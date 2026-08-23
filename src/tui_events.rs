//! TUI event loop and key handling.

use std::io;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::tui_app::{App, Mode};
use crate::tui_ui;

/// Rows moved by PgUp/PgDn and Ctrl-U/Ctrl-D.
const PAGE: isize = 10;

/// How long to wait for input before redrawing. Short enough that a finished
/// background download appears promptly, long enough to leave the CPU idle.
const POLL_INTERVAL: Duration = Duration::from_millis(150);

/// Run the interactive interface until the user quits.
pub fn run(app: &mut App) -> io::Result<()> {
    ratatui::run(|terminal| {
        // Draw once before touching the network, so the interface is up while
        // runtime discovery is still in flight.
        terminal.draw(|frame| tui_ui::draw(frame, app))?;
        app.start_discovery();

        while !app.should_quit {
            terminal.draw(|frame| tui_ui::draw(frame, app))?;
            app.poll_events();
            if event::poll(POLL_INTERVAL)?
                && let Event::Key(key) = event::read()?
                // Windows reports both press and release; act on press only.
                && key.kind == KeyEventKind::Press
            {
                handle_key(app, key);
            }
        }

        // The theme, use case and speed factors the user settled on are the
        // starting point for the next session.
        app.save_config();
        Ok(())
    })
}

pub fn handle_key(app: &mut App, key: KeyEvent) {
    match app.mode {
        Mode::Search => handle_search_key(app, key),
        // Any key dismisses a read-only panel or popup.
        Mode::Help | Mode::Plan | Mode::Comparison => app.close_popup(),
        Mode::SimulateHardware | Mode::AdvancedConfig => handle_form_key(app, key),
        Mode::Normal | Mode::Detail => handle_normal_key(app, key),
    }
}

fn handle_search_key(app: &mut App, key: KeyEvent) {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Esc | KeyCode::Enter => app.mode = Mode::Normal,
        KeyCode::Char('u') if ctrl => app.clear_search(),
        KeyCode::Backspace => {
            app.search.pop();
            app.refilter();
        }
        KeyCode::Char(c) => {
            app.search.push(c);
            app.refilter();
        }
        // Let the cursor move while the query is still being typed, so a
        // match can be selected without leaving search mode first.
        KeyCode::Up => app.move_selection(-1),
        KeyCode::Down => app.move_selection(1),
        _ => {}
    }
}

/// Keys for the two editable popups.
///
/// The hardware simulator and the speed tunables are the same interaction, so
/// they share one handler; only Enter and Ctrl-R differ, and those branch on
/// the mode at the end.
fn handle_form_key(app: &mut App, key: KeyEvent) {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

    // Ctrl-U clears the active field; a bare `u`/`j`/`k` is navigation, so
    // the modifier check has to come first.
    if ctrl && key.code == KeyCode::Char('u') {
        if let Some(form) = app.active_form_mut() {
            form.clear_active();
        }
        return;
    }
    if ctrl && key.code == KeyCode::Char('r') {
        if app.mode == Mode::SimulateHardware {
            app.reset_simulation_fields();
        }
        return;
    }

    match key.code {
        KeyCode::Esc => app.close_popup(),
        KeyCode::Enter => match app.mode {
            Mode::SimulateHardware => app.apply_simulation(),
            Mode::AdvancedConfig => app.apply_advanced_config(),
            _ => app.close_popup(),
        },
        KeyCode::Tab | KeyCode::Down | KeyCode::Char('j') => {
            if let Some(form) = app.active_form_mut() {
                form.focus_next();
            }
        }
        KeyCode::BackTab | KeyCode::Up | KeyCode::Char('k') => {
            if let Some(form) = app.active_form_mut() {
                form.focus_prev();
            }
        }
        KeyCode::Backspace => {
            if let Some(form) = app.active_form_mut() {
                form.backspace();
            }
        }
        KeyCode::Char(c) => {
            if let Some(form) = app.active_form_mut() {
                // The form itself rejects anything that is not part of a
                // decimal number, so `j`/`k` above never reach it as text.
                form.push(c);
            }
        }
        _ => {}
    }
}

fn handle_normal_key(app: &mut App, key: KeyEvent) {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
        KeyCode::Char('c') if ctrl => app.should_quit = true,

        // Navigation. The Ctrl-modified forms must be matched before the bare
        // letters, which mean something else entirely.
        KeyCode::Char('d') if ctrl => app.move_selection(PAGE),
        KeyCode::Char('u') if ctrl => app.move_selection(-PAGE),
        KeyCode::Char('j') | KeyCode::Down => app.move_selection(1),
        KeyCode::Char('k') | KeyCode::Up => app.move_selection(-1),
        KeyCode::PageDown => app.move_selection(PAGE),
        KeyCode::PageUp => app.move_selection(-PAGE),
        KeyCode::Char('g') | KeyCode::Home => app.select_first(),
        KeyCode::Char('G') | KeyCode::End => app.select_last(),

        // Filtering and ranking.
        KeyCode::Char('/') => {
            app.mode = Mode::Search;
            app.status.clear();
        }
        KeyCode::Char('f') => app.cycle_fit_filter(),
        KeyCode::Char('a') => app.cycle_availability(),
        KeyCode::Char('s') => app.cycle_sort(),
        KeyCode::Char('u') => app.cycle_use_case(),

        // Runtime integration.
        KeyCode::Char('d') => match app.selected_result().and_then(|r| r.ollama.clone()) {
            Some(tag) => app.start_pull(tag),
            None => {
                app.status = "no Ollama tag for this model — see the detail panel for the command"
                    .to_string()
            }
        },
        KeyCode::Char('r') => app.refresh_installed(),

        // Panels and popups.
        KeyCode::Char('p') => {
            if app.selected_result().is_some() {
                app.mode = Mode::Plan;
            }
        }
        KeyCode::Char('S') => app.open_simulation(),
        KeyCode::Char('A') => app.open_advanced_config(),
        KeyCode::Char('t') => app.cycle_theme(),
        KeyCode::Char('m') => app.mark_for_comparison(),
        KeyCode::Char('c') => {
            if app.can_compare() {
                app.mode = Mode::Comparison;
            } else if app.marked_result().is_none() {
                app.status = "mark a model first (press m)".to_string();
            } else {
                app.status = "select a different model to compare against".to_string();
            }
        }
        KeyCode::Enter => {
            app.mode = if app.mode == Mode::Detail {
                Mode::Normal
            } else {
                Mode::Detail
            };
        }
        KeyCode::Char('h' | '?') => app.mode = Mode::Help,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui_app::{Availability, FitFilter, SortColumn, tests::test_app};

    fn press(app: &mut App, code: KeyCode) {
        handle_key(app, KeyEvent::new(code, KeyModifiers::NONE));
    }

    fn press_ctrl(app: &mut App, code: KeyCode) {
        handle_key(app, KeyEvent::new(code, KeyModifiers::CONTROL));
    }

    #[test]
    fn quits_on_q_and_on_ctrl_c() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('q'));
        assert!(app.should_quit);

        let mut app = test_app();
        press_ctrl(&mut app, KeyCode::Char('c'));
        assert!(app.should_quit);
    }

    #[test]
    fn navigates_with_vim_keys() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('j'));
        assert_eq!(app.selected, 1);
        press(&mut app, KeyCode::Char('k'));
        assert_eq!(app.selected, 0);
        press(&mut app, KeyCode::Char('G'));
        assert_eq!(app.selected, app.visible.len() - 1);
        press(&mut app, KeyCode::Char('g'));
        assert_eq!(app.selected, 0);
    }

    #[test]
    fn ctrl_navigation_pages_rather_than_downloading() {
        // `d` downloads and `u` cycles the use case, so the Ctrl forms must
        // be matched first or paging would trigger the wrong action.
        let mut app = test_app();
        press_ctrl(&mut app, KeyCode::Char('d'));
        assert_eq!(app.selected, PAGE as usize);
        assert!(app.download_tag.is_none(), "Ctrl-D must not start a pull");

        let use_case_before = app.target;
        press_ctrl(&mut app, KeyCode::Char('u'));
        assert_eq!(app.selected, 0);
        assert_eq!(app.target, use_case_before, "Ctrl-U must not re-rank");
    }

    #[test]
    fn search_mode_types_and_filters() {
        let mut app = test_app();
        let total = app.visible.len();
        press(&mut app, KeyCode::Char('/'));
        assert_eq!(app.mode, Mode::Search);
        for c in "qwen".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        assert_eq!(app.search, "qwen");
        assert!(app.visible.len() < total);

        press(&mut app, KeyCode::Backspace);
        assert_eq!(app.search, "qwe");

        press_ctrl(&mut app, KeyCode::Char('u'));
        assert!(app.search.is_empty());
        assert_eq!(app.visible.len(), total);

        press(&mut app, KeyCode::Enter);
        assert_eq!(app.mode, Mode::Normal);
    }

    #[test]
    fn every_normal_key_is_typed_text_inside_search() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('/'));
        for c in "qfsu".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        assert!(!app.should_quit);
        assert_eq!(app.search, "qfsu");
        assert_eq!(app.fit_filter, FitFilter::All);
        assert_eq!(app.sort, SortColumn::Score);
    }

    #[test]
    fn arrow_keys_move_the_cursor_during_search() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('/'));
        press(&mut app, KeyCode::Down);
        assert_eq!(app.selected, 1);
        assert!(app.search.is_empty(), "arrows are not typed into the query");
    }

    #[test]
    fn filters_and_sorting_cycle() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('f'));
        assert_eq!(app.fit_filter, FitFilter::Runnable);
        assert!(app.visible.iter().all(|&i| app.results[i].is_runnable()));

        press(&mut app, KeyCode::Char('a'));
        assert_eq!(app.availability, Availability::Gguf);

        let sort_before = app.sort;
        press(&mut app, KeyCode::Char('s'));
        assert_ne!(app.sort, sort_before);
    }

    #[test]
    fn detail_and_help_toggle() {
        let mut app = test_app();
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.mode, Mode::Detail);
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.mode, Mode::Normal);

        press(&mut app, KeyCode::Char('h'));
        assert_eq!(app.mode, Mode::Help);
        press(&mut app, KeyCode::Char('x'));
        assert_eq!(app.mode, Mode::Normal);
    }

    #[test]
    fn navigation_works_while_the_detail_panel_is_open() {
        let mut app = test_app();
        press(&mut app, KeyCode::Enter);
        press(&mut app, KeyCode::Char('j'));
        assert_eq!(app.mode, Mode::Detail);
        assert_eq!(app.selected, 1);
    }

    #[test]
    fn theme_cycles_and_is_remembered_in_state() {
        let mut app = test_app();
        let before = app.theme;
        press(&mut app, KeyCode::Char('t'));
        assert_ne!(app.theme, before);
        assert!(app.status.contains("theme"));
    }

    #[test]
    fn comparison_needs_a_mark_and_two_distinct_models() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('c'));
        assert_eq!(app.mode, Mode::Normal);
        assert!(app.status.contains("mark a model"));

        press(&mut app, KeyCode::Char('m'));
        press(&mut app, KeyCode::Char('c'));
        assert_eq!(app.mode, Mode::Normal, "a model cannot compare with itself");
        assert!(app.status.contains("different model"));

        press(&mut app, KeyCode::Char('j'));
        press(&mut app, KeyCode::Char('c'));
        assert_eq!(app.mode, Mode::Comparison);
        press(&mut app, KeyCode::Char('x'));
        assert_eq!(app.mode, Mode::Normal, "any key closes the comparison");
    }

    #[test]
    fn plan_panel_opens_and_closes() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('p'));
        assert_eq!(app.mode, Mode::Plan);
        press(&mut app, KeyCode::Char('x'));
        assert_eq!(app.mode, Mode::Normal);
    }

    #[test]
    fn the_simulation_popup_edits_applies_and_cancels() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('S'));
        assert_eq!(app.mode, Mode::SimulateHardware);

        // Esc abandons the edit.
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.mode, Mode::Normal);

        press(&mut app, KeyCode::Char('S'));
        press_ctrl(&mut app, KeyCode::Char('u'));
        for c in "24".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.mode, Mode::Normal);
        assert!((app.hw.total_vram_gb() - 24.0).abs() < 0.01);
    }

    #[test]
    fn form_navigation_keys_are_not_typed_into_the_field() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('S'));
        press_ctrl(&mut app, KeyCode::Char('u'));
        // `j` and `k` move between fields; only digits reach the value.
        for c in "1j2k3".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        let typed: String = app
            .simulation
            .fields()
            .iter()
            .map(|f| f.value.clone())
            .collect();
        assert!(!typed.contains('j') && !typed.contains('k'), "{typed}");
    }

    #[test]
    fn tab_and_backtab_move_between_form_fields() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('S'));
        assert_eq!(app.simulation.active(), 0);
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.simulation.active(), 1);
        press(&mut app, KeyCode::BackTab);
        assert_eq!(app.simulation.active(), 0);
    }

    #[test]
    fn ctrl_r_restores_the_simulation_fields() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('S'));
        let original = app.simulation.fields()[0].value.clone();
        press_ctrl(&mut app, KeyCode::Char('u'));
        assert!(app.simulation.fields()[0].value.is_empty());
        press_ctrl(&mut app, KeyCode::Char('r'));
        assert_eq!(app.simulation.fields()[0].value, original);
    }

    #[test]
    fn the_advanced_config_popup_shares_the_same_handler() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('A'));
        assert_eq!(app.mode, Mode::AdvancedConfig);
        press_ctrl(&mut app, KeyCode::Char('u'));
        for c in "0.8".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.mode, Mode::Normal);
        assert!((app.cfg.efficiency - 0.8).abs() < 1e-9);
    }

    #[test]
    fn download_without_a_tag_explains_itself() {
        let mut app = test_app();
        // Point the cursor at a model with no Ollama tag, if the catalog has one.
        let untagged = app
            .visible
            .iter()
            .position(|&i| app.results[i].ollama.is_none());
        if let Some(row) = untagged {
            app.selected = row;
            press(&mut app, KeyCode::Char('d'));
            assert!(app.download_tag.is_none());
            assert!(app.status.contains("no Ollama tag"));
        }
    }

    #[test]
    fn refresh_reports_that_it_started() {
        let mut app = test_app();
        press(&mut app, KeyCode::Char('r'));
        // Discovery runs on a background thread; the key press only has to
        // acknowledge it, never to block on the result.
        assert!(app.status.contains("refreshing"), "{}", app.status);
    }

    #[test]
    fn unbound_keys_are_ignored() {
        let mut app = test_app();
        let selected = app.selected;
        for code in [KeyCode::Char('z'), KeyCode::F(5), KeyCode::Insert] {
            press(&mut app, code);
        }
        assert_eq!(app.mode, Mode::Normal);
        assert_eq!(app.selected, selected);
        assert!(!app.should_quit);
    }
}
