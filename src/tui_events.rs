//! TUI event loop and key handling.

use std::io;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::tui_app::{App, Mode};
use crate::tui_ui;

const PAGE: isize = 10;
const POLL_INTERVAL: Duration = Duration::from_millis(150);

/// Run the interactive interface until the user quits.
pub fn run(app: &mut App) -> io::Result<()> {
    ratatui::run(|terminal| {
        while !app.should_quit {
            terminal.draw(|frame| tui_ui::draw(frame, app))?;
            app.poll_downloads();
            if event::poll(POLL_INTERVAL)? {
                match event::read()? {
                    // Windows reports both press and release; act on press only.
                    Event::Key(key) if key.kind == KeyEventKind::Press => handle_key(app, key),
                    _ => {}
                }
            }
        }
        Ok(())
    })
}

pub fn handle_key(app: &mut App, key: KeyEvent) {
    match app.mode {
        Mode::Search => handle_search_key(app, key),
        Mode::Help => {
            // Any key dismisses the help popup.
            app.mode = Mode::Normal;
        }
        Mode::Plan => {
            // Any key exits plan mode.
            app.mode = Mode::Normal;
        }
        Mode::SimulateHardware => handle_simulation_key(app, key),
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
        KeyCode::Up => app.move_selection(-1),
        KeyCode::Down => app.move_selection(1),
        _ => {}
    }
}

fn handle_simulation_key(app: &mut App, key: KeyEvent) {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Esc => app.cancel_simulation(),
        KeyCode::Enter => app.apply_simulation(),
        KeyCode::Tab => app.sim_field = (app.sim_field + 1) % 3,
        KeyCode::Down if !ctrl => app.sim_field = (app.sim_field + 1) % 3,
        KeyCode::Up if !ctrl => app.sim_field = if app.sim_field == 0 { 2 } else { app.sim_field - 1 },
        KeyCode::Char('j') => app.sim_field = (app.sim_field + 1) % 3,
        KeyCode::Char('k') => app.sim_field = if app.sim_field == 0 { 2 } else { app.sim_field - 1 },
        KeyCode::Backspace => match app.sim_field {
            0 => { app.sim_vram_input.pop(); }
            1 => { app.sim_ram_input.pop(); }
            2 => { app.sim_cpu_input.pop(); }
            _ => {}
        },
        KeyCode::Char('u') if ctrl => match app.sim_field {
            0 => app.sim_vram_input.clear(),
            1 => app.sim_ram_input.clear(),
            2 => app.sim_cpu_input.clear(),
            _ => {}
        },
        KeyCode::Char('r') if ctrl => {
            // Reset to current hardware
            app.sim_vram_input = app.hw.gpus.first().map(|g| format!("{:.1}", g.vram_gb))
                .unwrap_or_else(|| "0.0".to_string());
            app.sim_ram_input = format!("{:.1}", app.hw.total_ram_gb);
            app.sim_cpu_input = format!("{}", app.hw.cpu_cores);
        }
        KeyCode::Char(c) if c.is_numeric() || c == '.' => match app.sim_field {
            0 => app.sim_vram_input.push(c),
            1 => app.sim_ram_input.push(c),
            2 => app.sim_cpu_input.push(c),
            _ => {}
        },
        _ => {}
    }
}

fn handle_normal_key(app: &mut App, key: KeyEvent) {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
        KeyCode::Char('c') if ctrl => app.should_quit = true,

        KeyCode::Char('j') | KeyCode::Down => app.move_selection(1),
        KeyCode::Char('k') | KeyCode::Up => app.move_selection(-1),
        KeyCode::PageDown => app.move_selection(PAGE),
        KeyCode::PageUp => app.move_selection(-PAGE),
        KeyCode::Char('d') if ctrl => app.move_selection(PAGE),
        KeyCode::Char('u') if ctrl => app.move_selection(-PAGE),
        KeyCode::Char('g') | KeyCode::Home => app.select_first(),
        KeyCode::Char('G') | KeyCode::End => app.select_last(),

        KeyCode::Char('/') => {
            app.mode = Mode::Search;
            app.status.clear();
        }
        KeyCode::Char('f') => app.cycle_fit_filter(),
        KeyCode::Char('a') => app.cycle_availability(),
        KeyCode::Char('s') => app.cycle_sort(),
        KeyCode::Char('u') => app.cycle_use_case(),

        KeyCode::Char('d') => match app.selected_result().and_then(|r| r.ollama.clone()) {
            Some(tag) => app.start_pull(tag),
            None => app.status = "no Ollama tag for this model".to_string(),
        },
        KeyCode::Char('r') => app.refresh_installed(),
        KeyCode::Char('S') => app.open_simulation(),
        KeyCode::Char('p') => {
            if app.selected_result().is_some() {
                app.mode = Mode::Plan;
            }
        }

        KeyCode::Enter => {
            app.mode = if app.mode == Mode::Detail {
                Mode::Normal
            } else {
                Mode::Detail
            };
        }
        KeyCode::Char('h') | KeyCode::Char('?') => app.mode = Mode::Help,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::SpeedConfig;
    use crate::hardware::{Backend, Hardware};
    use crate::models::{ModelDb, UseCase};
    use crate::tui_app::FitFilter;

    fn app() -> App {
        let mut hw = Hardware {
            cpu_brand: "Test CPU".into(),
            cpu_cores: 8,
            cpu_threads: 16,
            arch: "x86_64".into(),
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            gpus: Vec::new(),
            backend: Backend::CpuX86,
            simulated: true,
        };
        hw.set_vram(12.0);
        App::new(
            hw,
            ModelDb::embedded(),
            UseCase::General,
            SpeedConfig::default(),
        )
    }

    fn press(app: &mut App, code: KeyCode) {
        handle_key(app, KeyEvent::new(code, KeyModifiers::NONE));
    }

    fn press_ctrl(app: &mut App, code: KeyCode) {
        handle_key(app, KeyEvent::new(code, KeyModifiers::CONTROL));
    }

    #[test]
    fn quits_on_q() {
        let mut app = app();
        press(&mut app, KeyCode::Char('q'));
        assert!(app.should_quit);
    }

    #[test]
    fn navigates_with_vim_keys() {
        let mut app = app();
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
    fn search_mode_types_and_filters() {
        let mut app = app();
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
    fn q_inside_search_is_typed_not_a_quit() {
        let mut app = app();
        press(&mut app, KeyCode::Char('/'));
        press(&mut app, KeyCode::Char('q'));
        assert!(!app.should_quit);
        assert_eq!(app.search, "q");
    }

    #[test]
    fn filters_and_sorting_cycle() {
        let mut app = app();
        assert_eq!(app.fit_filter, FitFilter::All);
        press(&mut app, KeyCode::Char('f'));
        assert_eq!(app.fit_filter, FitFilter::Runnable);
        assert!(app.visible.iter().all(|&i| app.results[i].is_runnable()));

        let sort_before = app.sort;
        press(&mut app, KeyCode::Char('s'));
        assert_ne!(app.sort, sort_before);
    }

    #[test]
    fn detail_and_help_toggle() {
        let mut app = app();
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
    fn use_case_cycling_reranks() {
        let mut app = app();
        assert_eq!(app.target, UseCase::General);
        press(&mut app, KeyCode::Char('u'));
        assert_eq!(app.target, UseCase::Coding);
        assert!(!app.visible.is_empty());
    }

    #[test]
    fn navigation_works_while_the_detail_panel_is_open() {
        let mut app = app();
        press(&mut app, KeyCode::Enter);
        press(&mut app, KeyCode::Char('j'));
        assert_eq!(app.mode, Mode::Detail);
        assert_eq!(app.selected, 1);
    }

    #[test]
    fn r_refresh_updates_status() {
        let mut app = app();
        press(&mut app, KeyCode::Char('r'));
        // Ollama is absent in CI, so status will contain an error or "refreshed".
        // Just verify that it set the status and didn't panic.
        assert!(!app.status.is_empty());
    }

    #[test]
    fn d_with_ollama_tag_starts_download() {
        let mut app = app();
        // All seed models now have ollama tags, so the selected model should have one.
        if let Some(result) = app.selected_result() {
            if let Some(_tag) = &result.ollama {
                press(&mut app, KeyCode::Char('d'));
                // Verify download started (tag is set and status shows pulling).
                assert!(app.download_tag.is_some());
                assert!(app.status.contains("pulling"));
            }
        }
    }

    #[test]
    fn d_key_does_not_panic() {
        let mut test_app = app();
        // All 27 seed models have ollama tags, so this exercises the "with tag" branch.
        // The "no tag" branch is not covered with the current DB, but the code path
        // exists for future models without Ollama tags.
        if let Some(result) = test_app.selected_result() {
            if result.ollama.is_some() {
                press(&mut test_app, KeyCode::Char('d'));
                // Just verify it didn't panic and set a status.
                assert!(!test_app.status.is_empty());
            }
        }
    }
}
