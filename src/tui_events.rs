//! TUI event loop and key handling.

use std::io;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::tui_app::{App, Mode};
use crate::tui_ui;

const PAGE: isize = 10;

/// Run the interactive interface until the user quits.
pub fn run(app: &mut App) -> io::Result<()> {
    ratatui::run(|terminal| {
        while !app.should_quit {
            terminal.draw(|frame| tui_ui::draw(frame, app))?;
            match event::read()? {
                // Windows reports both press and release; act on press only.
                Event::Key(key) if key.kind == KeyEventKind::Press => handle_key(app, key),
                _ => {}
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
}
