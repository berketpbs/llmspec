//! TUI rendering.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Row, Table, TableState, Wrap};
use std::collections::HashSet;

use crate::display::{format_context, format_params, format_tps};
use crate::fit::{FitLevel, FitResult, RunMode};
use crate::tui_app::{App, Mode};

const HEADER: [&str; 12] = [
    "Model", "Provider", "Params", "Quant", "Mode", "Fit", "Inst", "Mem%", "Ctx", "tok/s",
    "Score", "Use Case",
];

const WIDTHS: [Constraint; 12] = [
    Constraint::Min(24),
    Constraint::Length(13),
    Constraint::Length(8),
    Constraint::Length(7),
    Constraint::Length(8),
    Constraint::Length(9),
    Constraint::Length(4),
    Constraint::Length(6),
    Constraint::Length(6),
    Constraint::Length(7),
    Constraint::Length(6),
    Constraint::Length(10),
];

pub fn draw(frame: &mut Frame, app: &mut App) {
    let detail_height = if app.mode == Mode::Detail { 13 } else { 0 };
    let layout = Layout::vertical([
        Constraint::Length(6),
        Constraint::Fill(1),
        Constraint::Length(detail_height),
        Constraint::Length(1),
    ]);
    let [system, table, detail, status] = frame.area().layout(&layout);

    render_system(frame, system, app);
    render_table(frame, table, app);
    if app.mode == Mode::Detail {
        render_detail(frame, detail, app);
    }
    render_status(frame, status, app);

    if app.mode == Mode::Help {
        render_help(frame, frame.area());
    }
}

fn fit_color(fit: FitLevel) -> Color {
    match fit {
        FitLevel::Perfect => Color::Green,
        FitLevel::Good => Color::Cyan,
        FitLevel::Marginal => Color::Yellow,
        FitLevel::TooTight => Color::Red,
    }
}

fn mode_color(mode: RunMode) -> Color {
    match mode {
        RunMode::Gpu => Color::Green,
        RunMode::Moe => Color::Magenta,
        RunMode::CpuGpu => Color::Yellow,
        RunMode::Cpu => Color::DarkGray,
    }
}

// ---------------------------------------------------------------------------
// System panel
// ---------------------------------------------------------------------------

fn render_system(frame: &mut Frame, area: Rect, app: &App) {
    let hw = &app.hw;
    let gpu = if hw.gpus.is_empty() {
        Line::from(vec![
            Span::styled("GPU  ", Style::new().dim()),
            Span::styled("none detected", Style::new().fg(Color::Yellow)),
        ])
    } else {
        let bandwidth = match hw.primary_bandwidth() {
            Some(bw) => format!("{bw:.0} GB/s"),
            None => "bandwidth unknown".to_string(),
        };
        Line::from(vec![
            Span::styled("GPU  ", Style::new().dim()),
            Span::raw(format!(
                "{} — {:.1} GB VRAM, {}",
                hw.primary_gpu_name(),
                hw.total_vram_gb(),
                bandwidth
            )),
        ])
    };

    let mut title = vec![Span::styled(
        " llmspec ",
        Style::new().fg(Color::Black).bg(Color::Cyan).bold(),
    )];
    if hw.simulated {
        title.push(Span::styled(
            " SIM ",
            Style::new().fg(Color::Black).bg(Color::Yellow).bold(),
        ));
    }

    let text = vec![
        Line::from(vec![
            Span::styled("CPU  ", Style::new().dim()),
            Span::raw(format!(
                "{} ({} cores / {} threads)",
                hw.cpu_brand, hw.cpu_cores, hw.cpu_threads
            )),
        ]),
        Line::from(vec![
            Span::styled("RAM  ", Style::new().dim()),
            Span::raw(format!(
                "{:.1} GB total, {:.1} GB available",
                hw.total_ram_gb, hw.available_ram_gb
            )),
        ]),
        gpu,
        Line::from(vec![
            Span::styled("Back ", Style::new().dim()),
            Span::styled(hw.backend.label(), Style::new().bold()),
            Span::styled("   Use case ", Style::new().dim()),
            Span::styled(app.target.as_str(), Style::new().fg(Color::Cyan).bold()),
        ]),
    ];

    let block = Block::bordered().title(Line::from(title));
    frame.render_widget(Paragraph::new(text).block(block), area);
}

// ---------------------------------------------------------------------------
// Model table
// ---------------------------------------------------------------------------

fn render_table(frame: &mut Frame, area: Rect, app: &mut App) {
    let header = Row::new(HEADER.map(|h| Span::styled(h, Style::new().bold())))
        .style(Style::new().fg(Color::White))
        .bottom_margin(0);

    let rows: Vec<Row> = app
        .visible
        .iter()
        .map(|&i| row_for(&app.results[i], &app.installed))
        .collect();

    let title = format!(
        " {} models · fit: {} · avail: {} · sort: {} ",
        app.visible.len(),
        app.fit_filter.label(),
        app.availability.label(),
        app.sort.label()
    );

    let table = Table::new(rows, WIDTHS)
        .header(header)
        .block(Block::bordered().title(title))
        .row_highlight_style(Style::new().bg(Color::Rgb(40, 45, 60)).bold())
        .highlight_symbol("› ");

    let mut state = TableState::default();
    if !app.visible.is_empty() {
        state.select(Some(app.selected));
    }
    frame.render_stateful_widget(table, area, &mut state);
}

fn row_for(r: &FitResult, installed: &HashSet<String>) -> Row<'static> {
    let score = format!("{:.1}", r.scores.composite);
    let inst = match &r.ollama {
        Some(tag) if installed.contains(tag) => {
            Span::styled("✓", Style::new().fg(Color::Green))
        }
        _ => Span::styled("–", Style::new().dim()),
    };
    Row::new(vec![
        Span::raw(r.name.clone()),
        Span::styled(r.provider.clone(), Style::new().dim()),
        Span::raw(format_params(r.params_b, r.active_params_b)),
        Span::raw(r.quant.label()),
        Span::styled(r.mode.label(), Style::new().fg(mode_color(r.mode))),
        Span::styled(r.fit.label(), Style::new().fg(fit_color(r.fit))),
        inst,
        Span::raw(format!("{:.0}%", r.mem_percent)),
        Span::raw(format_context(r.context)),
        Span::raw(format_tps(r.tokens_per_second)),
        Span::styled(score, Style::new().bold()),
        Span::styled(r.use_case.as_str(), Style::new().dim()),
    ])
}

// ---------------------------------------------------------------------------
// Detail panel
// ---------------------------------------------------------------------------

fn render_detail(frame: &mut Frame, area: Rect, app: &App) {
    let Some(r) = app.selected_result() else {
        return;
    };
    let capabilities = app
        .selected_model()
        .map(|m| m.capabilities.join(", "))
        .filter(|c| !c.is_empty())
        .unwrap_or_else(|| "—".to_string());

    let text = vec![
        Line::from(vec![
            Span::styled(r.name.clone(), Style::new().bold()),
            Span::styled(format!("  {}", r.model_id), Style::new().dim()),
        ]),
        Line::from(vec![
            Span::styled("Runs as    ", Style::new().dim()),
            Span::styled(
                format!("{} / {}", r.mode.label(), r.quant.label()),
                Style::new().fg(mode_color(r.mode)).bold(),
            ),
            Span::styled("   verdict ", Style::new().dim()),
            Span::styled(r.fit.label(), Style::new().fg(fit_color(r.fit)).bold()),
        ]),
        Line::from(vec![
            Span::styled("Memory     ", Style::new().dim()),
            Span::raw(format!(
                "{:.1} GB total · {:.1} GB resident · {:.0}% of pool",
                r.required_gb, r.resident_gb, r.mem_percent
            )),
        ]),
        Line::from(vec![
            Span::styled("Context    ", Style::new().dim()),
            Span::raw(format!(
                "{} used of {} max",
                format_context(r.context),
                format_context(r.max_context)
            )),
        ]),
        Line::from(vec![
            Span::styled("Throughput ", Style::new().dim()),
            Span::raw(format!(
                "~{} tok/s estimated",
                format_tps(r.tokens_per_second)
            )),
        ]),
        Line::from(vec![
            Span::styled("License    ", Style::new().dim()),
            Span::raw(r.license.clone()),
            Span::styled("   released ", Style::new().dim()),
            Span::raw(r.released.clone()),
            Span::styled("   caps ", Style::new().dim()),
            Span::raw(capabilities),
        ]),
        Line::raw(""),
        score_bar("Quality", r.scores.quality),
        score_bar("Speed", r.scores.speed),
        score_bar("Fit", r.scores.fit),
        score_bar("Context", r.scores.context),
    ];

    let block = Block::bordered().title(" Detail — Enter to close ");
    frame.render_widget(
        Paragraph::new(text).block(block).wrap(Wrap { trim: true }),
        area,
    );
}

fn score_bar(label: &str, score: f64) -> Line<'static> {
    const WIDTH: usize = 30;
    let filled = ((score / 100.0) * WIDTH as f64)
        .round()
        .clamp(0.0, WIDTH as f64) as usize;
    let color = if score >= 75.0 {
        Color::Green
    } else if score >= 50.0 {
        Color::Cyan
    } else if score >= 25.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    Line::from(vec![
        Span::styled(format!("{label:<11}"), Style::new().dim()),
        Span::styled("█".repeat(filled), Style::new().fg(color)),
        Span::styled("·".repeat(WIDTH - filled), Style::new().dim()),
        Span::raw(format!(" {score:.1}")),
    ])
}

// ---------------------------------------------------------------------------
// Status bar and help
// ---------------------------------------------------------------------------

fn render_status(frame: &mut Frame, area: Rect, app: &App) {
    let line = match app.mode {
        Mode::Search => Line::from(vec![
            Span::styled(" /", Style::new().fg(Color::Black).bg(Color::Cyan).bold()),
            Span::styled(
                format!("{} ", app.search),
                Style::new().fg(Color::Black).bg(Color::Cyan),
            ),
            Span::styled(
                "  Enter/Esc to accept · Ctrl-U to clear",
                Style::new().dim(),
            ),
        ]),
        _ => {
            let mut spans = vec![Span::styled(
                " j/k move · / search · f fit · a avail · s sort · u use case · d download · r refresh · Enter detail · h help · q quit",
                Style::new().dim(),
            )];
            if !app.status.is_empty() {
                spans.push(Span::styled(
                    format!("  │  {}", app.status),
                    Style::new().fg(Color::Cyan),
                ));
            }
            if !app.search.is_empty() {
                spans.push(Span::styled(
                    format!("  │  /{}", app.search),
                    Style::new().fg(Color::Yellow),
                ));
            }
            Line::from(spans)
        }
    };
    frame.render_widget(Paragraph::new(line), area);
}

const HELP: &[(&str, &str)] = &[
    ("j / k, ↑ / ↓", "move between models"),
    ("g / G", "jump to first / last"),
    ("PgUp / PgDn", "scroll by 10"),
    ("/", "search by name, provider, size or use case"),
    ("Ctrl-U", "clear the search"),
    (
        "f",
        "cycle fit filter: All, Runnable, Perfect, Good, Marginal",
    ),
    ("a", "cycle availability filter: All, GGUF Avail"),
    ("s", "cycle sort column"),
    ("u", "cycle target use case and re-rank"),
    ("d", "download selected model via Ollama"),
    ("r", "refresh installed models from Ollama"),
    ("Enter", "toggle the detail panel"),
    ("h / ?", "this help"),
    ("q / Esc", "quit"),
];

fn render_help(frame: &mut Frame, area: Rect) {
    let width = 64.min(area.width.saturating_sub(4));
    let height = (HELP.len() as u16 + 4).min(area.height.saturating_sub(2));
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    };

    let mut lines = vec![Line::raw("")];
    lines.extend(HELP.iter().map(|(key, description)| {
        Line::from(vec![
            Span::styled(format!("  {key:<14}"), Style::new().fg(Color::Cyan).bold()),
            Span::raw(*description),
        ])
    }));

    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::bordered()
                .title(" Key bindings — any key to close ")
                .style(Style::new().add_modifier(Modifier::BOLD)),
        ),
        popup,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use crate::fit::SpeedConfig;
    use crate::hardware::{Backend, Hardware};
    use crate::models::{ModelDb, UseCase};

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

    /// Render one frame and return the screen as text.
    fn screen(app: &mut App, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn renders_system_panel_and_models() {
        let mut app = app();
        let out = screen(&mut app, 140, 30);
        assert!(out.contains("llmspec"), "{out}");
        assert!(out.contains("SIM"), "simulated hardware should be badged");
        assert!(out.contains("Test CPU"));
        assert!(out.contains("12.0 GB VRAM"));
        // The top-ranked model should be on screen.
        let top = &app.results[app.visible[0]].name;
        assert!(out.contains(top.as_str()), "expected {top} in:\n{out}");
    }

    #[test]
    fn renders_detail_panel() {
        let mut app = app();
        app.mode = Mode::Detail;
        let out = screen(&mut app, 140, 40);
        assert!(out.contains("Detail"));
        assert!(out.contains("Quality"));
        assert!(out.contains("Throughput"));
    }

    #[test]
    fn renders_help_popup() {
        let mut app = app();
        app.mode = Mode::Help;
        let out = screen(&mut app, 140, 30);
        assert!(out.contains("Key bindings"));
        assert!(out.contains("cycle sort column"));
    }

    #[test]
    fn renders_search_prompt() {
        let mut app = app();
        app.mode = Mode::Search;
        app.search = "qwen".into();
        app.refilter();
        let out = screen(&mut app, 140, 30);
        assert!(out.contains("/qwen"), "{out}");
    }

    #[test]
    fn survives_a_tiny_terminal_and_an_empty_list() {
        let mut app = app();
        app.search = "no-such-model".into();
        app.refilter();
        assert!(app.visible.is_empty());
        // Must not panic on a cramped viewport with nothing to show.
        screen(&mut app, 40, 10);
        app.mode = Mode::Detail;
        screen(&mut app, 20, 8);
        app.mode = Mode::Help;
        screen(&mut app, 20, 6);
    }
}
