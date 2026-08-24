//! TUI rendering.
//!
//! Every colour comes from the active [`Palette`] rather than being named at
//! the call site, so switching themes repaints the whole interface and adding
//! one cannot leave part of the screen behind.

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Row, Table, TableState, Wrap};

use crate::display::{format_context, format_params, format_size_gb, format_tps};
use crate::fit::{FitLevel, FitResult, RunMode};
use crate::tui_app::{App, Mode};
use crate::tui_form::Form;
use crate::tui_theme::Palette;

/// Height of the hardware summary at the top, including its border.
const SYSTEM_HEIGHT: u16 = 6;

/// Height of the panel below the table. Sized to the tallest of the three
/// panels so that none of them is silently clipped.
const PANEL_HEIGHT: u16 = 16;

/// Table columns: heading, width, and whether the value is right-aligned.
const COLUMNS: &[(&str, Constraint)] = &[
    ("Model", Constraint::Min(22)),
    ("Provider", Constraint::Length(12)),
    ("Params", Constraint::Length(7)),
    ("Quant", Constraint::Length(7)),
    ("Mode", Constraint::Length(7)),
    ("Fit", Constraint::Length(9)),
    ("On disk", Constraint::Length(7)),
    ("Size", Constraint::Length(7)),
    ("Mem%", Constraint::Length(5)),
    ("Ctx", Constraint::Length(6)),
    ("tok/s", Constraint::Length(6)),
    ("Score", Constraint::Length(5)),
];

pub fn draw(frame: &mut Frame, app: &App) {
    let palette = app.theme.palette();
    let panel_height = if app.mode.is_panel() { PANEL_HEIGHT } else { 0 };

    let layout = Layout::vertical([
        Constraint::Length(SYSTEM_HEIGHT),
        Constraint::Fill(1),
        Constraint::Length(panel_height),
        Constraint::Length(1),
    ]);
    let [system, table, panel, status] = frame.area().layout(&layout);

    render_system(frame, system, app, &palette);
    render_table(frame, table, app, &palette);
    match app.mode {
        Mode::Detail => render_detail(frame, panel, app, &palette),
        Mode::Plan => render_plan(frame, panel, app, &palette),
        Mode::Comparison => render_comparison(frame, panel, app, &palette),
        _ => {}
    }
    render_status(frame, status, app, &palette);

    match app.mode {
        Mode::Help => render_help(frame, frame.area(), &palette),
        Mode::SimulateHardware => render_form(
            frame,
            frame.area(),
            &app.simulation,
            " Simulate hardware ",
            &palette,
        ),
        Mode::AdvancedConfig => render_form(
            frame,
            frame.area(),
            &app.speed_form,
            " Speed model ",
            &palette,
        ),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Shared styling
// ---------------------------------------------------------------------------

fn fit_style(fit: FitLevel, palette: &Palette) -> Style {
    let colour = match fit {
        FitLevel::Perfect => palette.good,
        FitLevel::Good => palette.ok,
        FitLevel::Marginal => palette.warn,
        FitLevel::TooTight => palette.bad,
    };
    Style::new().fg(colour)
}

fn mode_style(mode: RunMode, palette: &Palette) -> Style {
    let colour = match mode {
        RunMode::Gpu => palette.good,
        RunMode::Moe => palette.special,
        RunMode::CpuGpu => palette.warn,
        RunMode::Cpu => palette.dim,
    };
    Style::new().fg(colour)
}

fn dim(palette: &Palette) -> Style {
    Style::new().fg(palette.dim)
}

fn label(text: &str, palette: &Palette) -> Span<'static> {
    Span::styled(format!("{text:<11}"), dim(palette))
}

/// A bordered block in the theme's colours.
fn panel_block(title: &str, palette: &Palette) -> Block<'static> {
    Block::bordered()
        .title(Span::styled(
            title.to_string(),
            Style::new().fg(palette.accent),
        ))
        .border_style(dim(palette))
}

// ---------------------------------------------------------------------------
// System panel
// ---------------------------------------------------------------------------

fn render_system(frame: &mut Frame, area: Rect, app: &App, palette: &Palette) {
    let hw = &app.hw;

    let gpu = if hw.gpus.is_empty() {
        Line::from(vec![
            label("GPU", palette),
            Span::styled("none detected", Style::new().fg(palette.warn)),
        ])
    } else {
        let bandwidth = match hw.primary_bandwidth() {
            Some(bw) => format!("{bw:.0} GB/s"),
            None => "bandwidth unknown".to_string(),
        };
        Line::from(vec![
            label("GPU", palette),
            Span::raw(format!(
                "{} — {:.1} GB VRAM, {bandwidth}",
                hw.primary_gpu_name(),
                hw.total_vram_gb()
            )),
        ])
    };

    // Which runtimes are up is the first thing to check when a download or a
    // benchmark misbehaves, so it belongs on screen rather than in a menu.
    let runtimes = if app.discovery.runtimes.is_empty() {
        Span::styled("no local runtime", Style::new().fg(palette.warn))
    } else {
        Span::styled(
            app.discovery
                .runtimes
                .iter()
                .map(|r| r.name)
                .collect::<Vec<_>>()
                .join(", "),
            Style::new().fg(palette.good),
        )
    };

    let mut title = vec![Span::styled(
        " llmspec ",
        Style::new().fg(palette.selection).bg(palette.accent).bold(),
    )];
    if hw.simulated {
        title.push(Span::styled(
            " SIMULATED ",
            Style::new().fg(palette.selection).bg(palette.warn).bold(),
        ));
    }

    let text = vec![
        Line::from(vec![
            label("CPU", palette),
            Span::raw(format!(
                "{} ({} cores / {} threads)",
                hw.cpu_brand, hw.cpu_cores, hw.cpu_threads
            )),
        ]),
        Line::from(vec![
            label("RAM", palette),
            Span::raw(format!(
                "{:.1} GB total, {:.1} GB available",
                hw.total_ram_gb, hw.available_ram_gb
            )),
        ]),
        gpu,
        Line::from(vec![
            label("Backend", palette),
            Span::styled(hw.backend.label(), Style::new().bold()),
            Span::styled("   use case ", dim(palette)),
            Span::styled(app.target.as_str(), Style::new().fg(palette.accent).bold()),
            Span::styled("   runtimes ", dim(palette)),
            runtimes,
        ]),
    ];

    frame.render_widget(
        Paragraph::new(text).block(
            Block::bordered()
                .title(Line::from(title))
                .border_style(dim(palette)),
        ),
        area,
    );
}

// ---------------------------------------------------------------------------
// Model table
// ---------------------------------------------------------------------------

fn render_table(frame: &mut Frame, area: Rect, app: &App, palette: &Palette) {
    let header = Row::new(
        COLUMNS
            .iter()
            .map(|(name, _)| Span::styled(*name, Style::new().fg(palette.accent).bold())),
    );

    let rows: Vec<Row> = app
        .visible
        .iter()
        .map(|&i| {
            let result = &app.results[i];
            row_for(result, app.is_installed(result), palette)
        })
        .collect();

    let title = format!(
        " {} of {} models · fit {} · show {} · sort {} ",
        app.visible.len(),
        app.results.len(),
        app.fit_filter.label(),
        app.availability.label(),
        app.sort.label()
    );

    let widths: Vec<Constraint> = COLUMNS.iter().map(|(_, w)| *w).collect();
    let table = Table::new(rows, widths)
        .header(header)
        .block(panel_block(&title, palette))
        .row_highlight_style(Style::new().bg(palette.selection).bold())
        .highlight_symbol("› ");

    let mut state = TableState::default();
    if !app.visible.is_empty() {
        state.select(Some(app.selected));
    }
    frame.render_stateful_widget(table, area, &mut state);
}

fn row_for(r: &FitResult, installed: bool, palette: &Palette) -> Row<'static> {
    let on_disk = if installed {
        Span::styled("yes", Style::new().fg(palette.good))
    } else {
        Span::styled("–", dim(palette))
    };
    // A context far below the model's own maximum changes what the model is
    // useful for, so the number is marked rather than shown bare.
    let context = if r.context_is_reduced() {
        Span::styled(format_context(r.context), Style::new().fg(palette.warn))
    } else {
        Span::raw(format_context(r.context))
    };

    Row::new(vec![
        Span::raw(r.name.clone()),
        Span::styled(r.provider.clone(), dim(palette)),
        Span::raw(format_params(r.params_b, r.active_params_b)),
        Span::raw(r.quant.label()),
        Span::styled(r.mode.label(), mode_style(r.mode, palette)),
        Span::styled(r.fit.label(), fit_style(r.fit, palette)),
        on_disk,
        Span::raw(format_size_gb(r.download_gb)),
        Span::raw(format!("{:.0}%", r.mem_percent)),
        context,
        Span::raw(format_tps(r.tokens_per_second)),
        Span::styled(format!("{:.1}", r.scores.composite), Style::new().bold()),
    ])
}

// ---------------------------------------------------------------------------
// Detail panel
// ---------------------------------------------------------------------------

fn render_detail(frame: &mut Frame, area: Rect, app: &App, palette: &Palette) {
    let Some(r) = app.selected_result() else {
        return;
    };
    let capabilities = app
        .model_for(r)
        .map(|m| m.capabilities.join(", "))
        .filter(|c| !c.is_empty())
        .unwrap_or_else(|| "—".to_string());

    let context = if r.context_is_reduced() {
        Span::styled(
            format!(
                "{} of {} — short of this model's maximum on your hardware",
                format_context(r.context),
                format_context(r.max_context)
            ),
            Style::new().fg(palette.warn),
        )
    } else {
        Span::raw(format!("{} (full)", format_context(r.context)))
    };

    let mut text = vec![
        Line::from(vec![
            Span::styled(r.name.clone(), Style::new().fg(palette.accent).bold()),
            Span::styled(format!("  {}", r.model_id), dim(palette)),
        ]),
        Line::from(vec![
            label("Verdict", palette),
            Span::styled(r.fit.label(), fit_style(r.fit, palette).bold()),
            Span::styled("  running as ", dim(palette)),
            Span::styled(r.mode.label(), mode_style(r.mode, palette).bold()),
            Span::styled(format!(" at {}", r.quant.label()), dim(palette)),
        ]),
        Line::from(vec![
            label("Download", palette),
            Span::raw(format!("{} on disk", format_size_gb(r.download_gb))),
            Span::styled("   in memory ", dim(palette)),
            Span::raw(format!(
                "{:.1} GB ({:.0}% of the pool)",
                r.required_gb, r.mem_percent
            )),
        ]),
        Line::from(vec![label("Context", palette), context]),
        Line::from(vec![
            label("Throughput", palette),
            Span::raw(format!(
                "~{} tok/s estimated",
                format_tps(r.tokens_per_second)
            )),
            Span::styled("   verify with ", dim(palette)),
            Span::styled("llmspec bench", Style::new().fg(palette.accent)),
        ]),
        Line::from(vec![
            label("About", palette),
            Span::raw(format!("{} · {} · {}", r.license, r.released, capabilities)),
        ]),
    ];

    // The command to actually run the thing is the point of the whole tool.
    text.push(match app.commands_for(r) {
        Some((kind, install, run)) => Line::from(vec![
            label("Run it", palette),
            Span::styled(
                if app.is_installed(r) { run } else { install },
                Style::new().fg(palette.good),
            ),
            Span::styled(format!("   ({})", kind.label()), dim(palette)),
        ]),
        None => Line::from(vec![
            label("Run it", palette),
            Span::styled(
                "no packaged build for the detected runtime".to_string(),
                dim(palette),
            ),
        ]),
    });

    text.push(Line::raw(""));
    text.push(score_bar("Quality", r.scores.quality, palette));
    text.push(score_bar("Speed", r.scores.speed, palette));
    text.push(score_bar("Fit", r.scores.fit, palette));
    text.push(score_bar("Context", r.scores.context, palette));

    frame.render_widget(
        Paragraph::new(text)
            .block(panel_block(" Detail — Enter to close ", palette))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn score_bar(name: &str, score: f64, palette: &Palette) -> Line<'static> {
    const WIDTH: usize = 30;
    let filled = ((score / 100.0) * WIDTH as f64)
        .round()
        .clamp(0.0, WIDTH as f64) as usize;
    Line::from(vec![
        label(name, palette),
        Span::styled("█".repeat(filled), Style::new().fg(palette.score(score))),
        Span::styled("·".repeat(WIDTH - filled), dim(palette)),
        Span::raw(format!(" {score:.1}")),
    ])
}

// ---------------------------------------------------------------------------
// Plan panel
// ---------------------------------------------------------------------------

fn render_plan(frame: &mut Frame, area: Rect, app: &App, palette: &Palette) {
    let Some(result) = app.selected_result() else {
        return;
    };
    let Some(model) = app.model_for(result) else {
        return;
    };
    let plan = crate::fit::plan(model, result.quant, result.context, &app.cfg);

    let text = vec![
        Line::from(vec![
            Span::styled(
                plan.model_name.clone(),
                Style::new().fg(palette.accent).bold(),
            ),
            Span::styled(
                format!(
                    "  {} at {}, {} context",
                    format_params(plan.params_b, result.active_params_b),
                    plan.quantization.label(),
                    format_context(plan.context_length)
                ),
                dim(palette),
            ),
        ]),
        Line::raw(""),
        Line::from(vec![
            label("Min VRAM", palette),
            Span::raw(format!("{:.1} GB to hold it on the GPU", plan.min_vram_gb)),
        ]),
        Line::from(vec![
            label("Advised", palette),
            Span::raw(format!(
                "{:.1} GB, leaving headroom for the runtime",
                plan.recommended_vram_gb
            )),
        ]),
        Line::from(vec![
            label("Min RAM", palette),
            Span::raw(format!("{:.1} GB to run it on the CPU", plan.min_ram_gb)),
        ]),
        Line::raw(""),
        Line::from(vec![
            label("On a GPU", palette),
            Span::raw(format!("~{} tok/s", format_tps(plan.tps_gpu))),
            Span::styled("   on the CPU ", dim(palette)),
            Span::raw(format!("~{} tok/s", format_tps(plan.tps_cpu))),
        ]),
        Line::raw(""),
        Line::from(vec![
            label("Viable", palette),
            Span::raw(plan.viable_modes.join(", ")),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(text)
            .block(panel_block(" Hardware plan — any key to close ", palette))
            .wrap(Wrap { trim: true }),
        area,
    );
}

// ---------------------------------------------------------------------------
// Comparison panel
// ---------------------------------------------------------------------------

fn render_comparison(frame: &mut Frame, area: Rect, app: &App, palette: &Palette) {
    let (Some(marked), Some(selected)) = (app.marked_result(), app.selected_result()) else {
        return;
    };

    /// One comparison row: label, marked value, selected value, and which side
    /// wins (`None` when the metric has no better direction).
    type ComparisonRow = (&'static str, String, String, Option<bool>);

    let better = |a: f64, b: f64| Some(a > b);
    let rows: Vec<ComparisonRow> = vec![
        (
            "Score",
            format!("{:.1}", marked.scores.composite),
            format!("{:.1}", selected.scores.composite),
            better(marked.scores.composite, selected.scores.composite),
        ),
        (
            "Throughput",
            format!("{} tok/s", format_tps(marked.tokens_per_second)),
            format!("{} tok/s", format_tps(selected.tokens_per_second)),
            better(marked.tokens_per_second, selected.tokens_per_second),
        ),
        (
            "Verdict",
            marked.fit.label().to_string(),
            selected.fit.label().to_string(),
            Some(marked.fit > selected.fit),
        ),
        (
            "Runs as",
            format!("{} at {}", marked.mode.label(), marked.quant.label()),
            format!("{} at {}", selected.mode.label(), selected.quant.label()),
            None,
        ),
        (
            "Download",
            format_size_gb(marked.download_gb),
            format_size_gb(selected.download_gb),
            // Smaller is better here.
            Some(marked.download_gb < selected.download_gb),
        ),
        (
            "Memory",
            format!("{:.1} GB ({:.0}%)", marked.required_gb, marked.mem_percent),
            format!(
                "{:.1} GB ({:.0}%)",
                selected.required_gb, selected.mem_percent
            ),
            None,
        ),
        (
            "Context",
            format_context(marked.context),
            format_context(selected.context),
            Some(marked.context > selected.context),
        ),
        (
            "Parameters",
            format_params(marked.params_b, marked.active_params_b),
            format_params(selected.params_b, selected.active_params_b),
            None,
        ),
    ];

    let mut text = vec![
        Line::from(vec![
            Span::styled(format!("{:<12}", ""), dim(palette)),
            Span::styled(
                format!("{:<28}", truncate(&marked.name, 27)),
                Style::new().bold(),
            ),
            Span::styled(
                truncate(&selected.name, 27),
                Style::new().fg(palette.accent).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled(format!("{:<12}", ""), dim(palette)),
            Span::styled(format!("{:<28}", "marked (m)"), dim(palette)),
            Span::styled("selected", dim(palette)),
        ]),
        Line::raw(""),
    ];

    for (name, left, right, marked_wins) in rows {
        // Highlighting the winner is what makes the panel readable at a
        // glance; a metric with no better direction is left plain.
        let (left_style, right_style) = match marked_wins {
            Some(true) => (Style::new().fg(palette.good).bold(), Style::new()),
            Some(false) => (Style::new(), Style::new().fg(palette.good).bold()),
            None => (Style::new(), Style::new()),
        };
        text.push(Line::from(vec![
            Span::styled(format!("{name:<12}"), dim(palette)),
            Span::styled(format!("{:<28}", truncate(&left, 27)), left_style),
            Span::styled(truncate(&right, 27), right_style),
        ]));
    }

    frame.render_widget(
        Paragraph::new(text).block(panel_block(" Comparison — any key to close ", palette)),
        area,
    );
}

fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    text.chars()
        .take(width.saturating_sub(1))
        .chain(['…'])
        .collect()
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

/// Shown before the buttons. Not clickable — there is nothing to press for
/// "move the cursor".
const STATUS_PREFIX: &str = " j/k move  ";

/// Blank columns drawn between two buttons.
const BUTTON_GAP: u16 = 1;

/// The status bar's clickable hints, in the order they are drawn.
///
/// Each one carries the keystroke it stands for rather than an action of its
/// own, so a click is dispatched down exactly the same path as the key. The
/// full list of bindings lives in the help popup.
const STATUS_BUTTONS: &[(&str, KeyCode)] = &[
    ("/ search", KeyCode::Char('/')),
    ("f fit", KeyCode::Char('f')),
    ("a show", KeyCode::Char('a')),
    ("s sort", KeyCode::Char('s')),
    ("u use case", KeyCode::Char('u')),
    ("d download", KeyCode::Char('d')),
    ("Enter detail", KeyCode::Enter),
    ("h help", KeyCode::Char('h')),
    ("q quit", KeyCode::Char('q')),
];

/// Where each button sits on the status row: `(first column, one past the
/// last, keystroke)`.
///
/// Rendering and hit-testing both walk this, so a click cannot land on a
/// different label than the one drawn. Buttons that would run past the right
/// edge are dropped rather than clipped — a half-drawn label that still
/// answers to clicks is worse than no label.
pub fn status_button_layout(width: u16) -> Vec<(u16, u16, KeyCode)> {
    let mut x = STATUS_PREFIX.chars().count() as u16;
    let mut out = Vec::with_capacity(STATUS_BUTTONS.len());
    for (label, key) in STATUS_BUTTONS {
        // One padding space on each side of the label.
        let end = x + label.chars().count() as u16 + 2;
        if end > width {
            break;
        }
        out.push((x, end, *key));
        x = end + BUTTON_GAP;
    }
    out
}

/// The keystroke a click at `column` on the status row stands for.
///
/// `None` when the click missed every button, so a stray click in the gaps
/// does nothing rather than guessing at the nearest one.
pub fn status_button_at(width: u16, column: u16) -> Option<KeyCode> {
    status_button_layout(width)
        .into_iter()
        .find(|(start, end, _)| column >= *start && column < *end)
        .map(|(_, _, key)| key)
}

fn render_status(frame: &mut Frame, area: Rect, app: &App, palette: &Palette) {
    let line = if app.mode == Mode::Search {
        Line::from(vec![
            Span::styled(
                " / ",
                Style::new().fg(palette.selection).bg(palette.accent).bold(),
            ),
            Span::styled(
                format!("{} ", app.search),
                Style::new().fg(palette.selection).bg(palette.accent),
            ),
            Span::styled("  Enter or Esc to accept · Ctrl-U to clear", dim(palette)),
        ])
    } else if app.mode.is_popup() {
        // The normal hints name keys the popup has taken over, so showing
        // them while one is open would be actively wrong.
        Line::from(Span::styled(
            " a popup is open — Esc closes it",
            dim(palette),
        ))
    } else {
        let mut spans = vec![Span::styled(STATUS_PREFIX, dim(palette))];
        // Drawn from the same layout the click handler reads, so the two
        // cannot disagree about where a button is.
        let drawn = status_button_layout(area.width).len();
        for (label, _) in STATUS_BUTTONS.iter().take(drawn) {
            spans.push(Span::styled(
                format!(" {label} "),
                Style::new().fg(palette.selection).bg(palette.accent),
            ));
            spans.push(Span::raw(" ".repeat(BUTTON_GAP as usize)));
        }
        if !app.status.is_empty() {
            spans.push(Span::styled(
                format!("  │  {}", app.status),
                Style::new().fg(palette.accent),
            ));
        }
        if !app.search.is_empty() {
            spans.push(Span::styled(
                format!("  │  /{}", app.search),
                Style::new().fg(palette.warn),
            ));
        }
        Line::from(spans)
    };
    frame.render_widget(Paragraph::new(line), area);
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

/// Every binding, grouped. `None` in the key column starts a new group.
const HELP: &[(Option<&str>, &str)] = &[
    (None, "Moving around"),
    (Some("j / k, ↑ / ↓"), "move between models"),
    (Some("g / G, Home / End"), "jump to first / last"),
    (Some("PgUp / PgDn, Ctrl-U / Ctrl-D"), "scroll by ten"),
    (None, "Narrowing the list"),
    (Some("/"), "search by name, provider, size or capability"),
    (Some("Ctrl-U"), "clear the search (while searching)"),
    (
        Some("f"),
        "fit filter: all, runnable, perfect, good, marginal",
    ),
    (
        Some("a"),
        "show: all models, GGUF builds, already installed",
    ),
    (Some("s"), "sort column"),
    (Some("u"), "target use case, and re-rank for it"),
    (None, "Inspecting a model"),
    (Some("Enter"), "detail panel: memory, context, run command"),
    (Some("p"), "hardware plan: what this model would need"),
    (Some("m"), "mark a model, then"),
    (Some("c"), "compare the marked model with the selected one"),
    (None, "Running models"),
    (Some("d"), "download the selected model through Ollama"),
    (Some("r"), "re-probe local runtimes and installed models"),
    (None, "Changing the estimate"),
    (Some("S"), "simulate different VRAM, RAM or core count"),
    (Some("A"), "edit the speed model's tunables"),
    (None, "Other"),
    (Some("t"), "cycle the colour theme"),
    (Some("h / ?"), "this help"),
    (Some("Esc"), "close the open panel or popup"),
    (Some("q"), "quit"),
    (None, "Mouse"),
    (Some("click"), "press a button in the status bar"),
    (Some("wheel"), "move between models"),
];

fn render_help(frame: &mut Frame, area: Rect, palette: &Palette) {
    let mut lines = vec![Line::raw("")];
    for (key, description) in HELP {
        match key {
            Some(key) => lines.push(Line::from(vec![
                Span::styled(
                    format!("  {key:<30}"),
                    Style::new().fg(palette.accent).bold(),
                ),
                Span::raw(*description),
            ])),
            // Group heading.
            None => {
                lines.push(Line::raw(""));
                lines.push(Line::from(Span::styled(
                    format!("  {description}"),
                    Style::new().fg(palette.warn).bold(),
                )));
            }
        }
    }

    let popup = centred(area, 74, lines.len() as u16 + 2);
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines).block(panel_block(" Key bindings — any key to close ", palette)),
        popup,
    );
}

// ---------------------------------------------------------------------------
// Editable popups
// ---------------------------------------------------------------------------

fn render_form(frame: &mut Frame, area: Rect, form: &Form, title: &str, palette: &Palette) {
    let mut lines = vec![Line::raw("")];

    for (index, field) in form.fields().iter().enumerate() {
        let active = form.is_active(index);
        let (marker, style) = if active {
            ("› ", Style::new().fg(palette.accent).bold())
        } else {
            ("  ", Style::new())
        };
        lines.push(Line::from(vec![
            Span::styled(format!("  {marker}{:<12}", field.label), style),
            Span::styled(
                format!("{:<10}", field.value),
                if active {
                    Style::new()
                        .fg(palette.accent)
                        .add_modifier(Modifier::UNDERLINED)
                } else {
                    Style::new()
                },
            ),
            Span::styled(field.range_hint(), dim(palette)),
        ]));
    }

    // Explain only the field being edited: four permanent help lines crowd
    // out the values they describe.
    lines.push(Line::raw(""));
    if let Some(field) = form.fields().get(form.active()) {
        lines.push(Line::from(Span::styled(
            format!("  {}", field.help),
            dim(palette),
        )));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "  Tab/j/k move · Ctrl-U clear · Ctrl-R reset · Enter apply · Esc cancel",
        dim(palette),
    )));

    let popup = centred(area, 72, lines.len() as u16 + 2);
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines).block(panel_block(title, palette)),
        popup,
    );
}

/// Centre a popup, shrinking it to fit a small terminal.
fn centred(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use crate::providers::{DiscoveredRuntime, RuntimeKind};
    use crate::tui_app::tests::test_app;
    use crate::tui_theme::Theme;

    /// Render one frame and return the screen as text.
    fn screen(app: &App, width: u16, height: u16) -> String {
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
        let app = test_app();
        let out = screen(&app, 150, 30);
        assert!(out.contains("llmspec"), "{out}");
        assert!(out.contains("SIMULATED"), "simulated hardware is badged");
        assert!(out.contains("Test CPU"));
        assert!(out.contains("12.0 GB VRAM"));
        let top = &app.results[app.visible[0]].name;
        assert!(out.contains(top.as_str()), "expected {top} in:\n{out}");
    }

    #[test]
    fn the_system_panel_reports_runtime_status() {
        let mut app = test_app();
        assert!(screen(&app, 150, 30).contains("no local runtime"));

        app.discovery.runtimes = vec![DiscoveredRuntime {
            kind: RuntimeKind::LmStudio,
            name: RuntimeKind::LmStudio.label(),
            base_url: "http://127.0.0.1:1234".into(),
            model_count: 3,
            disk_gb: None,
        }];
        assert!(screen(&app, 150, 30).contains("LM Studio"));
    }

    #[test]
    fn the_table_shows_download_size_and_install_state() {
        let mut app = test_app();
        let out = screen(&app, 150, 30);
        assert!(out.contains("On disk"), "install column header missing");
        assert!(out.contains("Size"), "download-size column header missing");

        // Nothing is installed, so the column reads as absent everywhere.
        assert!(!out.contains("yes"), "nothing should be marked installed");

        let tag = app
            .results
            .iter()
            .find_map(|r| r.ollama.clone())
            .expect("catalog has ollama tags");
        app.discovery.installed.insert(&tag);
        app.refilter();
        assert!(screen(&app, 150, 30).contains("yes"));
    }

    #[test]
    fn the_detail_panel_shows_the_command_to_run_the_model() {
        let mut app = test_app();
        // Select a model with an Ollama tag so a command exists.
        let row = app
            .visible
            .iter()
            .position(|&i| app.results[i].ollama.is_some())
            .expect("a tagged model is visible");
        app.selected = row;
        app.mode = Mode::Detail;

        let out = screen(&app, 150, 44);
        assert!(out.contains("Detail"));
        assert!(out.contains("Run it"), "{out}");
        assert!(out.contains("ollama pull"), "{out}");
        assert!(out.contains("Download"), "download size is missing");
        assert!(out.contains("Quality") && out.contains("Context"));
    }

    #[test]
    fn an_installed_model_is_offered_the_run_command_not_the_pull() {
        let mut app = test_app();
        let row = app
            .visible
            .iter()
            .position(|&i| app.results[i].ollama.is_some())
            .unwrap();
        app.selected = row;
        let tag = app.results[app.visible[row]].ollama.clone().unwrap();
        app.discovery.installed.insert(&tag);
        app.mode = Mode::Detail;

        let out = screen(&app, 150, 44);
        assert!(out.contains("ollama run"), "{out}");
    }

    #[test]
    fn every_panel_fits_inside_its_allotted_height() {
        // The panels are sized by a constant; if one grows past it the last
        // line vanishes silently, so assert the final line of each is drawn.
        let mut app = test_app();

        app.mode = Mode::Detail;
        assert!(
            screen(&app, 150, 44).contains("Context"),
            "the detail panel's last score bar is clipped"
        );

        app.mode = Mode::Plan;
        assert!(
            screen(&app, 150, 44).contains("Viable"),
            "the plan panel's last line is clipped"
        );

        app.mark_for_comparison();
        app.move_selection(1);
        app.mode = Mode::Comparison;
        assert!(
            screen(&app, 150, 44).contains("Parameters"),
            "the comparison panel's last row is clipped"
        );
    }

    #[test]
    fn the_comparison_panel_uses_display_labels_not_debug_names() {
        let mut app = test_app();
        app.mark_for_comparison();
        app.move_selection(1);
        app.mode = Mode::Comparison;
        let out = screen(&app, 150, 44);

        assert!(out.contains("marked (m)"));
        // Debug formatting would print Q4KM and TooTight, which are not words.
        assert!(!out.contains("Q4KM"), "{out}");
        assert!(!out.contains("TooTight"), "{out}");
        assert!(!out.contains("CpuGpu"), "{out}");
    }

    #[test]
    fn the_help_popup_lists_every_binding() {
        let mut app = test_app();
        app.mode = Mode::Help;
        let out = screen(&app, 150, 46);
        assert!(out.contains("Key bindings"));
        // Keys that the earlier help omitted entirely.
        for expected in ["sort column", "colour theme", "hardware plan", "simulate"] {
            assert!(
                out.to_lowercase().contains(expected),
                "help is missing {expected}:\n{out}"
            );
        }
    }

    #[test]
    fn renders_search_prompt() {
        let mut app = test_app();
        app.mode = Mode::Search;
        app.search = "qwen".into();
        app.refilter();
        let out = screen(&app, 150, 30);
        assert!(out.contains("qwen"), "{out}");
    }

    #[test]
    fn form_popups_show_values_and_the_active_field_hint() {
        let mut app = test_app();
        app.open_simulation();
        let out = screen(&app, 150, 30);
        assert!(out.contains("Simulate hardware"));
        assert!(out.contains("VRAM (GB)"));
        assert!(out.contains("accelerator memory"), "active hint missing");

        app.open_advanced_config();
        let out = screen(&app, 150, 30);
        assert!(out.contains("Speed model"));
        assert!(out.contains("Efficiency"));
        assert!(out.contains("bandwidth"), "{out}");
    }

    #[test]
    fn a_reduced_context_is_visually_marked() {
        let app = test_app();
        // On 12 GB of VRAM at least one long-context model must run short.
        assert!(
            app.results.iter().any(FitResult::context_is_reduced),
            "expected some model to be placed below its native context"
        );
        // Rendering it must not panic and the row still shows a context.
        assert!(screen(&app, 150, 30).contains("Ctx"));
    }

    #[test]
    fn every_theme_renders() {
        let mut app = test_app();
        app.mode = Mode::Detail;
        for theme in Theme::ALL {
            app.theme = theme;
            let out = screen(&app, 150, 44);
            assert!(out.contains("llmspec"), "{} failed to render", theme.name());
        }
    }

    #[test]
    fn survives_a_tiny_terminal_and_an_empty_list() {
        let mut app = test_app();
        app.search = "no-such-model".into();
        app.refilter();
        assert!(app.visible.is_empty());

        // Must not panic on a cramped viewport with nothing to show.
        for (w, h) in [(40, 10), (20, 8), (10, 4), (1, 1)] {
            for mode in [
                Mode::Normal,
                Mode::Detail,
                Mode::Plan,
                Mode::Comparison,
                Mode::Help,
                Mode::SimulateHardware,
                Mode::AdvancedConfig,
                Mode::Search,
            ] {
                app.mode = mode;
                screen(&app, w, h);
            }
        }
    }
}
