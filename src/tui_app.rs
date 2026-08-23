//! TUI application state: filters, sorting, navigation and background work.

use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;

use crate::config::{Config, PersistedSpeed};
use crate::fit::{self, FitLevel, FitResult, SpeedConfig};
use crate::hardware::Hardware;
use crate::models::{Model, ModelDb, UseCase};
use crate::providers::{
    DiscoveredRuntime, InstalledIndex, ProviderRegistry, Runtime, RuntimeKind,
};
use crate::tui_form::{Field, Form};
use crate::tui_theme::Theme;

/// Work that finishes on a background thread and reports back to the UI.
///
/// Both variants exist because the alternative — doing the network call on the
/// event-loop thread — freezes the interface for as long as the runtime takes
/// to answer, which on a cold Ollama can be seconds.
#[derive(Debug)]
pub enum BackgroundEvent {
    /// A model download finished, successfully or not.
    PullFinished {
        tag: String,
        result: Result<(), String>,
    },
    /// Local runtimes were probed and their models listed.
    Discovered(Result<Discovery, String>),
}

/// What one pass of runtime discovery found.
#[derive(Debug, Default, Clone)]
pub struct Discovery {
    pub installed: InstalledIndex,
    pub runtimes: Vec<DiscoveredRuntime>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Search,
    Help,
    Detail,
    Plan,
    SimulateHardware,
    AdvancedConfig,
    Comparison,
}

impl Mode {
    /// True when the mode draws a panel below the table rather than over it.
    pub fn is_panel(self) -> bool {
        matches!(self, Mode::Detail | Mode::Plan | Mode::Comparison)
    }

    /// True when the mode draws a popup over the whole screen.
    pub fn is_popup(self) -> bool {
        matches!(
            self,
            Mode::Help | Mode::SimulateHardware | Mode::AdvancedConfig
        )
    }
}

// ---------------------------------------------------------------------------
// Filters and sorting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitFilter {
    All,
    Runnable,
    Perfect,
    Good,
    Marginal,
}

impl FitFilter {
    const CYCLE: [FitFilter; 5] = [
        FitFilter::All,
        FitFilter::Runnable,
        FitFilter::Perfect,
        FitFilter::Good,
        FitFilter::Marginal,
    ];

    pub fn label(self) -> &'static str {
        match self {
            FitFilter::All => "All",
            FitFilter::Runnable => "Runnable",
            FitFilter::Perfect => "Perfect",
            FitFilter::Good => "Good",
            FitFilter::Marginal => "Marginal",
        }
    }

    pub fn next(self) -> FitFilter {
        next_in(&Self::CYCLE, self)
    }

    fn accepts(self, result: &FitResult) -> bool {
        match self {
            FitFilter::All => true,
            FitFilter::Runnable => result.is_runnable(),
            FitFilter::Perfect => result.fit == FitLevel::Perfect,
            FitFilter::Good => result.fit == FitLevel::Good,
            FitFilter::Marginal => result.fit == FitLevel::Marginal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    /// Every model in the catalog.
    All,
    /// Only models with a published GGUF build.
    Gguf,
    /// Only models a local runtime already has on disk.
    Installed,
}

impl Availability {
    const CYCLE: [Availability; 3] = [
        Availability::All,
        Availability::Gguf,
        Availability::Installed,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Availability::All => "All",
            Availability::Gguf => "GGUF",
            Availability::Installed => "Installed",
        }
    }

    pub fn next(self) -> Availability {
        next_in(&Self::CYCLE, self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortColumn {
    Score,
    Params,
    Speed,
    Memory,
    Download,
    Context,
    Date,
    UseCase,
}

impl SortColumn {
    const CYCLE: [SortColumn; 8] = [
        SortColumn::Score,
        SortColumn::Params,
        SortColumn::Speed,
        SortColumn::Memory,
        SortColumn::Download,
        SortColumn::Context,
        SortColumn::Date,
        SortColumn::UseCase,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SortColumn::Score => "Score",
            SortColumn::Params => "Params",
            SortColumn::Speed => "tok/s",
            SortColumn::Memory => "Mem%",
            SortColumn::Download => "Size",
            SortColumn::Context => "Ctx",
            SortColumn::Date => "Date",
            SortColumn::UseCase => "Use Case",
        }
    }

    pub fn next(self) -> SortColumn {
        next_in(&Self::CYCLE, self)
    }

    /// Order two results by this column, best first.
    fn compare(self, a: &FitResult, b: &FitResult) -> std::cmp::Ordering {
        match self {
            SortColumn::Score => b.scores.composite.total_cmp(&a.scores.composite),
            SortColumn::Params => b.params_b.total_cmp(&a.params_b),
            SortColumn::Speed => b.tokens_per_second.total_cmp(&a.tokens_per_second),
            SortColumn::Memory => b.mem_percent.total_cmp(&a.mem_percent),
            // Smallest download first: this column exists to answer "what is
            // cheap to fetch", so ascending is the useful direction.
            SortColumn::Download => a.download_gb.total_cmp(&b.download_gb),
            SortColumn::Context => b.context.cmp(&a.context),
            SortColumn::Date => b.released.cmp(&a.released),
            SortColumn::UseCase => a.use_case.as_str().cmp(b.use_case.as_str()),
        }
    }
}

/// Step to the next entry in a cycle, wrapping at the end.
fn next_in<T: Copy + PartialEq>(cycle: &[T], current: T) -> T {
    let index = cycle.iter().position(|&c| c == current).unwrap_or(0);
    cycle[(index + 1) % cycle.len()]
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

/// Which speed tunable each advanced-config field edits.
const SPEED_FIELDS: [(&str, &str, (f64, f64)); 4] = [
    (
        "Efficiency",
        "share of peak memory bandwidth inference achieves",
        (0.01, 1.0),
    ),
    (
        "GPU factor",
        "multiplier applied to fully-resident placements",
        (0.1, 2.0),
    ),
    (
        "CPU offload",
        "penalty when dense weights spill to system RAM",
        (0.1, 1.0),
    ),
    (
        "MoE offload",
        "penalty when inactive experts stream from RAM",
        (0.1, 1.0),
    ),
];

pub struct App {
    pub hw: Hardware,
    pub db: ModelDb,
    pub cfg: SpeedConfig,
    pub target: UseCase,

    /// Every model, analysed against the current hardware.
    pub results: Vec<FitResult>,
    /// Indices into `results`, after filtering and sorting.
    pub visible: Vec<usize>,
    pub selected: usize,
    /// `model_id` to index into `db.models`, so the detail panel does not scan
    /// the catalog on every frame.
    model_index: HashMap<String, usize>,

    pub mode: Mode,
    pub search: String,
    pub fit_filter: FitFilter,
    pub availability: Availability,
    pub sort: SortColumn,
    pub status: String,
    pub should_quit: bool,
    pub theme: Theme,

    /// What the last runtime probe found. Empty until discovery completes.
    pub discovery: Discovery,
    /// Tag currently downloading, if any.
    pub download_tag: Option<String>,
    events_rx: mpsc::Receiver<BackgroundEvent>,
    events_tx: mpsc::Sender<BackgroundEvent>,
    /// True while a discovery request is in flight, so `r` does not stack them.
    refreshing: bool,

    /// Editable popups.
    pub simulation: Form,
    pub speed_form: Form,

    /// Model marked for side-by-side comparison, held by id so that filtering
    /// or re-sorting cannot silently move the mark to a different model.
    marked_model_id: Option<String>,
}

impl App {
    pub fn new(hw: Hardware, db: ModelDb, target: UseCase, cfg: SpeedConfig) -> App {
        let (events_tx, events_rx) = mpsc::channel();
        let model_index = db
            .models
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id.clone(), i))
            .collect();

        let mut app = App {
            results: Vec::new(),
            visible: Vec::new(),
            selected: 0,
            model_index,
            mode: Mode::Normal,
            search: String::new(),
            fit_filter: FitFilter::All,
            availability: Availability::All,
            sort: SortColumn::Score,
            status: String::new(),
            should_quit: false,
            theme: Theme::from_index(Config::load().theme),
            simulation: simulation_form(&hw),
            speed_form: speed_form(&cfg),
            hw,
            db,
            cfg,
            target,
            discovery: Discovery::default(),
            download_tag: None,
            events_rx,
            events_tx,
            refreshing: false,
            marked_model_id: None,
        };
        app.recompute();
        app
    }

    /// Start the first runtime discovery in the background.
    ///
    /// Kept out of [`App::new`] so that the first frame draws immediately:
    /// probing six runtimes is fast but not instant, and a UI that appears
    /// only after the network settles feels broken.
    pub fn start_discovery(&mut self) {
        self.spawn_discovery();
    }

    fn spawn_discovery(&mut self) {
        if self.refreshing {
            return;
        }
        self.refreshing = true;
        let tx = self.events_tx.clone();
        thread::spawn(move || {
            let mut registry = ProviderRegistry::new();
            let runtimes = registry.discover();
            let result = registry.list_all_models().map(|models| Discovery {
                installed: InstalledIndex::from_models(models),
                runtimes,
            });
            let _ = tx.send(BackgroundEvent::Discovered(result));
        });
    }

    /// Persist the settings worth carrying into the next session.
    ///
    /// A failure here is reported in the status line and otherwise ignored:
    /// losing a theme preference must never stop the TUI from exiting.
    pub fn save_config(&mut self) {
        let config = Config {
            theme: self.theme.index(),
            use_case: self.target,
            speed: PersistedSpeed::from(&self.cfg),
        };
        if let Err(e) = config.save() {
            self.status = format!("could not save settings: {e}");
        }
    }

    // -- analysis ----------------------------------------------------------

    /// Re-run the whole fit analysis, e.g. after the use case changes.
    pub fn recompute(&mut self) {
        self.results = fit::analyze_all(&self.db.models, &self.hw, self.target, &self.cfg);
        self.refilter();
    }

    /// Rebuild the visible row set from the current filters and sort order.
    pub fn refilter(&mut self) {
        let anchor = self.selected_result().map(|r| r.model_id.clone());

        let query = self.search.trim().to_ascii_lowercase();
        let mut visible: Vec<usize> = self
            .results
            .iter()
            .enumerate()
            .filter(|(_, r)| self.fit_filter.accepts(r))
            .filter(|(_, r)| self.availability_accepts(r))
            .filter(|(_, r)| query.is_empty() || self.matches_query(r, &query))
            .map(|(i, _)| i)
            .collect();

        let results = &self.results;
        let sort = self.sort;
        visible.sort_by(|&a, &b| {
            let (x, y) = (&results[a], &results[b]);
            // Models that cannot run stay at the bottom whatever the sort is.
            y.is_runnable()
                .cmp(&x.is_runnable())
                .then_with(|| sort.compare(x, y))
                .then_with(|| x.name.cmp(&y.name))
        });
        self.visible = visible;

        // Keep the cursor on the same model where possible.
        self.selected = anchor
            .and_then(|id| {
                self.visible
                    .iter()
                    .position(|&i| self.results[i].model_id == id)
            })
            .unwrap_or(0);
        self.clamp_selection();
    }

    fn availability_accepts(&self, result: &FitResult) -> bool {
        match self.availability {
            Availability::All => true,
            Availability::Gguf => result.gguf,
            Availability::Installed => self.is_installed(result),
        }
    }

    /// Match a row against a search query.
    ///
    /// Delegates to the catalog entry so the TUI's `/` and the CLI's `search`
    /// answer identically; only the fit-derived fields are matched here.
    fn matches_query(&self, result: &FitResult, query: &str) -> bool {
        match self.model_for(result) {
            Some(model) => model.matches(query),
            // A result with no catalog entry cannot happen today, but falling
            // back to the name is cheaper than an unwrap that might one day fire.
            None => result.name.to_ascii_lowercase().contains(query),
        }
    }

    /// True when a local runtime already has this model on disk.
    pub fn is_installed(&self, result: &FitResult) -> bool {
        self.discovery
            .installed
            .contains(result.ollama.as_deref(), &result.model_id)
    }

    /// The runtime whose commands the detail panel should suggest.
    ///
    /// Prefers whatever is actually running. With nothing running, a model
    /// with an Ollama tag gets Ollama — by far the most common way to run one
    /// locally — and anything else gets llama.cpp, which loads any GGUF.
    pub fn suggested_runtime(&self, result: &FitResult) -> RuntimeKind {
        if let Some(live) = self.discovery.runtimes.first() {
            return live.kind;
        }
        if result.ollama.is_some() {
            RuntimeKind::Ollama
        } else {
            RuntimeKind::LlamaCpp
        }
    }

    /// The name that runtime knows this model by.
    pub fn model_reference(&self, result: &FitResult, kind: RuntimeKind) -> Option<String> {
        if kind.uses_own_registry() {
            // A registry-backed runtime cannot be handed an upstream repo id;
            // without a tag there is no command worth printing.
            result.ollama.clone()
        } else {
            Some(result.model_id.clone())
        }
    }

    /// Install and run commands for the selected model, when they exist.
    pub fn commands_for(&self, result: &FitResult) -> Option<(RuntimeKind, String, String)> {
        let kind = self.suggested_runtime(result);
        let reference = self.model_reference(result, kind)?;
        Some((
            kind,
            kind.install_command(&reference),
            kind.run_command(&reference),
        ))
    }

    fn runtime_names(discovery: &Discovery) -> String {
        discovery
            .runtimes
            .iter()
            .map(|r| r.name)
            .collect::<Vec<_>>()
            .join(", ")
    }

    // -- selection ---------------------------------------------------------

    pub fn selected_result(&self) -> Option<&FitResult> {
        self.visible.get(self.selected).map(|&i| &self.results[i])
    }

    /// The catalog entry behind a result.
    pub fn model_for(&self, result: &FitResult) -> Option<&Model> {
        self.model_index
            .get(&result.model_id)
            .and_then(|&i| self.db.models.get(i))
    }

    fn clamp_selection(&mut self) {
        self.selected = self
            .selected
            .min(self.visible.len().saturating_sub(1))
            .min(self.visible.len());
        if self.visible.is_empty() {
            self.selected = 0;
        }
    }

    pub fn move_selection(&mut self, delta: isize) {
        if self.visible.is_empty() {
            return;
        }
        let last = self.visible.len() as isize - 1;
        self.selected = (self.selected as isize + delta).clamp(0, last) as usize;
    }

    pub fn select_first(&mut self) {
        self.selected = 0;
    }

    pub fn select_last(&mut self) {
        self.selected = self.visible.len().saturating_sub(1);
    }

    // -- filter cycling ----------------------------------------------------

    pub fn cycle_fit_filter(&mut self) {
        self.fit_filter = self.fit_filter.next();
        self.status = format!("fit filter: {}", self.fit_filter.label());
        self.refilter();
    }

    pub fn cycle_availability(&mut self) {
        self.availability = self.availability.next();
        // An empty "installed" list looks like a bug when the real cause is
        // that discovery has not found anything yet.
        self.status = if self.availability == Availability::Installed
            && self.discovery.installed.is_empty()
        {
            "showing installed only — nothing detected yet, press r to re-probe".to_string()
        } else {
            format!("showing: {}", self.availability.label())
        };
        self.refilter();
    }

    pub fn cycle_sort(&mut self) {
        self.sort = self.sort.next();
        self.status = format!("sorted by {}", self.sort.label());
        self.refilter();
    }

    pub fn cycle_use_case(&mut self) {
        self.target = next_in(&UseCase::ALL, self.target);
        self.status = format!("use case: {}", self.target.as_str());
        self.recompute();
    }

    pub fn cycle_theme(&mut self) {
        self.theme = self.theme.next();
        self.status = format!("theme: {}", self.theme.name());
    }

    pub fn clear_search(&mut self) {
        self.search.clear();
        self.refilter();
    }

    // -- runtime integration ----------------------------------------------

    /// Kick off a background pull. No-op while another download is running,
    /// because two concurrent pulls compete for the same disk and neither
    /// finishes sooner.
    pub fn start_pull(&mut self, tag: String) {
        if let Some(running) = &self.download_tag {
            self.status = format!("already downloading {running}");
            return;
        }
        self.status = format!("pulling {tag}…");
        self.download_tag = Some(tag.clone());
        let tx = self.events_tx.clone();
        thread::spawn(move || {
            let result = Runtime::new(RuntimeKind::Ollama).pull(&tag);
            let _ = tx.send(BackgroundEvent::PullFinished { tag, result });
        });
    }

    /// Ask for a fresh installed-model list without blocking the event loop.
    pub fn refresh_installed(&mut self) {
        if self.refreshing {
            self.status = "already refreshing…".to_string();
            return;
        }
        self.status = "refreshing installed models…".to_string();
        self.spawn_discovery();
    }

    /// Drain finished background work. Call once per event-loop tick.
    pub fn poll_events(&mut self) {
        while let Ok(event) = self.events_rx.try_recv() {
            match event {
                BackgroundEvent::PullFinished { tag, result } => {
                    self.download_tag = None;
                    match result {
                        Ok(()) => {
                            self.discovery.installed.insert(&tag);
                            self.status = format!("pulled {tag}");
                            // An install can change what the availability
                            // filter shows.
                            self.refilter();
                        }
                        Err(e) => self.status = format!("pull of {tag} failed: {e}"),
                    }
                }
                BackgroundEvent::Discovered(result) => {
                    self.refreshing = false;
                    match result {
                        Ok(discovery) => {
                            let count = discovery.installed.len();
                            self.status = match (discovery.runtimes.len(), count) {
                                (0, _) => "no local runtime is running".to_string(),
                                (_, 0) => {
                                    format!("{} running, no models installed", Self::runtime_names(&discovery))
                                }
                                (_, n) => {
                                    format!("{} — {n} model(s) installed", Self::runtime_names(&discovery))
                                }
                            };
                            self.discovery = discovery;
                            self.refilter();
                        }
                        // Keep whatever was already known; a failed probe is
                        // not evidence that the models went away.
                        Err(e) => self.status = format!("refresh failed: {e}"),
                    }
                }
            }
        }
    }

    // -- popups ------------------------------------------------------------

    pub fn open_simulation(&mut self) {
        self.simulation.reset(&simulation_values(&self.hw));
        self.mode = Mode::SimulateHardware;
    }

    /// Apply the simulated hardware and re-rank against it.
    pub fn apply_simulation(&mut self) {
        let vram = self.simulation.parse(0);
        let ram = self.simulation.parse(1);
        let cores = self.simulation.parse(2).map(|c| c as usize);

        if let Some(ram) = ram {
            self.hw.total_ram_gb = ram;
            self.hw.available_ram_gb = ram;
        }
        if let Some(cores) = cores {
            self.hw.cpu_cores = cores;
            // Keep threads at least equal to cores; a machine cannot have
            // fewer hardware threads than physical cores.
            self.hw.cpu_threads = self.hw.cpu_threads.max(cores);
        }
        if let Some(vram) = vram {
            self.hw.set_vram(vram);
        }
        // Everything downstream reads this flag to badge the numbers as
        // hypothetical rather than measured.
        self.hw.simulated = true;

        self.mode = Mode::Normal;
        self.recompute();
        self.status = "hardware simulation applied".to_string();
    }

    pub fn reset_simulation_fields(&mut self) {
        self.simulation.reset(&simulation_values(&self.hw));
    }

    pub fn open_advanced_config(&mut self) {
        self.speed_form.reset(&speed_values(&self.cfg));
        self.mode = Mode::AdvancedConfig;
    }

    pub fn apply_advanced_config(&mut self) {
        if let Some(v) = self.speed_form.parse(0) {
            self.cfg.efficiency = v;
        }
        if let Some(v) = self.speed_form.parse(1) {
            self.cfg.gpu_factor = v;
        }
        if let Some(v) = self.speed_form.parse(2) {
            self.cfg.cpu_offload_factor = v;
        }
        if let Some(v) = self.speed_form.parse(3) {
            self.cfg.moe_offload_factor = v;
        }
        self.mode = Mode::Normal;
        self.recompute();
        self.status = "speed model updated".to_string();
    }

    pub fn close_popup(&mut self) {
        self.mode = Mode::Normal;
    }

    /// The form the current mode is editing, if any.
    pub fn active_form_mut(&mut self) -> Option<&mut Form> {
        match self.mode {
            Mode::SimulateHardware => Some(&mut self.simulation),
            Mode::AdvancedConfig => Some(&mut self.speed_form),
            _ => None,
        }
    }

    // -- comparison --------------------------------------------------------

    /// Mark or unmark the selected model for comparison.
    pub fn mark_for_comparison(&mut self) {
        let Some(id) = self.selected_result().map(|r| r.model_id.clone()) else {
            return;
        };
        if self.marked_model_id.as_ref() == Some(&id) {
            self.marked_model_id = None;
            self.status = "comparison mark cleared".to_string();
        } else {
            let name = self.selected_result().map(|r| r.name.clone()).unwrap_or(id.clone());
            self.marked_model_id = Some(id);
            self.status = format!("marked {name} — press c to compare");
        }
    }

    /// The marked model's analysis, resolved by id so that it survives any
    /// change to the filters or the sort order.
    pub fn marked_result(&self) -> Option<&FitResult> {
        let id = self.marked_model_id.as_ref()?;
        self.results.iter().find(|r| &r.model_id == id)
    }

    /// True when there are two distinct models to compare.
    pub fn can_compare(&self) -> bool {
        match (self.marked_result(), self.selected_result()) {
            (Some(marked), Some(selected)) => marked.model_id != selected.model_id,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Form construction
// ---------------------------------------------------------------------------

fn simulation_values(hw: &Hardware) -> Vec<String> {
    vec![
        format!("{:.1}", hw.total_vram_gb()),
        format!("{:.1}", hw.total_ram_gb),
        hw.cpu_cores.to_string(),
    ]
}

fn simulation_form(hw: &Hardware) -> Form {
    let values = simulation_values(hw);
    Form::new(vec![
        Field::new(
            "VRAM (GB)",
            "accelerator memory available to the model",
            values[0].clone(),
            (0.0, 2048.0),
        ),
        Field::new(
            "RAM (GB)",
            "system memory, the ceiling for CPU and hybrid placement",
            values[1].clone(),
            (0.5, 8192.0),
        ),
        Field::new(
            "CPU cores",
            "physical cores; affects CPU-only throughput",
            values[2].clone(),
            (1.0, 512.0),
        ),
    ])
}

fn speed_values(cfg: &SpeedConfig) -> Vec<String> {
    vec![
        format!("{:.2}", cfg.efficiency),
        format!("{:.2}", cfg.gpu_factor),
        format!("{:.2}", cfg.cpu_offload_factor),
        format!("{:.2}", cfg.moe_offload_factor),
    ]
}

fn speed_form(cfg: &SpeedConfig) -> Form {
    let values = speed_values(cfg);
    Form::new(
        SPEED_FIELDS
            .iter()
            .zip(values)
            .map(|((label, help, range), value)| Field::new(label, help, value, *range))
            .collect(),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::hardware::Backend;

    pub(crate) fn test_hardware(vram: f64, ram: f64) -> Hardware {
        let mut hw = Hardware {
            cpu_brand: "Test CPU".into(),
            cpu_cores: 8,
            cpu_threads: 16,
            arch: "x86_64".into(),
            total_ram_gb: ram,
            available_ram_gb: ram * 0.75,
            gpus: Vec::new(),
            backend: Backend::CpuX86,
            simulated: true,
        };
        if vram > 0.0 {
            hw.set_vram(vram);
        }
        hw
    }

    /// An app that has never touched the network: `start_discovery` is not
    /// called, so `installed` stays empty and every test is deterministic.
    pub(crate) fn test_app() -> App {
        App::new(
            test_hardware(12.0, 32.0),
            ModelDb::embedded(),
            UseCase::General,
            SpeedConfig::default(),
        )
    }

    #[test]
    fn starts_with_every_model_visible() {
        let app = test_app();
        assert_eq!(app.visible.len(), app.results.len());
        assert!(app.selected_result().is_some());
    }

    #[test]
    fn search_narrows_the_list() {
        let mut app = test_app();
        app.search = "qwen".to_string();
        app.refilter();
        assert!(!app.visible.is_empty());
        assert!(app.visible.iter().all(|&i| {
            let r = &app.results[i];
            [&r.provider, &r.name, &r.model_id]
                .iter()
                .any(|f| f.to_ascii_lowercase().contains("qwen"))
        }));
    }

    #[test]
    fn search_matches_the_same_way_the_cli_does() {
        // The TUI's `/` and the CLI's `search` must not disagree, or a model
        // findable one way is invisible the other.
        let mut app = test_app();
        app.search = "coding".to_string();
        app.refilter();
        let tui: Vec<&str> = app
            .visible
            .iter()
            .map(|&i| app.results[i].model_id.as_str())
            .collect();
        let cli: Vec<&str> = app
            .db
            .models
            .iter()
            .filter(|m| m.matches("coding"))
            .map(|m| m.id.as_str())
            .collect();
        assert_eq!(tui.len(), cli.len());
        assert!(cli.iter().all(|id| tui.contains(id)));
    }

    #[test]
    fn fit_filter_cycles_through_every_level_and_restricts() {
        let mut app = test_app();
        let mut seen = vec![app.fit_filter];
        for _ in 0..FitFilter::CYCLE.len() - 1 {
            app.cycle_fit_filter();
            seen.push(app.fit_filter);
        }
        assert_eq!(seen.len(), FitFilter::CYCLE.len());
        app.cycle_fit_filter();
        assert_eq!(app.fit_filter, FitFilter::All, "the cycle wraps");

        app.fit_filter = FitFilter::Perfect;
        app.refilter();
        assert!(
            app.visible
                .iter()
                .all(|&i| app.results[i].fit == FitLevel::Perfect)
        );
    }

    #[test]
    fn availability_filter_covers_gguf_and_installed() {
        let mut app = test_app();
        app.availability = Availability::Gguf;
        app.refilter();
        assert!(app.visible.iter().all(|&i| app.results[i].gguf));

        app.availability = Availability::Installed;
        app.refilter();
        assert!(
            app.visible.is_empty(),
            "nothing is installed in a test app, so the filter shows nothing"
        );

        // Once a model is known to be installed it reappears.
        let tag = app
            .results
            .iter()
            .find_map(|r| r.ollama.clone())
            .expect("catalog has ollama tags");
        app.discovery.installed.insert(&tag);
        app.refilter();
        assert_eq!(app.visible.len(), 1);
    }

    #[test]
    fn selection_survives_refilter() {
        let mut app = test_app();
        app.move_selection(3);
        let id = app.selected_result().unwrap().model_id.clone();
        app.cycle_sort();
        assert_eq!(app.selected_result().unwrap().model_id, id);
    }

    #[test]
    fn navigation_stays_in_bounds() {
        let mut app = test_app();
        app.move_selection(-10);
        assert_eq!(app.selected, 0);
        app.select_last();
        let last = app.visible.len() - 1;
        assert_eq!(app.selected, last);
        app.move_selection(50);
        assert_eq!(app.selected, last);
    }

    #[test]
    fn empty_result_set_does_not_panic() {
        let mut app = test_app();
        app.search = "no-such-model-anywhere".to_string();
        app.refilter();
        assert!(app.visible.is_empty());
        assert!(app.selected_result().is_none());
        app.move_selection(1);
        app.select_last();
        assert_eq!(app.selected, 0);
    }

    #[test]
    fn changing_use_case_reranks() {
        let mut app = test_app();
        app.target = UseCase::Coding;
        app.recompute();
        let top = &app.results[app.visible[0]];
        assert!(top.scores.composite > 0.0);
        // Cycling visits every use case and comes back round.
        let start = app.target;
        for _ in 0..UseCase::ALL.len() {
            app.cycle_use_case();
        }
        assert_eq!(app.target, start);
    }

    #[test]
    fn sorting_by_download_size_puts_the_smallest_first() {
        let mut app = test_app();
        app.fit_filter = FitFilter::Runnable;
        app.sort = SortColumn::Download;
        app.refilter();
        let sizes: Vec<f64> = app
            .visible
            .iter()
            .map(|&i| app.results[i].download_gb)
            .collect();
        assert!(
            sizes.windows(2).all(|w| w[0] <= w[1]),
            "download column should ascend, got {sizes:?}"
        );
    }

    #[test]
    fn unrunnable_models_stay_last_under_every_sort() {
        let mut app = test_app();
        let mut sort = SortColumn::Score;
        for _ in 0..SortColumn::CYCLE.len() {
            app.sort = sort;
            app.refilter();
            let first_bad = app
                .visible
                .iter()
                .position(|&i| !app.results[i].is_runnable());
            if let Some(idx) = first_bad {
                assert!(
                    app.visible[idx..]
                        .iter()
                        .all(|&i| !app.results[i].is_runnable()),
                    "runnable model found after an unrunnable one under {}",
                    sort.label()
                );
            }
            sort = sort.next();
        }
    }

    #[test]
    fn comparison_mark_follows_the_model_not_the_row() {
        let mut app = test_app();
        app.move_selection(2);
        let marked_id = app.selected_result().unwrap().model_id.clone();
        app.mark_for_comparison();
        assert_eq!(app.marked_result().unwrap().model_id, marked_id);

        // Re-sorting and filtering move rows around; the mark must not follow
        // the old row index onto a different model.
        app.cycle_sort();
        app.search = "qwen".to_string();
        app.refilter();
        assert_eq!(
            app.marked_result().unwrap().model_id,
            marked_id,
            "the mark drifted to another model"
        );
    }

    #[test]
    fn marking_the_same_model_twice_clears_it() {
        let mut app = test_app();
        app.mark_for_comparison();
        assert!(app.marked_result().is_some());
        app.mark_for_comparison();
        assert!(app.marked_result().is_none());
        assert!(!app.can_compare());
    }

    #[test]
    fn comparing_a_model_with_itself_is_refused() {
        let mut app = test_app();
        app.mark_for_comparison();
        assert!(!app.can_compare(), "the mark is on the selected row");
        app.move_selection(1);
        assert!(app.can_compare());
    }

    #[test]
    fn simulation_applies_values_and_flags_the_numbers_as_simulated() {
        let mut app = test_app();
        app.hw.simulated = false;
        app.open_simulation();
        assert_eq!(app.mode, Mode::SimulateHardware);

        app.simulation.clear_active();
        for c in "48".chars() {
            app.simulation.push(c);
        }
        app.simulation.focus_next();
        app.simulation.clear_active();
        for c in "128".chars() {
            app.simulation.push(c);
        }
        app.apply_simulation();

        assert_eq!(app.mode, Mode::Normal);
        assert!((app.hw.total_vram_gb() - 48.0).abs() < 0.01);
        assert!((app.hw.total_ram_gb - 128.0).abs() < 0.01);
        assert!(app.hw.simulated, "simulated numbers must be badged as such");
    }

    #[test]
    fn simulation_leaves_blank_fields_alone() {
        let mut app = test_app();
        let ram_before = app.hw.total_ram_gb;
        app.open_simulation();
        app.simulation.focus_next();
        app.simulation.clear_active();
        app.apply_simulation();
        assert!((app.hw.total_ram_gb - ram_before).abs() < 0.01);
    }

    #[test]
    fn simulation_never_reports_fewer_threads_than_cores() {
        let mut app = test_app();
        app.open_simulation();
        // Field 2 is the core count; ask for more cores than there are threads.
        app.simulation.focus_next();
        app.simulation.focus_next();
        app.simulation.clear_active();
        for c in "64".chars() {
            app.simulation.push(c);
        }
        app.apply_simulation();
        assert_eq!(app.hw.cpu_cores, 64);
        assert!(app.hw.cpu_threads >= app.hw.cpu_cores);
    }

    #[test]
    fn speed_config_clamps_and_reranks() {
        let mut app = test_app();
        app.open_advanced_config();
        app.speed_form.clear_active();
        for c in "9.9".chars() {
            app.speed_form.push(c);
        }
        app.apply_advanced_config();
        assert_eq!(app.mode, Mode::Normal);
        assert!(
            (app.cfg.efficiency - 1.0).abs() < 1e-9,
            "efficiency should clamp to its 0.01..1.0 range, got {}",
            app.cfg.efficiency
        );
    }

    #[test]
    fn the_active_form_follows_the_mode() {
        let mut app = test_app();
        assert!(app.active_form_mut().is_none());
        app.open_simulation();
        assert!(app.active_form_mut().is_some());
        app.open_advanced_config();
        assert!(app.active_form_mut().is_some());
        app.close_popup();
        assert!(app.active_form_mut().is_none());
    }

    #[test]
    fn a_second_pull_is_refused_while_one_is_running() {
        let mut app = test_app();
        app.download_tag = Some("busy:7b".to_string());
        app.start_pull("other:7b".to_string());
        assert_eq!(app.download_tag.as_deref(), Some("busy:7b"));
        assert!(app.status.contains("already downloading"));
    }

    #[test]
    fn a_finished_pull_marks_the_model_installed() {
        let mut app = test_app();
        app.download_tag = Some("qwen2.5:7b".to_string());
        app.events_tx
            .send(BackgroundEvent::PullFinished {
                tag: "qwen2.5:7b".to_string(),
                result: Ok(()),
            })
            .unwrap();
        app.poll_events();
        assert!(app.download_tag.is_none());
        assert!(app.discovery.installed.contains(Some("qwen2.5:7b"), "whatever/model"));
        assert!(app.status.contains("pulled"));
    }

    #[test]
    fn a_failed_pull_reports_the_error_and_clears_the_slot() {
        let mut app = test_app();
        app.download_tag = Some("broken:7b".to_string());
        app.events_tx
            .send(BackgroundEvent::PullFinished {
                tag: "broken:7b".to_string(),
                result: Err("connection refused".to_string()),
            })
            .unwrap();
        app.poll_events();
        assert!(app.download_tag.is_none());
        assert!(app.status.contains("connection refused"));
    }

    #[test]
    fn a_discovery_result_updates_the_installed_set() {
        let mut app = test_app();
        app.refreshing = true;
        let mut installed = InstalledIndex::default();
        installed.insert("qwen2.5:7b");
        installed.insert("phi-4");
        app.events_tx
            .send(BackgroundEvent::Discovered(Ok(Discovery {
                installed,
                runtimes: vec![DiscoveredRuntime {
                    kind: RuntimeKind::Ollama,
                    name: RuntimeKind::Ollama.label(),
                    base_url: "http://127.0.0.1:11434".into(),
                    model_count: 2,
                    disk_gb: Some(9.0),
                }],
            })))
            .unwrap();
        app.poll_events();
        assert_eq!(app.discovery.installed.len(), 2);
        assert_eq!(app.discovery.runtimes.len(), 1);
        assert!(app.status.contains("Ollama"), "{}", app.status);
        assert!(!app.refreshing, "the refresh slot is released");
    }

    #[test]
    fn discovery_with_no_runtime_says_so() {
        let mut app = test_app();
        app.refreshing = true;
        app.events_tx
            .send(BackgroundEvent::Discovered(Ok(Discovery::default())))
            .unwrap();
        app.poll_events();
        assert!(app.status.contains("no local runtime"), "{}", app.status);
    }

    #[test]
    fn a_failed_discovery_is_reported_without_wiping_what_is_known() {
        let mut app = test_app();
        app.discovery.installed.insert("qwen2.5:7b");
        app.refreshing = true;
        app.events_tx
            .send(BackgroundEvent::Discovered(Err("no runtime".into())))
            .unwrap();
        app.poll_events();
        assert_eq!(app.discovery.installed.len(), 1, "known models are not lost");
        assert!(app.status.contains("no runtime"));
    }

    #[test]
    fn suggested_commands_follow_the_running_runtime() {
        let mut app = test_app();
        let result = app.results[0].clone();

        // With nothing running, a tagged model gets Ollama.
        let tagged = app
            .results
            .iter()
            .find(|r| r.ollama.is_some())
            .expect("catalog has tagged models")
            .clone();
        assert_eq!(app.suggested_runtime(&tagged), RuntimeKind::Ollama);
        let (kind, install, run) = app.commands_for(&tagged).expect("a tagged model has commands");
        assert_eq!(kind, RuntimeKind::Ollama);
        assert!(install.starts_with("ollama pull"));
        assert!(run.starts_with("ollama run"));

        // A live runtime wins over the fallback.
        app.discovery.runtimes = vec![DiscoveredRuntime {
            kind: RuntimeKind::Vllm,
            name: RuntimeKind::Vllm.label(),
            base_url: "http://127.0.0.1:8000".into(),
            model_count: 0,
            disk_gb: None,
        }];
        assert_eq!(app.suggested_runtime(&result), RuntimeKind::Vllm);
        let (_, install, _) = app.commands_for(&result).unwrap();
        assert!(install.contains(&result.model_id), "{install}");
    }

    #[test]
    fn a_registry_runtime_without_a_tag_has_no_command_to_offer() {
        let mut app = test_app();
        // Ollama cannot be handed an upstream repo id, so an untagged model
        // yields no command rather than a wrong one.
        app.discovery.runtimes = vec![DiscoveredRuntime {
            kind: RuntimeKind::Ollama,
            name: RuntimeKind::Ollama.label(),
            base_url: "http://127.0.0.1:11434".into(),
            model_count: 0,
            disk_gb: None,
        }];
        let untagged = app.results.iter().find(|r| r.ollama.is_none()).cloned();
        if let Some(result) = untagged {
            assert!(app.commands_for(&result).is_none());
        }
    }

    #[test]
    fn model_lookup_is_indexed_not_scanned() {
        let app = test_app();
        for result in app.results.iter().take(20) {
            let model = app.model_for(result).expect("every result has an entry");
            assert_eq!(model.id, result.model_id);
        }
    }

    #[test]
    fn mode_classification_is_exhaustive() {
        // Every mode either draws a panel, draws a popup, or is plain — and
        // never two of those at once, which would render both.
        for mode in [
            Mode::Normal,
            Mode::Search,
            Mode::Help,
            Mode::Detail,
            Mode::Plan,
            Mode::SimulateHardware,
            Mode::AdvancedConfig,
            Mode::Comparison,
        ] {
            assert!(!(mode.is_panel() && mode.is_popup()), "{mode:?}");
        }
        assert!(Mode::Detail.is_panel());
        assert!(Mode::Help.is_popup());
        assert!(!Mode::Normal.is_panel() && !Mode::Normal.is_popup());
    }
}
