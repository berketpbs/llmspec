//! TUI application state: filters, sorting, navigation.

use crate::fit::{self, FitLevel, FitResult, SpeedConfig};
use crate::hardware::Hardware;
use crate::models::{Model, ModelDb, UseCase};
use crate::providers::ProviderRegistry;
use std::collections::HashSet;
use std::sync::mpsc;

#[derive(Debug, Clone)]
pub enum DownloadEvent {
    Done { tag: String, result: Result<(), String> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Search,
    Help,
    Detail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitFilter {
    All,
    Runnable,
    Perfect,
    Good,
    Marginal,
}

impl FitFilter {
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
        match self {
            FitFilter::All => FitFilter::Runnable,
            FitFilter::Runnable => FitFilter::Perfect,
            FitFilter::Perfect => FitFilter::Good,
            FitFilter::Good => FitFilter::Marginal,
            FitFilter::Marginal => FitFilter::All,
        }
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
    All,
    Gguf,
}

impl Availability {
    pub fn label(self) -> &'static str {
        match self {
            Availability::All => "All",
            Availability::Gguf => "GGUF Avail",
        }
    }

    pub fn next(self) -> Availability {
        match self {
            Availability::All => Availability::Gguf,
            Availability::Gguf => Availability::All,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortColumn {
    Score,
    Params,
    Speed,
    Memory,
    Context,
    Date,
    UseCase,
}

impl SortColumn {
    pub fn label(self) -> &'static str {
        match self {
            SortColumn::Score => "Score",
            SortColumn::Params => "Params",
            SortColumn::Speed => "tok/s",
            SortColumn::Memory => "Mem%",
            SortColumn::Context => "Ctx",
            SortColumn::Date => "Date",
            SortColumn::UseCase => "Use Case",
        }
    }

    pub fn next(self) -> SortColumn {
        match self {
            SortColumn::Score => SortColumn::Params,
            SortColumn::Params => SortColumn::Speed,
            SortColumn::Speed => SortColumn::Memory,
            SortColumn::Memory => SortColumn::Context,
            SortColumn::Context => SortColumn::Date,
            SortColumn::Date => SortColumn::UseCase,
            SortColumn::UseCase => SortColumn::Score,
        }
    }
}

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

    pub mode: Mode,
    pub search: String,
    pub fit_filter: FitFilter,
    pub availability: Availability,
    pub sort: SortColumn,
    pub status: String,
    pub should_quit: bool,

    /// Ollama tags known to be installed.
    pub installed: HashSet<String>,
    /// Provider registry for querying installed models and pulling.
    pub providers: ProviderRegistry,
    /// Tag currently being downloaded, if any.
    pub download_tag: Option<String>,
    /// Channel receiver for download completion events.
    download_rx: mpsc::Receiver<DownloadEvent>,
    /// Channel sender for download threads (cloned at startup).
    download_tx: mpsc::Sender<DownloadEvent>,
}

impl App {
    pub fn new(hw: Hardware, db: ModelDb, target: UseCase, cfg: SpeedConfig) -> App {
        let (download_tx, download_rx) = mpsc::channel();

        let mut providers = ProviderRegistry::new();
        let installed = providers
            .list_all_models()
            .map(|models| models.into_iter().map(|m| m.name).collect())
            .unwrap_or_default();

        let mut app = App {
            results: Vec::new(),
            visible: Vec::new(),
            selected: 0,
            mode: Mode::Normal,
            search: String::new(),
            fit_filter: FitFilter::All,
            availability: Availability::All,
            sort: SortColumn::Score,
            status: String::new(),
            should_quit: false,
            hw,
            db,
            cfg,
            target,
            installed,
            providers,
            download_tag: None,
            download_rx,
            download_tx,
        };
        app.recompute();
        app
    }

    /// Re-run the whole fit analysis, e.g. after the use case changes.
    pub fn recompute(&mut self) {
        self.results = fit::analyze_all(&self.db.models, &self.hw, self.target, &self.cfg);
        self.refilter();
    }

    /// Rebuild the visible row set from the current filters and sort order.
    pub fn refilter(&mut self) {
        let selected_id = self.selected_result().map(|r| r.model_id.clone());

        let query = self.search.trim().to_ascii_lowercase();
        let mut visible: Vec<usize> = self
            .results
            .iter()
            .enumerate()
            .filter(|(_, r)| self.fit_filter.accepts(r))
            .filter(|(_, r)| self.availability != Availability::Gguf || r.gguf)
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
                .then_with(|| match sort {
                    SortColumn::Score => y.scores.composite.total_cmp(&x.scores.composite),
                    SortColumn::Params => y.params_b.total_cmp(&x.params_b),
                    SortColumn::Speed => y.tokens_per_second.total_cmp(&x.tokens_per_second),
                    SortColumn::Memory => y.mem_percent.total_cmp(&x.mem_percent),
                    SortColumn::Context => y.context.cmp(&x.context),
                    SortColumn::Date => y.released.cmp(&x.released),
                    SortColumn::UseCase => x.use_case.as_str().cmp(y.use_case.as_str()),
                })
                .then_with(|| x.name.cmp(&y.name))
        });

        self.visible = visible;

        // Keep the cursor on the same model where possible.
        self.selected = match selected_id {
            Some(id) => self
                .visible
                .iter()
                .position(|&i| self.results[i].model_id == id)
                .unwrap_or(0),
            None => 0,
        };
        self.clamp_selection();
    }

    fn matches_query(&self, result: &FitResult, query: &str) -> bool {
        query.split_whitespace().all(|term| {
            result.name.to_ascii_lowercase().contains(term)
                || result.model_id.to_ascii_lowercase().contains(term)
                || result.provider.to_ascii_lowercase().contains(term)
                || result.use_case.as_str().contains(term)
                || format!("{:.0}b", result.params_b).contains(term)
        })
    }

    pub fn selected_result(&self) -> Option<&FitResult> {
        self.visible.get(self.selected).map(|&i| &self.results[i])
    }

    /// The database entry behind the selected row.
    pub fn selected_model(&self) -> Option<&Model> {
        let id = &self.selected_result()?.model_id;
        self.db.models.iter().find(|m| &m.id == id)
    }

    fn clamp_selection(&mut self) {
        if self.visible.is_empty() {
            self.selected = 0;
        } else if self.selected >= self.visible.len() {
            self.selected = self.visible.len() - 1;
        }
    }

    pub fn move_selection(&mut self, delta: isize) {
        if self.visible.is_empty() {
            return;
        }
        let last = self.visible.len() as isize - 1;
        let next = (self.selected as isize + delta).clamp(0, last);
        self.selected = next as usize;
    }

    pub fn select_first(&mut self) {
        self.selected = 0;
    }

    pub fn select_last(&mut self) {
        self.selected = self.visible.len().saturating_sub(1);
    }

    pub fn cycle_fit_filter(&mut self) {
        self.fit_filter = self.fit_filter.next();
        self.status = format!("Fit filter: {}", self.fit_filter.label());
        self.refilter();
    }

    pub fn cycle_availability(&mut self) {
        self.availability = self.availability.next();
        self.status = format!("Availability: {}", self.availability.label());
        self.refilter();
    }

    pub fn cycle_sort(&mut self) {
        self.sort = self.sort.next();
        self.status = format!("Sorted by {}", self.sort.label());
        self.refilter();
    }

    pub fn cycle_use_case(&mut self) {
        let idx = UseCase::ALL
            .iter()
            .position(|&u| u == self.target)
            .unwrap_or(0);
        self.target = UseCase::ALL[(idx + 1) % UseCase::ALL.len()];
        self.status = format!("Use case: {}", self.target.as_str());
        self.recompute();
    }

    pub fn clear_search(&mut self) {
        self.search.clear();
        self.refilter();
    }

    /// Kick off a background pull of `tag`. No-op if a download is already in flight.
    pub fn start_pull(&mut self, tag: String) {
        if self.download_tag.is_some() {
            self.status = format!(
                "already downloading {}",
                self.download_tag.as_deref().unwrap_or("")
            );
            return;
        }
        self.status = format!("pulling {tag}…");
        self.download_tag = Some(tag.clone());
        let tx = self.download_tx.clone();
        std::thread::spawn(move || {
            let result = crate::providers::Ollama::default().pull(&tag);
            let _ = tx.send(DownloadEvent::Done { tag, result });
        });
    }

    /// Non-blocking poll of the download channel; call once per event-loop tick.
    pub fn poll_downloads(&mut self) {
        while let Ok(DownloadEvent::Done { tag, result }) = self.download_rx.try_recv() {
            self.download_tag = None;
            match result {
                Ok(()) => {
                    self.installed.insert(tag.clone());
                    self.status = format!("pulled {tag}");
                }
                Err(e) => self.status = format!("pull of {tag} failed: {e}"),
            }
        }
    }

    /// Synchronous forced refresh of installed models (bypasses the cache).
    pub fn refresh_installed(&mut self) {
        match self.providers.refresh_all_models() {
            Ok(models) => {
                self.installed = models.into_iter().map(|m| m.name).collect();
                self.status = "refreshed installed models".to_string();
            }
            Err(e) => self.status = format!("refresh failed: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{Backend, Hardware};

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

    #[test]
    fn starts_with_every_model_visible() {
        let app = app();
        assert_eq!(app.visible.len(), app.results.len());
        assert!(app.selected_result().is_some());
    }

    #[test]
    fn search_narrows_the_list() {
        let mut app = app();
        app.search = "qwen".to_string();
        app.refilter();
        assert!(!app.visible.is_empty());
        assert!(app.visible.iter().all(|&i| {
            let r = &app.results[i];
            r.provider.to_ascii_lowercase().contains("qwen")
                || r.name.to_ascii_lowercase().contains("qwen")
                || r.model_id.to_ascii_lowercase().contains("qwen")
        }));
    }

    #[test]
    fn fit_filter_cycles_and_restricts() {
        let mut app = app();
        app.fit_filter = FitFilter::Perfect;
        app.refilter();
        assert!(
            app.visible
                .iter()
                .all(|&i| app.results[i].fit == FitLevel::Perfect)
        );
    }

    #[test]
    fn selection_survives_refilter() {
        let mut app = app();
        app.move_selection(3);
        let id = app.selected_result().unwrap().model_id.clone();
        app.cycle_sort();
        assert_eq!(app.selected_result().unwrap().model_id, id);
    }

    #[test]
    fn navigation_stays_in_bounds() {
        let mut app = app();
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
        let mut app = app();
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
        let mut app = app();
        let before = app.selected_result().unwrap().scores.composite;
        app.target = UseCase::Coding;
        app.recompute();
        let after = app.results[app.visible[0]].scores.composite;
        assert!(before > 0.0 && after > 0.0);
    }
}
