//! Colour themes for the TUI.
//!
//! Every widget resolves its colours through a [`Palette`] rather than naming
//! them directly, so adding a theme is a table entry and cannot leave half the
//! screen on the old colours.
//!
//! The default theme uses the terminal's own named colours. That is
//! deliberate: it inherits whatever palette the user already configured and
//! stays readable on a light background, which a hardcoded dark theme does
//! not. The named themes are RGB, because a Dracula that renders in the
//! terminal's idea of "magenta" is not Dracula.
//!
//! Themes are identified in the config file by [`Theme::slug`], not by their
//! position in [`Theme::ALL`]. Older configs stored a bare index, so
//! [`Theme::from_legacy_index`] still resolves those against the ordering that
//! shipped first — which is why [`Theme::ALL`] is free to be reordered into
//! groups without silently changing anyone's theme.

use ratatui::style::Color;

use crate::config::ThemeRef;

/// Semantic colours the widgets draw with.
///
/// Named by role rather than by hue, so a theme decides what "a model that
/// barely fits" looks like without any widget having to agree in advance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Palette {
    /// Titles, active fields, the use-case badge.
    pub accent: Color,
    /// Body text.
    pub text: Color,
    /// Labels and anything secondary.
    pub dim: Color,
    /// Background of the selected row.
    pub selection: Color,
    /// Perfect fit, high scores, an installed model.
    pub good: Color,
    /// Good fit, mid scores.
    pub ok: Color,
    /// Marginal fit, low scores, warnings.
    pub warn: Color,
    /// Does not fit, errors.
    pub bad: Color,
    /// MoE placement — distinct from the fit colours on purpose, because it
    /// describes *how* a model runs, not *how well*.
    pub special: Color,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    Default,
    // Ports of established editor palettes.
    Dracula,
    Nord,
    Solarized,
    Gruvbox,
    Monokai,
    TokyoNight,
    CatppuccinMocha,
    RosePine,
    Everforest,
    Kanagawa,
    OneDark,
    // Palettes original to llmspec.
    Ocean,
    Forest,
    Sunset,
    Slate,
    Aurora,
    Matrix,
    Sakura,
    // Palettes chosen for legibility rather than looks.
    ColourblindSafe,
    HighContrast,
}

impl Theme {
    /// Cycling order, grouped: the default, then editor ports, then llmspec's
    /// own palettes, then the accessibility pair. Safe to reorder — nothing
    /// persists a position in this list.
    pub const ALL: [Theme; 21] = [
        Theme::Default,
        Theme::Dracula,
        Theme::Nord,
        Theme::Solarized,
        Theme::Gruvbox,
        Theme::Monokai,
        Theme::TokyoNight,
        Theme::CatppuccinMocha,
        Theme::RosePine,
        Theme::Everforest,
        Theme::Kanagawa,
        Theme::OneDark,
        Theme::Ocean,
        Theme::Forest,
        Theme::Sunset,
        Theme::Slate,
        Theme::Aurora,
        Theme::Matrix,
        Theme::Sakura,
        Theme::ColourblindSafe,
        Theme::HighContrast,
    ];

    /// The ordering the first release persisted as a bare integer. Frozen: a
    /// config written before themes were named still has to resolve to the
    /// theme its author picked, so nothing may be inserted or reordered here.
    const LEGACY_ORDER: [Theme; 10] = [
        Theme::Default,
        Theme::Dracula,
        Theme::Nord,
        Theme::Solarized,
        Theme::Gruvbox,
        Theme::Monokai,
        Theme::TokyoNight,
        Theme::Ocean,
        Theme::Forest,
        Theme::Sunset,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Theme::Default => "Default",
            Theme::Dracula => "Dracula",
            Theme::Nord => "Nord",
            Theme::Solarized => "Solarized",
            Theme::Gruvbox => "Gruvbox",
            Theme::Monokai => "Monokai",
            Theme::TokyoNight => "Tokyo Night",
            Theme::CatppuccinMocha => "Catppuccin Mocha",
            Theme::RosePine => "Rosé Pine",
            Theme::Everforest => "Everforest",
            Theme::Kanagawa => "Kanagawa",
            Theme::OneDark => "One Dark",
            Theme::Ocean => "Ocean",
            Theme::Forest => "Forest",
            Theme::Sunset => "Sunset",
            Theme::Slate => "Slate",
            Theme::Aurora => "Aurora",
            Theme::Matrix => "Matrix",
            Theme::Sakura => "Sakura",
            Theme::ColourblindSafe => "Colourblind Safe",
            Theme::HighContrast => "High Contrast",
        }
    }

    /// Stable identifier written to the config file. Unlike [`Theme::name`]
    /// this is part of the file format: renaming one orphans a saved
    /// preference, so it never changes once shipped.
    pub fn slug(self) -> &'static str {
        match self {
            Theme::Default => "default",
            Theme::Dracula => "dracula",
            Theme::Nord => "nord",
            Theme::Solarized => "solarized",
            Theme::Gruvbox => "gruvbox",
            Theme::Monokai => "monokai",
            Theme::TokyoNight => "tokyo-night",
            Theme::CatppuccinMocha => "catppuccin-mocha",
            Theme::RosePine => "rose-pine",
            Theme::Everforest => "everforest",
            Theme::Kanagawa => "kanagawa",
            Theme::OneDark => "one-dark",
            Theme::Ocean => "ocean",
            Theme::Forest => "forest",
            Theme::Sunset => "sunset",
            Theme::Slate => "slate",
            Theme::Aurora => "aurora",
            Theme::Matrix => "matrix",
            Theme::Sakura => "sakura",
            Theme::ColourblindSafe => "colourblind-safe",
            Theme::HighContrast => "high-contrast",
        }
    }

    /// Theme for a slug, tolerating the spellings a hand-edited config is
    /// likely to contain: any case, underscores or spaces for hyphens, and the
    /// American spelling of "colourblind".
    pub fn from_slug(slug: &str) -> Option<Theme> {
        let normalised = slug.trim().to_ascii_lowercase().replace(['_', ' '], "-");
        if normalised == "colorblind-safe" {
            return Some(Theme::ColourblindSafe);
        }
        Theme::ALL.iter().copied().find(|t| t.slug() == normalised)
    }

    /// Position in [`Theme::ALL`]. Cycling only — this is no longer persisted.
    pub fn index(self) -> usize {
        Theme::ALL.iter().position(|&t| t == self).unwrap_or(0)
    }

    /// Theme at `index` in the current cycling order.
    pub fn from_index(index: usize) -> Theme {
        Theme::ALL.get(index).copied().unwrap_or(Theme::Default)
    }

    /// Theme for an index written by a version that stored themes as numbers,
    /// falling back to the default when it points past the end of that list.
    pub fn from_legacy_index(index: usize) -> Theme {
        Theme::LEGACY_ORDER
            .get(index)
            .copied()
            .unwrap_or(Theme::Default)
    }

    /// Resolve a stored preference, in either the current or the legacy form.
    ///
    /// An unknown name is not an error: a config naming a theme this build
    /// does not have falls back to the default rather than refusing to start.
    pub fn from_ref(reference: &ThemeRef) -> Theme {
        match reference {
            ThemeRef::Name(name) => Theme::from_slug(name).unwrap_or(Theme::Default),
            ThemeRef::Index(index) => Theme::from_legacy_index(*index),
        }
    }

    /// How this theme is written back to the config file.
    pub fn to_ref(self) -> ThemeRef {
        ThemeRef::Name(self.slug().to_string())
    }

    pub fn next(self) -> Theme {
        Theme::from_index((self.index() + 1) % Theme::ALL.len())
    }

    pub fn palette(self) -> Palette {
        match self {
            // Named colours, so the terminal's own palette shows through.
            Theme::Default => Palette {
                accent: Color::Cyan,
                text: Color::Reset,
                dim: Color::DarkGray,
                selection: Color::Rgb(40, 45, 60),
                good: Color::Green,
                ok: Color::Cyan,
                warn: Color::Yellow,
                bad: Color::Red,
                special: Color::Magenta,
            },
            Theme::Dracula => Palette {
                accent: Color::Rgb(189, 147, 249),
                text: Color::Rgb(248, 248, 242),
                dim: Color::Rgb(98, 114, 164),
                selection: Color::Rgb(68, 71, 90),
                good: Color::Rgb(80, 250, 123),
                ok: Color::Rgb(139, 233, 253),
                warn: Color::Rgb(241, 250, 140),
                bad: Color::Rgb(255, 85, 85),
                special: Color::Rgb(255, 121, 198),
            },
            Theme::Nord => Palette {
                accent: Color::Rgb(136, 192, 208),
                text: Color::Rgb(216, 222, 233),
                dim: Color::Rgb(76, 86, 106),
                selection: Color::Rgb(59, 66, 82),
                good: Color::Rgb(163, 190, 140),
                ok: Color::Rgb(129, 161, 193),
                warn: Color::Rgb(235, 203, 139),
                bad: Color::Rgb(191, 97, 106),
                special: Color::Rgb(180, 142, 173),
            },
            Theme::Solarized => Palette {
                accent: Color::Rgb(38, 139, 210),
                text: Color::Rgb(147, 161, 161),
                dim: Color::Rgb(88, 110, 117),
                selection: Color::Rgb(7, 54, 66),
                good: Color::Rgb(133, 153, 0),
                ok: Color::Rgb(42, 161, 152),
                warn: Color::Rgb(181, 137, 0),
                bad: Color::Rgb(220, 50, 47),
                special: Color::Rgb(211, 54, 130),
            },
            Theme::Gruvbox => Palette {
                accent: Color::Rgb(250, 189, 47),
                text: Color::Rgb(235, 219, 178),
                dim: Color::Rgb(146, 131, 116),
                selection: Color::Rgb(60, 56, 54),
                good: Color::Rgb(184, 187, 38),
                ok: Color::Rgb(142, 192, 124),
                warn: Color::Rgb(254, 128, 25),
                bad: Color::Rgb(251, 73, 52),
                special: Color::Rgb(211, 134, 155),
            },
            Theme::Monokai => Palette {
                accent: Color::Rgb(102, 217, 239),
                text: Color::Rgb(248, 248, 242),
                dim: Color::Rgb(117, 113, 94),
                selection: Color::Rgb(62, 61, 50),
                good: Color::Rgb(166, 226, 46),
                ok: Color::Rgb(102, 217, 239),
                warn: Color::Rgb(253, 151, 31),
                bad: Color::Rgb(249, 38, 114),
                special: Color::Rgb(174, 129, 255),
            },
            Theme::TokyoNight => Palette {
                accent: Color::Rgb(122, 162, 247),
                text: Color::Rgb(192, 202, 245),
                dim: Color::Rgb(86, 95, 137),
                selection: Color::Rgb(41, 46, 66),
                good: Color::Rgb(158, 206, 106),
                ok: Color::Rgb(125, 207, 255),
                warn: Color::Rgb(224, 175, 104),
                bad: Color::Rgb(247, 118, 142),
                special: Color::Rgb(187, 154, 247),
            },
            Theme::CatppuccinMocha => Palette {
                accent: Color::Rgb(203, 166, 247),
                text: Color::Rgb(205, 214, 244),
                dim: Color::Rgb(108, 112, 134),
                selection: Color::Rgb(69, 71, 90),
                good: Color::Rgb(166, 227, 161),
                ok: Color::Rgb(148, 226, 213),
                warn: Color::Rgb(250, 179, 135),
                bad: Color::Rgb(243, 139, 168),
                special: Color::Rgb(245, 194, 231),
            },
            // Rosé Pine has no green: upstream is six accents on a plum base.
            // "Perfect" takes foam and "Good" takes rose, which is the mapping
            // its editor ports use for success and info.
            Theme::RosePine => Palette {
                accent: Color::Rgb(196, 167, 231),
                text: Color::Rgb(224, 222, 244),
                dim: Color::Rgb(110, 106, 134),
                selection: Color::Rgb(64, 61, 82),
                good: Color::Rgb(156, 207, 216),
                ok: Color::Rgb(235, 188, 186),
                warn: Color::Rgb(246, 193, 119),
                bad: Color::Rgb(235, 111, 146),
                special: Color::Rgb(196, 167, 231),
            },
            // Everforest is deliberately low-contrast, so "Good" takes its
            // blue rather than its aqua: aqua sits too close to the green
            // above it to tell apart down a table column.
            Theme::Everforest => Palette {
                accent: Color::Rgb(131, 192, 146),
                text: Color::Rgb(211, 198, 170),
                dim: Color::Rgb(122, 132, 120),
                selection: Color::Rgb(61, 72, 77),
                good: Color::Rgb(167, 192, 128),
                ok: Color::Rgb(127, 187, 179),
                warn: Color::Rgb(219, 188, 127),
                bad: Color::Rgb(230, 126, 128),
                special: Color::Rgb(214, 153, 182),
            },
            Theme::Kanagawa => Palette {
                accent: Color::Rgb(126, 156, 216),
                text: Color::Rgb(220, 215, 186),
                dim: Color::Rgb(114, 113, 105),
                selection: Color::Rgb(34, 50, 73),
                good: Color::Rgb(152, 187, 108),
                ok: Color::Rgb(122, 168, 159),
                warn: Color::Rgb(230, 195, 132),
                bad: Color::Rgb(255, 93, 98),
                special: Color::Rgb(149, 127, 184),
            },
            Theme::OneDark => Palette {
                accent: Color::Rgb(97, 175, 239),
                text: Color::Rgb(171, 178, 191),
                dim: Color::Rgb(92, 99, 112),
                selection: Color::Rgb(62, 68, 81),
                good: Color::Rgb(152, 195, 121),
                ok: Color::Rgb(86, 182, 194),
                warn: Color::Rgb(229, 192, 123),
                bad: Color::Rgb(224, 108, 117),
                special: Color::Rgb(198, 120, 221),
            },
            Theme::Ocean => Palette {
                accent: Color::Rgb(102, 204, 255),
                text: Color::Rgb(214, 232, 240),
                dim: Color::Rgb(84, 110, 122),
                selection: Color::Rgb(28, 56, 71),
                good: Color::Rgb(94, 214, 186),
                ok: Color::Rgb(102, 187, 255),
                warn: Color::Rgb(240, 190, 110),
                bad: Color::Rgb(239, 108, 116),
                special: Color::Rgb(150, 170, 255),
            },
            Theme::Forest => Palette {
                accent: Color::Rgb(140, 200, 120),
                text: Color::Rgb(222, 232, 214),
                dim: Color::Rgb(104, 122, 96),
                selection: Color::Rgb(38, 54, 38),
                good: Color::Rgb(150, 214, 110),
                ok: Color::Rgb(126, 190, 160),
                warn: Color::Rgb(226, 196, 110),
                bad: Color::Rgb(216, 106, 96),
                special: Color::Rgb(196, 176, 118),
            },
            Theme::Sunset => Palette {
                accent: Color::Rgb(255, 158, 100),
                text: Color::Rgb(248, 226, 214),
                dim: Color::Rgb(140, 102, 108),
                selection: Color::Rgb(72, 40, 52),
                good: Color::Rgb(255, 196, 110),
                ok: Color::Rgb(240, 150, 140),
                warn: Color::Rgb(238, 120, 90),
                bad: Color::Rgb(214, 74, 96),
                special: Color::Rgb(198, 130, 200),
            },
            // Low chroma throughout: the one theme here that does not compete
            // with the terminal for attention over a long session.
            Theme::Slate => Palette {
                accent: Color::Rgb(130, 160, 190),
                text: Color::Rgb(206, 212, 220),
                dim: Color::Rgb(108, 116, 128),
                selection: Color::Rgb(44, 50, 58),
                good: Color::Rgb(140, 182, 150),
                ok: Color::Rgb(140, 170, 200),
                warn: Color::Rgb(200, 180, 130),
                bad: Color::Rgb(200, 130, 130),
                special: Color::Rgb(160, 150, 190),
            },
            Theme::Aurora => Palette {
                accent: Color::Rgb(120, 220, 190),
                text: Color::Rgb(214, 226, 240),
                dim: Color::Rgb(86, 104, 130),
                selection: Color::Rgb(24, 36, 58),
                good: Color::Rgb(96, 224, 140),
                ok: Color::Rgb(120, 190, 240),
                warn: Color::Rgb(240, 210, 130),
                bad: Color::Rgb(240, 110, 140),
                special: Color::Rgb(190, 150, 255),
            },
            // A phosphor ramp, where brighter means better. "Does not fit"
            // breaks the monochrome on purpose: an amber alarm is the one
            // thing these terminals did in a second colour.
            Theme::Matrix => Palette {
                accent: Color::Rgb(0, 255, 120),
                text: Color::Rgb(170, 255, 190),
                dim: Color::Rgb(0, 120, 60),
                selection: Color::Rgb(0, 48, 24),
                good: Color::Rgb(0, 255, 120),
                ok: Color::Rgb(0, 200, 150),
                warn: Color::Rgb(190, 230, 60),
                bad: Color::Rgb(255, 96, 64),
                special: Color::Rgb(120, 255, 220),
            },
            Theme::Sakura => Palette {
                accent: Color::Rgb(240, 170, 200),
                text: Color::Rgb(240, 222, 230),
                dim: Color::Rgb(150, 120, 140),
                selection: Color::Rgb(58, 38, 52),
                good: Color::Rgb(170, 215, 160),
                ok: Color::Rgb(150, 200, 215),
                warn: Color::Rgb(240, 200, 130),
                bad: Color::Rgb(230, 110, 130),
                special: Color::Rgb(200, 160, 240),
            },
            // Okabe-Ito, the palette designed so none of its pairs collide
            // under the common colour-vision deficiencies. The fit verdicts
            // run along the blue-to-orange axis rather than red-to-green —
            // the axis a deuteranope cannot read, and the one every other
            // theme here uses.
            Theme::ColourblindSafe => Palette {
                accent: Color::Rgb(86, 180, 233),
                text: Color::Rgb(238, 238, 238),
                dim: Color::Rgb(150, 150, 150),
                selection: Color::Rgb(38, 42, 50),
                good: Color::Rgb(0, 158, 115),
                ok: Color::Rgb(86, 180, 233),
                warn: Color::Rgb(240, 228, 66),
                bad: Color::Rgb(213, 94, 0),
                special: Color::Rgb(204, 121, 167),
            },
            // Saturated primaries for projectors, glare and low vision. Every
            // colour clears 4.5:1 against black, which the tests check.
            Theme::HighContrast => Palette {
                accent: Color::Rgb(0, 255, 255),
                text: Color::Rgb(255, 255, 255),
                dim: Color::Rgb(176, 176, 176),
                selection: Color::Rgb(0, 0, 160),
                good: Color::Rgb(0, 255, 0),
                ok: Color::Rgb(0, 255, 255),
                warn: Color::Rgb(255, 255, 0),
                bad: Color::Rgb(255, 80, 80),
                special: Color::Rgb(255, 0, 255),
            },
        }
    }
}

impl Palette {
    /// Colour for a 0–100 score bar.
    pub fn score(&self, score: f64) -> Color {
        if score >= 75.0 {
            self.good
        } else if score >= 50.0 {
            self.ok
        } else if score >= 25.0 {
            self.warn
        } else {
            self.bad
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The RGB behind a palette entry. Only the named-colour default has
    /// entries this cannot answer for, and no test below inspects those.
    fn rgb(color: Color) -> Option<(f64, f64, f64)> {
        match color {
            Color::Rgb(r, g, b) => Some((r as f64, g as f64, b as f64)),
            _ => None,
        }
    }

    /// Relative luminance, WCAG 2.1.
    fn luminance(color: Color) -> f64 {
        let (r, g, b) = rgb(color).expect("luminance needs an RGB colour");
        let channel = |c: f64| {
            let c = c / 255.0;
            if c <= 0.03928 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    }

    fn contrast_against_black(color: Color) -> f64 {
        (luminance(color) + 0.05) / 0.05
    }

    /// Where a colour lands for a viewer who cannot separate red from green.
    ///
    /// Deuteranopia collapses the red-green opponent channel, leaving
    /// lightness and the blue-yellow channel to carry the difference. Two
    /// colours that agree on both are indistinguishable however far apart they
    /// look to everyone else.
    fn deuteranope_coords(color: Color) -> (f64, f64) {
        let (r, g, b) = rgb(color).expect("simulation needs an RGB colour");
        let lightness = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let blue_yellow = b - (r + g) / 2.0;
        (lightness, blue_yellow)
    }

    #[test]
    fn indexes_round_trip() {
        for theme in Theme::ALL {
            assert_eq!(Theme::from_index(theme.index()), theme);
        }
    }

    #[test]
    fn cycling_visits_every_theme_and_returns_to_the_start() {
        let mut theme = Theme::Default;
        let mut seen = Vec::new();
        for _ in 0..Theme::ALL.len() {
            seen.push(theme);
            theme = theme.next();
        }
        assert_eq!(theme, Theme::Default, "cycling wraps");
        assert_eq!(seen.len(), Theme::ALL.len());
        for candidate in Theme::ALL {
            assert!(
                seen.contains(&candidate),
                "{} was skipped",
                candidate.name()
            );
        }
    }

    #[test]
    fn an_out_of_range_stored_index_falls_back_to_default() {
        assert_eq!(Theme::from_index(999), Theme::Default);
        assert_eq!(Theme::from_legacy_index(999), Theme::Default);
    }

    #[test]
    fn slugs_round_trip_and_are_unique() {
        for theme in Theme::ALL {
            assert_eq!(
                Theme::from_slug(theme.slug()),
                Some(theme),
                "{} does not round-trip",
                theme.name()
            );
        }
        let mut slugs: Vec<_> = Theme::ALL.iter().map(|t| t.slug()).collect();
        let count = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two themes share a slug");
    }

    #[test]
    fn slugs_tolerate_the_spellings_a_hand_edited_config_may_use() {
        assert_eq!(Theme::from_slug("  Tokyo_Night "), Some(Theme::TokyoNight));
        assert_eq!(Theme::from_slug("ROSE PINE"), Some(Theme::RosePine));
        // Both spellings, because the display name is British and the word
        // most people will reach for is not.
        assert_eq!(
            Theme::from_slug("colorblind-safe"),
            Some(Theme::ColourblindSafe)
        );
        assert_eq!(
            Theme::from_slug("colourblind-safe"),
            Some(Theme::ColourblindSafe)
        );
        assert_eq!(Theme::from_slug("no-such-theme"), None);
    }

    #[test]
    fn legacy_indexes_still_resolve_to_the_theme_they_were_written_for() {
        // The numbers a pre-naming config could contain, and what they meant.
        // Reordering Theme::ALL must not move any of these.
        let shipped = [
            (0, Theme::Default),
            (1, Theme::Dracula),
            (2, Theme::Nord),
            (3, Theme::Solarized),
            (4, Theme::Gruvbox),
            (5, Theme::Monokai),
            (6, Theme::TokyoNight),
            (7, Theme::Ocean),
            (8, Theme::Forest),
            (9, Theme::Sunset),
        ];
        for (index, expected) in shipped {
            assert_eq!(
                Theme::from_legacy_index(index),
                expected,
                "legacy index {index} moved"
            );
        }
    }

    #[test]
    fn stored_preferences_resolve_in_both_forms() {
        for theme in Theme::ALL {
            assert_eq!(Theme::from_ref(&theme.to_ref()), theme);
        }
        assert_eq!(Theme::from_ref(&ThemeRef::Index(5)), Theme::Monokai);
        // A theme this build does not have must not stop the TUI starting.
        assert_eq!(
            Theme::from_ref(&ThemeRef::Name("retired-theme".into())),
            Theme::Default
        );
    }

    #[test]
    fn every_theme_distinguishes_its_verdict_colours() {
        // If two verdicts share a colour the table stops being readable at a
        // glance, which is the whole point of colouring them.
        for theme in Theme::ALL {
            let p = theme.palette();
            let verdicts = [p.good, p.ok, p.warn, p.bad];
            for (i, a) in verdicts.iter().enumerate() {
                for b in &verdicts[i + 1..] {
                    assert_ne!(a, b, "{} reuses a verdict colour", theme.name());
                }
            }
            assert_ne!(p.dim, p.text, "{} cannot dim anything", theme.name());
        }
    }

    #[test]
    fn score_colours_step_at_the_documented_thresholds() {
        let p = Theme::Default.palette();
        assert_eq!(p.score(100.0), p.good);
        assert_eq!(p.score(75.0), p.good);
        assert_eq!(p.score(74.9), p.ok);
        assert_eq!(p.score(50.0), p.ok);
        assert_eq!(p.score(49.9), p.warn);
        assert_eq!(p.score(25.0), p.warn);
        assert_eq!(p.score(0.0), p.bad);
    }

    #[test]
    fn the_default_theme_uses_the_terminals_own_colours() {
        // Named colours inherit the user's palette; RGB would override it and
        // break on a light background.
        let p = Theme::Default.palette();
        assert_eq!(p.good, Color::Green);
        assert_eq!(p.bad, Color::Red);
        assert_eq!(p.text, Color::Reset);
    }

    #[test]
    fn only_the_default_theme_uses_named_colours() {
        // Every other theme is RGB on purpose: a Dracula rendered in the
        // terminal's idea of "magenta" is not Dracula.
        for theme in Theme::ALL {
            if theme == Theme::Default {
                continue;
            }
            let p = theme.palette();
            for (role, colour) in [
                ("accent", p.accent),
                ("text", p.text),
                ("dim", p.dim),
                ("selection", p.selection),
                ("good", p.good),
                ("ok", p.ok),
                ("warn", p.warn),
                ("bad", p.bad),
                ("special", p.special),
            ] {
                assert!(
                    matches!(colour, Color::Rgb(..)),
                    "{}'s {role} is not RGB",
                    theme.name()
                );
            }
        }
    }

    #[test]
    fn the_colourblind_theme_keeps_its_verdicts_apart_without_red_and_green() {
        // The claim the theme makes: drop the red-green channel and the four
        // fit verdicts are still four different things. Anything closer than
        // this threshold on both remaining axes would read as one colour.
        let p = Theme::ColourblindSafe.palette();
        let verdicts = [
            ("good", p.good),
            ("ok", p.ok),
            ("warn", p.warn),
            ("bad", p.bad),
        ];
        for (i, (name_a, a)) in verdicts.iter().enumerate() {
            for (name_b, b) in &verdicts[i + 1..] {
                let (la, ba) = deuteranope_coords(*a);
                let (lb, bb) = deuteranope_coords(*b);
                assert!(
                    (la - lb).abs() > 30.0 || (ba - bb).abs() > 30.0,
                    "{name_a} and {name_b} collapse together for a deuteranope"
                );
            }
        }
    }

    #[test]
    fn a_red_green_theme_is_what_the_colourblind_theme_exists_to_avoid() {
        // Guards the test above from passing vacuously: the same measurement
        // on a conventional palette finds the collision it is looking for.
        let p = Theme::Gruvbox.palette();
        let (good_l, good_by) = deuteranope_coords(p.good);
        let (warn_l, warn_by) = deuteranope_coords(p.warn);
        assert!(
            (good_l - warn_l).abs() <= 30.0 && (good_by - warn_by).abs() <= 30.0,
            "Gruvbox no longer demonstrates the red-green collision"
        );
    }

    #[test]
    fn the_high_contrast_theme_is_legible_on_a_black_terminal() {
        // 4.5:1 is the WCAG AA floor for body text. `selection` is exempt: it
        // is painted as a background, never as text on one.
        let p = Theme::HighContrast.palette();
        for (role, colour) in [
            ("accent", p.accent),
            ("text", p.text),
            ("dim", p.dim),
            ("good", p.good),
            ("ok", p.ok),
            ("warn", p.warn),
            ("bad", p.bad),
            ("special", p.special),
        ] {
            let ratio = contrast_against_black(colour);
            assert!(
                ratio >= 4.5,
                "high contrast {role} is only {ratio:.1}:1 against black"
            );
        }
    }
}
