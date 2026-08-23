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

use ratatui::style::Color;

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
    Dracula,
    Nord,
    Solarized,
    Gruvbox,
    Monokai,
    TokyoNight,
    Ocean,
    Forest,
    Sunset,
}

impl Theme {
    pub const ALL: [Theme; 10] = [
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
            Theme::Ocean => "Ocean",
            Theme::Forest => "Forest",
            Theme::Sunset => "Sunset",
        }
    }

    /// Position in [`Theme::ALL`]; this is what gets persisted.
    pub fn index(self) -> usize {
        Theme::ALL.iter().position(|&t| t == self).unwrap_or(0)
    }

    /// Theme at `index`, falling back to the default when a stored config
    /// points past the end of the list.
    pub fn from_index(index: usize) -> Theme {
        Theme::ALL.get(index).copied().unwrap_or(Theme::Default)
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
}
