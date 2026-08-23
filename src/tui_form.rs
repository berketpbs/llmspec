//! A small labelled-field form for the TUI's editable popups.
//!
//! The hardware simulator and the speed tunables are the same interaction —
//! a short list of numeric fields, moved between with Tab or `j`/`k`, edited
//! in place, applied with Enter and abandoned with Esc. Expressing that once
//! keeps the two popups from drifting apart and keeps the key handler from
//! growing a `match field_index` arm per field.

/// One editable line in a form.
#[derive(Debug, Clone)]
pub struct Field {
    /// Prompt shown to the left of the value.
    pub label: &'static str,
    /// What the value means, shown under the fields.
    pub help: &'static str,
    /// Text as typed. Parsed only on apply, so a half-typed "1." is allowed.
    pub value: String,
    /// Accepted range, used to clamp on apply and to describe the field.
    pub range: (f64, f64),
}

/// A list of fields with a cursor.
#[derive(Debug, Clone)]
pub struct Form {
    fields: Vec<Field>,
    active: usize,
}

impl Form {
    pub fn new(fields: Vec<Field>) -> Form {
        Form { fields, active: 0 }
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    pub fn active(&self) -> usize {
        self.active
    }

    /// True when `index` is the field being edited.
    pub fn is_active(&self, index: usize) -> bool {
        index == self.active
    }

    /// Move the cursor, wrapping at both ends.
    pub fn focus_next(&mut self) {
        if !self.fields.is_empty() {
            self.active = (self.active + 1) % self.fields.len();
        }
    }

    pub fn focus_prev(&mut self) {
        if !self.fields.is_empty() {
            self.active = self.active.checked_sub(1).unwrap_or(self.fields.len() - 1);
        }
    }

    /// Append to the active field. Only characters that can appear in a
    /// decimal number are accepted, so the field cannot be typed into an
    /// unparseable state by accident.
    pub fn push(&mut self, c: char) {
        if !c.is_ascii_digit() && c != '.' {
            return;
        }
        if let Some(field) = self.fields.get_mut(self.active) {
            // A second decimal point would make the value unparseable.
            if c == '.' && field.value.contains('.') {
                return;
            }
            field.value.push(c);
        }
    }

    pub fn backspace(&mut self) {
        if let Some(field) = self.fields.get_mut(self.active) {
            field.value.pop();
        }
    }

    pub fn clear_active(&mut self) {
        if let Some(field) = self.fields.get_mut(self.active) {
            field.value.clear();
        }
    }

    /// Overwrite every value and return the cursor to the first field.
    pub fn reset(&mut self, values: &[String]) {
        for (field, value) in self.fields.iter_mut().zip(values) {
            field.value.clone_from(value);
        }
        self.active = 0;
    }

    /// The active field's value, clamped to its range.
    ///
    /// Returns `None` when the field is empty or holds something unparseable,
    /// which the caller reads as "leave this setting alone".
    pub fn parse(&self, index: usize) -> Option<f64> {
        let field = self.fields.get(index)?;
        let value: f64 = field.value.trim().parse().ok()?;
        if !value.is_finite() {
            return None;
        }
        Some(value.clamp(field.range.0, field.range.1))
    }
}

impl Field {
    pub fn new(label: &'static str, help: &'static str, value: String, range: (f64, f64)) -> Field {
        Field {
            label,
            help,
            value,
            range,
        }
    }

    /// `0.10 – 2.00`, for the popup's range column.
    pub fn range_hint(&self) -> String {
        let (lo, hi) = self.range;
        if lo.fract() == 0.0 && hi.fract() == 0.0 {
            format!("{lo:.0}–{hi:.0}")
        } else {
            format!("{lo:.2}–{hi:.2}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form() -> Form {
        Form::new(vec![
            Field::new("Alpha", "first", "1.0".into(), (0.0, 2.0)),
            Field::new("Beta", "second", "0.5".into(), (0.1, 1.0)),
            Field::new("Gamma", "third", "8".into(), (1.0, 64.0)),
        ])
    }

    #[test]
    fn focus_wraps_in_both_directions() {
        let mut f = form();
        assert_eq!(f.active(), 0);
        f.focus_prev();
        assert_eq!(
            f.active(),
            2,
            "moving up from the first field wraps to last"
        );
        f.focus_next();
        assert_eq!(f.active(), 0);
        f.focus_next();
        f.focus_next();
        f.focus_next();
        assert_eq!(f.active(), 0, "moving down past the last field wraps");
    }

    #[test]
    fn editing_only_touches_the_active_field() {
        let mut f = form();
        f.focus_next();
        f.push('7');
        assert_eq!(f.fields()[0].value, "1.0");
        assert_eq!(f.fields()[1].value, "0.57");
        f.backspace();
        assert_eq!(f.fields()[1].value, "0.5");
        f.clear_active();
        assert_eq!(f.fields()[1].value, "");
        assert_eq!(f.fields()[0].value, "1.0", "other fields are untouched");
    }

    #[test]
    fn only_number_characters_are_accepted() {
        let mut f = form();
        f.clear_active();
        for c in "1a2.b3".chars() {
            f.push(c);
        }
        assert_eq!(f.fields()[0].value, "12.3");
    }

    #[test]
    fn a_second_decimal_point_is_refused() {
        let mut f = form();
        f.clear_active();
        for c in "1.2.3".chars() {
            f.push(c);
        }
        assert_eq!(f.fields()[0].value, "1.23");
        assert_eq!(f.parse(0), Some(1.23));
    }

    #[test]
    fn parsing_clamps_to_the_declared_range() {
        let mut f = form();
        f.clear_active();
        f.push('9');
        assert_eq!(f.parse(0), Some(2.0), "9 is clamped to the 0..2 range");
        f.clear_active();
        f.push('0');
        assert_eq!(f.parse(0), Some(0.0));
    }

    #[test]
    fn empty_or_unparseable_values_are_none() {
        let mut f = form();
        f.clear_active();
        assert_eq!(f.parse(0), None, "an empty field means 'leave it alone'");
        f.push('.');
        assert_eq!(f.parse(0), None, "a lone decimal point is not a number");
        assert_eq!(f.parse(99), None, "an out-of-range index is not a panic");
    }

    #[test]
    fn reset_restores_values_and_the_cursor() {
        let mut f = form();
        f.focus_next();
        f.clear_active();
        f.reset(&["2.0".into(), "0.9".into(), "16".into()]);
        assert_eq!(f.active(), 0);
        assert_eq!(f.fields()[1].value, "0.9");
        assert_eq!(f.parse(2), Some(16.0));
    }

    #[test]
    fn range_hints_drop_pointless_decimals() {
        let f = form();
        assert_eq!(f.fields()[0].range_hint(), "0–2");
        assert_eq!(f.fields()[1].range_hint(), "0.10–1.00");
    }

    #[test]
    fn an_empty_form_does_not_panic() {
        let mut f = Form::new(Vec::new());
        f.focus_next();
        f.focus_prev();
        f.push('1');
        f.backspace();
        f.clear_active();
        assert_eq!(f.active(), 0);
        assert_eq!(f.parse(0), None);
    }
}
