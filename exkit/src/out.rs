// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::fmt::Display;
use std::io::Write;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Format {
    #[default]
    Text,
    Tsv,
}

pub fn format() -> Format {
    match std::env::var("QNTT_FORMAT").as_deref() {
        Ok("tsv") | Ok("TSV") => Format::Tsv,
        _ => Format::Text,
    }
}

pub fn flush(msg: impl Display) {
    println!("{msg}");
    let _ = std::io::stdout().flush();
}

pub fn note(msg: impl Display) {
    flush(format_args!("# {msg}"));
}

pub fn banner(title: impl Display) {
    flush(format_args!("== {title} =="));
}

pub fn banner_kv(title: impl Display, kv: &[(&str, String)]) {
    let tail: Vec<String> = kv.iter().map(|(k, v)| format!("{k}={v}")).collect();
    if tail.is_empty() {
        banner(title);
    } else {
        flush(format_args!("== {title} | {} ==", tail.join(" | ")));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Align {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Text,
    Int,
    Float(usize),
    Sci(usize),
    Ms(usize),
    Dur(usize),
}

#[derive(Clone, Debug)]
pub struct Col {
    name: &'static str,
    label: Option<&'static str>,
    width: usize,
    align: Align,
    kind: Kind,
}

impl Col {
    pub fn text(name: &'static str, width: usize) -> Col {
        Col {
            name,
            label: None,
            width,
            align: Align::Left,
            kind: Kind::Text,
        }
    }
    pub fn int(name: &'static str, width: usize) -> Col {
        Col {
            name,
            label: None,
            width,
            align: Align::Right,
            kind: Kind::Int,
        }
    }
    pub fn float(name: &'static str, width: usize, p: usize) -> Col {
        Col {
            name,
            label: None,
            width,
            align: Align::Right,
            kind: Kind::Float(p),
        }
    }
    pub fn sci(name: &'static str, width: usize, p: usize) -> Col {
        Col {
            name,
            label: None,
            width,
            align: Align::Right,
            kind: Kind::Sci(p),
        }
    }
    pub fn ms(name: &'static str, width: usize, p: usize) -> Col {
        Col {
            name,
            label: None,
            width,
            align: Align::Right,
            kind: Kind::Ms(p),
        }
    }
    pub fn dur(name: &'static str, width: usize, p: usize) -> Col {
        Col {
            name,
            label: None,
            width,
            align: Align::Right,
            kind: Kind::Dur(p),
        }
    }
    pub fn tag(name: &'static str, width: usize) -> Col {
        Col {
            name,
            label: None,
            width,
            align: Align::Right,
            kind: Kind::Text,
        }
    }
    pub fn header(mut self, label: &'static str) -> Col {
        self.label = Some(label);
        self
    }

    fn head(&self) -> &str {
        self.label.unwrap_or(self.name)
    }

    fn render(&self, c: &Cell, fmt: Format) -> String {
        let body = match (self.kind, c) {
            (_, Cell::Empty) => String::new(),
            (_, Cell::S(s)) => s.clone(),
            (Kind::Int, Cell::I(i)) => format!("{i}"),
            (Kind::Float(p), Cell::I(i)) => format!("{:.*}", p, *i as f64),
            (Kind::Sci(p), Cell::I(i)) => format!("{:.*e}", p, *i as f64),
            (_, Cell::I(i)) => format!("{i}"),
            (Kind::Float(p), Cell::F(x)) => format!("{x:.p$}"),
            (Kind::Sci(p), Cell::F(x)) => format!("{x:.p$e}"),
            (Kind::Int, Cell::F(x)) => {
                panic!(
                    "column `{}` is an int column but was given the float {x}",
                    self.name
                )
            }
            (_, Cell::F(x)) => format!("{x}"),
            (Kind::Ms(p), Cell::D(d)) => format!("{:.*}", p, d.as_secs_f64() * 1e3),
            (Kind::Dur(p), Cell::D(d)) => format!("{d:.p$?}"),
            (_, Cell::D(d)) => format!("{:.3}", d.as_secs_f64() * 1e3),
        };
        if fmt == Format::Tsv {
            return body;
        }
        match self.align {
            Align::Left => format!("{body:<width$}", width = self.width),
            Align::Right => format!("{body:>width$}", width = self.width),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Cell {
    S(String),
    I(i128),
    F(f64),
    D(Duration),
    Empty,
}

macro_rules! cell_from_int {
    ($($t:ty),*) => { $(
        impl From<$t> for Cell { fn from(v: $t) -> Cell { Cell::I(v as i128) } }
    )* };
}
cell_from_int!(u8, u16, u32, u64, usize, i8, i16, i32, i64, i128);

impl From<f64> for Cell {
    fn from(v: f64) -> Cell {
        Cell::F(v)
    }
}
impl From<f32> for Cell {
    fn from(v: f32) -> Cell {
        Cell::F(v as f64)
    }
}
impl From<&str> for Cell {
    fn from(v: &str) -> Cell {
        Cell::S(v.to_string())
    }
}
impl From<&String> for Cell {
    fn from(v: &String) -> Cell {
        Cell::S(v.clone())
    }
}
impl From<String> for Cell {
    fn from(v: String) -> Cell {
        Cell::S(v)
    }
}
impl From<Duration> for Cell {
    fn from(v: Duration) -> Cell {
        Cell::D(v)
    }
}
impl From<bool> for Cell {
    fn from(v: bool) -> Cell {
        Cell::S(if v { "yes" } else { "NO" }.to_string())
    }
}

#[macro_export]
macro_rules! row {
    ($($v:expr),* $(,)?) => { &[$($crate::out::Cell::from($v)),*] };
}

pub struct Table {
    cols: Vec<Col>,
    fmt: Format,
    rule: bool,
    started: bool,
}

impl Table {
    pub fn new(cols: Vec<Col>) -> Table {
        Table {
            cols,
            fmt: format(),
            rule: true,
            started: false,
        }
    }

    pub fn with_format(mut self, fmt: Format) -> Table {
        self.fmt = fmt;
        self
    }

    pub fn no_rule(mut self) -> Table {
        self.rule = false;
        self
    }

    pub fn start(mut self) -> Table {
        let heads: Vec<String> = self
            .cols
            .iter()
            .map(|c| match self.fmt {
                Format::Tsv => c.head().to_string(),
                Format::Text => match c.align {
                    Align::Left => format!("{:<w$}", c.head(), w = c.width),
                    Align::Right => format!("{:>w$}", c.head(), w = c.width),
                },
            })
            .collect();
        let line = self.join(&heads);
        flush(&line);
        if self.rule && self.fmt == Format::Text {
            flush("-".repeat(line.len()));
        }
        self.started = true;
        self
    }

    fn join(&self, parts: &[String]) -> String {
        match self.fmt {
            Format::Tsv => parts.join("\t"),
            Format::Text => parts.join(" ").trim_end().to_string(),
        }
    }

    pub fn row(&mut self, cells: &[Cell]) {
        assert_eq!(
            cells.len(),
            self.cols.len(),
            "row has {} cells but the table declares {} columns ({})",
            cells.len(),
            self.cols.len(),
            self.cols
                .iter()
                .map(|c| c.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let parts: Vec<String> = self
            .cols
            .iter()
            .zip(cells)
            .map(|(c, v)| c.render(v, self.fmt))
            .collect();
        flush(self.join(&parts));
    }

    pub fn note(&mut self, msg: impl Display) {
        note(msg);
    }

    pub fn skip(&mut self, name: &str, why: impl Display) {
        let w = self.cols.first().map(|c| c.width).unwrap_or(14);
        flush(format_args!("{name:<w$} SKIPPED: {why}"));
    }

    pub fn rule(&mut self) {
        if self.fmt == Format::Text {
            let w: usize = self
                .cols
                .iter()
                .map(|c| c.width + 1)
                .sum::<usize>()
                .saturating_sub(1);
            flush("-".repeat(w));
        }
    }

    pub fn finish(self) {
        let _ = std::io::stdout().flush();
    }
}

pub struct Summary {
    parts: Vec<String>,
}

impl Summary {
    pub fn new(target: &str) -> Summary {
        Summary {
            parts: vec!["SUMMARY".to_string(), target.to_string()],
        }
    }

    fn push(mut self, k: &'static str, v: String) -> Summary {
        assert!(!k.contains('='), "summary key `{k}` contains `=`");
        assert!(
            !v.contains('='),
            "summary value for `{k}` contains `=`: {v}"
        );
        self.parts.push(format!("{k}={}", v.replace(' ', "_")));
        self
    }

    pub fn s(self, k: &'static str, v: impl Display) -> Summary {
        self.push(k, v.to_string())
    }
    pub fn i(self, k: &'static str, v: impl Into<i128>) -> Summary {
        self.push(k, v.into().to_string())
    }
    pub fn f(self, k: &'static str, v: f64, p: usize) -> Summary {
        self.push(k, format!("{v:.p$}"))
    }
    pub fn e(self, k: &'static str, v: f64, p: usize) -> Summary {
        self.push(k, format!("{v:.p$e}"))
    }
    pub fn b(self, k: &'static str, v: bool) -> Summary {
        self.push(k, v.to_string())
    }

    pub fn emit(self) {
        flush(self.parts.join(" "));
    }

    pub fn line(&self) -> String {
        self.parts.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_column_owns_the_format_so_header_and_row_cannot_drift() {
        let c = Col::float("energy", 16, 8).header("energy (Ha)");
        assert_eq!(c.head(), "energy (Ha)");
        assert_eq!(c.render(&Cell::F(-1.5), Format::Text), "     -1.50000000");
        assert_eq!(c.render(&Cell::F(-1.5), Format::Tsv), "-1.50000000");
    }

    #[test]
    fn t_bool_renders_as_the_conv_tag() {
        let c = Col::tag("conv", 5);
        assert_eq!(c.render(&Cell::from(true), Format::Text), "  yes");
        assert_eq!(c.render(&Cell::from(false), Format::Text), "   NO");
    }

    #[test]
    #[should_panic(expected = "int column")]
    fn t_a_float_into_an_int_column_is_an_error_not_a_truncation() {
        Col::int("iters", 7).render(&Cell::F(3.5), Format::Text);
    }

    #[test]
    fn t_summary_line_shape() {
        let s = Summary::new("gad_wmodel")
            .s("cell", "d4su2/trunc")
            .b("converged", true)
            .i("steps", 908i64)
            .e("gnorm", 9.93e-9, 2);
        assert_eq!(
            s.line(),
            "SUMMARY gad_wmodel cell=d4su2/trunc converged=true steps=908 gnorm=9.93e-9"
        );
    }

    #[test]
    fn t_summary_masks_spaces_so_every_token_is_one_pair() {
        let s = Summary::new("t").s("note", "two words");
        assert_eq!(s.line().split_whitespace().count(), 3);
    }
}
