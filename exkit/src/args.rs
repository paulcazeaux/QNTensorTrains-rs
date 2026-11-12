// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use qntt::pointgroup::PointGroupKind;

use crate::out::Format;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Val {
    Str,
    Path,
    Usize,
    I64,
    F64,
    Bool,
    Enum(&'static [&'static str]),
    List(&'static Val),
}

impl Val {
    fn describe(&self) -> String {
        match self {
            Val::Str => "<str>".into(),
            Val::Path => "<path>".into(),
            Val::Usize => "<usize>".into(),
            Val::I64 => "<int>".into(),
            Val::F64 => "<float>".into(),
            Val::Bool => "<bool>".into(),
            Val::Enum(set) => format!("<{}>", set.join("|")),
            Val::List(inner) => format!("{}[,…]", inner.describe()),
        }
    }

    fn check(&self, name: &str, text: &str) -> Result<(), UsageError> {
        let bad = |want: &str| {
            Err(UsageError::BadValue {
                arg: name.to_string(),
                got: text.to_string(),
                want: want.to_string(),
            })
        };
        match self {
            Val::Str | Val::Path => Ok(()),
            Val::Usize => {
                if text.parse::<usize>().is_ok() {
                    Ok(())
                } else {
                    bad("a non-negative integer")
                }
            }
            Val::I64 => {
                if text.parse::<i64>().is_ok() {
                    Ok(())
                } else {
                    bad("an integer")
                }
            }
            Val::F64 => {
                if text.parse::<f64>().is_ok() {
                    Ok(())
                } else {
                    bad("a number")
                }
            }
            Val::Bool => match text {
                "true" | "false" | "1" | "0" | "yes" | "no" => Ok(()),
                _ => bad("true or false"),
            },
            Val::Enum(set) => {
                if set.contains(&text) {
                    Ok(())
                } else {
                    bad(&format!("one of: {}", set.join(", ")))
                }
            }
            Val::List(inner) => {
                for part in text.split(',').filter(|p| !p.is_empty()) {
                    inner.check(name, part)?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum Def {
    Required,
    Optional,
    Str(&'static str),
    Usize(usize),
    I64(i64),
    F64(f64),
    Bool(bool),
    List(&'static [&'static str]),
    Derived(&'static str),
}

impl Def {
    fn text(&self) -> Option<String> {
        match self {
            Def::Required | Def::Optional | Def::Derived(_) => None,
            Def::Str(s) => Some((*s).to_string()),
            Def::Usize(v) => Some(v.to_string()),
            Def::I64(v) => Some(v.to_string()),
            Def::F64(v) => Some(v.to_string()),
            Def::Bool(v) => Some(v.to_string()),
            Def::List(v) => Some(v.join(",")),
        }
    }

    fn help(&self) -> String {
        match self {
            Def::Required => "[required]".into(),
            Def::Optional => "[optional]".into(),
            Def::Derived(rule) => format!("[default {rule}]"),
            other => format!("[default {}]", other.text().unwrap()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Pos,
    Rest,
    Key,
    Opt,
    Flag,
    Switch,
}

#[derive(Clone, Debug)]
struct Arg {
    name: &'static str,
    doc: &'static str,
    val: Val,
    def: Def,
    slot: Slot,
}

#[derive(Clone, Debug, Default)]
pub struct Mode {
    args: Vec<Arg>,
}

macro_rules! builder_verbs {
    ($t:ty) => {
        impl $t {
            pub fn pos(
                mut self,
                name: &'static str,
                doc: &'static str,
                val: Val,
                def: Def,
            ) -> Self {
                self.args_mut().push(Arg {
                    name,
                    doc,
                    val,
                    def,
                    slot: Slot::Pos,
                });
                self
            }
            pub fn rest(
                mut self,
                name: &'static str,
                doc: &'static str,
                val: Val,
                def: Def,
            ) -> Self {
                self.args_mut().push(Arg {
                    name,
                    doc,
                    val,
                    def,
                    slot: Slot::Rest,
                });
                self
            }
            pub fn key(
                mut self,
                name: &'static str,
                doc: &'static str,
                val: Val,
                def: Def,
            ) -> Self {
                self.args_mut().push(Arg {
                    name,
                    doc,
                    val,
                    def,
                    slot: Slot::Key,
                });
                self
            }
            pub fn opt(
                mut self,
                name: &'static str,
                doc: &'static str,
                val: Val,
                def: Def,
            ) -> Self {
                self.args_mut().push(Arg {
                    name,
                    doc,
                    val,
                    def,
                    slot: Slot::Opt,
                });
                self
            }
            pub fn flag(mut self, name: &'static str, doc: &'static str) -> Self {
                self.args_mut().push(Arg {
                    name,
                    doc,
                    val: Val::Bool,
                    def: Def::Bool(false),
                    slot: Slot::Flag,
                });
                self
            }
            pub fn switch(mut self, name: &'static str, doc: &'static str) -> Self {
                self.args_mut().push(Arg {
                    name,
                    doc,
                    val: Val::Bool,
                    def: Def::Bool(false),
                    slot: Slot::Switch,
                });
                self
            }
        }
    };
}

impl Mode {
    fn args_mut(&mut self) -> &mut Vec<Arg> {
        &mut self.args
    }
}
builder_verbs!(Mode);

#[derive(Clone, Debug)]
struct ModeDecl {
    name: &'static str,
    doc: &'static str,
    mode: Mode,
}

pub type ModeClassifier = fn(&[String]) -> Option<&'static str>;

#[derive(Clone, Debug)]
struct EnvDecl {
    name: &'static str,
    kind: &'static str,
    doc: &'static str,
}

#[derive(Clone, Debug)]
pub struct Spec {
    name: &'static str,
    about: &'static str,
    notes: Vec<&'static str>,
    args: Vec<Arg>,
    modes: Vec<ModeDecl>,
    default_mode: Option<&'static str>,
    mode_at: usize,
    mode_classifier: Option<ModeClassifier>,
    pg: bool,
    envs: Vec<EnvDecl>,
}

impl Spec {
    fn args_mut(&mut self) -> &mut Vec<Arg> {
        &mut self.args
    }
}
builder_verbs!(Spec);

impl Spec {
    pub fn new(name: &'static str) -> Spec {
        Spec {
            name,
            about: "",
            notes: Vec::new(),
            args: Vec::new(),
            modes: Vec::new(),
            default_mode: None,
            mode_at: 0,
            mode_classifier: None,
            pg: false,
            envs: Vec::new(),
        }
    }

    pub fn about(mut self, s: &'static str) -> Spec {
        self.about = s;
        self
    }

    pub fn note(mut self, s: &'static str) -> Spec {
        self.notes.push(s);
        self
    }

    pub fn mode(
        mut self,
        name: &'static str,
        doc: &'static str,
        f: impl FnOnce(Mode) -> Mode,
    ) -> Spec {
        self.modes.push(ModeDecl {
            name,
            doc,
            mode: f(Mode::default()),
        });
        self
    }

    pub fn default_mode(mut self, name: &'static str) -> Spec {
        self.default_mode = Some(name);
        self
    }

    pub fn mode_at(mut self, i: usize) -> Spec {
        self.mode_at = i;
        self
    }

    pub fn mode_classifier(mut self, f: ModeClassifier) -> Spec {
        self.mode_classifier = Some(f);
        self
    }

    pub fn pg(mut self) -> Spec {
        self.pg = true;
        self
    }

    pub fn env(mut self, name: &'static str, kind: &'static str, doc: &'static str) -> Spec {
        self.envs.push(EnvDecl { name, kind, doc });
        self
    }

    pub fn parse(self) -> Matches {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        match self.parse_from(argv) {
            Ok(m) => m,
            Err(UsageError::HelpRequested(text)) => {
                print!("{text}");
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("error: {e}");
                eprintln!("run with --help for the full option list");
                std::process::exit(2);
            }
        }
    }

    pub fn parse_from<I: IntoIterator<Item = String>>(self, it: I) -> Result<Matches, UsageError> {
        let argv: Vec<String> = it.into_iter().collect();
        if argv.iter().any(|a| a == "-h" || a == "--help") {
            return Err(UsageError::HelpRequested(self.help()));
        }
        Parser::new(&self).run(argv)
    }

    pub fn help(&self) -> String {
        let mut s = String::new();
        if self.about.is_empty() {
            s.push_str(&format!("{}\n", self.name));
        } else {
            s.push_str(&format!("{} — {}\n", self.name, self.about));
        }
        s.push('\n');
        s.push_str(&format!(
            "usage: cargo run --release --example {} --{}\n",
            self.name,
            self.usage_tail()
        ));
        for n in &self.notes {
            s.push('\n');
            s.push_str(&wrap(n, 92));
        }
        if !self.modes.is_empty() {
            s.push_str("\nmodes:\n");
            for m in &self.modes {
                let dflt = if Some(m.name) == self.default_mode {
                    " (default)"
                } else {
                    ""
                };
                s.push_str(&format!("  {}{}\n      {}\n", m.name, dflt, m.doc));
                for a in &m.mode.args {
                    s.push_str(&arg_help(a, 6));
                }
            }
        }
        let plain: Vec<&Arg> = self.args.iter().collect();
        if !plain.is_empty() {
            s.push_str("\narguments:\n");
            for a in plain {
                s.push_str(&arg_help(a, 2));
            }
        }
        s.push_str("\noptions:\n");
        if self.pg {
            s.push_str(
                "  --pg [kind]            resolve the fixture's point group; a bare --pg means\n",
            );
            s.push_str("                         `auto`, otherwise name one (c1, ci, c2, cs, d2, c2v, …)\n");
        }
        s.push_str(
            "  --format <text|tsv>    output format                          [default text]\n",
        );
        s.push_str("  -h, --help             this text\n");
        if !self.envs.is_empty() {
            s.push_str("\nenvironment:\n");
            for e in &self.envs {
                s.push_str(&format!("  {}={}\n      {}\n", e.name, e.kind, e.doc));
            }
        }
        s
    }

    fn usage_tail(&self) -> String {
        let mut parts = Vec::new();
        if !self.modes.is_empty() {
            let alts: Vec<String> = self
                .modes
                .iter()
                .map(|m| {
                    let mut t = vec![m.name.to_string()];
                    t.extend(m.mode.args.iter().map(slot_token));
                    t.join(" ")
                })
                .collect();
            parts.push(format!("[{}]", alts.join(" | ")));
        }
        parts.extend(self.args.iter().map(slot_token));
        if parts.is_empty() {
            String::new()
        } else {
            format!(" {}", parts.join(" "))
        }
    }
}

fn slot_token(a: &Arg) -> String {
    let req = matches!(a.def, Def::Required);
    let body = match a.slot {
        Slot::Pos | Slot::Rest => a.name.to_string(),
        Slot::Key => format!("{}={}", a.name, a.val.describe()),
        Slot::Opt => format!("--{} {}", a.name, a.val.describe()),
        Slot::Flag => a.name.to_string(),
        Slot::Switch => format!("--{}", a.name),
    };
    let body = if a.slot == Slot::Rest {
        format!("{body} …")
    } else {
        body
    };
    if req {
        format!("<{body}>")
    } else {
        format!("[{body}]")
    }
}

fn arg_help(a: &Arg, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let lhs = match a.slot {
        Slot::Pos | Slot::Rest => a.name.to_string(),
        Slot::Key => format!("{}=", a.name),
        Slot::Opt => format!("--{}", a.name),
        Slot::Flag => a.name.to_string(),
        Slot::Switch => format!("--{}", a.name),
    };
    format!(
        "{pad}{:<12} {:<14} {}  {}\n",
        lhs,
        a.val.describe(),
        a.doc,
        a.def.help()
    )
}

fn wrap(text: &str, width: usize) -> String {
    let mut out = String::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > width {
            out.push_str(line.trim_end());
            out.push('\n');
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

#[derive(Clone, Debug)]
pub enum UsageError {
    HelpRequested(String),
    UnknownToken(String),
    MissingRequired(String),
    BadValue {
        arg: String,
        got: String,
        want: String,
    },
    UnknownMode {
        got: String,
        expected: Vec<&'static str>,
    },
    DuplicateKey(String),
    MissingOptValue(String),
}

impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UsageError::HelpRequested(_) => write!(f, "help requested"),
            UsageError::UnknownToken(t) => write!(f, "unknown argument `{t}`"),
            UsageError::MissingRequired(a) => write!(f, "missing required argument `{a}`"),
            UsageError::BadValue { arg, got, want } => {
                write!(
                    f,
                    "`{got}` is not a valid value for `{arg}` (expected {want})"
                )
            }
            UsageError::UnknownMode { got, expected } => write!(
                f,
                "`{got}` is not a mode (expected one of: {})",
                expected.join(", ")
            ),
            UsageError::DuplicateKey(k) => write!(f, "`{k}` given more than once"),
            UsageError::MissingOptValue(o) => write!(f, "`--{o}` needs a value"),
        }
    }
}

impl std::error::Error for UsageError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PgChoice {
    #[default]
    Unset,
    Auto,
    Kind(PointGroupKind),
}

struct Parser<'s> {
    spec: &'s Spec,
}

impl<'s> Parser<'s> {
    fn new(spec: &'s Spec) -> Parser<'s> {
        Parser { spec }
    }

    fn run(self, argv: Vec<String>) -> Result<Matches, UsageError> {
        let spec = self.spec;
        let mut fmt = crate::out::format();
        let mut pg = PgChoice::Unset;
        let mut values: BTreeMap<String, String> = BTreeMap::new();
        let mut lists: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut flags: Vec<String> = Vec::new();
        let mut positional: Vec<String> = Vec::new();

        let flag_names: Vec<&'static str> = spec
            .args
            .iter()
            .chain(spec.modes.iter().flat_map(|m| m.mode.args.iter()))
            .filter(|a| a.slot == Slot::Flag)
            .map(|a| a.name)
            .collect();
        let opt_names: Vec<(&'static str, Val)> = spec
            .args
            .iter()
            .chain(spec.modes.iter().flat_map(|m| m.mode.args.iter()))
            .filter(|a| matches!(a.slot, Slot::Opt | Slot::Switch))
            .map(|a| (a.name, a.val))
            .collect();
        let key_specs: Vec<(&'static str, Val)> = spec
            .args
            .iter()
            .chain(spec.modes.iter().flat_map(|m| m.mode.args.iter()))
            .filter(|a| a.slot == Slot::Key)
            .map(|a| (a.name, a.val))
            .collect();

        let mut it = argv.iter().peekable();
        while let Some(tok) = it.next() {
            if tok == "--format" {
                let v = it
                    .next()
                    .ok_or(UsageError::MissingOptValue("format".into()))?;
                fmt = match v.as_str() {
                    "text" => Format::Text,
                    "tsv" => Format::Tsv,
                    other => {
                        return Err(UsageError::BadValue {
                            arg: "--format".into(),
                            got: other.into(),
                            want: "text or tsv".into(),
                        });
                    }
                };
                continue;
            }
            if spec.pg && tok == "--pg" {
                pg = match it.peek().and_then(|s| PointGroupKind::from_name(s)) {
                    Some(k) => {
                        it.next();
                        PgChoice::Kind(k)
                    }
                    None => PgChoice::Auto,
                };
                continue;
            }
            if let Some(rest) = tok.strip_prefix("--") {
                let (nm, inline) = match rest.split_once('=') {
                    Some((a, b)) => (a, Some(b.to_string())),
                    None => (rest, None),
                };
                let Some(&(name, val)) = opt_names.iter().find(|(n, _)| *n == nm) else {
                    return Err(UsageError::UnknownToken(tok.clone()));
                };
                if val == Val::Bool && inline.is_none() {
                    flags.push(name.to_string());
                    continue;
                }
                let v = match inline {
                    Some(v) => v,
                    None => it
                        .next()
                        .cloned()
                        .ok_or_else(|| UsageError::MissingOptValue(name.to_string()))?,
                };
                val.check(name, &v)?;
                if values.insert(name.to_string(), v).is_some() {
                    return Err(UsageError::DuplicateKey(name.to_string()));
                }
                continue;
            }
            if let Some((k, v)) = tok.split_once('=') {
                let Some(&(name, val)) = key_specs.iter().find(|(n, _)| *n == k) else {
                    return Err(UsageError::UnknownToken(tok.clone()));
                };
                val.check(name, v)?;
                if let Val::List(_) = val {
                    let items = v
                        .split(',')
                        .filter(|p| !p.is_empty())
                        .map(str::to_string)
                        .collect();
                    if lists.insert(name.to_string(), items).is_some() {
                        return Err(UsageError::DuplicateKey(name.to_string()));
                    }
                } else if values.insert(name.to_string(), v.to_string()).is_some() {
                    return Err(UsageError::DuplicateKey(name.to_string()));
                }
                continue;
            }
            if let Some(&name) = flag_names.iter().find(|n| **n == tok.as_str()) {
                flags.push(name.to_string());
                continue;
            }
            positional.push(tok.clone());
        }

        let mut mode: &'static str = "";
        if !spec.modes.is_empty() {
            let names: Vec<&'static str> = spec.modes.iter().map(|m| m.name).collect();
            if let Some(f) = spec.mode_classifier {
                mode = f(&positional).or(spec.default_mode).unwrap_or("");
            } else if let Some(tok) = positional.get(spec.mode_at) {
                match names.iter().find(|n| **n == tok.as_str()) {
                    Some(n) => {
                        mode = n;
                        positional.remove(spec.mode_at);
                    }
                    None => match spec.default_mode {
                        Some(d) => mode = d,
                        None => {
                            return Err(UsageError::UnknownMode {
                                got: tok.clone(),
                                expected: names,
                            });
                        }
                    },
                }
            } else {
                mode = spec.default_mode.unwrap_or("");
            }
        }

        let active: Vec<&Arg> = spec
            .args
            .iter()
            .chain(
                spec.modes
                    .iter()
                    .filter(|m| m.name == mode)
                    .flat_map(|m| m.mode.args.iter()),
            )
            .collect();

        let mut queue = positional.into_iter();
        for a in active.iter().filter(|a| a.slot == Slot::Pos) {
            match queue.next() {
                Some(v) => {
                    a.val.check(a.name, &v)?;
                    values.insert(a.name.to_string(), v);
                }
                None => {
                    if matches!(a.def, Def::Required) {
                        return Err(UsageError::MissingRequired(a.name.to_string()));
                    }
                }
            }
        }
        let leftover: Vec<String> = queue.collect();
        if let Some(a) = active.iter().find(|a| a.slot == Slot::Rest) {
            for v in &leftover {
                a.val.check(a.name, v)?;
            }
            if leftover.is_empty() {
                if let Def::List(d) = a.def {
                    lists.insert(
                        a.name.to_string(),
                        d.iter().map(|s| s.to_string()).collect(),
                    );
                } else if matches!(a.def, Def::Required) {
                    return Err(UsageError::MissingRequired(a.name.to_string()));
                } else {
                    lists.insert(a.name.to_string(), Vec::new());
                }
            } else {
                lists.insert(a.name.to_string(), leftover);
            }
        } else if let Some(extra) = leftover.first() {
            return Err(UsageError::UnknownToken(extra.clone()));
        }

        for a in &active {
            match a.slot {
                Slot::Pos | Slot::Key | Slot::Opt
                    if !values.contains_key(a.name) && !lists.contains_key(a.name) =>
                {
                    if let Def::List(d) = a.def {
                        lists.insert(
                            a.name.to_string(),
                            d.iter().map(|s| s.to_string()).collect(),
                        );
                    } else if let Some(t) = a.def.text() {
                        values.insert(a.name.to_string(), t);
                    } else if matches!(a.def, Def::Required) {
                        return Err(UsageError::MissingRequired(a.name.to_string()));
                    }
                }
                _ => {}
            }
        }

        Ok(Matches {
            target: spec.name,
            mode,
            values,
            lists,
            flags,
            pg,
            fmt,
            echo: argv.join(" "),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Matches {
    target: &'static str,
    mode: &'static str,
    values: BTreeMap<String, String>,
    lists: BTreeMap<String, Vec<String>>,
    flags: Vec<String>,
    pg: PgChoice,
    fmt: Format,
    echo: String,
}

impl Matches {
    pub fn target(&self) -> &'static str {
        self.target
    }

    pub fn mode(&self) -> &'static str {
        self.mode
    }

    pub fn is(&self, mode: &str) -> bool {
        self.mode == mode
    }

    fn get(&self, name: &str) -> &str {
        self.values
            .get(name)
            .unwrap_or_else(|| panic!("`{name}` is not declared in {}'s spec", self.target))
    }

    pub fn str(&self, name: &str) -> &str {
        self.get(name)
    }
    pub fn strs(&self, name: &str) -> &[String] {
        self.lists
            .get(name)
            .unwrap_or_else(|| panic!("`{name}` is not a list in {}'s spec", self.target))
    }
    pub fn strs_opt(&self, name: &str) -> Option<&[String]> {
        self.lists.get(name).map(Vec::as_slice)
    }
    pub fn usize(&self, name: &str) -> usize {
        self.get(name).parse().expect("validated at parse")
    }
    pub fn i64(&self, name: &str) -> i64 {
        self.get(name).parse().expect("validated at parse")
    }
    pub fn f64(&self, name: &str) -> f64 {
        self.get(name).parse().expect("validated at parse")
    }
    pub fn path(&self, name: &str) -> PathBuf {
        PathBuf::from(self.get(name))
    }
    pub fn enum_of<T: FromStr>(&self, name: &str) -> T
    where
        T::Err: fmt::Debug,
    {
        self.get(name).parse().expect("validated at parse")
    }

    pub fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }

    pub fn str_opt(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }
    pub fn usize_opt(&self, name: &str) -> Option<usize> {
        self.values.get(name).and_then(|v| v.parse().ok())
    }
    pub fn f64_opt(&self, name: &str) -> Option<f64> {
        self.values.get(name).and_then(|v| v.parse().ok())
    }
    pub fn path_opt(&self, name: &str) -> Option<PathBuf> {
        self.values.get(name).map(PathBuf::from)
    }

    pub fn usize_or(&self, name: &str, fallback: usize) -> usize {
        self.usize_opt(name).unwrap_or(fallback)
    }
    pub fn f64_or(&self, name: &str, fallback: f64) -> f64 {
        self.f64_opt(name).unwrap_or(fallback)
    }
    pub fn str_or<'a>(&'a self, name: &str, fallback: &'a str) -> &'a str {
        self.str_opt(name).unwrap_or(fallback)
    }

    pub fn pg(&self) -> PgChoice {
        self.pg
    }
    pub fn format(&self) -> Format {
        self.fmt
    }
    pub fn echo(&self) -> &str {
        &self.echo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    fn n2() -> Spec {
        Spec::new("n2")
            .about("N2/cc-pVDZ diagnostics on the real integrals")
            .mode("matvec", "profile H·x", |m| {
                m.pos("R", "per-sector rank", Val::Usize, Def::Usize(16))
            })
            .mode("mals", "run the MALS workflow", |m| {
                m.pos("R", "bond-dim cap", Val::Usize, Def::Usize(50)).pos(
                    "sweeps",
                    "outer iterations",
                    Val::Usize,
                    Def::Usize(2),
                )
            })
            .mode("fast", "a quick run", |m| {
                m.pos("R", "per-sector rank", Val::Usize, Def::Usize(16))
                    .pos(
                        "m",
                        "sketch chain rank",
                        Val::Usize,
                        Def::Derived("Nup+Ndn"),
                    )
            })
            .default_mode("matvec")
    }

    #[test]
    fn t_mode_dispatch_binds_that_mode_positionals() {
        let m = n2().parse_from(argv("mals 40 3")).unwrap();
        assert_eq!(m.mode(), "mals");
        assert_eq!((m.usize("R"), m.usize("sweeps")), (40, 3));
    }

    #[test]
    fn t_default_mode_applies_with_no_arguments() {
        let m = n2().parse_from(argv("")).unwrap();
        assert_eq!(m.mode(), "matvec");
        assert_eq!(m.usize("R"), 16);
    }

    #[test]
    fn t_per_mode_positionals_cannot_share_an_index() {
        assert_eq!(n2().parse_from(argv("fast 32")).unwrap().usize("R"), 32);
        assert_eq!(n2().parse_from(argv("matvec 32")).unwrap().usize("R"), 32);
    }

    #[test]
    fn t_derived_default_is_absent_until_the_caller_supplies_it() {
        let m = n2().parse_from(argv("fast 32")).unwrap();
        assert_eq!(m.usize_opt("m"), None);
        assert_eq!(m.usize_or("m", 14), 14);
    }

    fn solver() -> Spec {
        Spec::new("rpa_solver")
            .pos(
                "fixture",
                "a name under testdata/, or a path",
                Val::Path,
                Def::Required,
            )
            .pos("M", "TT rank", Val::Usize, Def::Usize(32))
            .key(
                "arms",
                "solver arms to interleave",
                Val::List(&Val::Enum(&["lanczos", "davpc", "davnopc", "penpc"])),
                Def::List(&["lanczos", "davpc", "davnopc", "penpc"]),
            )
            .key(
                "bmax",
                "pooled cap for stage B",
                Val::Usize,
                Def::Derived("4*M"),
            )
            .flag("dense", "also run the dense oracle")
    }

    #[test]
    fn t_key_value_and_bare_flags_are_position_independent() {
        let m = solver()
            .parse_from(argv("h2o_631g dense arms=lanczos,davpc 48"))
            .unwrap();
        assert_eq!(m.str("fixture"), "h2o_631g");
        assert_eq!(m.usize("M"), 48);
        assert_eq!(m.strs("arms"), ["lanczos", "davpc"]);
        assert!(m.flag("dense"));
        assert_eq!(m.usize_or("bmax", 4 * 48), 192);
    }

    #[test]
    fn t_list_default_applies_when_the_key_is_absent() {
        let m = solver().parse_from(argv("h2o_631g")).unwrap();
        assert_eq!(m.strs("arms").len(), 4);
    }

    #[test]
    fn t_missing_required_is_named() {
        let e = solver().parse_from(argv("")).unwrap_err();
        assert!(
            matches!(e, UsageError::MissingRequired(ref a) if a == "fixture"),
            "{e}"
        );
    }

    #[test]
    fn t_enum_miss_lists_the_expected_set() {
        let e = solver()
            .parse_from(argv("h2o_631g arms=davison"))
            .unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("davison") && msg.contains("lanczos"), "{msg}");
    }

    #[test]
    fn t_unknown_token_is_refused_rather_than_ignored() {
        let e = Spec::new("weingarten").parse_from(argv("foo")).unwrap_err();
        assert!(
            matches!(e, UsageError::UnknownToken(ref t) if t == "foo"),
            "{e}"
        );
    }

    #[test]
    fn t_a_no_argument_spec_still_gets_help_and_format() {
        let s = Spec::new("weingarten").about("Weingarten diagnostics");
        let h = s.help();
        assert!(h.contains("--format"), "{h}");
        assert!(h.contains("--help"), "{h}");
        let m = Spec::new("weingarten")
            .parse_from(argv("--format tsv"))
            .unwrap();
        assert_eq!(m.format(), Format::Tsv);
    }

    #[test]
    fn t_help_is_requested_not_an_error() {
        let e = n2().parse_from(argv("--help")).unwrap_err();
        match e {
            UsageError::HelpRequested(t) => {
                assert!(t.contains("modes:"), "{t}");
                assert!(t.contains("matvec"), "{t}");
            }
            other => panic!("{other}"),
        }
    }

    #[test]
    fn t_pg_is_tri_state() {
        let s = || {
            Spec::new("t")
                .pg()
                .pos("f", "fixture", Val::Str, Def::Optional)
        };
        assert_eq!(s().parse_from(argv("x")).unwrap().pg(), PgChoice::Unset);
        assert_eq!(s().parse_from(argv("x --pg")).unwrap().pg(), PgChoice::Auto);
        assert!(matches!(
            s().parse_from(argv("x --pg d2h")).unwrap().pg(),
            PgChoice::Kind(_)
        ));
    }
}
