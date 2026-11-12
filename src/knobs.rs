// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! **Every environment variable the library reads**, in one place. They are MEASUREMENT
//! INSTRUMENTS, not a tuning interface: the compiled-in constants stay the defaults and the
//! documentation of record, and a run with none of these variables set behaves exactly as without
//! this module. (Examples and benches parse their own `QNTT_*` run parameters through
//! `qntt_exkit`; those are listed in each binary's `--help`, not here.)

use std::sync::OnceLock;

pub(crate) const NEVER: usize = usize::MAX;

pub(crate) struct Knob {
    name: &'static str,
    default: usize,
    cell: OnceLock<usize>,
}

impl Knob {
    const fn new(name: &'static str, default: usize) -> Self {
        Knob {
            name,
            default,
            cell: OnceLock::new(),
        }
    }

    #[inline]
    pub(crate) fn get(&self) -> usize {
        *self.cell.get_or_init(|| self.resolve(env_lookup))
    }

    fn resolve(&self, lookup: impl FnOnce(&str) -> Option<String>) -> usize {
        parse(self.name, lookup(self.name).as_deref(), self.default)
    }
}

pub(crate) fn env_lookup(name: &str) -> Option<String> {
    std::env::var_os(name).map(|v| {
        v.into_string()
            .unwrap_or_else(|v| panic!("{name}={v:?}: not valid UTF-8"))
    })
}

pub(crate) fn parse(name: &str, raw: Option<&str>, default: usize) -> usize {
    let Some(raw) = raw else {
        return default;
    };
    match raw.trim() {
        "never" => NEVER,
        "always" => 0,
        s => s.parse().unwrap_or_else(|_| {
            panic!("{name}={raw:?}: expected a decimal usize, `always` (= 0) or `never`")
        }),
    }
}

pub(crate) static COREOPS_BLOCKS: Knob = Knob::new(
    "QNTT_COREOPS_BLOCKS",
    crate::su2::coreops::PAR_MIN_COREOPS_BLOCKS,
);
pub(crate) static COREOPS_WORK: Knob = Knob::new(
    "QNTT_COREOPS_WORK",
    crate::su2::coreops::PAR_MIN_COREOPS_WORK,
);
pub(crate) static COREOPS_TASK_WORK: Knob = Knob::new(
    "QNTT_COREOPS_TASK_WORK",
    crate::su2::coreops::PAR_MIN_COREOPS_TASK_WORK,
);
pub(crate) static SPLIT_TARGET: Knob = Knob::new(
    "QNTT_SPLIT_TARGET",
    crate::hamiltonian::matvec::BLOCK_SPLIT_TARGET as usize,
);
pub(crate) static SPLIT_MIN_WIDTH: Knob =
    Knob::new("QNTT_SPLIT_MIN_WIDTH", crate::su2::shard::SPLIT_MIN_WIDTH);
pub(crate) static GEMM_PAR_MIN_WORK: Knob =
    Knob::new("QNTT_GEMM_PAR_MIN_WORK", crate::scalar::GEMM_PAR_MIN_WORK);
pub(crate) static GEMM_PAR_WIDTH: Knob =
    Knob::new("QNTT_GEMM_PAR_WIDTH", crate::scalar::GEMM_PAR_WIDTH);

static ALL: [&Knob; 7] = [
    &COREOPS_BLOCKS,
    &COREOPS_WORK,
    &COREOPS_TASK_WORK,
    &SPLIT_TARGET,
    &SPLIT_MIN_WIDTH,
    &GEMM_PAR_MIN_WORK,
    &GEMM_PAR_WIDTH,
];

/// The effective value of every knob, `NAME=value` space-separated (`never` for `usize::MAX`),
/// read through the gates' own accessors — so it also fixes them for the rest of the process.
pub fn line() -> String {
    ALL.iter()
        .map(|k| match k.get() {
            NEVER => format!("{}=never", k.name),
            v => format!("{}={v}", k.name),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) struct Flag {
    name: &'static str,
    default: bool,
    cell: OnceLock<bool>,
}

impl Flag {
    const fn off(name: &'static str) -> Self {
        Flag {
            name,
            default: false,
            cell: OnceLock::new(),
        }
    }

    const fn on(name: &'static str) -> Self {
        Flag {
            name,
            default: true,
            cell: OnceLock::new(),
        }
    }

    #[inline]
    pub(crate) fn get(&self) -> bool {
        *self.cell.get_or_init(|| self.read())
    }

    pub(crate) fn read(&self) -> bool {
        self.resolve(flag_lookup)
    }

    fn resolve(&self, lookup: impl FnOnce(&str) -> Option<String>) -> bool {
        truthy(lookup(self.name).as_deref(), self.default)
    }
}

fn flag_lookup(name: &str) -> Option<String> {
    std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
}

pub(crate) fn truthy(raw: Option<&str>, default: bool) -> bool {
    match raw {
        None | Some("") => default,
        Some("0" | "false") => false,
        Some(_) => true,
    }
}

pub(crate) static A_APPLY_EXACT: Flag = Flag::off("QNTT_A_APPLY_EXACT");
pub(crate) static B_APPLY_EXPLICIT: Flag = Flag::off("QNTT_B_APPLY_EXPLICIT");
pub(crate) static LS_SECANT: Flag = Flag::on("QNTT_LS_SECANT");
pub(crate) static LS_SECANT_ALL: Flag = Flag::on("QNTT_LS_SECANT_ALL");
pub(crate) static LS_SLOPE_FLOOR: Flag = Flag::on("QNTT_LS_SLOPE_FLOOR");
pub(crate) static JOINT_WHITE: Flag = Flag::on("QNTT_JOINT_WHITE");
pub(crate) static JOINT_CONDENSE: Flag = Flag::on("QNTT_JOINT_CONDENSE");
pub(crate) static JOINT_ACCURATE_METRIC: Flag = Flag::off("QNTT_JOINT_ACCURATE_METRIC");
pub(crate) static CG_TRACE: Flag = Flag::off("QNTT_CG_TRACE");
pub(crate) static LS_TRACE: Flag = Flag::off("QNTT_LS_TRACE");
pub(crate) static GAD_TRACE: Flag = Flag::off("QNTT_GAD_TRACE");
pub(crate) static PD_TRACE: Flag = Flag::off("QNTT_PD_TRACE");
pub(crate) static TB_TRACE: Flag = Flag::off("QNTT_TB_TRACE");
pub(crate) static OMEGA_TRACE: Flag = Flag::off("OMEGA_TRACE");

#[cfg(test)]
mod tests {
    use super::*;

    fn only(name: &'static str, value: &'static str) -> impl Fn(&str) -> Option<String> {
        move |n: &str| (n == name).then(|| value.to_string())
    }

    #[test]
    fn parse_default_number_always_never() {
        assert_eq!(parse("X", None, 7), 7);
        assert_eq!(parse("X", Some("123"), 7), 123);
        assert_eq!(parse("X", Some(" 0 "), 7), 0);
        assert_eq!(parse("X", Some("always"), 7), 0);
        assert_eq!(parse("X", Some("never"), 7), NEVER);
    }

    #[test]
    #[should_panic(expected = "QNTT_COREOPS_WORK=\"1<<20\"")]
    fn parse_malformed_panics_with_the_name() {
        parse("QNTT_COREOPS_WORK", Some("1<<20"), 0);
    }

    #[test]
    fn coreops_knobs_read_their_variables() {
        use crate::su2::coreops::{
            PAR_MIN_COREOPS_BLOCKS, PAR_MIN_COREOPS_TASK_WORK, PAR_MIN_COREOPS_WORK,
        };
        for (k, default) in [
            (&COREOPS_BLOCKS, PAR_MIN_COREOPS_BLOCKS),
            (&COREOPS_WORK, PAR_MIN_COREOPS_WORK),
            (&COREOPS_TASK_WORK, PAR_MIN_COREOPS_TASK_WORK),
        ] {
            assert_eq!(k.resolve(|_| None), default, "{}", k.name);
            assert_eq!(k.resolve(only(k.name, "4096")), 4096, "{}", k.name);
            assert_eq!(k.resolve(only(k.name, "never")), NEVER, "{}", k.name);
            assert_eq!(k.resolve(only("QNTT_GROW_ANCESTORS", "9")), default);
        }
        assert_eq!(COREOPS_WORK.name, "QNTT_COREOPS_WORK");
        assert_eq!(COREOPS_TASK_WORK.name, "QNTT_COREOPS_TASK_WORK");
        assert_eq!(COREOPS_BLOCKS.name, "QNTT_COREOPS_BLOCKS");
    }

    #[test]
    fn gemm_par_knobs_read_their_variables_and_gate_the_gemm() {
        use crate::scalar::{self, gemm_par_at};
        use faer::Par;
        assert_eq!(GEMM_PAR_MIN_WORK.name, "QNTT_GEMM_PAR_MIN_WORK");
        assert_eq!(GEMM_PAR_WIDTH.name, "QNTT_GEMM_PAR_WIDTH");
        assert_eq!(
            GEMM_PAR_MIN_WORK.resolve(|_| None),
            scalar::GEMM_PAR_MIN_WORK
        );
        assert_eq!(GEMM_PAR_WIDTH.resolve(|_| None), scalar::GEMM_PAR_WIDTH);
        assert_eq!(
            (scalar::GEMM_PAR_MIN_WORK, scalar::GEMM_PAR_WIDTH),
            (1 << 21, 4)
        );
        let floor = |v| GEMM_PAR_MIN_WORK.resolve(only("QNTT_GEMM_PAR_MIN_WORK", v));
        let width = |v| GEMM_PAR_WIDTH.resolve(only("QNTT_GEMM_PAR_WIDTH", v));
        assert_eq!(floor("never"), NEVER);
        assert_eq!(floor("always"), 0);
        assert_eq!(
            GEMM_PAR_WIDTH.resolve(only("QNTT_GEMM_PAR_MIN_WORK", "9")),
            4
        );
        let (w, d) = (scalar::GEMM_PAR_WIDTH, scalar::GEMM_PAR_MIN_WORK);
        assert_eq!(gemm_par_at(192, 192, 192, d, w, 8), Par::rayon(4));
        assert_eq!(gemm_par_at(64, 64, 64, d, w, 8), Par::Seq);
        assert_eq!(
            gemm_par_at(usize::MAX, 2, 2, floor("never"), w, 8),
            Par::Seq
        );
        assert_eq!(gemm_par_at(2, 2, 2, floor("always"), w, 8), Par::rayon(4));
        assert_eq!(gemm_par_at(4096, 1, 4096, 0, w, 8), Par::Seq);
        assert_eq!(gemm_par_at(1, 4096, 4096, 0, w, 8), Par::Seq);
        assert_eq!(gemm_par_at(192, 192, 192, d, width("0"), 8), Par::rayon(8));
        assert_eq!(gemm_par_at(192, 192, 192, d, width("1"), 8), Par::Seq);
        assert_eq!(gemm_par_at(192, 192, 192, d, width("16"), 8), Par::rayon(8));
        assert_eq!(gemm_par_at(1024, 1024, 1024, 0, w, 1), Par::Seq);
    }

    #[test]
    fn split_knobs_read_their_variables_and_shape_the_splitter() {
        use crate::su2::shard::Splitter;
        assert_eq!(SPLIT_TARGET.name, "QNTT_SPLIT_TARGET");
        assert_eq!(SPLIT_MIN_WIDTH.name, "QNTT_SPLIT_MIN_WIDTH");
        assert_eq!(SPLIT_TARGET.resolve(|_| None), 32);
        assert_eq!(SPLIT_MIN_WIDTH.resolve(|_| None), 8);
        let den = |v| SPLIT_TARGET.resolve(only("QNTT_SPLIT_TARGET", v));
        let minw = |v| SPLIT_MIN_WIDTH.resolve(only("QNTT_SPLIT_MIN_WIDTH", v));
        let (total, cost, width) = (3200, 1000, 64);
        assert_eq!(Splitter::with(true, total, 32, 8).count(cost, width), 8);
        assert_eq!(
            Splitter::with(true, total, den("64"), 8).count(cost, width),
            8
        );
        assert_eq!(
            Splitter::with(true, total, den("4"), 8).count(cost, width),
            2
        );
        assert_eq!(
            Splitter::with(true, total, den("0"), minw("0")).count(cost, width),
            16
        );
        assert_eq!(
            Splitter::with(true, total, den("never"), 8).count(cost, width),
            1
        );
        assert_eq!(
            Splitter::with(true, total, 32, minw("never")).count(cost, width),
            1
        );
    }

    #[test]
    fn truthy_one_rule_for_both_defaults() {
        for default in [false, true] {
            assert_eq!(truthy(None, default), default);
            assert_eq!(truthy(Some(""), default), default);
            assert!(!truthy(Some("0"), default));
            assert!(!truthy(Some("false"), default));
            for on in ["1", "true", "yes", "serial", " 0"] {
                assert!(truthy(Some(on), default), "{on:?}");
            }
        }
    }

    #[test]
    fn flags_read_their_variables() {
        let off = |f: &Flag| f.resolve(|_| None);
        assert!(!off(&A_APPLY_EXACT) && !off(&B_APPLY_EXPLICIT) && !off(&CG_TRACE));
        assert!(!off(&LS_TRACE) && !off(&GAD_TRACE) && !off(&PD_TRACE) && !off(&TB_TRACE));
        assert!(!off(&OMEGA_TRACE) && !off(&JOINT_ACCURATE_METRIC));
        for (f, name) in [
            (&LS_SECANT, "QNTT_LS_SECANT"),
            (&LS_SECANT_ALL, "QNTT_LS_SECANT_ALL"),
            (&LS_SLOPE_FLOOR, "QNTT_LS_SLOPE_FLOOR"),
            (&JOINT_WHITE, "QNTT_JOINT_WHITE"),
            (&JOINT_CONDENSE, "QNTT_JOINT_CONDENSE"),
        ] {
            assert!(f.resolve(|_| None), "{name}");
            assert!(f.resolve(only(name, "")), "{name}");
            assert!(f.resolve(only(name, "1")), "{name}");
            assert!(!f.resolve(only(name, "0")), "{name}");
            assert!(!f.resolve(only(name, "false")), "{name}");
        }
        assert!(!CG_TRACE.resolve(only("QNTT_CG_TRACE", "0")));
        assert!(TB_TRACE.resolve(only("QNTT_TB_TRACE", "1")));
        assert!(A_APPLY_EXACT.resolve(only("QNTT_A_APPLY_EXACT", "1")));
        assert!(!A_APPLY_EXACT.resolve(only("QNTT_B_APPLY_EXPLICIT", "1")));
        assert_eq!(OMEGA_TRACE.name, "OMEGA_TRACE");
    }
}
