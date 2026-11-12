// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

pub mod args;
pub mod cis;
pub mod fixture;
pub mod grad;
pub mod mem;
pub mod out;
pub mod ranks;
pub mod repo;
pub mod stats;

pub mod prelude {
    pub use crate::args::{Def, Matches, PgChoice, Spec, Val};
    pub use crate::fixture::Fixture;
    pub use crate::out::{Cell, Col, Format, Summary, Table, banner, banner_kv, flush, note};
    pub use crate::repo::Repo;
    pub use crate::row;
    pub use crate::stats::{Stats, median, stats, time, time_ms};
}

#[macro_export]
macro_rules! repo {
    () => {
        $crate::repo::Repo::from_manifest_dir(env!("CARGO_MANIFEST_DIR"))
    };
}

pub fn seeded(seed: u64) -> rand_xoshiro::Xoshiro256PlusPlus {
    use rand::SeedableRng;
    rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed)
}

pub fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub fn env_set(name: &str) -> bool {
    std::env::var_os(name).is_some()
}
