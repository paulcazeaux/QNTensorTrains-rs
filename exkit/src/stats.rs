// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub n: usize,
    pub min: f64,
    pub median: f64,
    pub max: f64,
    pub mean: f64,
}

impl Stats {
    pub fn spread(&self) -> f64 {
        if self.min == 0.0 {
            f64::INFINITY
        } else {
            self.max / self.min
        }
    }
}

pub fn median(xs: &[f64]) -> f64 {
    assert!(!xs.is_empty(), "median of an empty sample");
    assert!(
        !xs.iter().any(|x| x.is_nan()),
        "median of a sample containing NaN"
    );
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("NaN excluded above"));
    v[v.len() / 2]
}

pub fn stats(xs: &[f64]) -> Stats {
    assert!(!xs.is_empty(), "stats of an empty sample");
    assert!(
        !xs.iter().any(|x| x.is_nan()),
        "stats of a sample containing NaN"
    );
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("NaN excluded above"));
    Stats {
        n: v.len(),
        min: v[0],
        median: v[v.len() / 2],
        max: v[v.len() - 1],
        mean: v.iter().sum::<f64>() / v.len() as f64,
    }
}

pub fn ms(xs: &[Duration]) -> Vec<f64> {
    xs.iter().map(|d| d.as_secs_f64() * 1e3).collect()
}

pub fn secs(xs: &[Duration]) -> Vec<f64> {
    xs.iter().map(Duration::as_secs_f64).collect()
}

pub fn stats_ms(xs: &[Duration]) -> Stats {
    stats(&ms(xs))
}

pub fn median_dur(xs: &[Duration]) -> Duration {
    assert!(!xs.is_empty(), "median of an empty sample");
    let mut v = xs.to_vec();
    v.sort();
    v[v.len() / 2]
}

pub fn time_ms<R>(f: impl FnOnce() -> R) -> (R, f64) {
    let t0 = Instant::now();
    let r = f();
    (r, t0.elapsed().as_secs_f64() * 1e3)
}

pub fn time<R>(f: impl FnOnce() -> R) -> (R, Duration) {
    let t0 = Instant::now();
    let r = f();
    (r, t0.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_median_is_the_upper_one_on_even_n() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 3.0);
        assert_eq!(median(&[4.0, 3.0, 2.0, 1.0]), 3.0);
    }

    #[test]
    fn t_median_on_odd_n_is_the_middle() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn t_stats_agrees_with_median() {
        let xs = [5.0, 1.0, 4.0, 2.0];
        let s = stats(&xs);
        assert_eq!(s.median, median(&xs));
        assert_eq!((s.n, s.min, s.max), (4, 1.0, 5.0));
        assert_eq!(s.mean, 3.0);
        assert_eq!(s.spread(), 5.0);
    }

    #[test]
    #[should_panic(expected = "NaN")]
    fn t_median_refuses_nan() {
        median(&[1.0, f64::NAN, 2.0]);
    }

    #[test]
    #[should_panic(expected = "empty")]
    fn t_median_refuses_empty() {
        median(&[]);
    }
}
