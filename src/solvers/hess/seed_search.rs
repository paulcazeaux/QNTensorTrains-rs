// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

/// How far to walk along the proven negative direction before handing over to the polish.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeedSearchOpts {
    /// Largest step tried — the top rung of the ladder. Prefer
    /// [`SeedSearchOpts::from_residual`] over the bare default: the residual norm is the scale on
    /// which the state is already wrong, and no seed should be larger than that.
    pub eps0: f64,
    /// Geometric shrink factor, in `(0, 1)`.
    pub shrink: f64,
    /// Ladder length (an upper bound on retraction+energy evaluations).
    pub steps: usize,
    /// Required fraction of the model gain: accept `ε` when
    /// `E(x) − E(ε) ≥ c1 · |λ| · ε²`. Must lie in `(0, 1)`.
    pub c1: f64,
    /// Absolute floor on the accepted decrease, as a fraction of `|E(x)|`: also require
    /// `E(x) − E(ε) ≥ min_decrease_rel · |E(x)|`.
    pub min_decrease_rel: f64,
}

impl Default for SeedSearchOpts {
    fn default() -> Self {
        Self {
            eps0: 1e-1,
            shrink: 0.5,
            steps: 24,
            c1: 0.5,
            min_decrease_rel: 1e-12,
        }
    }
}

impl SeedSearchOpts {
    /// Defaults, with the ladder started at the base point's residual norm `‖H x − E x‖`.
    pub fn from_residual(residual: f64) -> Self {
        Self {
            eps0: residual,
            ..Self::default()
        }
    }
}

/// What [`geometric_seed_search`] found.
#[derive(Clone, Debug)]
pub struct SeedSearch<V> {
    /// The accepted seed, as returned by the caller's retraction.
    pub state: V,
    /// The accepted step.
    pub eps: f64,
    /// `E(seed)`.
    pub energy: f64,
    /// The model's predicted energy change at `eps`, `λ·eps²` relative to `E(x)` — negative.
    pub predicted: f64,
    /// `(ε, E(ε))` for every rung evaluated, in the order tried. The ratio
    /// `(E(x) − E) / (|λ|ε²)` across these is the retraction-penalty scaling, and worth logging:
    /// it should approach `1` from below as `ε` falls, and a run where it never does is one where
    /// the base point is too ill-conditioned for the witness to be actionable.
    pub trials: Vec<(f64, f64)>,
}

impl<V> SeedSearch<V> {
    /// Achieved decrease as a fraction of the model's prediction. `1` means the quadratic model
    /// was realized exactly; `< 1` means the retraction ate part of it.
    pub fn quality(&self, e_ref: f64) -> f64 {
        (e_ref - self.energy) / (-self.predicted)
    }
}

/// Geometric (backtracking) line search for the complex seed step, so that the seed handed to the
/// polish is a **true descent step** rather than a fixed guess.
pub fn geometric_seed_search<V, F>(
    e_ref: f64,
    lambda: f64,
    opts: &SeedSearchOpts,
    mut retract: F,
) -> Option<SeedSearch<V>>
where
    F: FnMut(f64) -> (V, f64),
{
    assert!(
        opts.eps0 > 0.0 && opts.eps0.is_finite(),
        "seed search: eps0 must be finite and positive, got {}",
        opts.eps0
    );
    assert!(
        opts.shrink > 0.0 && opts.shrink < 1.0,
        "seed search: shrink must lie in (0, 1), got {}",
        opts.shrink
    );
    assert!(
        opts.c1 > 0.0 && opts.c1 < 1.0,
        "seed search: c1 must lie in (0, 1), got {}",
        opts.c1
    );
    if !lambda.is_finite() || lambda >= 0.0 {
        return None;
    }

    let mut trials = Vec::with_capacity(opts.steps);
    let mut best: Option<SeedSearch<V>> = None;
    let mut worse_since_best = 0usize;
    let mut eps = opts.eps0;

    for _ in 0..opts.steps {
        let (v, e) = retract(eps);
        trials.push((eps, e));
        let predicted = lambda * eps * eps;
        let floor = opts.min_decrease_rel * e_ref.abs();
        let accepted = e.is_finite() && (e_ref - e) >= (opts.c1 * (-predicted)).max(floor);
        let improves = best.as_ref().is_none_or(|b| e < b.energy);
        if accepted && improves {
            best = Some(SeedSearch {
                state: v,
                eps,
                energy: e,
                predicted,
                trials: Vec::new(),
            });
            worse_since_best = 0;
        } else if best.is_some() {
            worse_since_best += 1;
            if worse_since_best >= 2 {
                break;
            }
        }
        eps *= opts.shrink;
    }

    best.map(|mut b| {
        b.trials = trials;
        b
    })
}

#[cfg(test)]
mod seed_search_tests {
    use super::*;

    fn model_at(e_ref: f64, lambda: f64, c: f64) -> impl FnMut(f64) -> ((), f64) {
        move |eps: f64| ((), e_ref + lambda * eps * eps + c * eps.powi(4))
    }

    fn model(lambda: f64, c: f64) -> impl FnMut(f64) -> ((), f64) {
        model_at(0.0, lambda, c)
    }

    #[test]
    fn accepts_the_deepest_rung_of_the_ladder() {
        let (lambda, c) = (-2.146868e-2, 1.0e4);
        let opts = SeedSearchOpts::default();
        let r =
            geometric_seed_search(0.0, lambda, &opts, model(lambda, c)).expect("a descent step");
        assert!(r.energy < 0.0, "seed must go DOWN, got {}", r.energy);
        let star = (-lambda / (2.0 * c)).sqrt();
        assert!(
            r.eps <= star / opts.shrink && r.eps >= star * opts.shrink,
            "eps {} not within a rung of the minimizer {star}",
            r.eps
        );
        let deeper: Vec<_> = r
            .trials
            .iter()
            .filter(|&&(eps, e)| e < r.energy && (0.0 - e) >= opts.c1 * (-lambda) * eps * eps)
            .collect();
        assert!(
            deeper.is_empty(),
            "a deeper accepted rung was passed over: {deeper:?}"
        );
        assert!(r.quality(0.0) > 0.5);
    }

    #[test]
    fn refuses_a_non_negative_direction() {
        let opts = SeedSearchOpts::default();
        assert!(geometric_seed_search(0.0, 0.0, &opts, model(0.0, 1.0)).is_none());
        assert!(geometric_seed_search(0.0, 3e-2, &opts, model(3e-2, 1.0)).is_none());
    }

    #[test]
    fn refuses_when_the_penalty_dominates_everywhere() {
        let lambda = -1e-12;
        let opts = SeedSearchOpts {
            steps: 4,
            ..SeedSearchOpts::default()
        };
        assert!(geometric_seed_search(0.0, lambda, &opts, model(lambda, 1.0)).is_none());
    }

    #[test]
    fn the_absolute_floor_refuses_an_unresolvable_step() {
        let e_ref = -76.119120676415;
        let (lambda, c) = (-1e-6, 1.0);
        let opts = SeedSearchOpts::from_residual(1e-2);
        assert!(geometric_seed_search(e_ref, lambda, &opts, model_at(e_ref, lambda, c)).is_none());
        let off = SeedSearchOpts {
            min_decrease_rel: 0.0,
            ..opts
        };
        let r = geometric_seed_search(e_ref, lambda, &off, model_at(e_ref, lambda, c))
            .expect("the relative test alone accepts");
        assert!(r.energy < e_ref);
        assert!(e_ref - r.energy < 1e-12 * e_ref.abs());
    }

    #[test]
    fn from_residual_only_moves_the_top_rung() {
        let o = SeedSearchOpts::from_residual(3.5e-3);
        let d = SeedSearchOpts::default();
        assert_eq!(o.eps0, 3.5e-3);
        assert_eq!((o.shrink, o.steps, o.c1), (d.shrink, d.steps, d.c1));
        assert_eq!(o.min_decrease_rel, d.min_decrease_rel);
    }

    #[test]
    fn stops_two_rungs_past_the_best() {
        let (lambda, c) = (-2.0e-2, 1.0e4);
        let opts = SeedSearchOpts::default();
        let r =
            geometric_seed_search(0.0, lambda, &opts, model(lambda, c)).expect("a descent step");
        assert!(
            r.trials.len() < opts.steps,
            "walked the whole ladder ({} rungs)",
            r.trials.len()
        );
        let k = r.trials.iter().position(|&(e, _)| e == r.eps).unwrap();
        assert_eq!(
            r.trials.len(),
            k + 3,
            "expected exactly two rungs past best"
        );
    }
}

#[cfg(test)]
mod driver_fork_tests {

    use crate::par;
    use crate::solvers::InnerProductVector;
    use crate::solvers::hess::davidson::recombine;

    #[derive(Clone)]
    struct FlatVec {
        data: Vec<f64>,
    }

    impl InnerProductVector for FlatVec {
        type Scalar = f64;
        fn dot(&self, o: &Self) -> f64 {
            self.data.iter().zip(&o.data).map(|(a, b)| a * b).sum()
        }
        fn norm(&self) -> f64 {
            self.dot(self).sqrt()
        }
        fn axpy(&mut self, a: f64, o: &Self) {
            for (x, &y) in self.data.iter_mut().zip(&o.data) {
                *x += a * y;
            }
        }
        fn scale(&mut self, a: f64) {
            for x in &mut self.data {
                *x *= a;
            }
        }
        fn for_each_coeff_mut<F: FnMut(&mut f64)>(&mut self, mut f: F) {
            for x in &mut self.data {
                f(x);
            }
        }
        fn zip_coeffs_mut<F: FnMut(&mut f64, f64)>(&mut self, o: &Self, mut f: F) {
            for (x, &y) in self.data.iter_mut().zip(&o.data) {
                f(x, y);
            }
        }
        fn coeff(&self, i: usize) -> f64 {
            self.data[i]
        }
        fn set_coeff(&mut self, i: usize, v: f64) {
            self.data[i] = v;
        }
        fn num_coeffs(&self) -> usize {
            self.data.len()
        }
    }

    fn basis(m: usize, n: usize) -> Vec<FlatVec> {
        (0..m)
            .map(|j| FlatVec {
                data: (0..n)
                    .map(|i| ((i * 31 + j * 17) % 97) as f64 / 97.0 - 0.5)
                    .collect(),
            })
            .collect()
    }

    fn coeffs(v: &[FlatVec]) -> Vec<f64> {
        v.iter().flat_map(|x| x.data.iter().copied()).collect()
    }

    fn bits_eq(a: &[f64], b: &[f64], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}: entry {i}");
        }
    }

    fn arms<R: Send>(f: impl Fn() -> R + Send + Sync) -> (R, R, R) {
        let _lock = par::lock_par_switches();
        let prev_on = par::set_vector_par(false);
        let ser = f();
        par::set_vector_par(true);
        let prev_floor = par::set_par_min_fork_work(0);
        let run = |nt: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap()
                .install(&f)
        };
        let p8 = run(8);
        let p16 = run(16);
        par::set_par_min_fork_work(prev_floor);
        par::set_vector_par(prev_on);
        (ser, p8, p16)
    }

    #[test]
    fn recombine_output_fork_is_bit_identical() {
        let (m, n, keep) = (24usize, 40usize, 6usize);
        let b = basis(m, n);
        let y: Vec<Vec<f64>> = (0..keep)
            .map(|j| {
                (0..m)
                    .map(|i| ((i * 7 + j * 13) % 19) as f64 / 19.0 - 0.5)
                    .collect()
            })
            .collect();
        let (ser, p8, p16) = arms(|| recombine(&b, &y, keep));
        assert_eq!(ser.len(), keep);
        bits_eq(&coeffs(&ser), &coeffs(&p8), "recombine at 8 threads");
        bits_eq(&coeffs(&ser), &coeffs(&p16), "recombine at 16 threads");
        assert!(coeffs(&ser).iter().any(|&x| x.abs() > 1e-3));
    }

    #[test]
    fn recombine_single_output_stays_serial_and_correct() {
        let (m, n) = (12usize, 16usize);
        let b = basis(m, n);
        let y = vec![
            (0..m)
                .map(|i| (i as f64) * 0.125 - 0.5)
                .collect::<Vec<f64>>(),
        ];
        let out = recombine(&b, &y, 1);
        assert_eq!(out.len(), 1);
        for i in 0..n {
            let expect: f64 = (0..m).map(|k| y[0][k] * b[k].data[i]).sum();
            assert!((out[0].data[i] - expect).abs() <= 1e-12 * expect.abs().max(1.0));
        }
    }
}
