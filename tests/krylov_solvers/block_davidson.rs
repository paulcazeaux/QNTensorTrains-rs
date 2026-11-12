// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use faer::Mat;
use faer::prelude::ReborrowMut;
use qntt::core::SparseCore;
use qntt::core::factor::dense_self_adjoint_eigen;
use qntt::qn::{QnSpace, RankGrid, Sector};
use qntt::solvers::InnerProductVector;
use qntt::solvers::als::{left_to_right_sketching, right_to_left_sketching};
use qntt::solvers::davidson::{DavidsonOpts, davidson, precond};
use qntt::solvers::davidson_block::{
    BlockDavidsonInfo, BlockDavidsonOpts, davidson_block, davidson_block_projected,
    davidson_block_projected_batched, home_roots,
};
use qntt::solvers::mals::{ContractedSparseCores, contract};
use qntt::su2::frame::Su2Sketch;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::Su2Space;
use qntt::su2::solvers::als as su2als;
use qntt::su2::solvers::mals::{Su2ContractedCores, contract as su2_contract};
use qntt::su2::vector::Su2TtVector;
use qntt::{C64, CoeffTensor, Family, Scalar, Sketch, SparseHamiltonian, TtVector};
use qntt_testkit::dense::{dense_eigen, dense_hamiltonian_matrix, dets_for};
use qntt_testkit::generic_truncated_tv;
use qntt_testkit::seeded;
use rand_xoshiro::Xoshiro256PlusPlus;

fn degenerate_tv(d: usize) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 4, 1], -1.0);
    v.set(&[1, 4, 0, 2], -1.0);
    (t, v)
}

fn gauge_integrals(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    theta: &[f64],
) -> (CoeffTensor<C64>, CoeffTensor<C64>) {
    let ph = |a: f64| C64::new(a.cos(), a.sin());
    let mut tc = CoeffTensor::<C64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            let c = t.get(&[i, j]);
            if c != 0.0 {
                tc.set(&[i, j], ph(theta[i] - theta[j]) * C64::new(c, 0.0));
            }
        }
    }
    let mut vc = CoeffTensor::<C64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let c = v.get(&[i, j, k, l]);
                    if c != 0.0 {
                        vc.set(
                            &[i, j, k, l],
                            ph(theta[i] - theta[j] + theta[k] - theta[l]) * C64::new(c, 0.0),
                        );
                    }
                }
            }
        }
    }
    (tc, vc)
}

fn abelian_local<T: Scalar>(
    space: QnSpace,
    h: &SparseHamiltonian<T>,
    k: usize,
    seed: u64,
) -> (
    TtVector<T>,
    Sketch<T>,
    Sketch<T>,
    ContractedSparseCores<T>,
    ContractedSparseCores<T>,
) {
    let mut rng = seeded(seed);
    let mut x: TtVector<T> = qntt::tt_randn(space, &qntt::max_ranks(&space), &mut rng);
    x.right_orthogonalize(false);
    x.move_core(k, false);
    x.check();
    let fl = left_to_right_sketching(h, &x)[k].clone();
    let fr = right_to_left_sketching(h, &x)[k + 1].clone();
    let cp0 = contract(x.core(k), x.core(k + 1));
    let d_diag = h.two_site_effective_diagonal(&fl, &fr, &cp0);
    (x, fl, fr, cp0, d_diag)
}

fn su2_local<T: Scalar>(
    space: Su2Space,
    h: &Su2SparseHamiltonian<T>,
    k: usize,
    seed: u64,
) -> (
    Su2TtVector<T>,
    Su2Sketch<T>,
    Su2Sketch<T>,
    Su2ContractedCores<T>,
    Su2ContractedCores<T>,
) {
    let mut rng = seeded(seed);
    let mut x: Su2TtVector<T> =
        qntt::su2::vector::tt_randn(space, &qntt::su2::qn::max_ranks(&space), &mut rng);
    x.right_orthogonalize(false);
    x.move_core(k, false);
    let fl = su2als::left_to_right_sketching(h, &x)[k].clone();
    let fr = su2als::right_to_left_sketching(h, &x)[k + 1].clone();
    let cp0 = su2_contract(x.core(k), x.core(k + 1));
    let d_diag = h.two_site_effective_diagonal(&fl, &fr, &cp0);
    (x, fl, fr, cp0, d_diag)
}

fn local_dense<V, Op>(op: &mut Op, template: &V) -> (Mat<V::Scalar>, Vec<f64>, f64)
where
    V: InnerProductVector,
    Op: FnMut(&V) -> V,
{
    let n = template.num_coeffs();
    let basis: Vec<V> = (0..n)
        .map(|j| {
            let mut e = template.clone();
            e.for_each_coeff_mut(|c| *c = V::Scalar::zero());
            e.set_coeff(j, V::Scalar::one());
            e
        })
        .collect();
    let mut images: Vec<V> = Vec::with_capacity(n);
    for e in &basis {
        images.push(op(e));
    }
    let mut m = Mat::<V::Scalar>::zeros(n, n);
    let mut g = vec![0.0f64; n];
    let mut goff = 0.0f64;
    for i in 0..n {
        g[i] = basis[i].dot(&basis[i]).re_s();
        for j in 0..n {
            *m.get_mut(i, j) = basis[i].dot(&images[j]);
            if i != j {
                goff = goff.max(basis[i].dot(&basis[j]).abs_s());
            }
        }
    }
    (m, g, goff)
}

fn local_raw<V, Op>(op: &mut Op, template: &V) -> Mat<V::Scalar>
where
    V: InnerProductVector,
    Op: FnMut(&V) -> V,
{
    let n = template.num_coeffs();
    let mut a = Mat::<V::Scalar>::zeros(n, n);
    for j in 0..n {
        let mut e = template.clone();
        e.for_each_coeff_mut(|c| *c = V::Scalar::zero());
        e.set_coeff(j, V::Scalar::one());
        let r = op(&e);
        for i in 0..n {
            *a.get_mut(i, j) = r.coeff(i);
        }
    }
    a
}

fn whiten<T: Scalar>(m: &Mat<T>, g: &[f64]) -> Mat<T> {
    let n = g.len();
    let mut b = Mat::<T>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            *b.get_mut(i, j) = *m.get(i, j) * T::from_f64(1.0 / (g[i] * g[j]).sqrt());
        }
    }
    b
}

fn herm_dev<T: Scalar>(a: &Mat<T>) -> f64 {
    let mut dev = 0.0f64;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            dev = dev.max((*a.get(i, j) - a.get(j, i).conj_s()).abs_s());
        }
    }
    dev
}

fn oracle_vector<V: InnerProductVector>(
    template: &V,
    y: &Mat<V::Scalar>,
    col: usize,
    g: &[f64],
) -> V {
    let mut q = template.clone();
    q.for_each_coeff_mut(|c| *c = V::Scalar::zero());
    for i in 0..g.len() {
        q.set_coeff(i, *y.get(i, col) * V::Scalar::from_f64(1.0 / g[i].sqrt()));
    }
    q
}

fn randn_block<V: InnerProductVector>(template: &V, rng: &mut Xoshiro256PlusPlus) -> V {
    let mut b = template.clone();
    b.for_each_coeff_mut(|c| *c = V::Scalar::sample_standard(rng));
    b
}

fn gram_dev<V: InnerProductVector>(us: &[V]) -> f64 {
    let mut dev = 0.0f64;
    for (i, ui) in us.iter().enumerate() {
        for (j, uj) in us.iter().enumerate() {
            let want = if i == j {
                V::Scalar::one()
            } else {
                V::Scalar::zero()
            };
            dev = dev.max((ui.dot(uj) - want).abs_s());
        }
    }
    dev
}

fn fresh_residual<V, Op>(op: &mut Op, u: &V, theta: f64) -> f64
where
    V: InnerProductVector,
    Op: FnMut(&V) -> V,
{
    let mut r = op(u);
    r.axpy(V::Scalar::from_f64(-theta), u);
    r.norm()
}

fn count_theta_cw<T: Scalar>(x: &Su2TtVector<T>) -> (usize, usize) {
    let space = x.space;
    let d = space.d;
    let mut n_theta = 0usize;
    let mut n_cw = 0usize;
    for b in 0..d {
        for m in space.bond_sectors(b) {
            if x.rank(b, m) == 0 {
                continue;
            }
            for succ in space.bond_sectors(b + 1) {
                if x.rank(b + 1, succ) == 0 {
                    continue;
                }
                if (qntt::su2::theta(m.two_s, succ.two_s) - 1.0).abs() > 1e-12 {
                    n_theta += 1;
                }
            }
        }
    }
    for b in 0..=d {
        for s in space.bond_sectors(b) {
            if x.rank(b, s) == 0 {
                continue;
            }
            if (qntt::su2::centerweight(s.two_s, space.two_s) - 1.0).abs() > 1e-12 {
                n_cw += 1;
            }
        }
    }
    (n_theta, n_cw)
}

fn assert_converged_block(info: &BlockDavidsonInfo, nroots: usize, tol: f64, tag: &str) {
    assert!(
        info.converged,
        "{tag}: block Davidson did not converge (nconv = {}, niter = {}, numops = {}, normres = {:?})",
        info.nconv, info.niter, info.numops, info.normres
    );
    assert_eq!(info.nconv, nroots, "{tag}: nconv");
    assert_eq!(info.normres.len(), nroots, "{tag}: normres length");
    for (i, r) in info.normres.iter().enumerate() {
        assert!(
            *r <= tol,
            "{tag}: root {i} residual {r:.3e} above tol {tol:.3e}"
        );
    }
}

fn assert_ascending(thetas: &[f64], tag: &str) {
    for i in 1..thetas.len() {
        assert!(
            thetas[i] >= thetas[i - 1],
            "{tag}: θ not ascending at {i}: {:?}",
            thetas
        );
    }
}

fn assert_exclusion_gap(ev: &[f64], nroots: usize, floor: f64, tag: &str) {
    assert!(
        ev.len() > nroots,
        "{tag}: local dimension {} has no level above root {}",
        ev.len(),
        nroots - 1
    );
    let gap = ev[nroots] - ev[nroots - 1];
    assert!(
        gap >= floor,
        "{tag}: exclusion gap E[{nroots}] − E[{}] = {gap:.4e} below the fixture floor {floor:.4e} \
         — this fixture no longer resolves {nroots} roots; change the fixture, never the floor",
        nroots - 1
    );
}

struct D1 {
    space: QnSpace,
    row_ranks: RankGrid,
    col_ranks: RankGrid,
    hd: Mat<f64>,
    s0: Sector,
    r: usize,
}

fn d1_fixture(r: usize) -> D1 {
    let space = QnSpace::new(1, 0, 0);
    let s0 = Sector::new(0, 0);
    let mut row_ranks = RankGrid::zeros(&space);
    row_ranks[s0] = r;
    let mut col_ranks = RankGrid::zeros(&space);
    col_ranks[s0] = 1;
    let mut hd = Mat::<f64>::zeros(r, r);
    for i in 0..r {
        *hd.get_mut(i, i) = 1.0 + i as f64;
    }
    for i in 0..r {
        for j in (i + 1)..r {
            let val = 0.05 / (1.0 + (i as f64 - j as f64).abs());
            *hd.get_mut(i, j) = val;
            *hd.get_mut(j, i) = val;
        }
    }
    D1 {
        space,
        row_ranks,
        col_ranks,
        hd,
        s0,
        r,
    }
}

impl D1 {
    fn zeros(&self) -> SparseCore<f64> {
        SparseCore::<f64>::zeros(self.space, 0, &self.row_ranks, &self.col_ranks)
    }

    fn diagonal(&self) -> SparseCore<f64> {
        let mut d = self.zeros();
        {
            let mut b = d.block_mut(Family::Empty, self.s0).unwrap();
            for i in 0..self.r {
                *b.rb_mut().get_mut(i, 0) = *self.hd.get(i, i);
            }
        }
        d
    }

    fn apply(&self, a: &SparseCore<f64>) -> SparseCore<f64> {
        let ab = a.block(Family::Empty, self.s0).unwrap();
        let mut out = self.zeros();
        {
            let mut ob = out.block_mut(Family::Empty, self.s0).unwrap();
            for i in 0..self.r {
                let mut acc = 0.0f64;
                for j in 0..self.r {
                    acc += *self.hd.get(i, j) * *ab.get(j, 0);
                }
                *ob.rb_mut().get_mut(i, 0) = acc;
            }
        }
        out
    }

    fn randn(&self, seed: u64) -> SparseCore<f64> {
        let mut rng = seeded(seed);
        let mut v0 = self.zeros();
        v0.fill_randn(&mut rng);
        v0
    }

    fn min_eig(&self) -> f64 {
        dense_self_adjoint_eigen(self.hd.as_ref()).0[0]
    }
}

fn assert_parity(
    fx: &D1,
    seed: u64,
    opts: &DavidsonOpts,
    expect_restart: bool,
    tag: &str,
) -> usize {
    let d_core = fx.diagonal();
    let v0 = fx.randn(seed);

    let mut op1 = |a: &SparseCore<f64>| fx.apply(a);
    let mut prec1 = |t: &mut SparseCore<f64>, theta: f64| precond(t, &d_core, theta);
    let (lambda, u1, info1) = davidson(&mut op1, &v0, &mut prec1, opts);

    let bopts = BlockDavidsonOpts {
        tol: opts.tol,
        maxsubspace: opts.maxsubspace,
        maxiter: opts.maxiter,
        nwarm: opts.nwarm,
        theta_tol: opts.theta_tol,
        restart_keep: opts.restart_keep,
        verbosity: opts.verbosity,
    };
    let mut op2 = |a: &SparseCore<f64>| fx.apply(a);
    let mut prec2 = |t: &mut SparseCore<f64>, theta: f64| precond(t, &d_core, theta);
    let (thetas, us, info2) =
        davidson_block(&mut op2, std::slice::from_ref(&v0), &mut prec2, 1, &bopts);

    assert!(
        (lambda - fx.min_eig()).abs() < 1e-6,
        "{tag}: the reference davidson run itself missed the smallest eigenvalue ({lambda} vs {})",
        fx.min_eig()
    );
    assert!(
        info1.converged,
        "{tag}: reference davidson did not converge"
    );

    assert_eq!(thetas.len(), 1, "{tag}: root count");
    assert_eq!(us.len(), 1, "{tag}: vector count");
    assert_eq!(info2.numops, info1.numops, "{tag}: numops");
    assert_eq!(info2.niter, info1.niter, "{tag}: niter");
    assert_eq!(info2.converged, info1.converged, "{tag}: converged");
    assert_eq!(info2.nconv, 1, "{tag}: nconv");
    assert_eq!(info2.dropped_guesses, 0, "{tag}: dropped_guesses");
    assert_eq!(
        thetas[0].to_bits(),
        lambda.to_bits(),
        "{tag}: θ {} vs λ {lambda} not bitwise equal",
        thetas[0]
    );
    assert_eq!(
        info2.normres[0].to_bits(),
        info1.normres.to_bits(),
        "{tag}: normres {} vs {} not bitwise equal",
        info2.normres[0],
        info1.normres
    );
    assert_eq!(
        us[0].num_coeffs(),
        u1.num_coeffs(),
        "{tag}: coefficient count"
    );
    for i in 0..u1.num_coeffs() {
        assert_eq!(
            us[0].coeff(i).to_bits(),
            u1.coeff(i).to_bits(),
            "{tag}: coefficient {i} ({} vs {}) not bitwise equal",
            us[0].coeff(i),
            u1.coeff(i)
        );
    }
    if expect_restart {
        assert!(
            info2.numops > opts.maxsubspace,
            "{tag}: numops {} ≤ maxsubspace {} — no restart was exercised by this arm",
            info2.numops,
            opts.maxsubspace
        );
    }
    info2.numops
}

#[test]
fn tms_a1_nroots1_bitwise_parity_with_davidson() {
    let fx = d1_fixture(8);

    let b1 = BlockDavidsonOpts::for_roots(1);
    let d1 = DavidsonOpts::default();
    assert_eq!(b1.tol, d1.tol, "for_roots(1).tol");
    assert_eq!(b1.maxsubspace, d1.maxsubspace, "for_roots(1).maxsubspace");
    assert_eq!(b1.maxiter, d1.maxiter, "for_roots(1).maxiter");
    assert_eq!(b1.nwarm, d1.nwarm, "for_roots(1).nwarm");
    assert_eq!(
        b1.restart_keep, d1.restart_keep,
        "for_roots(1).restart_keep"
    );
    assert_eq!(b1.verbosity, d1.verbosity, "for_roots(1).verbosity");
    assert!(b1.theta_tol.is_none(), "for_roots(1).theta_tol");

    let base = DavidsonOpts {
        tol: 1e-8,
        maxsubspace: 8,
        maxiter: 100,
        nwarm: 2,
        restart_keep: 1,
        verbosity: 0,
        theta_tol: None,
        stop_below: None,
    };

    assert_parity(&fx, 0x5A09, &base, false, "a1a default");

    let restarting = DavidsonOpts {
        maxsubspace: 4,
        maxiter: 300,
        ..base
    };
    let n_keep1 = assert_parity(&fx, 0x5A0A, &restarting, true, "a1b restart keep=1");

    let keep2 = DavidsonOpts {
        restart_keep: 2,
        ..restarting
    };
    let n_keep2 = assert_parity(&fx, 0x5A0A, &keep2, true, "a1c restart keep=2");

    let theta_exit = DavidsonOpts {
        theta_tol: Some(1e-5),
        ..base
    };
    assert_parity(&fx, 0x5A09, &theta_exit, false, "a1d theta_tol");

    println!("tms_a1: numops keep1={n_keep1} keep2={n_keep2} (maxsubspace = 4, r = 8)");
}

#[test]
fn tms_a1_batched_entry_is_bitwise_the_serial_entry() {
    use rayon::prelude::*;
    let fx = d1_fixture(8);
    let d_core = fx.diagonal();
    let seeds: Vec<SparseCore<f64>> = [0x5A11u64, 0x5A12, 0x5A13]
        .iter()
        .map(|s| fx.randn(*s))
        .collect();
    let nroots = 3usize;
    let base = BlockDavidsonOpts {
        tol: 1e-8,
        maxsubspace: 12,
        maxiter: 300,
        nwarm: 2,
        theta_tol: None,
        restart_keep: 3,
        verbosity: 0,
    };
    let arms = [
        ("no restart", base, false),
        (
            "forced restart",
            BlockDavidsonOpts {
                maxsubspace: 6,
                ..base
            },
            true,
        ),
    ];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("4-thread pool");
    for (tag, opts, expect_restart) in arms {
        let mut op1 = |a: &SparseCore<f64>| fx.apply(a);
        let mut prec1 = |t: &mut SparseCore<f64>, theta: f64| precond(t, &d_core, theta);
        let mut proj1 = |_t: &mut SparseCore<f64>| {};
        let (t1, u1, i1) =
            davidson_block_projected(&mut op1, &seeds, &mut prec1, &mut proj1, nroots, &opts);

        let mut forks = 0usize;
        let (t2, u2, i2) = pool.install(|| {
            let mut op_batch = |vs: &[SparseCore<f64>]| -> Vec<SparseCore<f64>> {
                forks += 1;
                vs.par_iter().map(|v| fx.apply(v)).collect()
            };
            let mut prec2 = |t: &mut SparseCore<f64>, theta: f64| precond(t, &d_core, theta);
            let mut proj2 = |_t: &mut SparseCore<f64>| {};
            davidson_block_projected_batched(
                &mut op_batch,
                &seeds,
                &mut prec2,
                &mut proj2,
                nroots,
                &opts,
            )
        });

        assert!(
            i1.converged,
            "{tag}: the reference run itself did not converge — the parity would be vacuous"
        );
        assert!(
            forks >= 2,
            "{tag}: only {forks} batches — the batched path was not exercised"
        );
        if expect_restart {
            assert!(
                i2.numops > opts.maxsubspace,
                "{tag}: numops {} ≤ maxsubspace {} — no restart in this arm",
                i2.numops,
                opts.maxsubspace
            );
        }
        assert_eq!(i2.numops, i1.numops, "{tag}: numops");
        assert_eq!(i2.niter, i1.niter, "{tag}: niter");
        assert_eq!(i2.converged, i1.converged, "{tag}: converged");
        assert_eq!(i2.nconv, i1.nconv, "{tag}: nconv");
        assert_eq!(
            i2.dropped_guesses, i1.dropped_guesses,
            "{tag}: dropped_guesses"
        );
        assert_eq!(t2.len(), t1.len(), "{tag}: root count");
        for i in 0..t1.len() {
            assert_eq!(
                t2[i].to_bits(),
                t1[i].to_bits(),
                "{tag}: θ[{i}] {} vs {} not bitwise equal",
                t2[i],
                t1[i]
            );
            assert_eq!(
                i2.normres[i].to_bits(),
                i1.normres[i].to_bits(),
                "{tag}: normres[{i}] {} vs {} not bitwise equal",
                i2.normres[i],
                i1.normres[i]
            );
            for c in 0..u1[i].num_coeffs() {
                assert_eq!(
                    u2[i].coeff(c).to_bits(),
                    u1[i].coeff(c).to_bits(),
                    "{tag}: root {i} coefficient {c} not bitwise equal"
                );
            }
        }
        println!(
            "tms_a1 batched [{tag}]: {} applies in {} iterations, {forks} batches, θ = {:?}",
            i2.numops, i2.niter, t2
        );
    }
}

#[test]
fn tms_a2_block_matches_dense_local_spectrum() {
    let d = 6usize;
    let (t, v) = generic_truncated_tv::<f64>(d);

    {
        let space = QnSpace::new(d, 1, 1);
        let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
        let (_x, fl, fr, cp0, dd) = abelian_local(space, &h, 2, 0xB10C_0A02);
        let n = cp0.num_coeffs();
        assert_eq!(n, 30, "a2A: local dimension");

        let mut op =
            |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
        let (m, g, goff) = local_dense(&mut op, &cp0);
        assert!(
            goff <= 1e-14,
            "a2A: abelian coefficient Gram off-diagonal {goff:.3e}"
        );
        for (i, gi) in g.iter().enumerate() {
            assert!(
                (gi - 1.0).abs() <= 1e-14,
                "a2A: abelian coefficient Gram diagonal {i} = {gi}"
            );
        }
        assert!(
            herm_dev(&m) <= 1e-12,
            "a2A: local operator not self-adjoint in the package metric ({:.3e})",
            herm_dev(&m)
        );
        let (ev, _) = dense_self_adjoint_eigen(m.as_ref());
        assert_exclusion_gap(&ev, 4, 0.05, "a2A");

        let mut rng = seeded(0xA2A_0001);
        let guesses = vec![
            cp0.clone(),
            randn_block(&cp0, &mut rng),
            randn_block(&cp0, &mut rng),
            randn_block(&cp0, &mut rng),
        ];
        let mut prec = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
        let opts = BlockDavidsonOpts {
            tol: 1e-10,
            ..BlockDavidsonOpts::for_roots(4)
        };
        let (thetas, us, info) = davidson_block(&mut op, &guesses, &mut prec, 4, &opts);

        assert_eq!(thetas.len(), 4, "a2A: root count");
        assert_eq!(info.dropped_guesses, 0, "a2A: dropped_guesses");
        assert_converged_block(&info, 4, opts.tol, "a2A");
        assert_ascending(&thetas, "a2A");
        for i in 0..4 {
            assert!(
                (thetas[i] - ev[i]).abs() <= 1e-10,
                "a2A: θ[{i}] = {} vs dense {} (Δ = {:.3e})",
                thetas[i],
                ev[i],
                (thetas[i] - ev[i]).abs()
            );
            let res = fresh_residual(&mut op, &us[i], thetas[i]);
            assert!(res <= 1e-9, "a2A: fresh residual of root {i} = {res:.3e}");
        }
        let gd = gram_dev(&us);
        assert!(gd <= 1e-12, "a2A: ‖Gram − I‖ = {gd:.3e}");
    }

    {
        let space = QnSpace::new(d, 2, 2);
        let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
        let (_x, fl, fr, cp0, dd) = abelian_local(space, &h, 2, 0xB10C_0A02);
        let dets = dets_for(d, 2, 2);
        let n = cp0.num_coeffs();
        assert_eq!(
            n,
            dets.len(),
            "a2B: the (2,3) pair of the full-rank (6,2,2) grid must host the COMPLETE sector"
        );

        let mut op =
            |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
        let (m, _g, goff) = local_dense(&mut op, &cp0);
        assert!(
            goff <= 1e-14,
            "a2B: coefficient Gram off-diagonal {goff:.3e}"
        );
        let (ev, _) = dense_self_adjoint_eigen(m.as_ref());
        assert_exclusion_gap(&ev, 3, 0.05, "a2B");

        let (sev, _) = dense_eigen(&dense_hamiltonian_matrix(d, &t, &v, &dets));
        for i in 0..3 {
            assert!(
                (ev[i] - sev[i]).abs() <= 1e-10,
                "a2B: the local oracle disagrees with the dense sector spectrum at {i}: {} vs {}",
                ev[i],
                sev[i]
            );
        }

        let mut rng = seeded(0xA2B_0001);
        let guesses = vec![
            cp0.clone(),
            randn_block(&cp0, &mut rng),
            randn_block(&cp0, &mut rng),
        ];
        let mut prec = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
        let opts = BlockDavidsonOpts {
            tol: 1e-10,
            ..BlockDavidsonOpts::for_roots(3)
        };
        let (thetas, us, info) = davidson_block(&mut op, &guesses, &mut prec, 3, &opts);

        assert_converged_block(&info, 3, opts.tol, "a2B");
        assert_ascending(&thetas, "a2B");
        for i in 0..3 {
            assert!(
                (thetas[i] - sev[i]).abs() <= 1e-10,
                "a2B: θ[{i}] = {} vs the dense SECTOR eigenvalue {} (Δ = {:.3e})",
                thetas[i],
                sev[i],
                (thetas[i] - sev[i]).abs()
            );
            let res = fresh_residual(&mut op, &us[i], thetas[i]);
            assert!(res <= 1e-9, "a2B: fresh residual of root {i} = {res:.3e}");
        }
        let gd = gram_dev(&us);
        assert!(gd <= 1e-12, "a2B: ‖Gram − I‖ = {gd:.3e}");
    }
}

#[test]
fn tms_a2su2_block_matches_dense_local_generalized_spectrum() {
    let d = 6usize;
    let (t, v) = generic_truncated_tv::<f64>(d);
    for (two_s, seed, want_n) in [(0usize, 0x50_2A02u64, 17usize), (2, 0x50_2A02, 15)] {
        let tag = format!("a2su2 2S={two_s}");
        let space = Su2Space::new(d, 2, two_s);
        let h = Su2SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
        let (x, fl, fr, cp0, dd) = su2_local(space, &h, 2, seed);
        let n = cp0.num_coeffs();
        assert_eq!(n, want_n, "{tag}: local dimension");

        let mut op = |cp: &Su2ContractedCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
        let (m, g, goff) = local_dense(&mut op, &cp0);

        assert!(
            goff <= 1e-14,
            "{tag}: Gram off-diagonal {goff:.3e} — expected diagonal"
        );
        assert!(
            g.iter().all(|gi| *gi > 0.0),
            "{tag}: non-positive Gram diagonal"
        );
        let n_weighted = g.iter().filter(|gi| (**gi - 1.0).abs() > 1e-12).count();
        assert!(
            n_weighted >= 2,
            "{tag}: only {n_weighted} centerweighted coefficient directions — the weighted path is \
             not live at this size"
        );

        let a_raw = local_raw(&mut op, &cp0);
        assert!(
            herm_dev(&a_raw) > 1e-3,
            "{tag}: the raw coefficient matrix came back self-adjoint ({:.3e}) — the centerweighted \
             two-site metric is not being exercised",
            herm_dev(&a_raw)
        );
        assert!(
            herm_dev(&m) <= 1e-12,
            "{tag}: ⟨e_i, op e_j⟩ not self-adjoint ({:.3e})",
            herm_dev(&m)
        );
        let b = whiten(&m, &g);
        assert!(
            herm_dev(&b) <= 1e-12,
            "{tag}: whitened operator not self-adjoint ({:.3e})",
            herm_dev(&b)
        );
        let (ev, _y) = dense_self_adjoint_eigen(b.as_ref());
        assert_exclusion_gap(&ev, 3, 0.05, &tag);

        let mut rng = seeded(0xA25_0001 + two_s as u64);
        let guesses = vec![
            cp0.clone(),
            randn_block(&cp0, &mut rng),
            randn_block(&cp0, &mut rng),
        ];
        let mut prec = |t: &mut Su2ContractedCores<f64>, theta: f64| precond(t, &dd, theta);
        let opts = BlockDavidsonOpts {
            tol: 1e-10,
            ..BlockDavidsonOpts::for_roots(3)
        };
        let (thetas, us, info) = davidson_block(&mut op, &guesses, &mut prec, 3, &opts);

        assert_converged_block(&info, 3, opts.tol, &tag);
        assert_ascending(&thetas, &tag);
        for i in 0..3 {
            assert!(
                (thetas[i] - ev[i]).abs() <= 1e-10,
                "{tag}: θ[{i}] = {} vs the generalized-pencil eigenvalue {} (Δ = {:.3e})",
                thetas[i],
                ev[i],
                (thetas[i] - ev[i]).abs()
            );
            let res = fresh_residual(&mut op, &us[i], thetas[i]);
            assert!(res <= 1e-9, "{tag}: fresh residual of root {i} = {res:.3e}");
        }
        let gd = gram_dev(&us);
        assert!(gd <= 1e-12, "{tag}: ‖Gram − I‖ (package dot) = {gd:.3e}");

        if two_s == 2 {
            let (n_theta, n_cw) = count_theta_cw(&x);
            assert!(
                n_theta >= 2 && n_cw >= 2,
                "{tag}: weighted paths not live (n_theta = {n_theta}, n_cw = {n_cw})"
            );
        }
    }
}

#[test]
fn tms_a3_identical_guesses_dropped_then_converge() {
    let d = 6usize;
    let (t, v) = generic_truncated_tv::<f64>(d);
    let space = QnSpace::new(d, 1, 1);
    let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let (_x, fl, fr, cp0, dd) = abelian_local(space, &h, 2, 0xB10C_0A02);
    let mut op = |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
    let (m, _g, _goff) = local_dense(&mut op, &cp0);
    let (ev, _) = dense_self_adjoint_eigen(m.as_ref());

    for nroots in [3usize, 4] {
        let tag = format!("a3 nroots={nroots}");
        assert_exclusion_gap(&ev, nroots, 0.05, &tag);
        let guesses = vec![cp0.clone(); nroots];
        let mut prec = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
        let opts = BlockDavidsonOpts {
            tol: 1e-10,
            ..BlockDavidsonOpts::for_roots(nroots)
        };
        let (thetas, us, info) = davidson_block(&mut op, &guesses, &mut prec, nroots, &opts);

        assert_eq!(
            info.dropped_guesses,
            nroots - 1,
            "{tag}: dropped_guesses (identical guesses must be dropped, not orthogonalized into noise)"
        );
        assert_converged_block(&info, nroots, opts.tol, &tag);
        assert_ascending(&thetas, &tag);
        for i in 0..nroots {
            assert!(
                (thetas[i] - ev[i]).abs() <= 1e-10,
                "{tag}: θ[{i}] = {} vs dense {}",
                thetas[i],
                ev[i]
            );
        }
        let gd = gram_dev(&us);
        assert!(gd <= 1e-10, "{tag}: ‖Gram − I‖ = {gd:.3e}");
    }
}

#[test]
fn tms_a4_degenerate_multiplet_roots_and_span() {
    let d = 6usize;
    let (t, v) = degenerate_tv(d);
    let space = QnSpace::new(d, 1, 1);
    let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);

    let mut found: Option<usize> = None;
    let mut census: Vec<(usize, usize, usize)> = Vec::new();
    for k in 0..d - 1 {
        let (_x, fl, fr, cp0, _dd) = abelian_local(space, &h, k, 0xD0_0001);
        let mut op =
            |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
        let (m, _g, _goff) = local_dense(&mut op, &cp0);
        let (ev, _) = dense_self_adjoint_eigen(m.as_ref());
        let deg = ev.iter().filter(|e| (**e + 0.5).abs() < 1e-8).count();
        census.push((k, cp0.num_coeffs(), deg));
        if deg == 2 && found.is_none() {
            found = Some(k);
        }
    }
    let k = found.unwrap_or_else(|| {
        panic!(
            "a4: no interior pair hosts the −0.5 doublet of the two-entry fixture \
             (per-pair (k, dim, multiplicity): {census:?})"
        )
    });
    assert_eq!(
        k, 1,
        "a4: the doublet moved off pair (1,2) — per-pair (k, dim, multiplicity): {census:?}"
    );

    let (_x, fl, fr, cp0, dd) = abelian_local(space, &h, k, 0xD0_0001);
    let mut op = |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
    let (m, g, _goff) = local_dense(&mut op, &cp0);
    let (ev, y) = dense_self_adjoint_eigen(m.as_ref());
    assert!(
        (ev[0] + 0.5).abs() <= 1e-10 && (ev[1] + 0.5).abs() <= 1e-10,
        "a4: dense local doublet is {} / {}",
        ev[0],
        ev[1]
    );
    assert_exclusion_gap(&ev, 2, 0.4, "a4");

    let mut rng = seeded(0xA4_0001);
    let guesses = vec![cp0.clone(), randn_block(&cp0, &mut rng)];
    let mut prec = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
    let opts = BlockDavidsonOpts {
        tol: 1e-10,
        ..BlockDavidsonOpts::for_roots(2)
    };
    let (thetas, us, info) = davidson_block(&mut op, &guesses, &mut prec, 2, &opts);

    assert_converged_block(&info, 2, opts.tol, "a4");
    for i in 0..2 {
        assert!(
            (thetas[i] + 0.5).abs() <= 1e-10,
            "a4: θ[{i}] = {} is not the degenerate eigenvalue −0.5",
            thetas[i]
        );
        let res = fresh_residual(&mut op, &us[i], thetas[i]);
        assert!(res <= 1e-9, "a4: fresh residual of root {i} = {res:.3e}");
    }
    assert!(
        (thetas[0] - thetas[1]).abs() <= 1e-10,
        "a4: the two members split by {:.3e}",
        (thetas[0] - thetas[1]).abs()
    );
    let gd = gram_dev(&us);
    assert!(gd <= 1e-10, "a4: ‖Gram − I‖ = {gd:.3e}");

    let qs: Vec<ContractedSparseCores<f64>> = (0..ev.len())
        .filter(|&c| (ev[c] + 0.5).abs() < 1e-8)
        .map(|c| oracle_vector(&cp0, &y, c, &g))
        .collect();
    assert_eq!(qs.len(), 2, "a4: analytic eigenspace dimension");
    for (i, u) in us.iter().enumerate() {
        let proj: f64 = qs.iter().map(|q| q.dot(u).abs_s().powi(2)).sum();
        assert!(
            (proj - 1.0).abs() <= 1e-8,
            "a4: root {i} projects onto the −0.5 eigenspace with weight {proj} (want 1)"
        );
    }
}

#[test]
fn tms_a5_forced_restart_converges() {
    let d = 6usize;
    let (t, v) = generic_truncated_tv::<f64>(d);
    let space = QnSpace::new(d, 1, 1);
    let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let (_x, fl, fr, cp0, dd) = abelian_local(space, &h, 2, 0xB10C_0A02);
    let mut op = |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
    let (m, _g, _goff) = local_dense(&mut op, &cp0);
    let (ev, _) = dense_self_adjoint_eigen(m.as_ref());
    let nroots = 3usize;
    assert_exclusion_gap(&ev, nroots, 0.05, "a5");

    let mut rng = seeded(0xA5_0001);
    let guesses = vec![
        cp0.clone(),
        randn_block(&cp0, &mut rng),
        randn_block(&cp0, &mut rng),
    ];

    let tight = BlockDavidsonOpts {
        tol: 1e-10,
        maxsubspace: 2 * nroots,
        restart_keep: nroots,
        maxiter: 400,
        ..BlockDavidsonOpts::for_roots(nroots)
    };
    let mut prec = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
    let (thetas, us, info) = davidson_block(&mut op, &guesses, &mut prec, nroots, &tight);
    assert_converged_block(&info, nroots, tight.tol, "a5 restarted");
    assert_ascending(&thetas, "a5 restarted");
    for i in 0..nroots {
        assert!(
            (thetas[i] - ev[i]).abs() <= 1e-10,
            "a5: θ[{i}] = {} vs dense {}",
            thetas[i],
            ev[i]
        );
    }
    let gd = gram_dev(&us);
    assert!(gd <= 1e-10, "a5: ‖Gram − I‖ = {gd:.3e}");
    assert!(
        info.numops > tight.maxsubspace,
        "a5: numops {} ≤ maxsubspace {} — no restart was exercised",
        info.numops,
        tight.maxsubspace
    );

    let roomy = BlockDavidsonOpts {
        tol: 1e-10,
        maxsubspace: 40,
        restart_keep: nroots,
        maxiter: 400,
        ..BlockDavidsonOpts::for_roots(nroots)
    };
    let mut prec2 = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
    let (thetas2, _us2, info2) = davidson_block(&mut op, &guesses, &mut prec2, nroots, &roomy);
    assert_converged_block(&info2, nroots, roomy.tol, "a5 unrestarted");
    for i in 0..nroots {
        assert!(
            (thetas2[i] - thetas[i]).abs() <= 1e-9,
            "a5: restarted and unrestarted runs disagree at root {i}: {} vs {}",
            thetas[i],
            thetas2[i]
        );
    }
    println!(
        "tms_a5: restarted niter={} numops={}; unrestarted niter={} numops={} (maxiter budget {})",
        info.niter, info.numops, info2.niter, info2.numops, tight.maxiter
    );
    assert!(
        info.niter < tight.maxiter,
        "a5: the restarted run used its whole {} iteration budget — the 3× headroom is gone",
        tight.maxiter
    );
}

fn a6_roots() -> (ContractedSparseCores<f64>, Vec<ContractedSparseCores<f64>>) {
    let d = 6usize;
    let (t, v) = generic_truncated_tv::<f64>(d);
    let space = QnSpace::new(d, 1, 1);
    let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let (_x, fl, fr, cp0, dd) = abelian_local(space, &h, 2, 0xB10C_0A02);
    let mut op = |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
    let mut rng = seeded(0xA6_0001);
    let guesses = vec![
        cp0.clone(),
        randn_block(&cp0, &mut rng),
        randn_block(&cp0, &mut rng),
        randn_block(&cp0, &mut rng),
    ];
    let mut prec = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
    let opts = BlockDavidsonOpts {
        tol: 1e-10,
        ..BlockDavidsonOpts::for_roots(4)
    };
    let (_thetas, us, info) = davidson_block(&mut op, &guesses, &mut prec, 4, &opts);
    assert_converged_block(&info, 4, opts.tol, "a6 setup");
    (cp0, us)
}

#[test]
fn tms_a6_home_roots_recovers_permutation() {
    let (_cp0, us) = a6_roots();

    let prev: Vec<ContractedSparseCores<f64>> =
        vec![us[2].clone(), us[0].clone(), us[3].clone(), us[1].clone()];
    let (perm, gram) = home_roots(&prev, &us);
    assert_eq!(perm, vec![2, 0, 3, 1], "a6: recovered permutation");
    assert_eq!(gram.len(), prev.len(), "a6: Gram rows = prev");
    for row in &gram {
        assert_eq!(row.len(), us.len(), "a6: Gram columns = us");
    }
    for (i, row) in gram.iter().enumerate() {
        for (j, gij) in row.iter().enumerate() {
            assert_eq!(
                gij.to_bits(),
                prev[i].dot(&us[j]).to_bits(),
                "a6: Gram[{i}][{j}] is not ⟨prev_i, us_j⟩"
            );
        }
        assert!(
            (row[perm[i]].abs_s() - 1.0).abs() <= 1e-10,
            "a6: |⟨prev_{i}, us_{}⟩| = {}",
            perm[i],
            row[perm[i]].abs_s()
        );
        for (j, gij) in row.iter().enumerate() {
            if j != perm[i] {
                assert!(
                    gij.abs_s() <= 1e-10,
                    "a6: off-assignment overlap ⟨prev_{i}, us_{j}⟩ = {}",
                    gij.abs_s()
                );
            }
        }
    }

    let mut p0 = us[2].clone();
    p0.axpy(0.1, &us[3]);
    let n0 = p0.norm();
    p0.scale(1.0 / n0);
    let mut p1 = us[0].clone();
    p1.axpy(-0.2, &us[1]);
    let n1 = p1.norm();
    p1.scale(1.0 / n1);
    let (perm2, gram2) = home_roots(&[p0, p1], &us);
    assert_eq!(perm2, vec![2, 0], "a6: mixture assignment");
    assert_eq!(gram2.len(), 2, "a6: mixture Gram rows");

    let (perm3, gram3) = home_roots(std::slice::from_ref(&us[1]), &us);
    assert_eq!(perm3, vec![1], "a6: single-previous assignment");
    assert_eq!(gram3.len(), 1, "a6: single-previous Gram rows");
    assert_eq!(gram3[0].len(), 4, "a6: single-previous Gram columns");
}

#[test]
#[should_panic(expected = "home_roots: need")]
fn tms_a6_home_roots_rejects_short_us() {
    let (_cp0, us) = a6_roots();
    let short: Vec<ContractedSparseCores<f64>> = us[..2].to_vec();
    let _ = home_roots(&us, &short);
}

#[test]
#[should_panic(expected = "home_roots:")]
fn tms_a6_home_roots_rejects_empty_prev() {
    let (_cp0, us) = a6_roots();
    let _ = home_roots(&[], &us);
}

#[test]
fn tms_a7_projected_variant_solves_on_q_perp() {
    let d = 6usize;
    let (t, v) = generic_truncated_tv::<f64>(d);
    let space = QnSpace::new(d, 1, 1);
    let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let (_x, fl, fr, cp0, dd) = abelian_local(space, &h, 2, 0xB10C_0A02);
    let mut op = |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
    let n = cp0.num_coeffs();
    let (m, g, _goff) = local_dense(&mut op, &cp0);
    let (ev, y) = dense_self_adjoint_eigen(m.as_ref());

    let eps = 0.1f64;
    let q0 = oracle_vector(&cp0, &y, 0, &g);
    let q1 = oracle_vector(&cp0, &y, 1, &g);
    let mut q = q0.clone();
    q.axpy(eps, &q1);
    let qn = q.norm();
    q.scale(1.0 / qn);
    let theta_q = q.dot(&op(&q));
    let rq = fresh_residual(&mut op, &q, theta_q);
    assert!(
        rq > 1e-3,
        "a7: the projected-out state is too close to an exact eigenvector (residual {rq:.3e}); \
         enlarge the admixture — never lower the anti-oracle threshold"
    );

    let qc: Vec<f64> = (0..n).map(|i| q.coeff(i)).collect();
    let sigma = ev[n - 1] + 1.0;
    let mut pm = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = sigma * qc[i] * qc[j];
            for a in 0..n {
                for b in 0..n {
                    let pia = if i == a { 1.0 } else { 0.0 } - qc[i] * qc[a];
                    let pbj = if b == j { 1.0 } else { 0.0 } - qc[b] * qc[j];
                    acc += pia * *m.get(a, b) * pbj;
                }
            }
            *pm.get_mut(i, j) = acc;
        }
    }
    let (pall, _pu) = dense_self_adjoint_eigen(pm.as_ref());
    assert!(
        (pall[n - 1] - sigma).abs() <= 1e-8,
        "a7: the shifted spurious direction is at {} , expected σ = {sigma}",
        pall[n - 1]
    );
    let perp: Vec<f64> = pall[..n - 1].to_vec();
    assert_exclusion_gap(&perp, 2, 0.05, "a7");

    let mut project = |b: &mut ContractedSparseCores<f64>| {
        let c = q.dot(b);
        b.axpy(-c, &q);
    };
    let mut rng = seeded(0xA7_0001);
    let guesses = vec![randn_block(&cp0, &mut rng), randn_block(&cp0, &mut rng)];
    let mut prec = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
    let opts = BlockDavidsonOpts {
        tol: 1e-10,
        ..BlockDavidsonOpts::for_roots(2)
    };
    let (thetas, us, info) =
        davidson_block_projected(&mut op, &guesses, &mut prec, &mut project, 2, &opts);

    assert_converged_block(&info, 2, opts.tol, "a7");
    assert!(
        info.niter < opts.maxiter,
        "a7: exhausted the iteration budget"
    );
    for i in 0..2 {
        assert!(
            (thetas[i] - perp[i]).abs() <= 1e-10,
            "a7: θ[{i}] = {} vs the dense q⊥ spectrum {} (Δ = {:.3e})",
            thetas[i],
            perp[i],
            (thetas[i] - perp[i]).abs()
        );
        assert!(
            q.dot(&us[i]).abs_s() <= 1e-12,
            "a7: root {i} is not orthogonal to the projected-out state (|⟨q,u⟩| = {})",
            q.dot(&us[i]).abs_s()
        );
    }
    let gd = gram_dev(&us);
    assert!(gd <= 1e-10, "a7: ‖Gram − I‖ = {gd:.3e}");

    let full = fresh_residual(&mut op, &us[0], thetas[0]);
    assert!(
        full > 1e-4,
        "a7: the full residual of the projected root is {full:.3e} — the projected-out state must \
         floor it (analytic floor ε(E₁−E₀)/(1+ε²) ≈ {:.3e})",
        eps * (ev[1] - ev[0]) / (1.0 + eps * eps)
    );
    assert!(
        (thetas[0] - ev[1]).abs() >= 1e-6,
        "a7: the projected ground coincides with E₁ — the projector is not biting"
    );
}

#[test]
fn tms_a8_c64_block_matches_dense_local_spectrum() {
    let d = 6usize;
    let (t, v) = generic_truncated_tv::<f64>(d);
    let theta: Vec<f64> = (0..d).map(|p| 0.3 + 0.4 * p as f64).collect();
    let (tc, vc) = gauge_integrals(d, &t, &v, &theta);
    let space = QnSpace::new(d, 2, 2);
    let h = SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);
    let (_x, fl, fr, cp0, dd) = abelian_local(space, &h, 2, 0xC64_0A08);
    let dets = dets_for(d, 2, 2);
    let n = cp0.num_coeffs();
    assert_eq!(
        n,
        dets.len(),
        "a8: the (2,3) pair must host the COMPLETE sector"
    );

    let mut op = |cp: &ContractedSparseCores<C64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
    let (m, _g, goff) = local_dense(&mut op, &cp0);
    assert!(
        goff <= 1e-14,
        "a8: coefficient Gram off-diagonal {goff:.3e}"
    );
    assert!(
        herm_dev(&m) <= 1e-12,
        "a8: local operator not self-adjoint ({:.3e})",
        herm_dev(&m)
    );
    let mut max_im = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            max_im = max_im.max(m.get(i, j).im_s().abs());
        }
    }
    assert!(
        max_im > 1e-3,
        "a8: the local matrix is real (max|Im| = {max_im:.3e})"
    );

    let (ev, _) = dense_self_adjoint_eigen(m.as_ref());
    assert_exclusion_gap(&ev, 3, 0.05, "a8");
    let (sev, _) = dense_eigen(&dense_hamiltonian_matrix(d, &t, &v, &dets));
    for i in 0..3 {
        assert!(
            (ev[i] - sev[i]).abs() <= 1e-10,
            "a8: gauge covariance broken at level {i}: {} vs the real spectrum {}",
            ev[i],
            sev[i]
        );
    }

    let mut rng = seeded(0xA8_0001);
    let guesses = vec![
        cp0.clone(),
        randn_block(&cp0, &mut rng),
        randn_block(&cp0, &mut rng),
    ];
    let mut prec = |t: &mut ContractedSparseCores<C64>, theta: f64| precond(t, &dd, theta);
    let opts = BlockDavidsonOpts {
        tol: 1e-10,
        ..BlockDavidsonOpts::for_roots(3)
    };
    let (thetas, us, info) = davidson_block(&mut op, &guesses, &mut prec, 3, &opts);

    assert_converged_block(&info, 3, opts.tol, "a8");
    assert_ascending(&thetas, "a8");
    for i in 0..3 {
        assert!(
            (thetas[i] - sev[i]).abs() <= 1e-10,
            "a8: θ[{i}] = {} vs the real dense sector eigenvalue {}",
            thetas[i],
            sev[i]
        );
        let res = fresh_residual(&mut op, &us[i], thetas[i]);
        assert!(res <= 1e-9, "a8: fresh residual of root {i} = {res:.3e}");
    }
    let gd = gram_dev(&us);
    assert!(gd <= 1e-12, "a8: ‖Gram − I‖ (complex dot) = {gd:.3e}");
}

fn a9_call(r: usize, nguesses: usize, zero_guesses: bool, nroots: usize, opts: &BlockDavidsonOpts) {
    let fx = d1_fixture(r);
    let d_core = fx.diagonal();
    let mut guesses: Vec<SparseCore<f64>> = Vec::new();
    for i in 0..nguesses {
        let mut g = fx.randn(0xA9_0000 + i as u64);
        if zero_guesses {
            g.scale(0.0);
        }
        guesses.push(g);
    }
    let mut op = |a: &SparseCore<f64>| fx.apply(a);
    let mut prec = |t: &mut SparseCore<f64>, theta: f64| precond(t, &d_core, theta);
    let _ = davidson_block(&mut op, &guesses, &mut prec, nroots, opts);
}

#[test]
#[should_panic(expected = "davidson_block: nroots")]
fn tms_a9_zero_nroots_panics() {
    a9_call(8, 1, false, 0, &BlockDavidsonOpts::for_roots(1));
}

#[test]
#[should_panic(expected = "davidson_block: at least one guess")]
fn tms_a9_empty_guess_list_panics() {
    a9_call(8, 0, false, 1, &BlockDavidsonOpts::for_roots(1));
}

#[test]
#[should_panic(expected = "davidson_block: every guess was dropped")]
fn tms_a9_all_guesses_dropped_panics() {
    a9_call(8, 2, true, 1, &BlockDavidsonOpts::for_roots(1));
}

#[test]
#[should_panic(expected = "davidson_block: maxsubspace")]
fn tms_a9_maxsubspace_below_two_nroots_panics() {
    let opts = BlockDavidsonOpts {
        maxsubspace: 7,
        restart_keep: 4,
        ..BlockDavidsonOpts::for_roots(4)
    };
    a9_call(8, 4, false, 4, &opts);
}

#[test]
#[should_panic(expected = "davidson_block: restart_keep")]
fn tms_a9_restart_keep_below_nroots_panics() {
    let opts = BlockDavidsonOpts {
        maxsubspace: 16,
        restart_keep: 3,
        ..BlockDavidsonOpts::for_roots(4)
    };
    a9_call(8, 4, false, 4, &opts);
}

#[test]
#[should_panic(expected = "davidson_block: restart_keep")]
fn tms_a9_restart_keep_at_maxsubspace_panics() {
    let opts = BlockDavidsonOpts {
        maxsubspace: 16,
        restart_keep: 16,
        ..BlockDavidsonOpts::for_roots(4)
    };
    a9_call(8, 4, false, 4, &opts);
}

#[test]
#[should_panic(expected = "davidson_block: local dimension")]
fn tms_a9_local_dimension_below_nroots_panics() {
    let opts = BlockDavidsonOpts {
        maxsubspace: 8,
        restart_keep: 3,
        ..BlockDavidsonOpts::for_roots(3)
    };
    a9_call(2, 1, false, 3, &opts);
}

#[test]
fn tms_a1_maxiter_exit_returns_every_root_the_subspace_hosts() {
    let fx = d1_fixture(8);
    let d_core = fx.diagonal();
    let v0 = fx.randn(0x1E_0001);
    let nroots = 3usize;
    for maxiter in [1usize, 2, 3] {
        let tag = format!("a1e maxiter={maxiter}");
        let opts = BlockDavidsonOpts {
            tol: 1e-14,
            maxiter,
            ..BlockDavidsonOpts::for_roots(nroots)
        };
        let mut op = |a: &SparseCore<f64>| fx.apply(a);
        let mut prec = |t: &mut SparseCore<f64>, theta: f64| precond(t, &d_core, theta);
        let (thetas, us, info) =
            davidson_block(&mut op, std::slice::from_ref(&v0), &mut prec, nroots, &opts);

        let hosted = nroots.min(info.numops);
        assert_eq!(
            thetas.len(),
            hosted,
            "{tag}: root count with a {}-column basis",
            info.numops
        );
        assert_eq!(us.len(), hosted, "{tag}: vector count");
        assert_eq!(info.normres.len(), hosted, "{tag}: normres length");
        assert_ascending(&thetas, &tag);
        let gd = gram_dev(&us);
        assert!(gd <= 1e-12, "{tag}: ‖Gram − I‖ = {gd:.3e}");
        println!(
            "{tag}: numops={} niter={} roots={}",
            info.numops,
            info.niter,
            thetas.len()
        );
    }
}

#[allow(clippy::type_complexity)]
fn a3_fixture() -> (
    SparseHamiltonian<f64>,
    Sketch<f64>,
    Sketch<f64>,
    ContractedSparseCores<f64>,
    ContractedSparseCores<f64>,
    Vec<f64>,
    Mat<f64>,
    Vec<f64>,
) {
    let d = 6usize;
    let (t, v) = generic_truncated_tv::<f64>(d);
    let space = QnSpace::new(d, 1, 1);
    let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let (_x, fl, fr, cp0, dd) = abelian_local(space, &h, 2, 0xB10C_0A02);
    let (m, g, _goff) = {
        let mut op =
            |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
        local_dense(&mut op, &cp0)
    };
    let (ev, y) = dense_self_adjoint_eigen(m.as_ref());
    (h, fl, fr, cp0, dd, ev, y, g)
}

#[test]
fn tms_a3_soft_locking_reactivates_a_displaced_root() {
    let (h, fl, fr, cp0, dd, ev, y, g) = a3_fixture();
    let mut op = |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
    let q1 = oracle_vector(&cp0, &y, 1, &g);
    let r1 = fresh_residual(&mut op, &q1, ev[1]);
    assert!(
        r1 <= 1e-12,
        "a3soft: the seeded level-1 eigenvector is not exact (residual {r1:.3e})"
    );

    let mut rng = seeded(0xA3_5017);
    let guesses = vec![q1.clone(), randn_block(&cp0, &mut rng)];
    let mut prec = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
    let opts = BlockDavidsonOpts {
        tol: 1e-10,
        ..BlockDavidsonOpts::for_roots(2)
    };
    let (thetas, us, info) = davidson_block(&mut op, &guesses, &mut prec, 2, &opts);

    assert_converged_block(&info, 2, opts.tol, "a3soft");
    for i in 0..2 {
        assert!(
            (thetas[i] - ev[i]).abs() <= 1e-10,
            "a3soft: θ[{i}] = {} vs dense {} (Δ = {:.3e})",
            thetas[i],
            ev[i],
            (thetas[i] - ev[i]).abs()
        );
        let res = fresh_residual(&mut op, &us[i], thetas[i]);
        assert!(
            res <= 1e-9,
            "a3soft: fresh residual of root {i} = {res:.3e}"
        );
    }
    assert!(
        q1.dot(&us[1]).abs_s() >= 1.0 - 1e-8,
        "a3soft: the seeded level-1 eigenvector did not end at position 1 (|⟨q1,u1⟩| = {})",
        q1.dot(&us[1]).abs_s()
    );
    assert!(
        q1.dot(&us[0]).abs_s() <= 1e-8,
        "a3soft: position 0 still carries the seeded excited vector (|⟨q1,u0⟩| = {})",
        q1.dot(&us[0]).abs_s()
    );
}

#[test]
fn tms_a3_theta_tol_is_keyed_per_root() {
    let (h, fl, fr, cp0, dd, ev, _y, _g) = a3_fixture();
    let mut op = |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
    let nroots = 3usize;
    assert_exclusion_gap(&ev, nroots, 0.05, "a3theta");
    let mut rng = seeded(0xA3_7701);
    let guesses = vec![
        cp0.clone(),
        randn_block(&cp0, &mut rng),
        randn_block(&cp0, &mut rng),
    ];
    let base = BlockDavidsonOpts {
        tol: 1e-10,
        ..BlockDavidsonOpts::for_roots(nroots)
    };

    let mut prec = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
    let (_th_none, _us_none, info_none) =
        davidson_block(&mut op, &guesses, &mut prec, nroots, &base);
    assert_converged_block(&info_none, nroots, base.tol, "a3theta none");
    let tight = BlockDavidsonOpts {
        theta_tol: Some(1e-16),
        ..base
    };
    let (th_tight, _us_tight, info_tight) =
        davidson_block(&mut op, &guesses, &mut prec, nroots, &tight);
    assert!(
        info_tight.converged && info_tight.nconv == nroots,
        "a3theta tight: nconv = {} of {nroots}",
        info_tight.nconv
    );
    assert!(
        info_tight.numops <= info_none.numops,
        "a3theta: a roundoff-scale θtol cost applies ({} vs {})",
        info_tight.numops,
        info_none.numops
    );
    for i in 0..nroots {
        assert!(
            (th_tight[i] - ev[i]).abs() <= 1e-10,
            "a3theta tight: root {i} at Δ = {:.3e} from its dense level {}",
            (th_tight[i] - ev[i]).abs(),
            ev[i]
        );
    }
    println!(
        "tms_a3 theta_tol=1e-16: numops={} niter={} (theta_tol=None: numops={} niter={})",
        info_tight.numops, info_tight.niter, info_none.numops, info_none.niter
    );

    let coarse = BlockDavidsonOpts {
        theta_tol: Some(1e-5),
        ..base
    };
    let (th_c, us_c, info_c) = davidson_block(&mut op, &guesses, &mut prec, nroots, &coarse);
    assert!(
        info_c.converged && info_c.nconv == nroots,
        "a3theta coarse: nconv = {} of {nroots} (niter = {})",
        info_c.nconv,
        info_c.niter
    );
    assert!(
        info_c.niter < info_none.niter,
        "a3theta: the coarse energy exit saved no iteration ({} vs {}) — the arm is vacuous",
        info_c.niter,
        info_none.niter
    );
    let deltas: Vec<f64> = (0..nroots).map(|i| (th_c[i] - ev[i]).abs()).collect();
    println!(
        "tms_a3 theta_tol=1e-5: niter={} Δθ={deltas:?}",
        info_c.niter
    );
    for i in 0..nroots {
        assert!(
            deltas[i] <= 1e-4,
            "a3theta coarse: root {i} stopped at Δ = {:.3e} from its dense level {} — a per-root \
             energy exit must not let an excited root ride root 0's plateau",
            deltas[i],
            ev[i]
        );
    }
    let gd = gram_dev(&us_c);
    assert!(gd <= 1e-8, "a3theta coarse: ‖Gram − I‖ = {gd:.3e}");
}

#[test]
fn tms_a5_stagnation_exit_when_the_basis_fills_the_local_space() {
    let fx = d1_fixture(8);
    let d_core = fx.diagonal();
    let v0 = fx.randn(0x57_A6);
    let opts = BlockDavidsonOpts {
        tol: 0.0,
        maxsubspace: 16,
        restart_keep: 2,
        maxiter: 50,
        ..BlockDavidsonOpts::for_roots(2)
    };
    let mut op = |a: &SparseCore<f64>| fx.apply(a);
    let mut prec = |t: &mut SparseCore<f64>, theta: f64| precond(t, &d_core, theta);
    let (thetas, us, info) =
        davidson_block(&mut op, std::slice::from_ref(&v0), &mut prec, 2, &opts);

    assert!(
        !info.converged,
        "a5stag: tol = 0 is unreachable — this run must not report convergence"
    );
    assert_eq!(
        info.numops, 8,
        "a5stag: the basis must exhaust the 8-dimensional local space exactly once"
    );
    assert!(
        info.niter <= 9,
        "a5stag: niter = {} — the loop kept iterating after the basis stopped growing",
        info.niter
    );
    assert_eq!(thetas.len(), 2, "a5stag: root count");
    let (ev, _) = dense_self_adjoint_eigen(fx.hd.as_ref());
    for i in 0..2 {
        assert!(
            (thetas[i] - ev[i]).abs() <= 1e-12,
            "a5stag: θ[{i}] = {} vs dense {}",
            thetas[i],
            ev[i]
        );
    }
    let gd = gram_dev(&us);
    assert!(gd <= 1e-10, "a5stag: ‖Gram − I‖ = {gd:.3e}");
    println!(
        "tms_a5 stagnation: numops={} niter={} converged={}",
        info.numops, info.niter, info.converged
    );
}

#[test]
fn tms_a5_short_return_on_an_exact_eigenvector_seed() {
    let fx = d1_fixture(8);
    let d_core = fx.diagonal();
    let (ev, y) = dense_self_adjoint_eigen(fx.hd.as_ref());
    let mut u_exact = fx.zeros();
    {
        let mut b = u_exact.block_mut(Family::Empty, fx.s0).unwrap();
        for i in 0..fx.r {
            *b.rb_mut().get_mut(i, 0) = *y.get(i, 0);
        }
    }
    let opts = BlockDavidsonOpts {
        tol: 1e-8,
        ..BlockDavidsonOpts::for_roots(2)
    };
    let mut op = |a: &SparseCore<f64>| fx.apply(a);
    let mut prec = |t: &mut SparseCore<f64>, theta: f64| precond(t, &d_core, theta);
    let (thetas, us, info) =
        davidson_block(&mut op, std::slice::from_ref(&u_exact), &mut prec, 2, &opts);

    assert_eq!(thetas.len(), 1, "a5short: root count");
    assert_eq!(us.len(), 1, "a5short: vector count");
    assert_eq!(info.normres.len(), 1, "a5short: normres length");
    assert_eq!(info.nconv, 1, "a5short: nconv");
    assert!(
        !info.converged,
        "a5short: one root of two came back — the run is not converged"
    );
    assert!(
        (thetas[0] - ev[0]).abs() <= 1e-12,
        "a5short: θ = {} vs dense {}",
        thetas[0],
        ev[0]
    );
}

#[test]
fn tms_a5_restart_keep_above_nroots_rebuilds_the_extra_columns() {
    let d = 6usize;
    let nroots = 3usize;
    let (t, v) = generic_truncated_tv::<f64>(d);

    {
        let space = QnSpace::new(d, 1, 1);
        let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
        let (_x, fl, fr, cp0, dd) = abelian_local(space, &h, 2, 0xB10C_0A02);
        let mut op =
            |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
        let (m, _g, _goff) = local_dense(&mut op, &cp0);
        let (ev, _) = dense_self_adjoint_eigen(m.as_ref());
        let mut rng = seeded(0xA5C_0001);
        let guesses = vec![
            cp0.clone(),
            randn_block(&cp0, &mut rng),
            randn_block(&cp0, &mut rng),
        ];
        let mut prec = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
        let opts = BlockDavidsonOpts {
            tol: 1e-10,
            maxsubspace: 4 * nroots,
            restart_keep: 2 * nroots,
            maxiter: 400,
            ..BlockDavidsonOpts::for_roots(nroots)
        };
        let (thetas, us, info) = davidson_block(&mut op, &guesses, &mut prec, nroots, &opts);
        assert_converged_block(&info, nroots, opts.tol, "a5keep f64");
        assert!(
            info.numops > opts.maxsubspace,
            "a5keep f64: numops {} ≤ maxsubspace {} — no restart was exercised",
            info.numops,
            opts.maxsubspace
        );
        for i in 0..nroots {
            assert!(
                (thetas[i] - ev[i]).abs() <= 1e-10,
                "a5keep f64: θ[{i}] = {} vs dense {}",
                thetas[i],
                ev[i]
            );
        }
        let gd = gram_dev(&us);
        assert!(gd <= 1e-10, "a5keep f64: ‖Gram − I‖ = {gd:.3e}");
        println!(
            "tms_a5 keep>nroots f64: niter={} numops={}",
            info.niter, info.numops
        );
    }

    {
        let theta: Vec<f64> = (0..d).map(|p| 0.3 + 0.4 * p as f64).collect();
        let (tc, vc) = gauge_integrals(d, &t, &v, &theta);
        let space = QnSpace::new(d, 1, 1);
        let h = SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);
        let (_x, fl, fr, cp0, dd) = abelian_local(space, &h, 2, 0xC64_0A05);
        let mut op =
            |cp: &ContractedSparseCores<C64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
        let (m, _g, _goff) = local_dense(&mut op, &cp0);
        let (ev, _) = dense_self_adjoint_eigen(m.as_ref());
        let mut rng = seeded(0xA5C_0002);
        let guesses = vec![
            cp0.clone(),
            randn_block(&cp0, &mut rng),
            randn_block(&cp0, &mut rng),
        ];
        let mut prec = |t: &mut ContractedSparseCores<C64>, theta: f64| precond(t, &dd, theta);
        let opts = BlockDavidsonOpts {
            tol: 1e-10,
            maxsubspace: 4 * nroots,
            restart_keep: 2 * nroots,
            maxiter: 400,
            ..BlockDavidsonOpts::for_roots(nroots)
        };
        let (thetas, us, info) = davidson_block(&mut op, &guesses, &mut prec, nroots, &opts);
        assert_converged_block(&info, nroots, opts.tol, "a5keep c64");
        assert!(
            info.numops > opts.maxsubspace,
            "a5keep c64: numops {} ≤ maxsubspace {} — no restart was exercised",
            info.numops,
            opts.maxsubspace
        );
        for i in 0..nroots {
            assert!(
                (thetas[i] - ev[i]).abs() <= 1e-10,
                "a5keep c64: θ[{i}] = {} vs dense {}",
                thetas[i],
                ev[i]
            );
        }
        let gd = gram_dev(&us);
        assert!(gd <= 1e-10, "a5keep c64: ‖Gram − I‖ = {gd:.3e}");
        println!(
            "tms_a5 keep>nroots c64: niter={} numops={}",
            info.niter, info.numops
        );
    }

    {
        let space = Su2Space::new(d, 2, 0);
        let h = Su2SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
        let (_x, fl, fr, cp0, dd) = su2_local(space, &h, 2, 0x50_2A02);
        let mut op = |cp: &Su2ContractedCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
        let (m, g, _goff) = local_dense(&mut op, &cp0);
        let (ev, _) = dense_self_adjoint_eigen(whiten(&m, &g).as_ref());
        let mut rng = seeded(0xA5C_0003);
        let guesses = vec![
            cp0.clone(),
            randn_block(&cp0, &mut rng),
            randn_block(&cp0, &mut rng),
        ];
        let mut prec = |t: &mut Su2ContractedCores<f64>, theta: f64| precond(t, &dd, theta);
        let opts = BlockDavidsonOpts {
            tol: 1e-10,
            maxsubspace: 4 * nroots,
            restart_keep: 2 * nroots,
            maxiter: 400,
            ..BlockDavidsonOpts::for_roots(nroots)
        };
        let (thetas, us, info) = davidson_block(&mut op, &guesses, &mut prec, nroots, &opts);
        assert_converged_block(&info, nroots, opts.tol, "a5keep su2");
        assert!(
            info.numops > opts.maxsubspace,
            "a5keep su2: numops {} ≤ maxsubspace {} — no restart was exercised",
            info.numops,
            opts.maxsubspace
        );
        for i in 0..nroots {
            assert!(
                (thetas[i] - ev[i]).abs() <= 1e-10,
                "a5keep su2: θ[{i}] = {} vs the generalized-pencil eigenvalue {}",
                thetas[i],
                ev[i]
            );
        }
        let gd = gram_dev(&us);
        assert!(gd <= 1e-10, "a5keep su2: ‖Gram − I‖ = {gd:.3e}");
        println!(
            "tms_a5 keep>nroots su2: niter={} numops={}",
            info.niter, info.numops
        );
    }
}

#[test]
fn tms_a6_home_roots_is_phase_blind() {
    let (_cp0, us) = a6_roots();
    let mut p0 = us[2].clone();
    p0.scale(-1.0);
    let p1 = us[0].clone();
    let (perm, gram) = home_roots(&[p0, p1], &us);
    assert_eq!(perm, vec![2, 0], "a6phase: sign-flipped previous roots");
    assert!(
        (gram[0][2] + 1.0).abs() <= 1e-10,
        "a6phase: the flipped overlap is {} , expected −1",
        gram[0][2]
    );

    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut rng = seeded(0xA6_0C64);
    let mut x: TtVector<C64> = qntt::tt_randn(space, &qntt::max_ranks(&space), &mut rng);
    x.right_orthogonalize(false);
    x.move_core(2, false);
    let template = contract(x.core(2), x.core(3));
    let basis: Vec<ContractedSparseCores<C64>> = (0..3)
        .map(|j| {
            let mut e = template.clone();
            e.for_each_coeff_mut(|c| *c = C64::new(0.0, 0.0));
            e.set_coeff(j, C64::new(1.0, 0.0));
            e
        })
        .collect();
    let gd = gram_dev(&basis);
    assert!(
        gd <= 1e-14,
        "a6phase: the C64 probe block is not orthonormal ({gd:.3e})"
    );
    let phi = 1.1f64;
    let ph = C64::new(phi.cos(), phi.sin());
    let mut r0 = basis[2].clone();
    r0.scale(ph);
    let mut r1 = basis[0].clone();
    r1.scale(-ph);
    let (permc, gramc) = home_roots(&[r0, r1], &basis);
    assert_eq!(permc, vec![2, 0], "a6phase: phase-rotated previous roots");
    assert!(
        (gramc[0][2].abs_s() - 1.0).abs() <= 1e-12,
        "a6phase: |⟨e^{{iφ}}u₂, u₂⟩| = {}",
        gramc[0][2].abs_s()
    );
    assert!(
        gramc[0][2].re_s() < 0.9,
        "a6phase: the phase rotation left the overlap essentially real ({})",
        gramc[0][2].re_s()
    );
}

#[allow(clippy::type_complexity)]
fn a7_fixture() -> (
    SparseHamiltonian<f64>,
    Sketch<f64>,
    Sketch<f64>,
    ContractedSparseCores<f64>,
    ContractedSparseCores<f64>,
    ContractedSparseCores<f64>,
) {
    let d = 6usize;
    let (t, v) = generic_truncated_tv::<f64>(d);
    let space = QnSpace::new(d, 1, 1);
    let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let (_x, fl, fr, cp0, dd) = abelian_local(space, &h, 2, 0xB10C_0A02);
    let (m, g, _goff) = {
        let mut op =
            |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
        local_dense(&mut op, &cp0)
    };
    let (_ev, y) = dense_self_adjoint_eigen(m.as_ref());
    let q0 = oracle_vector(&cp0, &y, 0, &g);
    let q1 = oracle_vector(&cp0, &y, 1, &g);
    let mut q = q0.clone();
    q.axpy(0.1, &q1);
    let qn = q.norm();
    q.scale(1.0 / qn);
    (h, fl, fr, cp0, dd, q)
}

#[test]
fn tms_a7_guess_annihilated_by_the_projector_is_dropped() {
    let (h, fl, fr, cp0, dd, q) = a7_fixture();
    let mut op = |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
    let mut project = |b: &mut ContractedSparseCores<f64>| {
        let c = q.dot(b);
        b.axpy(-c, &q);
    };
    let mut g_ann = q.clone();
    g_ann.scale(3.7);
    let mut probe = g_ann.clone();
    project(&mut probe);
    assert!(
        probe.norm() <= 1e-12 && g_ann.norm() > 1.0,
        "a7drop: the probe guess is not annihilated (‖Pg‖ = {:.3e}, ‖g‖ = {:.3e})",
        probe.norm(),
        g_ann.norm()
    );

    let mut rng = seeded(0xA7_D00D);
    let g2 = randn_block(&cp0, &mut rng);
    let opts = BlockDavidsonOpts {
        tol: 1e-10,
        ..BlockDavidsonOpts::for_roots(1)
    };
    let mut prec = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
    let (thetas, us, info) = davidson_block_projected(
        &mut op,
        &[g_ann, g2.clone()],
        &mut prec,
        &mut project,
        1,
        &opts,
    );
    assert_eq!(
        info.dropped_guesses, 1,
        "a7drop: the annihilated guess must be counted as dropped"
    );
    assert!(info.converged, "a7drop: the projected run did not converge");
    assert!(
        q.dot(&us[0]).abs_s() <= 1e-12,
        "a7drop: the root is not orthogonal to the projected-out state"
    );

    let mut prec2 = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
    let (th_ref, _us_ref, info_ref) = davidson_block_projected(
        &mut op,
        std::slice::from_ref(&g2),
        &mut prec2,
        &mut project,
        1,
        &opts,
    );
    assert_eq!(info_ref.dropped_guesses, 0, "a7drop: reference run");
    assert_eq!(info_ref.numops, info.numops, "a7drop: numops");
    assert_eq!(info_ref.niter, info.niter, "a7drop: niter");
    assert_eq!(
        thetas[0].to_bits(),
        th_ref[0].to_bits(),
        "a7drop: θ {} vs the single-guess reference {} — the annihilated guess left a trace",
        thetas[0],
        th_ref[0]
    );
}

#[test]
#[should_panic(expected = "davidson_block: every guess was dropped")]
fn tms_a7_all_guesses_annihilated_by_the_projector_panics() {
    let (h, fl, fr, _cp0, dd, q) = a7_fixture();
    let mut op = |cp: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(cp, &fl, &fr);
    let mut project = |b: &mut ContractedSparseCores<f64>| {
        let c = q.dot(b);
        b.axpy(-c, &q);
    };
    let mut g0 = q.clone();
    g0.scale(3.7);
    let mut g1 = q.clone();
    g1.scale(-2.0);
    let mut prec = |t: &mut ContractedSparseCores<f64>, theta: f64| precond(t, &dd, theta);
    let _ = davidson_block_projected(
        &mut op,
        &[g0, g1],
        &mut prec,
        &mut project,
        1,
        &BlockDavidsonOpts::for_roots(1),
    );
}
