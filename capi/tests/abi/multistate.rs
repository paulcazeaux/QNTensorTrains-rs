// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use qntt::solvers::excited::SsStart;
use qntt::solvers::multistate::StateAverageOptions;
use qntt::solvers::ssmals::{PenaltyRule, SsMode, StateSpecificOptions};
use qntt::{CoeffTensor, TtVector, dot, tt_state_in};
use qntt_capi::enums;
use qntt_capi::err::QNTT_NULL_ARG;

fn eigenvalues_sym(a: &mut [f64], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * n);
    let off = |a: &[f64]| -> f64 {
        let mut s = 0.0;
        for p in 0..n {
            for q in 0..n {
                if p != q {
                    s += a[p * n + q] * a[p * n + q];
                }
            }
        }
        s.sqrt()
    };
    let scale = (0..n * n).map(|i| a[i] * a[i]).sum::<f64>().sqrt().max(1.0);
    for _ in 0..100 {
        if off(a) <= 1e-14 * scale {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let tt = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (tt * tt + 1.0).sqrt();
                let s = tt * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    assert!(
        off(a) <= 1e-10 * scale,
        "Jacobi did not converge: off-diagonal norm {} (scale {scale})",
        off(a)
    );
    let mut ev: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    ev.sort_by(|x, y| x.partial_cmp(y).expect("no NaN in a converged spectrum"));
    ev
}

fn occupations(d: usize, n: usize) -> Vec<Vec<bool>> {
    (0..(1usize << d))
        .filter(|m| m.count_ones() as usize == n)
        .map(|m| (0..d).map(|k| (m >> k) & 1 == 1).collect())
        .collect()
}

fn dense_levels(
    d: usize,
    nup: usize,
    ndn: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
) -> Vec<f64> {
    let space = QnSpace::new(d, nup, ndn);
    let h: SparseHamiltonian<f64> = SparseHamiltonian::new(space, t, v, None, f64::EPSILON);
    let basis: Vec<TtVector<f64>> = occupations(d, nup)
        .iter()
        .flat_map(|u| {
            occupations(d, ndn)
                .into_iter()
                .map(move |dn| tt_state_in::<f64>(space, u, &dn))
        })
        .collect();
    let n = basis.len();
    assert!(n > 0, "empty determinant block ({nup}, {ndn})");
    let mut hm = vec![0.0f64; n * n];
    for (j, ej) in basis.iter().enumerate() {
        let hej = h.apply(ej);
        for (i, ei) in basis.iter().enumerate() {
            hm[i * n + j] = dot(ei, &hej);
        }
    }
    let mut asym = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            asym = asym.max((hm[i * n + j] - hm[j * n + i]).abs());
        }
    }
    assert!(
        asym < 1e-9,
        "the dense reference is not Hermitian (max {asym})"
    );
    eigenvalues_sym(&mut hm, n)
}

fn singlet_levels(
    d: usize,
    nup: usize,
    ndn: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
) -> Vec<f64> {
    let full = dense_levels(d, nup, ndn, t, v);
    let mut higher = dense_levels(d, nup + 1, ndn - 1, t, v);
    let mut out = Vec::new();
    for e in &full {
        match higher.iter().position(|y| (y - e).abs() < 1e-8) {
            Some(p) => {
                higher.remove(p);
            }
            None => out.push(*e),
        }
    }
    assert!(
        higher.is_empty(),
        "{} higher-Sz level(s) had no partner — the multiset difference is not the S = 0 spectrum",
        higher.len()
    );
    out
}

fn assert_gapped(levels: &[f64], nroots: usize, floor: f64, who: &str) {
    assert!(
        levels.len() > nroots,
        "{who}: the fixture resolves {} levels, need more than {nroots}",
        levels.len()
    );
    let gap = levels[nroots] - levels[nroots - 1];
    println!("{who}: gap to the first excluded level = {gap:.6} (floor {floor})");
    assert!(
        gap >= floor,
        "{who}: the gap to the first excluded level is {gap:.6}, below the fixture floor {floor}"
    );
}

fn h4_abelian_ham() -> (
    QnSpace,
    Vec<f64>,
    Vec<f64>,
    *mut qntt_capi::handle::Handle<SparseHamiltonian<f64>>,
) {
    let (norb, nelec, ms2, t, v) = read_h4_through_abi();
    let space = QnSpace::from_n_two_sz(norb, nelec, ms2);
    let mut h: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_ham_new(
                space.d,
                space.nup,
                space.ndn,
                t.as_ptr(),
                t.len(),
                v.as_ptr(),
                v.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK,
        "ham_new failed: {}",
        last_error()
    );
    (space, t, v, h)
}

fn randn_abelian(
    space: QnSpace,
    r: usize,
    seed: u64,
) -> *mut qntt_capi::handle::Handle<qntt::TtVector<f64>> {
    let flat = flat_ranks(&space, &uniform_ranks(space, r));
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                seed,
                &mut x,
            )
        },
        QNTT_OK,
        "vec_randn failed: {}",
        last_error()
    );
    x
}

fn abelian_coeffs(
    space: &QnSpace,
    x: *const qntt_capi::handle::Handle<qntt::TtVector<f64>>,
) -> Vec<f64> {
    let mut out = Vec::new();
    for site in 0..space.d {
        for family in 0..4i32 {
            for s in space.bond_sectors(site) {
                let (mut rows, mut cols) = (0usize, 0usize);
                assert_eq!(
                    unsafe {
                        qntt_f64_vec_block_get(
                            x,
                            site,
                            family,
                            s.nup,
                            s.ndn,
                            ptr::null_mut(),
                            0,
                            &mut rows,
                            &mut cols,
                        )
                    },
                    QNTT_OK
                );
                if rows * cols == 0 {
                    continue;
                }
                let mut buf = vec![0.0f64; rows * cols];
                assert_eq!(
                    unsafe {
                        qntt_f64_vec_block_get(
                            x,
                            site,
                            family,
                            s.nup,
                            s.ndn,
                            buf.as_mut_ptr(),
                            buf.len(),
                            &mut rows,
                            &mut cols,
                        )
                    },
                    QNTT_OK
                );
                out.extend_from_slice(&buf);
            }
        }
    }
    assert!(
        !out.is_empty(),
        "vacuous traversal: the train has no blocks"
    );
    out
}

macro_rules! ranks_of {
    ($f:ident, $x:expr) => {{
        let mut len = 0usize;
        assert_eq!(unsafe { $f($x, ptr::null_mut(), 0, &mut len) }, QNTT_OK);
        let mut r = vec![0usize; len];
        assert_eq!(
            unsafe { $f($x, r.as_mut_ptr(), r.len(), &mut len) },
            QNTT_OK
        );
        r
    }};
}

fn sa_full_rank(eps: f64, max_iter: usize) -> QnttStateAverageOptions {
    let mut o = QnttStateAverageOptions::defaults();
    o.mals.eps = eps;
    o.mals.max_iter = max_iter;
    o
}

#[test]
fn multistate_option_defaults_match_the_core_crate() {
    let mut sa = QnttStateAverageOptions::defaults();
    let derived = sa;
    assert_eq!(
        unsafe { qntt_state_average_options_default(&mut sa) },
        QNTT_OK
    );
    let core = StateAverageOptions::default();
    let mals = QnttMalsOptions::defaults();
    assert_eq!((sa.homing, sa.reorthonormalize), (0, 1));
    assert_eq!(sa.homing != 0, core.homing);
    assert_eq!(sa.reorthonormalize != 0, core.reorthonormalize);
    assert_eq!(sa.scale, core.scale);
    assert_eq!(
        (sa.maxsubspace, sa.restart_keep),
        (0, 0),
        "0 is the ABI spelling of `None` for both Davidson knobs"
    );
    assert!(core.maxsubspace.is_none() && core.restart_keep.is_none());
    assert_eq!(
        (
            sa.mals.eps,
            sa.mals.max_iter,
            sa.mals.seed,
            sa.mals.noise_type
        ),
        (mals.eps, mals.max_iter, mals.seed, mals.noise_type),
        "the nested MALS block is the same mirror the single-state filler writes"
    );
    assert_eq!(
        (
            derived.homing,
            derived.reorthonormalize,
            derived.scale,
            derived.maxsubspace,
            derived.restart_keep
        ),
        (
            sa.homing,
            sa.reorthonormalize,
            sa.scale,
            sa.maxsubspace,
            sa.restart_keep
        ),
        "the exported filler and `defaults()` must be one body"
    );

    let mut ss = QnttStateSpecificOptions::defaults();
    let ss_derived = ss;
    assert_eq!(
        unsafe { qntt_state_specific_options_default(&mut ss) },
        QNTT_OK
    );
    let core = StateSpecificOptions::default();
    assert_eq!(ss.roots, core.roots);
    assert_eq!(ss.mode, enums::ss_mode_code(core.mode));
    assert_eq!(ss.mode, 0, "the recorded default is Penalty (wire code 0)");
    assert_eq!(
        (ss.penalty_rule, ss.penalty_value),
        enums::penalty_rule_code(core.penalty),
        "the penalty pair must be the inverse of the core default"
    );
    assert_eq!(
        ss.penalty_rule, 0,
        "the recorded default is Auto (wire code 0)"
    );
    assert_eq!((ss.maxsubspace, ss.restart_keep), (0, 0));
    assert_eq!(
        (ss_derived.mode, ss_derived.penalty_rule, ss_derived.roots),
        (ss.mode, ss.penalty_rule, ss.roots)
    );
    assert_eq!(
        (
            ss.mals.eps,
            ss.mals.max_iter,
            ss.mals.seed,
            ss.mals.noise_type
        ),
        (mals.eps, mals.max_iter, mals.seed, mals.noise_type)
    );

    let mut ex = QnttExcitedOptions::defaults();
    let ex_derived = ex;
    assert_eq!(unsafe { qntt_excited_options_default(&mut ex) }, QNTT_OK);
    assert_eq!(
        (
            ex_derived.nroots,
            ex_derived.rank,
            ex_derived.eps,
            ex_derived.max_iter,
            ex_derived.seed,
            ex_derived.noise_type,
            ex_derived.mode,
            ex_derived.penalty_rule,
            ex_derived.ss_start,
            ex_derived.cg_eps,
            ex_derived.cg_variant,
            ex_derived.overlap_tol
        ),
        (
            ex.nroots,
            ex.rank,
            ex.eps,
            ex.max_iter,
            ex.seed,
            ex.noise_type,
            ex.mode,
            ex.penalty_rule,
            ex.ss_start,
            ex.cg_eps,
            ex.cg_variant,
            ex.overlap_tol
        ),
        "the exported filler and `defaults()` must be one body"
    );

    struct ExRef {
        who: &'static str,
        nroots: usize,
        rank: usize,
        eps: f64,
        max_iter: usize,
        seed: u64,
        polish: bool,
        lower_grad_tol: f64,
        trim_tol: f64,
        trim_passes: usize,
        cg_eps: f64,
        cg_max_iter: usize,
        cg_gradtol: f64,
        cg_seed: u64,
        cg_restarts: usize,
        polish_tol: f64,
        grad_gate: f64,
        overlap_tol: f64,
        keep_sa: bool,
        verbose: bool,
    }

    let ab = qntt::solvers::excited::ExcitedOpts::new(&QnSpace::new(6, 1, 1));
    let su2 = qntt::su2::solvers::excited::ExcitedOpts::new(&qntt::su2::qn::Su2Space::new(6, 2, 0));
    let refs = [
        ExRef {
            who: "abelian",
            nroots: ab.nroots,
            rank: ab.rank,
            eps: ab.eps,
            max_iter: ab.max_iter,
            seed: ab.seed,
            polish: ab.polish,
            lower_grad_tol: ab.lower_grad_tol,
            trim_tol: ab.trim_tol,
            trim_passes: ab.trim_passes,
            cg_eps: ab.cg.eps,
            cg_max_iter: ab.cg.max_iter,
            cg_gradtol: ab.cg.gradtol,
            cg_seed: ab.cg.seed,
            cg_restarts: ab.cg_restarts,
            polish_tol: ab.polish_tol,
            grad_gate: ab.grad_gate,
            overlap_tol: ab.overlap_tol,
            keep_sa: ab.keep_sa,
            verbose: ab.verbose,
        },
        ExRef {
            who: "su2",
            nroots: su2.nroots,
            rank: su2.rank,
            eps: su2.eps,
            max_iter: su2.max_iter,
            seed: su2.seed,
            polish: su2.polish,
            lower_grad_tol: su2.lower_grad_tol,
            trim_tol: su2.trim_tol,
            trim_passes: su2.trim_passes,
            cg_eps: su2.cg.eps,
            cg_max_iter: su2.cg.max_iter,
            cg_gradtol: su2.cg.gradtol,
            cg_seed: su2.cg.seed,
            cg_restarts: su2.cg_restarts,
            polish_tol: su2.polish_tol,
            grad_gate: su2.grad_gate,
            overlap_tol: su2.overlap_tol,
            keep_sa: su2.keep_sa,
            verbose: su2.verbose,
        },
    ];
    for r in &refs {
        let who = r.who;
        assert_eq!(ex.nroots, r.nroots, "{who}: nroots");
        assert_eq!(ex.rank, r.rank, "{who}: rank");
        assert_eq!(ex.eps, r.eps, "{who}: eps");
        assert_eq!(ex.max_iter, r.max_iter, "{who}: max_iter");
        assert_eq!(ex.seed, r.seed, "{who}: seed");
        assert_eq!(ex.polish != 0, r.polish, "{who}: polish");
        assert_eq!(ex.lower_grad_tol, r.lower_grad_tol, "{who}: lower_grad_tol");
        assert_eq!(ex.trim_tol, r.trim_tol, "{who}: trim_tol");
        assert_eq!(ex.trim_passes, r.trim_passes, "{who}: trim_passes");
        assert_eq!(ex.cg_eps, r.cg_eps, "{who}: cg_eps");
        assert_eq!(
            ex.cg_eps, 1e-13,
            "{who}: the pipeline override must survive the mirror"
        );
        assert_eq!(ex.cg_max_iter, r.cg_max_iter, "{who}: cg_max_iter");
        assert_eq!(ex.cg_gradtol, r.cg_gradtol, "{who}: cg_gradtol");
        assert_eq!(ex.cg_seed, r.cg_seed, "{who}: cg_seed");
        assert_eq!(ex.cg_restarts, r.cg_restarts, "{who}: cg_restarts");
        assert_eq!(ex.polish_tol, r.polish_tol, "{who}: polish_tol");
        assert_eq!(ex.grad_gate, r.grad_gate, "{who}: grad_gate");
        assert_eq!(ex.overlap_tol, r.overlap_tol, "{who}: overlap_tol");
        assert_eq!(ex.keep_sa != 0, r.keep_sa, "{who}: keep_sa");
        assert_eq!(ex.verbose != 0, r.verbose, "{who}: verbose");
    }
    assert_eq!(ex.noise_type, enums::noise_type_code(ab.noise_type));
    assert_eq!(
        ex.noise_type, 1,
        "the ladder's recorded noise flavour is Perturbative"
    );
    assert_eq!(ex.mode, enums::ss_mode_code(ab.mode));
    assert_eq!(
        (ex.penalty_rule, ex.penalty_value),
        enums::penalty_rule_code(ab.penalty)
    );
    assert_eq!(ex.ss_start, enums::ss_start_code(ab.ss_start));
    assert_eq!(
        ex.ss_start, 0,
        "the recorded default start is the state-average state"
    );
    assert_eq!(ex.cg_variant, enums::cg_variant_code(ab.cg.variant));
    assert_eq!(
        ex.cg_variant, 1,
        "the recorded CG default is Polak-Ribière (wire code 1)"
    );
}

#[test]
fn multistate_wire_codes_round_trip() {
    for (code, m) in [(0, SsMode::Penalty), (1, SsMode::Project)] {
        assert_eq!(enums::ss_mode(code).expect("known code"), m);
        assert_eq!(enums::ss_mode_code(m), code);
    }
    assert_eq!(enums::ss_start(0).expect("known code"), SsStart::Sa);
    assert_eq!(enums::ss_start_code(SsStart::Sa), 0);
    {
        let e = enums::ss_start(1).expect_err("X0 is refused without the feature");
        assert!(
            e.msg.contains("not available"),
            "the refusal must say the code is unavailable: {}",
            e.msg
        );
    }
    assert_eq!(
        enums::penalty_rule(0, 0.0).expect("auto"),
        PenaltyRule::Auto
    );
    assert_eq!(
        enums::penalty_rule(0, f64::NAN).expect("auto ignores its payload"),
        PenaltyRule::Auto
    );
    assert_eq!(enums::penalty_rule_code(PenaltyRule::Auto).0, 0);
    assert_eq!(
        enums::penalty_rule(1, 2.5).expect("fixed"),
        PenaltyRule::Fixed(2.5)
    );
    assert_eq!(enums::penalty_rule_code(PenaltyRule::Fixed(2.5)), (1, 2.5));
    assert_eq!(
        enums::penalty_rule(1, 0.0).expect("fixed(0) is a legal, documented weight"),
        PenaltyRule::Fixed(0.0)
    );

    for bad in [-1i32, 2, 7] {
        let e = enums::ss_mode(bad).expect_err("unknown ss mode");
        assert_eq!(e.code, QNTT_INVALID_ARG);
        assert!(
            e.msg.contains("penalty") && e.msg.contains("project"),
            "got {:?}",
            e.msg
        );
        let e = enums::ss_start(bad).expect_err("unknown ss start");
        assert_eq!(e.code, QNTT_INVALID_ARG);
        let e = enums::penalty_rule(bad, 1.0).expect_err("unknown penalty rule");
        assert_eq!(e.code, QNTT_INVALID_ARG);
    }
    for bad in [-1.0f64, f64::NAN, f64::INFINITY] {
        let e = enums::penalty_rule(1, bad).expect_err("Fixed(w) must be finite and ≥ 0");
        assert_eq!(e.code, QNTT_INVALID_ARG);
    }
    for code in [0i32, 1] {
        let v = enums::cg_variant(code).expect("known cg variant");
        assert_eq!(enums::cg_variant_code(v), code);
    }
}

#[test]
fn multistate_options_convert_to_the_core_structs() {
    let mut c = QnttStateAverageOptions::defaults();
    c.mals.eps = 1e-9;
    c.mals.max_iter = 7;
    c.mals.seed = 99;
    c.mals.conv_tol = 1e-11;
    c.homing = 1;
    c.reorthonormalize = 0;
    c.scale = 0.25;
    c.maxsubspace = 40;
    c.restart_keep = 6;
    let core = c
        .to_core(3, &[0.5, 0.3, 0.2], &[8, 0], &[1e-3, 0.0])
        .expect("a valid option block");
    assert_eq!(core.mals.eps, 1e-9);
    assert_eq!(core.mals.max_iter, 7);
    assert_eq!(core.mals.seed, 99);
    assert_eq!(core.mals.conv_tol, Some(1e-11));
    assert_eq!(core.mals.bond_dims, vec![8, 0]);
    assert_eq!(core.mals.noise, vec![1e-3, 0.0]);
    assert!(core.homing && !core.reorthonormalize);
    assert_eq!(core.scale, 0.25);
    assert_eq!(core.weights, Some(vec![0.5, 0.3, 0.2]));
    assert_eq!((core.maxsubspace, core.restart_keep), (Some(40), Some(6)));
    let core = QnttStateAverageOptions::defaults()
        .to_core(2, &[], &[0], &[])
        .expect("empty weights = 1/n, empty noise = noise-free");
    assert_eq!(core.weights, None);
    assert_eq!(core.maxsubspace, None);
    assert_eq!(core.restart_keep, None);
    assert_eq!(
        core.mals.noise,
        vec![0.0],
        "empty noise is the [0.0] convenience"
    );

    let bad = |nroots: usize, w: &[f64], bd: &[usize], ns: &[f64]| {
        QnttStateAverageOptions::defaults()
            .to_core(nroots, w, bd, ns)
            .expect_err("must be refused")
    };
    assert!(bad(0, &[], &[0], &[]).msg.to_lowercase().contains("root"));
    assert!(bad(3, &[0.5, 0.5], &[0], &[]).msg.contains("weights"));
    assert!(
        bad(2, &[0.4, 0.6], &[0], &[])
            .msg
            .contains("non-increasing")
    );
    assert!(bad(2, &[0.6, 0.5], &[0], &[]).msg.contains("sum to 1"));
    assert!(bad(2, &[0.5, -0.5], &[0], &[]).msg.contains("positive"));
    assert!(bad(2, &[], &[], &[]).msg.contains("bond_dims"));
    assert!(bad(2, &[], &[0], &[-1.0]).msg.contains("noise"));
    let mut rnd = QnttStateAverageOptions::defaults();
    rnd.mals.noise_type = 0;
    let e = rnd
        .to_core(2, &[], &[0], &[1e-3, 0.0])
        .expect_err("Random is refused");
    assert_eq!(e.code, QNTT_INVALID_ARG);
    assert!(
        e.msg.contains("Perturbative") || e.msg.contains("perturbative"),
        "got {:?}",
        e.msg
    );
    let mut small = QnttStateAverageOptions::defaults();
    small.maxsubspace = 3;
    assert!(small.to_core(2, &[], &[0], &[]).is_err());
    small.maxsubspace = 3;
    assert!(
        small.to_core(1, &[], &[0], &[]).is_ok(),
        "the 2n rule binds only at n ≥ 2"
    );

    let mut c = QnttStateSpecificOptions::defaults();
    c.mode = 1;
    c.penalty_rule = 1;
    c.penalty_value = 3.5;
    c.roots = 2;
    c.maxsubspace = 32;
    c.restart_keep = 4;
    let core = c.to_core(2, &[-1.0, -0.5], &[0], &[]).expect("valid");
    assert_eq!(core.mode, SsMode::Project);
    assert_eq!(core.penalty, PenaltyRule::Fixed(3.5));
    assert_eq!(core.roots, 2);
    assert_eq!(core.lower_energies, Some(vec![-1.0, -0.5]));
    assert_eq!((core.maxsubspace, core.restart_keep), (Some(32), Some(4)));
    let core = QnttStateSpecificOptions::defaults()
        .to_core(2, &[], &[0], &[])
        .expect("empty lower_energies = compute at entry");
    assert_eq!(core.lower_energies, None);
    let mut rnd = QnttStateSpecificOptions::defaults();
    rnd.mals.noise_type = 0;
    let core = rnd
        .to_core(0, &[], &[0], &[1e-3, 0.0])
        .expect("state-specific sweeps accept Random noise");
    assert_eq!(core.mals.noise_type, qntt::solvers::mals::NoiseType::Random);

    let bad = |nlower: usize, le: &[f64], f: fn(&mut QnttStateSpecificOptions)| {
        let mut o = QnttStateSpecificOptions::defaults();
        f(&mut o);
        o.to_core(nlower, le, &[0], &[])
            .expect_err("must be refused")
    };
    assert!(
        bad(0, &[], |o| o.roots = 0)
            .msg
            .to_lowercase()
            .contains("root")
    );
    assert!(bad(2, &[1.0], |_| {}).msg.contains("lower_energies"));
    assert!(bad(0, &[], |o| o.mode = 5).code == QNTT_INVALID_ARG);
    assert!(bad(0, &[], |o| o.penalty_rule = 5).code == QNTT_INVALID_ARG);
    assert!(
        bad(0, &[], |o| {
            o.penalty_rule = 1;
            o.penalty_value = -1.0;
        })
        .code
            == QNTT_INVALID_ARG
    );
    let mut ms = QnttStateSpecificOptions::defaults();
    ms.roots = 2;
    ms.maxsubspace = 3;
    let e = ms
        .to_core(0, &[], &[0], &[])
        .expect_err("3 < 2·2 is refused");
    assert_eq!(e.code, QNTT_INVALID_ARG);
    assert!(
        e.msg.contains("sa_restart_keep"),
        "the refusal must name the rule it front-runs, got {:?}",
        e.msg
    );
    ms.maxsubspace = 4;
    assert!(
        ms.to_core(0, &[], &[0], &[]).is_ok(),
        "maxsubspace = 2·roots is the boundary, and it is legal"
    );
    ms.roots = 1;
    ms.maxsubspace = 3;
    assert!(
        ms.to_core(0, &[], &[0], &[]).is_ok(),
        "the 2·roots rule binds only at roots ≥ 2"
    );
}

#[test]
fn excited_option_bridge_overwrites_new_and_keeps_empty_schedules() {
    let space = QnSpace::new(6, 1, 1);
    let base = qntt::solvers::excited::ExcitedOpts::new(&space);

    let mut c = QnttExcitedOptions::defaults();
    c.nroots = 4;
    c.rank = 12;
    c.eps = 1e-8;
    c.max_iter = 11;
    c.seed = 7;
    c.mode = 1;
    c.penalty_rule = 1;
    c.penalty_value = 2.0;
    c.ss_start = if false { 1 } else { 0 };
    c.polish = 0;
    c.lower_grad_tol = 1e-5;
    c.trim_tol = 1e-4;
    c.trim_passes = 1;
    c.cg_eps = 1e-12;
    c.cg_max_iter = 42;
    c.cg_gradtol = 1e-11;
    c.cg_seed = 21;
    c.cg_variant = 0;
    c.cg_restarts = 2;
    c.polish_tol = 1e-9;
    c.grad_gate = 1e-8;
    c.overlap_tol = 1e-5;
    c.keep_sa = 1;
    c.verbose = 0;

    let weights = [0.7f64, 0.1, 0.1, 0.1];
    let sched = ExcitedSchedules {
        weights: &weights,
        bond_dims: &[4, 8, 0],
        noise: &[1e-3, 0.0],
        ss_noise: &[1e-5, 0.0],
    };
    let o = excited_opts_abelian(&c, &sched, base.clone()).expect("a valid ladder option block");
    assert_eq!(o.nroots, 4);
    assert_eq!(o.rank, 12);
    assert_eq!(o.eps, 1e-8);
    assert_eq!(o.max_iter, 11);
    assert_eq!(o.seed, 7);
    assert_eq!(o.mode, SsMode::Project);
    assert_eq!(o.penalty, PenaltyRule::Fixed(2.0));
    assert_eq!(o.ss_start, SsStart::Sa);
    assert!(!o.polish);
    assert_eq!(o.lower_grad_tol, 1e-5);
    assert_eq!(o.trim_tol, 1e-4);
    assert_eq!(o.trim_passes, 1);
    assert_eq!(o.cg.eps, 1e-12);
    assert_eq!(o.cg.max_iter, 42);
    assert_eq!(o.cg.gradtol, 1e-11);
    assert_eq!(o.cg.seed, 21);
    assert_eq!(o.cg.variant, qntt::solvers::riemannian::CgVariant::Fr);
    assert_eq!(o.cg_restarts, 2);
    assert_eq!(o.polish_tol, 1e-9);
    assert_eq!(o.grad_gate, 1e-8);
    assert_eq!(o.overlap_tol, 1e-5);
    assert!(o.keep_sa);
    assert_eq!(o.weights, Some(weights.to_vec()));
    assert_eq!(o.bond_dims, Some(vec![4, 8, 0]));
    assert_eq!(o.noise, vec![1e-3, 0.0]);
    assert_eq!(o.ss_noise, vec![1e-5, 0.0]);
    assert_eq!(o.cg.preserve_grid, base.cg.preserve_grid);

    let empty = ExcitedSchedules {
        weights: &[],
        bond_dims: &[],
        noise: &[],
        ss_noise: &[],
    };
    let o = excited_opts_abelian(&QnttExcitedOptions::defaults(), &empty, base.clone())
        .expect("empty schedules are the defaults");
    assert_eq!(o.weights, None, "empty weights = 1/n");
    assert_eq!(
        o.bond_dims, None,
        "empty bond_dims = the default ramp of `rank`"
    );
    assert_eq!(
        o.noise, base.noise,
        "empty `noise` keeps the built-in ramp — NOT noise-free (pass [0.0] for that)"
    );
    assert_eq!(o.ss_noise, base.ss_noise);

    let su2_space = qntt::su2::qn::Su2Space::new(6, 2, 0);
    let su2_base = qntt::su2::solvers::excited::ExcitedOpts::new(&su2_space);
    let o = excited_opts_su2(&c, &sched, su2_base.clone()).expect("valid");
    assert_eq!(o.nroots, 4);
    assert_eq!(o.cg.variant, qntt::su2::riemannian::CgVariant::Fr);

    let refuse = |f: fn(&mut QnttExcitedOptions)| {
        let mut c = QnttExcitedOptions::defaults();
        f(&mut c);
        excited_opts_abelian(&c, &empty, base.clone()).expect_err("must be refused")
    };
    assert_eq!(refuse(|c| c.nroots = 0).code, QNTT_INVALID_ARG);
    let random = refuse(|c| c.noise_type = 0);
    assert_eq!(random.code, QNTT_INVALID_ARG);
    assert!(
        random.msg.contains("Perturbative") || random.msg.contains("perturbative"),
        "the ladder refuses Random noise with the driver's wording: {:?}",
        random.msg
    );
    assert_eq!(refuse(|c| c.noise_type = 9).code, QNTT_INVALID_ARG);
    assert_eq!(refuse(|c| c.mode = 9).code, QNTT_INVALID_ARG);
    assert_eq!(refuse(|c| c.ss_start = -1).code, QNTT_INVALID_ARG);
    assert_eq!(
        refuse(|c| {
            c.penalty_rule = 1;
            c.penalty_value = f64::NAN;
        })
        .code,
        QNTT_INVALID_ARG
    );

    let neg = ExcitedSchedules {
        weights: &[],
        bond_dims: &[0],
        noise: &[-1.0],
        ss_noise: &[0.0],
    };
    assert!(
        excited_opts_abelian(&QnttExcitedOptions::defaults(), &neg, base.clone())
            .expect_err("negative noise")
            .msg
            .contains("noise")
    );
    let w = [0.4f64, 0.6];
    let bad_w = ExcitedSchedules {
        weights: &w,
        bond_dims: &[0],
        noise: &[0.0],
        ss_noise: &[0.0],
    };
    assert!(
        excited_opts_abelian(&QnttExcitedOptions::defaults(), &bad_w, base)
            .expect_err("increasing weights")
            .msg
            .contains("non-increasing")
    );
}

#[test]
#[allow(clippy::needless_range_loop)]
fn state_average_nroots_one_is_mals_bitwise_through_the_abi() {
    let (space, _t, _v, h) = h4_abelian_ham();

    for (tag, bond_dims, noise) in [
        ("noise-free", vec![0usize], vec![]),
        (
            "perturbative ramp",
            vec![4usize, 0],
            vec![1e-3f64, 1e-4, 0.0, 0.0],
        ),
    ] {
        let x0 = randn_abelian(space, 2, 20260905);

        let mut mo = QnttMalsOptions::defaults();
        mo.eps = 1e-10;
        mo.max_iter = 12;
        let mut ref_sweeps = vec![QnttMalsSweep::default(); mo.max_iter];
        let (mut ref_lambda, mut ref_n) = (0.0f64, 0usize);
        let mut ref_x: *mut _ = ptr::null_mut();
        assert_eq!(
            unsafe {
                qntt_f64_mals(
                    h,
                    x0,
                    &mo,
                    bond_dims.as_ptr(),
                    bond_dims.len(),
                    noise.as_ptr(),
                    noise.len(),
                    &mut ref_lambda,
                    &mut ref_x,
                    ref_sweeps.as_mut_ptr(),
                    ref_sweeps.len(),
                    &mut ref_n,
                )
            },
            QNTT_OK,
            "{tag}: mals failed: {}",
            last_error()
        );
        assert!(ref_n > 0, "{tag}: vacuous reference (no sweep ran)");

        let mut so = QnttStateAverageOptions::defaults();
        so.mals = mo;
        let mut lambdas = [0.0f64];
        let mut states: [*mut _; 1] = [ptr::null_mut()];
        let mut sweeps = vec![QnttSaSweep::default(); mo.max_iter];
        let mut sweep_energies = vec![f64::NAN; mo.max_iter];
        let mut sweep_discarded = vec![f64::NAN; mo.max_iter];
        let mut nsweeps = 0usize;
        assert_eq!(
            unsafe {
                qntt_f64_state_average_mals(
                    h,
                    x0,
                    1,
                    &so,
                    ptr::null(),
                    0,
                    bond_dims.as_ptr(),
                    bond_dims.len(),
                    noise.as_ptr(),
                    noise.len(),
                    lambdas.as_mut_ptr(),
                    states.as_mut_ptr(),
                    states.len(),
                    sweeps.as_mut_ptr(),
                    sweeps.len(),
                    &mut nsweeps,
                    sweep_energies.as_mut_ptr(),
                    sweep_discarded.as_mut_ptr(),
                )
            },
            QNTT_OK,
            "{tag}: state_average_mals failed: {}",
            last_error()
        );

        assert_eq!(
            lambdas[0].to_bits(),
            ref_lambda.to_bits(),
            "{tag}: λ differs ({} vs {ref_lambda})",
            lambdas[0]
        );
        assert_eq!(nsweeps, ref_n, "{tag}: sweep count differs");
        for i in 0..ref_n {
            let (a, b) = (&sweeps[i], &ref_sweeps[i]);
            assert_eq!(
                a.energy.to_bits(),
                b.energy.to_bits(),
                "{tag}: sweep {i} energy"
            );
            assert_eq!(
                a.discarded_weight.to_bits(),
                b.discarded_weight.to_bits(),
                "{tag}: sweep {i} discarded weight"
            );
            assert_eq!(
                a.max_bond_dim, b.max_bond_dim,
                "{tag}: sweep {i} max_bond_dim"
            );
            assert_eq!(a.bond_dim, b.bond_dim, "{tag}: sweep {i} bond_dim");
            assert_eq!(
                a.noise.to_bits(),
                b.noise.to_bits(),
                "{tag}: sweep {i} noise"
            );
            assert_eq!(
                sweep_energies[i].to_bits(),
                a.energy.to_bits(),
                "{tag}: sweep {i} per-root energy"
            );
            assert_eq!(
                sweep_discarded[i].to_bits(),
                a.discarded_weight.to_bits(),
                "{tag}: sweep {i} per-root discarded weight"
            );
        }
        assert_eq!(sweep_energies[nsweeps - 1].to_bits(), lambdas[0].to_bits());

        let (mut oa, mut ob) = (0isize, 0isize);
        assert_eq!(unsafe { qntt_f64_vec_ortho(states[0], &mut oa) }, QNTT_OK);
        assert_eq!(unsafe { qntt_f64_vec_ortho(ref_x, &mut ob) }, QNTT_OK);
        assert_eq!(oa, ob, "{tag}: orthogonality centre");
        assert_eq!(
            ranks_of!(qntt_f64_vec_ranks, states[0]),
            ranks_of!(qntt_f64_vec_ranks, ref_x),
            "{tag}: rank grids"
        );
        let (ca, cb) = (
            abelian_coeffs(&space, states[0]),
            abelian_coeffs(&space, ref_x),
        );
        assert_eq!(ca.len(), cb.len(), "{tag}: block traversal length");
        for (i, (p, q)) in ca.iter().zip(cb.iter()).enumerate() {
            assert_eq!(
                p.to_bits(),
                q.to_bits(),
                "{tag}: coefficient {i} ({p} vs {q})"
            );
        }

        for hv in [x0, ref_x, states[0]] {
            unsafe { qntt_f64_vec_free(hv) };
        }
    }
    unsafe { qntt_f64_ham_free(h) };
}

#[test]
#[allow(clippy::needless_range_loop)]
fn su2_state_average_nroots_one_is_mals_bitwise_through_the_abi() {
    let (norb, nelec, _ms2, t, v) = read_h4_through_abi();
    let mut h: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_ham_new(
                norb,
                nelec,
                0,
                t.as_ptr(),
                t.len(),
                v.as_ptr(),
                v.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK,
        "su2 ham_new failed: {}",
        last_error()
    );
    let flat = su2_capped_ranks(norb, nelec, 0, 4);
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_f64_vec_randn(norb, nelec, 0, flat.as_ptr(), flat.len(), 31, &mut x0) },
        QNTT_OK
    );

    let mut mo = QnttMalsOptions::defaults();
    mo.eps = 1e-10;
    mo.max_iter = 10;
    let bond_dims = [0usize];
    let noise = [1e-3f64, 1e-4, 0.0];
    let mut ref_sweeps = vec![QnttMalsSweep::default(); mo.max_iter];
    let (mut ref_lambda, mut ref_n) = (0.0f64, 0usize);
    let mut ref_x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_mals(
                h,
                x0,
                &mo,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                &mut ref_lambda,
                &mut ref_x,
                ref_sweeps.as_mut_ptr(),
                ref_sweeps.len(),
                &mut ref_n,
            )
        },
        QNTT_OK,
        "su2 mals failed: {}",
        last_error()
    );

    let mut so = QnttStateAverageOptions::defaults();
    so.mals = mo;
    let mut lambdas = [0.0f64];
    let mut states: [*mut _; 1] = [ptr::null_mut()];
    let mut sweeps = vec![QnttSaSweep::default(); mo.max_iter];
    let mut nsweeps = 0usize;
    assert_eq!(
        unsafe {
            qntt_su2_f64_state_average_mals(
                h,
                x0,
                1,
                &so,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                lambdas.as_mut_ptr(),
                states.as_mut_ptr(),
                states.len(),
                sweeps.as_mut_ptr(),
                sweeps.len(),
                &mut nsweeps,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        },
        QNTT_OK,
        "su2 state_average_mals failed: {}",
        last_error()
    );

    assert_eq!(lambdas[0].to_bits(), ref_lambda.to_bits());
    assert_eq!(nsweeps, ref_n);
    for i in 0..ref_n {
        assert_eq!(sweeps[i].energy.to_bits(), ref_sweeps[i].energy.to_bits());
        assert_eq!(sweeps[i].max_bond_dim, ref_sweeps[i].max_bond_dim);
        assert_eq!(sweeps[i].noise.to_bits(), ref_sweeps[i].noise.to_bits());
    }
    assert_eq!(
        ranks_of!(qntt_su2_f64_vec_ranks, states[0]),
        ranks_of!(qntt_su2_f64_vec_ranks, ref_x)
    );
    let mut diff: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_f64_vec_add(states[0], ref_x, -1, &mut diff) },
        QNTT_OK
    );
    let (mut nd, mut nx) = (0.0f64, 0.0f64);
    assert_eq!(unsafe { qntt_su2_f64_vec_norm(diff, &mut nd) }, QNTT_OK);
    assert_eq!(unsafe { qntt_su2_f64_vec_norm(ref_x, &mut nx) }, QNTT_OK);
    assert!(nx > 0.0, "vacuous: the reference state is zero");
    assert_eq!(
        nd, 0.0,
        "SU(2) states differ: ‖Δ‖ = {nd} against ‖x‖ = {nx}"
    );

    for hv in [x0, ref_x, states[0], diff] {
        unsafe { qntt_su2_f64_vec_free(hv) };
    }
    unsafe { qntt_su2_f64_ham_free(h) };
}

#[test]
#[allow(clippy::needless_range_loop)]
fn state_specific_no_lower_is_mals_bitwise_through_the_abi() {
    let (space, _t, _v, h) = h4_abelian_ham();

    for (tag, noise_type, noise) in [
        ("noise-free", 1, vec![]),
        ("perturbative", 1, vec![1e-3f64, 1e-4, 0.0]),
        ("random", 0, vec![1e-2f64, 1e-3, 0.0]),
    ] {
        let x0 = randn_abelian(space, 2, 4242);
        let bond_dims = [0usize];

        let mut mo = QnttMalsOptions::defaults();
        mo.eps = 1e-10;
        mo.max_iter = 10;
        mo.seed = 5;
        mo.noise_type = noise_type;
        let mut ref_sweeps = vec![QnttMalsSweep::default(); mo.max_iter];
        let (mut ref_lambda, mut ref_n) = (0.0f64, 0usize);
        let mut ref_x: *mut _ = ptr::null_mut();
        assert_eq!(
            unsafe {
                qntt_f64_mals(
                    h,
                    x0,
                    &mo,
                    bond_dims.as_ptr(),
                    bond_dims.len(),
                    noise.as_ptr(),
                    noise.len(),
                    &mut ref_lambda,
                    &mut ref_x,
                    ref_sweeps.as_mut_ptr(),
                    ref_sweeps.len(),
                    &mut ref_n,
                )
            },
            QNTT_OK,
            "{tag}: mals failed: {}",
            last_error()
        );

        let mut so = QnttStateSpecificOptions::defaults();
        so.mals = mo;
        let mut lambda = 0.0f64;
        let mut x: *mut _ = ptr::null_mut();
        let mut sweeps = vec![QnttSsSweep::default(); mo.max_iter];
        let mut nsweeps = 0usize;
        let mut overlaps = [f64::NAN];
        assert_eq!(
            unsafe {
                qntt_f64_state_specific_mals(
                    h,
                    x0,
                    ptr::null(),
                    0,
                    &so,
                    bond_dims.as_ptr(),
                    bond_dims.len(),
                    noise.as_ptr(),
                    noise.len(),
                    ptr::null(),
                    0,
                    &mut lambda,
                    &mut x,
                    sweeps.as_mut_ptr(),
                    sweeps.len(),
                    &mut nsweeps,
                    overlaps.as_mut_ptr(),
                )
            },
            QNTT_OK,
            "{tag}: state_specific_mals failed: {}",
            last_error()
        );

        assert_eq!(lambda.to_bits(), ref_lambda.to_bits(), "{tag}: λ");
        assert_eq!(nsweeps, ref_n, "{tag}: sweep count");
        for i in 0..ref_n {
            let (a, b) = (&sweeps[i], &ref_sweeps[i]);
            assert_eq!(
                a.energy.to_bits(),
                b.energy.to_bits(),
                "{tag}: sweep {i} energy"
            );
            assert_eq!(
                a.discarded_weight.to_bits(),
                b.discarded_weight.to_bits(),
                "{tag}: sweep {i} discarded weight"
            );
            assert_eq!(a.max_bond_dim, b.max_bond_dim, "{tag}: sweep {i} rank");
            assert_eq!(
                a.noise.to_bits(),
                b.noise.to_bits(),
                "{tag}: sweep {i} noise"
            );
            assert_eq!(
                a.penalty, 0.0,
                "{tag}: the delegate path records no penalty"
            );
            assert_eq!(
                a.penalized.to_bits(),
                a.energy.to_bits(),
                "{tag}: θ == λ with no penalty"
            );
            assert_eq!(
                a.numops, 0,
                "{tag}: the delegate path reports no applies (documented)"
            );
        }
        assert!(
            overlaps[0].is_nan(),
            "{tag}: the overlap buffer was written at nlower = 0"
        );

        let (ca, cb) = (abelian_coeffs(&space, x), abelian_coeffs(&space, ref_x));
        for (i, (p, q)) in ca.iter().zip(cb.iter()).enumerate() {
            assert_eq!(p.to_bits(), q.to_bits(), "{tag}: coefficient {i}");
        }

        for hv in [x0, ref_x, x] {
            unsafe { qntt_f64_vec_free(hv) };
        }
    }
    unsafe { qntt_f64_ham_free(h) };
}

#[test]
#[allow(clippy::needless_range_loop)]
fn three_root_ladder_matches_the_dense_spectrum_through_the_abi() {
    let data = read_fcidump::<f64>(&h4_path(), TwoBodySymmetry::Eightfold).unwrap();
    let ev = dense_levels(data.norb, 2, 2, &data.t, &data.v);
    assert_gapped(&ev, 3, 0.05, "H4 (2,2)");

    let (space, _t, _v, h) = h4_abelian_ham();
    let n = 3usize;

    let x0 = randn_abelian(space, 4, 2024);
    let so = sa_full_rank(1e-10, 40);
    let bond_dims = [0usize];
    let noise = [0.0f64];
    let mut lambdas = vec![0.0f64; n];
    let mut states: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut sweeps = vec![QnttSaSweep::default(); so.mals.max_iter];
    let mut sweep_energies = vec![f64::NAN; so.mals.max_iter * n];
    let mut sweep_discarded = vec![f64::NAN; so.mals.max_iter * n];
    let mut nsweeps = 0usize;
    assert_eq!(
        unsafe {
            qntt_f64_state_average_mals(
                h,
                x0,
                n,
                &so,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                lambdas.as_mut_ptr(),
                states.as_mut_ptr(),
                states.len(),
                sweeps.as_mut_ptr(),
                sweeps.len(),
                &mut nsweeps,
                sweep_energies.as_mut_ptr(),
                sweep_discarded.as_mut_ptr(),
            )
        },
        QNTT_OK,
        "state_average_mals failed: {}",
        last_error()
    );
    println!(
        "SA λ = {lambdas:?} in {nsweeps} sweep(s); dense = {:?}",
        &ev[..n]
    );
    assert!(nsweeps > 0, "vacuous: no state-average iteration ran");
    assert!(
        nsweeps <= so.mals.max_iter,
        "the sweep count must fit its own budget"
    );
    for i in 0..n {
        assert!(
            (lambdas[i] - ev[i]).abs() <= 1e-8,
            "SA root {i}: {} vs dense {} (Δ {:.3e})",
            lambdas[i],
            ev[i],
            (lambdas[i] - ev[i]).abs()
        );
    }
    assert!(
        lambdas.windows(2).all(|p| p[0] <= p[1]),
        "the default `homing = 0` labels by energy: {lambdas:?}"
    );
    let last = &sweeps[nsweeps - 1];
    let avg: f64 = lambdas.iter().sum::<f64>() / n as f64;
    assert!(
        (last.energy - avg).abs() <= 1e-12 * avg.abs().max(1.0),
        "the sweep's energy is the weighted average: {} vs {avg}",
        last.energy
    );
    for i in 0..n {
        assert_eq!(
            sweep_energies[(nsweeps - 1) * n + i].to_bits(),
            lambdas[i].to_bits(),
            "the last sweep's per-root energy {i} is not the returned λ"
        );
        assert!(
            sweep_discarded[(nsweeps - 1) * n + i] >= 0.0,
            "a per-root discarded weight must be a weight"
        );
    }
    assert!(last.overlap_dev.is_finite() && last.overlap_dev >= 0.0);
    assert!(
        last.max_bond_dim >= n,
        "the shared basis must host {n} roots"
    );
    assert_eq!(last.noise, 0.0, "the schedule ends noise-free");
    assert!(last.numops > 0, "vacuous: no local apply was recorded");

    for i in 0..n {
        let mut nrm = 0.0f64;
        assert_eq!(unsafe { qntt_f64_vec_norm(states[i], &mut nrm) }, QNTT_OK);
        assert!(
            (nrm - 1.0).abs() <= 1e-8,
            "SA state {i} is not unit ({nrm})"
        );
        let mut rq = 0.0f64;
        assert_eq!(
            unsafe { qntt_f64_ham_rayleigh(h, states[i], &mut rq) },
            QNTT_OK
        );
        assert!(
            (rq - lambdas[i]).abs() <= 1e-8,
            "SA state {i}: RQ {rq} vs λ {}",
            lambdas[i]
        );
        for j in (i + 1)..n {
            let mut o = 0.0f64;
            assert_eq!(
                unsafe { qntt_f64_vec_dot(states[i], states[j], &mut o) },
                QNTT_OK
            );
            assert!(o.abs() <= 1e-10, "SA Gram ({i},{j}) = {o}");
        }
    }

    let lower: [*const qntt_capi::handle::Handle<qntt::TtVector<f64>>; 1] = [states[0] as *const _];
    for (tag, le, le_len) in [
        ("computed at entry", vec![], 0usize),
        ("supplied", vec![lambdas[0]], 1usize),
    ] {
        let mut sso = QnttStateSpecificOptions::defaults();
        sso.mals.eps = 1e-10;
        sso.mals.max_iter = 40;
        let mut lam = 0.0f64;
        let mut x: *mut _ = ptr::null_mut();
        let mut sweeps = vec![QnttSsSweep::default(); sso.mals.max_iter];
        let mut nsw = 0usize;
        let mut ov = vec![f64::NAN; sso.mals.max_iter];
        assert_eq!(
            unsafe {
                qntt_f64_state_specific_mals(
                    h,
                    x0,
                    lower.as_ptr(),
                    1,
                    &sso,
                    bond_dims.as_ptr(),
                    bond_dims.len(),
                    noise.as_ptr(),
                    noise.len(),
                    le.as_ptr(),
                    le_len,
                    &mut lam,
                    &mut x,
                    sweeps.as_mut_ptr(),
                    sweeps.len(),
                    &mut nsw,
                    ov.as_mut_ptr(),
                )
            },
            QNTT_OK,
            "SS ({tag}) failed: {}",
            last_error()
        );
        assert!(nsw > 0, "SS ({tag}): vacuous, no iteration ran");
        println!(
            "SS ({tag}): λ = {lam}, dense E1 = {}, overlap = {}",
            ev[1],
            ov[nsw - 1]
        );
        assert!(
            (lam - ev[1]).abs() <= 1e-8,
            "SS ({tag}): λ {lam} vs dense E1 {} (Δ {:.3e})",
            ev[1],
            (lam - ev[1]).abs()
        );
        let last = &sweeps[nsw - 1];
        assert_eq!(
            last.energy.to_bits(),
            lam.to_bits(),
            "SS ({tag}): the history's last plain energy IS the returned λ"
        );
        assert!(
            last.penalized >= last.energy - 1e-12,
            "SS ({tag}): θ ≥ λ (θ = {}, λ = {})",
            last.penalized,
            last.energy
        );
        assert!(
            last.penalty >= 1.0,
            "SS ({tag}): the Auto rule floors the weight at 1 (got {})",
            last.penalty
        );
        assert!(
            ov[nsw - 1] <= 1e-6,
            "SS ({tag}): residual overlap {} at full rank",
            ov[nsw - 1]
        );
        assert_eq!(last.converged, 1, "SS ({tag}) did not converge");
        assert!(
            last.numops > 0,
            "SS ({tag}): vacuous, the sweep path records applies"
        );
        let mut o = 0.0f64;
        assert_eq!(unsafe { qntt_f64_vec_dot(states[0], x, &mut o) }, QNTT_OK);
        let mut nrm = 0.0f64;
        assert_eq!(unsafe { qntt_f64_vec_norm(x, &mut nrm) }, QNTT_OK);
        assert!(
            o.abs() / nrm <= 1e-6,
            "SS ({tag}): the polished root still overlaps the lower state by {}",
            o.abs() / nrm
        );
        unsafe { qntt_f64_vec_free(x) };
    }

    let mut eo = QnttExcitedOptions::defaults();
    eo.nroots = n;
    eo.rank = 0;
    eo.eps = 1e-10;
    eo.max_iter = 30;
    eo.keep_sa = 1;
    let ss_noise = [0.0f64];
    let mut energies = vec![0.0f64; n];
    let mut roots: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut sa_states: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut info = QnttExcitedInfo::default();
    let mut rows = vec![QnttExcitedRoot::default(); n];
    let mut gram = vec![f64::NAN; n * n];
    let mut gram_len = 0usize;
    assert_eq!(
        unsafe {
            qntt_f64_excited_states(
                h,
                x0,
                ptr::null(),
                0,
                &eo,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                ss_noise.as_ptr(),
                ss_noise.len(),
                energies.as_mut_ptr(),
                roots.as_mut_ptr(),
                roots.len(),
                &mut info,
                rows.as_mut_ptr(),
                ptr::null_mut(),
                gram.as_mut_ptr(),
                gram.len(),
                &mut gram_len,
                sa_states.as_mut_ptr(),
                sa_states.len(),
            )
        },
        QNTT_OK,
        "excited_states failed: {}",
        last_error()
    );

    println!(
        "ladder e = {energies:?}, licence = {}, swaps = {}, overlap_dev = {:.3e}",
        info.licence, info.swaps, info.overlap_dev
    );
    for i in 0..n {
        assert!(
            (energies[i] - ev[i]).abs() <= 1e-8,
            "ladder root {i}: {} vs dense {} (Δ {:.3e})",
            energies[i],
            ev[i],
            (energies[i] - ev[i]).abs()
        );
        assert!(
            (rows[i].sa_energy - ev[i]).abs() <= 1e-8,
            "the state-average stage misses level {i} at full rank ({})",
            rows[i].sa_energy
        );
        assert!(
            (rows[i].ss_energy - ev[i]).abs() <= 1e-8,
            "the state-specific stage misses level {i} ({})",
            rows[i].ss_energy
        );
        assert_eq!(rows[i].converged, 1, "root {i} was not certified converged");
        assert_eq!(rows[i].stalled, 0, "root {i} stalled above the gate");
        assert_eq!(rows[i].swapped, 0, "root {i} was flagged out of order");
        assert!(
            rows[i].gradnorm <= eo.grad_gate,
            "root {i} came back above the exact gate ({:.3e})",
            rows[i].gradnorm
        );
        assert!(rows[i].j_ss.is_finite() && rows[i].j_pol.is_finite());
        assert!(rows[i].passes <= eo.trim_passes);
        if i == 0 {
            assert_eq!(
                rows[i].penalty, 0.0,
                "root 0 has nothing to be penalized against"
            );
        } else {
            assert!(
                rows[i].penalty >= 1.0,
                "root {i} penalty {}",
                rows[i].penalty
            );
        }
    }
    assert_eq!(info.licence, 1, "the full-rank ladder must be licensed");
    assert_eq!(info.swaps, 0);
    assert_eq!(
        info.enriched, 0,
        "a rank-4 random start already hosts three roots"
    );
    assert_eq!(gram_len, n * n, "the Gram is (nlower + nroots)² row-major");
    let mut off = 0.0f64;
    for a in 0..n {
        assert!(
            (gram[a * n + a] - 1.0).abs() <= 1e-10,
            "Gram diagonal {a} = {}",
            gram[a * n + a]
        );
        for b in 0..n {
            if a != b {
                off = off.max(gram[a * n + b]);
            }
        }
    }
    assert!(
        (info.overlap_dev - off).abs() <= 1e-15,
        "`overlap_dev` must be the largest off-diagonal Gram entry ({} vs {off})",
        info.overlap_dev
    );
    assert!(info.overlap_dev <= eo.overlap_tol);
    for i in 0..n {
        assert!(!sa_states[i].is_null(), "keep_sa did not emit state {i}");
        let mut rq = 0.0f64;
        assert_eq!(
            unsafe { qntt_f64_ham_rayleigh(h, sa_states[i], &mut rq) },
            QNTT_OK
        );
        assert!(
            (rq - rows[i].sa_energy).abs() <= 1e-8,
            "sa_states[{i}] RQ {rq} vs recorded {}",
            rows[i].sa_energy
        );
    }

    let mut eo2 = eo;
    eo2.polish = 0;
    eo2.keep_sa = 0;
    let mut e2 = vec![0.0f64; n];
    let mut r2: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut rows2 = vec![QnttExcitedRoot::default(); n];
    assert_eq!(
        unsafe {
            qntt_f64_excited_states(
                h,
                x0,
                ptr::null(),
                0,
                &eo2,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                ss_noise.as_ptr(),
                ss_noise.len(),
                e2.as_mut_ptr(),
                r2.as_mut_ptr(),
                r2.len(),
                ptr::null_mut(),
                rows2.as_mut_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            )
        },
        QNTT_OK,
        "unpolished ladder failed: {}",
        last_error()
    );
    for i in 0..n {
        assert!(
            rows2[i].gradnorm.is_nan(),
            "root {i}: `polish = 0` has no critical point to certify, so the gate is NaN"
        );
        assert_eq!(rows2[i].stalled, 0);
        assert_eq!(rows2[i].passes, 0);
        assert!(
            (e2[i] - ev[i]).abs() <= 1e-8,
            "unpolished root {i}: {} vs {}",
            e2[i],
            ev[i]
        );
    }

    for hv in states
        .into_iter()
        .chain(roots)
        .chain(sa_states)
        .chain(r2)
        .chain([x0])
    {
        unsafe { qntt_f64_vec_free(hv) };
    }
    unsafe { qntt_f64_ham_free(h) };
}

#[test]
#[allow(clippy::needless_range_loop)]
fn c64_ladder_matches_the_real_spectrum_through_the_abi() {
    let data = read_fcidump::<f64>(&h4_path(), TwoBodySymmetry::Eightfold).unwrap();
    let d = data.norb;
    let ev = dense_levels(d, 2, 2, &data.t, &data.v);
    assert_gapped(&ev, 2, 0.05, "H4 (2,2), C64 arm");

    let phi: Vec<f64> = (0..d).map(|p| 0.37 * (p as f64 + 1.0)).collect();
    let ph = |a: f64| C64::new(a.cos(), a.sin());
    let mut tc = vec![C64::new(0.0, 0.0); d * d];
    for i in 0..d {
        for j in 0..d {
            tc[i * d + j] = ph(phi[j] - phi[i]) * data.t.get(&[i, j]);
        }
    }
    let mut vc = vec![C64::new(0.0, 0.0); d * d * d * d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    vc[((i * d + j) * d + k) * d + l] =
                        ph(phi[j] - phi[i] + phi[l] - phi[k]) * data.v.get(&[i, j, k, l]);
                }
            }
        }
    }

    let space = QnSpace::new(d, 2, 2);
    let mut h: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_c64_ham_new(
                space.d,
                space.nup,
                space.ndn,
                tc.as_ptr(),
                tc.len(),
                vc.as_ptr(),
                vc.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK,
        "c64 ham_new failed: {}",
        last_error()
    );

    let flat = flat_ranks(&space, &uniform_ranks(space, 4));
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_c64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                11,
                &mut x0,
            )
        },
        QNTT_OK
    );

    let mut y0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_c64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                12,
                &mut y0,
            )
        },
        QNTT_OK
    );
    let (mut xhy, mut yhx) = (C64::new(0.0, 0.0), C64::new(0.0, 0.0));
    assert_eq!(unsafe { qntt_c64_ham_xthy(x0, h, y0, &mut xhy) }, QNTT_OK);
    assert_eq!(unsafe { qntt_c64_ham_xthy(y0, h, x0, &mut yhx) }, QNTT_OK);
    assert!(
        (xhy - yhx.conj()).norm() <= 1e-10 * xhy.norm().max(1.0),
        "the phase-gauged fixture is not Hermitian: ⟨x,Hy⟩ = {xhy}, conj⟨y,Hx⟩ = {}",
        yhx.conj()
    );

    let n = 2usize;
    let mut eo = QnttExcitedOptions::defaults();
    eo.nroots = n;
    eo.eps = 1e-10;
    eo.max_iter = 30;
    let bond_dims = [0usize];
    let noise = [0.0f64];
    let ss_noise = [0.0f64];
    let mut energies = vec![0.0f64; n];
    let mut roots: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut info = QnttExcitedInfo::default();
    let mut rows = vec![QnttExcitedRoot::default(); n];
    let mut gram = vec![f64::NAN; n * n];
    let mut gram_len = 0usize;
    assert_eq!(
        unsafe {
            qntt_c64_excited_states(
                h,
                x0,
                ptr::null(),
                0,
                &eo,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                ss_noise.as_ptr(),
                ss_noise.len(),
                energies.as_mut_ptr(),
                roots.as_mut_ptr(),
                roots.len(),
                &mut info,
                rows.as_mut_ptr(),
                ptr::null_mut(),
                gram.as_mut_ptr(),
                gram.len(),
                &mut gram_len,
                ptr::null_mut(),
                0,
            )
        },
        QNTT_OK,
        "c64 excited_states failed: {}",
        last_error()
    );

    for i in 0..n {
        assert!(
            (energies[i] - ev[i]).abs() <= 1e-8,
            "c64 root {i}: {} vs the real spectrum {} (Δ {:.3e})",
            energies[i],
            ev[i],
            (energies[i] - ev[i]).abs()
        );
        assert_eq!(rows[i].converged, 1);
    }
    assert_eq!(info.licence, 1, "the full-rank C64 ladder must be licensed");
    assert_eq!(gram_len, n * n);
    for a in 0..n {
        assert!(
            (gram[a * n + a] - 1.0).abs() <= 1e-10,
            "the Gram is |⟨a,b⟩|, real"
        );
    }

    let mut worst = 1.0f64;
    for r in &roots {
        let (mut bil, mut ses) = (C64::new(0.0, 0.0), C64::new(0.0, 0.0));
        assert_eq!(
            unsafe { qntt_c64_vec_dot_bilinear(*r, *r, &mut bil) },
            QNTT_OK
        );
        assert_eq!(unsafe { qntt_c64_vec_dot(*r, *r, &mut ses) }, QNTT_OK);
        worst = worst.min(bil.norm() / ses.norm());
    }
    println!("c64 arm: min |⟨x̄,x⟩|/⟨x,x⟩ over the roots = {worst:.6}");
    assert!(
        worst < 0.99,
        "the C64 arm is vacuous: every root is real up to a global phase (ρ = {worst})"
    );

    let lower: [*const qntt_capi::handle::Handle<qntt::TtVector<C64>>; 1] = [roots[0] as *const _];
    let mut sso = QnttStateSpecificOptions::defaults();
    sso.mals.eps = 1e-10;
    sso.mals.max_iter = 40;
    let mut sslam = 0.0f64;
    let mut ssx: *mut _ = ptr::null_mut();
    let mut sssweeps = vec![QnttSsSweep::default(); sso.mals.max_iter];
    let mut ssn = 0usize;
    let mut ssov = vec![f64::NAN; sso.mals.max_iter];
    assert_eq!(
        unsafe {
            qntt_c64_state_specific_mals(
                h,
                x0,
                lower.as_ptr(),
                1,
                &sso,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                ptr::null(),
                0,
                &mut sslam,
                &mut ssx,
                sssweeps.as_mut_ptr(),
                sssweeps.len(),
                &mut ssn,
                ssov.as_mut_ptr(),
            )
        },
        QNTT_OK,
        "c64 state_specific_mals failed: {}",
        last_error()
    );
    assert!(ssn > 0, "c64 SS: vacuous, no iteration ran");
    println!(
        "c64 SS: λ = {sslam}, dense E1 = {}, overlap = {}",
        ev[1],
        ssov[ssn - 1]
    );
    assert!(
        (sslam - ev[1]).abs() <= 1e-8,
        "c64 SS: λ {sslam} vs the real spectrum's E1 {} (Δ {:.3e})",
        ev[1],
        (sslam - ev[1]).abs()
    );
    assert!(
        ssov[ssn - 1] <= 1e-6,
        "c64 SS: residual overlap {} at full rank",
        ssov[ssn - 1]
    );
    let (mut sd, mut sn) = (C64::new(0.0, 0.0), 0.0f64);
    assert_eq!(
        unsafe { qntt_c64_vec_dot(lower[0] as *const _, ssx, &mut sd) },
        QNTT_OK
    );
    assert_eq!(unsafe { qntt_c64_vec_norm(ssx, &mut sn) }, QNTT_OK);
    assert!(
        sd.norm() / sn <= 1e-6,
        "c64 SS: the polished state still overlaps its lower rung by {}",
        sd.norm() / sn
    );
    unsafe { qntt_c64_vec_free(ssx) };

    for hv in roots.into_iter().chain([x0, y0]) {
        unsafe { qntt_c64_vec_free(hv) };
    }
    unsafe { qntt_c64_ham_free(h) };
}

#[test]
#[allow(clippy::needless_range_loop)]
fn su2_ladder_matches_the_dense_singlets_through_the_abi() {
    let data = read_fcidump::<f64>(&h4_path(), TwoBodySymmetry::Eightfold).unwrap();
    let s0 = singlet_levels(data.norb, 2, 2, &data.t, &data.v);
    assert_gapped(&s0, 3, 0.05, "H4 S = 0");

    let (norb, nelec, _ms2, t, v) = read_h4_through_abi();
    let mut h: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_ham_new(
                norb,
                nelec,
                0,
                t.as_ptr(),
                t.len(),
                v.as_ptr(),
                v.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK,
        "su2 ham_new failed: {}",
        last_error()
    );
    let flat = su2_capped_ranks(norb, nelec, 0, 64);
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_f64_vec_randn(norb, nelec, 0, flat.as_ptr(), flat.len(), 17, &mut x0) },
        QNTT_OK
    );

    let n = 3usize;
    let bond_dims = [0usize];
    let noise = [0.0f64];
    let ss_noise = [0.0f64];

    let so = sa_full_rank(1e-10, 40);
    let mut lambdas = vec![0.0f64; n];
    let mut states: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut nsweeps = 0usize;
    assert_eq!(
        unsafe {
            qntt_su2_f64_state_average_mals(
                h,
                x0,
                n,
                &so,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                lambdas.as_mut_ptr(),
                states.as_mut_ptr(),
                states.len(),
                ptr::null_mut(),
                0,
                &mut nsweeps,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        },
        QNTT_OK,
        "su2 state_average_mals failed: {}",
        last_error()
    );
    println!("su2 SA λ = {lambdas:?} (S = 0 levels {:?})", &s0[..n]);
    let mut got = lambdas.clone();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 0..n {
        assert!(
            (got[i] - s0[i]).abs() <= 1e-8,
            "su2 SA level {i}: {} vs singlet {} (Δ {:.3e})",
            got[i],
            s0[i],
            (got[i] - s0[i]).abs()
        );
    }

    let low = (0..n)
        .min_by(|a, b| lambdas[*a].partial_cmp(&lambdas[*b]).unwrap())
        .unwrap();
    let su2_lower: [*const qntt_capi::handle::Handle<qntt::su2::vector::Su2TtVector<f64>>; 1] =
        [states[low] as *const _];
    let mut sso = QnttStateSpecificOptions::defaults();
    sso.mals.eps = 1e-10;
    sso.mals.max_iter = 40;
    let mut sslam = 0.0f64;
    let mut ssx: *mut _ = ptr::null_mut();
    let mut sssweeps = vec![QnttSsSweep::default(); sso.mals.max_iter];
    let mut ssn = 0usize;
    let mut ssov = vec![f64::NAN; sso.mals.max_iter];
    assert_eq!(
        unsafe {
            qntt_su2_f64_state_specific_mals(
                h,
                x0,
                su2_lower.as_ptr(),
                1,
                &sso,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                ptr::null(),
                0,
                &mut sslam,
                &mut ssx,
                sssweeps.as_mut_ptr(),
                sssweeps.len(),
                &mut ssn,
                ssov.as_mut_ptr(),
            )
        },
        QNTT_OK,
        "su2 state_specific_mals failed: {}",
        last_error()
    );
    assert!(ssn > 0, "su2 SS: vacuous, no iteration ran");
    let which = s0.iter().position(|l| (l - sslam).abs() <= 1e-8);
    println!(
        "su2 SS: λ = {sslam} (S = 0 level {which:?}), overlap = {}",
        ssov[ssn - 1]
    );
    assert!(
        which.is_some(),
        "su2 SS: λ = {sslam} is not an S = 0 level of the fixture"
    );
    assert!(
        sslam > lambdas[low] - 1e-8,
        "su2 SS: the penalized run fell below the state it was pushed off ({sslam} vs {})",
        lambdas[low]
    );
    assert!(
        ssov[ssn - 1] <= 1e-6,
        "su2 SS: residual overlap {} at full rank",
        ssov[ssn - 1]
    );
    assert!(
        sssweeps[ssn - 1].penalty >= 1.0,
        "su2 SS: the Auto rule floors the weight at 1 (got {})",
        sssweeps[ssn - 1].penalty
    );
    let mut ssd = 0.0f64;
    assert_eq!(
        unsafe { qntt_su2_f64_vec_dot(su2_lower[0] as *const _, ssx, &mut ssd) },
        QNTT_OK
    );
    let mut ssnrm = 0.0f64;
    assert_eq!(unsafe { qntt_su2_f64_vec_norm(ssx, &mut ssnrm) }, QNTT_OK);
    assert!(
        ssd.abs() / ssnrm <= 1e-6,
        "su2 SS: the polished state still overlaps its lower rung by {}",
        ssd.abs() / ssnrm
    );
    unsafe { qntt_su2_f64_vec_free(ssx) };

    let mut eo = QnttExcitedOptions::defaults();
    eo.nroots = n;
    eo.eps = 1e-10;
    eo.max_iter = 30;
    let mut energies = vec![0.0f64; n];
    let mut roots: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut info = QnttExcitedInfo::default();
    let mut rows = vec![QnttExcitedRoot::default(); n];
    let mut gram = vec![f64::NAN; n * n];
    let mut gram_len = 0usize;
    assert_eq!(
        unsafe {
            qntt_su2_f64_excited_states(
                h,
                x0,
                ptr::null(),
                0,
                &eo,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                ss_noise.as_ptr(),
                ss_noise.len(),
                energies.as_mut_ptr(),
                roots.as_mut_ptr(),
                roots.len(),
                &mut info,
                rows.as_mut_ptr(),
                ptr::null_mut(),
                gram.as_mut_ptr(),
                gram.len(),
                &mut gram_len,
                ptr::null_mut(),
                0,
            )
        },
        QNTT_OK,
        "su2 excited_states failed: {}",
        last_error()
    );

    println!(
        "su2 ladder e = {energies:?}, licence = {}, overlap_dev = {:.3e}",
        info.licence, info.overlap_dev
    );
    for (i, e) in energies.iter().enumerate() {
        let which = s0.iter().position(|l| (l - e).abs() <= 1e-8);
        assert!(
            which.is_some(),
            "su2 root {i} = {e} is not an S = 0 level of the fixture"
        );
        println!("  su2 root {i} = S = 0 level {}", which.unwrap());
    }
    let mut got = energies.clone();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 0..n {
        assert!(
            (got[i] - s0[i]).abs() <= 1e-8,
            "su2 ladder level {i}: {} vs singlet {}",
            got[i],
            s0[i]
        );
    }
    assert_eq!(
        info.licence, 1,
        "the full-rank SU(2) ladder must be licensed"
    );
    assert!(info.overlap_dev <= eo.overlap_tol);
    for i in 0..n {
        assert_eq!(rows[i].converged, 1, "su2 root {i} not converged");
        assert!((gram[i * n + i] - 1.0).abs() <= 1e-10);
    }

    for hv in states.into_iter().chain(roots).chain([x0]) {
        unsafe { qntt_su2_f64_vec_free(hv) };
    }
    unsafe { qntt_su2_f64_ham_free(h) };
}

#[test]
fn multistate_error_paths() {
    let (space, _t, _v, h) = h4_abelian_ham();
    let x0 = randn_abelian(space, 3, 5);
    let bond_dims = [0usize];
    let noise = [0.0f64];
    let n = 3usize;
    let so = sa_full_rank(1e-6, 2);

    let run_sa = |nroots: usize,
                  opts: &QnttStateAverageOptions,
                  weights: &[f64],
                  bd: &[usize],
                  states_cap: usize,
                  hh: *const qntt_capi::handle::Handle<SparseHamiltonian<f64>>|
     -> (
        i32,
        Vec<*mut qntt_capi::handle::Handle<qntt::TtVector<f64>>>,
    ) {
        let mut lambdas = vec![0.0f64; nroots.max(1)];
        let mut states: Vec<*mut _> = vec![ptr::null_mut(); nroots.max(1)];
        let code = unsafe {
            qntt_f64_state_average_mals(
                hh,
                x0,
                nroots,
                opts,
                weights.as_ptr(),
                weights.len(),
                bd.as_ptr(),
                bd.len(),
                noise.as_ptr(),
                noise.len(),
                lambdas.as_mut_ptr(),
                states.as_mut_ptr(),
                states_cap,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        (code, states)
    };

    let (code, states) = run_sa(n, &so, &[], &bond_dims, n - 1, h);
    assert_eq!(code, QNTT_INVALID_ARG, "short out_states must be refused");
    assert!(
        last_error().contains("states"),
        "the message must name the buffer: {}",
        last_error()
    );
    assert!(
        states.iter().all(|p| p.is_null()),
        "a refused call must emit no handles"
    );

    assert_eq!(run_sa(0, &so, &[], &bond_dims, 1, h).0, QNTT_INVALID_ARG);
    let (code, _) = run_sa(n, &so, &[0.5, 0.5], &bond_dims, n, h);
    assert_eq!(code, QNTT_INVALID_ARG);
    assert!(last_error().contains("weights"), "got {}", last_error());
    let (code, _) = run_sa(2, &so, &[0.4, 0.6], &bond_dims, 2, h);
    assert_eq!(code, QNTT_INVALID_ARG);
    assert!(
        last_error().contains("non-increasing"),
        "got {}",
        last_error()
    );
    let (code, _) = run_sa(n, &so, &[], &[], n, h);
    assert_eq!(code, QNTT_INVALID_ARG);
    assert!(last_error().contains("bond_dims"), "got {}", last_error());
    let mut rnd = so;
    rnd.mals.noise_type = 0;
    assert_eq!(run_sa(n, &rnd, &[], &bond_dims, n, h).0, QNTT_INVALID_ARG);
    let mut bogus = so;
    bogus.mals.noise_type = 9;
    assert_eq!(run_sa(n, &bogus, &[], &bond_dims, n, h).0, QNTT_INVALID_ARG);

    let (code, _) = run_sa(n, &so, &[], &bond_dims, n, ptr::null());
    assert_eq!(code, QNTT_NULL_ARG);
    let mut lambdas = vec![0.0f64; n];
    assert_eq!(
        unsafe {
            qntt_f64_state_average_mals(
                h,
                x0,
                n,
                &so,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                lambdas.as_mut_ptr(),
                ptr::null_mut(),
                n,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        },
        QNTT_NULL_ARG,
        "out_states is required"
    );

    let su2_flat = su2_capped_ranks(4, 4, 0, 4);
    let mut su2_x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_vec_randn(4, 4, 0, su2_flat.as_ptr(), su2_flat.len(), 3, &mut su2_x)
        },
        QNTT_OK
    );
    let sso = QnttStateSpecificOptions::defaults();
    let wrong: [*const qntt_capi::handle::Handle<qntt::TtVector<f64>>; 1] = [su2_x as *const _];
    let (mut lam, mut out) = (0.0f64, ptr::null_mut());
    let code = unsafe {
        qntt_f64_state_specific_mals(
            h,
            x0,
            wrong.as_ptr(),
            1,
            &sso,
            bond_dims.as_ptr(),
            bond_dims.len(),
            noise.as_ptr(),
            noise.len(),
            ptr::null(),
            0,
            &mut lam,
            &mut out,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(
        code, QNTT_INVALID_ARG,
        "an SU(2) handle is not an abelian lower state"
    );
    assert!(last_error().contains("mismatch"), "got {}", last_error());
    assert!(out.is_null());

    let eo = QnttExcitedOptions::defaults();
    let mut e = vec![0.0f64; 2];
    let mut roots: Vec<*mut _> = vec![ptr::null_mut(); 2];
    let code = unsafe {
        qntt_c64_excited_states(
            h as *const _,
            x0 as *const _,
            ptr::null(),
            0,
            &eo,
            ptr::null(),
            0,
            bond_dims.as_ptr(),
            bond_dims.len(),
            noise.as_ptr(),
            noise.len(),
            noise.as_ptr(),
            noise.len(),
            e.as_mut_ptr(),
            roots.as_mut_ptr(),
            roots.len(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            0,
        )
    };
    assert_eq!(code, QNTT_INVALID_ARG);
    assert!(last_error().contains("mismatch"), "got {}", last_error());

    let mut eo = QnttExcitedOptions::defaults();
    eo.nroots = 2;
    eo.eps = 1e-8;
    eo.max_iter = 6;
    let mut energies = vec![0.0f64; 2];
    let mut roots: Vec<*mut _> = vec![ptr::null_mut(); 2];
    let mut gram = vec![f64::NAN; 1];
    let mut gram_len = 0usize;
    let ss_noise = [0.0f64];
    let code = unsafe {
        qntt_f64_excited_states(
            h,
            x0,
            ptr::null(),
            0,
            &eo,
            ptr::null(),
            0,
            bond_dims.as_ptr(),
            bond_dims.len(),
            noise.as_ptr(),
            noise.len(),
            ss_noise.as_ptr(),
            ss_noise.len(),
            energies.as_mut_ptr(),
            roots.as_mut_ptr(),
            roots.len(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            gram.as_mut_ptr(),
            gram.len(),
            &mut gram_len,
            ptr::null_mut(),
            0,
        )
    };
    assert_eq!(
        code, QNTT_INVALID_ARG,
        "a short Gram buffer is an error, not a truncation"
    );
    assert!(
        roots.iter().all(|p| p.is_null()),
        "handles must be emitted only after every fallible copy-out has succeeded"
    );

    let mut eo2 = eo;
    eo2.keep_sa = 1;
    let mut sa_states: Vec<*mut _> = vec![ptr::null_mut(); 1];
    let code = unsafe {
        qntt_f64_excited_states(
            h,
            x0,
            ptr::null(),
            0,
            &eo2,
            ptr::null(),
            0,
            bond_dims.as_ptr(),
            bond_dims.len(),
            noise.as_ptr(),
            noise.len(),
            ss_noise.as_ptr(),
            ss_noise.len(),
            energies.as_mut_ptr(),
            roots.as_mut_ptr(),
            roots.len(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            sa_states.as_mut_ptr(),
            sa_states.len(),
        )
    };
    assert_eq!(code, QNTT_INVALID_ARG);
    assert!(roots.iter().all(|p| p.is_null()));

    unsafe { qntt_su2_f64_vec_free(su2_x) };
    unsafe { qntt_f64_vec_free(x0) };
    unsafe { qntt_f64_ham_free(h) };
}

#[test]
fn a_determinant_start_is_refused_by_the_state_average_and_enriched_by_the_ladder() {
    let (space, _t, _v, h) = h4_abelian_ham();
    let occ: Vec<i32> = (0..space.d).map(|k| (k < space.nup) as i32).collect();
    let mut hf: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_tt_state(occ.as_ptr(), occ.as_ptr(), space.d, &mut hf) },
        QNTT_OK
    );

    let so = sa_full_rank(1e-6, 3);
    let bond_dims = [0usize];
    let noise = [0.0f64];
    let mut lambdas = [0.0f64; 2];
    let mut states: [*mut _; 2] = [ptr::null_mut(); 2];
    let code = unsafe {
        qntt_f64_state_average_mals(
            h,
            hf,
            2,
            &so,
            ptr::null(),
            0,
            bond_dims.as_ptr(),
            bond_dims.len(),
            noise.as_ptr(),
            noise.len(),
            lambdas.as_mut_ptr(),
            states.as_mut_ptr(),
            states.len(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(
        code, QNTT_PANIC,
        "hazard 5 is a core precondition, not an ABI check"
    );
    assert!(last_error().starts_with("panic:"), "got {}", last_error());
    assert!(
        last_error().contains("two-site local space"),
        "the core's pointed message must survive the boundary: {}",
        last_error()
    );
    assert!(states.iter().all(|p| p.is_null()));

    let mut eo = QnttExcitedOptions::defaults();
    eo.nroots = 2;
    eo.eps = 1e-10;
    eo.max_iter = 30;
    let ramp = [1e-3f64, 1e-4, 1e-5, 0.0];
    let ss_noise = [0.0f64];
    let mut energies = vec![0.0f64; 2];
    let mut roots: Vec<*mut _> = vec![ptr::null_mut(); 2];
    let mut info = QnttExcitedInfo::default();
    assert_eq!(
        unsafe {
            qntt_f64_excited_states(
                h,
                hf,
                ptr::null(),
                0,
                &eo,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                ramp.as_ptr(),
                ramp.len(),
                ss_noise.as_ptr(),
                ss_noise.len(),
                energies.as_mut_ptr(),
                roots.as_mut_ptr(),
                roots.len(),
                &mut info,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            )
        },
        QNTT_OK,
        "the ladder must enrich a determinant start: {}",
        last_error()
    );
    assert_eq!(
        info.enriched, 1,
        "the enrichment iteration must be recorded"
    );
    let data = read_fcidump::<f64>(&h4_path(), TwoBodySymmetry::Eightfold).unwrap();
    let ev = dense_levels(space.d, space.nup, space.ndn, &data.t, &data.v);
    println!("enriched ladder e = {energies:?} vs dense {:?}", &ev[..2]);
    assert!(
        (energies[0] - ev[0]).abs() <= 1e-8,
        "the enriched ladder's ground state: {} vs {}",
        energies[0],
        ev[0]
    );

    for hv in roots.into_iter().chain([hf]) {
        unsafe { qntt_f64_vec_free(hv) };
    }
    unsafe { qntt_f64_ham_free(h) };
}

fn assert_sa_row_mirrors(got: &QnttSaSweep, want: &qntt::solvers::multistate::SaSweep, s: usize) {
    for (what, g, w) in [
        ("energy", got.energy, want.energy),
        ("max_dev", got.max_dev, want.max_dev),
        ("overlap_dev", got.overlap_dev, want.overlap_dev),
        (
            "discarded_weight",
            got.discarded_weight,
            want.discarded_weight,
        ),
        ("noise", got.noise, want.noise),
    ] {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "sweep {s}: `{what}` is {g}, the core recorded {w}"
        );
    }
    assert_eq!(
        (got.max_bond_dim, got.bond_dim, got.numops),
        (want.max_bond_dim, want.bond_dim, want.numops),
        "sweep {s}: the integer columns do not mirror the core row"
    );
}

fn assert_ss_row_mirrors(got: &QnttSsSweep, want: &qntt::solvers::ssmals::SsSweep, s: usize) {
    for (what, g, w) in [
        ("energy", got.energy, want.energy),
        ("penalized", got.penalized, want.penalized),
        (
            "discarded_weight",
            got.discarded_weight,
            want.discarded_weight,
        ),
        ("noise", got.noise, want.noise),
        ("penalty", got.penalty, want.penalty),
    ] {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "sweep {s}: `{what}` is {g}, the core recorded {w}"
        );
    }
    assert_eq!(
        (
            got.max_bond_dim,
            got.bond_dim,
            got.swaps,
            got.numops,
            got.skipped
        ),
        (
            want.max_bond_dim,
            want.bond_dim,
            want.swaps,
            want.numops,
            want.skipped
        ),
        "sweep {s}: the integer columns do not mirror the core row"
    );
    assert_eq!(
        got.converged != 0,
        want.converged,
        "sweep {s}: `converged` does not mirror the core row"
    );
}

#[test]
#[allow(clippy::needless_range_loop)]
fn state_average_records_mirror_the_core_history_and_the_buffers_truncate() {
    let (space, _t, _v, h) = h4_abelian_ham();
    let n = 3usize;
    let x0 = randn_abelian(space, 4, 2024);
    let so = sa_full_rank(1e-10, 40);
    let bond_dims = [0usize];
    let noise = [0.0f64];

    let mut lambdas = vec![0.0f64; n];
    let mut states: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut sweeps = vec![QnttSaSweep::default(); so.mals.max_iter];
    let mut energies = vec![f64::NAN; so.mals.max_iter * n];
    let mut discarded = vec![f64::NAN; so.mals.max_iter * n];
    let mut nsweeps = 0usize;
    assert_eq!(
        unsafe {
            qntt_f64_state_average_mals(
                h,
                x0,
                n,
                &so,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                lambdas.as_mut_ptr(),
                states.as_mut_ptr(),
                states.len(),
                sweeps.as_mut_ptr(),
                sweeps.len(),
                &mut nsweeps,
                energies.as_mut_ptr(),
                discarded.as_mut_ptr(),
            )
        },
        QNTT_OK,
        "state_average_mals failed: {}",
        last_error()
    );
    assert!(
        nsweeps >= 2,
        "vacuous: the truncation arm below needs a run of at least two sweeps (got {nsweeps})"
    );

    let core_h = unsafe {
        qntt_capi::handle::borrow::<SparseHamiltonian<f64>>(h as *const _).expect("live handle")
    };
    let core_x0 =
        unsafe { qntt_capi::handle::borrow::<TtVector<f64>>(x0 as *const _).expect("live handle") };
    let core_opts = so
        .to_core(n, &[], &bond_dims, &noise)
        .expect("the same option block the ABI built");
    let (hist, core_states) =
        qntt::solvers::multistate::state_average_mals_with_history(core_h, core_x0, n, &core_opts);
    assert_eq!(
        hist.sweeps.len(),
        nsweeps,
        "the ABI and the core driver did not take the same trajectory"
    );
    for i in 0..n {
        assert_eq!(
            lambdas[i].to_bits(),
            hist.lambdas[i].to_bits(),
            "λ {i} is not the core's"
        );
    }
    for s in 0..nsweeps {
        assert_sa_row_mirrors(&sweeps[s], &hist.sweeps[s], s);
        for i in 0..n {
            assert_eq!(
                energies[s * n + i].to_bits(),
                hist.sweeps[s].energies[i].to_bits(),
                "companion energy [{s}·{n} + {i}] is not the core's per-root energy"
            );
            assert_eq!(
                discarded[s * n + i].to_bits(),
                hist.sweeps[s].discarded_weights[i].to_bits(),
                "companion discarded weight [{s}·{n} + {i}] is not the core's"
            );
        }
    }
    assert!(
        (0..nsweeps).any(|s| sweeps[s].max_dev.to_bits() != sweeps[s].overlap_dev.to_bits()),
        "vacuous: `max_dev` and `overlap_dev` coincided on every sweep"
    );
    assert!(
        (0..nsweeps).any(
            |s| (0..n).any(|i| energies[s * n + i].to_bits() != discarded[s * n + i].to_bits())
        ),
        "vacuous: the two companion buffers coincided everywhere"
    );
    drop(core_states);

    const SENT: f64 = -98765.4321;
    let mut lam2 = vec![0.0f64; n];
    let mut st2: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut sw2 = vec![QnttSaSweep::default(); 1];
    let mut e2 = vec![SENT; n + 1];
    let mut d2 = vec![SENT; n + 1];
    let mut nsw2 = 0usize;
    assert_eq!(
        unsafe {
            qntt_f64_state_average_mals(
                h,
                x0,
                n,
                &so,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                lam2.as_mut_ptr(),
                st2.as_mut_ptr(),
                st2.len(),
                sw2.as_mut_ptr(),
                1,
                &mut nsw2,
                e2.as_mut_ptr(),
                d2.as_mut_ptr(),
            )
        },
        QNTT_OK,
        "the truncated run failed: {}",
        last_error()
    );
    assert_eq!(
        nsw2, nsweeps,
        "a truncated buffer must still report the FULL sweep count"
    );
    assert_sa_row_mirrors(&sw2[0], &hist.sweeps[0], 0);
    for i in 0..n {
        assert_eq!(
            e2[i].to_bits(),
            energies[i].to_bits(),
            "the truncated companion must hold the FIRST sweep's root {i}"
        );
        assert_eq!(d2[i].to_bits(), discarded[i].to_bits());
    }
    assert_eq!(
        e2[n].to_bits(),
        SENT.to_bits(),
        "the energy companion wrote past `sweeps_cap · nroots`"
    );
    assert_eq!(
        d2[n].to_bits(),
        SENT.to_bits(),
        "the discarded-weight companion wrote past `sweeps_cap · nroots`"
    );

    for hv in states.into_iter().chain(st2).chain([x0]) {
        unsafe { qntt_f64_vec_free(hv) };
    }
    unsafe { qntt_f64_ham_free(h) };
}

#[test]
#[allow(clippy::needless_range_loop)]
fn excited_root_rows_mirror_the_core_ladder_records() {
    let (space, _t, _v, h) = h4_abelian_ham();
    let n = 3usize;
    let x0 = randn_abelian(space, 4, 2024);
    let bond_dims = [0usize];
    let noise = [0.0f64];
    let ss_noise = [0.0f64];
    let mut eo = QnttExcitedOptions::defaults();
    eo.nroots = n;
    eo.rank = 0;
    eo.eps = 1e-10;
    eo.max_iter = 30;

    let mut e = vec![0.0f64; n];
    let mut roots: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut info = QnttExcitedInfo::default();
    let mut rows = vec![QnttExcitedRoot::default(); n];
    assert_eq!(
        unsafe {
            qntt_f64_excited_states(
                h,
                x0,
                ptr::null(),
                0,
                &eo,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                ss_noise.as_ptr(),
                ss_noise.len(),
                e.as_mut_ptr(),
                roots.as_mut_ptr(),
                roots.len(),
                &mut info,
                rows.as_mut_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            )
        },
        QNTT_OK,
        "excited_states failed: {}",
        last_error()
    );

    let core_h = unsafe {
        qntt_capi::handle::borrow::<SparseHamiltonian<f64>>(h as *const _).expect("live handle")
    };
    let core_x0 =
        unsafe { qntt_capi::handle::borrow::<TtVector<f64>>(x0 as *const _).expect("live handle") };
    let sched = ExcitedSchedules {
        weights: &[],
        bond_dims: &bond_dims,
        noise: &noise,
        ss_noise: &ss_noise,
    };
    let core_opts = excited_opts_abelian(
        &eo,
        &sched,
        qntt::solvers::excited::ExcitedOpts::new(&space),
    )
    .expect("the same option block the ABI built");
    let (ce, _croots, cinfo) =
        qntt::solvers::excited::excited_states(core_h, core_x0, &[], &core_opts);

    let mut distinct_energies = 0usize;
    let mut distinct_j = 0usize;
    for i in 0..n {
        assert_eq!(
            e[i].to_bits(),
            ce[i].to_bits(),
            "energy {i} is not the core's"
        );
        for (what, g, w) in [
            ("sa_energy", rows[i].sa_energy, cinfo.sa_history.lambdas[i]),
            ("ss_energy", rows[i].ss_energy, cinfo.ss_energies[i]),
            ("j_ss", rows[i].j_ss, cinfo.j_ss[i]),
            ("j_pol", rows[i].j_pol, cinfo.j_pol[i]),
            ("gradnorm", rows[i].gradnorm, cinfo.gradnorms[i]),
            ("penalty", rows[i].penalty, cinfo.penalties[i]),
        ] {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "root {i}: `{what}` is {g}, the core recorded {w}"
            );
        }
        assert_eq!(rows[i].passes, cinfo.passes[i]);
        assert_eq!(rows[i].stalled != 0, cinfo.stalled[i]);
        assert_eq!(rows[i].converged != 0, cinfo.converged[i]);
        assert_eq!(
            rows[i].swapped != 0,
            cinfo.swapped.contains(&i),
            "root {i}: the per-root `swapped` flag is the core's `swapped` membership (nlower = 0)"
        );
        if rows[i].sa_energy.to_bits() != rows[i].ss_energy.to_bits() {
            distinct_energies += 1;
        }
        if rows[i].j_ss.to_bits() != rows[i].j_pol.to_bits() {
            distinct_j += 1;
        }
    }
    assert!(
        distinct_energies > 0,
        "vacuous: `sa_energy` and `ss_energy` were bit-identical on every root"
    );
    assert!(
        distinct_j > 0,
        "vacuous: `j_ss` and `j_pol` were bit-identical on every root"
    );
    assert_eq!(info.licence != 0, cinfo.licence);
    assert_eq!(info.enriched != 0, cinfo.enriched);
    assert_eq!(info.swaps, cinfo.swaps);
    assert_eq!(info.overlap_dev.to_bits(), cinfo.overlap_dev.to_bits());

    for hv in roots.into_iter().chain([x0]) {
        unsafe { qntt_f64_vec_free(hv) };
    }
    unsafe { qntt_f64_ham_free(h) };
}

#[test]
#[allow(clippy::needless_range_loop)]
fn state_specific_against_two_lower_states_records_them_in_the_callers_order() {
    let data = read_fcidump::<f64>(&h4_path(), TwoBodySymmetry::Eightfold).unwrap();
    let ev = dense_levels(data.norb, 2, 2, &data.t, &data.v);
    assert_gapped(&ev, 3, 0.05, "H4 (2,2)");

    let (space, _t, _v, h) = h4_abelian_ham();
    let n = 3usize;
    let x0 = randn_abelian(space, 4, 2024);
    let so = sa_full_rank(1e-10, 40);
    let bond_dims = [0usize];
    let noise = [0.0f64];

    let mut lambdas = vec![0.0f64; n];
    let mut states: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut nsweeps = 0usize;
    assert_eq!(
        unsafe {
            qntt_f64_state_average_mals(
                h,
                x0,
                n,
                &so,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                lambdas.as_mut_ptr(),
                states.as_mut_ptr(),
                states.len(),
                ptr::null_mut(),
                0,
                &mut nsweeps,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        },
        QNTT_OK,
        "state_average_mals failed: {}",
        last_error()
    );

    let lower: [*const qntt_capi::handle::Handle<qntt::TtVector<f64>>; 2] =
        [states[0] as *const _, states[1] as *const _];
    let mut sso = QnttStateSpecificOptions::defaults();
    sso.mals.eps = 1e-10;
    sso.mals.max_iter = 40;
    let mut lam = 0.0f64;
    let mut x: *mut _ = ptr::null_mut();
    let mut sweeps = vec![QnttSsSweep::default(); sso.mals.max_iter];
    let mut nsw = 0usize;
    let mut ov = vec![f64::NAN; sso.mals.max_iter * 2];
    assert_eq!(
        unsafe {
            qntt_f64_state_specific_mals(
                h,
                x0,
                lower.as_ptr(),
                2,
                &sso,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                ptr::null(),
                0,
                &mut lam,
                &mut x,
                sweeps.as_mut_ptr(),
                sweeps.len(),
                &mut nsw,
                ov.as_mut_ptr(),
            )
        },
        QNTT_OK,
        "SS against two lower states failed: {}",
        last_error()
    );
    println!(
        "SS(nlower = 2): λ = {lam}, dense E2 = {}, last overlaps = {:?}",
        ev[2],
        &ov[(nsw - 1) * 2..nsw * 2]
    );
    assert!(
        (lam - ev[2]).abs() <= 1e-8,
        "SS(nlower = 2): λ {lam} vs dense E2 {} (Δ {:.3e})",
        ev[2],
        (lam - ev[2]).abs()
    );
    assert!(
        nsw >= 2,
        "vacuous: a transposed write needs ≥ 2 sweeps to differ"
    );

    let core_h = unsafe {
        qntt_capi::handle::borrow::<SparseHamiltonian<f64>>(h as *const _).expect("live handle")
    };
    let core_x0 =
        unsafe { qntt_capi::handle::borrow::<TtVector<f64>>(x0 as *const _).expect("live handle") };
    let core_lower: Vec<TtVector<f64>> = (0..2)
        .map(|j| unsafe {
            qntt_capi::handle::borrow::<TtVector<f64>>(lower[j])
                .expect("live handle")
                .clone()
        })
        .collect();
    let core_opts = sso
        .to_core(2, &[], &bond_dims, &noise)
        .expect("the same option block the ABI built");
    let (chist, _cx) = qntt::solvers::ssmals::state_specific_mals_with_history(
        core_h,
        core_x0,
        &core_lower,
        &core_opts,
    );
    assert_eq!(
        chist.sweeps.len(),
        nsw,
        "the ABI and the core driver did not take the same trajectory"
    );
    let mut distinct = 0usize;
    for s in 0..nsw {
        for j in 0..2 {
            assert_eq!(
                ov[s * 2 + j].to_bits(),
                chist.sweeps[s].overlaps[j].to_bits(),
                "overlap [{s}·2 + {j}] is not the core's — the stride or the lower order is wrong"
            );
        }
        if ov[s * 2].to_bits() != ov[s * 2 + 1].to_bits() {
            distinct += 1;
        }
    }
    assert!(
        distinct > 0,
        "vacuous: the two lower states recorded the same overlap on every sweep, so a \
         transposed or reversed write would be indistinguishable"
    );

    let mut nx = 0.0f64;
    assert_eq!(unsafe { qntt_f64_vec_norm(x, &mut nx) }, QNTT_OK);
    for j in 0..2 {
        let mut d = 0.0f64;
        assert_eq!(
            unsafe { qntt_f64_vec_dot(lower[j] as *const _, x, &mut d) },
            QNTT_OK
        );
        let mut nl = 0.0f64;
        assert_eq!(
            unsafe { qntt_f64_vec_norm(lower[j] as *const _, &mut nl) },
            QNTT_OK
        );
        let want = d.abs() / (nl * nx);
        assert!(
            (ov[(nsw - 1) * 2 + j] - want).abs() <= 1e-8,
            "the last recorded overlap against lower {j} is {}, the states give {want}",
            ov[(nsw - 1) * 2 + j]
        );
    }

    const SENT: f64 = -98765.4321;
    let mut lam2 = 0.0f64;
    let mut x2: *mut _ = ptr::null_mut();
    let mut sw2 = vec![QnttSsSweep::default(); 1];
    let mut nsw2 = 0usize;
    let mut ov2 = vec![SENT; 2 * 2];
    assert_eq!(
        unsafe {
            qntt_f64_state_specific_mals(
                h,
                x0,
                lower.as_ptr(),
                2,
                &sso,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                ptr::null(),
                0,
                &mut lam2,
                &mut x2,
                sw2.as_mut_ptr(),
                1,
                &mut nsw2,
                ov2.as_mut_ptr(),
            )
        },
        QNTT_OK,
        "the truncated SS run failed: {}",
        last_error()
    );
    assert_eq!(
        nsw2, nsw,
        "a truncated buffer must still report the FULL sweep count"
    );
    assert_ss_row_mirrors(&sw2[0], &chist.sweeps[0], 0);
    for j in 0..2 {
        assert_eq!(
            ov2[j].to_bits(),
            ov[j].to_bits(),
            "the truncated overlap companion must hold the FIRST sweep's lower {j}"
        );
    }
    for k in 2..4 {
        assert_eq!(
            ov2[k].to_bits(),
            SENT.to_bits(),
            "the overlap companion wrote past `sweeps_cap · nlower` (slot {k})"
        );
    }

    for hv in states.into_iter().chain([x, x2, x0]) {
        unsafe { qntt_f64_vec_free(hv) };
    }
    unsafe { qntt_f64_ham_free(h) };
}

#[test]
#[allow(clippy::needless_range_loop)]
fn su2_c64_ladder_and_the_two_sweep_entry_points_through_the_abi() {
    let data = read_fcidump::<f64>(&h4_path(), TwoBodySymmetry::Eightfold).unwrap();
    let d = data.norb;
    let s0 = singlet_levels(d, 2, 2, &data.t, &data.v);
    assert_gapped(&s0, 3, 0.05, "H4 S = 0, su2_c64 arm");

    let phi: Vec<f64> = (0..d).map(|p| 0.37 * (p as f64 + 1.0)).collect();
    let ph = |a: f64| C64::new(a.cos(), a.sin());
    let mut tc = vec![C64::new(0.0, 0.0); d * d];
    for i in 0..d {
        for j in 0..d {
            tc[i * d + j] = ph(phi[j] - phi[i]) * data.t.get(&[i, j]);
        }
    }
    let mut vc = vec![C64::new(0.0, 0.0); d * d * d * d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    vc[((i * d + j) * d + k) * d + l] =
                        ph(phi[j] - phi[i] + phi[l] - phi[k]) * data.v.get(&[i, j, k, l]);
                }
            }
        }
    }

    let (norb, nelec, _ms2, _t, _v) = read_h4_through_abi();
    assert_eq!(norb, d);
    let mut h: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_c64_ham_new(
                norb,
                nelec,
                0,
                tc.as_ptr(),
                tc.len(),
                vc.as_ptr(),
                vc.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK,
        "su2_c64 ham_new failed: {}",
        last_error()
    );
    let flat = su2_capped_ranks(norb, nelec, 0, 64);
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_c64_vec_randn(norb, nelec, 0, flat.as_ptr(), flat.len(), 23, &mut x0) },
        QNTT_OK
    );

    let n = 3usize;
    let bond_dims = [0usize];
    let noise = [0.0f64];
    let ss_noise = [0.0f64];

    let so = sa_full_rank(1e-10, 40);
    let mut lambdas = vec![0.0f64; n];
    let mut states: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut nsweeps = 0usize;
    assert_eq!(
        unsafe {
            qntt_su2_c64_state_average_mals(
                h,
                x0,
                n,
                &so,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                lambdas.as_mut_ptr(),
                states.as_mut_ptr(),
                states.len(),
                ptr::null_mut(),
                0,
                &mut nsweeps,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        },
        QNTT_OK,
        "su2_c64 state_average_mals failed: {}",
        last_error()
    );
    println!("su2_c64 SA λ = {lambdas:?} (S = 0 levels {:?})", &s0[..n]);
    let mut got = lambdas.clone();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 0..n {
        assert!(
            (got[i] - s0[i]).abs() <= 1e-8,
            "su2_c64 SA level {i}: {} vs singlet {} (Δ {:.3e})",
            got[i],
            s0[i],
            (got[i] - s0[i]).abs()
        );
    }

    let low = (0..n)
        .min_by(|a, b| lambdas[*a].partial_cmp(&lambdas[*b]).unwrap())
        .unwrap();
    let lower: [*const qntt_capi::handle::Handle<qntt::su2::vector::Su2TtVector<C64>>; 1] =
        [states[low] as *const _];
    let mut sso = QnttStateSpecificOptions::defaults();
    sso.mals.eps = 1e-10;
    sso.mals.max_iter = 40;
    let mut sslam = 0.0f64;
    let mut ssx: *mut _ = ptr::null_mut();
    let mut sssweeps = vec![QnttSsSweep::default(); sso.mals.max_iter];
    let mut ssn = 0usize;
    let mut ssov = vec![f64::NAN; sso.mals.max_iter];
    assert_eq!(
        unsafe {
            qntt_su2_c64_state_specific_mals(
                h,
                x0,
                lower.as_ptr(),
                1,
                &sso,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                ptr::null(),
                0,
                &mut sslam,
                &mut ssx,
                sssweeps.as_mut_ptr(),
                sssweeps.len(),
                &mut ssn,
                ssov.as_mut_ptr(),
            )
        },
        QNTT_OK,
        "su2_c64 state_specific_mals failed: {}",
        last_error()
    );
    assert!(ssn > 0, "su2_c64 SS: vacuous, no iteration ran");
    let which = s0.iter().position(|l| (l - sslam).abs() <= 1e-8);
    println!(
        "su2_c64 SS: λ = {sslam} (S = 0 level {which:?}), overlap = {}",
        ssov[ssn - 1]
    );
    assert!(
        which.is_some(),
        "su2_c64 SS: λ = {sslam} is not an S = 0 level of the fixture"
    );
    assert!(
        sslam > lambdas[low] - 1e-8,
        "su2_c64 SS: the penalized run fell below the state it was pushed off"
    );
    assert!(
        ssov[ssn - 1] <= 1e-6,
        "su2_c64 SS: residual overlap {} at full rank",
        ssov[ssn - 1]
    );

    let mut eo = QnttExcitedOptions::defaults();
    eo.nroots = n;
    eo.eps = 1e-10;
    eo.max_iter = 30;
    let mut energies = vec![0.0f64; n];
    let mut roots: Vec<*mut _> = vec![ptr::null_mut(); n];
    let mut info = QnttExcitedInfo::default();
    let mut rows = vec![QnttExcitedRoot::default(); n];
    let mut gram = vec![f64::NAN; n * n];
    let mut gram_len = 0usize;
    assert_eq!(
        unsafe {
            qntt_su2_c64_excited_states(
                h,
                x0,
                ptr::null(),
                0,
                &eo,
                ptr::null(),
                0,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                ss_noise.as_ptr(),
                ss_noise.len(),
                energies.as_mut_ptr(),
                roots.as_mut_ptr(),
                roots.len(),
                &mut info,
                rows.as_mut_ptr(),
                ptr::null_mut(),
                gram.as_mut_ptr(),
                gram.len(),
                &mut gram_len,
                ptr::null_mut(),
                0,
            )
        },
        QNTT_OK,
        "su2_c64 excited_states failed: {}",
        last_error()
    );
    println!(
        "su2_c64 ladder e = {energies:?}, licence = {}, overlap_dev = {:.3e}",
        info.licence, info.overlap_dev
    );
    let mut got = energies.clone();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 0..n {
        assert!(
            (got[i] - s0[i]).abs() <= 1e-8,
            "su2_c64 ladder level {i}: {} vs singlet {}",
            got[i],
            s0[i]
        );
        assert_eq!(rows[i].converged, 1, "su2_c64 root {i} not converged");
        assert!((gram[i * n + i] - 1.0).abs() <= 1e-10);
    }
    assert_eq!(
        info.licence, 1,
        "the full-rank su2_c64 ladder must be licensed"
    );
    assert_eq!(gram_len, n * n);

    let mut worst = 1.0f64;
    for r in &roots {
        let (mut bil, mut ses) = (C64::new(0.0, 0.0), C64::new(0.0, 0.0));
        assert_eq!(
            unsafe { qntt_su2_c64_vec_dot_bilinear(*r, *r, &mut bil) },
            QNTT_OK
        );
        assert_eq!(unsafe { qntt_su2_c64_vec_dot(*r, *r, &mut ses) }, QNTT_OK);
        worst = worst.min(bil.norm() / ses.norm());
    }
    println!("su2_c64 arm: min |⟨x̄,x⟩|/⟨x,x⟩ over the roots = {worst:.6}");
    assert!(
        worst < 0.99,
        "the su2_c64 arm is vacuous: every root is real up to a global phase (ρ = {worst})"
    );

    for hv in states.into_iter().chain(roots).chain([ssx, x0]) {
        unsafe { qntt_su2_c64_vec_free(hv) };
    }
    unsafe { qntt_su2_c64_ham_free(h) };
}
