// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::Family;
use qntt::core::tangent::{
    TtTangent, min_schmidt, project_irrep, tangent_basis, tangent_basis_shifted, tangent_dim,
    tangent_dim_grids,
};
use qntt::core::weingarten::{RpaMetricBlock, Terms, weingarten_assemble_cached};
use qntt::pointgroup::{Irrep, PointGroup};
use qntt::round::RoundMode;
use qntt::solvers::hess::{
    ChannelGsOpts, ChannelOpts, ChannelSet, ChannelStatus, ConeOpts, CurvatureReg, IrrepSel,
    ModeCheckOpts, RpaKernel, RpaMetric, RpaModeKind, RpaReport, SectorCtx, SectorOpts,
    SectorReadout, SectorSpectrum, SympPrecond, SymplecticSpectrum, channel_response,
    effham_apply_shifted, effham_gram_shifted, hessian_grams, normal_residual,
    symplectic_eigenvalues, weingarten_operator,
};
use qntt::solvers::riemannian::{normalize_mut, tangent_bases};
use qntt::vector::symmetry_defect;
use qntt::{
    CoeffTensor, QnSpace, Scalar, SparseHamiltonian, TtVector, dot, max_ranks, rayleigh_quotient,
    xt_h_y,
};
use qntt_testkit::dense::{dense_eigen, dense_hamiltonian_matrix, dets_for};
use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::sector_util::*;

fn d6_reference() -> (
    QnSpace,
    CoeffTensor<f64>,
    CoeffTensor<f64>,
    SparseHamiltonian<f64>,
    TtVector<f64>,
) {
    let pg = d6_pg();
    let space = QnSpace::with_point_group(D6, 2, 2, pg, Irrep::IDENTITY);
    let (t, v) = d6_tv(pg);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9002);
    let mut x: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
    x.round_mut(1e-8, qntt::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x);
    x.check();
    (space, t, v, ham, x)
}

fn dense_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn rayleigh_tilde(ham: &SparseHamiltonian<f64>, y: &TtVector<f64>) -> f64 {
    xt_h_y(y, ham, y).re_s() / (2.0 * dot(y, y).re_s())
}

fn corrected_form(
    ham: &SparseHamiltonian<f64>,
    x: &TtVector<f64>,
    g_vec: &TtVector<f64>,
    xi: &TtTangent<f64>,
    h: f64,
) -> (f64, f64, f64) {
    let cp = xi.retract_hosvd(h);
    let cm = xi.retract_hosvd(-h);
    let m0 = rayleigh_tilde(ham, x);
    let mp = rayleigh_tilde(ham, &cp);
    let mm = rayleigh_tilde(ham, &cm);
    let q_uncorr = (mp - 2.0 * m0 + mm) / (h * h);
    let gp = dot(g_vec, &cp).re_s();
    let gm = dot(g_vec, &cm).re_s();
    let g0 = dot(g_vec, x).re_s();
    let corr = (gp + gm - 2.0 * g0) / (h * h);
    (q_uncorr - corr, q_uncorr, corr)
}

fn corrected_form_richardson(
    ham: &SparseHamiltonian<f64>,
    x: &TtVector<f64>,
    g_vec: &TtVector<f64>,
    xi: &TtTangent<f64>,
    h: f64,
) -> f64 {
    let (q1, _, _) = corrected_form(ham, x, g_vec, xi, h);
    let (q2, _, _) = corrected_form(ham, x, g_vec, xi, h / 2.0);
    (4.0 * q2 - q1) / 3.0
}

fn shifted_unit_tangents(
    sctx: &SectorCtx<'_, '_, f64>,
    space_out: QnSpace,
    seed: u64,
    n: usize,
) -> Vec<TtTangent<f64>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..(8 * n) {
        if out.len() == n {
            break;
        }
        let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
        let mut xi = sctx.project(&y);
        let nrm = xi.norm();
        if nrm <= 1e-8 {
            continue;
        }
        xi.scale_mut(1.0 / nrm);
        out.push(xi);
    }
    assert_eq!(
        out.len(),
        n,
        "shifted_unit_tangents: only {} of {n} draws survived P^phi at seed {seed:#x} — the \
         channel's numerical span is far below its structural dim and every arm looping over \
         these would be testing fewer directions than it says",
        out.len()
    );
    out
}

#[test]
fn tsr_relabeled_frames_are_structurally_legal() {
    let pg = fx4_pg();
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
    let (t, v) = fx4_tv(pg);
    let (x, _e0, _lv) = exact_typed_ground(space, &t, &v);
    let (bl, br) = tangent_bases(&x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7401);

    for p in 0..pg.order() {
        let phi = Irrep::new(p);
        let space_out = space.with_target(space.target.fuse(phi));
        let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
        let xi = TtTangent::project_shifted(&bl, &br, phi, &y);

        let emb = xi.to_vector();
        assert_eq!(emb.space, space_out, "phi {p}: iota^phi leaves space_out");
        assert!(emb.check(), "phi {p}: iota^phi is not a legal train");

        let vranks = xi.rank_col();
        let last = qntt::Sector::with_irrep(F_NUP, F_NDN, space_out.target);
        assert_eq!(
            vranks[D4][last], 1,
            "phi {p}: V' must carry rank 1 at the shifted boundary sector"
        );
        for (s, &r) in vranks[D4].iter() {
            assert!(r == 0 || s == last, "phi {p}: stray bond-d rank at {s:?}");
        }
        let first = qntt::Sector::with_irrep(0, 0, phi);
        assert_eq!(vranks[0][first], 1, "phi {p}: V' bond-0 label");
        for (s, &r) in vranks[0].iter() {
            assert!(r == 0 || s == first, "phi {p}: stray bond-0 rank at {s:?}");
        }
    }
}

#[test]
fn tsr_identity_phi_is_the_shipped_tangent() {
    let pg = fx4_pg();
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
    let (t, v) = fx4_tv(pg);
    let (x, _e0, _lv) = exact_typed_ground(space, &t, &v);
    let (bl, br) = tangent_bases(&x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7501);
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);

    let a = TtTangent::project(&bl, &br, &dir);
    let b = TtTangent::project_shifted(&bl, &br, Irrep::IDENTITY, &dir);
    assert!(
        a.norm() > 1e-6,
        "anti-vacuity: the reference projection is 0"
    );
    assert_eq!(b.phi(), Irrep::IDENTITY);
    assert_eq!(b.rank_col(), a.rank());
    assert_eq!(
        a.sub(&b).norm(),
        0.0,
        "project_shifted(IDENTITY) must be bit-identical to project"
    );
    assert_eq!(
        a.to_vector().to_dense(),
        b.to_vector().to_dense(),
        "to_vector at IDENTITY must be bit-identical"
    );
    assert_eq!(
        tangent_dim_grids(&space, &bl.ranks, Irrep::IDENTITY),
        tangent_dim(&bl)
    );
}

#[test]
fn tsr_projector_is_orthogonal_and_irrep_exact() {
    let pg = fx4_pg();
    let full = {
        let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
        let (t, v) = fx4_tv(pg);
        let (ham, x) = typed_minimizer(space, &t, &v, 0x7503, 1, 2);
        ("full-rank (2,1)", space, t, v, ham, x)
    };
    let trunc = {
        let (space, t, v, ham, x) = truncated_reference();
        ("truncated (2,2)", space, t, v, ham, x)
    };

    for (label, space, t, v, ham, x) in [full, trunc] {
        let (bl, br) = tangent_bases(&x);
        let lambda = rayleigh_quotient(&ham, &x).re_s();
        println!(
            "tsr projector [{label}]: sigma_min = {:e}, lambda = {lambda}, ||r_perp|| = {:e}",
            min_schmidt(&bl, &br),
            residual_norm(&ham, &x)
        );

        let channels = live_channels(&space, &bl.ranks);
        let live = channels.len();
        println!("tsr projector [{label}]: live channels {channels:?}");
        for phi in channels {
            let p = phi.index();
            let space_out = space.with_target(space.target.fuse(phi));
            let basis = tangent_basis_shifted(&bl, &br, phi);
            assert!(
                !basis.is_empty(),
                "[{label}] phi {p}: tangent_dim_grids says the channel is live but \
                 tangent_basis_shifted returned nothing"
            );

            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7503_0000 + p as u64);
            let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
            let z: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
            let py = TtTangent::project_shifted(&bl, &br, phi, &y);
            let pz = TtTangent::project_shifted(&bl, &br, phi, &z);
            let n = py.norm();
            assert!(
                n > 1e-6,
                "[{label}] phi {p}: anti-vacuity, the probe projected to 0"
            );
            let pp = TtTangent::project_shifted(&bl, &br, phi, &py.to_vector());
            assert!(
                py.sub(&pp).norm() <= 1e-11 * n,
                "[{label}] phi {p}: P^phi is not idempotent"
            );
            let iy = py.to_vector();
            assert!(
                (iy.norm() - n).abs() <= 1e-11 * n,
                "[{label}] phi {p}: iota is not an isometry"
            );

            let lhs = dot(&iy, &z);
            let rhs = dot(&y, &pz.to_vector());
            let sc = lhs.abs().max(rhs.abs()).max(1e-3);
            assert!(
                (lhs - rhs).abs() <= 1e-9 * sc,
                "[{label}] phi {p}: P^phi is oblique, {lhs} vs {rhs}"
            );

            let (gamma, defects) = symmetry_defect(&iy, pg);
            let worst = defects.iter().cloned().fold(0.0f64, f64::max);
            println!("tsr projector [{label}] phi={p}: gamma={gamma:?} defect={worst:e}");
            assert_eq!(
                gamma, space_out.target,
                "[{label}] phi {p}: iota(xi) does not live in Gamma0 + phi"
            );
            assert!(
                worst <= 1e-7,
                "[{label}] phi {p}: defect {worst:e} above the sqrt(eps) floor"
            );

            let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
            let g = effham_gram_shifted(&h_out, &bl, &br, phi, lambda, &basis);
            let m = basis.len();
            assert_eq!(g.len(), m * m, "[{label}] phi {p}: gram shape");
            let embedded: Vec<TtVector<f64>> = basis.iter().map(|b| b.to_vector()).collect();
            let happ: Vec<TtVector<f64>> = embedded.iter().map(|w| h_out.apply(w)).collect();
            let (mut worst_g, mut worst_sym, mut scale) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..m {
                for j in 0..m {
                    let want =
                        dot(&embedded[i], &happ[j]) - lambda * dot(&embedded[i], &embedded[j]);
                    worst_g = worst_g.max((g[i * m + j] - want).abs());
                    worst_sym = worst_sym.max((g[i * m + j] - g[j * m + i]).abs());
                    scale = scale.max(want.abs());
                }
            }
            println!(
                "tsr gram [{label}] phi={p}: dim {m}, |gram - dense| = {worst_g:e}, \
                 asymmetry = {worst_sym:e}, scale = {scale:e}"
            );
            assert!(
                scale > 1e-3,
                "[{label}] phi {p}: anti-vacuity, A^phi is identically 0"
            );
            assert!(
                worst_g <= 1e-9 * scale.max(1.0),
                "[{label}] phi {p}: effham_gram_shifted != the dense assembly ({worst_g:e})"
            );
            assert!(
                worst_sym <= 1e-10 * scale.max(1.0),
                "[{label}] phi {p}: A^phi is not symmetric ({worst_sym:e})"
            );

            let a0 = effham_apply_shifted(&h_out, &bl, &br, phi, lambda, &basis[0]);
            let quad = basis[0].dot(&a0);
            assert!(
                (quad - g[0]).abs() <= 1e-9 * scale.max(1.0),
                "[{label}] phi {p}: apply and gram disagree on the diagonal, {quad} vs {}",
                g[0]
            );
        }
        assert!(
            live >= 2,
            "[{label}] anti-vacuity: only {live} live channels on fx4_pg (measured inventory is 2)"
        );
    }
}

#[test]
fn tsr_sector_ctx_is_a_consistent_wrapper() {
    let (space, t, v, ham, x) = truncated_reference();
    let (bl, br) = tangent_bases(&x);
    let lambda = rayleigh_quotient(&ham, &x).re_s();
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_shifting_channel(&space, &bl.ranks);
    println!(
        "tsr wrapper: phi = {phi:?}, live = {:?}",
        live_channels(&space, &bl.ranks)
    );
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);

    assert_eq!(sctx.phi(), phi);
    assert_eq!(sctx.space_out(), space_out);
    assert!(
        (sctx.lambda() - lambda).abs() <= 1e-12 * lambda.abs().max(1.0),
        "SectorCtx::lambda must be the Rayleigh quotient at x: {} vs {lambda}",
        sctx.lambda()
    );
    let (cl, cr) = sctx.bases();
    assert_eq!(
        cl.space, space,
        "SectorCtx::bases must return the ORIGINAL frames (x's space), not the shifted ones"
    );
    assert_eq!(cr.space, space);
    assert_eq!(cl.to_dense(), bl.to_dense(), "left frame differs");
    assert_eq!(cr.to_dense(), br.to_dense(), "right frame differs");

    let dim = tangent_dim_grids(&space, &bl.ranks, phi);
    assert!(dim > 0, "anti-vacuity: the channel is empty");
    assert_eq!(sctx.dim(), dim, "SectorCtx::dim != tangent_dim_grids");
    let basis = sctx.basis();
    let want_basis = tangent_basis_shifted(&bl, &br, phi);
    assert_eq!(
        basis.len(),
        want_basis.len(),
        "SectorCtx::basis != tangent_basis_shifted"
    );
    for (j, (a, b)) in basis.iter().zip(want_basis.iter()).enumerate() {
        assert!(
            a.sub(b).norm() <= 1e-12,
            "SectorCtx::basis vector {j} differs from tangent_basis_shifted"
        );
    }

    let g_free = effham_gram_shifted(&h_out, &bl, &br, phi, lambda, &basis);
    let scale = g_free.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
    assert!(scale > 1e-3, "anti-vacuity: A^phi is identically zero");

    let g_exact = sctx.a_gram_exact(&basis);
    assert_eq!(g_exact.len(), g_free.len());
    for (j, (a, b)) in g_exact.iter().zip(g_free.iter()).enumerate() {
        assert_eq!(
            a, b,
            "a_gram_exact entry {j} must BE effham_gram_shifted at the SAME frames and lambda, \
             bit for bit — a re-derived context is invisible to any tolerance"
        );
    }

    let g_ctx = sctx.a_gram(&basis);
    assert_eq!(g_ctx.len(), g_free.len());
    let mut worst_g = 0.0f64;
    for (j, (a, b)) in g_ctx.iter().zip(g_free.iter()).enumerate() {
        worst_g = worst_g.max((a - b).abs());
        assert!(
            (a - b).abs() <= 1e-11 * scale,
            "a_gram (FUSED) entry {j}: {a} vs effham_gram_shifted {b} (scale {scale:e})"
        );
    }

    let a_free = effham_apply_shifted(&h_out, &bl, &br, phi, lambda, &basis[0]);
    let a_exact = sctx.a_apply_exact(&basis[0]);
    assert_eq!(
        a_exact.num_coeffs(),
        a_free.num_coeffs(),
        "a_apply_exact coefficient count"
    );
    for i in 0..a_free.num_coeffs() {
        assert_eq!(
            a_exact.coeff(i),
            a_free.coeff(i),
            "a_apply_exact coeff {i} must BE effham_apply_shifted at the bound frames, phi and \
             lambda, bit for bit"
        );
    }

    let a_ctx = sctx.a_apply(&basis[0]);
    let rel_a = a_ctx.sub(&a_free).norm() / a_free.norm().max(1.0);
    println!(
        "tsr wrapper: fused-vs-exact worst gram |da| = {worst_g:e} (scale {scale:e}), apply rel \
         = {rel_a:e}"
    );
    assert!(
        a_free.norm() > 1e-3,
        "anti-vacuity: A^phi annihilated basis[0] ({:e})",
        a_free.norm()
    );
    assert!(
        rel_a <= 1e-11,
        "a_apply (FUSED) vs effham_apply_shifted: relative deviation {rel_a:e} exceeds the \
         reassociation class (t8i model, 1e-11)"
    );

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x75C7);
    let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
    let p_ctx = sctx.project(&y);
    let p_free = TtTangent::project_shifted(&bl, &br, phi, &y);
    assert!(
        p_free.norm() > 1e-6,
        "anti-vacuity: the probe projected to 0"
    );
    assert!(
        p_ctx.sub(&p_free).norm() <= 1e-12 * p_free.norm(),
        "SectorCtx::project != project_shifted"
    );
}

#[test]
#[should_panic(expected = "not a cross-sector channel")]
fn tsr_identity_channel_is_refused_at_construction() {
    let (space, t, v, ham, x) = truncated_reference();
    let ctx = weingarten_operator(&ham, &x);
    let h_out = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let _ = SectorCtx::new(&ctx, &h_out, Irrep::IDENTITY);
}

#[test]
#[should_panic(expected = "shifted tangent")]
fn tsr_unshifted_hess_apply_refuses_a_shifted_tangent() {
    let (space, _t, _v, ham, x) = truncated_reference();
    let (bl, br) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_shifting_channel(&space, &bl.ranks);
    let space_out = space.with_target(space.target.fuse(phi));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x75F1);
    let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
    let xi = TtTangent::project_shifted(&bl, &br, phi, &y);
    assert!(
        xi.norm() > 1e-8,
        "fixture: the shifted direction must be nonzero for the refusal to mean anything"
    );
    let _ = ctx.hess_apply(&xi);
}

#[test]
#[should_panic(expected = "Phase B")]
fn tsr_rpa_readout_is_refused_in_sector_lowest_tda() {
    let (space, t, v, ham, x) = truncated_reference();
    let (bl, _br) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_shifting_channel(&space, &bl.ranks);
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    let opts = SectorOpts {
        readout: SectorReadout::Rpa,
        ..SectorOpts::default()
    };
    let _ = sctx.sector_lowest_tda(1, &opts);
}

#[test]
#[should_panic(expected = "sector_lowest")]
fn tsr_tda_readout_is_refused_in_sector_lowest() {
    let (space, t, v, ham, x) = truncated_reference();
    let (bl, _br) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_shifting_channel(&space, &bl.ranks);
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    let opts = SectorOpts {
        readout: SectorReadout::Tda,
        ..SectorOpts::default()
    };
    let _ = sctx.sector_lowest(1, &opts);
}

#[test]
fn tsr_dimension_pins() {
    let pg = fx4_pg();
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
    let (t, v) = fx4_tv(pg);
    let (x, _e0, _lv) = exact_typed_ground(space, &t, &v);
    let (bl, br) = tangent_bases(&x);

    let caps = max_ranks(&space);
    for (b, cap) in caps.iter().enumerate() {
        assert_eq!(
            bl.ranks[b], *cap,
            "the exact ground must be FULL rank at bond {b} for the emptiness pin to mean \
             anything (at a smaller grid an empty channel would say nothing about the GROUP)"
        );
    }
    for s in space.bond_sectors(1) {
        if s.nup == 0 && s.ndn == 0 {
            assert_eq!(
                caps[1][s], 0,
                "fx4_pg (2,1): bond 1's (0,0) sector must have typed cap 0 in every irrep — that \
                 is why fluxes 1 and 2 cannot be transported across bond 1, and why phi = 1, 2 \
                 are empty channels here"
            );
        }
    }

    let mut sum = 0usize;
    let mut dims: Vec<usize> = Vec::new();
    let mut sectors: Vec<usize> = Vec::new();
    for p in 0..pg.order() {
        let phi = Irrep::new(p);
        let dim = tangent_dim_grids(&space, &bl.ranks, phi);
        let basis = tangent_basis_shifted(&bl, &br, phi);
        let sector = dets_in_irrep(D4, F_NUP, F_NDN, pg, space.target.fuse(phi)).len();
        println!(
            "tsr dim phi={p}: dim {dim}, basis {}, sector {sector}",
            basis.len()
        );
        assert_eq!(basis.len(), dim, "phi {p}: basis length != structural dim");
        assert!(
            dim <= sector,
            "phi {p}: T^phi borrows x's suffix bases, so it is a SUBSPACE of the Gamma0+phi \
             sector — dim {dim} > sector {sector} means the enumerator invented directions"
        );
        dims.push(dim);
        sectors.push(sector);
        sum += dim;
    }
    assert_eq!(
        (dims.as_slice(), sectors.as_slice()),
        ([6, 0, 0, 4].as_slice(), [6, 6, 6, 6].as_slice()),
        "fx4_pg (2,1) at max_ranks: the MEASURED channel inventory. Channels 1 and 2 are empty \
         over non-empty sectors — an outcome, not a defect"
    );

    let untyped = tangent_dim(&bl.forget_point_group());
    println!("tsr dim: Sum_phi = {sum}, untyped tangent_dim = {untyped}");
    assert_eq!(sum, untyped, "Sum_phi dim T^phi != tangent_dim(untyped)");
    assert!(
        sum < dets_for(D4, F_NUP, F_NDN).len(),
        "fixture premise: on fx4_pg the untyped tangent space at the typed grid is STRICTLY \
         smaller than the particle sector ({sum} vs {}) — the pooled untyped grid r'' sits far \
         below the untyped caps because of the GROUP",
        dets_for(D4, F_NUP, F_NDN).len()
    );

    let pg_s = synth_pg();
    let space_s = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg_s, Irrep::IDENTITY);
    let caps_s = max_ranks(&space_s);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7601);
    let mut xs: TtVector<f64> = randomized_state(space_s, 3, 3, &mut rng);
    xs.round_ranks_mut(&caps_s);
    normalize_mut(&mut xs);
    xs.check();
    let (bls, brs) = tangent_bases(&xs);
    for (b, cap) in caps_s.iter().enumerate() {
        assert_eq!(
            bls.ranks[b], *cap,
            "synth_pg reference not at max_ranks, bond {b}"
        );
    }
    let mut sum_s = 0usize;
    let mut dims_s: Vec<usize> = Vec::new();
    for p in 0..pg_s.order() {
        let phi = Irrep::new(p);
        let dim = tangent_dim_grids(&space_s, &bls.ranks, phi);
        let sector = dets_in_irrep(D4, F_NUP, F_NDN, pg_s, space_s.target.fuse(phi)).len();
        println!(
            "tsr dim (synth) phi={p}: dim {dim}, basis {}, sector {sector}",
            tangent_basis_shifted(&bls, &brs, phi).len()
        );
        assert_eq!(
            tangent_basis_shifted(&bls, &brs, phi).len(),
            dim,
            "synth phi {p}: basis length != structural dim"
        );
        assert_eq!(
            dim, sector,
            "synth_pg (2,1) at max_ranks is the fixture where T^phi IS the whole sector: phi {p} \
             has dim {dim} over a {sector}-determinant sector"
        );
        dims_s.push(dim);
        sum_s += dim;
    }
    assert_eq!(
        dims_s,
        vec![12, 12, 0, 0],
        "synth_pg (2,1) measured channel inventory"
    );
    assert_eq!(
        sum_s,
        tangent_dim(&bls.forget_point_group()),
        "the sum identity must hold on synth_pg too"
    );
    assert_eq!(
        sum_s,
        dets_for(D4, F_NUP, F_NDN).len(),
        "where every channel is its whole sector, the untyped tangent space is the whole particle \
         sector"
    );

    let (space_t, _tt, _vt, ham_t, xt) = truncated_reference();
    let rperp = residual_norm(&ham_t, &xt);
    println!("tsr dim (truncated): ||r_perp|| = {rperp:e}");
    assert!(
        rperp > 1e-3,
        "the truncated reference must actually be truncated (||r_perp|| = {rperp:e})"
    );
    let (blt, brt) = tangent_bases(&xt);
    let mut sum_t = 0usize;
    let mut dims_t: Vec<usize> = Vec::new();
    let mut sectors_t: Vec<usize> = Vec::new();
    let mut strictly_smaller = false;
    for p in 0..pg.order() {
        let phi = Irrep::new(p);
        let dim = tangent_dim_grids(&space_t, &blt.ranks, phi);
        let got = tangent_basis_shifted(&blt, &brt, phi).len();
        assert!(got <= dim, "phi {p}: over-acceptance at truncated rank");
        let sector = dets_in_irrep(D4, T_NUP, T_NDN, pg, space_t.target.fuse(phi)).len();
        assert!(dim <= sector, "phi {p}: T^phi must stay inside its sector");
        if dim < sector {
            strictly_smaller = true;
        }
        dims_t.push(dim);
        sectors_t.push(sector);
        sum_t += dim;
    }
    assert_eq!(
        (dims_t.as_slice(), sectors_t.as_slice()),
        ([9, 0, 0, 6].as_slice(), [10, 8, 10, 8].as_slice()),
        "fx4_pg (2,2) truncated: the MEASURED channel inventory"
    );
    assert_eq!(
        sum_t,
        tangent_dim(&blt.forget_point_group()),
        "the sum identity is grid-algebraic and must not depend on the regime"
    );
    assert!(
        strictly_smaller,
        "anti-vacuity: no channel's shifted tangent space sat strictly inside its sector, so the \
         truncated half of this arm tested nothing the full-rank half did not"
    );
}

#[test]
fn tsr_untyped_hessian_grams_oracle() {
    struct Leg {
        label: &'static str,
        pg: PointGroup,
        space: QnSpace,
        t: CoeffTensor<f64>,
        v: CoeffTensor<f64>,
        ham: SparseHamiltonian<f64>,
        x: TtVector<f64>,
        want_live: usize,
        r_perp_nonzero: bool,
    }

    let pg4 = fx4_pg();
    let legs = {
        let full = {
            let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg4, Irrep::IDENTITY);
            let (t, v) = fx4_tv(pg4);
            let (ham, x) = typed_minimizer(space, &t, &v, 0x7505, 1, 2);
            Leg {
                label: "full-rank (2,1) minimizer",
                pg: pg4,
                space,
                t,
                v,
                ham,
                x,
                want_live: 2,
                r_perp_nonzero: false,
            }
        };
        let trunc = {
            let (space, t, v, ham, x) = truncated_reference();
            Leg {
                label: "truncated (2,2)",
                pg: pg4,
                space,
                t,
                v,
                ham,
                x,
                want_live: 2,
                r_perp_nonzero: true,
            }
        };
        let d6 = {
            let (space, t, v, ham, x) = d6_reference();
            Leg {
                label: "d = 6, all four channels live",
                pg: d6_pg(),
                space,
                t,
                v,
                ham,
                x,
                want_live: 4,
                r_perp_nonzero: true,
            }
        };
        [full, trunc, d6]
    };

    let mut curved_total = 0usize;
    for leg in legs {
        let mut curved = 0usize;
        let Leg {
            label,
            pg,
            space,
            t,
            v,
            ham,
            x,
            want_live,
            r_perp_nonzero,
        } = leg;
        let (bl_x, br_x) = tangent_bases(&x);
        let lambda = rayleigh_quotient(&ham, &x).re_s();

        let space_c1 = QnSpace::new(space.d, space.nup, space.ndn);
        let x_c1 = x.forget_point_group();
        let h_c1 = SparseHamiltonian::<f64>::new(space_c1, &t, &v, None, f64::EPSILON);
        let lam_c1 = rayleigh_quotient(&h_c1, &x_c1).re_s();
        assert!(
            (lam_c1 - lambda).abs() <= 1e-10 * lambda.abs().max(1.0),
            "[{label}] the embedding changed the Rayleigh quotient: {lam_c1} vs {lambda}"
        );
        let (defect_gamma, defects) = symmetry_defect(&x_c1, pg);
        let worst = defects.iter().cloned().fold(0.0f64, f64::max);
        assert_eq!(
            defect_gamma, space.target,
            "[{label}] wrong irrep read back"
        );
        assert!(
            worst <= 1e-7,
            "[{label}] the untyped embedding must stay a symmetry eigenstate: defect {worst:e}"
        );

        let ctx_c1 = weingarten_operator(&h_c1, &x_c1);
        let (bl_c1, br_c1) = ctx_c1.bases();
        let rperp = residual_norm(&ham, &x);
        println!(
            "tsr oracle [{label}]: sigma_min(c1) = {:e}, ||r_perp|| = {rperp:e}, lambda = \
             {lambda}, live = {:?}",
            min_schmidt(bl_c1, br_c1),
            live_channels(&space, &bl_x.ranks)
        );
        if r_perp_nonzero {
            assert!(
                rperp > 1e-3,
                "[{label}] fixture premise: this leg must carry a genuine normal residual \
                 (||r_perp|| = {rperp:e}); B is linear in it"
            );
        } else {
            assert!(
                rperp <= 1e-12,
                "[{label}] fixture premise: a FULL-RANK leg has no normal residual to carry \
                 (||r_perp|| = {rperp:e}, measured 3.5e-16) — the zero-B expectations below are \
                 keyed off it"
            );
        }
        let basis_c1 = tangent_basis(bl_c1, br_c1, false);
        assert!(
            !basis_c1.is_empty(),
            "[{label}] the untyped tangent basis is empty"
        );
        let ctx = weingarten_operator(&ham, &x);

        let mut checked = 0usize;
        for p in 0..pg.order() {
            let phi = Irrep::new(p);
            let gamma = space.target.fuse(phi);
            let space_out = space.with_target(gamma);

            let cands: Vec<TtTangent<f64>> = basis_c1
                .iter()
                .map(|b| project_irrep(b, bl_c1, br_c1, pg, gamma).0)
                .collect();
            let block = mgs_tangents(cands);

            let basis_phi = tangent_basis_shifted(&bl_x, &br_x, phi);
            assert_eq!(
                block.len(),
                basis_phi.len(),
                "[{label}] phi {p}: the untyped irrep block has {} directions but T^phi has {} — \
                 the two decompositions disagree on the DIMENSION, so comparing spectra would be \
                 meaningless",
                block.len(),
                basis_phi.len()
            );
            if block.is_empty() {
                assert_eq!(
                    tangent_dim_grids(&space, &bl_x.ranks, phi),
                    0,
                    "[{label}] phi {p}: an empty basis over a nonzero structural dim"
                );
                continue;
            }
            checked += 1;

            let n = block.len();
            let (a_c1, b_c1) = hessian_grams(&ctx_c1, &block);
            let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
            let a_phi = effham_gram_shifted(&h_out, &bl_x, &br_x, phi, lambda, &basis_phi);
            let e_c1 = gram_eigs(&a_c1, n);
            let e_phi = gram_eigs(&a_phi, n);
            let scale = e_c1.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
            println!(
                "tsr oracle [{label}] phi={p}: n = {n}, scale = {scale:e}, worst = {:e}",
                e_c1.iter()
                    .zip(e_phi.iter())
                    .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
            );
            assert!(
                scale > 1e-3,
                "[{label}] phi {p}: anti-vacuity, the untyped block spectrum is identically 0"
            );
            for (j, (a, b)) in e_c1.iter().zip(e_phi.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-9 * scale,
                    "[{label}] phi {p}: eigenvalue {j} of A^phi is {b} but the untyped \
                     hessian_grams block says {a} (scale {scale:e})"
                );
            }

            let (a2_phi, b_phi) = if phi == Irrep::IDENTITY {
                hessian_grams(&ctx, &basis_phi)
            } else {
                SectorCtx::new(&ctx, &h_out, phi).grams(&basis_phi)
            };
            let e_a2 = gram_eigs(&a2_phi, n);
            for (j, (a, b)) in e_phi.iter().zip(e_a2.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-10 * scale,
                    "[{label}] phi {p}: SectorCtx::grams().0 eigenvalue {j} is {b} but \
                     effham_gram_shifted says {a} — the (A, B) entry point and the A-only one \
                     must be the same operator"
                );
            }
            let e_b_c1 = gram_eigs(&b_c1, n);
            let e_b_phi = gram_eigs(&b_phi, n);
            let nb_c1 = e_b_c1.iter().fold(0.0f64, |m, z| m.max(z.abs()));
            let nb_phi = e_b_phi.iter().fold(0.0f64, |m, z| m.max(z.abs()));
            let na_phi = e_phi.iter().fold(0.0f64, |m, z| m.max(z.abs()));
            println!(
                "tsr oracle [{label}] phi={p}: ||B||/||A|| = {:.3e} (||A|| = {na_phi:e}, \
                 ||B||_phi = {nb_phi:e}, ||B||_c1 = {nb_c1:e}), worst |dB| = {:e}",
                nb_phi / na_phi.max(f64::MIN_POSITIVE),
                e_b_c1
                    .iter()
                    .zip(e_b_phi.iter())
                    .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
            );
            for (j, (a, b)) in e_b_c1.iter().zip(e_b_phi.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-9 * scale,
                    "[{label}] phi {p}: eigenvalue {j} of B^phi is {b} but the untyped \
                     hessian_grams block says {a} (scale {scale:e}) — B^phi is NOT the phi-block \
                     of the untyped Weingarten map"
                );
            }
            for (sign, name) in [(1.0f64, "A+B (= H^rr)"), (-1.0, "A-B (= H^ii)")] {
                let comb = |a: &[f64], b: &[f64]| -> Vec<f64> {
                    a.iter().zip(b.iter()).map(|(p, q)| p + sign * q).collect()
                };
                let e_lhs = gram_eigs(&comb(&a_c1, &b_c1), n);
                let e_rhs = gram_eigs(&comb(&a2_phi, &b_phi), n);
                for (j, (a, b)) in e_lhs.iter().zip(e_rhs.iter()).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-9 * scale,
                        "[{label}] phi {p}: eigenvalue {j} of {name} is {b} in the shifted \
                         channel but {a} in the untyped block (scale {scale:e}) — the A- and \
                         B-spectra can agree while their eigenvectors are misaligned, and this \
                         is the combination the Williamson consumes"
                    );
                }
            }
            let sites = slot_sites(&space, &bl_x.ranks, phi);
            if r_perp_nonzero && sites.len() >= 2 {
                curved += 1;
                assert!(
                    nb_phi > 1e-3 * na_phi,
                    "[{label}] phi {p}: ANTI-VACUITY — ||B^phi|| = {nb_phi:e} is under 1e-3 of \
                     ||A^phi|| = {na_phi:e} at a reference with ||r_perp|| > 0 whose T^phi \
                     reaches sites {sites:?}, where B must not vanish. A silently zero B (the \
                     documented failure mode: seeding pZ/qZ from the SHIFTED frames collapses \
                     the whole q chain) would pass every eigenvalue comparison above"
                );
            } else {
                assert!(
                    na_phi > 1e-3,
                    "[{label}] phi {p}: ||A^phi|| = {na_phi:e} — a relative bound on B against a \
                     vanishing A is vacuous (measured inventory 1.46 … 3.38)"
                );
                assert!(
                    nb_phi <= 1e-9 * na_phi,
                    "[{label}] phi {p}: B^phi = {nb_phi:e} where it must vanish (r_perp nonzero: \
                     {r_perp_nonzero}, T^phi slot sites: {sites:?}; ||A|| = {na_phi:e})"
                );
                assert!(
                    nb_c1 <= 1e-9 * na_phi,
                    "[{label}] phi {p}: the untyped B block is {nb_c1:e} where B must vanish"
                );
            }
        }
        assert_eq!(
            checked, want_live,
            "[{label}] anti-vacuity: {checked} channels carried a nonempty block, the measured \
             inventory of this fixture is {want_live}"
        );
        println!("tsr oracle [{label}]: {curved} channel(s) with a genuinely nonzero B");
        curved_total += curved;
    }
    assert!(
        curved_total >= 2,
        "ANTI-VACUITY across the whole arm: only {curved_total} (leg, channel) pairs carried a \
         nonzero B^phi, so the B half of this oracle would be a statement about the zero \
         operator. Measured inventory 2026-09-08: 1 on the truncated d = 4 leg (phi = 0 only — \
         its shifted channel is SINGLE-SITE and B is structurally 0 there) and 4 on the d = 6 leg"
    );
}

#[test]
fn tsr_full_rank_levels_are_the_sector_spectra() {
    let pg = synth_pg();
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
    let (t, v) = synth_tv(pg);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);

    let caps = max_ranks(&space);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7601);
    let mut x: TtVector<f64> = randomized_state(space, 3, 3, &mut rng);
    x.round_ranks_mut(&caps);
    normalize_mut(&mut x);
    x.check();
    let (bl, br) = tangent_bases(&x);
    for (b, cap) in caps.iter().enumerate() {
        assert_eq!(
            bl.ranks[b], *cap,
            "leg 1 needs the max_ranks grid at bond {b}: that is where T^phi is the whole sector"
        );
    }
    let lambda = rayleigh_quotient(&ham, &x).re_s();
    let g = grad_norm_at(&ham, &x);
    println!(
        "tsr sector spectra [random max_ranks]: lambda = {lambda}, ||grad|| = {g:e}, \
         sigma_min = {:e}, live = {:?}",
        min_schmidt(&bl, &br),
        live_channels(&space, &bl.ranks)
    );
    assert!(
        g > 1e-2,
        "anti-vacuity: leg 1 must run at a NON-critical point, or it is leg 2 in disguise and the \
         base-point-freedom of the identity is untested (||grad|| = {g:e})"
    );

    let channels = live_channels(&space, &bl.ranks);
    assert_eq!(channels.len(), 2, "synth_pg (2,1) has two live channels");
    let mut union: Vec<f64> = Vec::new();
    for phi in channels {
        let p = phi.index();
        let gamma = space.target.fuse(phi);
        let space_out = space.with_target(gamma);
        let basis = tangent_basis_shifted(&bl, &br, phi);
        let want = dense_levels(D4, &t, &v, pg, F_NUP, F_NDN, gamma);
        assert_eq!(
            basis.len(),
            want.len(),
            "phi {p}: the PRECONDITION of this leg is dim T^phi == dim(sector); T^phi has {} \
             directions and irrep {gamma:?} holds {} states",
            basis.len(),
            want.len()
        );
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        let a = effham_gram_shifted(&h_out, &bl, &br, phi, lambda, &basis);
        let got: Vec<f64> = gram_eigs(&a, basis.len())
            .into_iter()
            .map(|z| z + lambda)
            .collect();
        println!("tsr sector spectra phi={p}: got {got:?}\n                   want {want:?}");
        assert_multiset_eq(&got, &want, 1e-8, &format!("phi {p} levels"));
        union.extend(got);
    }
    union.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let all = dense_eigen(&dense_hamiltonian_matrix(
        D4,
        &t,
        &v,
        &dets_for(D4, F_NUP, F_NDN),
    ))
    .0;
    assert!(
        all.len() >= 4,
        "anti-vacuity: the particle sector has fewer than four levels"
    );
    assert_multiset_eq(
        &union,
        &all,
        1e-8,
        "union over the live phi vs the trivial-PG spectrum",
    );

    let (xe, e0, ref_levels) = exact_typed_ground(space, &t, &v);
    let (ble, bre) = tangent_bases(&xe);
    let lam_e = rayleigh_quotient(&ham, &xe).re_s();
    let ge = grad_norm_at(&ham, &xe);
    println!("tsr sector spectra [exact ground]: E0 = {e0}, lambda = {lam_e}, ||grad|| = {ge:e}");
    assert!(
        (lam_e - e0).abs() <= 1e-10 * e0.abs().max(1.0),
        "the exact ground's Rayleigh quotient must be E_0"
    );
    assert!(ge <= 1e-9, "the exact ground must be critical: {ge:e}");
    assert!(
        ref_levels.len() >= 2,
        "anti-vacuity: the reference irrep has fewer than two levels"
    );

    let channels_e = live_channels(&space, &ble.ranks);
    assert!(!channels_e.is_empty(), "anti-vacuity: no live channel");
    let mut strictly_smaller = false;
    for phi in channels_e {
        let p = phi.index();
        let gamma = space.target.fuse(phi);
        let space_out = space.with_target(gamma);
        let basis = tangent_basis_shifted(&ble, &bre, phi);
        let levels = dense_levels(D4, &t, &v, pg, F_NUP, F_NDN, gamma);
        println!(
            "tsr sector spectra [exact ground] phi={p}: dim {} vs sector dim {}",
            basis.len(),
            levels.len()
        );
        assert!(
            basis.len() <= levels.len(),
            "phi {p}: T^phi must stay inside its sector"
        );
        if basis.len() < levels.len() {
            strictly_smaller = true;
        }
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        let a = effham_gram_shifted(&h_out, &ble, &bre, phi, e0, &basis);
        let ritz: Vec<f64> = gram_eigs(&a, basis.len())
            .into_iter()
            .map(|z| z + e0)
            .collect();
        let (lo, hi) = (levels[0], levels[levels.len() - 1]);
        println!("tsr sector spectra [exact ground] phi={p}: ritz {ritz:?} in [{lo}, {hi}]");
        for (j, w) in ritz.iter().enumerate() {
            assert!(
                *w >= lo - 1e-8 && *w <= hi + 1e-8,
                "phi {p}: A^phi is a COMPRESSION of H|_{gamma:?} - E_0 to a subspace, so Ritz \
                 value {j} ({w}) must lie in [{lo}, {hi}]"
            );
        }
        assert!(
            ritz[0] >= lo - 1e-8,
            "phi {p}: the lowest Ritz value ({}) is below the channel's ground ({lo}) — a \
             compression cannot go under its own operator's minimum",
            ritz[0]
        );
    }
    assert!(
        strictly_smaller,
        "anti-vacuity: leg 2 must run where T^phi is STRICTLY inside its sector, or it is leg 1 \
         again and the interlacing assertions are the equality ones in disguise"
    );
}

#[test]
fn tsr_instability_reports_a_certified_negative() {
    let pg = synth_pg();
    let gamma0 = Irrep::new(1);
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, gamma0);
    let (t, v) = synth_tv(pg);
    let (x, e0, _lv) = exact_typed_ground(space, &t, &v);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);

    let lower = dense_levels(D4, &t, &v, pg, F_NUP, F_NDN, Irrep::IDENTITY);
    assert!(
        lower[0] < e0 - 1e-3,
        "fixture premise: irrep 0's ground ({}) must lie BELOW the reference ({e0})",
        lower[0]
    );
    let want = lower[0] - e0;

    let ctx = weingarten_operator(&ham, &x);
    let phi = Irrep::new(1);
    let space_out = space.with_target(gamma0.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    assert_eq!(sctx.phi(), phi);
    assert_eq!(sctx.space_out(), space_out);
    assert!(sctx.dim() > 0, "the lower channel must not be empty");

    let opts = SectorOpts {
        readout: SectorReadout::Tda,
        ..SectorOpts::default()
    };
    let out = sctx.sector_lowest_tda(3, &opts);
    let theta = out
        .theta()
        .expect("a Tda readout must publish theta(), not omega2()");
    assert!(
        out.omega2().is_none(),
        "a Tda readout must NOT publish omega2(): eig(A^phi) is an energy, omega^2 is not"
    );
    println!(
        "tsr instability: dim {} theta {:?} converged {} iters {} applies {} sigma_min {:e} \
         negatives {}",
        out.dim,
        theta,
        out.converged,
        out.iterations,
        out.applies,
        out.sigma_min,
        out.negatives().len()
    );
    assert_eq!(out.phi, phi);
    assert_eq!(out.target, space_out.target);
    assert!(out.dim > 0);
    assert!(
        theta.iter().all(|z| z.is_finite()),
        "the readout must not produce NaN/inf: {theta:?}"
    );
    assert!(
        theta[0] < 0.0,
        "the lowest A^phi eigenvalue must be negative here: {theta:?}"
    );
    assert!(
        out.negatives().iter().any(|w| w.certified()),
        "a negative direction must come with a CERTIFIED fresh-apply witness, not a bare Ritz \
         value: {:?}",
        out.negatives()
    );
    assert!(
        (theta[0] - want).abs() <= 1e-6 * want.abs().max(1.0),
        "at full rank the negative must be exactly E_0(lower) - E_0(ref) = {want}, got {}",
        theta[0]
    );
}

#[test]
fn tsr_lih_sector_gaps_track_the_channel_driver() {
    let (space, _pg, t, v, ham, x) = lih_typed_fixture();
    let space = *space;
    let g = grad_norm_at(ham, x);
    println!("tsr lih: ||grad|| = {g:e}, target {:?}", space.target);
    assert!(g < 1e-8, "LiH fixture not critical (||grad|| = {g:e})");

    let ctx = weingarten_operator(ham, x);
    let opts = ChannelOpts {
        channels: ChannelSet {
            irreps: IrrepSel::All,
            ..ChannelSet::default()
        },
        gs: ChannelGsOpts {
            chi: qntt::Pooled(4),
            max_iter: 20,
            ..ChannelGsOpts::default()
        },
        cone: ConeOpts {
            targets: 1,
            max_added: 0,
            ..ConeOpts::default()
        },
        ..ChannelOpts::default()
    };
    let spec = channel_response(t, v, space, &opts);
    let e_ref = spec.channels[spec.reference]
        .e_gs
        .expect("the reference channel must converge");

    let mut compared = 0usize;
    for ch in &spec.channels {
        let gamma = ch.label.irrep;
        if gamma == space.target {
            continue;
        }
        if ch.status != ChannelStatus::Ok {
            println!("tsr lih: channel {gamma:?} skipped ({:?})", ch.status);
            continue;
        }
        let Some(e_ch) = ch.e_gs else { continue };
        let channel_gap = e_ch - e_ref;

        let phi = space.target.fuse(gamma);
        let space_out = space.with_target(gamma);
        let h_out = SparseHamiltonian::<f64>::new(space_out, t, v, None, f64::EPSILON);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        if sctx.dim() == 0 {
            println!("tsr lih: channel {gamma:?} has an empty shifted tangent space");
            continue;
        }
        let out = sctx.sector_lowest_tda(1, &SectorOpts::default());
        let theta = out.theta().expect("a Tda readout publishes theta()");
        println!(
            "tsr lih {gamma:?}: sector TDA {:?} (dim {}, converged {}, sigma_min {:e}) vs \
             channel_response gap {channel_gap}",
            theta, out.dim, out.converged, out.sigma_min
        );
        assert!(
            out.converged,
            "channel {gamma:?}: the TDA readout did not converge"
        );
        assert!(!theta.is_empty(), "channel {gamma:?}: empty readout");
        let tda = theta[0];
        assert!(tda.is_finite() && channel_gap.is_finite());
        assert!(
            tda > 0.0,
            "channel {gamma:?}: LiH's ground state is A1, so every other irrep must sit ABOVE \
             the reference — a negative TDA gap here is a defect, not physics (got {tda})"
        );
        assert!(
            channel_gap > 0.0,
            "channel {gamma:?}: the channel driver disagrees on the ORDERING ({channel_gap})"
        );
        let ratio = tda / channel_gap;
        assert!(
            (0.2..=5.0).contains(&ratio),
            "channel {gamma:?}: sector TDA {tda} and channel gap {channel_gap} differ by more \
             than a factor of five (ratio {ratio}) — they are different objects (one linearizes \
             at a fixed rank-2 A1 reference, the other re-converges a separate ground state per \
             irrep), but not THAT different"
        );
        compared += 1;
    }
    assert!(
        compared >= 1,
        "anti-vacuity: no non-reference channel was comparable"
    );
}

#[test]
fn tsr_threaded_matches_serial() {
    let (space, t, v, ham, x) = d6_minimizer();
    let (bl, br) = tangent_bases(&x);
    let lambda = rayleigh_quotient(&ham, &x).re_s();
    let phi = first_curved_channel(&space, &bl.ranks);
    let dim = tangent_dim_grids(&space, &bl.ranks, phi);
    println!("tsr threading: phi = {phi:?}, dim = {dim}");
    assert!(
        dim > 1,
        "anti-vacuity: the channel must carry more than one direction"
    );
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);

    let ctx = weingarten_operator(&ham, &x);
    let run = || {
        let basis = tangent_basis_shifted(&bl, &br, phi);
        let gram = effham_gram_shifted(&h_out, &bl, &br, phi, lambda, &basis);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        let (a, b) = sctx.grams(&basis);
        let hrr: Vec<f64> = a.iter().zip(b.iter()).map(|(p, q)| p + q).collect();
        let w = signed_omega2_or_fail(
            &sctx.symplectic_eigenvalues(&basis),
            "tsr threading: the shifted dense symplectic driver",
        );
        let af: Vec<f64> = basis.iter().map(|u| sctx.a_apply_fused(u).norm()).collect();
        let nc = sctx.a_apply_fused(&basis[0]).num_coeffs();
        (basis.len(), gram, a, b, hrr, w, af, nc)
    };

    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build 1-thread rayon pool");
    let (ns, gs, as_, bs, hrrs, ws, afs, ncs) = serial_pool.install(|| {
        assert_eq!(
            rayon::current_num_threads(),
            1,
            "serial arm did not run in the 1-thread pool"
        );
        run()
    });
    let par_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("build 8-thread rayon pool");
    let (np, gp, ap, bp, hrrp, wp, afp, ncp) = par_pool.install(|| {
        assert!(
            rayon::current_num_threads() > 1,
            "parallel arm did not run multi-threaded ({} threads)",
            rayon::current_num_threads()
        );
        run()
    });

    assert!(ns > 1, "anti-vacuity: the shifted basis has {ns} vectors");
    assert_eq!(
        ns, np,
        "threaded vs serial: the MGS cull accepted a different number of directions"
    );
    assert_eq!(gs.len(), gp.len());
    let scale = gs.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    assert!(scale > 1e-3, "anti-vacuity: the Gram is identically zero");
    for (j, (a, b)) in gs.iter().zip(gp.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12 * scale,
            "threaded vs serial gram entry {j}: {a} vs {b}"
        );
    }
    let nb = bs.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    let na = as_.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    println!(
        "tsr threading: ||A|| = {na:e}, ||B|| = {nb:e}, |omega^2| = {}",
        ws.len()
    );
    assert!(
        nb > 1e-3 * na,
        "ANTI-VACUITY: ||B|| = {nb:e} vs ||A|| = {na:e} — a matches-serial pin over a B that is \
         identically zero forks nothing and proves nothing"
    );
    for (name, s, q, sc) in [
        ("A", &as_, &ap, na),
        ("B", &bs, &bp, nb),
        ("H^rr = A+B", &hrrs, &hrrp, na + nb),
    ] {
        assert_eq!(s.len(), q.len(), "{name}: gram size differs across pools");
        for (j, (a, b)) in s.iter().zip(q.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * sc.max(1.0),
                "threaded vs serial {name} gram entry {j}: {a} vs {b}"
            );
        }
    }
    assert_eq!(
        ws.len(),
        wp.len(),
        "the dense symplectic driver returned a different number of omega^2 across pools"
    );
    let wscale = ws.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
    for (j, (a, b)) in ws.iter().zip(wp.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-10 * wscale,
            "threaded vs serial signed omega^2[{j}]: {a} vs {b} (scale {wscale:e})"
        );
    }
    assert_eq!(
        afs.len(),
        afp.len(),
        "the fused shifted apply was run over a different number of directions across pools"
    );
    let af_scale = afs.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
    println!("tsr threading: fused |A xi| scale = {af_scale:e}, num_coeffs = {ncs}");
    assert!(
        af_scale > 1e-3,
        "ANTI-VACUITY: the fused shifted apply annihilated every direction (scale {af_scale:e}) \
         — a matches-serial pin over an annihilated direction forks nothing and proves nothing"
    );
    for (j, (a, b)) in afs.iter().zip(afp.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12 * af_scale,
            "threaded vs serial fused |A^phi xi| for direction {j}: {a} vs {b}"
        );
    }
    assert_eq!(
        ncs, ncp,
        "threaded vs serial: the fused shifted apply returned a different coefficient count \
         ({ncs} vs {ncp}) — a pool-dependent zero-width component passes every norm bound"
    );
    let d0 = dense_dot(&x.to_dense(), &x.to_dense());
    assert!((d0 - 1.0).abs() <= 1e-10, "the fixture must be normalized");
}

#[test]
fn tsr_empty_channel_is_a_reported_outcome() {
    let pg = fx4_pg();
    let (space, t, v, ham, x) = truncated_reference();
    let (bl, _br) = tangent_bases(&x);
    let empty: Vec<Irrep> = (0..pg.order())
        .map(Irrep::new)
        .filter(|&phi| tangent_dim_grids(&space, &bl.ranks, phi) == 0)
        .collect();
    println!(
        "tsr empty channel: empty {empty:?}, live {:?}",
        live_channels(&space, &bl.ranks)
    );
    assert!(
        !empty.is_empty(),
        "fixture premise: fx4_pg (2,2) must carry an empty channel (measured phi = 1, 2)"
    );

    let ctx = weingarten_operator(&ham, &x);
    for phi in empty {
        let p = phi.index();
        let space_out = space.with_target(space.target.fuse(phi));
        let sector = dets_in_irrep(D4, T_NUP, T_NDN, pg, space_out.target).len();
        assert!(
            sector > 0,
            "phi {p}: the target SECTOR must be non-empty, or the arm tests an empty Hilbert \
             space rather than an empty borrowed-basis channel"
        );
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        assert_eq!(sctx.dim(), 0, "phi {p}: dim must be 0");
        assert!(sctx.basis().is_empty(), "phi {p}: basis must be empty");

        let out = sctx.sector_lowest_tda(3, &SectorOpts::default());
        let theta = out.theta().expect("a Tda readout publishes theta()");
        println!(
            "tsr empty channel phi={p}: sector {sector}, dim {}, theta {:?}, converged {}, \
             applies {}, negatives {}, sigma_min {:e}",
            out.dim,
            theta,
            out.converged,
            out.applies,
            out.negatives().len(),
            out.sigma_min
        );
        assert!(theta.is_empty(), "phi {p}: theta must be empty");
        assert!(
            out.converged,
            "phi {p}: an unreachable channel is converged by definition — there is nothing left \
             to converge — and reporting it unconverged would look like a solver failure"
        );
        assert_eq!(
            out.applies, 0,
            "phi {p}: an empty channel must cost no applies"
        );
        assert_eq!(out.iterations, 0, "phi {p}: and no iterations");
        assert!(
            out.negatives().is_empty(),
            "phi {p}: no negatives to report"
        );
        assert!(
            krein_signs(out.rpa()).is_empty()
                && residuals_of(out.rpa()).is_empty()
                && !out.metric_indefinite(),
            "phi {p}: a Tda readout carries no signature, residual or metric report"
        );
        assert_eq!(
            out.index_lower_bound(),
            None,
            "phi {p}: Tda has no pencil inertia"
        );
        assert_eq!(out.dim, 0, "phi {p}: the readout must REPORT dim = 0");
        assert_eq!(out.phi, phi);
        assert_eq!(out.target, space_out.target);
        assert!(
            out.sigma_min.is_finite() && out.sigma_min > 0.0,
            "phi {p}: sigma_min is base-point data and must still be quoted ({})",
            out.sigma_min
        );

        for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Auto)] {
            let rpa = sctx.sector_lowest(
                3,
                &SectorOpts {
                    readout: SectorReadout::Rpa,
                    kernel,
                    ..SectorOpts::default()
                },
            );
            println!(
                "tsr empty channel phi={p} (Rpa {kernel:?}): omega2 {:?}, omega {:?}, converged {}, applies {}, \
                 negatives {}",
                rpa.omega2(),
                rpa.omega(),
                rpa.converged,
                rpa.applies,
                rpa.negatives().len()
            );
            assert_eq!(
                rpa.omega2().map(|w| w.len()),
                Some(0),
                "phi {p}: an Rpa readout on an empty channel must publish an EMPTY omega2, not None \
                 (which means 'no real omega^2 exists', a different statement)"
            );
            assert!(
                rpa.theta().is_none(),
                "phi {p}: an Rpa readout has no theta"
            );
            assert!(
                rpa.converged,
                "phi {p}: an unreachable channel is converged by definition"
            );
            assert_eq!(rpa.applies, 0, "phi {p}: no applies on an empty channel");
            assert_eq!(rpa.iterations, 0, "phi {p}: and no iterations");
            assert!(rpa.negatives().is_empty() && rpa.all_negatives_certified());
            assert!(
                krein_signs(rpa.rpa()).is_empty() && residuals_of(rpa.rpa()).is_empty(),
                "phi {p}: an empty Rpa readout carries no per-entry field"
            );
            assert!(
                !rpa.metric_indefinite() && !rpa.metric_indefinite(),
                "phi {p}: no solve ran, so no metric was met"
            );
            assert_eq!(
                rpa.index_lower_bound(),
                Some(0),
                "phi {p}: an empty pencil has no inertia"
            );
            assert_eq!(rpa.dim, 0);
            assert_eq!(rpa.phi, phi);
            assert_eq!(rpa.target, space_out.target);
            let r = rpa.rpa().expect("an Rpa readout");
            match (kernel, &r.report) {
                (RpaKernel::Paired, RpaReport::Paired { offset, .. }) => {
                    assert_eq!(offset.total(), 0, "phi {p}: an empty pencil has no offset")
                }
                (RpaKernel::Signed { metric, .. }, RpaReport::Signed { metric: m, .. }) => {
                    assert_eq!(*m, metric, "phi {p}: the requested lane, verbatim")
                }
                (k, rep) => panic!("phi {p}: requested {k:?}, reported {rep:?}"),
            }
        }
    }
}

#[test]
#[should_panic(expected = "flux")]
fn tsr_grams_refuses_a_wrong_flux_basis_vector() {
    let (space, t, v, ham, x) = truncated_reference();
    let (bl, br) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_shifting_channel(&space, &bl.ranks);
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);

    let mut basis = sctx.basis();
    assert!(!basis.is_empty(), "the channel must be non-empty");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7B50);
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let unshifted = TtTangent::project(&bl, &br, &dir);
    assert_eq!(unshifted.phi(), Irrep::IDENTITY);
    assert!(
        unshifted.norm() > 1e-8,
        "the wrong-flux probe must be nonzero, or the refusal could be about a zero vector"
    );
    basis.push(unshifted);
    let _ = sctx.grams(&basis);
}

#[test]
#[should_panic(expected = "flux")]
fn tsr_a_apply_fused_refuses_the_wrong_flux() {
    let (space, t, v, ham, x) = truncated_reference();
    let (bl, br) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_shifting_channel(&space, &bl.ranks);
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7B51);
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let unshifted = TtTangent::project(&bl, &br, &dir);
    assert_eq!(unshifted.phi(), Irrep::IDENTITY);
    assert!(
        unshifted.norm() > 1e-8,
        "the wrong-flux probe must be nonzero, or the refusal could be about a zero vector"
    );
    let _ = sctx.a_apply_fused(&unshifted);
}

#[test]
#[should_panic(expected = "flux")]
fn tsr_a_gram_refuses_a_wrong_flux_basis_vector() {
    let (space, t, v, ham, x) = truncated_reference();
    let (bl, br) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_shifting_channel(&space, &bl.ranks);
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);

    let mut basis = sctx.basis();
    assert!(!basis.is_empty(), "the channel must be non-empty");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7B52);
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let unshifted = TtTangent::project(&bl, &br, &dir);
    assert_eq!(unshifted.phi(), Irrep::IDENTITY);
    assert!(unshifted.norm() > 1e-8);
    basis.push(unshifted);
    let _ = sctx.a_gram(&basis);
}

#[test]
fn tsr_fd_oracle_certifies_a_plus_b_on_the_untyped_curve() {
    let (space, t, v, ham, x) = d6_minimizer();
    let (bl, br) = tangent_bases(&x);
    let lambda = rayleigh_quotient(&ham, &x).re_s();
    let ctx = weingarten_operator(&ham, &x);

    let space_c1 = QnSpace::new(space.d, space.nup, space.ndn);
    let x_c1 = x.forget_point_group();
    let h_c1 = SparseHamiltonian::<f64>::new(space_c1, &t, &v, None, f64::EPSILON);
    let ctx_c1 = weingarten_operator(&h_c1, &x_c1);
    let (bl_c1, br_c1) = ctx_c1.bases();
    let g_vec = ctx_c1.gradient().to_vector();
    let rperp = residual_norm(&ham, &x);
    println!(
        "tsr fd: sigma_min = {:e}, sigma_min(c1) = {:e}, ||r_perp|| = {rperp:e}, lambda = \
         {lambda}, ||grad_c1|| = {:e}, live = {:?}",
        min_schmidt(&bl, &br),
        min_schmidt(bl_c1, br_c1),
        ctx_c1.gradient().norm(),
        live_channels(&space, &bl.ranks)
    );
    assert!(
        rperp > 1e-6,
        "fixture premise: ||r_perp|| = {rperp:e} — B is linear in r_perp, so at a zero residual \
         this arm would certify A alone"
    );

    let embed = |xi: &TtTangent<f64>, what: &str| embed_untyped(xi, bl_c1, br_c1, what);

    let mut channels = 0usize;
    let mut curved = 0usize;
    let curved_set = curved_channels(&space, &bl.ranks);
    println!("tsr fd: curved (multi-site, B-carrying) channels = {curved_set:?}");
    for phi in live_channels(&space, &bl.ranks) {
        if phi == Irrep::IDENTITY {
            continue;
        }
        let b_live = curved_set.contains(&phi);
        let p = phi.index();
        let space_out = space.with_target(space.target.fuse(phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        assert!(
            sctx.dim() >= 2,
            "phi {p}: need >= 2 directions for the polarized entry"
        );
        let dirs = shifted_unit_tangents(&sctx, space_out, 0x7B20 + u64::from(phi.index()), 4);

        for (j, xi) in dirs.iter().take(3).enumerate() {
            let a_xi = sctx.a_apply(xi);
            let b_xi = sctx.b_apply(xi);
            let h_xi = sctx.hess_apply(xi);
            let q_ab = xi.dot(&h_xi);
            let s = q_ab.abs().max(1.0);

            let xi_c1 = embed(xi, &format!("phi {p} dir {j}"));
            let (_qc, _qu, corr) = corrected_form(&h_c1, &x_c1, &g_vec, &xi_c1, 2e-4);
            let q_rich = corrected_form_richardson(&h_c1, &x_c1, &g_vec, &xi_c1, 2e-4);
            let q_alg = xi_c1.dot(&ctx_c1.a_apply_fused(&xi_c1).add(&ctx_c1.b_apply(&xi_c1)));
            println!(
                "tsr fd phi={p} dir={j}: q_ab = {q_ab:.12} q_rich = {q_rich:.12} \
                 (|d| = {:e}) q_alg = {q_alg:.12} (|d| = {:e}) corr = {corr:e} \
                 ||A xi|| = {:e} ||B xi|| = {:e} ratio = {:.3e}",
                (q_rich - q_ab).abs(),
                (q_alg - q_ab).abs(),
                a_xi.norm(),
                b_xi.norm(),
                b_xi.norm() / a_xi.norm().max(f64::MIN_POSITIVE)
            );
            assert!(
                (q_alg - q_ab).abs() <= 1e-10 * s,
                "phi {p} dir {j}: the shifted <xi,(A+B)xi> = {q_ab} disagrees with the UNTYPED \
                 algebraic value {q_alg} at 1e-10 — this is the shifted assembly, not the FD"
            );
            assert!(
                (q_rich - q_ab).abs() <= 1e-6 * s,
                "phi {p} dir {j}: the corrected Richardson FD says {q_rich} but \
                 <xi,(A^phi+B^phi)xi> = {q_ab} (scale {s:e}); the algebraic cross-check passed, \
                 so this is the FD or the embedding"
            );
            if b_live {
                assert!(
                    b_xi.norm() > 1e-3 * a_xi.norm(),
                    "phi {p} dir {j}: ANTI-VACUITY — ||B^phi xi|| = {:e} is under 1e-3 of \
                     ||A^phi xi|| = {:e} on a channel whose T^phi reaches {:?}, so this arm \
                     would be an FD check on A alone",
                    b_xi.norm(),
                    a_xi.norm(),
                    slot_sites(&space, &bl.ranks, phi)
                );
            } else {
                assert!(
                    b_xi.norm() <= 1e-9 * a_xi.norm(),
                    "phi {p} dir {j}: T^phi is single-site, so B^phi must be identically zero \
                     (got {:e} against ||A^phi xi|| = {:e})",
                    b_xi.norm(),
                    a_xi.norm()
                );
            }
        }

        let (xi, eta) = (&dirs[0], &dirs[3]);
        let xi_c1 = embed(xi, "polarized xi");
        let eta_c1 = embed(eta, "polarized eta");
        let sum_c1 = xi_c1.add(&eta_c1);
        let q_x = corrected_form_richardson(&h_c1, &x_c1, &g_vec, &xi_c1, 2e-4);
        let q_e = corrected_form_richardson(&h_c1, &x_c1, &g_vec, &eta_c1, 2e-4);
        let q_s = corrected_form_richardson(&h_c1, &x_c1, &g_vec, &sum_c1, 2e-4);
        let fd_off = 0.5 * (q_s - q_x - q_e);
        let op_off = 0.5 * (eta.dot(&sctx.hess_apply(xi)) + xi.dot(&sctx.hess_apply(eta)));
        println!(
            "tsr fd phi={p} polarized: FD {fd_off:.12} vs operator {op_off:.12} \
             (|d| = {:e}, <xi,eta> = {:e})",
            (fd_off - op_off).abs(),
            xi.dot(eta)
        );
        assert!(
            (fd_off - op_off).abs() <= 1e-5,
            "phi {p}: the polarized second difference {fd_off} disagrees with \
             1/2(<eta,(A+B)xi> + <xi,(A+B)eta>) = {op_off} — an operator wrong by an \
             antisymmetric piece passes every diagonal assertion above"
        );
        channels += 1;
        if b_live {
            curved += 1;
        }
    }
    assert_eq!(
        channels, 3,
        "anti-vacuity: d6_pg (2,2) has exactly three live SHIFTING channels (measured phi = \
         1, 2, 3); {channels} were exercised"
    );
    assert!(
        curved >= 1,
        "ANTI-VACUITY: none of the {channels} channels exercised carries a nonzero B^phi, so \
         every FD number above certifies A alone"
    );
}

#[test]
fn tsr_b_phi_is_self_adjoint_and_consistent_with_hpp_hmm() {
    let (space, t, v, ham, x) = d6_reference();
    let (bl, br) = tangent_bases(&x);
    let lambda = rayleigh_quotient(&ham, &x).re_s();
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_curved_channel(&space, &bl.ranks);
    let p = phi.index();
    let rperp = residual_norm(&ham, &x);
    println!(
        "tsr b-props: phi = {p} ({phi:?}), dim = {}, sigma_min = {:e}, ||r_perp|| = {rperp:e}, \
         lambda = {lambda}",
        tangent_dim_grids(&space, &bl.ranks, phi),
        min_schmidt(&bl, &br)
    );
    assert!(
        rperp > 1e-6,
        "fixture premise: B is linear in r_perp, so symmetry at ||r_perp|| = {rperp:e} would be \
         the symmetry of the zero operator"
    );

    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    let dirs = shifted_unit_tangents(&sctx, space_out, 0x7B30, 6);

    for j in 0..3 {
        let (xi, eta) = (&dirs[2 * j], &dirs[2 * j + 1]);
        let b_xi = sctx.b_apply(xi);
        let b_eta = sctx.b_apply(eta);
        let lhs = xi.dot(&b_eta);
        let rhs = b_xi.dot(eta);
        let s = lhs.abs().max(rhs.abs()).max(1.0);
        println!(
            "tsr b-props pair {j}: <xi,B eta> = {lhs:.15e} <B xi,eta> = {rhs:.15e} \
             (|d| = {:e}, ||B xi|| = {:e})",
            (lhs - rhs).abs(),
            b_xi.norm()
        );
        assert!(
            b_xi.norm() > 1e-8,
            "pair {j}: anti-vacuity, B^phi annihilated the direction"
        );
        assert!(
            (lhs - rhs).abs() <= 1e-10 * s,
            "pair {j}: B^phi is not self-adjoint ({lhs} vs {rhs}) — the S1 and S3 chains carry \
             R^-1 at DIFFERENT irrep grades (suffix t+phi vs prefix s) and swapping them is \
             silent everywhere but here"
        );
    }

    let basis = sctx.basis();
    let n = basis.len();
    assert!(n >= 2, "the channel must carry at least two directions");
    let (a_g, b_g) = sctx.grams(&basis);
    let asym = |g: &[f64]| -> f64 {
        let mut m = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                m = m.max((g[i * n + j] - g[j * n + i]).abs());
            }
        }
        m
    };
    let nb = gram_spectral_norm(&b_g, n);
    let na = gram_spectral_norm(&a_g, n);
    println!(
        "tsr b-props grams: n = {n}, ||A|| = {na:e}, ||B|| = {nb:e}, ||B||/||A|| = {:.3e}, \
         asym(A) = {:e}, asym(B) = {:e}",
        nb / na.max(f64::MIN_POSITIVE),
        asym(&a_g),
        asym(&b_g)
    );
    assert!(
        nb > 1e-3 * na,
        "ANTI-VACUITY: ||B^phi|| = {nb:e} vs ||A^phi|| = {na:e} at a truncated reference"
    );
    assert!(
        asym(&b_g) <= 1e-10 * nb,
        "the dense B Gram is not symmetric (asym {:e}, ||B|| {nb:e})",
        asym(&b_g)
    );
    assert!(
        asym(&a_g) <= 1e-10 * na,
        "the dense A Gram is not symmetric (asym {:e}, ||A|| {na:e})",
        asym(&a_g)
    );

    for (j, xi) in dirs.iter().take(3).enumerate() {
        let a_xi = sctx.a_apply(xi);
        let b_xi = sctx.b_apply(xi);
        let scale = a_xi.norm().max(b_xi.norm()).max(1.0);
        let hpp = sctx.hpp_apply(xi);
        let hmm = sctx.hmm_apply(xi);
        let hess = sctx.hess_apply(xi);
        assert!(
            hpp.sub(&a_xi.add(&b_xi)).norm() <= 1e-12 * scale,
            "dir {j}: hpp_apply must be A + B (= H^rr)"
        );
        assert!(
            hmm.sub(&a_xi.sub(&b_xi)).norm() <= 1e-12 * scale,
            "dir {j}: hmm_apply must be A - B (= H^ii)"
        );
        assert!(
            hess.sub(&hpp).norm() <= 1e-12 * scale,
            "dir {j}: hess_apply and hpp_apply must be the same operator"
        );
        assert!(
            hpp.sub(&hmm).norm() > 1e-6 * scale,
            "dir {j}: ANTI-VACUITY — H^rr and H^ii coincide, so B is zero on this direction and \
             the three assertions above are statements about A"
        );
    }

    let z = normal_residual(&ham, &bl, &br, &x, lambda);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7B31);
    let dir0: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let xi0 = TtTangent::project(&bl, &br, &dir0);
    assert!(xi0.norm() > 1e-8, "the IDENTITY probe direction is zero");
    let via_ctx = ctx.b_apply(&xi0);
    let via_assemble =
        weingarten_assemble_cached(&bl, &br, &z, &xi0, ctx.bond_gauges(), Terms::all());
    let d_id = via_ctx.sub(&via_assemble).norm();
    println!(
        "tsr b-props IDENTITY: ||B xi|| = {:e}, two-decomposition |d| = {d_id:e}",
        via_ctx.norm()
    );
    assert!(
        via_ctx.norm() > 1e-8,
        "anti-vacuity: B is zero on the IDENTITY probe direction"
    );
    assert!(
        d_id <= 1e-10 * via_ctx.norm(),
        "at phi = IDENTITY the implicit/explicit two-source split and the single explicit-Z \
         assembly must agree ({d_id:e} vs ||B xi|| = {:e}) — they are two decompositions of the \
         same Z through the same generalized body",
        via_ctx.norm()
    );
}

#[test]
fn tsr_b_phi_vanishes_at_an_exact_eigenvector() {
    let pg = fx4_pg();
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
    let (t, v) = fx4_tv(pg);
    let (x, e0, _levels) = exact_typed_ground(space, &t, &v);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let (bl, br) = tangent_bases(&x);
    let rperp = residual_norm(&ham, &x);
    println!(
        "tsr b-zero: E_0 = {e0}, ||r_perp|| = {rperp:e}, sigma_min = {:e}, live = {:?}",
        min_schmidt(&bl, &br),
        live_channels(&space, &bl.ranks)
    );
    assert!(
        rperp <= 1e-12,
        "fixture premise: the exact typed ground must sit at the residual floor (measured \
         3.37e-16; got {rperp:e}); without it 'B vanishes' is not the claim being tested"
    );
    println!(
        "tsr b-zero: T^phi slot sites = {:?} — a SINGLE-SITE channel makes B vanish structurally \
         too, so this arm does not separate the two mechanisms (see the doc comment)",
        slot_sites(&space, &bl.ranks, first_shifting_channel(&space, &bl.ranks))
    );

    let ctx = weingarten_operator(&ham, &x);
    let phi = first_shifting_channel(&space, &bl.ranks);
    let p = phi.index();
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    assert!(sctx.dim() > 0, "phi {p}: the channel must be non-empty");

    for (j, xi) in shifted_unit_tangents(&sctx, space_out, 0x7B40, 3)
        .iter()
        .enumerate()
    {
        let a_xi = sctx.a_apply(xi);
        let b_xi = sctx.b_apply(xi);
        println!(
            "tsr b-zero phi={p} dir={j}: ||A xi|| = {:e}, ||B xi|| = {:e}",
            a_xi.norm(),
            b_xi.norm()
        );
        assert!(
            a_xi.norm() > 1e-3,
            "phi {p} dir {j}: ANTI-VACUITY — ||A^phi xi|| = {:e}; with a vanishing A the bound \
             below is satisfied by any B at all",
            a_xi.norm()
        );
        assert!(
            b_xi.norm() <= 1e-9 * a_xi.norm(),
            "phi {p} dir {j}: ||B^phi xi|| = {:e} at r_perp = 0 — B is LINEAR in the normal \
             residual and must vanish here (||A^phi xi|| = {:e})",
            b_xi.norm(),
            a_xi.norm()
        );
    }

    let basis = sctx.basis();
    let n = basis.len();
    let (a_g, b_g) = sctx.grams(&basis);
    let (na, nb) = (gram_spectral_norm(&a_g, n), gram_spectral_norm(&b_g, n));
    println!("tsr b-zero phi={p}: n = {n}, ||A|| = {na:e}, ||B|| = {nb:e}");
    assert!(na > 1e-3, "anti-vacuity: the A Gram is identically zero");
    assert!(
        nb <= 1e-9 * na,
        "phi {p}: the whole B Gram is {nb:e} at r_perp = 0 (||A|| = {na:e})"
    );
}

#[test]
fn tsr_williamson_union_matches_the_untyped_spectrum() {
    struct Leg {
        label: &'static str,
        space: QnSpace,
        t: CoeffTensor<f64>,
        v: CoeffTensor<f64>,
        ham: SparseHamiltonian<f64>,
        x: TtVector<f64>,
        want_live: usize,
    }
    let legs = {
        let (space, t, v, ham, x) = fx4_minimizer();
        let small = Leg {
            label: "fx4 (2,1) minimizer d=4",
            space,
            t,
            v,
            ham,
            x,
            want_live: 2,
        };
        let (space, t, v, ham, x) = d6_minimizer();
        let d6 = Leg {
            label: "d=6 minimizer, all four channels live",
            space,
            t,
            v,
            ham,
            x,
            want_live: 4,
        };
        [small, d6]
    };

    for leg in legs {
        let Leg {
            label,
            space,
            t,
            v,
            ham,
            x,
            want_live,
        } = leg;
        let (bl, br) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        let live = live_channels(&space, &bl.ranks);
        assert_eq!(
            live.len(),
            want_live,
            "[{label}] anti-vacuity: measured live-channel inventory is {want_live}, got {live:?}"
        );

        let space_c1 = QnSpace::new(space.d, space.nup, space.ndn);
        let x_c1 = x.forget_point_group();
        let h_c1 = SparseHamiltonian::<f64>::new(space_c1, &t, &v, None, f64::EPSILON);
        let ctx_c1 = weingarten_operator(&h_c1, &x_c1);
        let (bl_c1, br_c1) = ctx_c1.bases();
        println!(
            "tsr williamson [{label}]: sigma_min = {:e}, sigma_min(c1) = {:e}, \
             ||r_perp|| = {:e}, live = {live:?}",
            min_schmidt(&bl, &br),
            min_schmidt(bl_c1, br_c1),
            residual_norm(&ham, &x)
        );

        let dim_untyped = tangent_dim(bl_c1);
        let dim_0 = tangent_dim_grids(&space, &bl.ranks, Irrep::IDENTITY);
        let dim_shifted: usize = live
            .iter()
            .filter(|&&p| p != Irrep::IDENTITY)
            .map(|&p| tangent_dim_grids(&space, &bl.ranks, p))
            .sum();
        assert_eq!(
            dim_0 + dim_shifted - 1,
            dim_untyped - 1,
            "[{label}] dimension identity: (dim T^0 - 1) + sum_{{phi != 0}} dim T^phi = {} but \
             tangent_dim(x'') - 1 = {}",
            dim_0 + dim_shifted - 1,
            dim_untyped - 1
        );

        let basis_c1 = tangent_basis(bl_c1, br_c1, true);
        let basis_0 = tangent_basis(&bl, &br, true);
        let shifted: Vec<(Irrep, Vec<TtTangent<f64>>)> = live
            .iter()
            .copied()
            .filter(|&p| p != Irrep::IDENTITY)
            .map(|p| (p, tangent_basis_shifted(&bl, &br, p)))
            .collect();
        let n_shifted: usize = shifted.iter().map(|(_, b)| b.len()).sum();
        assert_eq!(
            basis_c1.len(),
            basis_0.len() + n_shifted,
            "[{label}] the untyped radial-deflated basis has {} directions but the blocks sum to \
             {} ({} in phi = 0 + {n_shifted} shifted) — comparing spectra over different counts \
             is meaningless",
            basis_c1.len(),
            basis_0.len() + n_shifted,
            basis_0.len()
        );

        let spec_c1 = symplectic_eigenvalues(&ctx_c1, &basis_c1);
        let spec_0 = symplectic_eigenvalues(&ctx, &basis_0);
        println!(
            "tsr williamson [{label}]: untyped {} (n = {}), phi=0 {} (n = {})",
            spectrum_label(&spec_c1),
            basis_c1.len(),
            spectrum_label(&spec_0),
            basis_0.len()
        );
        let w_untyped = signed_omega2_or_fail(&spec_c1, &format!("[{label}] untyped side"));
        let mut union = signed_omega2_or_fail(&spec_0, &format!("[{label}] phi = 0 block"));

        for (p, basis_phi) in &shifted {
            let space_out = space.with_target(space.target.fuse(*p));
            let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
            let sctx = SectorCtx::new(&ctx, &h_out, *p);
            let spec = sctx.symplectic_eigenvalues(basis_phi);
            println!(
                "tsr williamson [{label}] phi={}: {} (n = {})",
                p.index(),
                spectrum_label(&spec),
                basis_phi.len()
            );
            union.extend(signed_omega2_or_fail(
                &spec,
                &format!("[{label}] phi = {} block", p.index()),
            ));
        }

        union.sort_by(|a, b| a.partial_cmp(b).expect("finite omega^2"));
        let scale = w_untyped.iter().fold(0.0f64, |m, z| m.max(z.abs()));
        println!(
            "tsr williamson [{label}]: |union| = {}, |untyped| = {}, max|omega^2| = {scale:e}, \
             worst = {:e}",
            union.len(),
            w_untyped.len(),
            union
                .iter()
                .zip(w_untyped.iter())
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
        );
        assert!(
            scale > 1e-6,
            "[{label}] anti-vacuity: the untyped omega^2 spectrum is identically zero"
        );
        assert_multiset_eq(
            &union,
            &w_untyped,
            1e-8,
            &format!("[{label}] union_phi omega^2_phi vs the untyped signed spectrum"),
        );
    }
}

#[test]
fn tsr_rpa_readout_routes_instability_without_publishing_omega() {
    let pg = synth_pg();
    let gamma0 = Irrep::new(1);
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, gamma0);
    let (t, v) = synth_tv(pg);
    let (x, e0, _lv) = exact_typed_ground(space, &t, &v);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let (bl, br) = tangent_bases(&x);

    let lower = dense_levels(D4, &t, &v, pg, F_NUP, F_NDN, Irrep::IDENTITY);
    assert!(
        lower[0] < e0 - 1e-3,
        "fixture premise: irrep 0's ground ({}) must lie BELOW the reference ({e0})",
        lower[0]
    );

    let ctx = weingarten_operator(&ham, &x);
    let phi = Irrep::new(1);
    let space_out = space.with_target(gamma0.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    assert!(sctx.dim() > 0, "the lower channel must not be empty");

    let basis = sctx.basis();
    let n = basis.len();
    let (a_g, b_g) = sctx.grams(&basis);
    let dense = sctx.symplectic_eigenvalues(&basis);
    println!(
        "tsr rpa-instability: dim {} n {n} sigma_min {:e} ||r_perp|| {:e} ||A|| {:e} ||B|| {:e} \
         dense arm {} lowest eig(A) {:?}",
        sctx.dim(),
        min_schmidt(&bl, &br),
        residual_norm(&ham, &x),
        gram_spectral_norm(&a_g, n),
        gram_spectral_norm(&b_g, n),
        spectrum_label(&dense),
        gram_eigs(&a_g, n).first().copied()
    );
    assert!(
        gram_eigs(&a_g, n)[0] < -1e-3,
        "fixture premise: A^phi must be indefinite here (lowest eig = {})",
        gram_eigs(&a_g, n)[0]
    );
    match &dense {
        SymplecticSpectrum::Stable { omega } => panic!(
            "the dense readout came back Stable ({omega:?}) at a reference that is NOT the lowest \
             state of its target channel — an indefinite A^phi cannot leave both H^rr and H^ii PD"
        ),
        SymplecticSpectrum::Resonant { signed } => {
            assert!(
                signed.n_neg > 0,
                "Resonant with n_neg = 0 is Stable by another name"
            );
            assert!(
                signed.omega2.iter().all(|z| z.is_finite()),
                "the signed omega^2 must be finite: {:?}",
                signed.omega2
            );
        }
        SymplecticSpectrum::Unstable {
            reason,
            generator_eigs,
        } => {
            println!(
                "tsr rpa-instability: Unstable({reason:?}), 2n = {}",
                generator_eigs.len()
            );
            assert_eq!(
                generator_eigs.len(),
                2 * n,
                "the Unstable arm must report all 2n generator eigenvalues"
            );
            assert!(
                dense.omega_squared().is_none() && dense.unstable_pairs().is_none(),
                "no real omega^2 and no pair count exist on the Unstable arm"
            );
        }
    }

    let nb = gram_spectral_norm(&b_g, n);
    let na = gram_spectral_norm(&a_g, n);
    let eig_a = gram_eigs(&a_g, n);
    let min_abs_a = eig_a.iter().fold(f64::INFINITY, |m, z| m.min(z.abs()));
    println!(
        "tsr rpa-instability: ||B|| / ||A|| = {:e}, eig(A) = {eig_a:?}, min |a| = {min_abs_a:e}",
        nb / na
    );
    assert!(
        nb <= 1e-12 * na,
        "fixture premise: B == 0 at the exact reference (||B|| = {nb:e}, ||A|| = {na:e})"
    );
    assert!(
        min_abs_a >= 1e-4,
        "fixture premise: no eigenvalue of A^phi near 0 ({min_abs_a:e}), so every a^2 is far from \
         the sign of roundoff"
    );
    assert_eq!(
        n,
        sctx.dim(),
        "fixture premise: the basis spans T^phi, so k = dim is an EXHAUSTIVE solve"
    );
    let inertia = pencil_inertia(&a_g, &b_g, n);
    let negatives_of_a: Vec<f64> = eig_a.iter().copied().filter(|z| *z < 0.0).collect();

    let opts = SectorOpts {
        readout: SectorReadout::Rpa,
        kernel: RpaKernel::signed(RpaMetric::Auto),
        ..SectorOpts::default()
    };
    let out = sctx.sector_lowest(sctx.dim(), &opts);
    println!(
        "tsr rpa-instability iterative: dim {} omega2 {:?} signature {:?} converged {} iters {} \
         applies {} sigma_min {:e} negatives {} all certified {} flagged {} report {:?} index {:?}",
        out.dim,
        out.omega2(),
        krein_signs(out.rpa()),
        out.converged,
        out.iterations,
        out.applies,
        out.sigma_min,
        out.negatives().len(),
        out.all_negatives_certified(),
        out.metric_indefinite(),
        metric_report(out.rpa()),
        out.index_lower_bound()
    );
    assert_eq!(out.phi, phi);
    assert_eq!(out.target, space_out.target);
    assert_eq!(
        out.dim, n,
        "the readout must REPORT the channel's dim (the full basis length here)"
    );
    assert!(
        out.theta().is_none(),
        "an Rpa readout must not publish theta(): omega^2 is not an energy"
    );
    assert!(
        out.omega().is_none(),
        "no real omega may be published at a reference whose target channel holds a lower state \
         — a resonance is omega = i*sqrt(|omega^2|), which a Vec<f64> of omega cannot carry: {:?}",
        out.omega()
    );
    assert!(
        out.applies > 0,
        "the solve spent applies before answering and must report them honestly"
    );
    let w = out
        .omega2()
        .expect("an Rpa readout reports in Rpa units: an omega2, never None");
    assert!(
        out.metric_indefinite(),
        "the Ii metric W = A - B = A is indefinite here: the solve must be FLAGGED"
    );
    assert_eq!(
        krein_signs(out.rpa()).len(),
        w.len(),
        "one Krein signature per omega^2 entry"
    );
    assert_eq!(
        residuals_of(out.rpa()).len(),
        w.len(),
        "one residual per omega^2 entry"
    );
    assert!(w.iter().all(|z| z.is_finite()), "never NaN: {w:?}");
    assert!(
        w.iter().all(|&z| z >= 0.0),
        "B == 0 => M = A^2 is PSD: no negative omega^2 may appear ({w:?})"
    );
    assert!(
        out.negatives().is_empty(),
        "nothing negative, so nothing to certify: {:?}",
        out.negatives()
    );
    assert!(
        out.converged,
        "the exact point, exhaustive basis: the solve must converge (residuals {:?})",
        residuals_of(out.rpa())
    );
    assert_eq!(
        w.len(),
        n,
        "k = dim with no negative omega^2: every a^2 is listed exactly once"
    );
    let minus: Vec<f64> = w
        .iter()
        .zip(&krein_signs(out.rpa()))
        .filter(|(_, s)| **s == -1)
        .map(|(z, _)| *z)
        .collect();
    assert!(
        !minus.is_empty(),
        "at least one entry carries s = -1: the lower state, seen from x"
    );
    assert_eq!(
        minus.len(),
        negatives_of_a.len(),
        "one s = -1 entry per negative eigenvalue of A^phi"
    );
    let mut used = vec![false; negatives_of_a.len()];
    for &theta in &minus {
        let hit = negatives_of_a
            .iter()
            .enumerate()
            .filter(|(i, _)| !used[*i])
            .map(|(i, a)| (i, (theta - a * a).abs()))
            .min_by(|p, q| p.1.total_cmp(&q.1))
            .expect("an unmatched negative eigenvalue");
        assert!(
            hit.1 <= 1e-6 * theta.max(1.0),
            "s = -1 theta {theta} is not a^2 of an unmatched negative eigenvalue of A^phi \
             ({negatives_of_a:?})"
        );
        used[hit.0] = true;
    }
    let mut want: Vec<f64> = eig_a.iter().map(|a| a * a).collect();
    want.sort_by(f64::total_cmp);
    let mut got = w.to_vec();
    got.sort_by(f64::total_cmp);
    assert_multiset_eq(&got, &want, 1e-6 * na.max(1.0), "omega^2 vs a^2 at B == 0");
    assert_eq!(
        inertia.lhs(),
        inertia.rhs(),
        "dense counting convention (pencil_inertia): {inertia:?}"
    );
    assert_eq!(
        out.index_lower_bound(),
        Some(inertia.rhs()),
        "exhaustion: the projected pencil's inertia is n-(A+B) + n-(A-B) = 2 n-(A) ({inertia:?})"
    );
}

#[test]
fn tsr_sector_lowest_matches_the_dense_symplectic_driver() {
    for (label, (space, t, v, ham, x)) in [
        ("fx4 (2,1) minimizer d=4", fx4_minimizer()),
        ("d=6 minimizer", d6_minimizer()),
    ] {
        let (bl, br) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        let phi = curved_channels(&space, &bl.ranks)
            .first()
            .copied()
            .unwrap_or_else(|| first_shifting_channel(&space, &bl.ranks));
        let p = phi.index();
        let space_out = space.with_target(space.target.fuse(phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        let basis = sctx.basis();
        let n = basis.len();
        assert!(
            n >= 3,
            "[{label}] phi {p}: dim {n} is too small to iterate on"
        );

        let dense = sctx.symplectic_eigenvalues(&basis);
        println!(
            "tsr iterative [{label}] phi={p}: n = {n}, dim = {}, sigma_min = {:e}, \
             ||r_perp|| = {:e}, dense {}",
            sctx.dim(),
            min_schmidt(&bl, &br),
            residual_norm(&ham, &x),
            spectrum_label(&dense)
        );
        let w_dense = signed_omega2_or_fail(&dense, &format!("[{label}] phi = {p} dense"));
        let scale = w_dense.iter().fold(0.0f64, |m, z| m.max(z.abs()));
        assert!(
            scale > 1e-6,
            "[{label}] phi {p}: anti-vacuity, the dense omega^2 spectrum is zero"
        );

        let k = 2usize;
        let run = |metric: RpaMetric| {
            let opts = SectorOpts {
                readout: SectorReadout::Rpa,
                kernel: RpaKernel::signed(metric),
                ..SectorOpts::default()
            };
            sctx.sector_lowest(k, &opts)
        };
        let auto = run(RpaMetric::Auto);
        let resolved = match &auto.spectrum {
            SectorSpectrum::Rpa(r) => r.side(),
            SectorSpectrum::Tda { .. } => {
                panic!("[{label}] phi {p}: sector_lowest returned a Tda spectrum")
            }
        };
        println!(
            "tsr iterative [{label}] phi={p} Auto: resolved metric {resolved:?}, converged {}, \
             iters {}, applies {}, omega2 {:?}, negatives {} (all certified {})",
            auto.converged,
            auto.iterations,
            auto.applies,
            auto.omega2(),
            auto.negatives().len(),
            auto.all_negatives_certified()
        );
        assert_ne!(
            resolved,
            RpaMetric::Auto,
            "[{label}] phi {p}: the RESOLVED metric must be reported as Ii or Rr, never as the \
             unresolved Auto — an Auto reaching a consumer is the wrapper bug the preconditioner \
             factory panics on"
        );
        assert!(
            auto.converged,
            "[{label}] phi {p}: the Auto run did not converge in {} iterations / {} applies \
             (omega2 = {:?}); the assertions below are gated on convergence, so an unconverged \
             run would make this arm vacuous",
            auto.iterations,
            auto.applies,
            auto.omega2()
        );
        let w_it = auto.omega2().expect("an Rpa readout publishes omega2()");
        println!(
            "tsr iterative [{label}] phi={p}: dense label {} (recorded, Q12); signature {:?}, \
             report {:?}, index {:?}",
            spectrum_label(&dense),
            krein_signs(auto.rpa()),
            metric_report(auto.rpa()).map(|(n, g, _)| (n, g)),
            auto.index_lower_bound()
        );
        match &dense {
            SymplecticSpectrum::Stable { .. } => {
                assert!(
                    !w_it.is_empty() && w_it.len() <= k,
                    "[{label}] phi {p}: expected 1..={k} roots, got {}",
                    w_it.len()
                );
                assert!(
                    !auto.metric_indefinite() && !auto.metric_indefinite(),
                    "[{label}] phi {p}: a Stable channel's signed solve meets no indefinite metric"
                );
                assert_eq!(
                    krein_signs(auto.rpa()),
                    vec![1i8; w_it.len()],
                    "[{label}] phi {p}: every s = +1 on a Stable channel"
                );
                assert_eq!(
                    auto.index_lower_bound(),
                    Some(0),
                    "[{label}] phi {p}: a Stable channel has no inertia"
                );
            }
            SymplecticSpectrum::Resonant { signed } => assert_eq!(
                w_it.len(),
                signed.n_neg + k,
                "[{label}] phi {p}: a Resonant leg lists every negative plus the k lowest (Q12)"
            ),
            SymplecticSpectrum::Unstable { .. } => {
                unreachable!("signed_omega2_or_fail refused the Unstable arm above")
            }
        }
        assert_eq!(
            residuals_of(auto.rpa()).len(),
            w_it.len(),
            "one residual per omega^2 entry"
        );
        let tol = 10.0 * SectorOpts::default().tol * scale.max(1.0);
        for (j, w) in w_it.iter().enumerate() {
            assert!(
                (w - w_dense[j]).abs() <= tol,
                "[{label}] phi {p}: iterative omega^2[{j}] = {w} vs dense {} (tol {tol:e}, \
                 scale {scale:e})",
                w_dense[j]
            );
        }
        assert!(
            auto.omega().is_some()
                == (w_it.iter().all(|&z| z > 0.0) && auto.negatives().is_empty()),
            "[{label}] phi {p}: omega() must be Some exactly when every omega^2 is positive and \
             no negative witness is outstanding (omega2 = {w_it:?}, negatives = {})",
            auto.negatives().len()
        );

        let ii = run(RpaMetric::Ii);
        let rr = run(RpaMetric::Rr);
        println!(
            "tsr iterative [{label}] phi={p} forced: Ii converged {} ({} applies) omega2 {:?} | \
             Rr converged {} ({} applies) omega2 {:?}",
            ii.converged,
            ii.applies,
            ii.omega2(),
            rr.converged,
            rr.applies,
            rr.omega2()
        );
        assert!(
            ii.converged || rr.converged,
            "[{label}] phi {p}: NEITHER forced metric converged, so the free-choice claim was \
             never exercised"
        );
        if matches!(dense, SymplecticSpectrum::Stable { .. }) {
            assert!(
                flat_bits(&w_it) == flat_bits(&ii.omega2().expect("Rpa"))
                    && (auto.iterations, auto.applies) == (ii.iterations, ii.applies)
                    && flat_bits(&residuals_of(auto.rpa())) == flat_bits(&residuals_of(ii.rpa())),
                "[{label}] phi {p}: Auto must be the Ii call bit for bit on a Stable channel"
            );
        }
        if ii.converged && rr.converged {
            let (a, b) = (ii.omega2().expect("Rpa"), rr.omega2().expect("Rpa"));
            assert_multiset_eq(
                &a,
                &b,
                tol,
                &format!(
                    "[{label}] phi {p}: forced Ii vs forced Rr — spec(H^rr.H^ii) = \
                     spec(H^ii.H^rr), so which block carries the metric is a free choice"
                ),
            );
        }
    }
}

#[test]
fn tsr_fused_a_apply_matches_the_unfused_reference() {
    struct Leg {
        label: &'static str,
        space: QnSpace,
        t: CoeffTensor<f64>,
        v: CoeffTensor<f64>,
        ham: SparseHamiltonian<f64>,
        x: TtVector<f64>,
        want_live: usize,
        want_multisite: usize,
        full_basis: bool,
    }

    let pg4 = fx4_pg();
    let legs = {
        let full = {
            let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg4, Irrep::IDENTITY);
            let (t, v) = fx4_tv(pg4);
            let (ham, x) = typed_minimizer(space, &t, &v, 0x7505, 1, 2);
            Leg {
                label: "full-rank (2,1) minimizer",
                space,
                t,
                v,
                ham,
                x,
                want_live: 2,
                want_multisite: 0,
                full_basis: true,
            }
        };
        let trunc = {
            let (space, t, v, ham, x) = truncated_reference();
            Leg {
                label: "truncated (2,2)",
                space,
                t,
                v,
                ham,
                x,
                want_live: 2,
                want_multisite: 0,
                full_basis: true,
            }
        };
        let d6 = {
            let (space, t, v, ham, x) = d6_reference();
            Leg {
                label: "d = 6, all four channels live",
                space,
                t,
                v,
                ham,
                x,
                want_live: 4,
                want_multisite: 3,
                full_basis: false,
            }
        };
        [full, trunc, d6]
    };

    let mut multisite_total = 0usize;
    for leg in legs {
        let Leg {
            label,
            space,
            t,
            v,
            ham,
            x,
            want_live,
            want_multisite,
            full_basis,
        } = leg;
        let (bl, br) = tangent_bases(&x);
        let lambda = rayleigh_quotient(&ham, &x).re_s();
        let ctx = weingarten_operator(&ham, &x);
        let live = live_channels(&space, &bl.ranks);
        println!(
            "tsr fused apply [{label}]: live {live:?}, curved {:?}, ||r_perp|| = {:e}",
            curved_channels(&space, &bl.ranks),
            residual_norm(&ham, &x)
        );
        assert_eq!(
            live.len(),
            want_live,
            "[{label}] fixture premise: the measured live-channel inventory is {want_live}"
        );

        let mut multisite = 0usize;
        let mut checked_channels = 0usize;
        for phi in live {
            if phi == Irrep::IDENTITY {
                continue;
            }
            let slots = slot_sites(&space, &bl.ranks, phi);
            if slots.len() >= 2 {
                multisite += 1;
            }
            let space_out = space.with_target(space.target.fuse(phi));
            let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
            let sctx = SectorCtx::new(&ctx, &h_out, phi);
            let basis = sctx.basis();
            assert!(
                !basis.is_empty(),
                "[{label}] phi {}: a LIVE channel must carry a basis",
                phi.index()
            );
            let step = if full_basis {
                1
            } else {
                basis.len().div_ceil(8)
            };
            let idx: Vec<usize> = (0..basis.len()).step_by(step).collect();
            assert!(
                idx.len() >= 4,
                "[{label}] phi {}: only {} of {} directions would be checked — the identity is \
                 per-direction, but a subset this small stops discriminating a per-site slip",
                phi.index(),
                idx.len(),
                basis.len()
            );
            checked_channels += 1;

            let mut worst = 0.0f64;
            let mut weakest = f64::INFINITY;
            for &j in &idx {
                let xi = &basis[j];
                let fused = sctx.a_apply_fused(xi);
                let exact = sctx.a_apply_exact(xi);
                let free = effham_apply_shifted(&h_out, &bl, &br, phi, lambda, xi);

                assert_eq!(exact.num_coeffs(), free.num_coeffs());
                for i in 0..free.num_coeffs() {
                    assert_eq!(
                        exact.coeff(i),
                        free.coeff(i),
                        "[{label}] phi {}: a_apply_exact must BE effham_apply_shifted, coeff {i}",
                        phi.index()
                    );
                }

                let en = exact.norm();
                weakest = weakest.min(en);
                let rel = fused.sub(&exact).norm() / en.max(1.0);
                worst = worst.max(rel);

                assert_eq!(
                    fused.phi(),
                    phi,
                    "[{label}] phi {}: the image's flux",
                    phi.index()
                );
                assert_eq!(
                    fused.rank(),
                    xi.rank(),
                    "[{label}] phi {}: the image must live on the channel's own ROW grids",
                    phi.index()
                );
                assert_eq!(
                    fused.rank_col(),
                    xi.rank_col(),
                    "[{label}] phi {}: the image must live on the channel's own COLUMN grids, or \
                     A^phi's range is not T^phi",
                    phi.index()
                );
                assert_eq!(
                    fused.num_coeffs(),
                    exact.num_coeffs(),
                    "[{label}] phi {}: coefficient count",
                    phi.index()
                );
                for k in 0..space.d {
                    let cf = fused.component(k);
                    let ce = exact.component(k);
                    assert_eq!(
                        cf.space,
                        space_out,
                        "[{label}] phi {}: component {k} must come out stamped space_out — \
                         `sketched_hamiltonian` allocates in `h_out`'s space and the fused route \
                         must not restamp",
                        phi.index()
                    );
                    assert_eq!(
                        cf.row_ranks,
                        ce.row_ranks,
                        "[{label}] phi {}: component {k} row grid",
                        phi.index()
                    );
                    assert_eq!(
                        cf.col_ranks,
                        ce.col_ranks,
                        "[{label}] phi {}: component {k} col grid — a 0-column block at one \
                         sector passes every norm bound and is visible only here",
                        phi.index()
                    );
                    for f in Family::ALL {
                        for s in space_out.bond_sectors(k) {
                            assert_eq!(
                                cf.block_dims(f, s),
                                ce.block_dims(f, s),
                                "[{label}] phi {}: component {k} block dims at ({f:?}, {s:?})",
                                phi.index()
                            );
                        }
                    }
                }
            }
            println!(
                "tsr fused apply [{label}] phi={}: dim {}, slots {slots:?}, {} of {} directions, \
                 weakest ||A xi|| = {weakest:e}, worst rel = {worst:e}",
                phi.index(),
                sctx.dim(),
                idx.len(),
                basis.len()
            );
            assert!(
                weakest > 1e-3,
                "[{label}] phi {}: ANTI-VACUITY — A^phi annihilated a checked direction \
                 (||A xi|| = {weakest:e}); a relative bound over a zero image proves nothing",
                phi.index()
            );
            assert!(
                worst < 1e-11,
                "[{label}] phi {}: fused vs unfused shifted A-apply, worst relative deviation \
                 {worst:e} over {} directions — the contract is reassociation (t8i model), so \
                 anything above 1e-11 is a different operator, not roundoff",
                phi.index(),
                idx.len()
            );
        }
        assert!(
            checked_channels > 0,
            "[{label}] no shifted channel was reached"
        );
        assert_eq!(
            multisite,
            want_multisite,
            "[{label}] fixture premise: the measured multi-site shifted-channel count is \
             {want_multisite} (slot_sites per live channel: {:?})",
            live_channels(&space, &bl.ranks)
                .iter()
                .map(|&p| (p, slot_sites(&space, &bl.ranks, p)))
                .collect::<Vec<_>>()
        );
        multisite_total += multisite;
    }
    assert!(
        multisite_total >= 1,
        "ANTI-VACUITY: no leg reached a shifted channel whose T^phi varies two or more cores, so \
         the L-tilde/R-tilde TRANSPORT recursions were never fed non-zero data by any leg"
    );
}

#[test]
fn tsr_fused_a_gram_is_symmetric_and_matches_the_untyped_block() {
    let pg = fx4_pg();
    let legs: [(&str, Fixture); 2] = [
        ("full-rank (2,1) minimizer", {
            let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
            let (t, v) = fx4_tv(pg);
            let (ham, x) = typed_minimizer(space, &t, &v, 0x7505, 1, 2);
            (space, t, v, ham, x)
        }),
        ("truncated (2,2)", truncated_reference()),
    ];

    for (label, (space, t, v, ham, x)) in legs {
        let (bl, _br) = tangent_bases(&x);
        let lambda = rayleigh_quotient(&ham, &x).re_s();
        let ctx = weingarten_operator(&ham, &x);

        let space_c1 = QnSpace::new(space.d, space.nup, space.ndn);
        let x_c1 = x.forget_point_group();
        let h_c1 = SparseHamiltonian::<f64>::new(space_c1, &t, &v, None, f64::EPSILON);
        let lam_c1 = rayleigh_quotient(&h_c1, &x_c1).re_s();
        assert!(
            (lam_c1 - lambda).abs() <= 1e-10 * lambda.abs().max(1.0),
            "[{label}] the embedding changed the Rayleigh quotient: {lam_c1} vs {lambda}"
        );
        let ctx_c1 = weingarten_operator(&h_c1, &x_c1);
        let (bl_c1, br_c1) = ctx_c1.bases();
        let basis_c1 = tangent_basis(bl_c1, br_c1, false);
        assert!(
            !basis_c1.is_empty(),
            "[{label}] empty untyped tangent basis"
        );

        let mut checked = 0usize;
        for phi in live_channels(&space, &bl.ranks) {
            if phi == Irrep::IDENTITY {
                continue;
            }
            let p = phi.index();
            let gamma = space.target.fuse(phi);
            let space_out = space.with_target(gamma);
            let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
            let sctx = SectorCtx::new(&ctx, &h_out, phi);
            let basis = sctx.basis();
            let n = basis.len();
            assert!(
                n > 0,
                "[{label}] phi {p}: a live channel must carry a basis"
            );

            let cands: Vec<TtTangent<f64>> = basis_c1
                .iter()
                .map(|b| project_irrep(b, bl_c1, br_c1, pg, gamma).0)
                .collect();
            let block = mgs_tangents(cands);
            assert_eq!(
                block.len(),
                n,
                "[{label}] phi {p}: the untyped irrep block and T^phi disagree on the DIMENSION, \
                 so comparing spectra would be meaningless"
            );
            checked += 1;

            let (a_c1, _b_c1) = hessian_grams(&ctx_c1, &block);
            let e_c1 = gram_eigs(&a_c1, n);
            let scale = e_c1.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
            assert!(
                scale > 1e-3,
                "[{label}] phi {p}: ANTI-VACUITY, the untyped block spectrum is identically 0"
            );

            let g_fused = sctx.a_gram(&basis);
            let (g_paired, _) = sctx.grams(&basis);
            assert_eq!(g_fused.len(), n * n);
            assert_eq!(g_paired.len(), n * n);

            for (name, g) in [("a_gram", &g_fused), ("grams().0", &g_paired)] {
                let e = gram_eigs(g, n);
                for (j, (a, b)) in e_c1.iter().zip(e.iter()).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-9 * scale,
                        "[{label}] phi {p}: {name} eigenvalue {j} is {b} but the untyped \
                         hessian_grams block says {a} (scale {scale:e})"
                    );
                }
            }
            let worst_pair = g_fused
                .iter()
                .zip(g_paired.iter())
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                worst_pair <= 1e-12 * scale,
                "[{label}] phi {p}: a_gram and grams().0 disagree by {worst_pair:e} — one of the \
                 two shifted A drivers is not on the fused route"
            );
            let na = gram_spectral_norm(&g_fused, n);
            let asym = (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .fold(0.0f64, |m, (i, j)| {
                    m.max((g_fused[i * n + j] - g_fused[j * n + i]).abs())
                });
            println!(
                "tsr fused gram [{label}] phi={p}: n = {n}, scale = {scale:e}, ||A|| = {na:e}, \
                 ||G - G^T|| = {asym:e}, worst |a_gram - grams| = {worst_pair:e}"
            );
            assert!(
                asym <= 1e-11 * na.max(1.0),
                "[{label}] phi {p}: the FUSED A^phi Gram is not symmetric ({asym:e} at ||A|| = \
                 {na:e}) — L-tilde folds (bra u, ket v) and R-tilde folds (bra v, ket u), so a \
                 u/v slip breaks self-adjointness and NO eigenvalue comparison against a \
                 symmetrized reference sees it"
            );
        }
        assert!(
            checked >= 1,
            "[{label}] no live shifted channel — the arm would be vacuous"
        );
    }
}

#[test]
fn tsr_fused_a_apply_at_the_minimal_and_boundary_shapes() {
    {
        let (space, t, v, ham, x) = d3_minimal_channel();
        let (bl, br) = tangent_bases(&x);
        let lambda = rayleigh_quotient(&ham, &x).re_s();
        let ctx = weingarten_operator(&ham, &x);
        let phi = first_shifting_channel(&space, &bl.ranks);
        let space_out = space.with_target(space.target.fuse(phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        let basis = sctx.basis();
        let slots = slot_sites(&space, &bl.ranks, phi);
        println!(
            "tsr fused d3: phi = {phi:?}, dim = {}, basis = {}, slots = {slots:?}, ||r_perp|| = \
             {:e}, per-component coeffs = {:?}",
            sctx.dim(),
            basis.len(),
            residual_norm(&ham, &x),
            (0..3)
                .map(|k| basis[0].component(k).row_ranks.total())
                .collect::<Vec<_>>()
        );
        assert!(
            basis.len() >= 2,
            "ANTI-VACUITY: the d = 3 channel carries {} directions",
            basis.len()
        );

        let mut worst = 0.0f64;
        let mut weakest = f64::INFINITY;
        for (j, xi) in basis.iter().enumerate() {
            let fused = sctx.a_apply_fused(xi);
            let exact = sctx.a_apply_exact(xi);
            let free = effham_apply_shifted(&h_out, &bl, &br, phi, lambda, xi);
            assert_eq!(exact.num_coeffs(), free.num_coeffs());
            for i in 0..free.num_coeffs() {
                assert_eq!(
                    exact.coeff(i),
                    free.coeff(i),
                    "d = 3, direction {j}: a_apply_exact must BE effham_apply_shifted, coeff {i}"
                );
            }
            weakest = weakest.min(exact.norm());
            worst = worst.max(fused.sub(&exact).norm() / exact.norm().max(1.0));

            assert_eq!(fused.phi(), phi);
            assert_eq!(fused.rank(), xi.rank(), "d = 3, direction {j}: row grids");
            assert_eq!(
                fused.rank_col(),
                xi.rank_col(),
                "d = 3, direction {j}: column grids"
            );
            assert_eq!(fused.num_coeffs(), exact.num_coeffs());
            for k in 0..3 {
                let (cf, ce) = (fused.component(k), exact.component(k));
                assert_eq!(cf.space, space_out, "d = 3: component {k} stamp");
                assert_eq!(cf.row_ranks, ce.row_ranks, "d = 3: component {k} row grid");
                assert_eq!(cf.col_ranks, ce.col_ranks, "d = 3: component {k} col grid");
            }
        }
        println!("tsr fused d3: weakest ||A xi|| = {weakest:e}, worst rel = {worst:e}");
        assert!(
            weakest > 1e-3,
            "ANTI-VACUITY: A^phi annihilated a d = 3 direction (||A xi|| = {weakest:e})"
        );
        assert!(
            worst < 1e-11,
            "d = 3: fused vs unfused shifted A-apply, worst relative deviation {worst:e}"
        );
    }

    {
        let pg = PointGroup::new(1, &[Irrep::new(1)]);
        let mut t = CoeffTensor::<f64>::zeros(1, 2);
        t.set(&[0, 0], -0.7);
        let v = CoeffTensor::<f64>::zeros(1, 4);
        let space = QnSpace::with_point_group(1, 1, 0, pg, Irrep::new(1));
        let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7D01);
        let mut x: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
        normalize_mut(&mut x);
        x.check();
        let ctx = weingarten_operator(&ham, &x);
        let phi = Irrep::new(1);
        let space_out = space.with_target(space.target.fuse(phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);

        assert_eq!(
            max_ranks(&space_out)[1].total(),
            0,
            "d = 1 premise: Gamma0 + phi must be UNREACHABLE at bond 1 in space_out (that is what \
             makes Sketch::identity(space_out, 1, 1) all-zero-column, i.e. empty-channel kind A)"
        );
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        assert_eq!(
            sctx.dim(),
            0,
            "d = 1: an abelian shifted channel is necessarily empty"
        );
        assert!(sctx.basis().is_empty(), "d = 1: basis must be empty");
        assert!(
            sctx.a_gram(&[]).is_empty(),
            "d = 1: the Gram over an empty basis is empty"
        );

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7D02);
        let y: TtVector<f64> = randomized_state(space_out, 1, 1, &mut rng);
        let xi = sctx.project(&y);
        let fused = sctx.a_apply_fused(&xi);
        let exact = sctx.a_apply_exact(&xi);
        println!(
            "tsr fused d1: phi = {phi:?}, xi coeffs = {}, rank = {:?}, rank_col = {:?}",
            xi.num_coeffs(),
            xi.rank().iter().map(|g| g.total()).collect::<Vec<_>>(),
            xi.rank_col().iter().map(|g| g.total()).collect::<Vec<_>>()
        );
        assert_eq!(fused.phi(), phi, "d = 1: the image's flux");
        assert_eq!(fused.rank(), xi.rank(), "d = 1: row grids");
        assert_eq!(fused.rank_col(), xi.rank_col(), "d = 1: column grids");
        assert_eq!(
            fused.num_coeffs(),
            exact.num_coeffs(),
            "d = 1: coefficient count"
        );
        assert_eq!(
            fused.component(0).row_ranks,
            exact.component(0).row_ranks,
            "d = 1: the only component's row grid"
        );
        assert_eq!(
            fused.component(0).col_ranks,
            exact.component(0).col_ranks,
            "d = 1: the only component's col grid — the declared shape, not the sketch chain's \
             0-column one"
        );
        assert_eq!(
            fused.component(0).space,
            space_out,
            "d = 1: component stamp"
        );
        for i in 0..fused.num_coeffs() {
            assert_eq!(fused.coeff(i), 0.0, "d = 1: coeff {i} of an empty channel");
        }
    }
}

#[test]
fn tsr_fused_a_apply_on_an_empty_channel_returns_the_declared_zero() {
    {
        let pg = synth_pg();
        let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
        let (t, v) = synth_tv(pg);
        let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
        let caps = max_ranks(&space);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7D0A);
        let mut x: TtVector<f64> = qntt::vector::tt_randn(space, &caps, &mut rng);
        normalize_mut(&mut x);
        x.check();
        let (bl, br) = tangent_bases(&x);
        let lambda = rayleigh_quotient(&ham, &x).re_s();
        let ctx = weingarten_operator(&ham, &x);
        let dims: Vec<usize> = (0..pg.order())
            .map(|p| tangent_dim_grids(&space, &bl.ranks, Irrep::new(p)))
            .collect();
        println!("tsr fused empty [kind A] synth_pg (2,1) max_ranks: dims {dims:?}");

        let mut reached = 0usize;
        for p in 0..pg.order() {
            let phi = Irrep::new(p);
            if phi == Irrep::IDENTITY {
                continue;
            }
            let space_out = space.with_target(space.target.fuse(phi));
            let dets = dets_in_irrep(D4, F_NUP, F_NDN, pg, space_out.target).len();
            if dets > 0 {
                continue;
            }
            assert_eq!(
                max_ranks(&space_out)[D4].total(),
                0,
                "phi {p}: kind-A premise — an empty target SECTOR must leave the bond-d identity \
                 with no columns"
            );
            assert_eq!(
                dims[p as usize], 0,
                "phi {p}: an empty sector cannot carry T^phi"
            );
            let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
            assert_eq!(h_out.space, space_out);
            let sctx = SectorCtx::new(&ctx, &h_out, phi);
            assert_eq!(sctx.dim(), 0, "phi {p}: dim must be 0");
            assert!(sctx.basis().is_empty(), "phi {p}: basis must be empty");
            reached += 1;

            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7D0B ^ p as u64);
            let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
            let xi = sctx.project(&y);
            let free = TtTangent::project_shifted(&bl, &br, phi, &y);
            assert_eq!(xi.rank(), free.rank(), "phi {p}: project row grids");
            assert_eq!(xi.rank_col(), free.rank_col(), "phi {p}: project col grids");

            let fused = sctx.a_apply_fused(&xi);
            let exact = sctx.a_apply_exact(&xi);
            let oracle = effham_apply_shifted(&h_out, &bl, &br, phi, lambda, &xi);
            assert_eq!(exact.num_coeffs(), oracle.num_coeffs());
            println!(
                "tsr fused empty [kind A] phi={p}: dets {dets}, declared component grids \
                 (rows x cols) = {:?}, coeffs = {}",
                (0..D4)
                    .map(|k| (
                        fused.component(k).row_ranks.total(),
                        fused.component(k).col_ranks.total()
                    ))
                    .collect::<Vec<_>>(),
                fused.num_coeffs()
            );
            assert!(
                (0..D4).any(|k| fused.component(k).col_ranks.total() > 0),
                "phi {p}: ANTI-VACUITY — the declared component shapes are all empty, so the \
                 0-column failure mode this half exists for is not observable at this fixture"
            );
            assert_eq!(fused.phi(), phi, "phi {p}: flux");
            assert_eq!(fused.rank(), xi.rank(), "phi {p}: row grids");
            assert_eq!(fused.rank_col(), xi.rank_col(), "phi {p}: col grids");
            assert_eq!(
                fused.num_coeffs(),
                exact.num_coeffs(),
                "phi {p}: coefficient count"
            );
            for k in 0..D4 {
                let (cf, ce) = (fused.component(k), exact.component(k));
                assert_eq!(cf.space, space_out, "phi {p}: component {k} stamp");
                assert_eq!(
                    cf.row_ranks, ce.row_ranks,
                    "phi {p}: component {k} row grid"
                );
                assert_eq!(
                    cf.col_ranks, ce.col_ranks,
                    "phi {p}: component {k} col grid — the DECLARED shape \
                     SparseCore::zeros(space_out, k, r[k], r_col[k+1]), not the sketch chain's \
                     0-column one"
                );
                for f in Family::ALL {
                    for s in space_out.bond_sectors(k) {
                        assert_eq!(
                            cf.block_dims(f, s),
                            ce.block_dims(f, s),
                            "phi {p}: component {k} block dims at ({f:?}, {s:?})"
                        );
                    }
                }
            }
            for i in 0..fused.num_coeffs() {
                assert_eq!(
                    fused.coeff(i),
                    0.0,
                    "phi {p}: coeff {i} of an empty channel must be exactly 0.0"
                );
            }
            assert!(
                sctx.a_gram(&[]).is_empty(),
                "phi {p}: the Gram over an empty basis is empty"
            );
        }
        assert!(
            reached >= 1,
            "fixture premise: synth_pg (2,1) must carry a structurally EMPTY SECTOR (measured phi \
             = 2, 3), or the kind-A half is vacuous"
        );
    }

    {
        let pg = fx4_pg();
        let (space, t, v, ham, x) = truncated_reference();
        let (bl, _br) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        let empty: Vec<Irrep> = (0..pg.order())
            .map(Irrep::new)
            .filter(|&phi| tangent_dim_grids(&space, &bl.ranks, phi) == 0)
            .collect();
        assert!(
            !empty.is_empty(),
            "fixture premise: fx4_pg (2,2) must carry an empty CHANNEL over a non-empty sector \
             (measured phi = 1, 2)"
        );
        for phi in empty {
            let p = phi.index();
            let space_out = space.with_target(space.target.fuse(phi));
            let dets = dets_in_irrep(D4, T_NUP, T_NDN, pg, space_out.target).len();
            assert!(
                dets > 0,
                "phi {p}: kind-B premise — the target SECTOR must be NON-empty, or this is kind A"
            );
            assert!(
                max_ranks(&space_out)[D4].total() > 0,
                "phi {p}: kind-B premise — the bond-d identity DOES have a column here, so no \
                 branch may be taken"
            );
            let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
            let sctx = SectorCtx::new(&ctx, &h_out, phi);
            assert_eq!(sctx.dim(), 0, "phi {p}: dim must be 0");
            assert!(sctx.basis().is_empty(), "phi {p}: basis must be empty");
            let (a, b) = sctx.grams(&[]);
            assert!(a.is_empty() && b.is_empty(), "phi {p}: empty Grams");
            assert!(
                sctx.a_gram(&[]).is_empty() && sctx.a_gram_exact(&[]).is_empty(),
                "phi {p}: empty Gram from both the fused and the exact driver"
            );
            println!("tsr fused empty [kind B] phi={p}: sector {dets} dets, dim 0, Grams empty");
        }
    }
}

fn synth_exact_ground() -> Fixture {
    let pg = synth_pg();
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::new(1));
    let (t, v) = synth_tv(pg);
    let (x, _e0, _lv) = exact_typed_ground(space, &t, &v);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    (space, t, v, ham, x)
}

fn rpa_ii() -> SectorOpts {
    SectorOpts {
        readout: SectorReadout::Rpa,
        kernel: RpaKernel::signed(RpaMetric::Ii),
        ..SectorOpts::default()
    }
}

#[test]
fn tsr_curvature_reg_on_a_shifted_channel_is_the_untyped_regularized_block() {
    let (space, t, v, ham, x) = d6_minimizer();
    let (bl, br) = tangent_bases(&x);
    let phi = first_curved_channel(&space, &bl.ranks);
    let (f, raw_blk, reg_blk) = partial_hard_floor(&t, &v, &x, phi)
        .expect("fixture premise: a floor of the P2 rule partially regularizes the curved channel");
    println!(
        "tsr K1 untyped block: phi {}, floor {f:.6e}, untyped block max|B| raw {raw_blk:.3e} -> \
         Hard {reg_blk:.3e}, sigma_min {:e}",
        phi.index(),
        min_schmidt(&bl, &br)
    );
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let space_c1 = QnSpace::new(space.d, space.nup, space.ndn);
    let x_c1 = x.forget_point_group();
    let h_c1 = SparseHamiltonian::<f64>::new(space_c1, &t, &v, None, f64::EPSILON);
    let ctx_off = weingarten_operator(&ham, &x);
    let sctx_off = SectorCtx::new(&ctx_off, &h_out, phi);
    let dirs = shifted_unit_tangents(&sctx_off, space_out, 0x9A01, 4);
    for (label, reg) in [
        ("Hard", CurvatureReg::Hard(f)),
        ("Tikhonov", CurvatureReg::Tikhonov(f)),
    ] {
        let ctx = weingarten_operator(&ham, &x).with_curvature(reg);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        let ctx_c1 = weingarten_operator(&h_c1, &x_c1).with_curvature(reg);
        let (bl_c1, br_c1) = ctx_c1.bases();
        let (mut worst, mut moved, mut weakest) = (0.0f64, 0.0f64, f64::INFINITY);
        for (j, xi) in dirs.iter().take(3).enumerate() {
            let b_reg = sctx.b_apply(xi);
            let b_raw = sctx_off.b_apply(xi);
            let lhs = embed_untyped(&b_reg, bl_c1, br_c1, &format!("[{label}] B xi {j}"));
            let rhs = ctx_c1.b_apply(&embed_untyped(
                xi,
                bl_c1,
                br_c1,
                &format!("[{label}] xi {j}"),
            ));
            let s = b_reg.norm().max(1.0);
            let err = lhs.sub(&rhs).norm();
            println!(
                "tsr K1 [{label}] dir {j}: ||B_reg xi|| {:.6e} ||B_raw xi|| {:.6e} \
                 |embed(B_reg xi) - B''_reg embed(xi)| {err:.2e}",
                b_reg.norm(),
                b_raw.norm()
            );
            assert!(
                err <= 1e-10 * s,
                "[{label}] dir {j}: the shifted regularized B is not the untyped regularized \
                 block ({err:e} at scale {s:e})"
            );
            worst = worst.max(err / s);
            moved = moved.max(b_reg.sub(&b_raw).norm() / b_raw.norm().max(f64::MIN_POSITIVE));
            weakest = weakest.min(b_reg.norm());
        }
        let (xi, eta) = (&dirs[0], &dirs[3]);
        let typed = 0.5 * (eta.dot(&sctx.b_apply(xi)) + xi.dot(&sctx.b_apply(eta)));
        let exi = embed_untyped(xi, bl_c1, br_c1, &format!("[{label}] polarized xi"));
        let eeta = embed_untyped(eta, bl_c1, br_c1, &format!("[{label}] polarized eta"));
        let untyped = 0.5 * (eeta.dot(&ctx_c1.b_apply(&exi)) + exi.dot(&ctx_c1.b_apply(&eeta)));
        println!(
            "tsr K1 [{label}]: worst rel {worst:.2e}, moved {moved:.3e}, polarized typed \
             {typed:.12} untyped {untyped:.12}"
        );
        assert!(
            (typed - untyped).abs() <= 1e-10 * typed.abs().max(1.0),
            "[{label}] polarized entry: typed {typed} vs untyped {untyped}"
        );
        assert!(
            moved >= 1e-6,
            "[{label}] non-vacuity: the regularization did not move B^phi xi ({moved:e})"
        );
        assert!(
            weakest >= 1e-6,
            "[{label}] non-vacuity: B_reg xi vanished ({weakest:e}) — the identity would be 0 = 0"
        );
    }
}

#[test]
fn tsr_williamson_union_holds_under_curvature_reg() {
    let (space, t, v, ham, x) = d6_minimizer();
    let (bl, br) = tangent_bases(&x);
    let live = live_channels(&space, &bl.ranks);
    assert_eq!(
        live.len(),
        4,
        "fixture premise: all four channels live, got {live:?}"
    );
    let curved = first_curved_channel(&space, &bl.ranks);
    let (f, _, _) = partial_hard_floor(&t, &v, &x, curved)
        .expect("fixture premise: a floor of the P2 rule partially regularizes the curved channel");
    let space_c1 = QnSpace::new(space.d, space.nup, space.ndn);
    let x_c1 = x.forget_point_group();
    let h_c1 = SparseHamiltonian::<f64>::new(space_c1, &t, &v, None, f64::EPSILON);
    let ctx_off = weingarten_operator(&ham, &x);
    let curved_out = space.with_target(space.target.fuse(curved));
    let h_curved = SparseHamiltonian::<f64>::new(curved_out, &t, &v, None, f64::EPSILON);
    let basis_curved = tangent_basis_shifted(&bl, &br, curved);
    let plain_curved = signed_omega2_or_fail(
        &SectorCtx::new(&ctx_off, &h_curved, curved).symplectic_eigenvalues(&basis_curved),
        "plain curved channel",
    );
    for (label, reg) in [
        ("Hard", CurvatureReg::Hard(f)),
        ("Tikhonov", CurvatureReg::Tikhonov(f)),
    ] {
        let ctx = weingarten_operator(&ham, &x).with_curvature(reg);
        let ctx_c1 = weingarten_operator(&h_c1, &x_c1).with_curvature(reg);
        let (bl_c1, br_c1) = ctx_c1.bases();
        let dim_untyped = tangent_dim(bl_c1);
        let dim_0 = tangent_dim_grids(&space, &bl.ranks, Irrep::IDENTITY);
        let dim_shifted: usize = live
            .iter()
            .filter(|&&p| p != Irrep::IDENTITY)
            .map(|&p| tangent_dim_grids(&space, &bl.ranks, p))
            .sum();
        assert_eq!(
            dim_0 + dim_shifted - 1,
            dim_untyped - 1,
            "[{label}] dimension identity"
        );
        let basis_c1 = tangent_basis(bl_c1, br_c1, true);
        let basis_0 = tangent_basis(&bl, &br, true);
        let shifted: Vec<(Irrep, Vec<TtTangent<f64>>)> = live
            .iter()
            .copied()
            .filter(|&p| p != Irrep::IDENTITY)
            .map(|p| (p, tangent_basis_shifted(&bl, &br, p)))
            .collect();
        let n_shifted: usize = shifted.iter().map(|(_, b)| b.len()).sum();
        assert_eq!(
            basis_c1.len(),
            basis_0.len() + n_shifted,
            "[{label}] the untyped basis and the blocks disagree on the count"
        );
        let spec_c1 = symplectic_eigenvalues(&ctx_c1, &basis_c1);
        let spec_0 = symplectic_eigenvalues(&ctx, &basis_0);
        println!(
            "tsr K1 union [{label}] f {f:.6e}: untyped {}, phi=0 {}",
            spectrum_label(&spec_c1),
            spectrum_label(&spec_0)
        );
        let w_untyped = signed_omega2_or_fail(&spec_c1, &format!("[{label}] untyped side"));
        let mut union = signed_omega2_or_fail(&spec_0, &format!("[{label}] phi = 0 block"));
        let mut moved = 0.0f64;
        for (p, basis_phi) in &shifted {
            let space_out = space.with_target(space.target.fuse(*p));
            let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
            let sctx = SectorCtx::new(&ctx, &h_out, *p);
            let spec = sctx.symplectic_eigenvalues(basis_phi);
            println!(
                "tsr K1 union [{label}] phi={}: {} (n = {})",
                p.index(),
                spectrum_label(&spec),
                basis_phi.len()
            );
            let w = signed_omega2_or_fail(&spec, &format!("[{label}] phi = {} block", p.index()));
            if *p == curved {
                assert_eq!(w.len(), plain_curved.len());
                moved = w
                    .iter()
                    .zip(&plain_curved)
                    .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            }
            union.extend(w);
        }
        union.sort_by(|a, b| a.partial_cmp(b).expect("finite omega^2"));
        let scale = w_untyped.iter().fold(0.0f64, |m, z| m.max(z.abs()));
        println!(
            "tsr K1 union [{label}]: |union| = {}, |untyped| = {}, scale {scale:e}, curved \
             channel moved by {moved:.3e}",
            union.len(),
            w_untyped.len()
        );
        assert!(
            moved >= 1e-6 * scale,
            "[{label}] non-vacuity: the regularization did not move the curved channel's spectrum \
             ({moved:e})"
        );
        assert_multiset_eq(
            &union,
            &w_untyped,
            1e-8,
            &format!(
                "[{label}] union_phi of the regularized blocks vs the untyped regularized spectrum"
            ),
        );
    }
}

#[test]
fn tsr_signed_sector_solve_satisfies_the_index_identity() {
    let (checked, converged, nontrivial) = index_identity_over(vec![
        ("truncated (2,2)", truncated_reference()),
        ("synthetic exact ground", synth_exact_ground()),
        ("fx4 (2,1) minimizer", fx4_minimizer()),
        ("d6 minimizer", d6_minimizer()),
    ]);
    println!(
        "tsr index: {checked} channels checked, {converged} converged exhaustive, {nontrivial} \
         with nonzero inertia"
    );
    assert!(converged >= 1, "non-vacuity: no exhaustive solve converged");
    assert!(
        nontrivial >= 1,
        "non-vacuity: no converged channel carried inertia (every identity checked was 0 = 0)"
    );
}

fn index_identity_over(legs: Vec<(&str, Fixture)>) -> (usize, usize, usize) {
    let (mut converged, mut nontrivial, mut checked) = (0usize, 0usize, 0usize);
    for (label, (space, t, v, ham, x)) in legs {
        let (bl, _br) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        for phi in live_channels(&space, &bl.ranks) {
            if phi == Irrep::IDENTITY {
                continue;
            }
            let p = phi.index();
            let space_out = space.with_target(space.target.fuse(phi));
            let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
            let sctx = SectorCtx::new(&ctx, &h_out, phi);
            let basis = sctx.basis();
            let n = basis.len();
            let (a, b) = sctx.grams(&basis);
            let inertia = pencil_inertia(&a, &b, n);
            if inertia.min_abs_eig < 1e-8 {
                println!(
                    "tsr index [{label}] phi={p}: SKIPPED, min |eig(A +- B)| = {:e} < 1e-8",
                    inertia.min_abs_eig
                );
                continue;
            }
            assert_eq!(
                inertia.lhs(),
                inertia.rhs(),
                "[{label}] phi {p}: the dense pencil's counting convention ({inertia:?})"
            );
            let rhs = inertia.rhs();
            let full = sctx.sector_lowest(sctx.dim(), &rpa_ii());
            let one = sctx.sector_lowest(1, &rpa_ii());
            let ilb_full = full
                .index_lower_bound()
                .expect("an Rpa readout reports its inertia");
            let ilb_one = one
                .index_lower_bound()
                .expect("an Rpa readout reports its inertia");
            println!(
                "tsr index [{label}] phi={p}: n {n}, dense {inertia:?} (RHS {rhs}); exhaustive \
                 converged {} index {ilb_full} (flagged {}, report {:?}); k = 1 index {ilb_one}",
                full.converged,
                full.metric_indefinite(),
                metric_report(full.rpa())
            );
            assert!(
                ilb_one <= rhs,
                "[{label}] phi {p}: k = 1 index {ilb_one} exceeds the dense inertia {rhs} — not a \
                 lower bound"
            );
            assert!(
                ilb_full <= rhs,
                "[{label}] phi {p}: exhaustive index {ilb_full} exceeds the dense inertia {rhs}"
            );
            checked += 1;
            if full.converged {
                assert_eq!(
                    ilb_full, rhs,
                    "[{label}] phi {p}: a converged exhaustive solve must reproduce the dense \
                     inertia"
                );
                converged += 1;
                if rhs > 0 {
                    nontrivial += 1;
                }
            }
        }
    }
    (checked, converged, nontrivial)
}

#[test]
fn tsr_sector_mode_checks_of_an_empty_readout_are_empty() {
    let (space, t, v, ham, x) = truncated_reference();
    let (bl, _br) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let empty = (1..space.pg.order())
        .map(Irrep::new)
        .find(|&p| tangent_dim_grids(&space, &bl.ranks, p) == 0)
        .expect("fixture premise: an empty channel");
    let live = first_shifting_channel(&space, &bl.ranks);
    let opts = SectorOpts {
        readout: SectorReadout::Rpa,
        kernel: RpaKernel::signed(RpaMetric::Auto),
        ..SectorOpts::default()
    };
    for (phi, k) in [(empty, 3usize), (live, 0usize)] {
        let space_out = space.with_target(space.target.fuse(phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        let (resp, vecs) = sctx.sector_lowest_vectors(k, &opts);
        assert!(
            matches!(&resp.spectrum, SectorSpectrum::Rpa(r) if r.side() == RpaMetric::Auto),
            "route (a) reports the metric UNRESOLVED"
        );
        assert!(vecs.is_empty() && resp.omega2() == Some(Vec::new()));
        let checks = sctx.rpa_mode_checks(&resp, &vecs, &ModeCheckOpts::default());
        assert!(
            checks.is_empty(),
            "phi {:?} k {k}: an empty readout has no modes",
            phi
        );
    }
}

#[test]
#[should_panic(expected = "rpa_mode_checks")]
fn tsr_sector_mode_checks_refuse_a_tda_response() {
    let (space, t, v, ham, x) = truncated_reference();
    let (bl, _br) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_shifting_channel(&space, &bl.ranks);
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    let tda = sctx.sector_lowest_tda(2, &SectorOpts::default());
    let n = tda.theta().expect("a Tda readout").len();
    let vecs: Vec<qntt::solvers::hess::RpaAmplitudes<TtTangent<f64>>> = sctx
        .basis()
        .into_iter()
        .take(n)
        .map(|v| qntt::solvers::hess::RpaAmplitudes {
            x: v.clone(),
            y: v,
            x_im: None,
            y_im: None,
        })
        .collect();
    assert_eq!(vecs.len(), n, "the mode list is sized to the readout");
    let _ = sctx.rpa_mode_checks(&tda, &vecs, &ModeCheckOpts::default());
}

#[test]
fn tsr_resonant_channel_publishes_a_certified_negative_omega2() {
    let (space, t, v, ham, x) = pg4c_random_reference(0x7505);
    let (bl, _br) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_curved_channel(&space, &bl.ranks);
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    let basis = sctx.basis();
    let n = basis.len();
    assert_eq!(n, sctx.dim(), "fixture premise: the basis spans T^phi");
    let (a, b) = sctx.grams(&basis);
    let inertia = pencil_inertia(&a, &b, n);
    let dense = sctx.symplectic_eigenvalues(&basis);
    println!(
        "tsr resonant: phi {}, n {n}, dense {}, inertia {inertia:?}, ||A|| {:.3e} ||B|| {:.3e}",
        phi.index(),
        spectrum_label(&dense),
        gram_spectral_norm(&a, n),
        gram_spectral_norm(&b, n)
    );
    let SymplecticSpectrum::Resonant { signed } = &dense else {
        panic!(
            "fixture premise: the pg4c 0x7505 curved channel is Resonant, got {}",
            spectrum_label(&dense)
        )
    };
    assert_eq!(
        signed.metric,
        RpaMetricBlock::Ii,
        "fixture premise: H^ii is the PD block"
    );
    assert!(signed.n_neg >= 1, "fixture premise: a negative omega^2");
    assert!(
        inertia.min_abs_eig >= 1e-3,
        "fixture premise: margin min |eig(A +- B)| = {:e}",
        inertia.min_abs_eig
    );
    let scale = signed.omega2.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    assert!(
        signed.omega2[0] <= -1e-5 * scale,
        "fixture premise: the negative omega^2 {} is far from 0 (scale {scale:e})",
        signed.omega2[0]
    );
    assert_eq!(inertia.lhs(), inertia.rhs());

    let out = sctx.sector_lowest(sctx.dim(), &rpa_ii());
    let w = out.omega2().expect("Rpa").to_vec();
    println!(
        "tsr resonant iterative: omega2 {w:?} signature {:?} converged {} flagged {} negatives {:?} \
         index {:?}",
        krein_signs(out.rpa()),
        out.converged,
        out.metric_indefinite(),
        out.negatives(),
        out.index_lower_bound()
    );
    assert!(
        !out.metric_indefinite(),
        "W = H^ii is PD: the Ii lane meets no indefinite metric"
    );
    assert_eq!(
        krein_signs(out.rpa()),
        w.iter()
            .map(|&z| if z < 0.0 { 0 } else { 1 })
            .collect::<Vec<i8>>(),
        "the J = I path: every real mode s = +1"
    );
    assert!(
        out.converged,
        "an exhaustive solve on the J = I path converges"
    );
    let tol = 10.0 * SectorOpts::default().tol * scale.max(1.0);
    assert_multiset_eq(&w, &signed.omega2, tol, "iterative vs dense signed omega^2");
    let n_neg = w.iter().filter(|&&z| z < 0.0).count();
    assert_eq!(n_neg, signed.n_neg, "every negative omega^2 is listed");
    assert_eq!(out.negatives().len(), n_neg, "one witness per negative");
    for wit in &out.negatives() {
        assert!(
            (wit.theta - w[wit.index]).abs() <= 1e-8 * w[wit.index].abs().max(1.0),
            "the witness theta is the published omega^2 (omega^2 units on the Rpa arm)"
        );
    }
    assert!(
        out.all_negatives_certified(),
        "every negative carries a certified fresh-apply witness: {:?}",
        out.negatives()
    );
    assert!(out.omega().is_none(), "a resonance is never a real omega");
    assert_eq!(out.index_lower_bound(), Some(inertia.rhs()));

    let k = 2usize;
    let two = sctx.sector_lowest(k, &rpa_ii());
    let w2 = two.omega2().expect("Rpa");
    println!(
        "tsr resonant k = {k}: omega2 {w2:?} converged {}",
        two.converged
    );
    assert!(two.converged, "the k = {k} solve converges");
    assert_eq!(
        w2.len(),
        n_neg + k,
        "a Resonant channel lists n_neg + k entries"
    );
    assert_eq!(w2.iter().filter(|&&z| z < 0.0).count(), n_neg);
    let auto = sctx.sector_lowest(
        k,
        &SectorOpts {
            kernel: RpaKernel::signed(RpaMetric::Auto),
            ..rpa_ii()
        },
    );
    assert!(
        flat_bits(&auto.omega2().expect("Rpa")) == flat_bits(&w2)
            && (auto.iterations, auto.applies) == (two.iterations, two.applies),
        "Auto must be the Ii call bit for bit"
    );
}

#[test]
fn tsr_shifted_diagonal_is_the_embedded_rayleigh_quotient() {
    let mut checked = 0usize;
    for (label, (space, t, v, ham, x)) in [
        ("d = 3 minimal channel", d3_minimal_channel()),
        ("truncated (2,2)", truncated_reference()),
    ] {
        let (bl, _br) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        for phi in live_channels(&space, &bl.ranks) {
            if phi == Irrep::IDENTITY {
                continue;
            }
            let p = phi.index();
            let space_out = space.with_target(space.target.fuse(phi));
            let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
            let sctx = SectorCtx::new(&ctx, &h_out, phi);
            let lambda = sctx.lambda();
            let diag = sctx.a_diagonal();
            let r_col: Vec<qntt::RankGrid> =
                bl.ranks.iter().map(|g| g.relabel_irrep(phi)).collect();
            assert_eq!(diag.phi(), phi);
            assert_eq!(diag.rank(), &bl.ranks[..], "[{label}] phi {p}: row grids");
            assert_eq!(
                diag.rank_col(),
                &r_col[..],
                "[{label}] phi {p}: column grids"
            );
            for k in 0..space.d {
                let c = diag.component(k);
                assert_eq!(c.space, space_out, "[{label}] phi {p}: component {k} stamp");
                assert_eq!(
                    c.row_ranks, bl.ranks[k],
                    "[{label}] phi {p}: component {k} rows"
                );
                assert_eq!(
                    c.col_ranks,
                    r_col[k + 1],
                    "[{label}] phi {p}: component {k} cols"
                );
            }
            println!(
                "tsr K4 rayleigh [{label}] phi={p}: dim {}, per-component coefficients {:?}",
                sctx.dim(),
                (0..space.d)
                    .map(|k| diag.component(k).num_coeffs())
                    .collect::<Vec<_>>()
            );
            let mut worst = 0.0f64;
            for i in 0..diag.num_coeffs() {
                let mut e = diag.clone();
                e.for_each_coeff_mut(|c| *c = 0.0);
                e.set_coeff(i, 1.0);
                let y = e.to_vector();
                let ny = y.norm();
                assert!(
                    (ny - 1.0).abs() <= 1e-12,
                    "[{label}] phi {p} coeff {i}: premise ||iota(e_i)|| = 1, got {ny}"
                );
                let rq = xt_h_y(&y, &h_out, &y).re_s() - lambda;
                let got = diag.coeff(i);
                worst = worst.max((rq - got).abs() / (1.0 + rq.abs()));
                assert!(
                    (rq - got).abs() <= 1e-10 * (1.0 + rq.abs()),
                    "[{label}] phi {p} coeff {i}: a_diagonal {got} vs <y, H_out y> - lambda {rq}"
                );
                checked += 1;
            }
            println!("tsr K4 rayleigh [{label}] phi={p}: worst rel {worst:.2e}");
        }
    }
    assert!(checked >= 8, "non-vacuity: {checked} coefficients checked");
}

#[test]
fn tsr_sector_lowest_d0_reaches_the_same_spectrum() {
    let (space, t, v, ham, x) = d6_minimizer();
    let (bl, _br) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_curved_channel(&space, &bl.ranks);
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    let k = 3usize;
    let d0 = SympPrecond::DiagUnshifted { kappa: 0.0 };
    for (readout, kernel) in [
        (SectorReadout::Rpa, RpaKernel::Paired),
        (SectorReadout::Rpa, RpaKernel::signed(RpaMetric::Ii)),
        (SectorReadout::Tda, RpaKernel::Paired),
    ] {
        let run = |precond: Option<SympPrecond>| {
            let mut opts = SectorOpts {
                readout,
                kernel,
                ..SectorOpts::default()
            };
            if let Some(p) = precond {
                opts.precond = p;
            }
            match readout {
                SectorReadout::Rpa => sctx.sector_lowest(k, &opts),
                SectorReadout::Tda => sctx.sector_lowest_tda(k, &opts),
            }
        };
        let vals = |r: &qntt::solvers::hess::SectorResponse| -> Vec<f64> {
            r.omega2()
                .or_else(|| r.theta().map(<[f64]>::to_vec))
                .expect("one arm")
                .to_vec()
        };
        let off = run(Some(SympPrecond::Off));
        let with = run(Some(d0));
        let dflt = run(None);
        let (wo, wd) = (vals(&off), vals(&with));
        println!(
            "tsr D0 [{readout:?} {kernel:?}]: Off {wo:?} ({} iters, {} applies, converged {}) | D0 \
             {wd:?} ({} iters, {} applies, converged {})",
            off.iterations,
            off.applies,
            off.converged,
            with.iterations,
            with.applies,
            with.converged
        );
        assert!(
            off.converged && with.converged,
            "[{readout:?}] both runs converge"
        );
        assert_eq!(wo.len(), wd.len(), "[{readout:?}] the same number of roots");
        let scale = wo.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
        let tol = 10.0 * SectorOpts::default().tol * scale;
        for (j, (a, b)) in wo.iter().zip(&wd).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "[{readout:?}] root {j}: Off {a} vs D0 {b} (tol {tol:e})"
            );
        }
        assert!(
            flat_bits(&vals(&dflt)) == flat_bits(&wo)
                && (dflt.iterations, dflt.applies) == (off.iterations, off.applies)
                && flat_bits(&residuals_of(dflt.rpa())) == flat_bits(&residuals_of(off.rpa())),
            "[{readout:?}] the default (Auto -> Off) must be the explicit Off run bit for bit"
        );
        assert!(
            flat_bits(&wd) != flat_bits(&wo)
                || (with.iterations, with.applies) != (off.iterations, off.applies),
            "[{readout:?}] non-vacuity: the D0 run is bitwise the Off run — the preconditioner \
             never reached the kernel"
        );
    }
}

#[test]
fn tsr_sector_lowest_honours_max_subspace() {
    let (space, t, v, ham, x) = d6_minimizer();
    let (bl, _br) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_curved_channel(&space, &bl.ranks);
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    let dim = sctx.dim();
    let k = 3usize;
    let tight = 2 * (k + (k + 2));
    assert!(
        dim > tight,
        "premise: dim {dim} leaves room below it for the tight bound {tight}"
    );
    let tol = 1e-10;
    let run = |kernel: RpaKernel, k: usize, max_subspace: usize| {
        sctx.sector_lowest(
            k,
            &SectorOpts {
                tol,
                readout: SectorReadout::Rpa,
                kernel,
                max_subspace,
                ..SectorOpts::default()
            },
        )
    };
    for (kernel, k, tight) in [
        (RpaKernel::signed(RpaMetric::Ii), k, tight),
        (RpaKernel::Paired, 1, 4 * (1 + (1 + 2))),
    ] {
        assert!(
            dim > tight,
            "premise ({kernel:?}): dim {dim} > tight bound {tight}"
        );
        let (dflt, wide, cut) = (
            run(kernel, k, 0),
            run(kernel, k, dim + 10),
            run(kernel, k, tight),
        );
        let w = |r: &qntt::solvers::hess::SectorResponse| -> Vec<f64> {
            r.omega2().expect("an Rpa readout").to_vec()
        };
        println!(
            "tsr max_subspace {kernel:?} k {k}: dim {dim}; default {:?} ({} iters, {} applies, converged {}) | tight \
             {tight}: {:?} ({} iters, {} applies, converged {})",
            w(&dflt),
            dflt.iterations,
            dflt.applies,
            dflt.converged,
            w(&cut),
            cut.iterations,
            cut.applies,
            cut.converged
        );
        assert!(
            flat_bits(&w(&wide)) == flat_bits(&w(&dflt))
                && (wide.iterations, wide.applies) == (dflt.iterations, dflt.applies)
                && flat_bits(&residuals_of(wide.rpa())) == flat_bits(&residuals_of(dflt.rpa())),
            "a bound above dim clamps to dim, as the auto bound does here: the same solve bit for bit"
        );
        assert!(
            flat_bits(&w(&cut)) != flat_bits(&w(&dflt))
                || (cut.iterations, cut.applies) != (dflt.iterations, dflt.applies),
            "non-vacuity: the tight-bound run is bitwise the default one — the bound never reached \
             the kernel"
        );
        assert!(dflt.converged && cut.converged, "both runs converge");
        let (wd, wc) = (w(&dflt), w(&cut));
        assert_eq!(wd.len(), wc.len(), "the same number of roots");
        let scale = wd.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
        for (j, (a, b)) in wd.iter().zip(&wc).enumerate() {
            assert!(
                (a - b).abs() <= 10.0 * tol * scale,
                "root {j}: default {a} vs tight bound {b}"
            );
        }
    }
}

#[test]
fn tsr_sector_d0_survives_a_collapsed_tda_model_at_a_top_reference() {
    let pg = d6_pg();
    let (t, v) = d6_tv(pg);
    let d0 = SympPrecond::DiagUnshifted { kappa: 0.0 };
    let mut collapsed = 0usize;
    'search: for g0 in 0..pg.order() {
        let target = Irrep::new(g0);
        let ndets = dets_in_irrep(D6, 2, 2, pg, target).len();
        if ndets == 0 {
            continue;
        }
        let space = QnSpace::with_point_group(D6, 2, 2, pg, target);
        let (x, lam, _) = exact_typed_eigvec(space, &t, &v, ndets - 1);
        let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
        let ctx = weingarten_operator(&ham, &x);
        let (bl, _) = ctx.bases();
        for p in 1..pg.order() {
            let phi = Irrep::new(p);
            if tangent_dim_grids(&space, &bl.ranks, phi) == 0 {
                continue;
            }
            let space_out = space.with_target(space.target.fuse(phi));
            let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
            let sctx = SectorCtx::new(&ctx, &h_out, phi);
            let dg = sctx.a_diagonal();
            let dmax = (0..dg.num_coeffs())
                .map(|i| dg.coeff(i))
                .fold(f64::NEG_INFINITY, f64::max);
            if dmax > 0.0 {
                continue;
            }
            collapsed += 1;
            let k = 2usize.min(sctx.dim());
            let tda = |pc: SympPrecond| {
                sctx.sector_lowest_tda(
                    k,
                    &SectorOpts {
                        readout: SectorReadout::Tda,
                        precond: pc,
                        ..SectorOpts::default()
                    },
                )
            };
            let rpa = |pc: SympPrecond, kernel: RpaKernel| {
                sctx.sector_lowest(
                    k,
                    &SectorOpts {
                        readout: SectorReadout::Rpa,
                        precond: pc,
                        kernel,
                        ..SectorOpts::default()
                    },
                )
            };
            let (paired, signed) = (RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii));
            for (arm, off, with) in [
                ("Tda", tda(SympPrecond::Off), tda(d0)),
                ("Rpa paired", rpa(SympPrecond::Off, paired), rpa(d0, paired)),
                ("Rpa signed", rpa(SympPrecond::Off, signed), rpa(d0, signed)),
            ] {
                let wo = off
                    .theta()
                    .map(<[f64]>::to_vec)
                    .or_else(|| off.omega2())
                    .expect("one arm")
                    .to_vec();
                let wd = with
                    .theta()
                    .map(<[f64]>::to_vec)
                    .or_else(|| with.omega2())
                    .expect("one arm")
                    .to_vec();
                println!(
                    "tsr D0 collapsed [{arm}] G0 {g0} phi {p} lambda {lam:.6} max D_A {dmax:.4e}: \
                     Off {wo:?} ({} iters, {} applies, conv {}) | D0 {wd:?} ({} iters, {} applies, \
                     conv {})",
                    off.iterations,
                    off.applies,
                    off.converged,
                    with.iterations,
                    with.applies,
                    with.converged
                );
                assert!(
                    off.converged,
                    "[{arm}] G0 {g0} phi {p}: premise, Off converges"
                );
                assert!(
                    with.converged,
                    "[{arm}] G0 {g0} phi {p}: D0 stopped unconverged after {} iterations on a \
                     collapsed model (the overflow)",
                    with.iterations
                );
                assert_eq!(wo.len(), wd.len(), "[{arm}] G0 {g0} phi {p}: root count");
                let scale = wo.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
                let tol = 10.0 * SectorOpts::default().tol * scale;
                for (j, (a, b)) in wo.iter().zip(&wd).enumerate() {
                    assert!(
                        (a - b).abs() <= tol,
                        "[{arm}] G0 {g0} phi {p} root {j}: Off {a} vs D0 {b} (tol {tol:e})"
                    );
                }
                if arm == "Rpa paired" {
                    assert!(
                        flat_bits(&wo) == flat_bits(&wd)
                            && (off.iterations, off.applies) == (with.iterations, with.applies),
                        "[{arm}] G0 {g0} phi {p}: a collapsed D_A model must skip D0 (the guard), \
                         bitwise the Off run"
                    );
                }
            }
            break 'search;
        }
    }
    assert!(
        collapsed >= 1,
        "fixture premise: no live channel at a top eigenvector has a collapsed D_A model"
    );
}

fn paired_sector_leg(
    label: &str,
    sc: &SectorCtx<'_, '_, f64>,
    k: usize,
    stable: bool,
    cross: bool,
) -> usize {
    let basis = sc.basis();
    let n = basis.len();
    let (a, b) = sc.grams(&basis);
    let dense = dense_pencil(&a, &b, n);
    let (want_r, want_c) = dense.wanted(k);
    let opts = |kernel: RpaKernel| SectorOpts {
        readout: SectorReadout::Rpa,
        tol: 1e-10,
        kernel,
        ..SectorOpts::default()
    };
    let (resp, amps) = sc.sector_lowest_vectors(k, &opts(RpaKernel::Paired));
    let r = resp.rpa().expect("an Rpa readout");
    let got_r = r.omega2();
    let got_c: Vec<(f64, f64)> = {
        let mut c: Vec<_> = r
            .complex()
            .map(|m| (m.omega2.re, m.omega2.im.abs()))
            .collect();
        c.sort_by(|p, q| p.0.total_cmp(&q.0));
        c
    };
    let rel = |g: f64, w: f64| (g - w).abs() / w.abs().max(f64::MIN_POSITIVE);
    let worst = got_r
        .iter()
        .zip(&want_r)
        .map(|(&g, &w)| rel(g, w))
        .chain(
            got_c
                .iter()
                .zip(&want_c)
                .flat_map(|(g, w)| [rel(g.0, w.0), rel(g.1, w.1)]),
        )
        .fold(0.0f64, f64::max);
    println!(
        "tsr paired [{label}] k {k}: n {n}, dim {}, built {}, pairs {}, converged {}, omega2 {got_r:?} \
         (dense {want_r:?}), complex {got_c:?} (dense {want_c:?}), worst rel {worst:.2e}, index {:?}",
        sc.dim(),
        resp.iterations,
        resp.applies,
        resp.converged,
        resp.index_lower_bound()
    );
    assert!(
        resp.converged,
        "[{label}] k {k}: the paired solve converges"
    );
    assert!(
        matches!(r.report, RpaReport::Paired { .. }),
        "[{label}] k {k}: the paired kernel ran ({:?})",
        r.report
    );
    assert_eq!(
        got_r.len(),
        want_r.len(),
        "[{label}] k {k}: real/imaginary mode count"
    );
    assert_eq!(
        got_c.len(),
        want_c.len(),
        "[{label}] k {k}: complex pair count"
    );
    assert!(
        worst <= 1e-8,
        "[{label}] k {k}: omega^2 off the dense pencil by {worst:e} (relative)"
    );
    assert!(
        r.modes
            .iter()
            .filter(|m| m.kind == RpaModeKind::Imaginary)
            .all(|m| m.certificate.is_some_and(|w| w.certified())),
        "[{label}] k {k}: every imaginary mode carries a certified witness"
    );
    if k >= sc.dim() {
        assert_eq!(
            resp.index_lower_bound(),
            Some(pencil_inertia(&a, &b, n).rhs()),
            "[{label}] k = dim: the projected inertia IS n-(A+B) + n-(A-B)"
        );
    }
    assert_eq!(
        resp.omega().is_some(),
        stable,
        "[{label}] k {k}: omega() is published on a Stable channel only"
    );
    let checks = sc.rpa_mode_checks(&resp, &amps, &ModeCheckOpts::default());
    assert_eq!(checks.len(), r.modes.len());
    for (j, c) in checks.iter().enumerate() {
        let res = r.modes[j].residual;
        if c.eta.is_finite() {
            assert!(c.eta <= 1e-6, "[{label}] k {k} mode {j}: eta {}", c.eta);
        }
        if c.residual_fresh.is_finite() {
            assert!(
                (c.residual_fresh - res).abs() <= 1e-9 + 1e-3 * res,
                "[{label}] k {k} mode {j}: fresh residual {} vs the readout's {res}",
                c.residual_fresh
            );
        }
    }
    assert!(
        checks
            .iter()
            .filter(|c| c.omega2 > 0.0)
            .all(|c| c.eta.is_finite() && c.residual_fresh.is_finite()),
        "[{label}] k {k}: every positive mode is checked"
    );
    if cross {
        let signed = sc.sector_lowest(k, &opts(RpaKernel::signed(RpaMetric::Ii)));
        let ws = signed.omega2().expect("Rpa");
        assert!(
            signed.converged,
            "[{label}] k {k}: the signed solve converges"
        );
        assert_eq!(ws.len(), got_r.len(), "[{label}] k {k}: cross-kernel count");
        for (j, (p, s)) in got_r.iter().zip(&ws).enumerate() {
            assert!(
                rel(*s, *p) <= 1e-8,
                "[{label}] k {k} mode {j}: paired {p} vs signed {s}"
            );
        }
    }
    got_r.len() + got_c.len()
}

#[test]
fn tsr_paired_sector_readout_matches_the_dense_pencil() {
    let mut compared = 0usize;
    for (label, (space, t, v, ham, x)) in [("fx4", fx4_minimizer()), ("d6", d6_minimizer())] {
        let (bl, _) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        let phis: Vec<Irrep> = if label == "d6" {
            vec![first_curved_channel(&space, &bl.ranks)]
        } else {
            live_channels(&space, &bl.ranks)
                .into_iter()
                .filter(|&p| p != Irrep::IDENTITY)
                .collect()
        };
        assert!(
            !phis.is_empty(),
            "[{label}] premise: a live shifted channel"
        );
        for phi in phis {
            let space_out = space.with_target(space.target.fuse(phi));
            let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
            let sc = SectorCtx::new(&ctx, &h_out, phi);
            let spec = sc.symplectic_eigenvalues(&sc.basis());
            assert!(
                matches!(spec, SymplecticSpectrum::Stable { .. }),
                "[{label}] phi {phi:?}: premise Stable, got {}",
                spectrum_label(&spec)
            );
            let tag = format!("{label} phi {}", phi.index());
            for k in [2, sc.dim()] {
                compared += paired_sector_leg(&tag, &sc, k, true, true);
            }
        }
    }
    for (seed, resonant) in [(0x7505u64, true), (7, false)] {
        let (space, t, v, ham, x) = pg4c_random_reference(seed);
        let ctx = weingarten_operator(&ham, &x);
        let phi = Irrep::new(1);
        let space_out = space.with_target(space.target.fuse(phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        let sc = SectorCtx::new(&ctx, &h_out, phi);
        let spec = sc.symplectic_eigenvalues(&sc.basis());
        let label = format!("pg4c {seed:#x} {}", spectrum_label(&spec));
        assert_eq!(
            matches!(spec, SymplecticSpectrum::Resonant { .. }),
            resonant,
            "[{label}] premise"
        );
        if !resonant {
            assert!(
                matches!(spec, SymplecticSpectrum::Unstable { .. }),
                "[{label}] premise"
            );
        }
        compared += paired_sector_leg(&label, &sc, sc.dim(), false, resonant);
    }
    println!("tsr paired: {compared} modes compared");
}
