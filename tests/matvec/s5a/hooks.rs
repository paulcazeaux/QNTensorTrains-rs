// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn s5a_d1t_sketched_hamiltonian_threaded_matches_serial() {
    let mut rng = seeded(0x5a2f);
    let space = Su2Space::new(8, 4, 0);
    let t = ha_tensor::<f64>(
        8,
        &[
            (0, 1, -1.0),
            (1, 2, -0.5),
            (2, 3, -0.7),
            (3, 4, -0.3),
            (4, 5, -0.6),
            (5, 6, -0.4),
            (6, 7, -0.5),
        ],
    );
    let h = Su2SparseHamiltonian::new(space, &t, &zero_v(8), None, 1e-12);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 4, 8, &mut rng);
    x.right_orthogonalize(true);

    let k = 4usize;
    let fl = left_to_right_chain(&h, &x)[k].clone();
    let fr = right_to_left_chain(&h, &x)[k + 1].clone();
    let ws = su2_onesite_workspace(&h, x.core(k));
    let mut v = Su2Core::<f64>::zeros(space, k, &x.core(k).row_ranks, &x.core(k).col_ranks);
    v.fill_randn(&mut rng);

    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| h.sketched_hamiltonian(&v, &fl, &fr, &ws))
    };
    let y1 = run(1);
    let y8 = run(8);

    let mut maxdiff = 0.0f64;
    let mut nonzero = 0.0f64;
    for fam in Su2Family::ALL {
        for s in space.bond_sectors(k) {
            match (y1.block(fam, s), y8.block(fam, s)) {
                (Some(a), Some(b)) => {
                    assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
                    for j in 0..a.ncols() {
                        for i in 0..a.nrows() {
                            maxdiff = maxdiff.max((*a.get(i, j) - *b.get(i, j)).abs());
                            nonzero = nonzero.max(a.get(i, j).abs());
                        }
                    }
                }
                (None, None) => {}
                _ => panic!("threaded vs serial block presence mismatch at {fam:?} {s:?}"),
            }
        }
    }
    assert!(
        nonzero > 1e-6,
        "d1t: vacuous — sketched_hamiltonian output is all zero"
    );
    assert!(
        maxdiff < 1e-12,
        "d1t: threaded vs serial sketched_hamiltonian max block diff {maxdiff:e}"
    );
}

#[test]
fn s5a_d1_onesite_hermiticity() {
    let mut rng = seeded(0x5a20);
    let space = Su2Space::new(6, 2, 0);
    let t = ha_tensor::<f64>(6, &[(0, 1, -1.0), (1, 2, -0.5)]);
    let h = Su2SparseHamiltonian::new(space, &t, &zero_v(6), None, 1e-12);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    x.right_orthogonalize(true);
    let fl = Su2Sketch::<f64>::identity(space, 0, 1);
    let chain = right_to_left_chain(&h, &x);
    let fr = &chain[1];
    let ws = su2_onesite_workspace(&h, x.core(0));
    let lop = |v: &Su2Core<f64>| h.sketched_hamiltonian(v, &fl, fr, &ws);
    let mut big = Su2Core::<f64>::zeros(space, 0, &x.core(0).row_ranks, &x.core(0).col_ranks);
    big.fill_randn(&mut rng);
    let mut big2 = Su2Core::<f64>::zeros(space, 0, &x.core(0).row_ranks, &x.core(0).col_ranks);
    big2.fill_randn(&mut seeded(0x5a20 + 1));
    let lx = lop(&big);
    let ly = lop(&big2);
    let lhs = big.dot_center(&ly);
    let rhs = lx.dot_center(&big2);
    assert!(
        InnerProductVector::norm(&ly) > 1e-6,
        "d1: non-vacuity — Lop(y) should not be the zero operator"
    );
    assert!(
        (lhs - rhs).abs()
            <= 1e-10 * InnerProductVector::norm(&big) * InnerProductVector::norm(&big2),
        "d1: one-site local operator not self-adjoint in the centerweighted metric"
    );
}

#[test]
fn s5a_d2_twosite_hermiticity() {
    let mut rng = seeded(0x5a21);
    let space = Su2Space::new(6, 2, 0);
    let t = ha_tensor::<f64>(6, &[(0, 1, -1.0), (1, 2, -0.5)]);
    let h = Su2SparseHamiltonian::new(space, &t, &zero_v(6), None, 1e-12);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    x.right_orthogonalize(true);
    let fl = Su2Sketch::<f64>::identity(space, 0, 1);
    let chain = right_to_left_chain(&h, &x);
    let fr = &chain[2];
    let tc = contract(x.core(0), x.core(1));
    let lop = |v: &Su2ContractedCores<f64>| h.two_site_sketched_hamiltonian(v, &fl, fr);
    let mut big = Su2ContractedCores::<f64>::zeros(space, 0, &tc.row_ranks, &tc.col_ranks);
    for &m in &big.mid_qn.clone() {
        if let Some(mut b) = big.block_mut(m) {
            let (nr, nc) = (b.nrows(), b.ncols());
            for j in 0..nc {
                for i in 0..nr {
                    *b.as_mut().get_mut(i, j) = f64::sample_standard(&mut rng);
                }
            }
        }
    }
    let mut rng2 = seeded(0x5a21 + 1);
    let mut big2 = big.zeros_like();
    big2.for_each_coeff_mut(|v| *v = f64::sample_standard(&mut rng2));
    let lx = lop(&big);
    let ly = lop(&big2);
    let lhs = big.dot_cores(&ly);
    let rhs = lx.dot_cores(&big2);
    assert!(
        ly.norm() > 1e-6,
        "d2: non-vacuity — Lop(y) should not be the zero operator"
    );
    let cos = big.dot_cores(&big2).abs() / (big.norm() * big2.norm()).max(1e-30);
    assert!(
        cos < 0.9,
        "d2: vacuous — the two probe tensors are nearly parallel (|cos| = {cos})"
    );
    assert!(
        (lhs - rhs).abs() <= 1e-10 * big.norm() * big2.norm(),
        "d2: two-site local operator not self-adjoint in the ContractedSparseCores metric"
    );
}

#[test]
fn s5a_d2t_two_site_sketched_hamiltonian_threaded_matches_serial() {
    let mut rng = seeded(0x5a3f);
    let space = Su2Space::new(8, 4, 0);
    let t = ha_tensor::<f64>(
        8,
        &[
            (0, 1, -1.0),
            (1, 2, -0.5),
            (2, 3, -0.7),
            (3, 4, -0.3),
            (4, 5, -0.6),
            (5, 6, -0.4),
            (6, 7, -0.5),
        ],
    );
    let h = Su2SparseHamiltonian::new(space, &t, &zero_v(8), None, 1e-12);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 4, 8, &mut rng);
    x.right_orthogonalize(true);

    let k = 3usize;
    let fl = left_to_right_chain(&h, &x)[k].clone();
    let fr = right_to_left_chain(&h, &x)[k + 2].clone();
    let tc = contract(x.core(k), x.core(k + 1));
    let mut v = Su2ContractedCores::<f64>::zeros(space, k, &tc.row_ranks, &tc.col_ranks);
    v.for_each_coeff_mut(|c| *c = f64::sample_standard(&mut rng));

    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| h.two_site_sketched_hamiltonian(&v, &fl, &fr))
    };
    let y1 = run(1);
    let y8 = run(8);

    let mut maxdiff = 0.0f64;
    let mut nonzero = 0.0f64;
    for &m in &y1.mid_qn.clone() {
        match (y1.block(m), y8.block(m)) {
            (Some(a), Some(b)) => {
                assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
                for j in 0..a.ncols() {
                    for i in 0..a.nrows() {
                        maxdiff = maxdiff.max((*a.get(i, j) - *b.get(i, j)).abs());
                        nonzero = nonzero.max(a.get(i, j).abs());
                    }
                }
            }
            (None, None) => {}
            _ => panic!("threaded vs serial mid-block presence mismatch at {m:?}"),
        }
    }
    assert!(
        nonzero > 1e-6,
        "d2t: vacuous — two_site_sketched_hamiltonian output is all zero"
    );
    assert!(
        maxdiff < 1e-12,
        "d2t: threaded vs serial two_site_sketched_hamiltonian max block diff {maxdiff:e}"
    );
}

#[test]
fn s5a_d3_twosite_matvec_naive_consistency() {
    let mut rng = seeded(0x5a22);
    let space = Su2Space::new(6, 4, 2);
    let t = ha_tensor::<f64>(6, &[(0, 1, -1.0), (1, 2, -0.7), (2, 3, 0.3)]);
    let v = hb_tensor::<f64>(6, &[(2, 0, 3, 1, -1.0), (1, 3, 0, 2, -1.0)]);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.right_orthogonalize(true);
    let fl = Su2Sketch::<f64>::identity(space, 0, 1);
    let chain = right_to_left_chain(&h, &x);
    let fr = &chain[2];
    let tc = contract(x.core(0), x.core(1));
    let want = h.two_site_sketched_hamiltonian(&tc, &fl, fr);
    let (xk, xk1) = factor_qc(&tc);
    let l = qntt::su2::hamiltonian::matvec::sketch_h_mul(&fl, &h, &xk);
    let r = qntt::su2::hamiltonian::matvec::h_sketch_mul(&h, &xk1, fr);
    let want2 = contract(&l, &r);
    let mut nrm = 0.0f64;
    let mut refn = 0.0f64;
    let mut compared = 0usize;
    for &m in &want.mid_qn {
        let Some(a) = want.block(m) else { continue };
        if a.nrows() == 0 || a.ncols() == 0 {
            continue;
        }
        let b = want2
            .block(m)
            .unwrap_or_else(|| panic!("d3: the naive build dropped mid block {m:?}"));
        let (d, r) = block_diff_and_norm(a, b);
        nrm += d * d;
        refn += r * r;
        compared += 1;
    }
    assert!(compared > 0, "d3: vacuous — no mid block was compared");
    assert!(
        refn.sqrt() > 1e-8,
        "d3: vacuous — the two-site matvec output is ~zero (||want||_F = {})",
        refn.sqrt()
    );
    assert!(
        nrm.sqrt() <= 1e-12 * refn.sqrt().max(1.0),
        "d3: naive contract(L,R) should equal two_site_sketched_hamiltonian exactly"
    );
}

#[test]
fn s5a_d6_twosite_workspace_matches_no_workspace() {
    let mut rng = seeded(0x5a29);
    let space = Su2Space::new(6, 4, 2);
    let t = ha_tensor::<f64>(6, &[(0, 1, -1.0), (1, 2, -0.7), (2, 3, 0.3)]);
    let v = hb_tensor::<f64>(6, &[(2, 0, 3, 1, -1.0), (1, 3, 0, 2, -1.0)]);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.right_orthogonalize(true);
    let fl = Su2Sketch::<f64>::identity(space, 0, 1);
    let chain = right_to_left_chain(&h, &x);
    let fr = &chain[2];
    let tc = contract(x.core(0), x.core(1));

    let want = h.two_site_sketched_hamiltonian(&tc, &fl, fr);
    let mut ws = qntt::su2::hamiltonian::matvec::su2_twosite_workspace::<f64>();
    let got1 = h.two_site_sketched_hamiltonian_ws(&mut ws, &tc, &fl, fr);
    let got2 = h.two_site_sketched_hamiltonian_ws(&mut ws, &tc, &fl, fr);
    assert!(ws.is_some(), "d6: the workspace should be populated");

    let mut compared = 0usize;
    let mut refn = 0.0f64;
    for &m in &want.mid_qn {
        let Some(a) = want.block(m) else { continue };
        if a.nrows() == 0 || a.ncols() == 0 {
            continue;
        }
        for (tag, got) in [("first", &got1), ("cached", &got2)] {
            let b = got
                .block(m)
                .unwrap_or_else(|| panic!("d6: the {tag} ws build dropped mid block {m:?}"));
            let (d, r) = block_diff_and_norm(a, b);
            assert!(
                d <= 1e-13 * r.max(1.0),
                "d6: the {tag} workspace build diverges from the no-workspace form at {m:?}"
            );
            refn += r * r;
        }
        compared += 1;
    }
    assert!(compared > 0, "d6: vacuous — no mid block was compared");
    assert!(
        refn.sqrt() > 1e-8,
        "d6: vacuous — the two-site matvec output is ~zero"
    );

    let tc2 = contract(x.core(2), x.core(3));
    let lchain = left_to_right_chain(&h, &x);
    let (fl2, fr2) = (&lchain[2], &chain[4]);
    let want2 = h.two_site_sketched_hamiltonian(&tc2, fl2, fr2);
    let got3 = h.two_site_sketched_hamiltonian_ws(&mut ws, &tc2, fl2, fr2);
    for &m in &want2.mid_qn {
        let Some(a) = want2.block(m) else { continue };
        if a.nrows() == 0 || a.ncols() == 0 {
            continue;
        }
        let b = got3
            .block(m)
            .unwrap_or_else(|| panic!("d6: the rebuilt ws build dropped mid block {m:?}"));
        let (d, r) = block_diff_and_norm(a, b);
        assert!(
            d <= 1e-13 * r.max(1.0),
            "d6: reusing the workspace across a rank change diverges from the no-workspace form \
             at {m:?} — the staleness key does not cover the outer ranks"
        );
    }
}

#[test]
fn s5a_c1_onesite_effective_diagonal_vs_bruteforce() {
    let mut rng = seeded(0x5a23);
    let space = Su2Space::new(6, 4, 2);
    let mut t = ha_tensor::<f64>(6, &[(0, 1, -1.0)]);
    for i in 0..6 {
        t.set(&[i, i], 0.3 + 0.1 * i as f64);
    }
    let v = hb_tensor::<f64>(6, &[(2, 0, 3, 1, -0.3), (1, 3, 0, 2, -0.3)]);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.right_orthogonalize(true);
    for &k in &[0usize, 2, 5] {
        let mut x2 = x.clone();
        x2.move_core(k, false);
        let lchain = left_to_right_chain(&h, &x2);
        let rchain = right_to_left_chain(&h, &x2);
        let fl = &lchain[k];
        let fr = &rchain[k + 1];
        let ws = su2_onesite_workspace(&h, x2.core(k));
        let mut lop = |v: &Su2Core<f64>| h.sketched_hamiltonian(v, fl, fr, &ws);
        let d = h.effective_diagonal(fl, fr, x2.core(k), &ws);
        let o = effective_diagonal_bruteforce(&mut lop, x2.core(k));
        let n_o = InnerProductVector::norm(&o);
        assert!(
            n_o > 1e-8,
            "c1: non-vacuity at k={k} — the brute-force diagonal should not be ~0"
        );
        let mut diff = d.clone();
        InnerProductVector::axpy(&mut diff, -1.0, &o);
        let dev = InnerProductVector::norm(&diff);
        assert!(
            dev <= 1e-10 * n_o.max(1.0),
            "c1: effective_diagonal disagrees with the brute-force oracle at k={k} (dev={dev}, ||O||={n_o})"
        );
    }
}

#[test]
fn s5a_c3_twosite_effective_diagonal_vs_bruteforce() {
    let mut rng = seeded(0x5a25);
    let space = Su2Space::new(6, 4, 2);
    let mut t = ha_tensor::<f64>(6, &[(0, 1, -1.0)]);
    for i in 0..6 {
        t.set(&[i, i], 0.3 + 0.1 * i as f64);
    }
    let v = hb_tensor::<f64>(6, &[(2, 0, 3, 1, -0.3), (1, 3, 0, 2, -0.3)]);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.right_orthogonalize(true);
    for &k in &[0usize, 2, 4] {
        let mut x2 = x.clone();
        x2.move_core(k, false);
        let lchain = left_to_right_chain(&h, &x2);
        let rchain = right_to_left_chain(&h, &x2);
        let fl = &lchain[k];
        let fr = &rchain[k + 2];
        let tc = contract(x2.core(k), x2.core(k + 1));
        let mut lop = |v: &Su2ContractedCores<f64>| h.two_site_sketched_hamiltonian(v, fl, fr);
        let d = h.two_site_effective_diagonal(fl, fr, &tc);
        let o = effective_diagonal_bruteforce(&mut lop, &tc);
        let n_o = InnerProductVector::norm(&o);
        assert!(
            n_o > 1e-8,
            "c3: non-vacuity at k={k} — the brute-force diagonal should not be ~0"
        );
        let mut diff = d.clone();
        InnerProductVector::axpy(&mut diff, -1.0, &o);
        let dev = InnerProductVector::norm(&diff);
        assert!(
            dev <= 1e-10 * n_o.max(1.0),
            "c3: two-site effective_diagonal disagrees with the brute-force oracle at k={k}"
        );
    }
}

#[test]
fn s5a_c2_c4_effective_diagonal_complex() {
    let mut rng = seeded(0x5a24);
    let space = Su2Space::new(6, 4, 2);
    let mut t = ha_tensor::<C64>(6, &[(0, 1, -1.0)]);
    for i in 0..6 {
        t.set(&[i, i], C64::from_f64(0.3 + 0.1 * i as f64));
    }
    let v = hb_tensor::<C64>(6, &[(2, 0, 3, 1, -0.3), (1, 3, 0, 2, -0.3)]);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let mut x: Su2TtVector<C64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.right_orthogonalize(true);

    let k = 2usize;
    x.move_core(k, false);
    let max_im = (0..x.core(k).num_coeffs())
        .map(|i| x.core(k).coeff(i).im.abs())
        .fold(0.0f64, f64::max);
    assert!(max_im > 1e-6, "c2: vacuous — the state is real-valued");

    let lchain = left_to_right_chain(&h, &x);
    let rchain = right_to_left_chain(&h, &x);

    let (fl, fr) = (&lchain[k], &rchain[k + 1]);
    let ws = su2_onesite_workspace(&h, x.core(k));
    let mut lop = |v: &Su2Core<C64>| h.sketched_hamiltonian(v, fl, fr, &ws);
    let d = h.effective_diagonal(fl, fr, x.core(k), &ws);
    let o = effective_diagonal_bruteforce(&mut lop, x.core(k));
    let n_o = InnerProductVector::norm(&o);
    assert!(
        n_o > 1e-8,
        "c2: non-vacuity — the brute-force diagonal is ~0"
    );
    let mut diff = d.clone();
    InnerProductVector::axpy(&mut diff, -C64::from_f64(1.0), &o);
    assert!(
        InnerProductVector::norm(&diff) <= 1e-10 * n_o.max(1.0),
        "c2: complex one-site effective_diagonal disagrees with the brute-force oracle"
    );
    let max_im_d = (0..d.num_coeffs())
        .map(|i| d.coeff(i).im.abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_im_d <= 1e-12 * n_o.max(1.0),
        "c2: the one-site effective diagonal of a Hermitian H must be real (max |Im| = {max_im_d})"
    );

    let fr2 = &rchain[k + 2];
    let tc = contract(x.core(k), x.core(k + 1));
    let mut lop2 = |v: &Su2ContractedCores<C64>| h.two_site_sketched_hamiltonian(v, fl, fr2);
    let d2 = h.two_site_effective_diagonal(fl, fr2, &tc);
    let o2 = effective_diagonal_bruteforce(&mut lop2, &tc);
    let n_o2 = InnerProductVector::norm(&o2);
    assert!(
        n_o2 > 1e-8,
        "c4: non-vacuity — the brute-force diagonal is ~0"
    );
    let mut diff2 = d2.clone();
    InnerProductVector::axpy(&mut diff2, -C64::from_f64(1.0), &o2);
    assert!(
        InnerProductVector::norm(&diff2) <= 1e-10 * n_o2.max(1.0),
        "c4: complex two-site effective_diagonal disagrees with the brute-force oracle"
    );
    let max_im_d2 = (0..d2.num_coeffs())
        .map(|i| d2.coeff(i).im.abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_im_d2 <= 1e-12 * n_o2.max(1.0),
        "c4: the two-site effective diagonal of a Hermitian H must be real"
    );
}

#[test]
fn s5a_d4_sketching_step_right_reference() {
    let mut rng = seeded(0x5a27);
    let space = Su2Space::new(6, 4, 2);
    let t = ha_tensor::<f64>(6, &[(0, 1, -1.0), (2, 3, -0.6)]);
    let h = Su2SparseHamiltonian::new(space, &t, &zero_v(6), None, 1e-12);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.left_orthogonalize(true);
    let k = 2usize;
    let lchain = left_to_right_chain(&h, &x);
    let fl = &lchain[k];
    let ws = su2_onesite_workspace(&h, x.core(k));
    let c = h.sketching_step_right(&x, fl, &ws);

    let reference = adjoint_core_times_core(x.core(k), &sketch_h_mul(fl, &h, x.core(k)));
    let mut compared = 0usize;
    let mut refn_tot = 0.0f64;
    for s in c.sectors() {
        let Some(want) = reference.block(s) else {
            continue;
        };
        if want.nrows() == 0 || want.ncols() == 0 {
            continue;
        }
        let got = c
            .block(s)
            .unwrap_or_else(|| panic!("d4: sketching_step_right dropped sector {s:?}"));
        let (diff, refn) = block_diff_and_norm(want, got);
        assert!(
            diff <= 1e-12 * refn.max(1.0),
            "d4: sketching_step_right must equal the UNWEIGHTED adjoint(Xₖ)·(Fᴸ·H·Xₖ) reference \
             at sector {s:?} (diff={diff}, ||ref||={refn}) — a θ-weighted fold here is the \
             mirror-image error of a plain adjoint in sketching_step_left"
        );
        refn_tot += refn * refn;
        compared += 1;
    }
    assert!(compared > 0, "d4: vacuous — no sector was compared");
    assert!(
        refn_tot.sqrt() > 1e-8 && c.norm() > 1e-8,
        "d4: vacuous — the environment sketch is ~zero (||ref||={}, ||c||={})",
        refn_tot.sqrt(),
        c.norm()
    );
}

#[test]
fn s5a_d5_sketching_step_left_reference() {
    let mut rng = seeded(0x5a28);
    let space = Su2Space::new(6, 4, 2);
    let t = ha_tensor::<f64>(6, &[(0, 1, -1.0), (2, 3, -0.6)]);
    let h = Su2SparseHamiltonian::new(space, &t, &zero_v(6), None, 1e-12);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.right_orthogonalize(true);
    let k = 2usize;
    let ws = su2_onesite_workspace(&h, x.core(k));
    let rchain = right_to_left_chain(&h, &x);
    let fr = &rchain[k + 1];
    let n_theta = x
        .core(k)
        .row_qn
        .iter()
        .filter(|&&l| l.two_s > 0 && Su2Family::Up.apply(l, x.space.pg.irrep_of(k)).is_some())
        .count();
    assert!(
        n_theta >= 1,
        "d5: non-vacuity precondition — need at least one two_s>0 row sector with an Up branch"
    );
    let c = h.sketching_step_left(&x, fr, &ws);
    assert!(
        c.norm() > 1e-8,
        "d5: vacuous — the environment sketch is ~zero (||c|| = {})",
        c.norm()
    );
    let reference = &right_to_left_chain_projected(&h, &x, false)[k];
    let sabotaged = &right_to_left_chain_plain(&h, &x)[k];
    let (mut dev_ref, mut dev_bad) = (0.0f64, 0.0f64);
    let mut compared = 0usize;
    for s in c.sectors() {
        let Some(a) = c.block(s) else { continue };
        if a.nrows() == 0 || a.ncols() == 0 {
            continue;
        }
        let r = reference
            .block(s)
            .unwrap_or_else(|| panic!("d5: the rproject reference dropped sector {s:?}"));
        let b = sabotaged
            .block(s)
            .unwrap_or_else(|| panic!("d5: the plain-adjoint build dropped sector {s:?}"));
        for j in 0..a.ncols() {
            for i in 0..a.nrows() {
                dev_ref = dev_ref.max((*a.get(i, j) - *r.get(i, j)).abs());
                dev_bad = dev_bad.max((*a.get(i, j) - *b.get(i, j)).abs());
            }
        }
        compared += 1;
    }
    assert!(compared > 0, "d5: vacuous — no sector was compared");
    assert!(
        dev_ref <= 1e-12,
        "d5: sketching_step_left must equal the θ²-weighted `rproject` reference (dev={dev_ref})"
    );
    assert!(
        dev_bad > 1e-6,
        "d5: sketching_step_left must DIFFER from the plain-adjoint `rproject_plain` build — a \
         plain adjoint is a silent O(1) error (dev={dev_bad})"
    );
}
