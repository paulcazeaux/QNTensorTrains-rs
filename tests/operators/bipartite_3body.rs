// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::coeff::ThreeBody;
use qntt::hamiltonian::Construction;
use qntt::hamiltonian::assembly::{BlockStat, assemble_with, bipartite_probe};
use qntt::{
    CoeffTensor, Orbital, QnSpace, Scalar, SparseHamiltonian, Spin, TtVector, rayleigh_quotient,
};
use qntt_testkit::dense_apply_string;
use qntt_testkit::randomized_state;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::hash::{Hash, Hasher};

fn dense_apply_hamiltonian_full<T: Scalar>(
    d: usize,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    w: &CoeffTensor<T>,
    vec: &[T],
) -> Vec<T> {
    let len = vec.len();
    let mut out = vec![T::zero(); len];
    let spins = [Spin::Up, Spin::Dn];
    let accumulate = |ops: &[(Orbital, bool)], coeff: T, out: &mut [T]| {
        let term = dense_apply_string(d, ops, vec);
        for (o, tm) in out.iter_mut().zip(term) {
            *o += tm * coeff;
        }
    };

    for i in 0..d {
        for j in 0..d {
            let c = t.get(&[i, j]);
            if c == T::zero() {
                continue;
            }
            for &s in &spins {
                accumulate(
                    &[(Orbital::new(i, s), true), (Orbital::new(j, s), false)],
                    c,
                    &mut out,
                );
            }
        }
    }

    let half = T::from_f64(0.5);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let c = v.get(&[i, j, k, l]);
                    if c == T::zero() {
                        continue;
                    }
                    for &s1 in &spins {
                        for &s2 in &spins {
                            let (oi, oj) = (Orbital::new(i, s1), Orbital::new(j, s1));
                            let (ok, ol) = (Orbital::new(k, s2), Orbital::new(l, s2));
                            if oi != ok && oj != ol {
                                accumulate(
                                    &[(oi, true), (ok, true), (ol, false), (oj, false)],
                                    c * half,
                                    &mut out,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    let sixth = T::from_f64(1.0 / 6.0);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    for m in 0..d {
                        for n in 0..d {
                            let c = w.get(&[i, j, k, l, m, n]);
                            if c == T::zero() {
                                continue;
                            }
                            for &s1 in &spins {
                                for &s2 in &spins {
                                    for &s3 in &spins {
                                        let (oi, oj) = (Orbital::new(i, s1), Orbital::new(j, s1));
                                        let (ok, ol) = (Orbital::new(k, s2), Orbital::new(l, s2));
                                        let (om, on) = (Orbital::new(m, s3), Orbital::new(n, s3));
                                        let cd = oi != ok && oi != om && ok != om;
                                        let ad = oj != ol && oj != on && ol != on;
                                        if cd && ad {
                                            accumulate(
                                                &[
                                                    (oi, true),
                                                    (ok, true),
                                                    (om, true),
                                                    (on, false),
                                                    (ol, false),
                                                    (oj, false),
                                                ],
                                                c * sixth,
                                                &mut out,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    out
}

fn random_coeffs(d: usize, rng: &mut Xoshiro256PlusPlus) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            t.set(&[i, j], rng.sample::<f64, _>(StandardNormal));
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if v.get(&[i, j, k, l]) != 0.0 {
                        continue;
                    }
                    let x: f64 = rng.sample(StandardNormal);
                    v.set(&[i, j, k, l], x);
                    v.set(&[k, l, i, j], x);
                }
            }
        }
    }
    (t, v)
}

fn random_w(d: usize, count: usize, rng: &mut Xoshiro256PlusPlus) -> CoeffTensor<f64> {
    let mut w = CoeffTensor::<f64>::zeros(d, 6);
    for _ in 0..count {
        let idx = [
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
        ];
        w.set(&idx, rng.random_range(-1.0..1.0));
    }
    w
}

fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in a.iter().zip(b) {
        num += (x - y) * (x - y);
        den += y * y;
    }
    (num / den.max(1e-300)).sqrt()
}

fn build_bipartite_w(
    space: QnSpace,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    w: Option<&dyn ThreeBody<f64>>,
) -> SparseHamiltonian<f64> {
    SparseHamiltonian::new_with(space, t, v, w, f64::EPSILON, Construction::Bipartite)
}

fn build_median_w(
    space: QnSpace,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    w: Option<&dyn ThreeBody<f64>>,
) -> SparseHamiltonian<f64> {
    SparseHamiltonian::new_with(space, t, v, w, f64::EPSILON, Construction::Median)
}

fn hash_assembled(a: &qntt::hamiltonian::assembly::Assembled<f64>) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let nirrep = a.state_qns[0].nirrep();
    a.coo.len().hash(&mut h);
    for (k, nest) in a.coo.iter().enumerate() {
        for (yf, ysec, xf, xsec, cell) in nest.iter_grid_order() {
            (
                k, yf as u8, ysec.nup, ysec.ndn, xf as u8, xsec.nup, xsec.ndn,
            )
                .hash(&mut h);
            if nirrep > 1 {
                (ysec.irrep.index(), xsec.irrep.index()).hash(&mut h);
            }
            cell.rows.len().hash(&mut h);
            for ((&r, &c), &val) in cell.rows.iter().zip(&cell.cols).zip(&cell.vals) {
                r.hash(&mut h);
                c.hash(&mut h);
                val.to_bits().hash(&mut h);
            }
        }
    }
    a.state_qns.len().hash(&mut h);
    for (b, grid) in a.state_qns.iter().enumerate() {
        for (s, list) in grid.iter() {
            if list.is_empty() {
                continue;
            }
            (b, s.nup, s.ndn).hash(&mut h);
            if nirrep > 1 {
                s.irrep.index().hash(&mut h);
            }
            for sec in list {
                (sec.nup, sec.ndn).hash(&mut h);
                if nirrep > 1 {
                    sec.irrep.index().hash(&mut h);
                }
            }
        }
    }
    h.finish()
}

#[test]
fn b3_dense_oracle_multishape_w() {
    let shapes = [
        QnSpace::new(6, 3, 3),
        QnSpace::from_n_two_sz(5, 4, 0),
        QnSpace::new(6, 2, 1),
    ];
    for (si, &space) in shapes.iter().enumerate() {
        let d = space.d;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB3_0000 ^ si as u64);
        let (t, v) = random_coeffs(d, &mut rng);
        let w = random_w(d, 40, &mut rng);

        let h_bip = build_bipartite_w(space, &t, &v, Some(&w));
        let h_med = build_median_w(space, &t, &v, Some(&w));

        let tol_cross = (50.0 * d as f64 * f64::EPSILON).max(1e-12);
        for seed in 0..3u64 {
            let mut r2 = Xoshiro256PlusPlus::seed_from_u64(0xDA7A_0000 ^ (si as u64) << 8 ^ seed);
            let x: TtVector<f64> = randomized_state(space, 1, 3, &mut r2);
            let xd = x.to_dense();

            let refv = dense_apply_hamiltonian_full(d, &t, &v, &w, &xd);
            let y_bip = h_bip.apply(&x).to_dense();
            let y_med = h_med.apply(&x).to_dense();

            let e_ref = rel_diff(&y_bip, &refv);
            assert!(
                e_ref < 1e-9,
                "shape {si} (d={d}) seed {seed}: bipartite apply vs dense JW oracle (w on) rel-err {e_ref:.2e}"
            );
            let e_cross = rel_diff(&y_bip, &y_med);
            assert!(
                e_cross < tol_cross,
                "shape {si} (d={d}) seed {seed}: bipartite vs median apply (w on) rel-err \
                 {e_cross:.2e} (tol {tol_cross:.2e})"
            );
        }
    }
}

#[test]
fn b3_dense_oracle_comp_channel_nl3_witness() {
    let space = QnSpace::new(6, 3, 3);
    let d = space.d;

    type WitnessSystem = (u64, CoeffTensor<f64>, CoeffTensor<f64>, CoeffTensor<f64>);
    let mut chosen: Option<WitnessSystem> = None;
    for trial in 0..12u64 {
        let seed = 0x30DE_0000 ^ trial;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let (t, v) = random_coeffs(d, &mut rng);
        let w = random_w(d, 60, &mut rng);
        let blocks: Vec<BlockStat<f64>> =
            bipartite_probe(space, &t, &v, Some(&w as &dyn ThreeBody<f64>), f64::EPSILON);
        let hit = blocks.iter().any(|blk| {
            blk.bond > 0
                && blk.bond < d
                && blk
                    .comp_channels
                    .iter()
                    .any(|c| c.distinct_nl_up.contains(&3) || c.distinct_nl_dn.contains(&3))
        });
        if hit {
            chosen = Some((seed, t, v, w));
            break;
        }
    }
    let (seed, t, v, w) = chosen.expect(
        "VACUOUS: no interior Comp channel with an nl=3 rider materialized across 12 trial \
         seeds at QnSpace::new(6,3,3) — perturb the shape/coeffs until bipartite_probe witnesses \
         one (the guard is the spec, the trials are a starting point); this is an escalation-worthy \
         finding if it persists, per bipartite-design.md §1 step 2 / spec 'bipartite-3body-path' §10",
    );
    println!("b3_dense_oracle_comp_channel_nl3_witness: witnessed at seed {seed:#x}");

    let h_bip = build_bipartite_w(space, &t, &v, Some(&w));
    let h_med = build_median_w(space, &t, &v, Some(&w));
    let tol_cross = (50.0 * d as f64 * f64::EPSILON).max(1e-12);
    for trial in 0..4u64 {
        let mut r2 = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0xC0DE_0000 ^ trial);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut r2);
        let xd = x.to_dense();
        let refv = dense_apply_hamiltonian_full(d, &t, &v, &w, &xd);
        let y_bip = h_bip.apply(&x).to_dense();
        let y_med = h_med.apply(&x).to_dense();
        let e_ref = rel_diff(&y_bip, &refv);
        assert!(
            e_ref < 1e-9,
            "Comp-witness system, trial {trial}: bipartite vs dense JW oracle rel-err {e_ref:.2e}"
        );
        let e_cross = rel_diff(&y_bip, &y_med);
        assert!(
            e_cross < tol_cross,
            "Comp-witness system, trial {trial}: bipartite vs median rel-err {e_cross:.2e}"
        );
    }
}

#[test]
fn b3_cross_oracle_small_d_w() {
    let space = QnSpace::new(6, 3, 3);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB3C4_0553);
    let (t, v) = random_coeffs(d, &mut rng);
    let w = random_w(d, 50, &mut rng);
    let h_bip = build_bipartite_w(space, &t, &v, Some(&w));
    let h_med = build_median_w(space, &t, &v, Some(&w));

    for seed in 0..4u64 {
        let mut r2 = Xoshiro256PlusPlus::seed_from_u64(0x005E_EDB3 ^ seed);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut r2);
        let y_bip = h_bip.apply(&x).to_dense();
        let y_med = h_med.apply(&x).to_dense();
        let e = rel_diff(&y_bip, &y_med);
        assert!(
            e < 1e-12,
            "seed {seed}: cross-construction dense outputs (w on) differ rel-err {e:.2e}"
        );
        let q_bip = rayleigh_quotient(&h_bip, &x);
        let q_med = rayleigh_quotient(&h_med, &x);
        let rq = (q_bip - q_med).abs() / q_med.abs().max(1e-12);
        assert!(
            rq < 1e-12,
            "seed {seed}: cross-construction Rayleigh quotient (w on) differs rel {rq:.2e} \
             (bip {q_bip}, med {q_med})"
        );
    }
}

#[test]
fn b3_coo_artifact_bit_identity_w() {
    let space = QnSpace::new(6, 3, 3);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB17_3B0D);
    let (t, v) = random_coeffs(d, &mut rng);
    let w = random_w(d, 40, &mut rng);
    let w_dyn: Option<&dyn ThreeBody<f64>> = Some(&w);

    let a1 = assemble_with(space, &t, &v, w_dyn, f64::EPSILON, Construction::Bipartite);
    let a2 = assemble_with(space, &t, &v, w_dyn, f64::EPSILON, Construction::Bipartite);
    assert_eq!(
        hash_assembled(&a1),
        hash_assembled(&a2),
        "run-to-run 3-body COO artifact hash differs (w-on bipartite construction is not \
         seed-free)"
    );

    assert_eq!(a1.coo.len(), a2.coo.len());
    for k in 0..a1.coo.len() {
        let (g1, g2) = (a1.coo[k].iter_grid_order(), a2.coo[k].iter_grid_order());
        assert_eq!(
            g1.len(),
            g2.len(),
            "populated cell count differs at site {k}"
        );
        for (e1, e2) in g1.iter().zip(&g2) {
            let (yf, ysec, xf, xsec, cell1) = *e1;
            let cell2 = e2.4;
            assert_eq!(
                (yf, ysec, xf, xsec),
                (e2.0, e2.1, e2.2, e2.3),
                "populated cell key differs at site {k}"
            );
            assert_eq!(
                cell1.rows, cell2.rows,
                "row mismatch site {k} yf {yf} xf {xf} ysec {ysec:?} xsec {xsec:?}"
            );
            assert_eq!(cell1.cols, cell2.cols, "col mismatch site {k}");
            let b1: Vec<u64> = cell1.vals.iter().map(|x| x.to_bits()).collect();
            let b2: Vec<u64> = cell2.vals.iter().map(|x| x.to_bits()).collect();
            assert_eq!(b1, b2, "val (bits) mismatch site {k}");
        }
    }
}

#[test]
fn b3_coo_artifact_bit_identity_pinned_w() {
    const PINNED_B3_COO_HASH: u64 = 15628791015691602128;
    let space = QnSpace::new(6, 3, 3);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB17_3B0D);
    let (t, v) = random_coeffs(d, &mut rng);
    let w = random_w(d, 40, &mut rng);
    let w_dyn: Option<&dyn ThreeBody<f64>> = Some(&w);

    let a = assemble_with(space, &t, &v, w_dyn, f64::EPSILON, Construction::Bipartite);
    let h = hash_assembled(&a);
    assert_eq!(
        h, PINNED_B3_COO_HASH,
        "pinned 3-body (w-on) COO hash changed — if intentional, update PINNED_B3_COO_HASH to {h}"
    );
}

#[test]
fn r3_cover_switch_leaves_the_balance_window_on_3body_terms() {
    use qntt::hamiltonian::assembly::switch_profile;
    let space = QnSpace::new(6, 3, 3);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x83_0FF0);
    let (t, v) = random_coeffs(d, &mut rng);
    let w = random_w(d, 40, &mut rng);
    let prof = switch_profile(space, &t, &v, Some(&w as &dyn ThreeBody<f64>), f64::EPSILON);
    let three: Vec<_> = prof
        .iter()
        .filter(|p| p.nbody == 3 && p.cover_bond.is_some())
        .collect();
    assert!(!three.is_empty(), "no 3-body terms survived the sweep");
    let outside = three
        .iter()
        .filter(|p| {
            let c = p.cover_bond.unwrap();
            c < p.window.0 || c > p.window.1
        })
        .count();
    let differ = three
        .iter()
        .filter(|p| p.cover_bond != Some(p.median_bond))
        .count();
    println!(
        "SWITCH_PROFILE_3BODY d={d} terms={} differ={differ} outside_balance_window={outside}",
        three.len()
    );
    assert!(
        outside > 0,
        "VACUOUS for the widened OpenIdx: every 3-body cover switch stayed inside the balance \
         window, so the [_; 3] encoding would have sufficed on this system"
    );
}
