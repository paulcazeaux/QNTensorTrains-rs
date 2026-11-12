// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::{Scalar, tt_state};
use qntt_testkit::randomized_state;
use std::path::{Path, PathBuf};

fn state_from_sites<T: Scalar>(d: usize, up_sites: &[usize], dn_sites: &[usize]) -> TtVector<T> {
    let mut occ_up = vec![false; d];
    let mut occ_dn = vec![false; d];
    for &s in up_sites {
        occ_up[s] = true;
    }
    for &s in dn_sites {
        occ_dn[s] = true;
    }
    tt_state(&occ_up, &occ_dn)
}

fn draw_distinct(d: usize, n: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..d).collect();
    for i in 0..n {
        let j = i + rng.random_range(0..(d - i));
        pool.swap(i, j);
    }
    pool.truncate(n);
    pool
}

fn be_fcidump_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/Be.ezfio.FCIDUMP")
}

fn frob_diff<T: Scalar>(a: &Sparse3Body<T>, b: &Sparse3Body<T>) -> f64 {
    use std::collections::HashSet;
    let mut keys: HashSet<[usize; 6]> = HashSet::new();
    keys.extend(a.support().map(|(k, _)| k));
    keys.extend(b.support().map(|(k, _)| k));
    keys.iter()
        .map(|&k| {
            let d = a.get(&k) - b.get(&k);
            d.abs_s() * d.abs_s()
        })
        .sum::<f64>()
        .sqrt()
}
use qntt::coeff::{Sparse3Body, ThreeBody};
use qntt::fcidump::{ThreeBodySymmetry, TwoBodySymmetry, read_tcdump};
use qntt::hamiltonian::Construction;
use qntt::integrals::{IntegralsOptions, ThreeBodyOptions, load_fcidump};
use qntt::{CoeffTensor, QnSpace, SparseHamiltonian, TtVector, rayleigh_quotient};
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::io::Write as _;

fn tc_opts() -> IntegralsOptions {
    IntegralsOptions {
        symmetry: TwoBodySymmetry::Pairswap,
        three_body: Some(ThreeBodyOptions::default()),
        ..IntegralsOptions::default()
    }
}

#[test]
fn t7b_be_header_and_symmetry() {
    let be_path = be_fcidump_path();
    let load = load_fcidump::<f64>(&be_path, None, &tc_opts()).expect("load_fcidump(Be)");
    let data = &load.ints;
    assert_eq!(data.norb, 9);
    assert_eq!(data.nelec, 4);
    assert_eq!(data.ms2, 0);
    assert!(load.w.is_none());
    assert_eq!(load.space.nup, 2);
    assert_eq!(load.space.ndn, 2);
    assert!((data.e0 - (-1.887566535800648e-3)).abs() < 1e-12);
    assert!((data.t.get(&[0, 0]) - (-7.93871537813441)).abs() < 1e-11);

    let norb = data.norb;
    let mut max_asym = 0.0f64;
    for i in 0..norb {
        for j in 0..norb {
            for k in 0..norb {
                for l in 0..norb {
                    let a = data.v.get(&[i, j, k, l]);
                    let b = data.v.get(&[k, l, i, j]);
                    max_asym = max_asym.max((a - b).abs());
                }
            }
        }
    }
    assert!(max_asym < 1e-12, "pairswap symmetry violated: {max_asym}");

    let mut max_t_asym = 0.0f64;
    for i in 0..norb {
        for j in 0..norb {
            max_t_asym = max_t_asym.max((data.t.get(&[j, i]) - data.t.get(&[i, j])).abs());
        }
    }
    assert!(
        max_t_asym > 1e-3,
        "t unexpectedly (near-)hermitian: {max_t_asym}"
    );
}

#[test]
fn t7b_load_fcidump_bad_header_rejected() {
    for (nelec, ms2, tag) in [(2i64, 6i64, "ms2>nelec"), (3i64, 0i64, "odd-parity")] {
        let path = std::env::temp_dir().join(format!(
            "qntt_phase7_badhdr_{tag}_{}.fcidump.tmp",
            std::process::id()
        ));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, " &FCI NORB=  2 , NELEC=  {nelec} , MS2=  {ms2} ,").unwrap();
            writeln!(f, " &end").unwrap();
            writeln!(f, " 1.0  1  1  0  0").unwrap();
            writeln!(f, " 0.0  0  0  0  0").unwrap();
        }
        let res = load_fcidump::<f64>(&path, None, &tc_opts());
        let _ = std::fs::remove_file(&path);
        assert!(res.is_err(), "bad header ({tag}) was not rejected");
    }
}

#[test]
fn t7b_be_determinant_molecule_gate() {
    let be_path = be_fcidump_path();
    let (h, load) = SparseHamiltonian::<f64>::from_fcidump(&be_path, None, &tc_opts())
        .expect("from_fcidump(Be)");
    let data = &load.ints;
    let x: TtVector<f64> = state_from_sites(9, &[0, 1], &[0, 1]);
    x.check();
    assert!((x.norm_lr(false) - 1.0).abs() < 1e-12);

    let rq = rayleigh_quotient(&h, &x);

    let mut eref = 2.0 * (data.t.get(&[0, 0]) + data.t.get(&[1, 1]));
    for &i in &[0usize, 1] {
        for &j in &[0usize, 1] {
            eref += 2.0 * data.v.get(&[i, i, j, j]) - data.v.get(&[i, j, j, i]);
        }
    }
    assert!(
        (rq - eref).abs() < 1e-10 * eref.abs(),
        "Slater-Condon oracle mismatch: rq={rq} eref={eref}"
    );

    assert!(
        (rq - (-14.577233846433735)).abs() < 1e-9,
        "molecule gate: rq={rq}"
    );
    assert!(
        (rq + data.e0 - (-14.579121412969537)).abs() < 1e-9,
        "molecule gate (total incl. core): rq+e0={}",
        rq + data.e0
    );
}

#[test]
fn t7b_tcdump_48fold_roundtrip_and_build() {
    let d = 7usize;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7B20);

    let entries: Vec<(f64, [usize; 6])> = (0..4)
        .map(|_| {
            let x: f64 = rng.sample(StandardNormal);
            let s = draw_distinct(d, 6, &mut rng);
            (x, [s[0], s[1], s[2], s[3], s[4], s[5]])
        })
        .collect();

    const PERM3: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut wref = CoeffTensor::<f64>::zeros(d, 6);
    for &(x, [i, j, k, l, m, n]) in &entries {
        let pairs = [(i, l), (j, m), (k, n)];
        for sigma in PERM3 {
            for mask in 0u8..8 {
                let mut p = [(0usize, 0usize); 3];
                for (q, slot) in p.iter_mut().enumerate() {
                    let (a, b) = pairs[sigma[q]];
                    *slot = if (mask >> q) & 1 == 1 { (b, a) } else { (a, b) };
                }
                let idx = [p[0].0, p[0].1, p[1].0, p[1].1, p[2].0, p[2].1];
                wref.set(&idx, x);
            }
        }
    }

    let tcpath = std::env::temp_dir().join("qntt_phase7_t7b_48fold.tcdump.tmp");
    {
        let mut f = std::fs::File::create(&tcpath).expect("create temp tcdump");
        writeln!(f, " &TCDUMP NORB=  {d} ,").unwrap();
        writeln!(f, " &end").unwrap();
        for &(x, [i, j, k, l, m, n]) in &entries {
            writeln!(
                f,
                " {x}  {}  {}  {}  {}  {}  {}",
                i + 1,
                j + 1,
                k + 1,
                l + 1,
                m + 1,
                n + 1
            )
            .unwrap();
        }
    }

    let w = read_tcdump::<f64>(&tcpath, d, ThreeBodySymmetry::Full48).expect("read_tcdump");
    let wref_sparse = Sparse3Body::from_dense(&wref);
    assert!(
        frob_diff(&w, &wref_sparse) < 1e-12,
        "w != 48-fold reference expansion"
    );
    assert_eq!(
        w.nnz(),
        48 * entries.len(),
        "collision-free orbit expansion count"
    );
    for &(x, [i, j, k, l, m, n]) in &entries {
        assert!(
            (w.get(&[i, l, j, m, k, n]) - x).abs() < 1e-12,
            "TCDUMP slot map ⟨ijk|lmn⟩ → w[i,l,j,m,k,n] mismatch"
        );
    }

    let fcpath = std::env::temp_dir().join("qntt_phase7_t7b_48fold.fcidump.tmp");
    {
        let mut f = std::fs::File::create(&fcpath).expect("create temp fcidump");
        writeln!(f, " &FCI NORB=  {d} , NELEC=  4 , MS2=  0 ,").unwrap();
        writeln!(f, " &end").unwrap();
        for i in 0..d {
            writeln!(f, " {}  {}  {}  0  0", 1.0 / (i + 1) as f64, i + 1, i + 1).unwrap();
        }
        writeln!(f, " 0.25  1  2  3  4").unwrap();
        writeln!(f, " 0.0  0  0  0  0").unwrap();
    }

    let load = load_fcidump::<f64>(&fcpath, Some(&tcpath), &tc_opts()).expect("load_fcidump");
    let data = &load.ints;
    let w_data = load.w.as_ref().expect("three-body channel present");
    let mut wref_neg = wref.clone();
    for v in wref_neg.data.iter_mut() {
        *v = -*v;
    }
    let wref_neg_sparse = Sparse3Body::from_dense(&wref_neg);
    assert!(
        frob_diff(w_data, &wref_neg_sparse) < 1e-12,
        "three_body_sign=-1 not applied (H̄ should contain −(1/6)·L)"
    );
    assert!((data.v.get(&[0, 1, 2, 3]) - 0.25).abs() < 1e-12);
    assert!((data.v.get(&[2, 3, 0, 1]) - 0.25).abs() < 1e-12);
    assert!(
        data.v.get(&[1, 0, 2, 3]).abs() < 1e-15,
        "non-hermitian dump must not manufacture the eightfold image"
    );

    let (h, load2) = SparseHamiltonian::<f64>::from_fcidump(&fcpath, Some(&tcpath), &tc_opts())
        .expect("from_fcidump");
    let data2 = &load2.ints;
    let space = QnSpace::new(d, 2, 2);
    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(0x7B21);
    let x0: TtVector<f64> = randomized_state(space, 1, 3, &mut rng2);
    x0.check();
    let y = &h * &x0;
    let w_dyn: Option<&dyn ThreeBody<f64>> = load2.w.as_ref().map(|w| w as &dyn ThreeBody<f64>);
    let href = SparseHamiltonian::new_with(
        space,
        &data2.t,
        &data2.v,
        w_dyn,
        f64::EPSILON,
        Construction::Median,
    );
    let yref = &href * &x0;
    let diff = &y - &yref;
    assert!(
        diff.norm_lr(false) <= 1e-12 * y.norm_lr(false),
        "H built via from_fcidump != SparseHamiltonian::new_with(data, Median)"
    );

    let _ = std::fs::remove_file(&tcpath);
    let _ = std::fs::remove_file(&fcpath);
}
