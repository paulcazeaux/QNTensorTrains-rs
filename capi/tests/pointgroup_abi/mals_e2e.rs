// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::io::Write;
use std::ptr;

use qntt::{
    CoeffTensor, Irrep, PointGroup, PointGroupKind, QnSpace, SparseHamiltonian, TtVector, dot,
    tt_state_in,
};
use qntt_capi::err::QNTT_OK;
use qntt_capi::exports::*;
use qntt_capi::fcidump::*;
use qntt_capi::opts::QnttMalsOptions;
use qntt_capi::pointgroup::*;

use crate::util::{c2v_coeffs, c2v_orbsym, c2v_pg, last_error};
use crate::{PG_AUTO, PG_C2V};

fn assert_fixture_nonvacuous(pg: &PointGroup, d: usize, t: &CoeffTensor<f64>) {
    assert!(pg.order() > 1, "fixture group must be nontrivial");
    assert!(
        (1..d).any(|k| pg.irrep_of(k) != pg.irrep_of(0)),
        "fixture group must be inhomogeneous across sites"
    );
    assert!(
        (0..d).any(|i| (0..d).any(|j| pg.irrep_of(i) != pg.irrep_of(j) && t.get(&[i, j]) == 0.0)),
        "vacuous: t has no symmetry-forbidden (necessarily-zero) entry"
    );
}

fn determinants_with_irrep(
    pg: &PointGroup,
    d: usize,
    nup: usize,
    ndn: usize,
    target: Irrep,
) -> Vec<(Vec<bool>, Vec<bool>)> {
    let mut out = Vec::new();
    for idx in 0..(1usize << (2 * d)) {
        let dig: Vec<usize> = (0..d).map(|k| (idx >> (2 * k)) & 3).collect();
        let cu = dig.iter().filter(|&&x| x == 1 || x == 3).count();
        let cd = dig.iter().filter(|&&x| x == 2 || x == 3).count();
        if cu != nup || cd != ndn {
            continue;
        }
        let mut ir = Irrep::IDENTITY;
        for (k, &x) in dig.iter().enumerate() {
            if x == 1 || x == 2 {
                ir = ir.fuse(pg.irrep_of(k));
            }
        }
        if ir != target {
            continue;
        }
        out.push((
            dig.iter().map(|&x| x == 1 || x == 3).collect(),
            dig.iter().map(|&x| x == 2 || x == 3).collect(),
        ));
    }
    out
}

fn smallest_eigenvalue_sym(a: &mut [f64], n: usize) -> f64 {
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
    (0..n).map(|i| a[i * n + i]).fold(f64::INFINITY, f64::min)
}

fn dense_target_block_ground_state(
    pg: &PointGroup,
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    nup: usize,
    ndn: usize,
    target: Irrep,
) -> f64 {
    let dets = determinants_with_irrep(pg, d, nup, ndn, target);
    let n = dets.len();
    assert!(n > 0, "empty target-irrep determinant block");
    let n_particle = (0..(1usize << (2 * d)))
        .filter(|&idx| {
            let dig: Vec<usize> = (0..d).map(|k| (idx >> (2 * k)) & 3).collect();
            dig.iter().filter(|&&x| x == 1 || x == 3).count() == nup
                && dig.iter().filter(|&&x| x == 2 || x == 3).count() == ndn
        })
        .count();
    assert!(
        n < n_particle,
        "vacuous: the target-irrep block ({n}) is the WHOLE particle block ({n_particle})"
    );

    let space = QnSpace::with_point_group(d, nup, ndn, *pg, target);
    let h: SparseHamiltonian<f64> = SparseHamiltonian::new(space, t, v, None, f64::EPSILON);
    let basis: Vec<TtVector<f64>> = dets
        .iter()
        .map(|(u, dn)| tt_state_in::<f64>(space, u, dn))
        .collect();
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
    assert!(asym < 1e-9, "reference block is not Hermitian (max {asym})");
    smallest_eigenvalue_sym(&mut hm, n)
}

fn temp_fcidump_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("qntt_pg5abi_{tag}_{}.FCIDUMP", std::process::id()));
    p
}

#[allow(clippy::too_many_arguments)]
fn write_c2v_fcidump(
    path: &std::path::Path,
    norb: usize,
    nelec: usize,
    ms2: i64,
    orbsym_molpro: &[u8],
    isym_molpro: u8,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
) {
    let mut f = std::fs::File::create(path).expect("create temp fcidump");
    let orbsym_str: String = orbsym_molpro.iter().map(|x| format!("{x},")).collect();
    writeln!(f, " &FCI NORB=  {norb} , NELEC=  {nelec} , MS2=  {ms2} ,").unwrap();
    writeln!(f, " ORBSYM={orbsym_str}").unwrap();
    writeln!(f, " ISYM={isym_molpro},").unwrap();
    writeln!(f, " &end").unwrap();
    for i in 0..norb {
        for j in 0..norb {
            for k in 0..norb {
                for l in 0..norb {
                    let x = v.get(&[i, j, k, l]);
                    if x != 0.0 {
                        writeln!(f, " {x:.17e}  {} {} {} {}", i + 1, j + 1, k + 1, l + 1).unwrap();
                    }
                }
            }
        }
    }
    for i in 0..norb {
        for j in 0..norb {
            let x = t.get(&[i, j]);
            if x != 0.0 {
                writeln!(f, " {x:.17e}  {} {}  0  0", i + 1, j + 1).unwrap();
            }
        }
    }
    writeln!(f, " 0.0  0  0  0  0").unwrap();
}

fn c2v_fcidump_to_mals_through_abi(tag: &str, pg_kind: i32) -> (f64, f64, i32) {
    let d = 4usize;
    let (nup, ndn) = (2usize, 2usize);
    let target = Irrep::new(2);
    let pg = c2v_pg(d);
    let orbsym = c2v_orbsym(d);
    let isym = PointGroupKind::C2v.molpro_from_irrep(target).unwrap();
    let (t, v) = c2v_coeffs(&pg, d);
    assert_fixture_nonvacuous(&pg, d, &t);

    let lambda_exact = dense_target_block_ground_state(&pg, d, &t, &v, nup, ndn, target);

    let path = temp_fcidump_path(tag);
    write_c2v_fcidump(&path, d, nup + ndn, 0, &orbsym, isym, &t, &v);
    let path_c = std::ffi::CString::new(path.to_str().unwrap()).unwrap();

    let mut dump: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_fcidump_read_pg(path_c.as_ptr(), 0, pg_kind, &mut dump) },
        QNTT_OK,
        "fcidump_read_pg failed: {}",
        last_error()
    );
    let _ = std::fs::remove_file(&path);

    let mut norb = 0usize;
    let (mut nelec, mut ms2, mut e0) = (0usize, 0i64, 0.0f64);
    assert_eq!(
        unsafe { qntt_f64_fcidump_info(dump, &mut norb, &mut nelec, &mut ms2, &mut e0) },
        QNTT_OK
    );
    assert_eq!((norb, nelec, ms2), (d, nup + ndn, 0));

    let (mut kind_out, mut isym_out, mut target_out) = (0i32, 0i64, 0i32);
    assert_eq!(
        unsafe {
            qntt_f64_fcidump_point_group(dump, &mut kind_out, &mut isym_out, &mut target_out)
        },
        QNTT_OK,
        "fcidump_point_group failed: {}",
        last_error()
    );
    assert_eq!(isym_out, i64::from(isym), "raw ISYM round trip");
    assert_eq!(
        target_out,
        i32::from(target.index()),
        "ISYM must resolve to the 0-based XOR target the file encodes"
    );

    let mut orbsym_len = 0usize;
    assert_eq!(
        unsafe { qntt_f64_fcidump_orbsym(dump, ptr::null_mut(), 0, &mut orbsym_len) },
        QNTT_OK
    );
    let mut orbsym_out = vec![0u8; orbsym_len];
    assert_eq!(
        unsafe {
            qntt_f64_fcidump_orbsym(dump, orbsym_out.as_mut_ptr(), orbsym_len, &mut orbsym_len)
        },
        QNTT_OK
    );
    assert_eq!(
        orbsym_out, orbsym,
        "ORBSYM must survive the file round trip"
    );

    let mut t_len = 0usize;
    unsafe { qntt_f64_fcidump_t(dump, ptr::null_mut(), 0, &mut t_len) };
    let mut t_abi = vec![0.0f64; t_len];
    assert_eq!(
        unsafe { qntt_f64_fcidump_t(dump, t_abi.as_mut_ptr(), t_len, &mut t_len) },
        QNTT_OK
    );
    let mut v_len = 0usize;
    unsafe { qntt_f64_fcidump_v(dump, ptr::null_mut(), 0, &mut v_len) };
    let mut v_abi = vec![0.0f64; v_len];
    assert_eq!(
        unsafe { qntt_f64_fcidump_v(dump, v_abi.as_mut_ptr(), v_len, &mut v_len) },
        QNTT_OK
    );
    let tdiff = t_abi
        .iter()
        .zip(&t.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let vdiff = v_abi
        .iter()
        .zip(&v.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        tdiff < 1e-12 && vdiff < 1e-12,
        "integral round trip: {tdiff}/{vdiff}"
    );

    let mut h: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_ham_new_pg(
                d,
                nup,
                ndn,
                kind_out,
                orbsym_out.as_ptr(),
                orbsym_out.len(),
                target_out,
                t_abi.as_ptr(),
                t_abi.len(),
                v_abi.as_ptr(),
                v_abi.len(),
                ptr::null(),
                f64::EPSILON,
                2,
                &mut h,
            )
        },
        QNTT_OK,
        "ham_new_pg failed: {}",
        last_error()
    );

    let mut ranks_len = 0usize;
    assert_eq!(
        unsafe {
            qntt_space_max_ranks_pg(
                d,
                nup,
                ndn,
                kind_out,
                orbsym_out.as_ptr(),
                orbsym_out.len(),
                target_out,
                ptr::null_mut(),
                0,
                &mut ranks_len,
            )
        },
        QNTT_OK
    );
    let mut ranks = vec![0usize; ranks_len];
    assert_eq!(
        unsafe {
            qntt_space_max_ranks_pg(
                d,
                nup,
                ndn,
                kind_out,
                orbsym_out.as_ptr(),
                orbsym_out.len(),
                target_out,
                ranks.as_mut_ptr(),
                ranks_len,
                &mut ranks_len,
            )
        },
        QNTT_OK
    );
    for r in ranks.iter_mut() {
        *r = (*r).min(3);
    }
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn_pg(
                d,
                nup,
                ndn,
                kind_out,
                orbsym_out.as_ptr(),
                orbsym_out.len(),
                target_out,
                ranks.as_ptr(),
                ranks.len(),
                70_002,
                &mut x0,
            )
        },
        QNTT_OK,
        "vec_randn_pg failed: {}",
        last_error()
    );

    let mut opts = QnttMalsOptions::defaults();
    assert_eq!(unsafe { qntt_mals_options_default(&mut opts) }, QNTT_OK);
    opts.eps = 1e-10;
    opts.max_iter = 30;
    let bond_dims = [64usize];
    let (mut lambda, mut nsweeps) = (0.0f64, 0usize);
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_mals(
                h,
                x0,
                &opts,
                bond_dims.as_ptr(),
                bond_dims.len(),
                ptr::null(),
                0,
                &mut lambda,
                &mut x,
                ptr::null_mut(),
                0,
                &mut nsweeps,
            )
        },
        QNTT_OK,
        "mals failed: {}",
        last_error()
    );
    assert!(nsweeps > 0, "MALS reported no sweeps");

    let (mut nbits, mut tgt, mut irr_len) = (0i32, 0i32, 0usize);
    assert_eq!(
        unsafe {
            qntt_f64_vec_point_group(x, &mut nbits, &mut tgt, ptr::null_mut(), 0, &mut irr_len)
        },
        QNTT_OK
    );
    assert_eq!((nbits, tgt, irr_len), (2, target_out, d));

    unsafe {
        qntt_f64_vec_free(x);
        qntt_f64_vec_free(x0);
        qntt_f64_ham_free(h);
        qntt_f64_fcidump_free(dump);
    }
    (lambda, lambda_exact, kind_out)
}

#[test]
fn pg_fcidump_to_mals_through_the_abi_explicit_kind() {
    let (lambda, exact, kind) = c2v_fcidump_to_mals_through_abi("explicit", PG_C2V);
    assert_eq!(kind, PG_C2V, "the explicit kind must be reported back");
    assert!(
        (lambda - exact).abs() <= 1e-8 * exact.abs().max(1.0),
        "MALS through the ABI must reach the dense target-block ground state: got {lambda}, want \
         {exact}"
    );
}

#[test]
fn pg_fcidump_to_mals_through_the_abi_auto_kind() {
    let (lambda_auto, exact, kind) = c2v_fcidump_to_mals_through_abi("auto", PG_AUTO);
    assert_eq!(kind, PG_C2V, "AUTO on max(ORBSYM) = 4 must guess C2v");
    assert!(
        (lambda_auto - exact).abs() <= 1e-8 * exact.abs().max(1.0),
        "MALS under QNTT_PG_AUTO must reach the dense target-block ground state: got \
         {lambda_auto}, want {exact}"
    );
    let (lambda_explicit, _, _) = c2v_fcidump_to_mals_through_abi("auto_ref", PG_C2V);
    assert_eq!(
        lambda_auto, lambda_explicit,
        "the kind affects irrep NAMES only — AUTO and explicit C2v must agree exactly"
    );
}
