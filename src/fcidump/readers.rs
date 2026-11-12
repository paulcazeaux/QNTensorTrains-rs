// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rustc_hash::FxHashSet;
use std::path::Path;

use crate::coeff::{CoeffTensor, Sparse3Body};
use crate::scalar::Scalar;

use super::parser::{PERM3, read_entries, read_namelist_header};
use super::{FcidumpData, FcidumpError, ThreeBodySymmetry, TwoBodySymmetry};

/// Read a two-body FCIDUMP file: parse the optional `&FCI ... &end`/`/`-terminated namelist header
/// (or headerless, data from line 1), classify each 4-index entry by its RAW (pre-shift) file
/// indices — `i=j=k=l=0` ⇒ core energy, `k=l=0` ⇒ one-body `t[i-1,j-1]`, else two-body
/// `v[i-1,j-1,k-1,l-1]` — then complete the requested `symmetry` group (explicit entries always
/// win; symmetry only fills unset slots). FCIDUMP indices are 1-based throughout; every index
/// formula above drops one to land in this crate's 0-based convention.
pub fn read_fcidump<T: Scalar>(
    path: &Path,
    symmetry: TwoBodySymmetry,
) -> Result<FcidumpData<T>, FcidumpError> {
    parse_fcidump(&std::fs::read_to_string(path)?, symmetry)
}

/// [`read_fcidump`] on an in-memory dump — every parsing rule, index shift and symmetry-completion
/// image documented there applies verbatim; the only difference is that the text is already read.
pub fn parse_fcidump<T: Scalar>(
    content: &str,
    symmetry: TwoBodySymmetry,
) -> Result<FcidumpData<T>, FcidumpError> {
    let lines: Vec<&str> = content.lines().collect();
    let (params, datastart) = read_namelist_header(&lines);
    let entries = read_entries::<4>(&lines[datastart.min(lines.len())..]);

    let mut e0 = T::zero();
    let mut t1: Vec<(f64, usize, usize)> = Vec::new();
    let mut v2: Vec<(f64, [usize; 4])> = Vec::new();
    let mut norb = params.get("NORB").unwrap_or(0).max(0) as usize;
    for &(x, [i, j, k, l]) in &entries {
        let (i, j, k, l) = (i as usize, j as usize, k as usize, l as usize);
        if i == 0 && j == 0 && k == 0 && l == 0 {
            e0 = T::from_f64(x);
        } else if k == 0 && l == 0 {
            if i == 0 || j == 0 {
                return Err(FcidumpError::ArgumentError(format!(
                    "malformed one-body FCIDUMP entry (zero orbital index): {x}  {i}  {j}  0  0"
                )));
            }
            t1.push((x, i, j));
            norb = norb.max(i).max(j);
        } else {
            if i == 0 || j == 0 || k == 0 || l == 0 {
                return Err(FcidumpError::ArgumentError(format!(
                    "malformed two-body FCIDUMP entry (zero orbital index): {x}  {i}  {j}  {k}  {l}"
                )));
            }
            v2.push((x, [i, j, k, l]));
            norb = norb.max(i).max(j).max(k).max(l);
        }
    }

    let mut t = CoeffTensor::<T>::zeros(norb, 2);
    let mut tset: FxHashSet<[usize; 2]> = FxHashSet::default();
    for &(x, i, j) in &t1 {
        t.set(&[i - 1, j - 1], T::from_f64(x));
        tset.insert([i - 1, j - 1]);
    }
    if symmetry == TwoBodySymmetry::Eightfold {
        for &(x, i, j) in &t1 {
            let key = [j - 1, i - 1];
            if tset.insert(key) {
                t.set(&key, T::from_f64(x));
            }
        }
    }

    let mut v = CoeffTensor::<T>::zeros(norb, 4);
    let mut vset: FxHashSet<[usize; 4]> = FxHashSet::default();
    for &(x, idx) in &v2 {
        let key = [idx[0] - 1, idx[1] - 1, idx[2] - 1, idx[3] - 1];
        v.set(&key, T::from_f64(x));
        vset.insert(key);
    }
    if symmetry != TwoBodySymmetry::None {
        for &(x, [i, j, k, l]) in &v2 {
            let images: &[[usize; 4]] = match symmetry {
                TwoBodySymmetry::Eightfold => &[
                    [j, i, k, l],
                    [i, j, l, k],
                    [j, i, l, k],
                    [k, l, i, j],
                    [l, k, i, j],
                    [k, l, j, i],
                    [l, k, j, i],
                ],
                TwoBodySymmetry::Pairswap => &[[k, l, i, j]],
                TwoBodySymmetry::None => &[],
            };
            for im in images {
                let key = [im[0] - 1, im[1] - 1, im[2] - 1, im[3] - 1];
                if vset.insert(key) {
                    v.set(&key, T::from_f64(x));
                }
            }
        }
    }

    let nelec = params.get("NELEC").unwrap_or(0).max(0) as usize;
    let ms2 = params.get("MS2").unwrap_or(0);
    let orbsym = params
        .list("ORBSYM")
        .map(<[i64]>::to_vec)
        .unwrap_or_default();
    let isym = params.get("ISYM");
    Ok(FcidumpData {
        t,
        v,
        e0,
        norb,
        nelec,
        ms2,
        orbsym,
        isym,
    })
}

/// Read a three-body TCDUMP file: parse entries,
/// slot-map each physicists'-convention `⟨ijk|lmn⟩` sextuple to the package pair-slot order
/// `w[i,l,j,m,k,n]` (0-based, `-1` each of the six raw file indices), then complete the requested
/// `symmetry` group (`Full48` = 6 pair-perms × 8 per-pair transposes, `Pairperm` = 6 pair-perms
/// only). Returns [`Sparse3Body`] — see the `fcidump` module doc (NOT a dense `CoeffTensor`,
/// unrepresentable at FCIDUMP scale). `norb` seeds the orbital count
/// (grown by the max index seen, or the file's own `NORB=` header field); `d == 0` ⇒
/// `ArgumentError`.
pub fn read_tcdump<T: Scalar>(
    path: &Path,
    norb: usize,
    symmetry: ThreeBodySymmetry,
) -> Result<Sparse3Body<T>, FcidumpError> {
    let content = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();
    let (params, datastart) = read_namelist_header(&lines);
    let entries = read_entries::<6>(&lines[datastart.min(lines.len())..]);

    let norb_hdr = params.get("NORB").unwrap_or(0).max(0) as usize;
    let max_idx = entries
        .iter()
        .flat_map(|(_, idx)| idx.iter().copied())
        .max()
        .unwrap_or(0)
        .max(0) as usize;
    let d = norb.max(norb_hdr).max(max_idx);
    if d == 0 {
        return Err(FcidumpError::ArgumentError(
            "could not determine the orbital count; pass norb explicitly".to_string(),
        ));
    }
    if d > 126 {
        return Err(FcidumpError::ArgumentError(format!(
            "TCDUMP orbital count {d} exceeds the supported maximum of 126"
        )));
    }
    for &(_, idx) in &entries {
        if idx.iter().any(|&v| v < 1) {
            return Err(FcidumpError::ArgumentError(format!(
                "malformed TCDUMP entry (index < 1; the format has no zero sentinel): {idx:?}"
            )));
        }
    }

    let mut w = Sparse3Body::<T>::zeros(d);
    for &(x, idx) in &entries {
        let z = idx.map(|v| (v - 1) as usize);
        let (i, j, k, l, m, n) = (z[0], z[1], z[2], z[3], z[4], z[5]);
        w.set(&[i, l, j, m, k, n], T::from_f64(x));
    }
    if symmetry != ThreeBodySymmetry::None {
        let n_masks: u8 = if symmetry == ThreeBodySymmetry::Full48 {
            8
        } else {
            1
        };
        for &(x, idx) in &entries {
            let z = idx.map(|v| (v - 1) as usize);
            let (i, j, k, l, m, n) = (z[0], z[1], z[2], z[3], z[4], z[5]);
            let pairs = [[i, l], [j, m], [k, n]];
            for sigma in PERM3 {
                for mask in 0u8..n_masks {
                    let mut p = [[0usize; 2]; 3];
                    for (q, slot) in p.iter_mut().enumerate() {
                        let pr = pairs[sigma[q]];
                        *slot = if (mask >> q) & 1 == 1 {
                            [pr[1], pr[0]]
                        } else {
                            pr
                        };
                    }
                    let im = [p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1]];
                    w.set_if_absent(&im, T::from_f64(x));
                }
            }
        }
    }
    Ok(w)
}

#[cfg(test)]
mod phase7_tests {
    use crate::coeff::CoeffTensor;
    use crate::fcidump::{ThreeBodySymmetry, TwoBodySymmetry, read_fcidump, read_tcdump};
    use crate::scalar::Scalar;
    use rand::{RngExt, SeedableRng};
    use rand_distr::StandardNormal;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::io::Write as _;

    fn coeff_frob_diff<T: Scalar>(a: &CoeffTensor<T>, b: &CoeffTensor<T>) -> f64 {
        assert_eq!(
            a.data.len(),
            b.data.len(),
            "coeff_frob_diff: shape mismatch"
        );
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d.abs_s() * d.abs_s()
            })
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn t7b_fcidump_8fold_roundtrip() {
        fn images8(i: usize, j: usize, k: usize, l: usize) -> [(usize, usize, usize, usize); 8] {
            [
                (i, j, k, l),
                (j, i, k, l),
                (i, j, l, k),
                (j, i, l, k),
                (k, l, i, j),
                (l, k, i, j),
                (k, l, j, i),
                (l, k, j, i),
            ]
        }

        let d = 4usize;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7B10);

        let mut traw = vec![vec![0.0f64; d]; d];
        for row in traw.iter_mut() {
            for x in row.iter_mut() {
                *x = rng.sample(StandardNormal);
            }
        }
        let mut t0 = CoeffTensor::<f64>::zeros(d, 2);
        #[allow(clippy::needless_range_loop)]
        for i in 0..d {
            for j in 0..d {
                t0.set(&[i, j], 0.5 * (traw[i][j] + traw[j][i]));
            }
        }

        let mut v0 = CoeffTensor::<f64>::zeros(d, 4);
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    for l in 0..d {
                        let x: f64 = rng.sample(StandardNormal);
                        v0.set(&[i, j, k, l], x);
                    }
                }
            }
        }
        let mut vsym = CoeffTensor::<f64>::zeros(d, 4);
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    for l in 0..d {
                        let s: f64 = images8(i, j, k, l)
                            .iter()
                            .map(|&(a, b, c, e)| v0.get(&[a, b, c, e]))
                            .sum();
                        vsym.set(&[i, j, k, l], s / 8.0);
                    }
                }
            }
        }
        let e0ref: f64 = rng.sample(StandardNormal);

        let path = std::env::temp_dir().join("qntt_phase7_t7b_8fold.fcidump.tmp");
        {
            let mut f = std::fs::File::create(&path).expect("create temp fcidump");
            writeln!(f, " &FCI NORB=  {d} , NELEC=  2 , MS2=  0 ,").unwrap();
            writeln!(f, " ORBSYM={}", "1,".repeat(d)).unwrap();
            writeln!(f, " ISYM=0,").unwrap();
            writeln!(f, " &end").unwrap();
            for i in 0..d {
                for j in 0..d {
                    for k in 0..d {
                        for l in 0..d {
                            let minimum = images8(i, j, k, l).into_iter().min().unwrap();
                            if (i, j, k, l) == minimum {
                                writeln!(
                                    f,
                                    " {}  {}  {}  {}  {}",
                                    vsym.get(&[i, j, k, l]),
                                    i + 1,
                                    j + 1,
                                    k + 1,
                                    l + 1
                                )
                                .unwrap();
                            }
                        }
                    }
                }
            }
            for i in 0..d {
                for j in i..d {
                    writeln!(f, " {}  {}  {}  0  0", t0.get(&[i, j]), i + 1, j + 1).unwrap();
                }
            }
            writeln!(f, " {e0ref}  0  0  0  0").unwrap();
        }

        let data = read_fcidump::<f64>(&path, TwoBodySymmetry::Eightfold).expect("read_fcidump");
        let _ = std::fs::remove_file(&path);

        assert!(
            coeff_frob_diff(&data.t, &t0) < 1e-12,
            "t reconstruction mismatch"
        );
        assert!(
            coeff_frob_diff(&data.v, &vsym) < 1e-12,
            "v reconstruction mismatch"
        );
        assert!((data.e0 - e0ref).abs() < 1e-12, "e0 mismatch");
        assert_eq!(data.norb, d);
        assert_eq!(data.nelec, 2);
    }

    #[test]
    fn t7b_tcdump_zero_index_rejected() {
        let path = std::env::temp_dir().join("qntt_phase7_zero_idx.tcdump.tmp");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "1.0 0 1 2 1 2 1").unwrap();
        }
        let res = read_tcdump::<f64>(&path, 4, ThreeBodySymmetry::None);
        let _ = std::fs::remove_file(&path);
        assert!(res.is_err(), "zero-index TCDUMP line was not rejected");
    }

    #[test]
    fn t7b_tcdump_norb_over_cap_rejected() {
        let path = std::env::temp_dir().join("qntt_phase7_over_cap.tcdump.tmp");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "1.0 1 2 3 4 5 200").unwrap();
        }
        let res = read_tcdump::<f64>(&path, 4, ThreeBodySymmetry::None);
        let _ = std::fs::remove_file(&path);
        assert!(res.is_err(), "norb > 126 TCDUMP was not rejected");
    }

    #[test]
    fn t7b_fcidump_zero_index_rejected() {
        for (line, tag) in [
            ("1.0  0  5  0  0", "one-body"),
            ("1.0  1  0  2  3", "two-body"),
        ] {
            let path = std::env::temp_dir().join(format!("qntt_phase7_fci_zero_{tag}.fcidump.tmp"));
            {
                let mut f = std::fs::File::create(&path).unwrap();
                writeln!(f, " &FCI NORB=  5 , NELEC=  2 , MS2=  0 ,").unwrap();
                writeln!(f, " &end").unwrap();
                writeln!(f, "{line}").unwrap();
                writeln!(f, " 0.0  0  0  0  0").unwrap();
            }
            let res = read_fcidump::<f64>(&path, TwoBodySymmetry::None);
            let _ = std::fs::remove_file(&path);
            assert!(
                res.is_err(),
                "{tag} zero-index FCIDUMP line was not rejected"
            );
        }
    }

    #[test]
    fn t7b_explicit_entries_win_precedence() {
        let path = std::env::temp_dir().join("qntt_phase7_prec_v.fcidump.tmp");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, " &FCI NORB=  4 , NELEC=  2 , MS2=  0 ,").unwrap();
            writeln!(f, " &end").unwrap();
            writeln!(f, " 0.25  1  2  3  4").unwrap();
            writeln!(f, " 0.75  3  4  1  2").unwrap();
            writeln!(f, " 0.0  0  0  0  0").unwrap();
        }
        let data = read_fcidump::<f64>(&path, TwoBodySymmetry::Pairswap).expect("read_fcidump");
        let _ = std::fs::remove_file(&path);
        assert!(
            (data.v.get(&[0, 1, 2, 3]) - 0.25).abs() < 1e-15,
            "explicit v[1234] lost"
        );
        assert!(
            (data.v.get(&[2, 3, 0, 1]) - 0.75).abs() < 1e-15,
            "explicit v[3412]=0.75 overwritten by its pair-swap image (guard deleted?)"
        );

        let path = std::env::temp_dir().join("qntt_phase7_prec_t.fcidump.tmp");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, " &FCI NORB=  3 , NELEC=  2 , MS2=  0 ,").unwrap();
            writeln!(f, " &end").unwrap();
            writeln!(f, " 0.3  1  2  0  0").unwrap();
            writeln!(f, " 0.7  2  1  0  0").unwrap();
            writeln!(f, " 0.0  0  0  0  0").unwrap();
        }
        let data = read_fcidump::<f64>(&path, TwoBodySymmetry::Eightfold).expect("read_fcidump");
        let _ = std::fs::remove_file(&path);
        assert!(
            (data.t.get(&[0, 1]) - 0.3).abs() < 1e-15,
            "explicit t[12] lost"
        );
        assert!(
            (data.t.get(&[1, 0]) - 0.7).abs() < 1e-15,
            "explicit t[21]=0.7 overwritten by its transpose image (guard deleted?)"
        );

        let path = std::env::temp_dir().join("qntt_phase7_prec_tc.tcdump.tmp");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, " &TCDUMP NORB=  6 ,").unwrap();
            writeln!(f, " &end").unwrap();
            writeln!(f, " 1.0  1 2 3 4 5 6").unwrap();
            writeln!(f, " 2.0  4 2 3 1 5 6").unwrap();
        }
        let w = read_tcdump::<f64>(&path, 6, ThreeBodySymmetry::Full48).expect("read_tcdump");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            w.nnz(),
            48,
            "the two entries should share one 48-fold orbit"
        );
        assert!(
            (w.get(&[0, 3, 1, 4, 2, 5]) - 1.0).abs() < 1e-15,
            "explicit TCDUMP slot 1 overwritten by a symmetry image"
        );
        assert!(
            (w.get(&[3, 0, 1, 4, 2, 5]) - 2.0).abs() < 1e-15,
            "explicit TCDUMP slot 2 overwritten by a symmetry image"
        );
    }

    #[test]
    fn t7b_parser_branches() {
        let path = std::env::temp_dir().join("qntt_phase7_headerless.fcidump.tmp");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, " 1.0  1  1  0  0").unwrap();
            writeln!(f, " 0.0  0  0  0  0").unwrap();
        }
        let data = read_fcidump::<f64>(&path, TwoBodySymmetry::None).expect("read headerless");
        let _ = std::fs::remove_file(&path);
        assert_eq!(data.norb, 1);
        assert!((data.t.get(&[0, 0]) - 1.0).abs() < 1e-15);

        let path = std::env::temp_dir().join("qntt_phase7_slash.fcidump.tmp");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, " &FCI NORB=  2 , NELEC=  2 , MS2=  0 ,").unwrap();
            writeln!(f, " /").unwrap();
            writeln!(f, " 4.0  3  3  0  0").unwrap();
            writeln!(f, " 0.0  0  0  0  0").unwrap();
        }
        let data =
            read_fcidump::<f64>(&path, TwoBodySymmetry::None).expect("read slash-terminated");
        let _ = std::fs::remove_file(&path);
        assert_eq!(data.norb, 3, "norb should grow past the header NORB=2 to 3");
        assert!((data.t.get(&[2, 2]) - 4.0).abs() < 1e-15);

        let path = std::env::temp_dir().join("qntt_phase7_nosym.fcidump.tmp");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, " &FCI NORB=  4 , NELEC=  2 , MS2=  0 ,").unwrap();
            writeln!(f, " &end").unwrap();
            writeln!(f, " 0.5  1  2  3  4").unwrap();
            writeln!(f, " 0.0  0  0  0  0").unwrap();
        }
        let data = read_fcidump::<f64>(&path, TwoBodySymmetry::None).expect("read None-sym");
        let _ = std::fs::remove_file(&path);
        assert!((data.v.get(&[0, 1, 2, 3]) - 0.5).abs() < 1e-15);
        assert!(
            data.v.get(&[2, 3, 0, 1]).abs() < 1e-15,
            "TwoBodySymmetry::None must not fill the pair-swap image"
        );

        let path = std::env::temp_dir().join("qntt_phase7_pairperm.tcdump.tmp");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, " &TCDUMP NORB=  6 ,").unwrap();
            writeln!(f, " &end").unwrap();
            writeln!(f, " 1.0  1 2 3 4 5 6").unwrap();
        }
        let w = read_tcdump::<f64>(&path, 6, ThreeBodySymmetry::Pairperm).expect("read Pairperm");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            w.nnz(),
            6,
            "Pairperm should emit exactly 6 pair-permutation images"
        );
        assert!((w.get(&[1, 4, 0, 3, 2, 5]) - 1.0).abs() < 1e-15);
        assert!(
            w.get(&[3, 0, 1, 4, 2, 5]).abs() < 1e-15,
            "Pairperm must not emit per-pair transpose images"
        );
    }
}
