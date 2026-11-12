// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::*;

#[test]
fn a7_fcidump_screen_pg_is_ok_idempotent_and_a_noop_without_orbsym() {
    let h2o_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/h2o_c2v/H2O.FCIDUMP");
    let path_c = std::ffi::CString::new(h2o_path.to_str().unwrap()).unwrap();
    let mut dump: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_fcidump_read_pg(path_c.as_ptr(), 2, PG_AUTO, &mut dump) },
        QNTT_OK
    );
    assert_eq!(
        unsafe { qntt_f64_fcidump_screen_pg(dump, PG_AUTO) },
        QNTT_OK
    );

    let mut len1 = 0usize;
    unsafe { qntt_f64_fcidump_t(dump, ptr::null_mut(), 0, &mut len1) };
    let mut t1 = vec![0.0f64; len1];
    assert_eq!(
        unsafe { qntt_f64_fcidump_t(dump, t1.as_mut_ptr(), t1.len(), &mut len1) },
        QNTT_OK
    );
    assert_eq!(
        unsafe { qntt_f64_fcidump_screen_pg(dump, PG_AUTO) },
        QNTT_OK
    );
    let mut len2 = 0usize;
    let mut t2 = vec![0.0f64; len1];
    assert_eq!(
        unsafe { qntt_f64_fcidump_t(dump, t2.as_mut_ptr(), t2.len(), &mut len2) },
        QNTT_OK
    );
    assert_eq!(t1, t2, "a second screen must be a byte-identical no-op");
    unsafe { qntt_f64_fcidump_free(dump) };

    let h4_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/h4.FCIDUMP");
    let path_c2 = std::ffi::CString::new(h4_path.to_str().unwrap()).unwrap();
    let mut dump2: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_fcidump_read_pg(path_c2.as_ptr(), 2, PG_AUTO, &mut dump2) },
        QNTT_OK
    );
    let mut lb = 0usize;
    unsafe { qntt_f64_fcidump_t(dump2, ptr::null_mut(), 0, &mut lb) };
    let mut before = vec![0.0f64; lb];
    assert_eq!(
        unsafe { qntt_f64_fcidump_t(dump2, before.as_mut_ptr(), before.len(), &mut lb) },
        QNTT_OK
    );
    assert_eq!(
        unsafe { qntt_f64_fcidump_screen_pg(dump2, PG_AUTO) },
        QNTT_OK
    );
    let mut la = 0usize;
    let mut after = vec![0.0f64; lb];
    assert_eq!(
        unsafe { qntt_f64_fcidump_t(dump2, after.as_mut_ptr(), after.len(), &mut la) },
        QNTT_OK
    );
    assert_eq!(before, after, "ORBSYM-free dump: screening must be a no-op");
    unsafe { qntt_f64_fcidump_free(dump2) };

    let n2_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/n2_d2h/N2.FCIDUMP");
    if n2_path.is_file() {
        let path_c3 = std::ffi::CString::new(n2_path.to_str().unwrap()).unwrap();
        let mut dump3: *mut _ = ptr::null_mut();
        assert_eq!(
            unsafe { qntt_f64_fcidump_read_pg(path_c3.as_ptr(), 2, PG_AUTO, &mut dump3) },
            QNTT_OK
        );
        let mut lt = 0usize;
        unsafe { qntt_f64_fcidump_t(dump3, ptr::null_mut(), 0, &mut lt) };
        let mut t_before = vec![0.0f64; lt];
        assert_eq!(
            unsafe { qntt_f64_fcidump_t(dump3, t_before.as_mut_ptr(), t_before.len(), &mut lt) },
            QNTT_OK
        );
        let mut lv = 0usize;
        unsafe { qntt_f64_fcidump_v(dump3, ptr::null_mut(), 0, &mut lv) };
        let mut v_before = vec![0.0f64; lv];
        assert_eq!(
            unsafe { qntt_f64_fcidump_v(dump3, v_before.as_mut_ptr(), v_before.len(), &mut lv) },
            QNTT_OK
        );

        assert_eq!(
            unsafe { qntt_f64_fcidump_screen_pg(dump3, PG_AUTO) },
            QNTT_OK
        );

        let mut t_after = vec![0.0f64; t_before.len()];
        let mut lt2 = 0usize;
        assert_eq!(
            unsafe { qntt_f64_fcidump_t(dump3, t_after.as_mut_ptr(), t_after.len(), &mut lt2) },
            QNTT_OK
        );
        let mut v_after = vec![0.0f64; v_before.len()];
        let mut lv2 = 0usize;
        assert_eq!(
            unsafe { qntt_f64_fcidump_v(dump3, v_after.as_mut_ptr(), v_after.len(), &mut lv2) },
            QNTT_OK
        );

        assert_ne!(
            t_before, t_after,
            "n2_d2h carries forbidden one-body entries per its own census — screening must \
             actually change the t buffer"
        );
        assert_ne!(
            v_before, v_after,
            "n2_d2h carries forbidden two-body entries per its own census — screening must \
             actually change the v buffer"
        );
        unsafe { qntt_f64_fcidump_free(dump3) };
    }
}

#[test]
fn a8_bond_sectors_pg_matches_rust_and_allows_partial_null_arrays() {
    let d = 4usize;
    let (n, two_s) = (3usize, 1usize);
    let target = Irrep::new(0);
    let orbsym = c2v_orbsym(d);
    let pg = c2v_pg(d);
    let bond = 2usize;

    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let mut want = space.bond_sectors(bond);
    want.sort_by_key(|s| (s.n, s.two_s, s.irrep.index()));
    assert!(!want.is_empty(), "fixture bug: bond 2 must have sectors");
    assert!(
        want.iter().any(|s| s.n != s.two_s),
        "fixture bug: need at least one sector with n != 2s to discriminate an axis swap"
    );

    let mut len = 0usize;
    assert_eq!(
        unsafe {
            qntt_su2_space_bond_sectors_pg(
                d,
                n,
                two_s,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(target.index()),
                bond,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut len,
            )
        },
        QNTT_OK
    );
    assert_eq!(len, want.len());

    let mut out_n = vec![0usize; len];
    let mut out_s = vec![0usize; len];
    let mut out_ir = vec![0u8; len];
    let mut len2 = 0usize;
    assert_eq!(
        unsafe {
            qntt_su2_space_bond_sectors_pg(
                d,
                n,
                two_s,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(target.index()),
                bond,
                out_n.as_mut_ptr(),
                out_s.as_mut_ptr(),
                out_ir.as_mut_ptr(),
                len,
                &mut len2,
            )
        },
        QNTT_OK
    );
    assert_eq!(len2, len);

    let mut got: Vec<(usize, usize, u8)> =
        (0..len).map(|k| (out_n[k], out_s[k], out_ir[k])).collect();
    got.sort();
    let want_tuples: Vec<(usize, usize, u8)> = want
        .iter()
        .map(|s| (s.n, s.two_s, s.irrep.index()))
        .collect();
    assert_eq!(
        got, want_tuples,
        "qntt_su2_space_bond_sectors_pg must match Su2Space::bond_sectors exactly (array \
         identity, not just array order — a swapped n/2s source would still sum to the same \
         multiset in a badly-chosen fixture, so this compares tuples)"
    );

    let mut a = vec![0usize; len];
    let mut b = vec![0usize; len];
    let mut len3 = 0usize;
    assert_eq!(
        unsafe {
            qntt_su2_space_bond_sectors_pg(
                d,
                n,
                two_s,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(target.index()),
                bond,
                a.as_mut_ptr(),
                b.as_mut_ptr(),
                ptr::null_mut(),
                len,
                &mut len3,
            )
        },
        QNTT_OK,
        "a null irrep array with the others sized must succeed"
    );
    assert_eq!(len3, len);
    assert_eq!(a, out_n);
    assert_eq!(b, out_s);

    let mut irreps = vec![0u8; len];
    let mut len4 = 0usize;
    assert_eq!(
        unsafe {
            qntt_su2_space_bond_sectors_pg(
                d,
                n,
                two_s,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(target.index()),
                bond,
                ptr::null_mut(),
                ptr::null_mut(),
                irreps.as_mut_ptr(),
                len,
                &mut len4,
            )
        },
        QNTT_OK,
        "a null n/2s pair with irrep sized must succeed"
    );
    assert_eq!(len4, len);
    assert_eq!(irreps, out_ir);
}

const SU2_PG_FN_NAMES: [&str; 10] = [
    "vec_zeros_pg",
    "vec_from_ranks_pg",
    "vec_randn_pg",
    "tt_csf_pg",
    "vec_ranks_pg",
    "vec_block_get_pg",
    "vec_block_set_pg",
    "vec_point_group",
    "ham_new_pg",
    "ham_point_group",
];
const SU2_PG_SCALARS: [(&str, &str); 2] = [("su2_f64", "double"), ("su2_c64", "qntt_complex")];

fn slice_between<'a>(text: &'a str, start_marker: &str, end_marker: &str) -> &'a str {
    let start = text
        .find(start_marker)
        .unwrap_or_else(|| panic!("start marker not found (source moved?): {start_marker:?}"))
        + start_marker.len();
    let end = text[start..].find(end_marker).unwrap_or_else(|| {
        panic!("end marker not found after start (source moved?): {end_marker:?}")
    }) + start;
    &text[start..end]
}

#[test]
fn a9_su2_pg_macro_symbols_are_literally_declared_and_exported() {
    let header = include_str!("../../include/qntt.h");
    let exports_src = include_str!("../../src/exports.rs");

    for (sfx, _) in SU2_PG_SCALARS {
        assert_eq!(
            exports_src
                .lines()
                .filter(|l| l.trim()
                    == format!(
                        "export_su2_pg!({sfx}, {});",
                        if sfx == "su2_f64" { "f64" } else { "C64" }
                    ))
                .count(),
            1,
            "export_su2_pg! must be instantiated at exactly {sfx} (once)"
        );
    }
    for (sfx, ctype) in SU2_PG_SCALARS {
        assert_eq!(
            header
                .lines()
                .filter(|l| l.trim() == format!("QNTT_DECLARE_SU2_PG({sfx}, {ctype})"))
                .count(),
            1,
            "QNTT_DECLARE_SU2_PG must be instantiated at exactly ({sfx}, {ctype}) (once)"
        );
    }

    let header_body = slice_between(
        header,
        "#define QNTT_DECLARE_SU2_PG(SFX, T)",
        "QNTT_DECLARE_SU2_PG(su2_f64, double)",
    );
    let exports_body = slice_between(
        exports_src,
        "macro_rules! export_su2_pg {",
        "export_su2_pg!(su2_f64, f64);",
    );

    for fn_name in SU2_PG_FN_NAMES {
        let header_pat = format!("qntt_##SFX##_{fn_name}(");
        assert!(
            header_body.contains(&header_pat),
            "QNTT_DECLARE_SU2_PG body is missing `{header_pat}` — a header declaration for \
             qntt_su2_{{f64,c64}}_{fn_name} was renamed or deleted"
        );

        let sig_needle = format!("pub unsafe extern \"C\" fn [<qntt_ $sfx _{fn_name}>](");
        let sig_pos = exports_body.find(&sig_needle).unwrap_or_else(|| {
            panic!(
                "export_su2_pg! body is missing `{sig_needle}` — qntt_su2_{{f64,c64}}_{fn_name}_pg \
                 was renamed, deleted, or its signature reshaped"
            )
        });
        let line_start = exports_body[..sig_pos].rfind('\n').map_or(0, |p| p + 1);
        let prev_line = exports_body[..line_start]
            .lines()
            .next_back()
            .unwrap_or("")
            .trim();
        assert_eq!(
            prev_line, "#[unsafe(no_mangle)]",
            "qntt_su2_{{f64,c64}}_{fn_name} must be preceded by #[unsafe(no_mangle)] — found {prev_line:?}"
        );
    }

    for (sfx, _) in SU2_PG_SCALARS {
        for fn_name in SU2_PG_FN_NAMES {
            let want = format!("qntt_{sfx}_{fn_name}");
            let declared = header_body
                .replace("##SFX##", sfx)
                .contains(&format!("{want}("));
            let exported = exports_body
                .replace("[<qntt_ $sfx _", &format!("qntt_{sfx}_"))
                .replace(">](", "(")
                .contains(&format!("{want}("));
            assert!(
                declared && exported,
                "{want} must be both declared in qntt.h and exported by the library \
                 (declared={declared}, exported={exported})"
            );
        }
    }
}
