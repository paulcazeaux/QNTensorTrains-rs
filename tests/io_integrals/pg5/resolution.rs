// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg5::util::*;
use qntt::fcidump::{PointGroupData, TwoBodySymmetry, read_fcidump, read_fcidump_with_point_group};
use qntt::{
    IntegralsOptions, Irrep, PointGroupKind, PointGroupPolicy, QnSpace, SparseHamiltonian,
    TtVector, rayleigh_quotient, tt_state,
};

#[test]
fn guess_per_order() {
    let cases: &[(&[i64], PointGroupKind)] = &[
        (&[1], PointGroupKind::C1),
        (&[1, 2], PointGroupKind::Cs),
        (&[1, 2, 3], PointGroupKind::C2v),
        (&[1, 2, 3, 4], PointGroupKind::C2v),
        (&[1, 5], PointGroupKind::D2h),
        (&[1, 8], PointGroupKind::D2h),
    ];
    for &(orbsym, want) in cases {
        let data = fcidump_stub(orbsym.len(), 0, 0, orbsym.to_vec(), None);
        let pg = data
            .point_group(None)
            .unwrap_or_else(|e| panic!("orbsym {orbsym:?}: {e}"));
        assert_eq!(pg.kind, want, "orbsym {orbsym:?}");
        assert!(pg.guessed, "orbsym {orbsym:?}: guessed flag must be set");
    }

    for bad in [vec![9i64], vec![1, 9]] {
        let data = fcidump_stub(bad.len(), 0, 0, bad.clone(), None);
        assert!(
            data.point_group(None).is_err(),
            "orbsym {bad:?} should be rejected"
        );
    }

    assert_eq!(PointGroupKind::default_for_order(1), PointGroupKind::C1);
    assert_eq!(PointGroupKind::default_for_order(2), PointGroupKind::Cs);
    assert_eq!(PointGroupKind::default_for_order(4), PointGroupKind::C2v);
    assert_eq!(PointGroupKind::default_for_order(8), PointGroupKind::D2h);
}

#[test]
#[should_panic(expected = "point-group order must be 1, 2, 4 or 8")]
fn default_for_order_refuses_unsupported_order() {
    let _ = PointGroupKind::default_for_order(3);
}

#[test]
fn explicit_kind_overrides_guess() {
    let data = fcidump_stub(4, 0, 0, vec![1, 2, 3, 4], None);
    let guess = data.point_group(None).unwrap();
    assert_eq!(guess.kind, PointGroupKind::C2v);
    assert!(guess.guessed);

    let explicit = data.point_group(Some(PointGroupKind::D2)).unwrap();
    assert_eq!(explicit.kind, PointGroupKind::D2);
    assert!(!explicit.guessed);

    assert_ne!(
        explicit.pg.planes(),
        guess.pg.planes(),
        "D2 and C2v must assign different XOR labels to the same Molpro ids"
    );
}

#[test]
fn explicit_kind_too_small_rejected() {
    let data = fcidump_stub(4, 0, 0, vec![1, 2, 3, 4], None);
    let err = data
        .point_group(Some(PointGroupKind::Ci))
        .expect_err("id 3 exceeds Ci's order 2");
    let msg = err.to_string();
    assert!(msg.contains("Ci"), "got {msg:?}");
    assert!(msg.contains("3"), "got {msg:?}");
    assert!(msg.contains("Some(2)"), "got {msg:?}");
}

#[test]
fn orbsym_length_mismatch_rejected() {
    let data = fcidump_stub(4, 0, 0, vec![1, 2, 3], None);
    let err = data
        .point_group(None)
        .expect_err("length mismatch must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("ORBSYM"), "got {msg:?}");
    assert!(msg.contains('3') && msg.contains('4'), "got {msg:?}");
}

#[test]
fn orbsym_zero_id_rejected() {
    let data = fcidump_stub(4, 0, 0, vec![0, 1, 1, 1], None);
    let err = data
        .point_group(None)
        .expect_err("id 0 is not a 1-based Molpro id");
    assert!(err.to_string().contains("ORBSYM"), "got {}", err);
}

#[test]
fn orbsym_id_over_8_rejected() {
    let data = fcidump_stub(4, 0, 0, vec![9, 1, 1, 1], None);
    let err = data
        .point_group(None)
        .expect_err("id > 8 exceeds every abelian group's order");
    assert!(err.to_string().contains('8'), "got {}", err);
}

#[test]
fn isym_out_of_range_rejected() {
    let data = fcidump_stub(4, 0, 0, vec![1, 2, 3, 4], Some(9));
    let err = data
        .point_group(None)
        .expect_err("ISYM=9 exceeds C2v's order 4");
    let msg = err.to_string();
    assert!(msg.contains("ISYM"), "got {msg:?}");
    assert!(msg.contains('9'), "got {msg:?}");
}

#[test]
fn norb_over_127_with_nontrivial_group_rejected() {
    let d = 200usize;
    let orbsym: Vec<i64> = (0..d).map(|k| [2i64, 3, 4, 1][k % 4]).collect();
    let data = fcidump_stub(d, 0, 0, orbsym, None);
    let err = data
        .point_group(None)
        .expect_err("d>127 with a nontrivial group must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("127") || msg.contains("128"), "got {msg:?}");
}

#[test]
fn norb_over_128_with_trivial_group_rejected_cleanly() {
    let data = fcidump_stub(129, 4, 0, vec![], None);
    let err = data
        .point_group(None)
        .expect_err("an ORBSYM-free dump at norb=129 must be rejected, not silently resolve");
    let msg = err.to_string();
    assert!(msg.contains("128"), "got {msg:?}");
    let err2 = data
        .qn_space(None)
        .expect_err("qn_space must also reject norb=129 cleanly");
    assert!(err2.to_string().contains("128"));
}

#[test]
fn norb_128_with_trivial_group_still_resolves() {
    let data = fcidump_stub(128, 4, 0, vec![], None);
    let pg = data
        .point_group(None)
        .unwrap_or_else(|e| panic!("norb=128 must still resolve: {e}"));
    assert_eq!(pg, PointGroupData::TRIVIAL);
    data.qn_space(None)
        .unwrap_or_else(|e| panic!("norb=128 qn_space must still succeed: {e}"));
}

#[test]
fn explicit_c1_over_nontrivial_orbsym_rejected() {
    let data = fcidump_stub(4, 0, 0, vec![1, 2, 3, 4], None);
    let err = data
        .point_group(Some(PointGroupKind::C1))
        .expect_err("explicit C1 over a non-all-ones ORBSYM must be rejected");
    let msg = err.to_string();
    assert!(msg.contains('2'), "got {msg:?}");
}

#[test]
fn qn_space_rejects_inconsistent_header_instead_of_panicking() {
    let bad_parity = fcidump_stub(4, 3, 0, vec![], None);
    let err = bad_parity
        .qn_space(None)
        .expect_err("odd NELEC with even MS2 must be rejected, not panic");
    assert!(err.to_string().contains("parity"), "got {err}");

    let bad_ms2 = fcidump_stub(4, 4, 5, vec![], None);
    let err = bad_ms2
        .qn_space(None)
        .expect_err("|MS2| > NELEC must be rejected, not panic");
    assert!(err.to_string().contains("MS2"), "got {err}");
}

#[test]
fn su2_hamiltonian_rejects_inconsistent_header_instead_of_accepting_it() {
    let mut bad_parity = fcidump_stub(4, 4, 1, vec![], None);
    let err = match bad_parity.su2_hamiltonian(Some(2), &IntegralsOptions::default()) {
        Ok(_) => panic!("mismatched NELEC/MS2 parity must be rejected, not silently accepted"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("parity"), "got {err}");
}

#[test]
fn c1_fixtures_resolve_trivial() {
    for path in all_c1_fixtures() {
        let data = read_fcidump::<f64>(&path, TwoBodySymmetry::Eightfold)
            .unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let pg = data
            .point_group(None)
            .unwrap_or_else(|e| panic!("{path:?}: {e}"));
        assert_eq!(pg, PointGroupData::TRIVIAL, "{path:?}");
    }
}

#[test]
fn pg_aware_load_equals_plain_load() {
    for path in all_c1_fixtures() {
        let (with_pg, _pgdata) =
            read_fcidump_with_point_group::<f64>(&path, TwoBodySymmetry::Eightfold, None)
                .unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let plain = read_fcidump::<f64>(&path, TwoBodySymmetry::Eightfold).unwrap();
        assert_eq!(
            with_pg.t.data, plain.t.data,
            "{path:?}: t must be bit-identical"
        );
        assert_eq!(
            with_pg.v.data, plain.v.data,
            "{path:?}: v must be bit-identical"
        );
        assert_eq!(with_pg.e0, plain.e0, "{path:?}");
        assert_eq!(
            (with_pg.norb, with_pg.nelec, with_pg.ms2),
            (plain.norb, plain.nelec, plain.ms2),
            "{path:?}"
        );
    }
}

#[test]
fn pg_aware_space_equals_from_n_two_sz() {
    for path in all_c1_fixtures() {
        let data = read_fcidump::<f64>(&path, TwoBodySymmetry::Eightfold).unwrap();
        let space = data
            .qn_space(None)
            .unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let want = QnSpace::from_n_two_sz(data.norb, data.nelec, data.ms2);
        assert_eq!(space, want, "{path:?}");
    }
}

#[test]
fn pg_aware_hamiltonian_matches() {
    let path = be_path();
    let plain_opts = IntegralsOptions {
        symmetry: TwoBodySymmetry::Pairswap,
        ..IntegralsOptions::default()
    };
    let pg_opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..plain_opts
    };
    let (h_pg, load_pg) = SparseHamiltonian::<f64>::from_fcidump(&path, None, &pg_opts)
        .expect("Be with the point-group-aware path");
    let (h_plain, load_plain) =
        SparseHamiltonian::<f64>::from_fcidump(&path, None, &plain_opts).expect("Be plain path");
    let (data_pg, data_plain, pgdata) = (&load_pg.ints, &load_plain.ints, load_pg.pg);

    assert_eq!(pgdata, PointGroupData::TRIVIAL);
    assert_eq!(data_pg.t.data, data_plain.t.data);
    assert_eq!(data_pg.v.data, data_plain.v.data);

    let mut occ = vec![false; 9];
    occ[0] = true;
    occ[1] = true;
    let x: TtVector<f64> = tt_state(&occ, &occ);
    assert!(x.check());

    let rq_pg = rayleigh_quotient(&h_pg, &x);
    let rq_plain = rayleigh_quotient(&h_plain, &x);
    assert_eq!(
        rq_pg, rq_plain,
        "the point-group-aware and plain paths must be bit-identical at |G|=1"
    );
    assert!(
        (rq_pg - (-14.577233846433735)).abs() < 1e-9,
        "molecule gate re-pinned through the point-group-aware path: rq={rq_pg}"
    );
}

#[test]
#[should_panic(expected = "invalid point-group table:")]
fn point_group_new_refuses_out_of_range_irrep() {
    use qntt::PointGroup;
    let _ = PointGroup::new(1, &[Irrep(0), Irrep(2)]);
}
