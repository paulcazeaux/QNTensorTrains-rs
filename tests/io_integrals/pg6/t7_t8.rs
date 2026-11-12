// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg6::util::*;
use qntt::PointGroupKind;

#[test]
fn pg6_n2_dump_exercises_the_symmetry_noise_screen() {
    const SYMMETRY_FORBIDDEN_TOL: f64 = 1e-8;
    let meta = n2_meta();
    let (mut data, pgdata) = load_n2(None);

    assert_eq!(pgdata.kind, PointGroupKind::D2h);
    assert!(pgdata.guessed);
    assert_eq!(data.isym, Some(1));
    assert_eq!(data.orbsym, meta_vec_i64(&meta, "orbsym_molpro"));

    let (n1, max1, n2, max2) = screen_forbidden(&mut data.t, &mut data.v, pgdata.pg);
    assert_eq!(
        n1,
        meta_usize(&meta, "forbidden_1body"),
        "one-body forbidden-entry count must match the meta's file-level census exactly"
    );
    assert_eq!(
        n2,
        meta_usize(&meta, "forbidden_2body"),
        "two-body forbidden-entry count must match the meta's file-level census exactly"
    );

    let observed_max = max1.max(max2);
    let want_max = meta_f64(&meta, "forbidden_max");
    assert!(
        (observed_max - want_max).abs() < 1e-16,
        "combined forbidden max {observed_max} vs meta's {want_max}"
    );

    assert!(
        max1 < SYMMETRY_FORBIDDEN_TOL,
        "one-body max {max1} unexpectedly exceeds the {SYMMETRY_FORBIDDEN_TOL:e} noise floor"
    );
    assert!(
        max2 < SYMMETRY_FORBIDDEN_TOL,
        "two-body max {max2} unexpectedly exceeds the {SYMMETRY_FORBIDDEN_TOL:e} noise floor"
    );
}
