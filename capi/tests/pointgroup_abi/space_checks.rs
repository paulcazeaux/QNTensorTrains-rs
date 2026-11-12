// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt_capi::err::{QNTT_INVALID_ARG, QNTT_OK};

use crate::util::{c2v_orbsym, check_pg, last_error};
use crate::{PG_AUTO, PG_C2V};

#[test]
fn pg_space_rejects_its_six_failure_modes() {
    let d = 4usize;
    let good = c2v_orbsym(d);

    assert_eq!(check_pg(d, 2, 2, PG_C2V, &good, 2), QNTT_OK);

    assert_eq!(check_pg(d, 2, 2, 8, &good, 0), QNTT_INVALID_ARG);
    assert!(
        last_error().contains("point group kind 8"),
        "{}",
        last_error()
    );
    assert_eq!(check_pg(d, 2, 2, PG_AUTO, &good, 0), QNTT_INVALID_ARG);

    assert_eq!(check_pg(d, 5, 2, PG_C2V, &good, 0), QNTT_INVALID_ARG);
    assert!(last_error().contains("abelian space:"), "{}", last_error());

    let big = c2v_orbsym(128);
    assert_eq!(check_pg(128, 0, 0, PG_C2V, &big, 0), QNTT_INVALID_ARG);
    assert!(last_error().contains("127-site"), "{}", last_error());
    let ok127 = c2v_orbsym(127);
    assert_eq!(check_pg(127, 0, 0, PG_C2V, &ok127, 0), QNTT_OK);

    assert_eq!(check_pg(d, 2, 2, PG_C2V, &good[..3], 0), QNTT_INVALID_ARG);
    assert!(
        last_error().contains("orbsym has 3 entries"),
        "{}",
        last_error()
    );

    let mut bad = good.clone();
    bad[1] = 5;
    assert_eq!(check_pg(d, 2, 2, PG_C2V, &bad, 0), QNTT_INVALID_ARG);
    assert!(last_error().contains('5'), "{}", last_error());

    assert_eq!(check_pg(d, 2, 2, PG_C2V, &good, 4), QNTT_INVALID_ARG);
    assert!(last_error().contains("target irrep 4"), "{}", last_error());
    assert_eq!(check_pg(d, 2, 2, PG_C2V, &good, -1), QNTT_INVALID_ARG);

    assert_eq!(check_pg(d, 9, 9, PG_C2V, &bad, 7), QNTT_INVALID_ARG);
    assert!(
        last_error().starts_with("abelian space:"),
        "the (d, a, b) complaint must be reported first, got: {}",
        last_error()
    );
}
