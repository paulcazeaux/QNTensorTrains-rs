// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn orbsym_list_captured_in_order() {
    let mut params = HeaderParams::default();
    let mut open: Option<String> = None;
    parse_header_params(" ORBSYM=1,2,3,4,", &mut params, &mut open);
    assert_eq!(params.list("ORBSYM"), Some(&[1i64, 2, 3, 4][..]));
    assert_eq!(params.get("ORBSYM"), Some(1));
}

#[test]
fn scalar_params_unchanged_by_the_run() {
    let lines = [
        " &FCI NORB=           9 , NELEC=           4 , MS2=           0 ,",
        " &FCI NORB=  24,NELEC=10,MS2=0,",
        " &FCI NORB=   4,NELEC= 4,MS2=0,",
        " &FCI NORB=  28,NELEC=14,MS2=0,",
    ];
    let want = [(9i64, 4i64, 0i64), (24, 10, 0), (4, 4, 0), (28, 14, 0)];
    for (line, &(norb, nelec, ms2)) in lines.iter().zip(&want) {
        let mut params = HeaderParams::default();
        let mut open: Option<String> = None;
        parse_header_params(line, &mut params, &mut open);
        assert_eq!(params.get("NORB"), Some(norb), "{line:?}");
        assert_eq!(params.get("NELEC"), Some(nelec), "{line:?}");
        assert_eq!(params.get("MS2"), Some(ms2), "{line:?}");
    }
}

#[test]
fn orbsym_continuation_across_lines() {
    let lines = [
        "&FCI NORB=  6 , NELEC=  6 , MS2=  0 ,",
        " ORBSYM=1,2,3,",
        " 1,2,3,",
        " ISYM=1,",
        "&END",
    ];
    let (params, datastart) = read_namelist_header(&lines);
    assert_eq!(datastart, lines.len());
    assert_eq!(params.list("ORBSYM"), Some(&[1i64, 2, 3, 1, 2, 3][..]));
    assert_eq!(params.get("ISYM"), Some(1));

    let lines2 = ["&FCI NORB=4,", " ORBSYM=1,2,", "ISYM=1,", "&END"];
    let (params2, _) = read_namelist_header(&lines2);
    assert_eq!(params2.list("ORBSYM"), Some(&[1i64, 2][..]));
    assert_eq!(params2.get("ISYM"), Some(1));
}

#[test]
fn isym_zero_absent_and_positive() {
    let (p0, _) = read_namelist_header(&["&FCI", "ISYM=0,", "&END"]);
    assert_eq!(p0.get("ISYM"), Some(0));

    let (pnone, _) = read_namelist_header(&["&FCI", "NORB=4,", "&END"]);
    assert_eq!(pnone.get("ISYM"), None);

    let (p3, _) = read_namelist_header(&["&FCI", "ISYM=3,", "&END"]);
    assert_eq!(p3.get("ISYM"), Some(3));
}

#[test]
fn header_params_first_occurrence_still_wins() {
    let (params, _) = read_namelist_header(&["&FCI NORB=9,", "NORB=99,", "&END"]);
    assert_eq!(params.get("NORB"), Some(9));
}

#[test]
fn headerless_and_slash_terminated_unchanged() {
    let (params, datastart) = read_namelist_header(&["1.0  1 1 1 1"]);
    assert_eq!(datastart, 0);
    assert_eq!(params.get("NORB"), None);

    let (params, datastart) = read_namelist_header(&["&FCI NORB=4,", "/"]);
    assert_eq!(datastart, 2);
    assert_eq!(params.get("NORB"), Some(4));
}

#[test]
fn unterminated_header_continuation_does_not_swallow_a_data_line() {
    let lines = [
        " &FCI NORB=  4 , NELEC=  2 , MS2=  0 ,",
        " ORBSYM=1,1,1,",
        " 2.5  1 1 0 0",
        " 0.0  0 0 0 0",
    ];
    let (params, _datastart) = read_namelist_header(&lines);
    assert_eq!(params.list("ORBSYM"), Some(&[1i64, 1, 1][..]), "{lines:?}");
}
