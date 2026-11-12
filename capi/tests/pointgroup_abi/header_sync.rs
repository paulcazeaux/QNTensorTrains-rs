// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

const PG_SUFFIXES: [&str; 2] = ["f64", "c64"];

fn is_pg_symbol(name: &str) -> bool {
    name.starts_with("qntt_pg_") || name.ends_with("_pg") || name.ends_with("_point_group")
}

fn expand(name: &str) -> Vec<String> {
    if name.contains('@') {
        PG_SUFFIXES.iter().map(|s| name.replace('@', s)).collect()
    } else {
        vec![name.to_string()]
    }
}

fn ident_from(bytes: &[u8], start: usize) -> String {
    let mut end = start;
    while end < bytes.len()
        && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_' || bytes[end] == b'@')
    {
        end += 1;
    }
    String::from_utf8_lossy(&bytes[start..end]).into_owned()
}

#[test]
fn header_declares_every_point_group_symbol() {
    let header = include_str!("../../include/qntt.h").replace("##SFX##", "@");
    let sources = [
        include_str!("../../src/exports.rs"),
        include_str!("../../src/pointgroup.rs"),
        include_str!("../../src/pointgroup/space.rs"),
        include_str!("../../src/pointgroup/tables.rs"),
        include_str!("../../src/pointgroup/vector.rs"),
        include_str!("../../src/fcidump.rs"),
    ];

    let exports_src = sources[0];
    for (macro_name, src) in [
        ("export_abelian_pg!(", exports_src),
        ("export_fcidump!(", include_str!("../../src/fcidump.rs")),
    ] {
        let invocations: Vec<&str> = src
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with(macro_name))
            .collect();
        assert_eq!(
            invocations.len(),
            PG_SUFFIXES.len(),
            "{macro_name} instantiation set changed: {invocations:?} — update PG_SUFFIXES"
        );
        for (inv, sfx) in invocations.iter().zip(PG_SUFFIXES) {
            assert!(
                inv.starts_with(&format!("{macro_name}{sfx},")),
                "{macro_name} instantiated at an unexpected suffix: {inv}"
            );
        }
    }
    for (macro_name, src) in [
        ("QNTT_DECLARE_ABELIAN_PG(", &header),
        ("QNTT_DECLARE_SCALAR(", &header),
    ] {
        let invocations: Vec<&str> = src
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with(macro_name))
            .collect();
        assert_eq!(
            invocations.len(),
            PG_SUFFIXES.len(),
            "{macro_name} instantiation set changed: {invocations:?}"
        );
        for (inv, sfx) in invocations.iter().zip(PG_SUFFIXES) {
            assert!(
                inv.starts_with(&format!("{macro_name}{sfx},")),
                "{macro_name} instantiated at an unexpected suffix: {inv}"
            );
        }
    }

    let hb = header.as_bytes();
    let mut declared: Vec<String> = Vec::new();
    let mut i = 0usize;
    while let Some(rel) = header[i..].find("qntt_") {
        let at = i + rel;
        let name = ident_from(hb, at);
        i = at + name.len().max(1);
        if hb.get(at + name.len()) == Some(&b'(') && is_pg_symbol(&name) {
            declared.extend(expand(&name));
        }
    }
    declared.sort();
    declared.dedup();

    let mut exported: Vec<String> = Vec::new();
    for src in sources {
        let norm = src.replace("[<qntt_ $sfx _", "qntt_@_").replace(">](", "(");
        let mut no_mangle = false;
        for line in norm.lines() {
            let tl = line.trim();
            if tl.starts_with("#[") {
                no_mangle |= tl.contains("no_mangle");
                continue;
            }
            if let Some(pos) = tl.find("extern \"C\" fn ") {
                let start = pos + "extern \"C\" fn ".len();
                let name = ident_from(tl.as_bytes(), start);
                if no_mangle && is_pg_symbol(&name) {
                    exported.extend(expand(&name));
                }
                no_mangle = false;
                continue;
            }
            if !tl.is_empty() && !tl.starts_with("//") {
                no_mangle = false;
            }
        }
    }
    exported.sort();
    exported.dedup();

    assert!(
        declared.len() >= 30,
        "the header scan found only {} point-group declarations — the scan itself is broken",
        declared.len()
    );
    let missing_from_lib: Vec<&String> =
        declared.iter().filter(|n| !exported.contains(n)).collect();
    let missing_from_header: Vec<&String> =
        exported.iter().filter(|n| !declared.contains(n)).collect();
    assert!(
        missing_from_lib.is_empty(),
        "declared in qntt.h but not exported by the library: {missing_from_lib:?}"
    );
    assert!(
        missing_from_header.is_empty(),
        "exported by the library but not declared in qntt.h: {missing_from_header:?}"
    );
    assert_eq!(declared, exported);
}
