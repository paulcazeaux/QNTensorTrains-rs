// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::HashMap;

fn parse_fortran_float(s: &str) -> Option<f64> {
    let rewritten: String = s
        .chars()
        .map(|c| if c == 'd' || c == 'D' { 'E' } else { c })
        .collect();
    rewritten.parse::<f64>().ok()
}

#[derive(Default)]
pub(super) struct HeaderParams {
    scalars: HashMap<String, i64>,
    lists: HashMap<String, Vec<i64>>,
}

impl HeaderParams {
    pub(super) fn get(&self, key: &str) -> Option<i64> {
        self.scalars.get(key).copied()
    }

    pub(super) fn list(&self, key: &str) -> Option<&[i64]> {
        self.lists.get(key).map(|v| v.as_slice())
    }
}

fn parse_int_run(line: &str, start: usize, out: &mut Vec<i64>) -> Option<usize> {
    let bytes = line.as_bytes();
    let n = bytes.len();
    let mut end_last: Option<usize> = None;
    let mut j = start;
    loop {
        let mut p = j;
        if end_last.is_some() {
            while p < n && bytes[p].is_ascii_whitespace() {
                p += 1;
            }
            if p >= n || bytes[p] != b',' {
                break;
            }
            p += 1;
            while p < n && bytes[p].is_ascii_whitespace() {
                p += 1;
            }
        }
        let num_start = p;
        if p < n && bytes[p] == b'-' {
            p += 1;
        }
        let dig_start = p;
        while p < n && bytes[p].is_ascii_digit() {
            p += 1;
        }
        if p == dig_start {
            break;
        }
        let Ok(val) = line[num_start..p].parse::<i64>() else {
            break;
        };
        out.push(val);
        end_last = Some(p);
        j = p;
    }
    end_last
}

fn run_continues(line: &str, end: usize) -> bool {
    let bytes = line.as_bytes();
    let n = bytes.len();
    let mut p = end;
    while p < n && bytes[p].is_ascii_whitespace() {
        p += 1;
    }
    if p >= n || bytes[p] != b',' {
        return false;
    }
    p += 1;
    while p < n && bytes[p].is_ascii_whitespace() {
        p += 1;
    }
    p >= n
}

fn parse_header_params(line: &str, params: &mut HeaderParams, open: &mut Option<String>) {
    let bytes = line.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    if let Some(key) = open.take() {
        let mut p = 0;
        while p < n && bytes[p].is_ascii_whitespace() {
            p += 1;
        }
        if p < n && (bytes[p].is_ascii_digit() || bytes[p] == b'-') {
            let mut run: Vec<i64> = Vec::new();
            if let Some(end_last) = parse_int_run(line, p, &mut run) {
                let mut q = end_last;
                while q < n && bytes[q].is_ascii_whitespace() {
                    q += 1;
                }
                let terminates_cleanly = q >= n || bytes[q] == b',';
                if terminates_cleanly {
                    if let Some(prev) = params.lists.get_mut(&key) {
                        prev.extend_from_slice(&run);
                    }
                    i = end_last;
                    if run_continues(line, end_last) {
                        *open = Some(key);
                    }
                }
            }
        }
    }
    while i < n {
        if !bytes[i].is_ascii_alphabetic() {
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        while i < n && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
            i += 1;
        }
        let name = &line[start..i];
        let mut j = i;
        while j < n && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j >= n || bytes[j] != b'=' {
            continue;
        }
        j += 1;
        while j < n && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        let mut run: Vec<i64> = Vec::new();
        let Some(end_last) = parse_int_run(line, j, &mut run) else {
            continue;
        };
        let key = name.to_ascii_uppercase();
        let fresh = !params.scalars.contains_key(&key);
        if fresh {
            params.scalars.insert(key.clone(), run[0]);
            params.lists.insert(key.clone(), run);
        }
        i = end_last;
        *open = if fresh && run_continues(line, end_last) {
            Some(key)
        } else {
            None
        };
    }
}

fn is_header_terminator(line: &str) -> bool {
    line.to_ascii_lowercase().contains("&end") || line.trim() == "/"
}

pub(super) fn read_namelist_header(lines: &[&str]) -> (HeaderParams, usize) {
    let mut params = HeaderParams::default();
    let mut open: Option<String> = None;
    let mut datastart = 0usize;
    if lines
        .first()
        .is_some_and(|l| l.trim_start().starts_with('&'))
    {
        for (ni, line) in lines.iter().enumerate() {
            parse_header_params(line, &mut params, &mut open);
            if is_header_terminator(line) {
                datastart = ni + 1;
                break;
            }
        }
    }
    (params, datastart)
}

pub(crate) fn read_entries<const N: usize>(lines: &[&str]) -> Vec<(f64, [i64; N])> {
    let mut out = Vec::new();
    for line in lines {
        let tok: Vec<&str> = line.split_whitespace().collect();
        if tok.len() != N + 1 {
            continue;
        }
        let Some(val) = parse_fortran_float(tok[0]) else {
            continue;
        };
        let mut idx = [0i64; N];
        for (p, slot) in idx.iter_mut().enumerate() {
            *slot = tok[p + 1].parse::<i64>().unwrap_or(-1);
        }
        if idx.iter().any(|&x| x < 0) {
            continue;
        }
        out.push((val, idx));
    }
    out
}

pub(super) const PERM3: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

#[cfg(test)]
mod tests;

#[cfg(test)]
mod phase7_tests {
    use crate::fcidump::{TwoBodySymmetry, read_fcidump};
    use std::io::Write as _;

    #[test]
    fn t7b_fortran_d_exponent() {
        let path = std::env::temp_dir().join("qntt_phase7_dexp.fcidump.tmp");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, " &FCI NORB=  2 , NELEC=  2 , MS2=  0 ,").unwrap();
            writeln!(f, " &end").unwrap();
            writeln!(f, " 1.5D-3  1  1  0  0").unwrap();
            writeln!(f, " 2.5d+00  2  2  0  0").unwrap();
            writeln!(f, " 0.0  0  0  0  0").unwrap();
        }
        let data = read_fcidump::<f64>(&path, TwoBodySymmetry::None).expect("read_fcidump");
        let _ = std::fs::remove_file(&path);
        assert!(
            (data.t.get(&[0, 0]) - 1.5e-3).abs() < 1e-15,
            "D-exponent not parsed"
        );
        assert!(
            (data.t.get(&[1, 1]) - 2.5).abs() < 1e-15,
            "lowercase d-exponent not parsed"
        );
    }
}
