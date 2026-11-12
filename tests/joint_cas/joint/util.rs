// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::{TtTangent, projective_transport};
use qntt::integrals::Integrals;
use qntt::rotation::{expm_antisym, orbital_gradient, rotate_integrals};
use qntt::solvers::riemannian::{normalize_mut, tangent_bases};
use qntt::{CoeffTensor, QnSpace, SparseHamiltonian, TtVector, dot, rayleigh_quotient, tt_state};

pub(crate) use qntt_testkit::dense::*;

pub(crate) fn dense_gpq(
    d: usize,
    p: usize,
    q: usize,
    dets: &[(Vec<bool>, Vec<bool>)],
) -> Vec<Vec<f64>> {
    let epq = dense_epq_matrix(d, p, q, dets);
    let eqp = dense_epq_matrix(d, q, p, dets);
    mat_add(&epq, &mat_scale(&eqp, -1.0))
}

pub(crate) fn dense_state_vector(dets: &[(Vec<bool>, Vec<bool>)], x: &TtVector<f64>) -> Vec<f64> {
    dets.iter()
        .map(|(ou, od)| {
            let s: TtVector<f64> = tt_state(ou, od);
            dot(x, &s)
        })
        .collect()
}

pub(crate) fn joint_energy_dense(
    hd: &[Vec<f64>],
    c: &[f64],
    kd: &[Vec<f64>],
    zeta: &[f64],
    s: f64,
) -> f64 {
    let expo = mat_expm(&mat_scale(kd, -s));
    let disp: Vec<f64> = c.iter().zip(zeta).map(|(&ci, &zi)| ci + s * zi).collect();
    let psi = mat_vec(&expo, &disp);
    let hpsi = mat_vec(hd, &psi);
    dot_vec(&psi, &hpsi) / dot_vec(&psi, &psi)
}

pub(crate) fn q2_dense(
    hd: &[Vec<f64>],
    c: &[f64],
    kd: &[Vec<f64>],
    zeta: &[f64],
    e0: f64,
    h: f64,
) -> f64 {
    let ep2 = joint_energy_dense(hd, c, kd, zeta, 2.0 * h);
    let ep1 = joint_energy_dense(hd, c, kd, zeta, h);
    let em1 = joint_energy_dense(hd, c, kd, zeta, -h);
    let em2 = joint_energy_dense(hd, c, kd, zeta, -2.0 * h);
    (-ep2 + 16.0 * ep1 - 30.0 * e0 + 16.0 * em1 - em2) / (24.0 * h * h)
}

pub(crate) fn coeff2_flat(c: &CoeffTensor<f64>, d: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            out[i * d + j] = c.get(&[i, j]);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn joint_covector_fd_central(
    t_cur: &CoeffTensor<f64>,
    v_cur: &CoeffTensor<f64>,
    space: QnSpace,
    pairs: &[(usize, usize)],
    a_dir: &[f64],
    x_base: &TtVector<f64>,
    xi_dir: &TtTangent<f64>,
    h: f64,
) -> (TtTangent<f64>, Vec<f64>) {
    let d = space.d;
    let mut k = CoeffTensor::<f64>::zeros(d, 2);
    for (&(p, q), &ai) in pairs.iter().zip(a_dir) {
        k.set(&[p, q], ai);
        k.set(&[q, p], -ai);
    }
    let kflat = coeff2_flat(&k, d);

    let one = |s: f64| -> (TtTangent<f64>, Vec<f64>) {
        let step = scale_flat(&kflat, -s);
        let u = expm_antisym(&step, d);
        let ints_s = Integrals::<f64> {
            t: t_cur.clone(),
            v: v_cur.clone(),
            e0: 0.0,
            norb: d,
            nelec: space.n(),
            ms2: space.nup as i64 - space.ndn as i64,
            orbsym: Vec::new(),
            isym: None,
        };
        let rotated = rotate_integrals(&ints_s, &u);
        let hs = SparseHamiltonian::new(space, &rotated.t, &rotated.v, None, f64::EPSILON);
        let mut xs = xi_dir.retract_hosvd(s);
        normalize_mut(&mut xs);
        let lam_s = rayleigh_quotient(&hs, &xs);
        let hx = hs.apply(&xs);
        let res = &hx - &(&xs * lam_s);
        let (bl_s, br_s) = tangent_bases(&xs);
        let g_own = TtTangent::project(&bl_s, &br_s, &res);
        let g_xi = projective_transport(x_base, &g_own);
        let g = orbital_gradient(&hs, &xs);
        let gc: Vec<f64> = pairs.iter().map(|&(p, q)| 0.5 * g.get(&[p, q])).collect();
        (g_xi, gc)
    };

    let (mut gxi_p, gc_p) = one(h);
    let (gxi_m, gc_m) = one(-h);
    gxi_p.axpy(-1.0, &gxi_m);
    gxi_p.scale_mut(1.0 / (2.0 * h));
    let gc: Vec<f64> = gc_p
        .iter()
        .zip(&gc_m)
        .map(|(&p, &m)| (p - m) / (2.0 * h))
        .collect();
    (gxi_p, gc)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

pub(crate) fn json_parse(text: &str) -> Result<Json, String> {
    let b = text.as_bytes();
    let mut i = 0usize;
    let v = json_value(b, &mut i)?;
    json_ws(b, &mut i);
    if i != b.len() {
        return Err(format!("trailing input at byte {i}"));
    }
    Ok(v)
}

pub(crate) fn json_get<'a>(v: &'a Json, path: &str) -> Option<&'a Json> {
    let mut cur = v;
    for seg in path.split('/').filter(|s| !s.is_empty()) {
        cur = match cur {
            Json::Obj(members) => members.iter().find(|(k, _)| k == seg).map(|(_, x)| x)?,
            Json::Arr(items) => items.get(seg.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(cur)
}

pub(crate) fn json_num(v: &Json, path: &str) -> Option<f64> {
    match json_get(v, path)? {
        Json::Num(x) => Some(*x),
        _ => None,
    }
}

pub(crate) fn json_nums(v: &Json, path: &str) -> Option<Vec<f64>> {
    match json_get(v, path)? {
        Json::Arr(items) => items
            .iter()
            .map(|x| match x {
                Json::Num(y) => Some(*y),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn json_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\n' | b'\r') {
        *i += 1;
    }
}

fn json_lit(b: &[u8], i: &mut usize, want: &[u8]) -> Result<(), String> {
    if b.len() >= *i + want.len() && &b[*i..*i + want.len()] == want {
        *i += want.len();
        Ok(())
    } else {
        Err(format!(
            "expected `{}` at byte {i}",
            std::str::from_utf8(want).unwrap_or("?")
        ))
    }
}

fn json_value(b: &[u8], i: &mut usize) -> Result<Json, String> {
    json_ws(b, i);
    match b.get(*i) {
        None => Err(format!("unexpected end of input at byte {i}")),
        Some(b'{') => {
            *i += 1;
            let mut members: Vec<(String, Json)> = Vec::new();
            json_ws(b, i);
            if b.get(*i) == Some(&b'}') {
                *i += 1;
                return Ok(Json::Obj(members));
            }
            loop {
                json_ws(b, i);
                let key = json_string(b, i)?;
                json_ws(b, i);
                json_lit(b, i, b":")?;
                let val = json_value(b, i)?;
                members.push((key, val));
                json_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b'}') => {
                        *i += 1;
                        return Ok(Json::Obj(members));
                    }
                    _ => return Err(format!("expected `,` or `}}` at byte {i}")),
                }
            }
        }
        Some(b'[') => {
            *i += 1;
            let mut items: Vec<Json> = Vec::new();
            json_ws(b, i);
            if b.get(*i) == Some(&b']') {
                *i += 1;
                return Ok(Json::Arr(items));
            }
            loop {
                items.push(json_value(b, i)?);
                json_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b']') => {
                        *i += 1;
                        return Ok(Json::Arr(items));
                    }
                    _ => return Err(format!("expected `,` or `]` at byte {i}")),
                }
            }
        }
        Some(b'"') => Ok(Json::Str(json_string(b, i)?)),
        Some(b't') => {
            json_lit(b, i, b"true")?;
            Ok(Json::Bool(true))
        }
        Some(b'f') => {
            json_lit(b, i, b"false")?;
            Ok(Json::Bool(false))
        }
        Some(b'n') => {
            json_lit(b, i, b"null")?;
            Ok(Json::Null)
        }
        Some(_) => {
            let start = *i;
            while *i < b.len()
                && (b[*i].is_ascii_digit() || matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E'))
            {
                *i += 1;
            }
            let s = std::str::from_utf8(&b[start..*i]).map_err(|e| e.to_string())?;
            s.parse::<f64>()
                .map(Json::Num)
                .map_err(|e| format!("bad number `{s}` at byte {start}: {e}"))
        }
    }
}

fn json_string(b: &[u8], i: &mut usize) -> Result<String, String> {
    json_lit(b, i, b"\"")?;
    let mut out: Vec<u8> = Vec::new();
    loop {
        let c = *b.get(*i).ok_or_else(|| "unterminated string".to_string())?;
        *i += 1;
        match c {
            b'"' => return String::from_utf8(out).map_err(|e| e.to_string()),
            b'\\' => {
                let e = *b.get(*i).ok_or_else(|| "unterminated escape".to_string())?;
                *i += 1;
                match e {
                    b'"' => out.push(b'"'),
                    b'\\' => out.push(b'\\'),
                    b'/' => out.push(b'/'),
                    b'b' => out.push(8),
                    b'f' => out.push(12),
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'u' => {
                        let hex = std::str::from_utf8(
                            b.get(*i..*i + 4).ok_or_else(|| "short \\u".to_string())?,
                        )
                        .map_err(|x| x.to_string())?;
                        let cp = u32::from_str_radix(hex, 16).map_err(|x| x.to_string())?;
                        let ch = char::from_u32(cp).ok_or_else(|| "bad codepoint".to_string())?;
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        *i += 4;
                    }
                    _ => return Err(format!("bad escape `\\{}` at byte {i}", e as char)),
                }
            }
            _ => out.push(c),
        }
    }
}
