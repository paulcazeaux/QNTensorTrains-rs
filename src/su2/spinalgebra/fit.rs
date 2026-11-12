// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::BTreeMap;
use std::rc::Rc;

use rustc_hash::FxHashMap;

use super::helpers::{descending, pow4};
use super::siteops::{dense_term, occ_count};
use super::{Coo, PatternKey, SiteOpId, SpinAlgebra, TensorOp, TermChannel};

fn amul(cols: &[Vec<f64>], w: &[f64], nrows: usize) -> Vec<f64> {
    let mut v = vec![0.0f64; nrows];
    for (c, col) in cols.iter().enumerate() {
        let wc = w[c];
        if wc == 0.0 {
            continue;
        }
        for (i, vi) in v.iter_mut().enumerate() {
            *vi += col[i] * wc;
        }
    }
    v
}

#[allow(clippy::too_many_arguments)]
fn walk_paths(
    p: usize,
    q: usize,
    twok: i32,
    chan_ks: &[Vec<i32>],
    suffix: &[i32],
    ranks: &mut Vec<i32>,
    choice: &mut Vec<usize>,
    paths: &mut Vec<(Vec<i32>, Vec<usize>)>,
) {
    if p == q {
        if twok == 0 {
            paths.push((ranks.clone(), choice.clone()));
        }
        return;
    }
    for (ci, &opk) in chan_ks[p].iter().enumerate() {
        let lo = (twok - opk).abs();
        let hi = twok + opk;
        let mut twok2 = lo;
        while twok2 <= hi {
            if twok2 <= suffix[p + 1] {
                ranks.push(twok2);
                choice.push(ci);
                walk_paths(p + 1, q, twok2, chan_ks, suffix, ranks, choice, paths);
                ranks.pop();
                choice.pop();
            }
            twok2 += 2;
        }
    }
}

#[allow(dead_code)]
impl SpinAlgebra {
    /// Analyze a spin-summed term (tokens in operator order); enumerate admissible operator-bond
    /// rank paths, least-squares-fit each channel's weight against `dense_term`, and return the
    /// surviving channels (`term_channels`). **THE `residual ≤
    /// 1e-12·max(1,‖D‖)` assert is the layer's convention-certifier — keep it a release `assert!`.**
    /// Cached on the canonical [`PatternKey`] (position- and d-independent); iterative refinement
    /// (2 rounds of `w += A\(b-Aw)`) pushes weights to ≤1-ulp accuracy, load-bearing for the
    /// freeze's exact-zero cancellation drop.
    pub fn term_channels(&mut self, tokens: &[(usize, bool)]) -> Rc<Vec<TermChannel>> {
        let mut sites: Vec<usize> = tokens.iter().map(|t| t.0).collect();
        sites.sort_unstable();
        sites.dedup();
        let q = sites.len();
        let pos = |s: usize| sites.iter().position(|&x| x == s).unwrap();
        let pattern: PatternKey = tokens.iter().map(|&(s, c)| (pos(s), c)).collect();
        if let Some(v) = self.channel_cache.get(&pattern) {
            return Rc::clone(v);
        }
        let minitokens = &pattern;

        let groups: Vec<Vec<bool>> = (0..q)
            .map(|p| {
                minitokens
                    .iter()
                    .filter(|t| t.0 == p)
                    .map(|t| t.1)
                    .collect()
            })
            .collect();
        let mut chans: Vec<Rc<Vec<SiteOpId>>> = Vec::with_capacity(q);
        for g in &groups {
            chans.push(self.site_channel_ops(g));
        }
        let mut tens: Vec<BTreeMap<i32, TensorOp>> = Vec::with_capacity(q);
        for g in &groups {
            tens.push(self.group_tensors(g));
        }

        let chan_ks: Vec<Vec<i32>> = chans
            .iter()
            .map(|ids| ids.iter().map(|&id| self.siteops[id].two_k).collect())
            .collect();
        let site_max: Vec<i32> = chan_ks
            .iter()
            .map(|ks| ks.iter().copied().max().unwrap_or(0))
            .collect();
        let mut suffix = vec![0i32; q + 1];
        for i in (0..q).rev() {
            suffix[i] = suffix[i + 1] + site_max[i];
        }

        let mut paths: Vec<(Vec<i32>, Vec<usize>)> = Vec::new();
        walk_paths(
            0,
            q,
            0,
            &chan_ks,
            &suffix,
            &mut vec![0],
            &mut Vec::new(),
            &mut paths,
        );
        assert!(
            !paths.is_empty(),
            "no admissible rank path for pattern {pattern:?}"
        );

        let d = dense_term(q, minitokens);
        let mut dc: Vec<Coo> = Vec::with_capacity(paths.len());
        for (ranks, choice) in &paths {
            let tensors: Vec<Option<TensorOp>> = (0..q)
                .map(|p| {
                    let opk = self.siteops[chans[p][choice[p]]].two_k;
                    Some(tens[p][&opk].clone())
                })
                .collect();
            dc.push(self.dense_channel(q, ranks, &tensors));
        }

        let mut row_of: FxHashMap<(usize, usize), usize> = FxHashMap::default();
        let mut assign = |ij: (usize, usize), n: &mut usize| {
            row_of.entry(ij).or_insert_with(|| {
                let r = *n;
                *n += 1;
                r
            });
        };
        let mut nrows = 0usize;
        for c in &dc {
            for &ij in c.keys() {
                assign(ij, &mut nrows);
            }
        }
        for &ij in d.keys() {
            assign(ij, &mut nrows);
        }
        let ncols = dc.len();

        let w: Vec<f64> = if nrows == 0 {
            vec![0.0; ncols]
        } else {
            let mut colvecs: Vec<Vec<f64>> = vec![vec![0.0; nrows]; ncols];
            for (c, chan) in dc.iter().enumerate() {
                for (&ij, &val) in chan {
                    colvecs[c][row_of[&ij]] = val;
                }
            }
            let mut b = vec![0.0f64; nrows];
            for (&ij, &val) in &d {
                b[row_of[&ij]] = val;
            }
            let colrefs: Vec<&[f64]> = colvecs.iter().map(|c| c.as_slice()).collect();
            let mut w = crate::core::factor::dense_lstsq_colpiv::<f64>(nrows, &colrefs, &b);
            for _ in 0..2 {
                let aw = amul(&colvecs, &w, nrows);
                let resid: Vec<f64> = (0..nrows).map(|i| b[i] - aw[i]).collect();
                let delta = crate::core::factor::dense_lstsq_colpiv::<f64>(nrows, &colrefs, &resid);
                for (wc, dc) in w.iter_mut().zip(delta.iter()) {
                    *wc += *dc;
                }
            }
            let aw = amul(&colvecs, &w, nrows);
            let rn = (0..nrows)
                .map(|i| (aw[i] - b[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                rn <= 1e-12 * f64::max(1.0, bn),
                "channel fit failed for pattern {pattern:?}: residual {rn} (‖D‖={bn})"
            );
            w
        };

        let mut out: Vec<TermChannel> = Vec::new();
        for (i, (ranks, choice)) in paths.iter().enumerate() {
            if w[i].abs() <= 1e-13 {
                continue;
            }
            out.push(TermChannel {
                ranks: ranks.clone(),
                siteops: (0..q).map(|p| chans[p][choice[p]]).collect(),
                weight: w[i],
            });
        }
        let rc = Rc::new(out);
        self.channel_cache.insert(pattern, Rc::clone(&rc));
        rc
    }

    /// Coupled product's dense-reference operator on `q` mini-sites, propagating through ONE chain
    /// channel. `tensors[p] = None` is a transport (identity) bond; `ranks` is the length-`q+1`
    /// operator- bond rank path. Sparse left-to-right propagation over `(dout,din,parity,twoq)`
    /// states, collapsing to `(dout,din)` at the closed `twoq == 0` operator bond.
    #[allow(clippy::needless_range_loop)]
    pub fn dense_channel(&mut self, q: usize, ranks: &[i32], tensors: &[Option<TensorOp>]) -> Coo {
        let mut states: BTreeMap<(usize, usize, i32, i32), f64> = BTreeMap::new();
        states.insert((0, 0, 0, 0), 1.0);
        for p in 0..q {
            let (km, kp) = (ranks[p], ranks[p + 1]);
            let w4 = pow4(p);
            let mut new: BTreeMap<(usize, usize, i32, i32), f64> = BTreeMap::new();
            for (&(dout, din, par, twoq), &amp) in &states {
                match &tensors[p] {
                    None => {
                        if km != kp {
                            continue;
                        }
                        for s_in in 0..4 {
                            let key = (
                                dout + s_in * w4,
                                din + s_in * w4,
                                (par + occ_count(s_in) as i32) % 2,
                                twoq,
                            );
                            *new.entry(key).or_insert(0.0) += amp;
                        }
                    }
                    Some(t) => {
                        for (imu, twomu) in descending(t.two_k) {
                            let twoqp = twoq + twomu;
                            if twoqp.abs() > kp {
                                continue;
                            }
                            let cg = self.clebschgordan(km, twoq, t.two_k, twomu, kp, twoqp);
                            if cg == 0.0 {
                                continue;
                            }
                            let comp = &t.components[imu];
                            for s_in in 0..4 {
                                let jw = if t.odd && par == 1 { -1.0 } else { 1.0 };
                                for s_out in 0..4 {
                                    let c = comp[s_out][s_in];
                                    if c == 0.0 {
                                        continue;
                                    }
                                    let key = (
                                        dout + s_out * w4,
                                        din + s_in * w4,
                                        (par + occ_count(s_in) as i32) % 2,
                                        twoqp,
                                    );
                                    *new.entry(key).or_insert(0.0) += amp * cg * c * jw;
                                }
                            }
                        }
                    }
                }
            }
            states = new;
        }
        let mut d: Coo = BTreeMap::new();
        for (&(dout, din, _par, twoq), &amp) in &states {
            if twoq == 0 {
                *d.entry((dout, din)).or_insert(0.0) += amp;
            }
        }
        d
    }
}
