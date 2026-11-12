// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::rc::Rc;

use super::helpers::frob;
use super::siteops::elementary_tensor;
use super::{SiteOp, SiteOpId, SpinAlgebra, TensorOp};

fn pack_group(cdags: &[bool]) -> u64 {
    debug_assert!(
        cdags.len() <= 32,
        "pack_group cache key is injective only for <= 32 tokens (got {})",
        cdags.len()
    );
    let mut bits = 0u64;
    for (i, &c) in cdags.iter().enumerate() {
        if c {
            bits |= 1u64 << i;
        }
    }
    ((cdags.len() as u64) << 32) | bits
}

#[allow(dead_code)]
impl SpinAlgebra {
    /// All nonvanishing irreducible-rank channels of the ordered product of elementary tensors on
    /// ONE site, sorted by `two_k` (`site_channel_ops`). **THE
    /// identity-interning conversion site**: memoized on the token group, returns
    /// [`SiteOpId`]s into [`Self::siteops`] so repeated calls on an equal group yield the SAME ids.
    pub fn site_channel_ops(&mut self, cdags: &[bool]) -> Rc<Vec<SiteOpId>> {
        let key = pack_group(cdags);
        if let Some(v) = self.siteops_cache.get(&key) {
            return Rc::clone(v);
        }
        let tensors = self.group_tensors(cdags);
        let mut ops: Vec<SiteOp> = Vec::new();
        for t in tensors.values() {
            let red = self.reduce_siteop(t);
            ops.push(SiteOp {
                two_k: t.two_k,
                flux: t.flux,
                odd: t.odd,
                red,
            });
        }
        ops.sort_by_key(|o| o.two_k);
        let ids: Vec<SiteOpId> = ops
            .into_iter()
            .map(|op| {
                let id = self.siteops.len();
                self.siteops.push(op);
                id
            })
            .collect();
        let rc = Rc::new(ids);
        self.siteops_cache.insert(key, Rc::clone(&rc));
        rc
    }

    pub(super) fn group_tensors(&mut self, cdags: &[bool]) -> BTreeMap<i32, TensorOp> {
        let mut acc = vec![elementary_tensor(cdags[0])];
        for &c in &cdags[1..] {
            let t2 = elementary_tensor(c);
            let mut nxt: BTreeMap<i32, TensorOp> = BTreeMap::new();
            for t1 in &acc {
                let lo = (t1.two_k - t2.two_k).abs();
                let hi = t1.two_k + t2.two_k;
                let mut kk = lo;
                while kk <= hi {
                    let cpl = self.couple(t1, &t2, kk);
                    match nxt.entry(kk) {
                        Entry::Occupied(mut e) => {
                            let p = e.get_mut();
                            for i in 0..=(kk as usize) {
                                for a in 0..4 {
                                    for b in 0..4 {
                                        p.components[i][a][b] += cpl.components[i][a][b];
                                    }
                                }
                            }
                        }
                        Entry::Vacant(e) => {
                            e.insert(cpl);
                        }
                    }
                    kk += 2;
                }
            }
            acc = nxt.into_values().collect();
        }
        acc.into_iter()
            .filter(|t| t.components.iter().any(|c| frob(c) > 1e-13))
            .map(|t| (t.two_k, t))
            .collect()
    }
}
