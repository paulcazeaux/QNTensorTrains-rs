use super::{
    Chain, OPEN_TYPES, SEL_N, SiteCores, StoredFamilies, close_core, fold_left, fold_right,
    id_core, sel_close, sel_n, spin_ix, type_ix,
};
use crate::core::SparseCore;
use crate::core::factor::Sketch;
use crate::orbital::Spin;
use crate::rdm::{active_spins, assert_unit_boundaries, close_trace, pair_sign};
use crate::vector::TtVector;

struct FrozenRtpCache<'a> {
    sc: &'a SiteCores<'a>,
    r_next: &'a Sketch<f64>,
    built: Vec<Option<Sketch<f64>>>,
}

impl<'a> FrozenRtpCache<'a> {
    fn new(sc: &'a SiteCores<'a>, r_next: &'a Sketch<f64>) -> Self {
        FrozenRtpCache {
            sc,
            r_next,
            built: (0..SEL_N * SEL_N).map(|_| None).collect(),
        }
    }

    fn get(&mut self, bsel: usize, ksel: usize) -> &Sketch<f64> {
        let idx = bsel * SEL_N + ksel;
        if self.built[idx].is_none() {
            let built = fold_right(self.sc.sel(ksel), self.r_next, self.sc.sel(bsel));
            self.built[idx] = Some(built);
        }
        self.built[idx].as_ref().expect("just built")
    }
}

#[allow(clippy::needless_range_loop)]
pub(super) fn frozen_generalized_gamma2(x: &TtVector<f64>) -> Vec<f64> {
    let space = x.space;
    let d = space.d;
    assert!(
        space.pg.is_trivial(),
        "generalized_gamma2: point-group gating is not implemented yet — under |G| > 1 only \
         per-side identity-flux cells survive, and computing the rest silently wrong would be \
         worse than this panic"
    );
    let mut gamma = vec![0.0f64; d * d * d * d];
    if d == 0 {
        return gamma;
    }
    assert_unit_boundaries(x, "x");
    let spins = active_spins(&space);
    if spins.is_empty() {
        return gamma;
    }
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * d + j) * d + k) * d + l;

    let mut l_env: Vec<Sketch<f64>> = Vec::with_capacity(d);
    l_env.push(Sketch::identity(space, 0, 1));
    for m in 0..d - 1 {
        let next = fold_left(x.core(m), &l_env[m], x.core(m));
        l_env.push(next);
    }
    let mut r_env: Vec<Option<Sketch<f64>>> = vec![None; d + 1];
    r_env[d] = Some(Sketch::identity(space, d, 1));
    for m in (1..d).rev() {
        let prev = r_env[m + 1].as_ref().expect("right env built rightward");
        r_env[m] = Some(fold_right(x.core(m), prev, x.core(m)));
    }

    let mut rk: StoredFamilies = vec![vec![vec![Vec::new(); d]; 2]; 2];
    let mut rb: StoredFamilies = vec![vec![vec![Vec::new(); d]; 2]; 2];
    for &t in OPEN_TYPES.iter() {
        let ti = type_ix(t);
        for &s in spins.iter() {
            let si = spin_ix(s);
            let closes: Vec<Option<SparseCore<f64>>> = (0..d)
                .map(|q| (q >= 1).then(|| close_core(x.core(q), t, s)))
                .collect();
            let ids: Vec<Option<SparseCore<f64>>> = (0..d)
                .map(|q| (q >= 1 && q + 1 < d).then(|| id_core(x.core(q), t, s)))
                .collect();
            for p in 1..d {
                let r_plain = r_env[p + 1].as_ref().expect("plain right env");
                let close_p = closes[p].as_ref().expect("close core at p ≥ 1");

                let mut by_bond: Vec<Option<Sketch<f64>>> = vec![None; p + 1];
                let mut cur = fold_right(close_p, r_plain, x.core(p));
                by_bond[p] = Some(cur.clone());
                for site in (1..p).rev() {
                    let idc = ids[site].as_ref().expect("identity at 1 ≤ site < d−1");
                    cur = fold_right(idc, &cur, x.core(site));
                    by_bond[site] = Some(cur.clone());
                }
                rk[ti][si][p] = by_bond;

                let mut by_bond_b: Vec<Option<Sketch<f64>>> = vec![None; p + 1];
                let mut cur_b = fold_right(x.core(p), r_plain, close_p);
                by_bond_b[p] = Some(cur_b.clone());
                for site in (1..p).rev() {
                    let idc = ids[site].as_ref().expect("identity at 1 ≤ site < d−1");
                    cur_b = fold_right(x.core(site), &cur_b, idc);
                    by_bond_b[site] = Some(cur_b.clone());
                }
                rb[ti][si][p] = by_bond_b;
            }
        }
    }

    let mut bo: Vec<(Chain, Sketch<f64>)> = Vec::new();
    let mut ko: Vec<(Chain, Sketch<f64>)> = Vec::new();
    let mut bd1: Vec<(usize, Spin, Sketch<f64>)> = Vec::new();
    let mut kd1: Vec<(usize, Spin, Sketch<f64>)> = Vec::new();
    let mut bd2: Vec<((usize, usize), Spin, Sketch<f64>)> = Vec::new();
    let mut kd2: Vec<((usize, usize), Spin, Sketch<f64>)> = Vec::new();
    let mut boko: Vec<(Chain, Chain, Sketch<f64>)> = Vec::new();
    let mut bd1ko: Vec<(usize, Spin, Chain, Sketch<f64>)> = Vec::new();
    let mut bokd1: Vec<(Chain, usize, Spin, Sketch<f64>)> = Vec::new();

    for m in 0..d {
        let xm = x.core(m);
        let r_next = r_env[m + 1].as_ref().expect("plain right env at m+1");
        let sc = SiteCores::new(xm, m, d, &spins);
        let mut rtp = FrozenRtpCache::new(&sc, r_next);

        for &sb in spins.iter() {
            for &sk_ in spins.iter() {
                let v = close_trace(&l_env[m], rtp.get(sel_n(sb), sel_n(sk_)));
                gamma[idx(m, m, m, m)] += v;
            }
        }
        for (ch, sk) in bo.iter() {
            let (cre, ann) = ch.pair(m);
            let sign = pair_sign(cre, ann, ch.spin);
            for &sk_spin in spins.iter() {
                let v = close_trace(sk, rtp.get(sel_close(ch.t, ch.spin), sel_n(sk_spin)));
                gamma[idx(ann, cre, m, m)] += sign * v;
            }
        }
        for (ch, sk) in ko.iter() {
            let (cre, ann) = ch.pair(m);
            let sign = pair_sign(cre, ann, ch.spin);
            for &sb in spins.iter() {
                let v = close_trace(sk, rtp.get(sel_n(sb), sel_close(ch.t, ch.spin)));
                gamma[idx(m, m, cre, ann)] += sign * v;
            }
        }
        for (a, _sb, sk) in bd1.iter() {
            for &sk_spin in spins.iter() {
                let v = close_trace(sk, rtp.get(0, sel_n(sk_spin)));
                gamma[idx(*a, *a, m, m)] += v;
            }
        }
        for (b, _sk, sk) in kd1.iter() {
            for &sb in spins.iter() {
                let v = close_trace(sk, rtp.get(sel_n(sb), 0));
                gamma[idx(m, m, *b, *b)] += v;
            }
        }
        for ((cre, ann), _sb, sk) in bd2.iter() {
            let sign = pair_sign(*cre, *ann, Spin::Up);
            for &sk_spin in spins.iter() {
                let v = close_trace(sk, rtp.get(0, sel_n(sk_spin)));
                gamma[idx(*ann, *cre, m, m)] += sign * v;
            }
        }
        for ((cre, ann), _sk, sk) in kd2.iter() {
            let sign = pair_sign(*cre, *ann, Spin::Up);
            for &sb in spins.iter() {
                let v = close_trace(sk, rtp.get(sel_n(sb), 0));
                gamma[idx(m, m, *cre, *ann)] += sign * v;
            }
        }
        for (chb, chk, sk) in boko.iter() {
            let (bc, ba) = chb.pair(m);
            let (kc, ka) = chk.pair(m);
            let sign = pair_sign(bc, ba, chb.spin) * pair_sign(kc, ka, chk.spin);
            let v = close_trace(
                sk,
                rtp.get(sel_close(chb.t, chb.spin), sel_close(chk.t, chk.spin)),
            );
            gamma[idx(ba, bc, kc, ka)] += sign * v;
        }
        for (a, _sb, chk, sk) in bd1ko.iter() {
            let (kc, ka) = chk.pair(m);
            let sign = pair_sign(kc, ka, chk.spin);
            let v = close_trace(sk, rtp.get(0, sel_close(chk.t, chk.spin)));
            gamma[idx(*a, *a, kc, ka)] += sign * v;
        }
        for (chb, b, _sk, sk) in bokd1.iter() {
            let (bc, ba) = chb.pair(m);
            let sign = pair_sign(bc, ba, chb.spin);
            let v = close_trace(sk, rtp.get(sel_close(chb.t, chb.spin), 0));
            gamma[idx(ba, bc, *b, *b)] += sign * v;
        }

        if m + 1 >= d {
            continue;
        }

        for ((cre, ann), _sb, sk) in bd2.iter() {
            let sign_b = pair_sign(*cre, *ann, Spin::Up);
            for &tk in OPEN_TYPES.iter() {
                for &sk_spin in spins.iter() {
                    let opc = sc.open(tk, sk_spin);
                    let transient = fold_left(xm, sk, opc);
                    let ti = type_ix(tk);
                    for p in m + 1..d {
                        let Some(stored) = rk[ti][spin_ix(sk_spin)][p]
                            .get(m + 1)
                            .and_then(|o| o.as_ref())
                        else {
                            continue;
                        };
                        let v = close_trace(&transient, stored);
                        let ch = Chain {
                            t: tk,
                            spin: sk_spin,
                            site: m,
                        };
                        let (kc, ka) = ch.pair(p);
                        let sign = sign_b * pair_sign(kc, ka, sk_spin);
                        gamma[idx(*ann, *cre, kc, ka)] += sign * v;
                    }
                }
            }
        }
        for ((cre, ann), _sk, sk) in kd2.iter() {
            let sign_k = pair_sign(*cre, *ann, Spin::Up);
            for &tb in OPEN_TYPES.iter() {
                for &sb in spins.iter() {
                    let opc = sc.open(tb, sb);
                    let transient = fold_left(opc, sk, xm);
                    let ti = type_ix(tb);
                    for p in m + 1..d {
                        let Some(stored) =
                            rb[ti][spin_ix(sb)][p].get(m + 1).and_then(|o| o.as_ref())
                        else {
                            continue;
                        };
                        let v = close_trace(&transient, stored);
                        let ch = Chain {
                            t: tb,
                            spin: sb,
                            site: m,
                        };
                        let (bc, ba) = ch.pair(p);
                        let sign = sign_k * pair_sign(bc, ba, sb);
                        gamma[idx(ba, bc, *cre, *ann)] += sign * v;
                    }
                }
            }
        }
        for (ch, sk) in bo.iter() {
            let (bc, ba) = ch.pair(m);
            let sign_b = pair_sign(bc, ba, ch.spin);
            let close_b = sc.close(ch.t, ch.spin);
            for &tk in OPEN_TYPES.iter() {
                for &sk_spin in spins.iter() {
                    let opk = sc.open(tk, sk_spin);
                    let transient = fold_left(close_b, sk, opk);
                    let ti = type_ix(tk);
                    for p in m + 1..d {
                        let Some(stored) = rk[ti][spin_ix(sk_spin)][p]
                            .get(m + 1)
                            .and_then(|o| o.as_ref())
                        else {
                            continue;
                        };
                        let v = close_trace(&transient, stored);
                        let chk = Chain {
                            t: tk,
                            spin: sk_spin,
                            site: m,
                        };
                        let (kc, ka) = chk.pair(p);
                        let sign = sign_b * pair_sign(kc, ka, sk_spin);
                        gamma[idx(ba, bc, kc, ka)] += sign * v;
                    }
                }
            }
        }
        for (ch, sk) in ko.iter() {
            let (kc, ka) = ch.pair(m);
            let sign_k = pair_sign(kc, ka, ch.spin);
            let close_k = sc.close(ch.t, ch.spin);
            for &tb in OPEN_TYPES.iter() {
                for &sb in spins.iter() {
                    let opb = sc.open(tb, sb);
                    let transient = fold_left(opb, sk, close_k);
                    let ti = type_ix(tb);
                    for p in m + 1..d {
                        let Some(stored) =
                            rb[ti][spin_ix(sb)][p].get(m + 1).and_then(|o| o.as_ref())
                        else {
                            continue;
                        };
                        let v = close_trace(&transient, stored);
                        let chb = Chain {
                            t: tb,
                            spin: sb,
                            site: m,
                        };
                        let (bc, ba) = chb.pair(p);
                        let sign = sign_k * pair_sign(bc, ba, sb);
                        gamma[idx(ba, bc, kc, ka)] += sign * v;
                    }
                }
            }
        }
        for (chb, chk, sk) in boko.iter() {
            let (bc, ba) = chb.pair(m);
            let sign_b = pair_sign(bc, ba, chb.spin);
            let close_b = sc.close(chb.t, chb.spin);
            let id_k = sc.id(chk.t, chk.spin);
            let transient = fold_left(close_b, sk, id_k);
            let ti = type_ix(chk.t);
            for p in m + 1..d {
                let Some(stored) = rk[ti][spin_ix(chk.spin)][p]
                    .get(m + 1)
                    .and_then(|o| o.as_ref())
                else {
                    continue;
                };
                let v = close_trace(&transient, stored);
                let (kc, ka) = chk.pair(p);
                let sign = sign_b * pair_sign(kc, ka, chk.spin);
                gamma[idx(ba, bc, kc, ka)] += sign * v;
            }
        }
        for (chb, chk, sk) in boko.iter() {
            let (kc, ka) = chk.pair(m);
            let sign_k = pair_sign(kc, ka, chk.spin);
            let close_k = sc.close(chk.t, chk.spin);
            let id_b = sc.id(chb.t, chb.spin);
            let transient = fold_left(id_b, sk, close_k);
            let ti = type_ix(chb.t);
            for p in m + 1..d {
                let Some(stored) = rb[ti][spin_ix(chb.spin)][p]
                    .get(m + 1)
                    .and_then(|o| o.as_ref())
                else {
                    continue;
                };
                let v = close_trace(&transient, stored);
                let (bc, ba) = chb.pair(p);
                let sign = sign_k * pair_sign(bc, ba, chb.spin);
                gamma[idx(ba, bc, kc, ka)] += sign * v;
            }
        }

        let mut new_bo: Vec<(Chain, Sketch<f64>)> = Vec::new();
        let mut new_ko: Vec<(Chain, Sketch<f64>)> = Vec::new();
        let mut new_bd1: Vec<(usize, Spin, Sketch<f64>)> = Vec::new();
        let mut new_kd1: Vec<(usize, Spin, Sketch<f64>)> = Vec::new();
        let mut new_bd2: Vec<((usize, usize), Spin, Sketch<f64>)> = Vec::new();
        let mut new_kd2: Vec<((usize, usize), Spin, Sketch<f64>)> = Vec::new();
        let mut new_boko: Vec<(Chain, Chain, Sketch<f64>)> = Vec::new();
        let mut new_bd1ko: Vec<(usize, Spin, Chain, Sketch<f64>)> = Vec::new();
        let mut new_bokd1: Vec<(Chain, usize, Spin, Sketch<f64>)> = Vec::new();

        for &t in OPEN_TYPES.iter() {
            for &s in spins.iter() {
                let opc = sc.open(t, s);
                new_bo.push((
                    Chain {
                        t,
                        spin: s,
                        site: m,
                    },
                    fold_left(opc, &l_env[m], xm),
                ));
                new_ko.push((
                    Chain {
                        t,
                        spin: s,
                        site: m,
                    },
                    fold_left(xm, &l_env[m], opc),
                ));
            }
        }
        for &s in spins.iter() {
            new_bd1.push((m, s, fold_left(sc.n(s), &l_env[m], xm)));
            new_kd1.push((m, s, fold_left(xm, &l_env[m], sc.n(s))));
        }
        for (ch, sk) in bo.iter() {
            let close_b = sc.close(ch.t, ch.spin);
            new_bd2.push((ch.pair(m), ch.spin, fold_left(close_b, sk, xm)));
        }
        for (ch, sk) in ko.iter() {
            let close_k = sc.close(ch.t, ch.spin);
            new_kd2.push((ch.pair(m), ch.spin, fold_left(xm, sk, close_k)));
        }
        for (ch, sk) in bo.iter() {
            let id_b = sc.id(ch.t, ch.spin);
            for &tk in OPEN_TYPES.iter() {
                for &sk_spin in spins.iter() {
                    let opc = sc.open(tk, sk_spin);
                    new_boko.push((
                        *ch,
                        Chain {
                            t: tk,
                            spin: sk_spin,
                            site: m,
                        },
                        fold_left(id_b, sk, opc),
                    ));
                }
            }
        }
        for (ch, sk) in ko.iter() {
            let id_k = sc.id(ch.t, ch.spin);
            for &tb in OPEN_TYPES.iter() {
                for &sb in spins.iter() {
                    let opc = sc.open(tb, sb);
                    new_boko.push((
                        Chain {
                            t: tb,
                            spin: sb,
                            site: m,
                        },
                        *ch,
                        fold_left(opc, sk, id_k),
                    ));
                }
            }
        }
        for &tb in OPEN_TYPES.iter() {
            for &sb in spins.iter() {
                let opb = sc.open(tb, sb);
                for &tk in OPEN_TYPES.iter() {
                    for &sk_spin in spins.iter() {
                        let opk = sc.open(tk, sk_spin);
                        new_boko.push((
                            Chain {
                                t: tb,
                                spin: sb,
                                site: m,
                            },
                            Chain {
                                t: tk,
                                spin: sk_spin,
                                site: m,
                            },
                            fold_left(opb, &l_env[m], opk),
                        ));
                    }
                }
            }
        }
        for (a, sb, sk) in bd1.iter() {
            for &tk in OPEN_TYPES.iter() {
                for &sk_spin in spins.iter() {
                    let opc = sc.open(tk, sk_spin);
                    new_bd1ko.push((
                        *a,
                        *sb,
                        Chain {
                            t: tk,
                            spin: sk_spin,
                            site: m,
                        },
                        fold_left(xm, sk, opc),
                    ));
                }
            }
        }
        for (ch, sk) in ko.iter() {
            let id_k = sc.id(ch.t, ch.spin);
            for &sb in spins.iter() {
                new_bd1ko.push((m, sb, *ch, fold_left(sc.n(sb), sk, id_k)));
            }
        }
        for &sb in spins.iter() {
            for &tk in OPEN_TYPES.iter() {
                for &sk_spin in spins.iter() {
                    let opk = sc.open(tk, sk_spin);
                    new_bd1ko.push((
                        m,
                        sb,
                        Chain {
                            t: tk,
                            spin: sk_spin,
                            site: m,
                        },
                        fold_left(sc.n(sb), &l_env[m], opk),
                    ));
                }
            }
        }
        for (b, sk_spin, sk) in kd1.iter() {
            for &tb in OPEN_TYPES.iter() {
                for &sb in spins.iter() {
                    let opc = sc.open(tb, sb);
                    new_bokd1.push((
                        Chain {
                            t: tb,
                            spin: sb,
                            site: m,
                        },
                        *b,
                        *sk_spin,
                        fold_left(opc, sk, xm),
                    ));
                }
            }
        }
        for (ch, sk) in bo.iter() {
            let id_b = sc.id(ch.t, ch.spin);
            for &sk_spin in spins.iter() {
                new_bokd1.push((*ch, m, sk_spin, fold_left(id_b, sk, sc.n(sk_spin))));
            }
        }
        for &tb in OPEN_TYPES.iter() {
            for &sb in spins.iter() {
                let opb = sc.open(tb, sb);
                for &sk_spin in spins.iter() {
                    new_bokd1.push((
                        Chain {
                            t: tb,
                            spin: sb,
                            site: m,
                        },
                        m,
                        sk_spin,
                        fold_left(opb, &l_env[m], sc.n(sk_spin)),
                    ));
                }
            }
        }

        for (ch, sk) in bo.iter_mut() {
            let id_b = sc.id(ch.t, ch.spin);
            *sk = fold_left(id_b, sk, xm);
        }
        for (ch, sk) in ko.iter_mut() {
            let id_k = sc.id(ch.t, ch.spin);
            *sk = fold_left(xm, sk, id_k);
        }
        for (_a, _s, sk) in bd1.iter_mut() {
            *sk = fold_left(xm, sk, xm);
        }
        for (_b, _s, sk) in kd1.iter_mut() {
            *sk = fold_left(xm, sk, xm);
        }
        for (_p, _s, sk) in bd2.iter_mut() {
            *sk = fold_left(xm, sk, xm);
        }
        for (_p, _s, sk) in kd2.iter_mut() {
            *sk = fold_left(xm, sk, xm);
        }
        for (chb, chk, sk) in boko.iter_mut() {
            let id_b = sc.id(chb.t, chb.spin);
            let id_k = sc.id(chk.t, chk.spin);
            *sk = fold_left(id_b, sk, id_k);
        }
        for (_a, _s, chk, sk) in bd1ko.iter_mut() {
            let id_k = sc.id(chk.t, chk.spin);
            *sk = fold_left(xm, sk, id_k);
        }
        for (chb, _b, _s, sk) in bokd1.iter_mut() {
            let id_b = sc.id(chb.t, chb.spin);
            *sk = fold_left(id_b, sk, xm);
        }

        bo.extend(new_bo);
        ko.extend(new_ko);
        bd1.extend(new_bd1);
        kd1.extend(new_kd1);
        bd2.extend(new_bd2);
        kd2.extend(new_kd2);
        boko.extend(new_boko);
        bd1ko.extend(new_bd1ko);
        bokd1.extend(new_bokd1);
    }

    gamma
}
