// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Elementary fermionic operators acting on a [`TtVector`]: one-, two-, and three-body
//! creation/annihilation strings applied as a left-to-right operator-state sweep over the cores.
//! Each entry point is a thin wrapper over a shared driver (the forward
//! sweep, flux threading, `scale_mut`, and `check` are identical across arities; only the
//! creation/annihilation lists, corner-gate, and initial state differ) that dispatches per-site to
//! one of the eleven on-site kernels (`Id`, `Adag`, `A`, `N`, `AdagupAdagdn`, `AupAdn`, `S₊`, `S₋`,
//! `AN`, `AdagN`, `NN`) via a `(upop,dnop)` predicate (`select_site_op`), which collapses the whole
//! site-coincidence cascade into one table.

use faer::prelude::ReborrowMut;

use crate::coeff::CoeffTensor;
use crate::core::{Family, SparseCore};
use crate::orbital::{Orbital, Spin, fermionic_sign};
use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::{QnSpace, RankGrid, Sector, SectorGrid};
use crate::scalar::Scalar;
use crate::vector::{TtVector, tt_zeros};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SiteOp {
    Id,
    A,
    Adag,
    N,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct State {
    pub fu: i64,
    pub nlu: i64,
    pub nru: i64,
    pub fd: i64,
    pub nld: i64,
    pub nrd: i64,
    pub delta: Irrep,
}

#[derive(Clone, Copy)]
enum Sgn {
    Jw0,
    Jw1,
    Plus,
    Minus,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn shift_qn(
    space: &QnSpace,
    caps: Option<&[RankGrid]>,
    b: usize,
    fu: i64,
    nlu: i64,
    nru: i64,
    fd: i64,
    nld: i64,
    nrd: i64,
    delta: Irrep,
) -> Vec<Sector> {
    let nup_max = space.nup as i64 - nru;
    let ndn_max = space.ndn as i64 - nrd;
    space
        .bond_sectors(b)
        .into_iter()
        .filter(|s| {
            let nup = s.nup as i64;
            let ndn = s.ndn as i64;
            nlu <= nup
                && nup <= nup_max
                && nld <= ndn
                && ndn <= ndn_max
                && nup - fu >= 0
                && ndn - fd >= 0
                && space.contains(
                    b,
                    Sector::with_irrep(
                        (nup - fu) as usize,
                        (ndn - fd) as usize,
                        s.irrep.fuse(delta),
                    ),
                )
                && caps.is_none_or(|c| c[b][*s] > 0)
        })
        .collect()
}

#[allow(clippy::type_complexity)]
fn descriptor(
    up: SiteOp,
    dn: SiteOp,
) -> (
    (i64, i64, i64),
    (i64, i64, i64),
    &'static [(Family, Family, Sgn)],
) {
    use Family::{Dn, Empty, Occ, Up};
    use Sgn::{Jw0, Jw1, Minus, Plus};
    use SiteOp as O;
    match (up, dn) {
        (O::Id, O::Id) => (
            (0, 0, 0),
            (0, 0, 0),
            &[
                (Empty, Empty, Plus),
                (Up, Up, Plus),
                (Dn, Dn, Plus),
                (Occ, Occ, Plus),
            ],
        ),
        (O::Adag, O::Id) => ((1, 1, -1), (0, 0, 0), &[(Up, Empty, Jw0), (Occ, Dn, Jw0)]),
        (O::Id, O::Adag) => ((0, 0, 0), (1, 1, -1), &[(Dn, Empty, Jw0), (Occ, Up, Jw1)]),
        (O::A, O::Id) => ((-1, 0, 0), (0, 0, 0), &[(Empty, Up, Jw0), (Dn, Occ, Jw0)]),
        (O::Id, O::A) => ((0, 0, 0), (-1, 0, 0), &[(Empty, Dn, Jw0), (Up, Occ, Jw1)]),
        (O::N, O::Id) => ((0, 1, -1), (0, 0, 0), &[(Up, Up, Plus), (Occ, Occ, Plus)]),
        (O::Id, O::N) => ((0, 0, 0), (0, 1, -1), &[(Dn, Dn, Plus), (Occ, Occ, Plus)]),
        (O::Adag, O::Adag) => ((1, 1, -1), (1, 1, -1), &[(Occ, Empty, Plus)]),
        (O::A, O::A) => ((-1, 0, 0), (-1, 0, 0), &[(Empty, Occ, Minus)]),
        (O::Adag, O::A) => ((1, 1, -1), (-1, 0, 0), &[(Up, Dn, Plus)]),
        (O::A, O::Adag) => ((-1, 0, 0), (1, 1, -1), &[(Dn, Up, Minus)]),
        (O::A, O::N) => ((-1, 0, 0), (0, 1, -1), &[(Dn, Occ, Jw0)]),
        (O::N, O::A) => ((0, 1, -1), (-1, 0, 0), &[(Up, Occ, Jw1)]),
        (O::Adag, O::N) => ((1, 1, -1), (0, 1, -1), &[(Occ, Dn, Jw0)]),
        (O::N, O::Adag) => ((0, 1, -1), (1, 1, -1), &[(Occ, Up, Jw1)]),
        (O::N, O::N) => ((0, 1, -1), (0, 1, -1), &[(Occ, Occ, Plus)]),
    }
}

pub(crate) fn apply_site_op<T: Scalar>(
    a: &SparseCore<T>,
    caps: Option<&[RankGrid]>,
    up: SiteOp,
    dn: SiteOp,
    st: State,
) -> (SparseCore<T>, State) {
    apply_site_op_into(a, a.space, caps, up, dn, st)
}

pub(crate) fn apply_site_op_into<T: Scalar>(
    a: &SparseCore<T>,
    space_out: QnSpace,
    caps: Option<&[RankGrid]>,
    up: SiteOp,
    dn: SiteOp,
    st: State,
) -> (SparseCore<T>, State) {
    debug_assert!(
        space_out.d == a.space.d
            && space_out.nup == a.space.nup
            && space_out.ndn == a.space.ndn
            && space_out.pg == a.space.pg,
        "apply_site_op_into: the output space may differ from the input space in `target` only"
    );
    let (dup, ddn, writes) = descriptor(up, dn);
    let space = space_out;
    let k = a.k;
    let g = space.pg.irrep_of(k);
    let dg = if (dup.0 + ddn.0).rem_euclid(2) == 1 {
        g
    } else {
        Irrep::IDENTITY
    };
    let out = State {
        fu: st.fu + dup.0,
        nlu: st.nlu + dup.1,
        nru: st.nru + dup.2,
        fd: st.fd + ddn.0,
        nld: st.nld + ddn.1,
        nrd: st.nrd + ddn.2,
        delta: st.delta.fuse(dg),
    };
    debug_assert!(
        writes
            .iter()
            .all(|&(tgt, srcfam, _)| tgt.flux(g).fuse(srcfam.flux(g)) == dg),
        "site-{k} operator irrep flux disagrees with the descriptor's family writes"
    );

    let ql = shift_qn(
        &space, caps, k, st.fu, st.nlu, st.nru, st.fd, st.nld, st.nrd, st.delta,
    );
    let qr = shift_qn(
        &space,
        caps,
        k + 1,
        out.fu,
        out.nlu,
        out.nru,
        out.fd,
        out.nld,
        out.nrd,
        out.delta,
    );

    let mut rowranks = RankGrid::zeros(&space);
    for &s in &ql {
        let src = Sector::with_irrep(
            (s.nup as i64 - st.fu) as usize,
            (s.ndn as i64 - st.fd) as usize,
            s.irrep.fuse(st.delta),
        );
        rowranks[s] = a.row_ranks[src];
    }
    let mut colranks = RankGrid::zeros(&space);
    for &s in &qr {
        let src = Sector::with_irrep(
            (s.nup as i64 - out.fu) as usize,
            (s.ndn as i64 - out.fd) as usize,
            s.irrep.fuse(out.delta),
        );
        colranks[s] = a.col_ranks[src];
    }

    let mut b = SparseCore::zeros(space, k, &rowranks, &colranks);
    let mut qr_set = SectorGrid::<bool>::filled(&space, false);
    for &s in &qr {
        qr_set[s] = true;
    }

    for &s in &ql {
        let src = Sector::with_irrep(
            (s.nup as i64 - st.fu) as usize,
            (s.ndn as i64 - st.fd) as usize,
            s.irrep.fuse(st.delta),
        );
        let jw0 = if (src.nup + src.ndn) % 2 == 1 {
            -1.0
        } else {
            1.0
        };
        for &(tgt, srcfam, sgn) in writes {
            let t = tgt.apply(s, space.pg.irrep_of(k));
            if t.nup <= space.nup && t.ndn <= space.ndn && qr_set[t] {
                let sign = match sgn {
                    Sgn::Jw0 => jw0,
                    Sgn::Jw1 => -jw0,
                    Sgn::Plus => 1.0,
                    Sgn::Minus => -1.0,
                };
                if let Some(srcb) = a.block(srcfam, src) {
                    let sval = T::from_f64(sign);
                    let mut dstb = b
                        .block_mut(tgt, s)
                        .expect("target block admissible by shift_qn construction");
                    for c in 0..srcb.ncols() {
                        for r in 0..srcb.nrows() {
                            *dstb.rb_mut().get_mut(r, c) = sval * *srcb.get(r, c);
                        }
                    }
                }
            }
        }
    }

    (b, out)
}

pub(crate) fn select_site_op(site: usize, cre: &[Orbital], ann: &[Orbital]) -> (SiteOp, SiteOp) {
    let hit = |list: &[Orbital], spin: Spin| list.iter().any(|o| o.site == site && o.spin == spin);
    let pick = |c: bool, an: bool| {
        if c && an {
            SiteOp::N
        } else if c {
            SiteOp::Adag
        } else if an {
            SiteOp::A
        } else {
            SiteOp::Id
        }
    };
    (
        pick(hit(cre, Spin::Up), hit(ann, Spin::Up)),
        pick(hit(cre, Spin::Dn), hit(ann, Spin::Dn)),
    )
}

fn operator_cores<T: Scalar>(
    tt: &TtVector<T>,
    cre: &[Orbital],
    ann: &[Orbital],
    space_out: QnSpace,
) -> (Vec<SparseCore<T>>, State) {
    let d = space_out.d;

    let nru = cre.iter().filter(|o| o.spin == Spin::Up).count() as i64;
    let nrd = cre.iter().filter(|o| o.spin == Spin::Dn).count() as i64;
    let mut st = State {
        fu: 0,
        nlu: 0,
        nru,
        fd: 0,
        nld: 0,
        nrd,
        delta: Irrep::IDENTITY,
    };

    let caps = (!space_out.pg.is_trivial()).then(|| crate::qn::max_ranks(&space_out));
    let caps = caps.as_deref();

    let mut cores = Vec::with_capacity(d);
    for k in 0..d {
        let (up, dn) = select_site_op(k, cre, ann);
        let (c, next) = apply_site_op_into(tt.core(k), space_out, caps, up, dn, st);
        cores.push(c);
        st = next;
    }
    (cores, st)
}

fn finish_operator<T: Scalar>(
    space_out: QnSpace,
    cores: Vec<SparseCore<T>>,
    cre: &[Orbital],
    ann: &[Orbital],
    coeff: T,
) -> TtVector<T> {
    let mut seq: Vec<Orbital> = Vec::with_capacity(cre.len() + ann.len());
    seq.extend_from_slice(cre);
    seq.extend_from_slice(ann);
    let sign = fermionic_sign(&seq);

    let mut out = TtVector::new(space_out, cores);
    out.scale_mut(T::from_f64(sign as f64) * coeff);
    debug_assert!(out.check());
    out
}

pub(crate) fn build_operator<T: Scalar>(
    tt: &TtVector<T>,
    cre: &[Orbital],
    ann: &[Orbital],
    coeff: T,
) -> TtVector<T> {
    let space = tt.space;
    let (cores, st) = operator_cores(tt, cre, ann, space);
    assert!(
        st.delta == Irrep::IDENTITY,
        "operator string has total irrep flux {:?} != IDENTITY: it maps target {:?} out of the \
         space and has no representation over this QnSpace",
        st.delta,
        space.target
    );

    finish_operator(space, cores, cre, ann, coeff)
}

fn build_operator_cross_target<T: Scalar>(
    tt: &TtVector<T>,
    cre: &[Orbital],
    ann: &[Orbital],
    coeff: T,
    space_out: QnSpace,
    gamma_mu: Irrep,
) -> TtVector<T> {
    let (cores, st) = operator_cores(tt, cre, ann, space_out);
    assert_eq!(
        st.delta, gamma_mu,
        "operator string has total irrep flux {:?}, but the cross-target output space was built \
         for {:?} (target {:?} -> {:?})",
        st.delta, gamma_mu, tt.space.target, space_out.target
    );
    finish_operator(space_out, cores, cre, ann, coeff)
}

/// The distinct irrep fluxes `Γ_p ⊗ Γ_q` carried by the nonzero support of a one-body property
/// matrix `m`, **ascending by [`Irrep`] index**.
///
/// # Panics
/// If `m` is not rank-2.
pub fn property_flux_classes<T: Scalar>(m: &CoeffTensor<T>, pg: PointGroup) -> Vec<Irrep> {
    assert_eq!(
        m.rank, 2,
        "property_flux_classes: property matrix must be rank-2 (got rank {})",
        m.rank
    );
    let mut classes: Vec<Irrep> = Vec::new();
    for p in 0..m.d {
        for q in 0..m.d {
            if m.get(&[p, q]).abs_s() != 0.0 {
                let g = pg.irrep_of(p).fuse(pg.irrep_of(q));
                if !classes.contains(&g) {
                    classes.push(g);
                }
            }
        }
    }
    classes.sort_unstable();
    classes
}

/// `y = μ̂·x` for a **single-flux** one-body property `μ̂ = Σ_{p,q} m[p,q]·E_pq`, with the output
/// living in the FUSED-target space `y.space.target = x.space.target ⊗ Γ_μ`.
///
/// # Panics
/// - `m` is not rank-2, or not `d × d`, or not symmetric
///   ([`assert_property_hermitian`](crate::integrals::assert_property_hermitian));
/// - `pg != x.space.pg` (a C1 space with an external table needs no cross-target map at all, and
///   mixing the two would need an irrep re-blocking that does not exist);
/// - `m`'s support carries more than one flux class — call once per
///   [`property_flux_classes`] entry.
///
/// `m ≡ 0` is **accepted**, not an error: it returns the exact zero train in the unchanged-target
/// space, mirroring `property_hamiltonian`'s own `m ≡ 0` arm.
pub fn apply_cross_target<T: Scalar>(
    x: &TtVector<T>,
    m: &CoeffTensor<T>,
    pg: PointGroup,
) -> TtVector<T> {
    let space = x.space;
    let d = space.d;
    assert!(
        m.rank == 2 && m.d == d,
        "apply_cross_target: property matrix must be rank-2 over d={d} sites (got rank {}, d {})",
        m.rank,
        m.d
    );
    assert_eq!(
        pg, space.pg,
        "apply_cross_target: `pg` must be the state's own point group; a C1 space carrying an \
         external irrep table needs no cross-target map (its target is already IDENTITY and every \
         property is representable there)"
    );
    crate::integrals::assert_property_hermitian(m, "apply_cross_target");

    let classes = property_flux_classes(m, pg);
    assert!(
        classes.len() <= 1,
        "apply_cross_target: the property carries {} distinct irrep flux classes {:?} — each maps \
         the target to a different output space, so there is no single output train. Split `m` by \
         `property_flux_classes` and call once per class.",
        classes.len(),
        classes
    );
    let Some(&gamma_mu) = classes.first() else {
        return tt_zeros(space);
    };
    let space_out =
        QnSpace::with_point_group(d, space.nup, space.ndn, pg, space.target.fuse(gamma_mu));

    let mut acc: Option<TtVector<T>> = None;
    for p in 0..d {
        for q in 0..d {
            let c = m.get(&[p, q]);
            if c.abs_s() == 0.0 {
                continue;
            }
            for spin in [Spin::Up, Spin::Dn] {
                if (spin == Spin::Up && space.nup < 1) || (spin == Spin::Dn && space.ndn < 1) {
                    continue;
                }
                let o = Orbital::new(p, spin);
                let term = build_operator_cross_target(
                    x,
                    &[o],
                    &[Orbital::new(q, spin)],
                    c,
                    space_out,
                    gamma_mu,
                );
                acc = Some(match acc {
                    None => term,
                    Some(a) => &a + &term,
                });
            }
        }
    }
    acc.unwrap_or_else(|| tt_zeros(space_out))
}

/// `a†_i a_j` (one-body, same-spin hop/density).
/// Requires `i.spin == j.spin`.
pub fn adag_i_a_j<T: Scalar>(tt: &TtVector<T>, i: Orbital, j: Orbital, t: T) -> TtVector<T> {
    let space = tt.space;
    assert_eq!(i.spin, j.spin, "AdagᵢAⱼ requires i.spin == j.spin");
    assert!(
        i.site < space.d && j.site < space.d,
        "orbital site out of range"
    );

    if (i.spin == Spin::Up && space.nup < 1) || (i.spin == Spin::Dn && space.ndn < 1) {
        return tt_zeros(space);
    }
    build_operator(tt, &[i], &[j], t)
}

/// `a†_i a†_k a_l a_j` (two-body). Requires `i.spin == j.spin`, `k.spin == l.spin`, `i ≠ k`, `j ≠
/// l`.
pub fn adag_i_adag_k_a_l_a_j<T: Scalar>(
    tt: &TtVector<T>,
    i: Orbital,
    j: Orbital,
    k: Orbital,
    l: Orbital,
    w: T,
) -> TtVector<T> {
    let space = tt.space;
    assert_eq!(i.spin, j.spin, "AdagᵢAdagₖAₗAⱼ requires i.spin == j.spin");
    assert_eq!(k.spin, l.spin, "AdagᵢAdagₖAₗAⱼ requires k.spin == l.spin");
    assert_ne!(i, k, "AdagᵢAdagₖAₗAⱼ requires i ≠ k");
    assert_ne!(j, l, "AdagᵢAdagₖAₗAⱼ requires j ≠ l");
    assert!(
        i.site < space.d && j.site < space.d && k.site < space.d && l.site < space.d,
        "orbital site out of range"
    );

    let same_up = i.spin == Spin::Up && k.spin == Spin::Up;
    let same_dn = i.spin == Spin::Dn && k.spin == Spin::Dn;
    let mixed = i.spin != k.spin;
    if (same_up && space.nup < 2)
        || (same_dn && space.ndn < 2)
        || (mixed && (space.nup < 1 || space.ndn < 1))
    {
        return tt_zeros(space);
    }

    build_operator(tt, &[i, k], &[l, j], w)
}

/// `a†_i a†_k a†_m a_n a_l a_j` (three-body). Requires `i.spin==j.spin`, `k.spin==l.spin`,
/// `m.spin==n.spin`, all three creation orbitals pairwise distinct, and all three annihilation
/// orbitals pairwise distinct.
#[allow(clippy::too_many_arguments)]
pub fn adag_i_adag_k_adag_m_a_n_a_l_a_j<T: Scalar>(
    tt: &TtVector<T>,
    i: Orbital,
    j: Orbital,
    k: Orbital,
    l: Orbital,
    m: Orbital,
    n: Orbital,
    w: T,
) -> TtVector<T> {
    let space = tt.space;
    assert_eq!(
        i.spin, j.spin,
        "AdagᵢAdagₖAdagₘAₙAₗAⱼ requires i.spin == j.spin"
    );
    assert_eq!(
        k.spin, l.spin,
        "AdagᵢAdagₖAdagₘAₙAₗAⱼ requires k.spin == l.spin"
    );
    assert_eq!(
        m.spin, n.spin,
        "AdagᵢAdagₖAdagₘAₙAₗAⱼ requires m.spin == n.spin"
    );
    assert_ne!(i, k, "AdagᵢAdagₖAdagₘAₙAₗAⱼ requires i ≠ k");
    assert_ne!(i, m, "AdagᵢAdagₖAdagₘAₙAₗAⱼ requires i ≠ m");
    assert_ne!(k, m, "AdagᵢAdagₖAdagₘAₙAₗAⱼ requires k ≠ m");
    assert_ne!(j, l, "AdagᵢAdagₖAdagₘAₙAₗAⱼ requires j ≠ l");
    assert_ne!(j, n, "AdagᵢAdagₖAdagₘAₙAₗAⱼ requires j ≠ n");
    assert_ne!(l, n, "AdagᵢAdagₖAdagₘAₙAₗAⱼ requires l ≠ n");
    assert!(
        i.site < space.d
            && j.site < space.d
            && k.site < space.d
            && l.site < space.d
            && m.site < space.d
            && n.site < space.d,
        "orbital site out of range"
    );

    let npairs_up = [i, k, m].iter().filter(|o| o.spin == Spin::Up).count();
    let npairs_dn = 3 - npairs_up;
    if space.nup < npairs_up || space.ndn < npairs_dn {
        return tt_zeros(space);
    }

    build_operator(tt, &[i, k, m], &[n, l, j], w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qn::max_ranks;
    use crate::vector::tt_randn;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn capped_randn(space: QnSpace, cap: usize, seed: u64) -> TtVector<f64> {
        let d = space.d;
        let caps = max_ranks(&space);
        let ranks: Vec<RankGrid> = (0..=d)
            .map(|b| {
                let mut r = RankGrid::zeros(&space);
                for s in space.bond_sectors(b) {
                    r[s] = if b == 0 || b == d { 1 } else { cap }.min(caps[b][s]);
                }
                r
            })
            .collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        tt_randn(space, &ranks, &mut rng)
    }

    #[test]
    fn t7_cross_target_moment_readout_is_an_ordinary_dot() {
        let d = 4usize;
        let pg = PointGroup::new(
            2,
            &[Irrep::new(0), Irrep::new(1), Irrep::new(2), Irrep::new(1)],
        );
        let gamma0 = Irrep::IDENTITY;
        let space = QnSpace::with_point_group(d, 2, 1, pg, gamma0);
        let x = capped_randn(space, 2, 0xE400);

        let mut m = CoeffTensor::<f64>::zeros(d, 2);
        m.set(&[0, 1], 0.75);
        m.set(&[1, 0], 0.75);
        let classes = property_flux_classes(&m, pg);
        assert_eq!(classes.len(), 1, "fixture must be single-flux");
        let gamma_mu = classes[0];
        assert_ne!(gamma_mu, Irrep::IDENTITY, "fixture must be cross-irrep");

        let y = apply_cross_target(&x, &m, pg);
        let space_out = y.space;
        assert_eq!(space_out.target, gamma0.fuse(gamma_mu));

        let w = capped_randn(space_out, 2, 0xE401);
        let moment = crate::ops::dot(&w, &y);

        let (xd, wd, yd) = (x.to_dense(), w.to_dense(), y.to_dense());
        let want: f64 = wd.iter().zip(yd.iter()).map(|(a, b)| a * b).sum();
        let scale = wd
            .iter()
            .fold(0.0f64, |a, &v| a.max(v.abs()))
            .max(1.0)
            .max(yd.iter().fold(0.0f64, |a, &v| a.max(v.abs())));
        assert!(
            (moment - want).abs() <= 1e-12 * scale,
            "fused-space dot {moment:e} != dense wᵀ(μ̂x) {want:e}"
        );
        assert!(
            moment.abs() > 1e-6,
            "anti-vacuity: the cross-space moment came out zero"
        );
        assert_ne!(x.space, w.space, "the two targets must genuinely differ");
        assert!(xd.iter().any(|v| v.abs() > 0.0));
    }

    #[test]
    fn t7_cross_target_wrong_caps_is_silently_zero() {
        let d = 4usize;
        let pg = PointGroup::new(
            2,
            &[Irrep::new(0), Irrep::new(1), Irrep::new(2), Irrep::new(1)],
        );
        let space = QnSpace::with_point_group(d, 2, 1, pg, Irrep::IDENTITY);
        let x = capped_randn(space, 2, 0xE400);
        let (p, q) = (0usize, 1usize);
        let gamma_mu = pg.irrep_of(p).fuse(pg.irrep_of(q));
        let space_out =
            QnSpace::with_point_group(d, space.nup, space.ndn, pg, space.target.fuse(gamma_mu));

        let cre = [Orbital::new(p, Spin::Up)];
        let ann = [Orbital::new(q, Spin::Up)];

        let bad_caps = max_ranks(&space);
        let mut st = State {
            fu: 0,
            nlu: 0,
            nru: 1,
            fd: 0,
            nld: 0,
            nrd: 0,
            delta: Irrep::IDENTITY,
        };
        let mut worst = 0.0f64;
        for k in 0..d {
            let (up, dn) = select_site_op(k, &cre, &ann);
            let (c, next) = apply_site_op_into(x.core(k), space_out, Some(&bad_caps), up, dn, st);
            for f in Family::ALL {
                for &s in &space_out.bond_sectors(k) {
                    if let Some(b) = c.block(f, s) {
                        for j in 0..b.ncols() {
                            for i in 0..b.nrows() {
                                worst = worst.max(b[(i, j)].abs());
                            }
                        }
                    }
                }
            }
            st = next;
        }
        assert_eq!(
            worst, 0.0,
            "the input-space caps must prune EVERY path to the fused target (measured largest \
             surviving amplitude {worst:e}) — this is the silent failure the caps rule prevents"
        );

        let good = build_operator_cross_target(&x, &cre, &ann, 1.0, space_out, gamma_mu);
        assert!(
            good.norm() > 1e-6,
            "anti-vacuity: the correctly-capped sweep must not be zero either"
        );
    }
}

#[cfg(test)]
mod phase4_tests {
    use crate::{Orbital, Spin, fermionic_sign};

    fn eps_ref(mut p: Vec<Orbital>) -> i8 {
        let mut sign: i8 = 1;
        let mut remaining = p.len();
        while remaining > 0 {
            let mut pos = 0;
            for i in 1..remaining {
                if p[i] >= p[pos] {
                    pos = i;
                }
            }
            if (remaining - 1 - pos) % 2 == 1 {
                sign = -sign;
            }
            p.remove(pos);
            remaining -= 1;
        }
        sign
    }

    #[test]
    fn t1_perm_sign() {
        let distinct = vec![
            Orbital::new(2, Spin::Up),
            Orbital::new(0, Spin::Dn),
            Orbital::new(4, Spin::Up),
            Orbital::new(1, Spin::Dn),
        ];
        assert_eq!(
            fermionic_sign(&distinct),
            eps_ref(distinct.clone()),
            "all-distinct orbital string"
        );

        let tied = vec![
            Orbital::new(1, Spin::Up),
            Orbital::new(3, Spin::Dn),
            Orbital::new(1, Spin::Up),
            Orbital::new(0, Spin::Dn),
        ];
        assert_eq!(
            fermionic_sign(&tied),
            eps_ref(tied.clone()),
            "tied (equal-orbital) string"
        );

        let i = Orbital::new(2, Spin::Up);
        let j = Orbital::new(4, Spin::Dn);
        assert_eq!(fermionic_sign(&[i, j]), 1, "i<j => +1");
        assert_eq!(fermionic_sign(&[j, i]), -1, "i>j => -1");
        assert_eq!(fermionic_sign(&[i, i]), 1, "i==j (tie) => +1");
    }
}
