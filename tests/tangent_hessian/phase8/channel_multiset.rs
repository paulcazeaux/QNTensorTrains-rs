// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::fcidump::{PointGroupData, screen_point_group_forbidden};
use qntt::integrals::{Integrals, IntegralsLoad};
use qntt::pointgroup::{Irrep, PointGroup, PointGroupKind};
use qntt::solvers::hess::{
    ChannelGsOpts, ChannelOpts, ChannelSet, ChannelSpectrum, ChannelStatus, ConeOpts, IrrepSel,
    SpinSel, channel_response, channel_response_from_load,
};
use qntt::{CoeffTensor, QnSpace, SparseHamiltonian, TtVector, dot, tt_state};

fn synth_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[Irrep::new(0), Irrep::new(1), Irrep::new(1), Irrep::new(0)],
    )
}

fn synth_tv(pg: PointGroup) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let d = 4usize;
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        t.set(&[p, p], 0.15 * (p as f64 - 0.5 * d as f64));
    }
    t.set(&[0, 3], -0.9);
    t.set(&[3, 0], -0.9);
    t.set(&[1, 2], -1.1);
    t.set(&[2, 1], -1.1);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                v.set(&[p, p, q, q], 0.5 / (1.0 + (p as f64 - q as f64).abs()));
            }
        }
    }
    for idx in [[0, 3, 1, 2], [1, 2, 0, 3], [3, 0, 2, 1], [2, 1, 3, 0]] {
        v.set(&idx, 0.21);
    }
    screen_point_group_forbidden(&mut t, &mut v, pg)
        .expect("the fixture is symmetry-allowed by construction — the screen must be a no-op");
    (t, v)
}

fn exact_spectrum(
    space: QnSpace,
    ham: &SparseHamiltonian<f64>,
    filter: Option<(PointGroup, Irrep)>,
) -> Vec<f64> {
    let d = space.d;
    let states: Vec<TtVector<f64>> = combinations(d, space.nup)
        .into_iter()
        .flat_map(|up| {
            combinations(d, space.ndn)
                .into_iter()
                .map(move |dn| (up.clone(), dn))
        })
        .filter(|(up, dn)| match filter {
            None => true,
            Some((pg, tgt)) => {
                up.iter()
                    .chain(dn.iter())
                    .fold(Irrep::IDENTITY, |ir, &s| ir.fuse(pg.irrep_of(s)))
                    == tgt
            }
        })
        .map(|(up, dn)| {
            let mut occ_up = vec![false; d];
            let mut occ_dn = vec![false; d];
            for &s in &up {
                occ_up[s] = true;
            }
            for &s in &dn {
                occ_dn[s] = true;
            }
            tt_state(&occ_up, &occ_dn)
        })
        .collect();
    let n = states.len();
    if n == 0 {
        return Vec::new();
    }
    let mut hmat = vec![vec![0.0f64; n]; n];
    for (j, sj) in states.iter().enumerate() {
        let hs = ham * sj;
        for (i, si) in states.iter().enumerate() {
            hmat[i][j] = dot(si, &hs);
        }
    }
    sym_eigvals_jacobi(hmat)
}

fn merged_energies<S>(spec: &ChannelSpectrum<S>) -> Vec<f64> {
    let mut e: Vec<f64> = spec
        .levels
        .iter()
        .filter(|l| !l.redundant)
        .map(|l| l.energy)
        .collect();
    e.sort_by(|a, b| a.partial_cmp(b).unwrap());
    e
}

#[test]
fn t8q_channel_multiset_matches_the_trivial_pg_spectrum() {
    let d = 4usize;
    let pg = synth_pg();
    let (t, v) = synth_tv(pg);
    let (nup, ndn) = (2usize, 1usize);

    let plain = QnSpace::new(d, nup, ndn);
    let hplain = SparseHamiltonian::new(plain, &t, &v, None, f64::EPSILON);
    let full = exact_spectrum(plain, &hplain, None);
    let per_irrep: Vec<Vec<f64>> = (0..pg.order())
        .map(|ir| exact_spectrum(plain, &hplain, Some((pg, Irrep::new(ir)))))
        .collect();
    assert_eq!(full.len(), 24, "sector (2,1) at d=4 has 24 determinants");
    assert_eq!(
        (
            per_irrep[0].len(),
            per_irrep[1].len(),
            per_irrep[2].len(),
            per_irrep[3].len()
        ),
        (12, 12, 0, 0),
        "the synthetic group must split (2,1) as 12 + 12 with two structurally empty irreps"
    );

    let base = QnSpace::with_point_group(d, nup, ndn, pg, Irrep::IDENTITY);
    let opts = ChannelOpts {
        channels: ChannelSet {
            irreps: IrrepSel::All,
            spin: SpinSel::RefOnly,
        },
        cone: ConeOpts {
            targets: 11,
            max_added: 40,
            topup_seed: Some(0xC4C0),
            conv_tol: 0.0,
            ..Default::default()
        },
        eps_h: f64::EPSILON,
        ..Default::default()
    };
    let spec = channel_response(&t, &v, base, &opts);

    println!(
        "t8q channels: {} enumerated, reference {}",
        spec.channels.len(),
        spec.reference
    );
    for (i, c) in spec.channels.iter().enumerate() {
        println!(
            "  ch {i}: irrep {} spin_delta {} status {:?} e_gs {:?} e0 {:?}",
            c.label.irrep.index(),
            c.label.spin_delta,
            c.status,
            c.e_gs,
            c.cone.as_ref().map(|r| r.e0)
        );
    }

    assert_eq!(
        spec.channels.len(),
        pg.order() as usize,
        "IrrepSel::All must enumerate every irrep in 0..pg.order(), empties included"
    );
    assert_eq!(
        spec.reference, 0,
        "the reference channel is enumerated FIRST so every later gap is immediately computable"
    );
    assert_eq!(
        (
            spec.channels[0].label.irrep,
            spec.channels[0].label.spin_delta
        ),
        (Irrep::IDENTITY, 0),
        "channel 0 must be the base space's own (target irrep, reference spin sector)"
    );

    for ir in [2usize, 3] {
        let c = &spec.channels[ir];
        assert_eq!(
            c.status,
            ChannelStatus::Empty,
            "irrep {ir} is max_ranks-empty at bond d and must be reported Empty, not solved"
        );
        assert!(
            c.cone.is_none() && c.e_gs.is_none(),
            "irrep {ir}: an empty channel carries no solve"
        );
    }

    for ir in [0usize, 1] {
        let c = &spec.channels[ir];
        assert_eq!(
            c.status,
            ChannelStatus::Ok,
            "irrep {ir}: {:?} — the channel solve did not complete",
            c.status
        );
        let cone = c
            .cone
            .as_ref()
            .expect("an Ok channel carries a cone readout");
        assert_eq!(
            cone.energies.len(),
            12,
            "irrep {ir}: targets = 11 must report all 12 levels of a 12-determinant channel"
        );
        for (j, (got, want)) in cone.energies.iter().zip(per_irrep[ir].iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-8,
                "irrep {ir} level {j}: channel CI {got} vs the determinant-filtered exact {want} \
                 — at full augmentation the channel CI is exact (measured floor ~1e-14)"
            );
        }
    }

    let merged = merged_energies(&spec);
    assert_eq!(
        merged.len(),
        full.len(),
        "the union of the channel spectra must have as many levels as the trivial-PG sector"
    );
    for (j, (got, want)) in merged.iter().zip(full.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-8,
            "merged level {j}: {got} vs the trivial-PG exact {want} — the per-channel composition \
             is not the full spectrum"
        );
    }

    let e_ref0 = spec.channels[spec.reference]
        .cone
        .as_ref()
        .expect("the reference channel solved")
        .e0;
    for l in &spec.levels {
        let gap = l
            .gap
            .expect("the reference channel solved, so every gap is Some");
        assert!(
            (gap - (l.energy - e_ref0)).abs() <= 1e-10,
            "level (ch {}, k {}): gap {gap} != energy − E_ref0 ({})",
            l.channel,
            l.k,
            l.energy - e_ref0
        );
    }

    let ref_top = spec
        .levels
        .iter()
        .filter(|l| l.channel == spec.reference)
        .map(|l| l.energy)
        .fold(f64::NEG_INFINITY, f64::max);
    let other_below = spec
        .levels
        .iter()
        .any(|l| l.channel != spec.reference && l.energy < ref_top);
    assert!(
        other_below,
        "no level from a non-reference channel lies below the reference channel's highest \
         reported level ({ref_top}) — the composition would be answering nothing LR could not"
    );
}

#[test]
fn t8q_spin_channel_reaches_the_partner_sector_ground() {
    let d = 4usize;
    let space22 = QnSpace::new(d, 2, 2);
    let ham22 = generic_truncated_h::<f64>(space22);
    let e22 = exact_config_spectrum(space22, &ham22);
    let space31 = QnSpace::new(d, 3, 1);
    let ham31 = generic_truncated_h::<f64>(space31);
    let e31 = exact_config_spectrum(space31, &ham31);
    println!(
        "t8q spin: (2,2) dim {} ground {:.12}, (3,1) dim {} ground {:.12}",
        e22.len(),
        e22[0],
        e31.len(),
        e31[0]
    );
    assert_eq!((e22.len(), e31.len()), (36, 16), "fixture sector sizes");

    let (t, v) = generic_truncated_tv(d);
    let opts = ChannelOpts {
        channels: ChannelSet {
            irreps: IrrepSel::RefOnly,
            spin: SpinSel::Raised(1),
        },
        cone: ConeOpts {
            targets: 4,
            max_added: 40,
            topup_seed: Some(0xC4C1),
            conv_tol: 0.0,
            ..Default::default()
        },
        eps_h: f64::EPSILON,
        ..Default::default()
    };
    let spec = channel_response(&t, &v, space22, &opts);
    for (i, c) in spec.channels.iter().enumerate() {
        println!(
            "  ch {i}: ({},{}) spin_delta {} status {:?} e_gs {:?} e0 {:?}",
            c.space.unwrap().nup,
            c.space.unwrap().ndn,
            c.label.spin_delta,
            c.status,
            c.e_gs,
            c.cone.as_ref().map(|r| r.e0)
        );
    }

    assert_eq!(
        spec.channels.len(),
        2,
        "RefOnly × Raised(1) is the reference channel plus one spin-raised partner"
    );
    assert_eq!(spec.reference, 0);
    let sector = |i: usize| {
        let sp = spec.channels[i]
            .space
            .expect("a feasible channel carries its typed space");
        (sp.nup, sp.ndn)
    };
    assert_eq!(sector(0), (2, 2), "channel 0 is the reference sector");
    assert_eq!(
        sector(1),
        (3, 1),
        "Raised(1) from (2,2) is the partner sector (nup + 1, ndn − 1)"
    );
    assert_eq!(spec.channels[1].label.spin_delta, 1);

    assert!(
        spec.levels
            .iter()
            .all(|l| l.redundant == (spec.channels[l.channel].label.spin_delta > 0)),
        "on the abelian layer exactly the spin-raised channels' levels are redundant copies"
    );
    for l in spec.levels.iter().filter(|l| l.redundant) {
        assert!(
            e22.iter()
                .any(|e| (e - l.energy).abs() < 1e-7 * e.abs().max(1.0)),
            "level {:.12} of the (3,1) channel is marked redundant but has NO degenerate partner \
             in the reference sector's exact spectrum — the redundancy claim would be false",
            l.energy
        );
    }

    for (i, want) in [(0usize, e22[0]), (1, e31[0])] {
        let c = &spec.channels[i];
        assert_eq!(c.status, ChannelStatus::Ok, "channel {i}: {:?}", c.status);
        let cone = c
            .cone
            .as_ref()
            .expect("an Ok channel carries a cone readout");
        assert!(
            (cone.e0 - want).abs() < 1e-8,
            "channel {i}: relaxed-reference CI ground {} vs the exact sector ground {want}",
            cone.e0
        );
        let e_gs = c
            .e_gs
            .expect("an Ok channel reports its fixed-rank ground energy");
        assert!(
            e_gs >= want - 1e-10 && e_gs - want < 1e-2,
            "channel {i}: e_gs {e_gs} is not a sane fixed-rank upper bound on {want}"
        );
    }

    let gap =
        spec.channels[1].cone.as_ref().unwrap().e0 - spec.channels[0].cone.as_ref().unwrap().e0;
    println!(
        "t8q spin: partner-channel gap {gap:.9} (exact {:.9})",
        e31[0] - e22[0]
    );
    assert!(
        (gap - (e31[0] - e22[0])).abs() < 1e-8,
        "the spin-channel gap {gap} does not match the exact sector-ground difference {}",
        e31[0] - e22[0]
    );
}

fn residued_load() -> (IntegralsLoad<f64>, PointGroup) {
    let d = 4usize;
    let pg = synth_pg();
    let (mut t, v) = synth_tv(pg);
    t.set(&[0, 1], 5e-14);
    t.set(&[1, 0], 5e-14);
    assert!(
        t.get(&[0, 1]).abs() > f64::EPSILON,
        "the injected residue must survive the operator's own eps screen, else the fixture is \
         vacuous"
    );
    let load = IntegralsLoad {
        ints: Integrals {
            t,
            v,
            e0: 0.0,
            norb: d,
            nelec: 3,
            ms2: 1,
            orbsym: Vec::new(),
            isym: None,
        },
        w: None,
        meta: None,
        pg: PointGroupData {
            kind: PointGroupKind::C2v,
            pg,
            target: Irrep::IDENTITY,
            guessed: false,
        },
        space: QnSpace::with_point_group(d, 2, 1, pg, Irrep::IDENTITY),
    };
    (load, pg)
}

fn one_channel_opts() -> ChannelOpts {
    ChannelOpts {
        channels: ChannelSet {
            irreps: IrrepSel::RefOnly,
            spin: SpinSel::RefOnly,
        },
        gs: ChannelGsOpts {
            chi: qntt::Pooled(3),
            ..Default::default()
        },
        cone: ConeOpts {
            targets: 2,
            max_added: 4,
            conv_tol: 0.0,
            ..Default::default()
        },
        eps_h: f64::EPSILON,
        ..Default::default()
    }
}

#[test]
fn t8q_channel_response_from_load_screens_its_input() {
    let (load, pg) = residued_load();
    let opts = one_channel_opts();
    let spec = channel_response_from_load(&load, &opts);

    let (mut ts, mut vs) = (load.ints.t.clone(), load.ints.v.clone());
    screen_point_group_forbidden(&mut ts, &mut vs, pg)
        .expect("a 5e-14 residue is screened, not refused");
    assert_eq!(
        ts.get(&[0, 1]),
        0.0,
        "the screen must have removed the forbidden entry"
    );
    let control = channel_response(&ts, &vs, load.space, &opts);

    let got = spec.channels[0]
        .cone
        .as_ref()
        .expect("the reference channel solved");
    let want = control.channels[0]
        .cone
        .as_ref()
        .expect("the control channel solved");
    println!(
        "t8q from_load: e0 {:.12} vs hand-screened {:.12} (added {} / {})",
        got.e0, want.e0, got.added, want.added
    );
    assert_eq!(
        got.energies, want.energies,
        "the wrapper's screen must be the caller's screen — same data, same numbers"
    );
    assert_eq!(
        spec.channels[0].label.name.as_deref(),
        Some("A1"),
        "ChannelOpts::kind is filled from the load, so irreps come out named"
    );
}

#[test]
#[should_panic]
fn t8q_unscreened_integrals_kill_the_raw_entry_point() {
    let (load, _pg) = residued_load();
    let _ = channel_response(&load.ints.t, &load.ints.v, load.space, &one_channel_opts());
}

#[test]
fn t8q_channel_reports_a_non_critical_ground_state() {
    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let opts = ChannelOpts {
        channels: ChannelSet {
            irreps: IrrepSel::RefOnly,
            spin: SpinSel::RefOnly,
        },
        gs: ChannelGsOpts {
            chi: qntt::Pooled(3),
            polish_restarts: 0,
            grad_accept: 1e-30,
            ..Default::default()
        },
        cone: ConeOpts {
            targets: 1,
            max_added: 2,
            conv_tol: 0.0,
            ..Default::default()
        },
        eps_h: f64::EPSILON,
        ..Default::default()
    };
    let spec = channel_response(&t, &v, QnSpace::new(d, 2, 2), &opts);
    let c = &spec.channels[0];
    println!(
        "t8q not-critical: status {:?} grad {:?} e0 {:?}",
        c.status,
        c.grad_norm,
        c.cone.as_ref().map(|r| r.e0)
    );
    match c.status {
        ChannelStatus::GsNotCritical { grad_norm } => {
            assert_eq!(
                Some(grad_norm),
                c.grad_norm,
                "the status must carry the SAME gradient the result reports"
            );
            assert!(
                grad_norm > 1e-30 && grad_norm.is_finite(),
                "the reported gradient {grad_norm:.3e} does not exceed the acceptance it failed"
            );
        }
        ref other => panic!(
            "grad_accept = 1e-30 is unreachable by any polish, so the channel must be reported \
             GsNotCritical; got {other:?}"
        ),
    }
    assert!(
        c.cone.is_some() && !spec.levels.is_empty(),
        "a non-critical ground state is a CAVEAT, not a failure: the readout is still reported"
    );
}

#[test]
#[should_panic(expected = "the loaded integrals are not compatible")]
fn t8q_channel_response_from_load_refuses_genuine_mismatch() {
    let (mut load, _pg) = residued_load();
    load.ints.t.set(&[0, 1], 0.5);
    load.ints.t.set(&[1, 0], 0.5);
    let _ = channel_response_from_load(&load, &one_channel_opts());
}
