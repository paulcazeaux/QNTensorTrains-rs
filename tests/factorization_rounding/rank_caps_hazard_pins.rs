// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::gad::{GadOpts, RankAdaptMode, RankAdaptOpts, gad};
use qntt::solvers::hess::{ChannelGsOpts, channel_ground_state};
use qntt::solvers::mals::NoiseType;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::Su2Space;
use qntt::su2::vector::Su2TtVector;
use qntt::{Pooled, QnSpace, SparseHamiltonian, TtVector};
use qntt_testkit::generic_truncated_tv;
use qntt_testkit::{randomized_state, randomized_su2_state, seeded};

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn word(&mut self, w: u64) {
        for b in w.to_le_bytes() {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn get(self) -> u64 {
        self.0
    }
}

fn ranks_digest(x: &TtVector<f64>) -> u64 {
    let mut h = Fnv::new();
    for (b, g) in x.ranks.iter().enumerate() {
        h.word(b as u64);
        for (s, r) in g.iter() {
            h.word(s.nup as u64);
            h.word(s.ndn as u64);
            h.word(*r as u64);
        }
    }
    h.get()
}

fn ranks_digest_su2(x: &Su2TtVector<f64>) -> u64 {
    let mut h = Fnv::new();
    for (b, g) in x.ranks.iter().enumerate() {
        h.word(b as u64);
        for (s, r) in g.iter() {
            h.word(s.n as u64);
            h.word(s.two_s as u64);
            h.word(*r as u64);
        }
    }
    h.get()
}

#[track_caller]
fn pin(name: &str, got: u64, want: u64) {
    assert_eq!(
        got, want,
        "pin `{name}` moved: got {got:#018x}, expected {want:#018x}"
    );
}

#[track_caller]
fn assert_binds(name: &str, capped: u64, uncapped: u64) {
    assert_ne!(
        capped, uncapped,
        "cap in `{name}` does not bind on this fixture (capped ranks == uncapped ranks) — \
         the pin would be vacuous"
    );
}

#[track_caller]
fn assert_repeats(name: &str, a: u64, b: u64) {
    assert_eq!(a, b, "`{name}` is not repeatable at a fixed seed");
}

const SEED: u64 = 0x5EED_CA95;

fn space() -> QnSpace {
    QnSpace::new(6, 3, 3)
}

fn su2_space() -> Su2Space {
    Su2Space::new(8, 6, 0)
}

fn hamiltonian(sp: QnSpace) -> SparseHamiltonian<f64> {
    let (t, v) = generic_truncated_tv(sp.d);
    SparseHamiltonian::new(sp, &t, &v, None, f64::EPSILON)
}

fn hamiltonian_su2(sp: Su2Space) -> Su2SparseHamiltonian<f64> {
    let (t, v) = generic_truncated_tv(sp.d);
    Su2SparseHamiltonian::new(sp, &t, &v, None, f64::EPSILON)
}

fn unit_state() -> TtVector<f64> {
    let mut x = randomized_state::<f64, _>(space(), 2, 4, &mut seeded(SEED));
    let n = x.norm();
    x.scale_mut(1.0 / n);
    x
}

fn unit_state_su2() -> Su2TtVector<f64> {
    let mut x = randomized_su2_state::<f64, _>(su2_space(), 2, 4, &mut seeded(SEED));
    let n = x.norm();
    x.scale_mut(1.0 / n);
    x
}

#[test]
fn rchp_gad_rank_adapt_chi_abelian() {
    let h = hamiltonian(space());
    let x0 = unit_state();
    let run = |chi: usize| {
        let opts = GadOpts {
            gradtol: 1e-8,
            max_steps: 6,
            k_davidson: 4,
            certify: false,
            probe_budget: 20,
            preserve_grid: true,
            rank_adapt: Some(RankAdaptOpts {
                mode: RankAdaptMode::Both,
                chi: Pooled(chi),
                round_eps: 1e-10,
                period: 1,
            }),
            ..GadOpts::default()
        };
        gad(&h, &x0, None, &opts).0
    };
    let x = run(49);
    assert_binds(
        "gad rank_adapt chi=49",
        ranks_digest(&x),
        ranks_digest(&run(0)),
    );
    assert_repeats(
        "gad rank_adapt chi=49",
        ranks_digest(&x),
        ranks_digest(&run(49)),
    );
    pin(
        "gad rank_adapt chi=49 abelian",
        ranks_digest(&x),
        0xea00_0a8c_462e_358f,
    );
}

#[test]
fn rchp_gad_rank_adapt_chi_su2() {
    let h = hamiltonian_su2(su2_space());
    let x0 = unit_state_su2();
    let run = |chi: usize| {
        let opts = GadOpts {
            gradtol: 1e-8,
            max_steps: 6,
            k_davidson: 4,
            certify: false,
            probe_budget: 20,
            preserve_grid: true,
            rank_adapt: Some(RankAdaptOpts {
                mode: RankAdaptMode::Both,
                chi: Pooled(chi),
                round_eps: 1e-10,
                period: 1,
            }),
            ..GadOpts::default()
        };
        qntt::su2::hess::gad(&h, &x0, None, &opts).0
    };
    let x = run(31);
    assert_binds(
        "su2 gad rank_adapt chi=31",
        ranks_digest_su2(&x),
        ranks_digest_su2(&run(0)),
    );
    assert_repeats(
        "su2 gad rank_adapt chi=31",
        ranks_digest_su2(&x),
        ranks_digest_su2(&run(31)),
    );
    pin(
        "gad rank_adapt chi=31 su2",
        ranks_digest_su2(&x),
        0x3cef_1f27_c358_2caa,
    );
}

#[test]
fn rchp_channel_gs_chi_abelian() {
    let sp = space();
    let h = hamiltonian(sp);
    let run = |chi: usize| {
        let gs = ChannelGsOpts {
            chi: Pooled(chi),
            eps: 1e-10,
            noise: vec![0.0],
            noise_type: NoiseType::Perturbative,
            max_iter: 3,
            seed: 0xC4A1,
            polish_gradtol: 1e-9,
            polish_restarts: 1,
            ..ChannelGsOpts::default()
        };
        channel_ground_state(&h, sp, &gs).0
    };
    let x = run(4);
    assert_binds("channel_gs chi=4", ranks_digest(&x), ranks_digest(&run(0)));
    assert_repeats("channel_gs chi=4", ranks_digest(&x), ranks_digest(&run(4)));
    pin(
        "channel_gs chi=4 abelian",
        ranks_digest(&x),
        0x0499_a08f_a2f7_ae02,
    );
}

#[test]
fn rchp_channel_gs_chi_su2() {
    let sp = su2_space();
    let h = hamiltonian_su2(sp);
    let run = |chi: usize| {
        let gs = ChannelGsOpts {
            chi: Pooled(chi),
            eps: 1e-10,
            noise: vec![0.0],
            noise_type: NoiseType::Perturbative,
            max_iter: 5,
            seed: 0xC4A1,
            polish_gradtol: 1e-9,
            polish_restarts: 1,
            ..ChannelGsOpts::default()
        };
        qntt::su2::hess::channel_ground_state(&h, sp, &gs).0
    };
    let x = run(4);
    assert_binds(
        "su2 channel_gs chi=4",
        ranks_digest_su2(&x),
        ranks_digest_su2(&run(0)),
    );
    assert_repeats(
        "su2 channel_gs chi=4",
        ranks_digest_su2(&x),
        ranks_digest_su2(&run(4)),
    );
    pin(
        "channel_gs chi=4 su2",
        ranks_digest_su2(&x),
        0xb348_aaf5_9990_c94d,
    );
}
