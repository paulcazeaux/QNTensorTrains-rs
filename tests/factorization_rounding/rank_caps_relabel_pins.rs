// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::mals::{MalsOptions, mals};
use qntt::solvers::tdvp::{TdvpOptions, tdvp};
use qntt::su2::qn::Su2Space;
use qntt::su2::vector::Su2TtVector;
use qntt::{QnSpace, RoundMode, SparseHamiltonian, TtVector};
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

fn digest(x: &TtVector<f64>) -> u64 {
    ranks_digest(x)
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

fn digest_su2(x: &Su2TtVector<f64>) -> u64 {
    ranks_digest_su2(x)
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

const SEED: u64 = 0x5EED_CA95;

fn space() -> QnSpace {
    QnSpace::new(6, 3, 3)
}

fn su2_spaces() -> [Su2Space; 2] {
    [Su2Space::new(6, 6, 0), Su2Space::new(6, 5, 1)]
}

fn fixture() -> TtVector<f64> {
    randomized_state(space(), 3, 6, &mut seeded(SEED))
}

fn fixture_su2(sp: Su2Space) -> Su2TtVector<f64> {
    randomized_su2_state(sp, 3, 6, &mut seeded(SEED))
}

fn hamiltonian(sp: QnSpace) -> SparseHamiltonian<f64> {
    let (t, v) = generic_truncated_tv(sp.d);
    SparseHamiltonian::new(sp, &t, &v, None, f64::EPSILON)
}

#[test]
fn rcp_round_rmax_abelian() {
    let x = fixture();
    let y = x.round(1e-12, qntt::Pooled(5), RoundMode::Rel);
    let y2 = x.round(1e-12, qntt::Pooled(5), RoundMode::Rel);
    let free = x.round(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    assert_binds("round(rmax=5)", ranks_digest(&y), ranks_digest(&free));
    assert_eq!(digest(&y), digest(&y2), "round is not deterministic");
    pin("round(rmax=5) abelian", digest(&y), 0x3303_dba4_d94c_a461);
}

#[test]
fn rcp_round_global_rmax_abelian() {
    let x = fixture();
    let y = x.round_global(1e-12, qntt::PerSector(2), RoundMode::Rel);
    let y2 = x.round_global(1e-12, qntt::PerSector(2), RoundMode::Rel);
    let free = x.round_global(1e-12, qntt::PerSector::NONE, RoundMode::Rel);
    assert_binds(
        "round_global(rmax=2)",
        ranks_digest(&y),
        ranks_digest(&free),
    );
    assert_eq!(digest(&y), digest(&y2), "round_global is not deterministic");
    pin(
        "round_global(rmax=2) abelian",
        digest(&y),
        0xf01e_40f1_3e85_d0e2,
    );
}

#[test]
fn rcp_round_lr_rmax_abelian() {
    let lr = |rmax: usize| {
        let mut y = fixture();
        y.right_orthogonalize(false);
        y.round_lr_mut(1e-12, qntt::Pooled(rmax), RoundMode::Rel);
        y
    };
    let y = lr(5);
    let y2 = lr(5);
    let free = lr(0);
    assert_binds(
        "round_lr_mut(rmax=5)",
        ranks_digest(&y),
        ranks_digest(&free),
    );
    assert_eq!(digest(&y), digest(&y2), "round_lr_mut is not deterministic");
    pin(
        "round_lr_mut(rmax=5) abelian",
        digest(&y),
        0x9f18_2584_9c38_c0a1,
    );
}

#[test]
fn rcp_round_rmax_su2() {
    const PINS: [u64; 2] = [0x61de_4059_c4fc_7ca1, 0x2019_8753_ae9d_3ce2];
    for (i, sp) in su2_spaces().into_iter().enumerate() {
        let x = fixture_su2(sp);
        let y = x.round(1e-12, qntt::Pooled(5), RoundMode::Rel);
        let y2 = x.round(1e-12, qntt::Pooled(5), RoundMode::Rel);
        let free = x.round(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        assert_binds(
            "su2 round(rmax=5)",
            ranks_digest_su2(&y),
            ranks_digest_su2(&free),
        );
        assert_eq!(digest_su2(&y), digest_su2(&y2));
        pin(
            &format!("su2 round(rmax=5) space {i}"),
            digest_su2(&y),
            PINS[i],
        );
    }
}

#[test]
fn rcp_round_global_rmax_su2() {
    const PINS: [u64; 2] = [0x2920_ac04_a97c_a7e1, 0x177b_5371_b075_04c2];
    for (i, sp) in su2_spaces().into_iter().enumerate() {
        let x = fixture_su2(sp);
        let y = x.round_global(1e-12, qntt::PerSector(2), RoundMode::Rel);
        let y2 = x.round_global(1e-12, qntt::PerSector(2), RoundMode::Rel);
        let free = x.round_global(1e-12, qntt::PerSector::NONE, RoundMode::Rel);
        assert_binds(
            "su2 round_global(rmax=2)",
            ranks_digest_su2(&y),
            ranks_digest_su2(&free),
        );
        assert_eq!(digest_su2(&y), digest_su2(&y2));
        pin(
            &format!("su2 round_global(rmax=2) space {i}"),
            digest_su2(&y),
            PINS[i],
        );
    }
}

#[test]
fn rcp_round_lr_rmax_su2() {
    const PINS: [u64; 2] = [0x956a_aacc_af91_c2e1, 0x2ac2_75b0_0d8e_7760];
    for (i, sp) in su2_spaces().into_iter().enumerate() {
        let lr = |rmax: usize| {
            let mut y = fixture_su2(sp);
            y.right_orthogonalize(false);
            y.round_lr_mut(1e-12, qntt::Pooled(rmax), RoundMode::Rel);
            y
        };
        let y = lr(5);
        let y2 = lr(5);
        let free = lr(0);
        assert_binds(
            "su2 round_lr_mut(rmax=5)",
            ranks_digest_su2(&y),
            ranks_digest_su2(&free),
        );
        assert_eq!(digest_su2(&y), digest_su2(&y2));
        pin(
            &format!("su2 round_lr_mut(rmax=5) space {i}"),
            digest_su2(&y),
            PINS[i],
        );
    }
}

fn mals_start() -> TtVector<f64> {
    let mut x = randomized_state::<f64, _>(space(), 1, 2, &mut seeded(SEED ^ 0x1111));
    let n = x.norm();
    x.scale_mut(1.0 / n);
    x
}

#[test]
fn rcp_mals_bond_dims_abelian() {
    let sp = space();
    let h = hamiltonian(sp);
    let x0 = mals_start();
    let opts = |bd: Vec<usize>| MalsOptions {
        eps: 1e-6,
        bond_dims: bd,
        noise: vec![0.0],
        max_iter: 2,
        conv_tol: Some(0.0),
        ..MalsOptions::default()
    };
    let (e, x) = mals(&h, &x0, &opts(vec![4, 6]));
    let (e2, x2) = mals(&h, &x0, &opts(vec![4, 6]));
    let (_, xf) = mals(&h, &x0, &opts(vec![0]));
    assert_binds("mals bond_dims=[4,6]", ranks_digest(&x), ranks_digest(&xf));
    assert_eq!(digest(&x), digest(&x2), "mals not deterministic");
    assert!(
        (e - e2).abs() <= 1e-12 * e.abs().max(1.0),
        "mals energy not deterministic"
    );
    pin("mals state abelian", digest(&x), 0x9ece_fc6e_7064_4702);
}

#[test]
fn rcp_tdvp_rmax_abelian() {
    let sp = space();
    let h = hamiltonian(sp);
    let x0 = mals_start();
    let opts = |rmax: usize| TdvpOptions {
        dtau: 0.05,
        eps: 0.0,
        trunc: 1e-10,
        max_sweeps: 1,
        chi: qntt::Pooled(rmax),
        ..TdvpOptions::default()
    };
    let (_, x) = tdvp(&h, &x0, &opts(4));
    let (_, x2) = tdvp(&h, &x0, &opts(4));
    let (_, xf) = tdvp(&h, &x0, &opts(0));
    assert_binds("tdvp rmax=4", ranks_digest(&x), ranks_digest(&xf));
    assert_eq!(digest(&x), digest(&x2), "tdvp not deterministic");
    pin("tdvp state abelian", digest(&x), 0x27b5_c305_eec6_b182);
}
