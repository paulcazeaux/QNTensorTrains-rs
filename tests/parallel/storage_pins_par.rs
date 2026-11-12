// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::TtTangent;
use qntt::qn::QnSpace;
use qntt::solvers::InnerProductVector;
use qntt::su2::qn::Su2Space;
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::Su2TtVector;
use qntt::{Irrep, PointGroup, TtVector};
use qntt_testkit::{randomized_state, randomized_su2_state};

use crate::lock;
use crate::vector_algebra_par::{coeffs, parallel_arm, seeded, serial_arm, verbs};

const RECORDING_FAMILIES: [&str; 2] = ["aarch64", "x86_64-amd"];

const ROWS: &[(&str, u64)] = &[
    ("aarch64/ab_c1_cores", 0xCC3560CA24E2408B),
    ("aarch64/ab_c1_tangent", 0xA86F970D20F7DE95),
    ("aarch64/ab_g4_cores", 0x3BFD830EF2811BA4),
    ("aarch64/ab_g4_tangent", 0xF9262C25D8211257),
    ("aarch64/su2_g2_cores", 0x40791DD6ECB37B15),
    ("aarch64/su2_g2_tangent", 0x610875117A8C1E18),
    ("aarch64/su2_s0_cores", 0xC4630BF2E6CA5F3C),
    ("aarch64/su2_s0_tangent", 0x9B75503EF268E242),
    ("aarch64/su2_s1_cores", 0xD5B0C56BE1578564),
    ("aarch64/su2_s1_tangent", 0x33765B8D46F6EA6F),
    ("x86_64-amd/ab_c1_cores", 0xCC3560CA24E2408B),
    ("x86_64-amd/ab_c1_tangent", 0xE7C84416CBF803C4),
    ("x86_64-amd/ab_g4_cores", 0x3BFD830EF2811BA4),
    ("x86_64-amd/ab_g4_tangent", 0x27F407CC9C34458C),
    ("x86_64-amd/su2_g2_cores", 0x40791DD6ECB37B15),
    ("x86_64-amd/su2_g2_tangent", 0x04DD69088276C6C9),
    ("x86_64-amd/su2_s0_cores", 0xC4630BF2E6CA5F3C),
    ("x86_64-amd/su2_s0_tangent", 0x765D928426A6705E),
    ("x86_64-amd/su2_s1_cores", 0xD5B0C56BE1578564),
    ("x86_64-amd/su2_s1_tangent", 0xE0586BA23CA22A24),
];

struct Fold(u64);

impl Fold {
    fn new() -> Self {
        Fold(0x6A09_E667_F3BC_C908)
    }
    fn word(&mut self, w: u64) {
        let mut z = (self.0.rotate_left(23) ^ w).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        self.0 = z ^ (z >> 31);
    }
    fn floats(&mut self, v: &[f64]) {
        self.word(v.len() as u64);
        for x in v {
            self.word(x.to_bits());
        }
    }
    fn get(self) -> u64 {
        self.0
    }
    fn verbs(&mut self, (d, n, a, s): (f64, f64, Vec<f64>, Vec<f64>)) {
        self.word(d.to_bits());
        self.word(n.to_bits());
        self.floats(&a);
        self.floats(&s);
    }
}

fn family() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse4a") {
            "x86_64-amd"
        } else {
            "x86_64-intel"
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::env::consts::ARCH
    }
}

struct Pins {
    failures: Vec<String>,
    recording: bool,
}

impl Pins {
    fn check(&mut self, output: &str, got: u64) {
        let fam = family();
        let key = format!("{fam}/{output}");
        match ROWS.iter().find(|(k, _)| *k == key) {
            Some(&(_, want)) => {
                println!("{key}: 0x{got:016X} (want 0x{want:016X})");
                if got != want {
                    self.failures.push(format!(
                        "{key}: digest 0x{got:016X} ≠ the recorded 0x{want:016X}"
                    ));
                }
            }
            None => {
                println!("RECORD (\"{key}\", 0x{got:016X}),");
                self.recording |= RECORDING_FAMILIES.contains(&fam);
            }
        }
    }

    fn finish(self) {
        assert!(
            self.failures.is_empty(),
            "parallel storage digests moved:\n{}",
            self.failures.join("\n")
        );
        assert!(
            !self.recording,
            "RECORD: paste the printed rows into ROWS (only rows that reproduce across two runs, \
             on the dense-grid storage)"
        );
    }
}

fn fold_both<V: InnerProductVector<Scalar = f64> + Send + Sync>(f: &mut Fold, u: &V, v: &V) {
    f.verbs(parallel_arm(8, || verbs(u, v)));
    f.verbs(serial_arm(|| verbs(u, v)));
}

fn pg(nbits: u8, d: usize) -> PointGroup {
    let order = 1u8 << nbits;
    let irreps: Vec<Irrep> = (0..d)
        .map(|k| Irrep::new((k as u8 * 3 + 1) % order))
        .collect();
    PointGroup::new(nbits, &irreps)
}

#[test]
fn storage_pins_parallel_arms() {
    let _g = lock();
    let mut pins = Pins {
        failures: Vec::new(),
        recording: false,
    };

    let g4 = pg(2, 6);
    let target = g4.irrep_of_sites(0..2).fuse(g4.irrep_of_sites(0..3));
    for (tag, space) in [
        ("c1", QnSpace::new(6, 3, 2)),
        ("g4", QnSpace::with_point_group(6, 2, 3, g4, target)),
    ] {
        let mut rng = seeded(0x5711_0A11 ^ space.pg.order() as u64);
        let x: TtVector<f64> = randomized_state(space, 0, 3, &mut rng);
        let y: TtVector<f64> = qntt::tt_randn(space, &x.ranks.clone(), &mut rng);
        let mut f = Fold::new();
        for k in 0..space.d {
            fold_both(&mut f, x.core(k), y.core(k));
        }
        pins.check(&format!("ab_{tag}_cores"), f.get());

        let (mut bl, mut br) = (x.clone(), x.clone());
        bl.left_orthogonalize(true);
        br.right_orthogonalize(true);
        let d1: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let d2: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let (u, v) = (
            TtTangent::project(&bl, &br, &d1),
            TtTangent::project(&bl, &br, &d2),
        );
        let mut f = Fold::new();
        fold_both(&mut f, &u, &v);
        let mut a = u.clone();
        parallel_arm(8, || a.axpby(0.375, &v, -1.25));
        f.floats(&coeffs(&a));
        pins.check(&format!("ab_{tag}_tangent"), f.get());
    }

    for (tag, space) in [
        ("s0", Su2Space::new(6, 4, 0)),
        ("s1", Su2Space::new(5, 3, 1)),
        (
            "g2",
            Su2Space::with_point_group(6, 4, 0, pg(1, 6), Irrep::IDENTITY),
        ),
    ] {
        let mut rng = seeded(0x5712_57A7 ^ (space.d as u64) << 8 ^ space.two_s as u64);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 0, 3, &mut rng);
        let y: Su2TtVector<f64> = qntt::su2::vector::tt_randn(space, &x.ranks.clone(), &mut rng);
        let mut f = Fold::new();
        for k in 0..space.d {
            fold_both(&mut f, x.core(k), y.core(k));
        }
        pins.check(&format!("su2_{tag}_cores"), f.get());

        let (mut bl, mut br) = (x.clone(), x.clone());
        bl.left_orthogonalize(true);
        br.right_orthogonalize(true);
        let d1: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let d2: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let (u, v) = (
            Su2TtTangent::project(&bl, &br, &d1),
            Su2TtTangent::project(&bl, &br, &d2),
        );
        let mut f = Fold::new();
        fold_both(&mut f, &u, &v);
        let mut a = u.clone();
        parallel_arm(8, || a.axpby(0.375, &v, -1.25));
        f.floats(&coeffs(&a));
        pins.check(&format!("su2_{tag}_tangent"), f.get());
    }
    pins.finish();
}
