use std::sync::atomic::Ordering;

use super::{GAMMA2_MIRROR, gamma2_with, generalized_gamma2, set_gamma2_mirror};
use crate::qn::QnSpace;
use crate::test_util::seeded;
use crate::vector::TtVector;

fn capped(space: QnSpace, lo: usize, hi: usize, seed: u64) -> TtVector<f64> {
    use rand::RngExt;
    let mut rng = seeded(seed);
    let caps = crate::qn::max_ranks(&space);
    let d = space.d;
    let ranks: Vec<crate::qn::RankGrid> = (0..=d)
        .map(|b| {
            let mut r = crate::qn::RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                let draw = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
                r[s] = draw.min(caps[b][s]);
            }
            r
        })
        .collect();
    crate::vector::tt_randn(space, &ranks, &mut rng)
}

fn is_truncated(x: &TtVector<f64>) -> bool {
    let caps = crate::qn::max_ranks(&x.space);
    (0..=x.space.d).any(|b| {
        x.space
            .bond_sectors(b)
            .into_iter()
            .any(|s| x.ranks[b][s] < caps[b][s])
    })
}

struct Fixture {
    label: &'static str,
    x: TtVector<f64>,
    spinful: bool,
}

fn fixtures() -> Vec<Fixture> {
    let mk = |label, d, nup, ndn, lo, hi, seed, truncated: bool| {
        let x = capped(QnSpace::new(d, nup, ndn), lo, hi, seed);
        if truncated {
            assert!(
                is_truncated(&x),
                "{label}: must sit strictly below max_ranks"
            );
        }
        Fixture {
            label,
            x,
            spinful: nup > 0 && ndn > 0,
        }
    };
    vec![
        mk("d=1 (1,1)", 1, 1, 1, 8, 8, 0xF4_0101, false),
        mk("d=2 (1,0)", 2, 1, 0, 8, 8, 0xF4_0201, false),
        mk("d=2 (1,1)", 2, 1, 1, 8, 8, 0xF4_0202, false),
        mk("d=3 (1,0)", 3, 1, 0, 8, 8, 0xF4_0301, false),
        mk("d=3 (1,1)", 3, 1, 1, 8, 8, 0xF4_0302, false),
        mk("d=4 (2,0)", 4, 2, 0, 1, 2, 0xF4_0401, false),
        mk("d=4 (0,2)", 4, 0, 2, 1, 2, 0xF4_0402, false),
        mk("d=4 (2,1) truncated", 4, 2, 1, 1, 2, 0xF4_0403, true),
        mk("d=5 (3,1) truncated", 5, 3, 1, 1, 2, 0xF4_0501, true),
        mk("d=6 (3,0) truncated", 6, 3, 0, 1, 2, 0xF4_0601, true),
    ]
}

fn idx(d: usize, i: usize, j: usize, k: usize, l: usize) -> usize {
    ((i * d + j) * d + k) * d + l
}

fn unflat(d: usize, n: usize) -> (usize, usize, usize, usize) {
    (n / (d * d * d), (n / (d * d)) % d, (n / d) % d, n % d)
}

fn mirror_of(d: usize, n: usize) -> usize {
    let (i, j, k, l) = unflat(d, n);
    idx(d, l, k, j, i)
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, a| m.max(a.abs()))
}

fn apply_e(v: &[f64], k: usize, l: usize, spin: usize) -> Vec<f64> {
    let (pk, pl) = (2 * k + spin, 2 * l + spin);
    let mut y = vec![0.0f64; v.len()];
    for (n, &c) in v.iter().enumerate() {
        if c == 0.0 || (n >> pl) & 1 == 0 {
            continue;
        }
        let n1 = n ^ (1 << pl);
        if (n1 >> pk) & 1 == 1 {
            continue;
        }
        let parity = (n & ((1 << pl) - 1)).count_ones() + (n1 & ((1 << pk) - 1)).count_ones();
        y[n1 | (1 << pk)] += if parity % 2 == 0 { c } else { -c };
    }
    y
}

struct DenseGamma2 {
    total: Vec<f64>,
    parts: [[Vec<f64>; 2]; 2],
}

fn dense_gamma2(x: &TtVector<f64>) -> DenseGamma2 {
    let d = x.space.d;
    let v = x.to_dense();
    let ey: Vec<Vec<Vec<f64>>> = (0..2)
        .map(|s| {
            (0..d * d)
                .map(|kl| apply_e(&v, kl / d, kl % d, s))
                .collect()
        })
        .collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
    let n4 = d.pow(4);
    let mut parts: [[Vec<f64>; 2]; 2] = Default::default();
    for (sb, row) in parts.iter_mut().enumerate() {
        for (sk, part) in row.iter_mut().enumerate() {
            *part = (0..n4)
                .map(|e| {
                    let (i, j, k, l) = unflat(d, e);
                    dot(&ey[sb][j * d + i], &ey[sk][k * d + l])
                })
                .collect();
        }
    }
    let total = (0..n4)
        .map(|e| parts[0][0][e] + parts[0][1][e] + parts[1][0][e] + parts[1][1][e])
        .collect();
    DenseGamma2 { total, parts }
}

fn assert_vs_dense(got: &[f64], dense: &[f64], d: usize, what: &str) {
    let scale = max_abs(dense).max(1e-30);
    let (worst, at) = got
        .iter()
        .zip(dense)
        .enumerate()
        .map(|(n, (a, b))| ((a - b).abs(), n))
        .fold((0.0f64, 0usize), |w, v| if v.0 > w.0 { v } else { w });
    eprintln!("{what}: vs dense worst rel {:e}", worst / scale);
    assert!(
        worst <= 1e-12 * scale,
        "{what}: worst |walk − dense| = {worst:e} at {:?} (scale {scale:e}): walk {:e}, dense {:e}",
        unflat(d, at),
        got[at],
        dense[at]
    );
}

#[test]
fn both_modes_match_the_dense_oracle() {
    for f in fixtures() {
        let d = f.x.space.d;
        let dense = dense_gamma2(&f.x);
        assert!(
            dense.total.iter().any(|v| *v != 0.0),
            "{}: all-zero dense Γ₂ (vacuous)",
            f.label
        );
        let full = gamma2_with(&f.x, false, false);
        let mir = gamma2_with(&f.x, false, true);
        assert_vs_dense(&full, &dense.total, d, &format!("{} Full", f.label));
        assert_vs_dense(&mir, &dense.total, d, &format!("{} Mirrored", f.label));

        if f.spinful {
            let scale = max_abs(&dense.total);
            let (mut off, mut diag) = (0.0f64, 0.0f64);
            for i in 0..d {
                for j in 0..d {
                    let e = idx(d, i, j, j, i);
                    assert_eq!(
                        dense.parts[0][1][e].to_bits(),
                        dense.parts[1][0][e].to_bits(),
                        "{}: c_UD == c_DU at self-mirror {:?} (real inner product)",
                        f.label,
                        (i, j, j, i)
                    );
                    let c = dense.parts[0][1][e].abs();
                    if i == j {
                        diag = diag.max(c);
                    } else {
                        off = off.max(c);
                    }
                }
            }
            eprintln!(
                "{}: cross-spin self-mirror parts (rel): off-diagonal {:e} (weight-2 term), \
                 diagonal {:e} (∅ cross-spin term)",
                f.label,
                off / scale,
                diag / scale
            );
            if d >= 2 {
                assert!(
                    off >= 1e-6 * scale,
                    "{}: no off-diagonal self-mirror cross-spin content — the weight-2 rule \
                     would be untested",
                    f.label
                );
            }
            assert!(
                diag >= 1e-6 * scale,
                "{}: no diagonal ∅ cross-spin content — the ∅ weight would be untested",
                f.label
            );
        }
    }
}

#[test]
fn mirrored_matches_full_within_roundoff() {
    let mut spinful_differing = 0usize;
    for f in fixtures() {
        let full = gamma2_with(&f.x, false, false);
        let mir = gamma2_with(&f.x, false, true);
        assert_eq!(mir.len(), full.len(), "{}: length", f.label);
        let scale = max_abs(&full);
        assert!(
            scale > 0.0,
            "{}: all-zero Full reference (vacuous)",
            f.label
        );
        let (worst, at) = mir
            .iter()
            .zip(&full)
            .enumerate()
            .map(|(n, (a, b))| ((a - b).abs(), n))
            .fold((0.0f64, 0usize), |w, v| if v.0 > w.0 { v } else { w });
        let differing = mir
            .iter()
            .zip(&full)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        if f.spinful {
            spinful_differing += differing;
        }
        let rel = worst / scale;
        eprintln!(
            "{}: Mirrored vs Full max|Δ| {worst:e}, rel {rel:e}, margin {:.3} of the 1e-13 bound; \
             {differing} of {} cells differ in bits",
            f.label,
            rel / 1e-13,
            full.len()
        );
        assert!(
            worst <= 1e-13 * scale,
            "{}: max|Mirrored − Full| = {worst:e} at {:?} (scale {scale:e}): Mirrored {:e}, \
             Full {:e}",
            f.label,
            unflat(f.x.space.d, at),
            mir[at],
            full[at]
        );
    }
    assert!(
        spinful_differing > 0,
        "Mirrored and Full agree bit for bit on every spinful fixture — the Mirrored branch \
         looks never taken"
    );
}

#[test]
fn mirrored_output_is_exactly_symmetric_and_pins_the_forced_cells() {
    for f in fixtures() {
        let d = f.x.space.d;
        let full = gamma2_with(&f.x, false, false);
        let mir = gamma2_with(&f.x, false, true);
        for e in 0..mir.len() {
            let me = mirror_of(d, e);
            assert_eq!(
                mir[e].to_bits(),
                mir[me].to_bits(),
                "{}: Mirrored Γ₂{:?} = {:e} vs its mirror Γ₂{:?} = {:e} — not bitwise symmetric",
                f.label,
                unflat(d, e),
                mir[e],
                unflat(d, me),
                mir[me]
            );
            assert!(
                !(mir[e] == 0.0 && mir[e].is_sign_negative()),
                "{}: Mirrored cell {:?} is −0.0",
                f.label,
                unflat(d, e)
            );
        }
        for m in 0..d {
            let e = idx(d, m, m, m, m);
            assert_eq!(
                mir[e].to_bits(),
                full[e].to_bits(),
                "{}: ∅ cell {:?}: Mirrored {:e} vs Full {:e} — only ∅ lands here, unchanged",
                f.label,
                (m, m, m, m),
                mir[e],
                full[e]
            );
        }
        if !f.spinful {
            for e in 0..mir.len() {
                let me = mirror_of(d, e);
                let b = mir[e].to_bits();
                assert!(
                    b == full[e].to_bits() || b == full[me].to_bits(),
                    "{}: one-channel cell {:?}: Mirrored {:e} is neither Full[e] = {:e} nor \
                     Full[μe] = {:e} bitwise",
                    f.label,
                    unflat(d, e),
                    mir[e],
                    full[e],
                    full[me]
                );
            }
        }
        if d == 1 {
            for (e, (a, b)) in mir.iter().zip(&full).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{}: cell {e}: d = 1 has only ∅, so the modes must agree bitwise",
                    f.label
                );
            }
        }
    }
}

#[test]
fn gamma2_mirror_switch_defaults_on_and_routes_the_public_entry() {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_gamma2_mirror(self.0);
        }
    }
    let _lock = crate::par::lock_par_switches();
    assert!(
        GAMMA2_MIRROR.load(Ordering::Relaxed),
        "GAMMA2_MIRROR must default to true (the Mirrored walk is the default)"
    );
    let _restore = Restore(true);

    let x = capped(QnSpace::new(5, 3, 1), 1, 2, 0xF4_0502);
    assert!(
        is_truncated(&x),
        "d=5 (3,1): must sit strictly below max_ranks"
    );
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("rayon pool");
    let want_full = gamma2_with(&x, false, false);
    let want_mir = gamma2_with(&x, false, true);
    let mut expect_prev = true;
    for (mode, want) in [(false, &want_full), (true, &want_mir), (false, &want_full)] {
        assert_eq!(
            set_gamma2_mirror(mode),
            expect_prev,
            "set_gamma2_mirror must return the previous value"
        );
        expect_prev = mode;
        assert_eq!(
            GAMMA2_MIRROR.load(Ordering::Relaxed),
            mode,
            "switch not set"
        );
        let got = pool1.install(|| generalized_gamma2(&x));
        let bad = got
            .iter()
            .zip(want.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            bad, 0,
            "public entry with GAMMA2_MIRROR = {mode}: {bad} cells differ from \
             gamma2_with(x, false, {mode})"
        );
    }
    let differing = want_full
        .iter()
        .zip(&want_mir)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!("switch test: Full and Mirrored differ in {differing} cells");
    assert!(
        differing > 0,
        "Full and Mirrored bitwise identical on a spinful truncated state — the routing check \
         cannot tell the modes apart"
    );
}
