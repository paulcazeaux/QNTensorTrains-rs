use super::*;
use crate::bipartite::util::*;

#[test]
fn b_cross_oracle_small_d() {
    let space = QnSpace::from_n_two_sz(6, 4, 0);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x2A11_E1E1);
    let (t, v) = random_coeffs(d, &mut rng);
    let h_bip = build_bipartite(space, &t, &v);
    let h_med = build_median(space, &t, &v);

    for seed in 0..3u64 {
        let mut r2 = Xoshiro256PlusPlus::seed_from_u64(0x5EED ^ seed);
        let x: TtVector<f64> = randomized_state(space, 2, 5, &mut r2);
        let y_bip = h_bip.apply(&x).to_dense();
        let y_med = h_med.apply(&x).to_dense();
        let e = rel_diff(&y_bip, &y_med);
        assert!(
            e < 1e-12,
            "seed {seed}: cross-construction dense outputs differ rel-err {e:.2e}"
        );
        let q_bip = rayleigh_quotient(&h_bip, &x);
        let q_med = rayleigh_quotient(&h_med, &x);
        let rq = (q_bip - q_med).abs() / q_med.abs().max(1e-12);
        assert!(
            rq < 1e-12,
            "seed {seed}: cross-construction Rayleigh quotient differs rel {rq:.2e} \
             (bip {q_bip}, med {q_med})"
        );
    }
}

#[test]
fn b_coo_artifact_bit_identity() {
    let space = QnSpace::from_n_two_sz(6, 4, 0);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB17_1DE);
    let (t, v) = random_coeffs(d, &mut rng);

    let a1 = assemble_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);
    let a2 = assemble_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);
    assert_eq!(
        hash_assembled(&a1),
        hash_assembled(&a2),
        "run-to-run COO artifact hash differs (construction is not seed-free)"
    );

    assert_eq!(a1.coo.len(), a2.coo.len());
    for k in 0..a1.coo.len() {
        let (g1, g2) = (a1.coo[k].iter_grid_order(), a2.coo[k].iter_grid_order());
        assert_eq!(
            g1.len(),
            g2.len(),
            "populated cell count differs at site {k}"
        );
        for (e1, e2) in g1.iter().zip(&g2) {
            let (yf, ysec, xf, xsec, cell1) = *e1;
            let cell2 = e2.4;
            assert_eq!(
                (yf, ysec, xf, xsec),
                (e2.0, e2.1, e2.2, e2.3),
                "populated cell key differs at site {k}"
            );
            assert_eq!(
                cell1.rows, cell2.rows,
                "row mismatch site {k} yf {yf} xf {xf} ysec {ysec:?} xsec {xsec:?}"
            );
            assert_eq!(cell1.cols, cell2.cols, "col mismatch site {k}");
            let b1: Vec<u64> = cell1.vals.iter().map(|x| x.to_bits()).collect();
            let b2: Vec<u64> = cell2.vals.iter().map(|x| x.to_bits()).collect();
            assert_eq!(b1, b2, "val (bits) mismatch site {k}");
        }
    }
}

#[test]
fn b_hx_bit_identity() {
    let space = QnSpace::from_n_two_sz(6, 6, 0);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x77DE);
    let (t, v) = random_coeffs(d, &mut rng);
    let x: TtVector<f64> = randomized_state(space, 1, 6, &mut rng);

    let h1 = build_bipartite(space, &t, &v);
    let y1a = h1.apply(&x);
    let y1b = h1.apply(&x);
    assert_cores_bit_identical(&y1a, &y1b, "same-h apply determinism");

    let h2 = build_bipartite(space, &t, &v);
    let y2 = h2.apply(&x);
    assert_cores_bit_identical(&y1a, &y2, "rebuilt-h apply bit-identity");
}

#[test]
fn b_coo_artifact_bit_identity_pinned() {
    const PINNED_COO_HASH: u64 = 2623915936533678639;
    let space = QnSpace::from_n_two_sz(6, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xF1_1ED);
    let (t, v) = random_coeffs(space.d, &mut rng);
    let a = assemble_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);
    assert_eq!(
        hash_assembled(&a),
        PINNED_COO_HASH,
        "pinned COO hash changed — if intentional, update PINNED_COO_HASH"
    );
}

#[test]
fn m_coo_artifact_bit_identity() {
    let space = QnSpace::from_n_two_sz(6, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB17_1DE);
    let (t, v) = random_coeffs(space.d, &mut rng);
    let a1 = assemble_with(space, &t, &v, None, f64::EPSILON, Construction::Median);
    let a2 = assemble_with(space, &t, &v, None, f64::EPSILON, Construction::Median);
    assert_eq!(
        hash_assembled(&a1),
        hash_assembled(&a2),
        "run-to-run Median COO artifact hash differs (construction is not seed-free)"
    );
}

#[test]
fn m_coo_artifact_bit_identity_pinned() {
    const PINNED_MEDIAN_COO_HASH: u64 = 13630325454864305721;
    let space = QnSpace::from_n_two_sz(6, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xF1_1ED);
    let (t, v) = random_coeffs(space.d, &mut rng);
    let a = assemble_with(space, &t, &v, None, f64::EPSILON, Construction::Median);
    assert_eq!(
        hash_assembled(&a),
        PINNED_MEDIAN_COO_HASH,
        "pinned Median COO hash changed — if intentional, update PINNED_MEDIAN_COO_HASH"
    );
}

#[test]
fn m_construction_thread_count_independent() {
    let space = QnSpace::from_n_two_sz(6, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7147_0EAD);
    let (t, v) = random_coeffs(space.d, &mut rng);
    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| {
                hash_assembled(&assemble_with(
                    space,
                    &t,
                    &v,
                    None,
                    f64::EPSILON,
                    Construction::Median,
                ))
            })
    };
    assert_eq!(
        run(1),
        run(8),
        "Median construction differs under 1- vs 8-thread pools"
    );
}

#[test]
fn b_construction_thread_count_independent() {
    let space = QnSpace::from_n_two_sz(6, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7147_0EAD);
    let (t, v) = random_coeffs(space.d, &mut rng);
    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| {
                hash_assembled(&assemble_with(
                    space,
                    &t,
                    &v,
                    None,
                    f64::EPSILON,
                    Construction::Bipartite,
                ))
            })
    };
    assert_eq!(
        run(1),
        run(8),
        "bipartite construction differs under 1- vs 8-thread pools"
    );
}

#[test]
fn b_matvec_threaded_bit_identical() {
    let space = QnSpace::from_n_two_sz(6, 6, 0);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0004_A7EC);
    let (t, v) = random_coeffs(d, &mut rng);
    let x: TtVector<f64> = randomized_state(space, 1, 6, &mut rng);
    let h = build_bipartite(space, &t, &v);

    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let y_serial = serial.install(|| h.apply(&x));
    let y_parallel = h.apply(&x);
    assert_cores_bit_identical(
        &y_serial,
        &y_parallel,
        "bipartite matvec threaded-vs-serial",
    );
}

#[test]
fn b_als_threaded_matches_serial() {
    let space = QnSpace::from_n_two_sz(8, 4, 0);
    let d = space.d;
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for (p, q, val) in [
        (0, 1, -1.0),
        (1, 2, -0.5),
        (2, 3, -0.7),
        (3, 4, -0.4),
        (4, 5, -0.6),
        (5, 6, -0.3),
        (6, 7, -0.5),
    ] {
        t.set(&[p, q], val);
        t.set(&[q, p], val);
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for (i, j, k, l, val) in [(2, 0, 3, 1, -1.0), (5, 3, 6, 4, -0.5)] {
        v.set(&[i, j, k, l], val);
        v.set(&[k, l, i, j], val);
    }
    let h = build_bipartite(space, &t, &v);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5B0D_0A15);
    let x0: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
    let opts = AlsOptions {
        eps: 1e-6,
        max_iter: 2,
        verbose: false,
    };
    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| als(&h, &x0, &opts).0)
    };
    let l1 = run(1);
    let l8 = run(8);
    assert!(
        (l1 - l8).abs() <= 1e-10 * l1.abs().max(1.0),
        "als on bipartite h: 1-thread λ={l1} vs 8-thread λ={l8}"
    );
}

#[test]
fn b_mals_threaded_matches_serial() {
    let space = QnSpace::from_n_two_sz(8, 4, 0);
    let d = space.d;
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for (p, q, val) in [(0, 1, -1.0), (2, 3, -0.7), (4, 5, -0.6), (6, 7, -0.5)] {
        t.set(&[p, q], val);
        t.set(&[q, p], val);
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for (i, j, k, l, val) in [(2, 0, 3, 1, -1.0), (5, 3, 6, 4, -0.5)] {
        v.set(&[i, j, k, l], val);
        v.set(&[k, l, i, j], val);
    }
    let h = build_bipartite(space, &t, &v);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5B10_0A15);
    let x0: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
    let opts = MalsOptions {
        eps: 1e-6,
        bond_dims: vec![20, 20],
        max_iter: 2,
        verbose: false,
        ..Default::default()
    };
    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| mals(&h, &x0, &opts).0)
    };
    let l1 = run(1);
    let l8 = run(8);
    assert!(
        (l1 - l8).abs() <= 1e-10 * l1.abs().max(1.0),
        "mals on bipartite h: 1-thread λ={l1} vs 8-thread λ={l8}"
    );
}
