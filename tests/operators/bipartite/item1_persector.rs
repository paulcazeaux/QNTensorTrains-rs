use super::*;
use crate::bipartite::util::*;

#[test]
fn b_apply_dense_oracle_multishape() {
    let shapes = [
        QnSpace::from_n_two_sz(4, 4, 0),
        QnSpace::new(5, 2, 1),
        QnSpace::new(6, 1, 1),
    ];
    for (si, &space) in shapes.iter().enumerate() {
        let d = space.d;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB1_0000 ^ si as u64);
        let (t, v) = random_coeffs(d, &mut rng);

        let h_bip = build_bipartite(space, &t, &v);
        let h_med = build_median(space, &t, &v);

        let tol_cross = (50.0 * d as f64 * f64::EPSILON).max(1e-12);
        for seed in 0..4u64 {
            let mut r2 = Xoshiro256PlusPlus::seed_from_u64(0xDA7A_0000 ^ (si as u64) << 8 ^ seed);
            let x: TtVector<f64> = randomized_state(space, 1, 4, &mut r2);
            let xd = x.to_dense();

            let refv = dense_apply_hamiltonian_12(d, &t, &v, &xd);
            let y_bip = h_bip.apply(&x).to_dense();
            let y_med = h_med.apply(&x).to_dense();

            let e_ref = rel_diff(&y_bip, &refv);
            assert!(
                e_ref < 1e-9,
                "shape {si} (d={d}) seed {seed}: bipartite apply vs dense JW oracle rel-err {e_ref:.2e}"
            );
            let e_cross = rel_diff(&y_bip, &y_med);
            assert!(
                e_cross < tol_cross,
                "shape {si} (d={d}) seed {seed}: bipartite vs median apply rel-err {e_cross:.2e} \
                 (tol {tol_cross:.2e})"
            );
        }
    }
}

#[test]
fn b_apply_dense_oracle_heterogeneous_nl_comp_merge() {
    let space = QnSpace::new(6, 2, 1);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0E7E_0000);
    let (t, v) = random_coeffs(d, &mut rng);

    let blocks = bipartite_probe(space, &t, &v, None, f64::EPSILON);
    let witnessed = blocks.iter().any(|blk| {
        blk.comp_channels.iter().any(|c| {
            c.num_terms >= 2 && (c.distinct_nl_up.len() >= 2 || c.distinct_nl_dn.len() >= 2)
        })
    });
    assert!(
        witnessed,
        "VACUOUS: no heterogeneous-nl Comp merge materialized on this system — perturb the shape \
         / coeffs until the probe witnesses one (the guard is the spec, the system is a starting \
         point)"
    );

    let h_bip = build_bipartite(space, &t, &v);
    let h_med = build_median(space, &t, &v);
    let tol_cross = (50.0 * d as f64 * f64::EPSILON).max(1e-12);
    for seed in 0..4u64 {
        let mut r2 = Xoshiro256PlusPlus::seed_from_u64(0xC0_1DE ^ seed);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut r2);
        let xd = x.to_dense();
        let refv = dense_apply_hamiltonian_12(d, &t, &v, &xd);
        let y_bip = h_bip.apply(&x).to_dense();
        let y_med = h_med.apply(&x).to_dense();
        assert!(
            rel_diff(&y_bip, &refv) < 1e-9,
            "heterogeneous-nl seed {seed}: bipartite vs dense JW oracle rel-err {:.2e}",
            rel_diff(&y_bip, &refv)
        );
        assert!(
            rel_diff(&y_bip, &y_med) < tol_cross,
            "heterogeneous-nl seed {seed}: bipartite vs median rel-err {:.2e}",
            rel_diff(&y_bip, &y_med)
        );
    }
}

#[test]
fn b_persector_dense_oracle_and_cross() {
    let shapes = [
        QnSpace::from_n_two_sz(4, 4, 0),
        QnSpace::new(5, 2, 1),
        QnSpace::new(6, 1, 1),
        QnSpace::new(6, 2, 1),
        QnSpace::from_n_two_sz(6, 4, 0),
    ];
    for (si, &space) in shapes.iter().enumerate() {
        let d = space.d;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDEC0_0000 ^ si as u64);
        let (t, v) = random_coeffs(d, &mut rng);

        if si == 3 {
            let blocks = bipartite_probe(space, &t, &v, None, f64::EPSILON);
            let witnessed = blocks.iter().any(|blk| {
                blk.comp_channels.iter().any(|c| {
                    c.num_terms >= 2 && (c.distinct_nl_up.len() >= 2 || c.distinct_nl_dn.len() >= 2)
                })
            });
            assert!(
                witnessed,
                "VACUOUS: no heterogeneous-nl Comp merge on the (6,2,1) shape"
            );
        }

        let h_ps = build_persector(space, &t, &v);
        let h_med = build_median(space, &t, &v);
        let tol_cross = (50.0 * d as f64 * f64::EPSILON).max(1e-12);
        for seed in 0..4u64 {
            let mut r2 = Xoshiro256PlusPlus::seed_from_u64(0x0D3C_00DE ^ (si as u64) << 8 ^ seed);
            let x: TtVector<f64> = randomized_state(space, 1, 4, &mut r2);
            let xd = x.to_dense();
            let refv = dense_apply_hamiltonian_12(d, &t, &v, &xd);
            let y_ps = h_ps.apply(&x).to_dense();
            let y_med = h_med.apply(&x).to_dense();
            let e_ref = rel_diff(&y_ps, &refv);
            assert!(
                e_ref < 1e-9,
                "shape {si} (d={d}) seed {seed}: per-sector apply vs dense JW oracle rel-err {e_ref:.2e}"
            );
            let e_cross = rel_diff(&y_ps, &y_med);
            assert!(
                e_cross < tol_cross,
                "shape {si} (d={d}) seed {seed}: per-sector vs median apply rel-err {e_cross:.2e} \
                 (tol {tol_cross:.2e})"
            );
        }
    }
}

#[test]
fn b_persector_run_to_run_deterministic() {
    let space = QnSpace::from_n_two_sz(6, 4, 0);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD37_E121);
    let (t, v) = random_coeffs(d, &mut rng);
    let a1 = assemble_with(
        space,
        &t,
        &v,
        None,
        f64::EPSILON,
        Construction::BipartitePerSector,
    );
    let a2 = assemble_with(
        space,
        &t,
        &v,
        None,
        f64::EPSILON,
        Construction::BipartitePerSector,
    );
    assert_eq!(
        hash_assembled(&a1),
        hash_assembled(&a2),
        "decoupled construction is not run-to-run deterministic"
    );
}
