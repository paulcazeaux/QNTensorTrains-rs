use super::*;
use crate::bipartite::util::*;

fn dense_rank(nrows: usize, ncols: usize, rowmajor: &[f64], rel_tol: f64) -> usize {
    if nrows == 0 || ncols == 0 {
        return 0;
    }
    assert_eq!(rowmajor.len(), nrows * ncols, "coeff matrix shape mismatch");
    let mut m = faer::Mat::<f64>::zeros(nrows, ncols);
    for r in 0..nrows {
        for c in 0..ncols {
            *m.get_mut(r, c) = rowmajor[r * ncols + c];
        }
    }
    let svals = m.singular_values().expect("svd");
    let smax = svals.first().copied().unwrap_or(0.0);
    if smax <= 0.0 {
        return 0;
    }
    svals.iter().filter(|&&s| s > rel_tol * smax).count()
}

fn matching_equals_rank_on(space: QnSpace, t: &CoeffTensor<f64>, v: &CoeffTensor<f64>) {
    let d = space.d;
    let blocks: Vec<BlockStat<f64>> = bipartite_probe(space, t, v, None, f64::EPSILON);
    assert!(!blocks.is_empty(), "probe returned no blocks");
    let mut nontrivial = 0usize;
    let mut boundary_seen = 0usize;
    for blk in &blocks {
        if blk.bond == 0 || blk.bond == d {
            boundary_seen += 1;
            assert!(
                blk.matching_size <= 1,
                "boundary bond {} block has matching_size {} > 1",
                blk.bond,
                blk.matching_size
            );
            continue;
        }
        let rank = dense_rank(blk.coeff_nrows, blk.coeff_ncols, &blk.coeff_rowmajor, 1e-9);
        assert_eq!(
            blk.matching_size,
            rank,
            "bond {} sector {:?} flux ({},{}): matching_size {} != coeff rank {} \
             (coeff {}x{})",
            blk.bond,
            blk.sector,
            blk.flux_up,
            blk.flux_dn,
            blk.matching_size,
            rank,
            blk.coeff_nrows,
            blk.coeff_ncols
        );
        if blk.matching_size >= 2 {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial > 0,
        "vacuous: no interior block had matching_size ≥ 2 (rank equality proves nothing)"
    );
    assert!(
        boundary_seen > 0,
        "vacuous: no boundary-bond (0 or d) block was ever probed — the boundary-collapse \
         assertion above never ran"
    );
}

#[test]
fn b_matching_size_equals_coeff_rank() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4A_11C11);
    let (t4, v4) = random_coeffs(4, &mut rng);
    matching_equals_rank_on(QnSpace::from_n_two_sz(4, 4, 0), &t4, &v4);

    let (t6, v6) = random_coeffs(6, &mut rng);
    matching_equals_rank_on(QnSpace::new(6, 1, 1), &t6, &v6);

    let (t6b, v6b) = random_coeffs(6, &mut rng);
    matching_equals_rank_on(QnSpace::new(6, 2, 1), &t6b, &v6b);
}

#[test]
fn b_bond_dim_le_median_elementwise() {
    let space = QnSpace::from_n_two_sz(8, 4, 0);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB0AD_D115);
    let (t, v) = random_coeffs(d, &mut rng);
    let bip = build_bipartite(space, &t, &v).automaton_bond_dims();
    let med = build_median(space, &t, &v).automaton_bond_dims();
    assert_eq!(bip.len(), med.len());
    for (b, (&c, &o)) in bip.iter().zip(&med).enumerate() {
        assert!(c <= o, "bond {b}: bipartite {c} > median {o}");
    }
    let (sb, sm): (usize, usize) = (bip.iter().sum(), med.iter().sum());
    println!(
        "BOND_DIM median max={} sum={sm} profile={med:?}",
        med.iter().max().unwrap()
    );
    println!(
        "BOND_DIM bipartite max={} sum={sb} profile={bip:?}",
        bip.iter().max().unwrap()
    );
    assert!(
        sb < sm,
        "no total bond-dim reduction (median {sm} -> bipartite {sb})"
    );
}

#[test]
fn b_bond_dim_h4_target_profile() {
    let path = Path::new("testdata/h4.FCIDUMP");
    let data = read_fcidump::<f64>(path, TwoBodySymmetry::Eightfold).expect("read h4 FCIDUMP");
    let space = QnSpace::from_n_two_sz(data.norb, data.nelec, data.ms2);
    let ps = build_persector(space, &data.t, &data.v);
    let ps_max = *ps.automaton_bond_dims().iter().max().unwrap();
    let ps_cmax = *ps.automaton_channel_counts().iter().max().unwrap();
    println!("H4 per-sector: sector-summed max={ps_max}  distinct max={ps_cmax}");
    assert!(
        ps_max < 336,
        "H4 per-sector max {ps_max} not below median 336"
    );
    assert!(
        (58..=90).contains(&ps_cmax),
        "H4 per-sector distinct-channel max {ps_cmax} outside block2-convention band 58..=90"
    );

    let bip = build_bipartite(space, &data.t, &data.v);
    let bd = bip.automaton_bond_dims();
    let cc = bip.automaton_channel_counts();
    let max = *bd.iter().max().unwrap();
    let cmax = *cc.iter().max().unwrap();
    println!(
        "H4 per-flux (default): sector-summed max={max} profile={bd:?}  distinct max={cmax} profile={cc:?}"
    );
    assert!(max < 336, "H4 per-flux max {max} not below median 336");
    assert!(
        cmax <= ps_cmax,
        "H4 per-flux distinct-channel max {cmax} exceeds per-sector {ps_cmax} (per-flux should merge ≥ per-sector)"
    );
}

#[test]
fn b_debug_asserts_pass_on_valid_build() {
    let space = QnSpace::new(6, 2, 1);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDEB6_A55E);
    let (t, v) = random_coeffs(d, &mut rng);
    let _ = assemble_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);
}
