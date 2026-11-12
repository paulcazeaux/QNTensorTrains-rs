use super::*;
use crate::bipartite::util::*;

fn build_cover_switch(
    space: QnSpace,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
) -> SparseHamiltonian<f64> {
    SparseHamiltonian::new_with(
        space,
        t,
        v,
        None,
        f64::EPSILON,
        Construction::MedianCoverSwitch,
    )
}

fn build_bip_median_switch(
    space: QnSpace,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
) -> SparseHamiltonian<f64> {
    SparseHamiltonian::new_with(
        space,
        t,
        v,
        None,
        f64::EPSILON,
        Construction::BipartiteMedianSwitch,
    )
}

#[test]
fn r_switch_arms_dense_oracle_and_cross() {
    let shapes = [
        QnSpace::from_n_two_sz(4, 4, 0),
        QnSpace::new(5, 2, 1),
        QnSpace::new(6, 1, 1),
        QnSpace::new(6, 2, 1),
    ];
    for (si, &space) in shapes.iter().enumerate() {
        let d = space.d;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5171_C400u64.wrapping_add(si as u64));
        let (t, v) = random_coeffs(d, &mut rng);
        let h_med = build_median(space, &t, &v);
        let arms = [
            ("median+cover", build_cover_switch(space, &t, &v)),
            ("bipartite+median", build_bip_median_switch(space, &t, &v)),
        ];
        let tol_cross = (50.0 * d as f64 * f64::EPSILON).max(1e-12);
        for seed in 0..4u64 {
            let mut r2 = Xoshiro256PlusPlus::seed_from_u64(0x5EED_0000 ^ (si as u64) << 8 ^ seed);
            let x: TtVector<f64> = randomized_state(space, 1, 4, &mut r2);
            let xd = x.to_dense();
            let refv = dense_apply_hamiltonian_12(d, &t, &v, &xd);
            let y_med = h_med.apply(&x).to_dense();
            for (name, h) in &arms {
                let y = h.apply(&x).to_dense();
                let e_ref = rel_diff(&y, &refv);
                assert!(
                    e_ref < 1e-9,
                    "{name}, shape {si} (d={d}) seed {seed}: apply vs dense JW oracle \
                     rel-err {e_ref:.2e}"
                );
                let e_cross = rel_diff(&y, &y_med);
                assert!(
                    e_cross < tol_cross,
                    "{name}, shape {si} (d={d}) seed {seed}: vs median apply rel-err \
                     {e_cross:.2e} (tol {tol_cross:.2e})"
                );
            }
        }
    }
}

#[test]
fn r_cover_switch_is_non_vacuous_and_measured() {
    use qntt::hamiltonian::assembly::switch_profile;
    let space = QnSpace::from_n_two_sz(8, 4, 0);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5177_C4E5);
    let (t, v) = random_coeffs(d, &mut rng);
    let prof = switch_profile(space, &t, &v, None, f64::EPSILON);
    let live: Vec<_> = prof.iter().filter(|p| p.cover_bond.is_some()).collect();
    assert!(!live.is_empty(), "no terms survived the sweep");
    let stats = |pick: fn(&&&qntt::hamiltonian::assembly::TermSwitch) -> Option<usize>| {
        let differ = live
            .iter()
            .filter(|p| pick(p) != Some(p.median_bond))
            .count();
        let outside = live
            .iter()
            .filter(|p| pick(p).is_some_and(|c| c < p.window.0 || c > p.window.1))
            .count();
        (differ, outside)
    };
    let (differ_bip, outside_bip) = stats(|p| p.cover_bond);
    let (differ_med, outside_med) = stats(|p| p.median_cover_bond);
    let cross = live
        .iter()
        .filter(|p| p.cover_bond != p.median_cover_bond)
        .count();
    let pct = |n: usize| 100.0 * n as f64 / live.len() as f64;
    println!(
        "SWITCH_PROFILE d={d} terms={}\n  bipartite-identity optimum: differs from positional \
         median on {differ_bip} ({:.1}%), outside balance window {outside_bip} ({:.1}%)\n  \
         median-identity optimum:    differs from positional median on {differ_med} ({:.1}%), \
         outside balance window {outside_med} ({:.1}%)\n  the two optima disagree on {cross} \
         ({:.1}%)",
        live.len(),
        pct(differ_bip),
        pct(outside_bip),
        pct(differ_med),
        pct(outside_med),
        pct(cross)
    );
    assert!(
        differ_med > 0,
        "VACUOUS: the median-identity cover reproduced the positional median switch on every \
         term — the reconciliation arm would be a no-op"
    );
}

#[test]
fn r_median_label_overspecification() {
    use qntt::hamiltonian::assembly::ChannelDesc;
    use std::collections::BTreeSet;

    for (name, space, t, v) in [
        {
            let d =
                read_fcidump::<f64>(Path::new("testdata/h4.FCIDUMP"), TwoBodySymmetry::Eightfold)
                    .expect("read h4 FCIDUMP");
            (
                "H4",
                QnSpace::from_n_two_sz(d.norb, d.nelec, d.ms2),
                d.t,
                d.v,
            )
        },
        {
            let d = read_fcidump::<f64>(
                Path::new("testdata/h2o.FCIDUMP"),
                TwoBodySymmetry::Eightfold,
            )
            .expect("read h2o FCIDUMP");
            (
                "H2O",
                QnSpace::from_n_two_sz(d.norb, d.nelec, d.ms2),
                d.t,
                d.v,
            )
        },
    ] {
        let a = assemble_with(space, &t, &v, None, f64::EPSILON, Construction::Median);
        let mut worst = (0usize, 0usize, 0usize);
        for b in 0..=space.d {
            let mut full = BTreeSet::new();
            let mut proj = BTreeSet::new();
            for s in space.bond_sectors(b) {
                for c in &a.states[b][s] {
                    let ChannelDesc::Median(st, open) = c else {
                        unreachable!("median arm")
                    };
                    full.insert((*st, *open));
                    proj.insert(((st.up.flux, st.dn.flux), *open));
                }
            }
            if full.len() > worst.1 {
                worst = (b, full.len(), proj.len());
            }
        }
        let (b, full, proj) = worst;
        println!(
            "OVERSPEC {name}: peak bond {b} — median distinct labels {full}, \
             with (nl,nr) projected away {proj}  ({:.2}× splitting)",
            full as f64 / proj as f64
        );
        assert!(
            proj < full,
            "{name}: projecting (nl,nr) away merged nothing — the tag would not be \
             over-specification on this system"
        );
    }
}
