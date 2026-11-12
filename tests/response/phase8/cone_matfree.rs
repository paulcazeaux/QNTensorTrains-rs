// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::QnSpace;
use qntt::solvers::hess::{ConeOpts, MatfreeOpts, cone_response, cone_response_matfree};

#[test]
fn t8s_matfree_matches_dense_and_exact_at_full_augmentation() {
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let evals = exact_config_spectrum(space, &ham);

    let opts = ConeOpts {
        targets: 4,
        max_added: 3,
        topup_seed: Some(0x8E82),
        ..Default::default()
    };
    let dense = cone_response(&ham, &x, &opts);
    let mf = cone_response_matfree(&ham, &x, &opts, &MatfreeOpts::default());

    assert_eq!(
        dense.added, mf.added,
        "selection trajectories diverged (added)"
    );
    assert_eq!(
        dense.rounds, mf.rounds,
        "selection trajectories diverged (rounds)"
    );
    for (i, (a, b)) in dense.energies.iter().zip(mf.energies.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-8 * a.abs().max(1.0),
            "energy {i}: dense {a} vs matfree {b} (calibrated floor ~4e-13)"
        );
    }
    for (k, (g, e)) in mf
        .gaps
        .iter()
        .zip(evals[1..].iter().map(|e| e - evals[0]))
        .enumerate()
    {
        assert!(
            (g - e).abs() < 1e-8 * e.max(1.0),
            "gap {k}: matfree {g} vs exact {e}"
        );
    }
    assert!(
        (mf.e0 - evals[0]).abs() < 1e-8,
        "matfree relaxed reference must reach E0 at full augmentation"
    );
    assert_eq!(mf.guards.gram, 0.0);
}

#[test]
fn t8s_matfree_matches_dense_targeted() {
    let space = QnSpace::new(5, 2, 2);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);

    let opts = ConeOpts {
        targets: 6,
        max_added: 8,
        ..Default::default()
    };
    let dense = cone_response(&ham, &x, &opts);
    let mf = cone_response_matfree(&ham, &x, &opts, &MatfreeOpts::default());

    assert_eq!(dense.seeded, mf.seeded);
    assert_eq!(
        dense.added, mf.added,
        "selection trajectories diverged (added)"
    );
    assert_eq!(
        dense.rounds, mf.rounds,
        "selection trajectories diverged (rounds)"
    );
    for (i, (a, b)) in dense.energies.iter().zip(mf.energies.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-8 * a.abs().max(1.0),
            "energy {i}: dense {a} vs matfree {b}"
        );
    }
    for (i, (ca, cb)) in dense.character.iter().zip(mf.character.iter()).enumerate() {
        for k in 0..3 {
            assert!(
                (ca[k] - cb[k]).abs() < 1e-6,
                "character[{i}][{k}]: dense {} vs matfree {}",
                ca[k],
                cb[k]
            );
        }
    }
}

#[test]
fn t8s_matfree_matches_dense_molecular() {
    let (_space, ham, x, _rng) = lih_sto3g_ghost_fixture();

    let opts = ConeOpts {
        targets: 4,
        max_added: 8,
        ..Default::default()
    };
    let dense = cone_response(&ham, &x, &opts);
    let mf = cone_response_matfree(&ham, &x, &opts, &MatfreeOpts::default());

    assert_eq!(dense.added, mf.added);
    assert_eq!(dense.rounds, mf.rounds);
    for (i, (a, b)) in dense.energies.iter().zip(mf.energies.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-8 * a.abs().max(1.0),
            "energy {i}: dense {a} vs matfree {b}"
        );
    }
    for (i, (ca, cb)) in dense.character.iter().zip(mf.character.iter()).enumerate() {
        for k in 0..3 {
            assert!(
                (ca[k] - cb[k]).abs() < 1e-6,
                "character[{i}][{k}]: dense {} vs matfree {}",
                ca[k],
                cb[k]
            );
        }
    }
}

#[test]
#[should_panic(expected = "gram_check requires the dense path")]
fn t8s_matfree_rejects_gram_check() {
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let opts = ConeOpts {
        targets: 2,
        max_added: 2,
        gram_check: true,
        ..Default::default()
    };
    let _ = cone_response_matfree(&ham, &x, &opts, &MatfreeOpts::default());
}

#[test]
fn t8s_matfree_stats_are_self_consistent() {
    use qntt::solvers::hess::cone_response_matfree_with_stats;

    let space = QnSpace::new(5, 2, 2);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let opts = ConeOpts {
        targets: 6,
        max_added: 8,
        ..Default::default()
    };
    let (res, st) = cone_response_matfree_with_stats(&ham, &x, &opts, &MatfreeOpts::default());

    assert_eq!(
        st.applies, st.lanczos_iters,
        "one apply per Lanczos iteration"
    );
    assert_eq!(
        st.round_iters.iter().sum::<usize>(),
        st.lanczos_iters,
        "per-solve iteration counts must partition the total"
    );
    assert_eq!(st.round_iters.len(), res.rounds + 1);
    assert!(st.restarts >= 1, "at least one fill restart per solve");
    assert!(st.dense_equivalent_applies > res.added);
    assert!(
        st.t_h_apply.as_nanos() > 0 && st.t_assemble_tangent.as_nanos() > 0,
        "stage timers must be populated on a matfree run"
    );
}

#[test]
fn t8s_stage_stats_are_self_consistent_on_both_paths() {
    use qntt::solvers::hess::{cone_response_matfree_with_stats, cone_response_with_stats};

    let space = QnSpace::new(5, 2, 2);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let opts = ConeOpts {
        targets: 6,
        max_added: 8,
        ..Default::default()
    };

    let (dres, dst) = cone_response_with_stats(&ham, &x, &opts);
    let (mres, mfs) = cone_response_matfree_with_stats(&ham, &x, &opts, &MatfreeOpts::default());
    let mst = &mfs.stages;

    for (tag, res, st) in [("dense", &dres, &dst), ("matfree", &mres, mst)] {
        assert!(st.n_tangent > 0, "{tag}: tangent block must be nonempty");
        assert_eq!(
            st.n_accept_ok, res.added,
            "{tag}: accepted normals must equal ConeResponse::added"
        );
        assert!(
            st.n_accept_calls >= st.n_accept_ok,
            "{tag}: every acceptance is also a call"
        );
        assert!(
            st.n_cand >= res.rounds,
            "{tag}: each round emits at least one candidate"
        );
        assert!(
            st.t_tangent_basis.as_nanos() > 0 && st.t_iota.as_nanos() > 0,
            "{tag}: basis and iota stages must be timed"
        );
        assert!(
            st.t_cand_hv.as_nanos() > 0,
            "{tag}: the targeted generator's H*v must be timed"
        );
        assert!(
            st.accounted() <= st.t_total,
            "{tag}: accounted {:?} exceeds total {:?} — the stage timers overlap",
            st.accounted(),
            st.t_total
        );
        assert_eq!(
            st.unaccounted(),
            st.t_total - st.accounted(),
            "{tag}: unaccounted must be the exact residual"
        );
        assert!(
            st.max_rank_hv >= st.max_rank_cand && st.max_rank_cand > 0,
            "{tag}: unrounded H*v rank {} must dominate the rounded candidate rank {}",
            st.max_rank_hv,
            st.max_rank_cand
        );
    }

    assert!(
        dst.t_reference_block.as_nanos() > 0 && dst.t_xrow.as_nanos() == 0,
        "dense builds the reference block and no x-row"
    );
    assert!(
        mst.t_xrow.as_nanos() > 0 && mst.t_reference_block.as_nanos() == 0,
        "matfree builds the x-row and no reference block"
    );
    assert_eq!(
        dst.n_tangent, mst.n_tangent,
        "same point, same tangent block"
    );
    assert_eq!(
        dst.bytes_iota, mst.bytes_iota,
        "same point, same iota footprint"
    );
    assert!(
        dst.bytes_refs > dst.bytes_iota,
        "refs duplicates iota and adds x: {} vs {}",
        dst.bytes_refs,
        dst.bytes_iota
    );
}
