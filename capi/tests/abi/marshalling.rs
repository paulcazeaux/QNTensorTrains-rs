// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn bond_sectors_match_the_core_enumeration() {
    let space = QnSpace::new(6, 3, 3);
    for b in 0..=space.d {
        let expect = space.bond_sectors(b);

        let mut len = 0usize;
        assert_eq!(
            unsafe {
                qntt_space_bond_sectors(
                    space.d,
                    space.nup,
                    space.ndn,
                    b,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    0,
                    &mut len,
                )
            },
            QNTT_OK
        );
        assert_eq!(len, expect.len());

        let (mut a, mut c) = (vec![0usize; len], vec![0usize; len]);
        assert_eq!(
            unsafe {
                qntt_space_bond_sectors(
                    space.d,
                    space.nup,
                    space.ndn,
                    b,
                    a.as_mut_ptr(),
                    c.as_mut_ptr(),
                    len,
                    &mut len,
                )
            },
            QNTT_OK
        );
        for (i, s) in expect.iter().enumerate() {
            assert_eq!((a[i], c[i]), (s.nup, s.ndn), "bond {b}, position {i}");
        }
    }
}

#[test]
fn space_max_ranks_matches_qn_max_ranks_at_trivial_group() {
    for &(d, nup, ndn) in &[(4usize, 2usize, 1usize), (6, 3, 2), (8, 4, 3)] {
        let space = QnSpace::new(d, nup, ndn);
        let want = qntt::max_ranks(&space);
        let want_flat = flat_ranks(&space, &want);

        let mut len = 0usize;
        assert_eq!(
            unsafe { qntt_space_max_ranks(d, nup, ndn, ptr::null_mut(), 0, &mut len) },
            QNTT_OK
        );
        assert_eq!(len, want_flat.len());
        let mut flat = vec![0usize; len];
        assert_eq!(
            unsafe { qntt_space_max_ranks(d, nup, ndn, flat.as_mut_ptr(), flat.len(), &mut len) },
            QNTT_OK
        );
        assert_eq!(
            flat, want_flat,
            "d={d} nup={nup} ndn={ndn}: ABI max_ranks must match qntt::max_ranks"
        );
    }
}

#[test]
fn rank_grids_round_trip_through_the_flat_layout() {
    let space = QnSpace::new(6, 3, 3);
    let ranks = uniform_ranks(space, 3);
    let flat = flat_ranks(&space, &ranks);

    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_from_ranks(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                &mut x,
            )
        },
        QNTT_OK
    );

    let mut len = 0usize;
    assert_eq!(
        unsafe { qntt_f64_vec_ranks(x, ptr::null_mut(), 0, &mut len) },
        QNTT_OK
    );
    assert_eq!(len, flat.len());
    let mut back = vec![0usize; len];
    assert_eq!(
        unsafe { qntt_f64_vec_ranks(x, back.as_mut_ptr(), len, &mut len) },
        QNTT_OK
    );
    assert_eq!(
        back, flat,
        "rank grids must survive the round trip verbatim"
    );

    unsafe { qntt_f64_vec_free(x) };
}

#[test]
fn block_round_trip_reproduces_the_train() {
    let space = QnSpace::new(6, 3, 3);
    let ranks = uniform_ranks(space, 2);
    let flat = flat_ranks(&space, &ranks);

    let mut src: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                42,
                &mut src,
            )
        },
        QNTT_OK
    );

    let mut dst: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_from_ranks(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                &mut dst,
            )
        },
        QNTT_OK
    );

    let mut copied = 0usize;
    for site in 0..space.d {
        for family in 0..4i32 {
            for s in space.bond_sectors(site) {
                let (mut rows, mut cols) = (0usize, 0usize);
                assert_eq!(
                    unsafe {
                        qntt_f64_vec_block_get(
                            src,
                            site,
                            family,
                            s.nup,
                            s.ndn,
                            ptr::null_mut(),
                            0,
                            &mut rows,
                            &mut cols,
                        )
                    },
                    QNTT_OK
                );
                if rows * cols == 0 {
                    continue;
                }
                let mut buf = vec![0.0f64; rows * cols];
                assert_eq!(
                    unsafe {
                        qntt_f64_vec_block_get(
                            src,
                            site,
                            family,
                            s.nup,
                            s.ndn,
                            buf.as_mut_ptr(),
                            buf.len(),
                            &mut rows,
                            &mut cols,
                        )
                    },
                    QNTT_OK
                );
                assert_eq!(
                    unsafe {
                        qntt_f64_vec_block_set(
                            dst,
                            site,
                            family,
                            s.nup,
                            s.ndn,
                            buf.as_ptr(),
                            rows,
                            cols,
                        )
                    },
                    QNTT_OK
                );
                copied += 1;
            }
        }
    }
    assert!(copied > 0, "the fixture should have blocks to copy");

    let mut diff: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_vec_add(src, dst, -1, &mut diff) },
        QNTT_OK
    );
    let (mut nd, mut ns) = (0.0f64, 0.0f64);
    assert_eq!(unsafe { qntt_f64_vec_norm(diff, &mut nd) }, QNTT_OK);
    assert_eq!(unsafe { qntt_f64_vec_norm(src, &mut ns) }, QNTT_OK);
    assert!(
        nd <= 1e-14 * ns.max(1.0),
        "block round trip changed the train: ‖Δ‖ = {nd:e}, ‖x‖ = {ns:e}"
    );

    for h in [src, dst, diff] {
        unsafe { qntt_f64_vec_free(h) };
    }
}

#[test]
fn block_set_rejects_a_shape_mismatch() {
    let space = QnSpace::new(4, 2, 2);
    let ranks = uniform_ranks(space, 1);
    let flat = flat_ranks(&space, &ranks);
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_from_ranks(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                &mut x,
            )
        },
        QNTT_OK
    );

    let mut found = None;
    'outer: for site in 0..space.d {
        for family in 0..4i32 {
            for s in space.bond_sectors(site) {
                let (mut rows, mut cols) = (0usize, 0usize);
                unsafe {
                    qntt_f64_vec_block_get(
                        x,
                        site,
                        family,
                        s.nup,
                        s.ndn,
                        ptr::null_mut(),
                        0,
                        &mut rows,
                        &mut cols,
                    )
                };
                if rows * cols > 0 {
                    found = Some((site, family, s, rows, cols));
                    break 'outer;
                }
            }
        }
    }
    let (site, family, s, rows, cols) = found.expect("rank-1 train has blocks");
    let bogus = vec![1.0f64; rows * cols + 1];
    let status = unsafe {
        qntt_f64_vec_block_set(
            x,
            site,
            family,
            s.nup,
            s.ndn,
            bogus.as_ptr(),
            rows + 1,
            cols,
        )
    };
    assert_eq!(status, QNTT_INVALID_ARG);
    assert!(last_error().contains("source is"), "got {:?}", last_error());

    unsafe { qntt_f64_vec_free(x) };
}

#[test]
fn to_dense_matches_the_rust_api_and_refuses_oversized_requests() {
    let space = QnSpace::new(4, 2, 2);
    let ranks = uniform_ranks(space, 2);
    let flat = flat_ranks(&space, &ranks);
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                7,
                &mut x,
            )
        },
        QNTT_OK
    );

    let mut len = 0usize;
    assert_eq!(
        unsafe { qntt_f64_vec_to_dense(x, ptr::null_mut(), 0, &mut len) },
        QNTT_OK
    );
    assert_eq!(len, 4usize.pow(space.d as u32));

    let mut tiny = vec![0.0f64; 1];
    assert_eq!(
        unsafe { qntt_f64_vec_to_dense(x, tiny.as_mut_ptr(), tiny.len(), &mut len) },
        QNTT_INVALID_ARG
    );

    let mut dense = vec![0.0f64; len];
    assert_eq!(
        unsafe { qntt_f64_vec_to_dense(x, dense.as_mut_ptr(), dense.len(), &mut len) },
        QNTT_OK
    );

    use rand::SeedableRng;
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
    let direct = qntt::tt_randn::<f64, _>(space, &ranks, &mut rng);
    let expect = direct.to_dense();
    assert_eq!(dense.len(), expect.len());
    for (i, (&a, &b)) in dense.iter().zip(expect.iter()).enumerate() {
        assert!((a - b).abs() <= 1e-15, "entry {i}: {a} vs {b}");
    }

    unsafe { qntt_f64_vec_free(x) };
}
