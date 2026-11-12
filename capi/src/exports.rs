// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The exported symbols: four families generated from one macro body.

use std::os::raw::c_int;

use qntt::C64;
use qntt::Sparse3Body;
use qntt::TtVector;
use qntt::su2::vector::Su2TtVector;

use crate::api;
use crate::err::catch;
use crate::handle::{Handle, borrow, release};
use crate::layer::{Abelian, Su2};
use crate::opts::*;
use crate::pointgroup as pg;
use crate::su2_pointgroup as su2pg;

macro_rules! export_layer {
    ($sfx:ident, $L:ty, $T:ty) => {
        paste::paste! {

            /// Release a vector handle (no-op on `NULL`).
            ///
            /// # Safety
            /// `x` must be null or a live vector handle not already released.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_free>](x: *mut Handle<<$L as crate::layer::Layer>::Vector>) {
                unsafe { release(x) }
            }

            /// All-zero train (every bond rank zero).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_zeros>](
                d: usize, a: usize, b: usize,
                out: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::vec_zeros::<$L>(d, a, b, out) })
            }

            /// Zero-filled train at prescribed per-bond ranks.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_from_ranks>](
                d: usize, a: usize, b: usize,
                ranks: *const usize, ranks_len: usize,
                out: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::vec_from_ranks::<$L>(d, a, b, ranks, ranks_len, out) })
            }

            /// Gaussian random train at prescribed per-bond ranks.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_randn>](
                d: usize, a: usize, b: usize,
                ranks: *const usize, ranks_len: usize, seed: u64,
                out: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::vec_randn::<$L>(d, a, b, ranks, ranks_len, seed, out) })
            }

            /// Deep copy.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_clone>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                out: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::vec_clone::<$L>(x, out) })
            }

            /// `(d, a, b)` of the train's space.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_space>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                out_d: *mut usize, out_a: *mut usize, out_b: *mut usize,
            ) -> c_int {
                catch(|| unsafe { api::vec_space::<$L>(x, out_d, out_a, out_b) })
            }

            /// Flattened per-bond rank grids.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_ranks>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                out: *mut usize, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| unsafe { api::vec_ranks::<$L>(x, out, cap, out_len) })
            }

            /// Orthogonality centre, `-1` if none.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_ortho>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>, out: *mut isize,
            ) -> c_int {
                catch(|| unsafe { api::vec_ortho::<$L>(x, out) })
            }

            /// Structural self-check (`1` = consistent).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_check>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>, out: *mut i32,
            ) -> c_int {
                catch(|| unsafe { api::vec_check::<$L>(x, out) })
            }

            /// Sites, largest per-sector rank, largest summed bond dimension.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_sizes>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                out_sites: *mut usize, out_max_sector_rank: *mut usize, out_max_bond_rank: *mut usize,
            ) -> c_int {
                catch(|| unsafe { api::vec_sizes::<$L>(x, out_sites, out_max_sector_rank, out_max_bond_rank) })
            }


            /// `(x, y)` — the BILINEAR (unconjugated) contraction; `rho = |(x,x)|/⟨x,x⟩`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_dot_bilinear>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                y: *const Handle<<$L as crate::layer::Layer>::Vector>,
                out: *mut $T,
            ) -> c_int {
                catch(|| unsafe { api::vec_dot_bilinear::<$L>(x, y, out) })
            }

            /// Schmidt values at every interior bond, concatenated, with their multiplet weights
            /// and the per-bond counts. `Σ wᵢ σᵢ² = ‖x‖²` at every bond.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_schmidt>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                out: *mut f64, cap: usize, out_len: *mut usize,
                out_weights: *mut f64, weights_cap: usize,
                out_counts: *mut usize, counts_cap: usize, out_nbonds: *mut usize,
            ) -> c_int {
                catch(|| unsafe {
                    api::vec_schmidt::<$L>(
                        x, out, cap, out_len, out_weights, weights_cap,
                        out_counts, counts_cap, out_nbonds,
                    )
                })
            }

            /// Riemannian gradient norm at `x` — the criticality gate.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _grad_norm>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                out: *mut f64,
            ) -> c_int {
                catch(|| unsafe { api::grad_norm::<$L>(h, x, out) })
            }


            /// Write the train to `path` (truncating).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_save>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                path: *const std::os::raw::c_char,
            ) -> c_int {
                catch(|| unsafe { api::vec_save::<$L>(x, path) })
            }

            /// Read a train written by `qntt_*_vec_save` at the SAME layer and scalar.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_load>](
                path: *const std::os::raw::c_char,
                out: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::vec_load::<$L>(path, out) })
            }

            /// Copy one core block out, column-major.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_block_get>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                site: usize, family: i32, sa: usize, sb: usize,
                out: *mut $T, cap: usize, out_rows: *mut usize, out_cols: *mut usize,
            ) -> c_int {
                catch(|| unsafe {
                    api::vec_block_get::<$L>(x, site, family, sa, sb, out, cap, out_rows, out_cols)
                })
            }

            /// Overwrite one core block from a column-major buffer (unshared handle only).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_block_set>](
                x: *mut Handle<<$L as crate::layer::Layer>::Vector>,
                site: usize, family: i32, sa: usize, sb: usize,
                src: *const $T, rows: usize, cols: usize,
            ) -> c_int {
                catch(|| unsafe { api::vec_block_set::<$L>(x, site, family, sa, sb, src, rows, cols) })
            }

            /// Layer norm.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_norm>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>, out: *mut f64,
            ) -> c_int {
                catch(|| unsafe { api::vec_norm::<$L>(x, out) })
            }

            /// `⟨x, y⟩`, conjugating `x`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_dot>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                y: *const Handle<<$L as crate::layer::Layer>::Vector>,
                out: *mut $T,
            ) -> c_int {
                catch(|| unsafe { api::vec_dot::<$L>(x, y, out) })
            }

            /// `a · x` as a new train.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_scaled>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                a: *const $T,
                out: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::vec_scaled::<$L>(x, a, out) })
            }

            /// `x + y` (`sign >= 0`) or `x - y`, unrounded.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_add>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                y: *const Handle<<$L as crate::layer::Layer>::Vector>,
                sign: i32,
                out: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::vec_add::<$L>(x, y, sign, out) })
            }

            /// Deterministic SVD rounding into a new train.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_round>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                eps: f64, chi: usize, mode: i32,
                out: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::vec_round::<$L>(x, eps, chi, mode, out) })
            }

            /// In-place deterministic SVD rounding (unshared handle only).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_round_mut>](
                x: *mut Handle<<$L as crate::layer::Layer>::Vector>,
                eps: f64, chi: usize, mode: i32,
            ) -> c_int {
                catch(|| unsafe { api::vec_round_mut::<$L>(x, eps, chi, mode) })
            }

            /// Left-orthonormalization sweep (unshared handle only).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_left_orthonormalize>](
                x: *mut Handle<<$L as crate::layer::Layer>::Vector>,
                keep_rank: i32, out_norm: *mut f64,
            ) -> c_int {
                catch(|| unsafe { api::vec_left_orthonormalize::<$L>(x, keep_rank, out_norm) })
            }

            /// `round(Σ αᵢ xᵢ, tol)`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_round_sum>](
                alpha: *const $T,
                vecs: *const *const Handle<<$L as crate::layer::Layer>::Vector>,
                n: usize, tol: f64,
                out: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::vec_round_sum::<$L>(alpha, vecs, n, tol, out) })
            }


            /// Release a Hamiltonian handle (no-op on `NULL`).
            ///
            /// # Safety
            /// `h` must be null or a live handle not already released.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _ham_free>](h: *mut Handle<<$L as crate::layer::Layer>::Ham>) {
                unsafe { release(h) }
            }

            /// Compile the second-quantized operator from row-major `t`/`v` (and optional `w`).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _ham_new>](
                d: usize, a: usize, b: usize,
                t: *const $T, t_len: usize,
                v: *const $T, v_len: usize,
                w: *const Handle<Sparse3Body<$T>>,
                eps: f64, construction: i32,
                out: *mut *mut Handle<<$L as crate::layer::Layer>::Ham>,
            ) -> c_int {
                catch(|| unsafe {
                    api::ham_new::<$L>(d, a, b, t, t_len, v, v_len, w, eps, construction, out)
                })
            }

            /// `(d, a, b)` of the operator's space.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _ham_space>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                out_d: *mut usize, out_a: *mut usize, out_b: *mut usize,
            ) -> c_int {
                catch(|| unsafe { api::ham_space::<$L>(h, out_d, out_a, out_b) })
            }

            /// Automaton bond dimensions, where the layer reports them.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _ham_bond_dims>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                out: *mut usize, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| unsafe { api::ham_bond_dims::<$L>(h, out, cap, out_len) })
            }

            /// `H·x`, unrounded.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _ham_apply>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                out: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::ham_apply::<$L>(h, x, out) })
            }

            /// `⟨x, H·x⟩` — raw expectation, NOT normalized (see `ham_rayleigh`).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _ham_expectation>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                out: *mut $T,
            ) -> c_int {
                catch(|| unsafe { api::ham_expectation::<$L>(h, x, out) })
            }

            /// `⟨x, H·x⟩ / ⟨x, x⟩`, safe on an unnormalized train.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _ham_rayleigh>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                out: *mut $T,
            ) -> c_int {
                catch(|| unsafe { api::ham_rayleigh::<$L>(h, x, out) })
            }

            /// `⟨x, H·y⟩`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _ham_xthy>](
                x: *const Handle<<$L as crate::layer::Layer>::Vector>,
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                y: *const Handle<<$L as crate::layer::Layer>::Vector>,
                out: *mut $T,
            ) -> c_int {
                catch(|| unsafe { api::ham_xthy::<$L>(x, h, y, out) })
            }


            /// One-site ALS (DMRG).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _als>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                x0: *const Handle<<$L as crate::layer::Layer>::Vector>,
                opts: *const QnttAlsOptions,
                out_lambda: *mut $T,
                out_x: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::als::<$L>(h, x0, opts, out_lambda, out_x) })
            }

            /// Two-site rank-adaptive MALS (DMRG).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _mals>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                x0: *const Handle<<$L as crate::layer::Layer>::Vector>,
                opts: *const QnttMalsOptions,
                bond_dims: *const usize, bond_dims_len: usize,
                noise: *const f64, noise_len: usize,
                out_lambda: *mut f64,
                out_x: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
                out_sweeps: *mut QnttMalsSweep, sweeps_cap: usize, out_nsweeps: *mut usize,
            ) -> c_int {
                catch(|| unsafe {
                    api::mals::<$L>(
                        h, x0, opts, bond_dims, bond_dims_len, noise, noise_len,
                        out_lambda, out_x, out_sweeps, sweeps_cap, out_nsweeps,
                    )
                })
            }

            /// Imaginary-time two-site TDVP.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _tdvp>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                x0: *const Handle<<$L as crate::layer::Layer>::Vector>,
                opts: *const QnttTdvpOptions,
                out_lambda: *mut f64,
                out_x: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::tdvp::<$L>(h, x0, opts, out_lambda, out_x) })
            }

            /// Riemannian optimization on the fixed-rank manifold (`use_cg != 0` ⇒ CG).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _riemannian>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                x0: *const Handle<<$L as crate::layer::Layer>::Vector>,
                opts: *const QnttRiemannianOpts,
                use_cg: i32,
                out_lambda: *mut f64,
                out_x: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
            ) -> c_int {
                catch(|| unsafe { api::riemannian::<$L>(h, x0, opts, use_cg, out_lambda, out_x) })
            }

            /// Rounded-Lanczos ground state with optional per-iteration history.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _lanczos>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                x0: *const Handle<<$L as crate::layer::Layer>::Vector>,
                opts: *const QnttLanczosOptions,
                out_lambda: *mut f64,
                out_x: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
                out_breakdown: *mut i32,
                out_iters: *mut usize,
                hist_lambdas: *mut f64,
                hist_res: *mut f64,
                hist_ranks: *mut usize,
                hist_rounded_ranks: *mut usize,
                hist_cap: usize,
            ) -> c_int {
                catch(|| unsafe {
                    api::lanczos_solve::<$L>(
                        h, x0, opts, out_lambda, out_x, out_breakdown, out_iters,
                        hist_lambdas, hist_res, hist_ranks, hist_rounded_ranks, hist_cap,
                    )
                })
            }


            /// State-average two-site MALS: the `nroots` lowest states sharing one train.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _state_average_mals>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                x0: *const Handle<<$L as crate::layer::Layer>::Vector>,
                nroots: usize,
                opts: *const QnttStateAverageOptions,
                weights: *const f64, weights_len: usize,
                bond_dims: *const usize, bond_dims_len: usize,
                noise: *const f64, noise_len: usize,
                out_lambdas: *mut f64,
                out_states: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
                states_cap: usize,
                out_sweeps: *mut QnttSaSweep, sweeps_cap: usize, out_nsweeps: *mut usize,
                out_sweep_energies: *mut f64, out_sweep_discarded: *mut f64,
            ) -> c_int {
                catch(|| unsafe {
                    api::state_average_mals::<$L>(
                        h, x0, nroots, opts, weights, weights_len, bond_dims, bond_dims_len,
                        noise, noise_len, out_lambdas, out_states, states_cap,
                        out_sweeps, sweeps_cap, out_nsweeps,
                        out_sweep_energies, out_sweep_discarded,
                    )
                })
            }

            /// State-specific two-site MALS: one state against fixed lower states.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _state_specific_mals>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                x0: *const Handle<<$L as crate::layer::Layer>::Vector>,
                lower: *const *const Handle<<$L as crate::layer::Layer>::Vector>,
                nlower: usize,
                opts: *const QnttStateSpecificOptions,
                bond_dims: *const usize, bond_dims_len: usize,
                noise: *const f64, noise_len: usize,
                lower_energies: *const f64, lower_energies_len: usize,
                out_lambda: *mut f64,
                out_x: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
                out_sweeps: *mut QnttSsSweep, sweeps_cap: usize, out_nsweeps: *mut usize,
                out_sweep_overlaps: *mut f64,
            ) -> c_int {
                catch(|| unsafe {
                    api::state_specific_mals::<$L>(
                        h, x0, lower, nlower, opts, bond_dims, bond_dims_len, noise, noise_len,
                        lower_energies, lower_energies_len, out_lambda, out_x,
                        out_sweeps, sweeps_cap, out_nsweeps, out_sweep_overlaps,
                    )
                })
            }

            /// The excited-state ladder: state average, then a state-specific sweep and a
            /// penalized Riemannian polish per root, with the certification gates.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _excited_states>](
                h: *const Handle<<$L as crate::layer::Layer>::Ham>,
                x0: *const Handle<<$L as crate::layer::Layer>::Vector>,
                lower: *const *const Handle<<$L as crate::layer::Layer>::Vector>,
                nlower: usize,
                opts: *const QnttExcitedOptions,
                weights: *const f64, weights_len: usize,
                bond_dims: *const usize, bond_dims_len: usize,
                noise: *const f64, noise_len: usize,
                ss_noise: *const f64, ss_noise_len: usize,
                out_energies: *mut f64,
                out_roots: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
                roots_cap: usize,
                out_info: *mut QnttExcitedInfo,
                out_root_info: *mut QnttExcitedRoot,
                out_lower_gradnorms: *mut f64,
                out_gram: *mut f64, gram_cap: usize, out_gram_len: *mut usize,
                out_sa_states: *mut *mut Handle<<$L as crate::layer::Layer>::Vector>,
                sa_states_cap: usize,
            ) -> c_int {
                catch(|| unsafe {
                    api::excited_states::<$L>(
                        h, x0, lower, nlower, opts, weights, weights_len,
                        bond_dims, bond_dims_len, noise, noise_len, ss_noise, ss_noise_len,
                        out_energies, out_roots, roots_cap, out_info, out_root_info,
                        out_lower_gradnorms, out_gram, gram_cap, out_gram_len,
                        out_sa_states, sa_states_cap,
                    )
                })
            }








        }
    };
}

export_layer!(f64, Abelian<f64>, f64);
export_layer!(c64, Abelian<C64>, C64);
export_layer!(su2_f64, Su2<f64>, f64);
export_layer!(su2_c64, Su2<C64>, C64);

macro_rules! export_space {
    ($p:ident, $L:ty) => {
        paste::paste! {
            /// Validate a space descriptor (`(d, Nup, Ndn)` abelian, `(d, N, 2S)` SU(2)).
            #[unsafe(no_mangle)]
            pub extern "C" fn [<$p _check>](d: usize, a: usize, b: usize) -> c_int {
                catch(|| api::space_check::<$L>(d, a, b))
            }

            /// `(rows, cols)` of the layer's sector grid.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<$p _grid_extent>](
                d: usize, a: usize, b: usize, out_rows: *mut usize, out_cols: *mut usize,
            ) -> c_int {
                catch(|| unsafe { api::space_grid_extent::<$L>(d, a, b, out_rows, out_cols) })
            }

            /// Admissible sectors at `bond`, canonical order.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<$p _bond_sectors>](
                d: usize, a: usize, b: usize, bond: usize,
                out_a: *mut usize, out_b: *mut usize, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| unsafe {
                    api::space_bond_sectors::<$L>(d, a, b, bond, out_a, out_b, cap, out_len)
                })
            }

            /// Maximal admissible rank grids, flattened over all `d+1` bonds.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<$p _max_ranks>](
                d: usize, a: usize, b: usize,
                out: *mut usize, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| unsafe { api::space_max_ranks::<$L>(d, a, b, out, cap, out_len) })
            }
        }
    };
}

export_space!(qntt_space, Abelian<f64>);
export_space!(qntt_su2_space, Su2<f64>);

/// `(Nup, Ndn)` from a total electron count and `2·Sz` (abelian only).
///
/// # Safety
/// Out-pointers must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_space_from_n_two_sz(
    d: usize,
    n: usize,
    two_sz: i64,
    out_nup: *mut usize,
    out_ndn: *mut usize,
) -> c_int {
    catch(|| {
        let s = qntt::QnSpace::from_n_two_sz(d, n, two_sz);
        unsafe {
            *crate::err::ptr_mut(out_nup, "out_nup")? = s.nup;
            *crate::err::ptr_mut(out_ndn, "out_ndn")? = s.ndn;
        }
        Ok(())
    })
}

macro_rules! export_abelian_extra {
    ($sfx:ident, $T:ty) => {
        paste::paste! {
            /// Rank-1 product state from per-site occupations (`occ_up[k]`, `occ_dn[k]` nonzero =
            /// occupied). Both arrays have `d` entries.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _tt_state>](
                occ_up: *const i32, occ_dn: *const i32, d: usize,
                out: *mut *mut Handle<TtVector<$T>>,
            ) -> c_int {
                catch(|| {
                    let up = unsafe { crate::err::ptr_slice(occ_up, d, "occ_up")? };
                    let dn = unsafe { crate::err::ptr_slice(occ_dn, d, "occ_dn")? };
                    let up: Vec<bool> = up.iter().map(|&v| v != 0).collect();
                    let dn: Vec<bool> = dn.iter().map(|&v| v != 0).collect();
                    unsafe { crate::handle::emit(qntt::tt_state::<$T>(&up, &dn), out) }
                })
            }

            /// Dense materialization in the `4^d` Jordan–Wigner basis. Refuses to allocate when
            /// `4^d` overflows or exceeds `cap`; call with `cap = 0` to query the size.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_to_dense>](
                x: *const Handle<TtVector<$T>>,
                out: *mut $T, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| {
                    let x = unsafe { borrow(x)? };
                    let d = x.space.d;
                    let n = 4usize.checked_pow(d as u32).ok_or_else(|| {
                        crate::err::Error::invalid(format!("4^{d} overflows a machine word"))
                    })?;
                    if !out_len.is_null() {
                        unsafe { *out_len = n };
                    }
                    if cap == 0 {
                        return Ok(());
                    }
                    if cap < n {
                        return Err(crate::err::Error::invalid(format!(
                            "dense buffer holds {cap} scalars, need 4^{d} = {n}"
                        )));
                    }
                    let dense = x.to_dense();
                    unsafe { crate::err::copy_out(&dense, out, cap, std::ptr::null_mut(), "dense") }
                })
            }
        }
    };
}

export_abelian_extra!(f64, f64);
export_abelian_extra!(c64, C64);

macro_rules! export_abelian_pg {
    ($sfx:ident, $T:ty) => {
        paste::paste! {
            /// All-zero train in a point-group space.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_zeros_pg>](
                d: usize, a: usize, b: usize,
                pg_kind: i32, orbsym: *const u8, orbsym_len: usize, target: i32,
                out: *mut *mut Handle<TtVector<$T>>,
            ) -> c_int {
                catch(|| unsafe {
                    pg::vec_zeros_pg::<$T>(d, a, b, pg_kind, orbsym, orbsym_len, target, out)
                })
            }

            /// Zero-filled train at prescribed per-bond ranks (`(d+1)·|G|·rows·cols` entries).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_from_ranks_pg>](
                d: usize, a: usize, b: usize,
                pg_kind: i32, orbsym: *const u8, orbsym_len: usize, target: i32,
                ranks: *const usize, ranks_len: usize,
                out: *mut *mut Handle<TtVector<$T>>,
            ) -> c_int {
                catch(|| unsafe {
                    pg::vec_from_ranks_pg::<$T>(
                        d, a, b, pg_kind, orbsym, orbsym_len, target, ranks, ranks_len, out,
                    )
                })
            }

            /// Gaussian random train; ranks at irrep-unreachable sectors are masked to zero.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_randn_pg>](
                d: usize, a: usize, b: usize,
                pg_kind: i32, orbsym: *const u8, orbsym_len: usize, target: i32,
                ranks: *const usize, ranks_len: usize, seed: u64,
                out: *mut *mut Handle<TtVector<$T>>,
            ) -> c_int {
                catch(|| unsafe {
                    pg::vec_randn_pg::<$T>(
                        d, a, b, pg_kind, orbsym, orbsym_len, target, ranks, ranks_len, seed, out,
                    )
                })
            }

            /// A determinant placed in a point-group space; its total irrep must be `target`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _tt_state_pg>](
                occ_up: *const i32, occ_dn: *const i32, d: usize,
                pg_kind: i32, orbsym: *const u8, orbsym_len: usize, target: i32,
                out: *mut *mut Handle<TtVector<$T>>,
            ) -> c_int {
                catch(|| unsafe {
                    pg::tt_state_pg::<$T>(
                        occ_up, occ_dn, d, pg_kind, orbsym, orbsym_len, target, out,
                    )
                })
            }

            /// The 0/1 indicator of the `(Nup, Ndn, target)` block.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _tt_ones_pg>](
                d: usize, a: usize, b: usize,
                pg_kind: i32, orbsym: *const u8, orbsym_len: usize, target: i32,
                out: *mut *mut Handle<TtVector<$T>>,
            ) -> c_int {
                catch(|| unsafe {
                    pg::tt_ones_pg::<$T>(d, a, b, pg_kind, orbsym, orbsym_len, target, out)
                })
            }

            /// The train's per-bond rank grids, flattened with the irrep axis.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_ranks_pg>](
                x: *const Handle<TtVector<$T>>,
                out: *mut usize, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| unsafe { pg::vec_ranks_pg::<$T>(x, out, cap, out_len) })
            }

            /// Copy one block out, addressing the sector by `(nup, ndn, irrep)`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_block_get_pg>](
                x: *const Handle<TtVector<$T>>,
                site: usize, family: i32, sa: usize, sb: usize, irrep: u8,
                out: *mut $T, cap: usize, out_rows: *mut usize, out_cols: *mut usize,
            ) -> c_int {
                catch(|| unsafe {
                    pg::vec_block_get_pg::<$T>(
                        x, site, family, sa, sb, irrep, out, cap, out_rows, out_cols,
                    )
                })
            }

            /// Overwrite one block, addressing the sector by `(nup, ndn, irrep)`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_block_set_pg>](
                x: *mut Handle<TtVector<$T>>,
                site: usize, family: i32, sa: usize, sb: usize, irrep: u8,
                src: *const $T, rows: usize, cols: usize,
            ) -> c_int {
                catch(|| unsafe {
                    pg::vec_block_set_pg::<$T>(x, site, family, sa, sb, irrep, src, rows, cols)
                })
            }

            /// The train's packed point group: `nbits`, the XOR target, and `d` per-site XOR
            /// irreps. The KIND is not recoverable — see `include/qntt.h`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_point_group>](
                x: *const Handle<TtVector<$T>>,
                out_nbits: *mut i32, out_target: *mut i32,
                out_irreps: *mut u8, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| unsafe {
                    pg::vec_point_group::<$T>(x, out_nbits, out_target, out_irreps, cap, out_len)
                })
            }

            /// Compile the second-quantized operator over a point-group space.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _ham_new_pg>](
                d: usize, a: usize, b: usize,
                pg_kind: i32, orbsym: *const u8, orbsym_len: usize, target: i32,
                t: *const $T, t_len: usize, v: *const $T, v_len: usize,
                w: *const Handle<Sparse3Body<$T>>, eps: f64, construction: i32,
                out: *mut *mut Handle<qntt::SparseHamiltonian<$T>>,
            ) -> c_int {
                catch(|| unsafe {
                    pg::ham_new_pg::<$T>(
                        d, a, b, pg_kind, orbsym, orbsym_len, target,
                        t, t_len, v, v_len, w, eps, construction, out,
                    )
                })
            }

            /// The operator's packed point group; see `*_vec_point_group`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _ham_point_group>](
                h: *const Handle<qntt::SparseHamiltonian<$T>>,
                out_nbits: *mut i32, out_target: *mut i32,
                out_irreps: *mut u8, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| unsafe {
                    pg::ham_point_group::<$T>(h, out_nbits, out_target, out_irreps, cap, out_len)
                })
            }
        }
    };
}

export_abelian_pg!(f64, f64);
export_abelian_pg!(c64, C64);

macro_rules! export_su2_extra {
    ($sfx:ident, $T:ty) => {
        paste::paste! {
            /// Rank-1 configuration state function from per-site genealogical branch codes
            /// (`0` empty, `1` spin-raising, `2` spin-lowering, `3` doubly occupied — NOT
            /// electron spins; see `src/su2/mod.rs`).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _tt_csf>](
                codes: *const i32, d: usize,
                out: *mut *mut Handle<Su2TtVector<$T>>,
            ) -> c_int {
                catch(|| {
                    let raw = unsafe { crate::err::ptr_slice(codes, d, "codes")? };
                    let fams: Result<Vec<_>, _> = raw
                        .iter()
                        .map(|&c| <Su2<$T> as crate::layer::Layer>::family(c))
                        .collect();
                    unsafe {
                        crate::handle::emit(qntt::su2::vector::tt_csf::<$T>(fams?.as_slice()), out)
                    }
                })
            }
        }
    };
}

export_su2_extra!(su2_f64, f64);
export_su2_extra!(su2_c64, C64);

macro_rules! export_su2_pg {
    ($sfx:ident, $T:ty) => {
        paste::paste! {
            /// All-zero SU(2) train in a point-group space.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_zeros_pg>](
                d: usize, a: usize, b: usize,
                pg_kind: i32, orbsym: *const u8, orbsym_len: usize, target: i32,
                out: *mut *mut Handle<Su2TtVector<$T>>,
            ) -> c_int {
                catch(|| unsafe {
                    su2pg::vec_zeros_pg::<$T>(d, a, b, pg_kind, orbsym, orbsym_len, target, out)
                })
            }

            /// Zero-filled SU(2) train at prescribed per-bond ranks (`(d+1)·|G|·rows·cols`
            /// entries; reduced multiplet ranks, not abelian state counts).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_from_ranks_pg>](
                d: usize, a: usize, b: usize,
                pg_kind: i32, orbsym: *const u8, orbsym_len: usize, target: i32,
                ranks: *const usize, ranks_len: usize,
                out: *mut *mut Handle<Su2TtVector<$T>>,
            ) -> c_int {
                catch(|| unsafe {
                    su2pg::vec_from_ranks_pg::<$T>(
                        d, a, b, pg_kind, orbsym, orbsym_len, target, ranks, ranks_len, out,
                    )
                })
            }

            /// Gaussian random SU(2) train; ranks at irrep-unreachable sectors are masked to zero.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_randn_pg>](
                d: usize, a: usize, b: usize,
                pg_kind: i32, orbsym: *const u8, orbsym_len: usize, target: i32,
                ranks: *const usize, ranks_len: usize, seed: u64,
                out: *mut *mut Handle<Su2TtVector<$T>>,
            ) -> c_int {
                catch(|| unsafe {
                    su2pg::vec_randn_pg::<$T>(
                        d, a, b, pg_kind, orbsym, orbsym_len, target, ranks, ranks_len, seed, out,
                    )
                })
            }

            /// A rank-1 CSF in a point-group space. `(N, 2S)` and the total irrep are DERIVED from
            /// the genealogical walk; a `target` that disagrees is rejected.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _tt_csf_pg>](
                codes: *const i32, d: usize,
                pg_kind: i32, orbsym: *const u8, orbsym_len: usize, target: i32,
                out: *mut *mut Handle<Su2TtVector<$T>>,
            ) -> c_int {
                catch(|| unsafe {
                    su2pg::tt_csf_pg::<$T>(codes, d, pg_kind, orbsym, orbsym_len, target, out)
                })
            }

            /// The train's per-bond rank grids, flattened with the irrep axis.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_ranks_pg>](
                x: *const Handle<Su2TtVector<$T>>,
                out: *mut usize, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| unsafe { su2pg::vec_ranks_pg::<$T>(x, out, cap, out_len) })
            }

            /// Copy one block out, addressing the sector by `(n, 2s, irrep)`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_block_get_pg>](
                x: *const Handle<Su2TtVector<$T>>,
                site: usize, family: i32, sa: usize, sb: usize, irrep: u8,
                out: *mut $T, cap: usize, out_rows: *mut usize, out_cols: *mut usize,
            ) -> c_int {
                catch(|| unsafe {
                    su2pg::vec_block_get_pg::<$T>(
                        x, site, family, sa, sb, irrep, out, cap, out_rows, out_cols,
                    )
                })
            }

            /// Overwrite one block, addressing the sector by `(n, 2s, irrep)`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_block_set_pg>](
                x: *mut Handle<Su2TtVector<$T>>,
                site: usize, family: i32, sa: usize, sb: usize, irrep: u8,
                src: *const $T, rows: usize, cols: usize,
            ) -> c_int {
                catch(|| unsafe {
                    su2pg::vec_block_set_pg::<$T>(x, site, family, sa, sb, irrep, src, rows, cols)
                })
            }

            /// The train's packed point group: `nbits`, the XOR target, and `d` per-site XOR
            /// irreps. The KIND is not recoverable — see `include/qntt.h`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_point_group>](
                x: *const Handle<Su2TtVector<$T>>,
                out_nbits: *mut i32, out_target: *mut i32,
                out_irreps: *mut u8, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| unsafe {
                    su2pg::vec_point_group::<$T>(x, out_nbits, out_target, out_irreps, cap, out_len)
                })
            }

            /// Compile the SU(2) operator over a point-group space (`construction` must be the
            /// default: the SU(2) assembly has only one).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _ham_new_pg>](
                d: usize, a: usize, b: usize,
                pg_kind: i32, orbsym: *const u8, orbsym_len: usize, target: i32,
                t: *const $T, t_len: usize, v: *const $T, v_len: usize,
                w: *const Handle<Sparse3Body<$T>>, eps: f64, construction: i32,
                out: *mut *mut Handle<qntt::su2::hamiltonian::Su2SparseHamiltonian<$T>>,
            ) -> c_int {
                catch(|| unsafe {
                    su2pg::ham_new_pg::<$T>(
                        d, a, b, pg_kind, orbsym, orbsym_len, target,
                        t, t_len, v, v_len, w, eps, construction, out,
                    )
                })
            }

            /// The operator's packed point group; see `*_vec_point_group`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _ham_point_group>](
                h: *const Handle<qntt::su2::hamiltonian::Su2SparseHamiltonian<$T>>,
                out_nbits: *mut i32, out_target: *mut i32,
                out_irreps: *mut u8, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| unsafe {
                    su2pg::ham_point_group::<$T>(h, out_nbits, out_target, out_irreps, cap, out_len)
                })
            }
        }
    };
}

export_su2_pg!(su2_f64, f64);
export_su2_pg!(su2_c64, C64);

macro_rules! export_scalar {
    ($sfx:ident, $T:ty) => {
        paste::paste! {
            /// Release a three-body coefficient handle (no-op on `NULL`).
            ///
            /// # Safety
            /// `w` must be null or a live handle not already released.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _w_free>](w: *mut Handle<Sparse3Body<$T>>) {
                unsafe { release(w) }
            }

            /// Empty sparse three-body coefficient tensor over `norb` orbitals.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _w_new>](
                norb: usize, out: *mut *mut Handle<Sparse3Body<$T>>,
            ) -> c_int {
                catch(|| unsafe { api::w_new::<$T>(norb, out) })
            }

            /// Bulk COO fill (`idx` holds `6·n` 0-based indices).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _w_set_coo>](
                w: *mut Handle<Sparse3Body<$T>>,
                idx: *const usize, vals: *const $T, n: usize,
            ) -> c_int {
                catch(|| unsafe { api::w_set_coo::<$T>(w, idx, vals, n) })
            }

            /// Stored nonzero count.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _w_nnz>](
                w: *const Handle<Sparse3Body<$T>>, out: *mut usize,
            ) -> c_int {
                catch(|| unsafe { api::w_nnz::<$T>(w, out) })
            }
        }
    };
}

export_scalar!(f64, f64);
export_scalar!(c64, C64);

/// Fill `out` with the ALS keyword defaults.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_als_options_default(out: *mut QnttAlsOptions) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttAlsOptions::defaults() };
        Ok(())
    })
}

/// Fill `out` with the MALS keyword defaults (the bond-dimension schedule defaults to `[0]`,
/// i.e. uncapped, and is passed separately).
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_mals_options_default(out: *mut QnttMalsOptions) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttMalsOptions::defaults() };
        Ok(())
    })
}

/// Fill `out` with the TDVP keyword defaults.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_tdvp_options_default(out: *mut QnttTdvpOptions) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttTdvpOptions::defaults() };
        Ok(())
    })
}

/// Fill `out` with the Lanczos keyword defaults.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_lanczos_options_default(out: *mut QnttLanczosOptions) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttLanczosOptions::defaults() };
        Ok(())
    })
}

/// Fill `out` with the Riemannian keyword defaults.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_riemannian_options_default(out: *mut QnttRiemannianOpts) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttRiemannianOpts::defaults() };
        Ok(())
    })
}

/// Fill `out` with the state-average keyword defaults (the weight, bond-dimension and noise
/// schedules are passed separately).
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_state_average_options_default(
    out: *mut QnttStateAverageOptions,
) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttStateAverageOptions::defaults() };
        Ok(())
    })
}

/// Fill `out` with the state-specific keyword defaults (the schedules and the lower-state
/// reference energies are passed separately).
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_state_specific_options_default(
    out: *mut QnttStateSpecificOptions,
) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttStateSpecificOptions::defaults() };
        Ok(())
    })
}

/// Fill `out` with the excited-state ladder's keyword defaults.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_excited_options_default(out: *mut QnttExcitedOptions) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttExcitedOptions::defaults() };
        Ok(())
    })
}

/// Fill `out` with the matrix-free RPA keyword defaults.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_symplectic_options_default(
    out: *mut QnttSymplecticLowestOpts,
) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttSymplecticLowestOpts::defaults() };
        Ok(())
    })
}

/// Fill `out` with the CROSS-SECTOR (shifted-channel) readout's keyword defaults: the `Tda`
/// readout and the `Auto` metric, on the library's own `tol`/`max_iter`. `seed` is not a field —
/// it is a call parameter of the sector entry points.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_sector_options_default(out: *mut QnttSectorOpts) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttSectorOpts::defaults() };
        Ok(())
    })
}

/// Fill `out` with the in-sector Tamm–Dancoff readout's (`qntt_*_tda`) keyword defaults: the
/// library's [`TdaDavidsonOpts`](qntt::solvers::hess::TdaDavidsonOpts) with
/// `precond = QNTT_TDA_PRECOND_AUTO` and `precond_kappa = 10`, the cap `AUTO` resolves to.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_tda_options_default(out: *mut QnttTdaOpts) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttTdaOpts::defaults() };
        Ok(())
    })
}

/// Fill `out` with the keyword defaults of `qntt_*_sector_alpha`: [`SectorStaticOpts`](qntt::solvers::hess::SectorStaticOpts)' own
/// `tol`/`max_iter`/`tt_grad_tol`, `has_sym_assert_rel = 0` (so the library recomputes `100·tol`
/// from the `tol` actually passed) and `QNTT_PROP_ROUTE_MATERIALIZED`, the only route a channel
/// has.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_sector_alpha_options_default(out: *mut QnttSectorAlphaOpts) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttSectorAlphaOpts::defaults() };
        Ok(())
    })
}

/// Fill `out` with the keyword defaults of `qntt_*_sector_transition_moments`:
/// [`SectorMomentOpts`](qntt::solvers::hess::SectorMomentOpts)' own `tt_grad_tol`/`degeneracy_tol`/`psd_tol` and
/// `QNTT_PROP_ROUTE_MATERIALIZED`. `reserved` is zeroed.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_sector_moment_options_default(
    out: *mut QnttSectorMomentOpts,
) -> c_int {
    catch(|| {
        unsafe { *crate::err::ptr_mut(out, "out")? = QnttSectorMomentOpts::defaults() };
        Ok(())
    })
}
