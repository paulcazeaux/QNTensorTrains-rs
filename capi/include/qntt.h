/*
 * Copyright (C) 2023-2026 Paul Cazeaux
 *
 * This file is part of qntt. qntt is free software: you can
 * redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your
 * option) any later version. See LICENSE for details.
 */

/*
 * qntt — C ABI for quantum-number-preserving block-sparse tensor trains.
 *
 * Build:  cargo build --release -p qntt-capi   →  libqntt_capi.{dylib,so,dll}
 *
 * ---------------------------------------------------------------------------------------------
 * Contract
 * ---------------------------------------------------------------------------------------------
 * - Every function returns int32_t; QNTT_OK (0) is success. Results are written through
 *   out-pointers, never returned by value.
 * - OUT-PARAMETERS MUST NOT ALIAS: no two out-pointers of one call may name overlapping storage,
 *   and none may overlap an input buffer of the same call. The library writes them as distinct
 *   objects and an aliased pair is undefined behaviour, not a defined last-writer-wins.
 * - Every input buffer must be readable for the length passed alongside it and aligned for its
 *   element type (8 bytes for a double or a size_t); a length that overstates the allocation is
 *   undefined behaviour, not a QNTT_INVALID_ARG, because the length IS the caller's promise.
 * - On a non-OK return, output memory is unspecified and qntt_last_error() holds a message for
 *   the CALLING THREAD (the slot is thread-local and cleared at the start of every call).
 * - Handles are opaque, refcounted and type-tagged. Each has a *_free that is a no-op on NULL.
 *   Passing a handle of the wrong type or scalar is reported as QNTT_INVALID_ARG, not undefined
 *   behaviour.
 * - Buffer-out functions take (ptr, cap, out_len). Call with cap = 0 and ptr = NULL to query the
 *   required length; a non-zero cap smaller than that length is an error, never a truncation.
 * - A few functions mutate in place (*_vec_block_set, *_vec_round_mut,
 *   *_vec_left_orthonormalize, *_w_set_coo). They require the SOLE reference to
 *   the object; an aliased handle is refused.
 *
 * ---------------------------------------------------------------------------------------------
 * Symbol families
 * ---------------------------------------------------------------------------------------------
 *   qntt_f64_*        abelian (Nup, Ndn) layer, double
 *   qntt_c64_*        abelian (Nup, Ndn) layer, double _Complex
 *   qntt_su2_f64_*    SU(2) spin-adapted layer, double
 *   qntt_su2_c64_*    SU(2) spin-adapted layer, double _Complex
 *
 * Space queries are scalar-independent: qntt_space_* and qntt_su2_space_*.
 *
 * ---------------------------------------------------------------------------------------------
 * Conventions
 * ---------------------------------------------------------------------------------------------
 * - EVERYTHING IS 0-BASED: sites 0..d-1, bonds 0..d, sector labels count electrons strictly to
 *   the left of a bond.
 * - A space is (d, a, b): abelian (d, Nup, Ndn), SU(2) (d, N, 2S).
 * - A sector is the pair (a, b): abelian (nup, ndn), SU(2) (n, 2s).
 * - Rank grids cross as dense (rows x cols) blocks of size_t with the SECOND axis fastest; a
 *   per-bond stack is d+1 such blocks back to back. Query (rows, cols) with *_grid_extent.
 * - Coefficient tensors (t, v) are flat ROW-MAJOR (first axis most significant), d^2 and d^4.
 * - Core blocks are copied COLUMN-MAJOR (Fortran's layout).
 * - Block families are 0 empty, 1 up, 2 down, 3 doubly occupied. On the SU(2) layer 1 and 2 are
 *   the S+1/2 and S-1/2 genealogical COUPLING BRANCHES, not electron spins: both add exactly one
 *   electron.
 *
 * ---------------------------------------------------------------------------------------------
 * Point-group symmetry (the *_pg family)
 * ---------------------------------------------------------------------------------------------
 * - The GROUP is abelian (C1..D2h); the LAYER need not be — both layers carry it, with
 *   mirror-image symbol families.
 * - A point group crosses as the quadruple (pg_kind, orbsym, orbsym_len, target), spelled out
 *   immediately after the space triple. Variable-length data travels alongside as caller-owned
 *   arrays, like the MALS bond-dimension and noise schedules.
 * - orbsym holds 1-BASED MOLPRO IDS, one per site (exactly what an FCIDUMP file writes);
 *   orbsym_len must equal d. target, and every irrep the library RETURNS, are 0-BASED XOR
 *   INDICES, because they are sector labels. Convert with qntt_pg_irrep_from_molpro /
 *   qntt_pg_molpro_from_irrep rather than shifting by hand.
 * - Rank grids gain an irrep axis at |G| > 1: d+1 blocks of (|G| x rows x cols), LAST axis
 *   fastest, so flat[b*(nir*rows*cols) + ir*(rows*cols) + i*cols + j]. At |G| = 1 that is
 *   byte-for-byte the two-axis layout of the non-_pg symbols.
 * - SITE CAP: the *_pg family accepts d <= 128 at QNTT_PG_C1 and d <= 127 at every non-trivial
 *   kind, and returns QNTT_INVALID_ARG above that. This applies to C1 TOO: a _pg call always
 *   builds a point-group-aware space. The non-_pg symbols are the any-d path for trivial-group
 *   work.
 * - Both layers carry the point group: qntt_su2_<scalar>_*_pg plus the scalar-free
 *   qntt_su2_space_*_pg; the qntt_pg_* group tables are layer-independent and shared. Two
 *   asymmetries: no qntt_su2_*_tt_ones_pg (SU(2) has no tt_ones), and tt_state_pg's counterpart
 *   is tt_csf_pg, which DERIVES the total irrep and rejects a target that disagrees.
 * - The KIND is not recoverable from a built object: a space stores the packed per-site table
 *   (nbits + planes), not the kind, and nbits does not determine it (D2, C2v and C2h all have
 *   nbits = 2 with different Molpro numbering). *_vec_point_group / *_ham_point_group therefore
 *   report XOR irreps only. To rebuild a descriptor, pick any kind of the right order (all
 *   same-order abelian groups are numerically identical; only irrep NAMES differ) and map the
 *   XOR indices back with qntt_pg_molpro_from_irrep.
 * - RULE L: qntt_*_vec_ranks, qntt_*_vec_block_get and qntt_*_vec_block_set address the
 *   identity-irrep plane only and now return QNTT_INVALID_ARG on a train with a non-trivial
 *   point group; use their _pg twins. No pre-existing caller can hit this — before this family
 *   existed, no exported symbol could build such a train. This now applies on the SU(2) layer
 *   too (a |G| > 1 SU(2) train first became constructible with the qntt_su2_*_pg family).
 * - The qntt_pg_*_name / qntt_pg_irrep_name string-outs copy exactly *out_len bytes and append a
 *   NUL only when the buffer has room (cap > *out_len). Size-query with cap = 0, then allocate
 *   *out_len + 1 if you want a C string.
 */

#ifndef QNTT_H
#define QNTT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <complex>
typedef std::complex<double> qntt_complex;
extern "C" {
#else
#include <complex.h>
typedef double _Complex qntt_complex;
#endif

/* --- status codes ---------------------------------------------------------------------------- */

#define QNTT_OK             0
#define QNTT_NULL_ARG       1
#define QNTT_INVALID_ARG    2
#define QNTT_PANIC          3   /* a Rust panic, typically a core-crate assertion, was caught */
#define QNTT_IO             4
#define QNTT_NOT_CONVERGED  5

/* --- enum wire codes ------------------------------------------------------------------------- */

#define QNTT_ROUND_ABS 0
#define QNTT_ROUND_REL 1

#define QNTT_CONSTRUCTION_DEFAULT              0
#define QNTT_CONSTRUCTION_MEDIAN               1
#define QNTT_CONSTRUCTION_BIPARTITE            2
#define QNTT_CONSTRUCTION_BIPARTITE_PER_SECTOR 3

#define QNTT_SYM2_NONE      0
#define QNTT_SYM2_PAIRSWAP  1
#define QNTT_SYM2_EIGHTFOLD 2

#define QNTT_SYM3_NONE     0
#define QNTT_SYM3_PAIRPERM 1
#define QNTT_SYM3_FULL48   2

#define QNTT_CG_FLETCHER_REEVES 0
#define QNTT_CG_POLAK_RIBIERE   1

#define QNTT_FAMILY_EMPTY 0
#define QNTT_FAMILY_UP    1
#define QNTT_FAMILY_DN    2
#define QNTT_FAMILY_OCC   3

/* Which Hessian part qntt_*_pd_probe targets. On the f64 families: RR = A + B (indefinite => the
 * point is a saddle of the fixed-rank energy), II = A - B (indefinite => a physical instability
 * against complex perturbations). On the c64 families there is ONE object, the complex Riemannian
 * Hessian h, reached with RR; II is rejected, because -J h J > 0 <=> h > 0 and that single probe
 * certifies the whole RPA precondition. */
#define QNTT_HESS_RR 0
#define QNTT_HESS_II 1

/* Which quantity a CROSS-SECTOR readout reports (QnttSectorOpts.readout). One readout per entry
 * point in the library, so this selects the arm rather than post-filtering a result: TDA reports
 * eig(A^phi), an ENERGY; RPA the signed omega^2 of the pair (A^phi, B^phi). They are different
 * quantities in different units and are not interchangeable. */
#define QNTT_SECTOR_READOUT_TDA 0
#define QNTT_SECTOR_READOUT_RPA 1

/* Which block carries the W-metric (QnttSectorOpts.metric, and the out_metric of both
 * qntt_*_rpa_signed and qntt_*_sector_rpa). AUTO as a REQUEST is II on both, with no probe: the
 * request names the side q = X -+ Y the Weingarten fraction reads (the default paired kernel has no
 * metric) and, under kernel = QNTT_RPA_KERNEL_SIGNED, also the signed solve's lane (no retry).
 * As an ECHO it means no metric was resolved (the run returned before choosing one, or the
 * readout has none).
 * NOT the QNTT_HESS_* codes above: those number RR first and this encoding numbers II first. The
 * asymmetry is frozen, not a choice -- it is what qntt_*_rpa_signed's out_metric has shipped
 * since the metric-selection work landed, and what the Julia RPA_METRICS map reads. */
#define QNTT_RPA_METRIC_II   0
#define QNTT_RPA_METRIC_RR   1
#define QNTT_RPA_METRIC_AUTO 2

/* The kernel of qntt_*_rpa_signed and qntt_*_sector_rpa (the `kernel` REQUEST argument, resp.
 * QnttSectorOpts.kernel) and the out_kernel ECHO: the paired Davidson (0, the library default,
 * so a zeroed request gets it) or the signed W-metric Davidson -- on the f64 layers on the lane `metric` names (RR the swap; II and AUTO the
 * H^ii lane, no probe), on the C64 layers always RR. Any other request is QNTT_INVALID_ARG before
 * any work. From C the echo equals the request (no solver option without a paired arm is
 * reachable through QnttSymplecticLowestOpts). */
#define QNTT_RPA_KERNEL_PAIRED 0
#define QNTT_RPA_KERNEL_SIGNED 1

/* Curvature regularization of qntt_*_rpa_signed and qntt_f64_sector_rpa (curvature_kind;
 * curvature_value is its scale). OFF: the exact Riemannian Hessian, curvature_value ignored.
 * HARD: B -> P.B.P, P the column projector off the right Schmidt directions with Schmidt value
 * below curvature_value. TIKHONOV: B -> D.B.D, D = V diag(s^2/(s^2 + value^2)) V^T. s is
 * absolute; on su2_f64 it is the RAW sigma of qntt_*_vec_schmidt (not sqrt(weight)*sigma): at a
 * singlet one value cuts the same directions on both layers. Under HARD the bond inverse is
 * floored at curvature_value too, under TIKHONOV damped to s/(s^2 + value^2) unless D is the
 * identity to 1e-15 (then the plain inverse, which agrees with it to ~1e-15 there) -- same
 * directions, same operator: no 1/s of a cut or damped value enters the apply. A and the tangent
 * space are untouched. On a shifted channel the same projector acts at each column label t XOR
 * phi -- the phi block of the untyped regularized curvature term. */
#define QNTT_CURVATURE_REG_OFF      0
#define QNTT_CURVATURE_REG_HARD     1
#define QNTT_CURVATURE_REG_TIKHONOV 2

/* Preconditioner of qntt_*_sector_rpa (precond; precond_kappa caps its dynamic range, <= 0 or
 * non-finite = auto 1e3). OFF: none. D0: the channel's fixed diagonal inverse -- of D_A^2 on the
 * RPA readout, of D_A on TDA, D_A the shifted effective-Hamiltonian diagonal minus lambda. AUTO:
 * the library's sector default (currently OFF). */
#define QNTT_SECTOR_PRECOND_OFF  0
#define QNTT_SECTOR_PRECOND_D0   1
#define QNTT_SECTOR_PRECOND_AUTO 2

/* Which route builds a channel's property covector (QnttSectorAlphaOpts.route,
 * QnttSectorMomentOpts.route). MATERIALIZED is the only route a shifted channel HAS:
 * project_hmul_env needs the property as a SparseHamiltonian on x's OWN space, which a
 * cross-irrep property is not. SHIFTED is named here -- and REFUSED with QNTT_INVALID_ARG --
 * because it is PropRoute's own default in the joint layer, so a caller reusing that vocabulary
 * must get a message rather than a panic deep inside the solve. */
#define QNTT_PROP_ROUTE_SHIFTED      0
#define QNTT_PROP_ROUTE_MATERIALIZED 1

/* Which physics refusal a cross-sector OBSERVABLE returned, written through out_refusal on every
 * path -- INCLUDING every caller-error path, and BEFORE any other argument is inspected, so
 * *out_refusal is defined even when the call returns QNTT_NULL_ARG for a null handle, a null opts
 * or a null out_info. The single exception is out_refusal == NULL itself, which is QNTT_NULL_ARG
 * with nothing to write. NONE on QNTT_OK. Every non-NONE value comes back with
 * QNTT_NOT_CONVERGED and the full diagnosis in qntt_last_error(); no out-buffer and no out_len is
 * written on those paths. These are a statement about the STATE; a caller error is
 * QNTT_INVALID_ARG instead, and no seventh STATUS was added because the six above are frozen and
 * a new one would break every existing caller's exhaustive switch. */
#define QNTT_SECTOR_REFUSAL_NONE                  0
#define QNTT_SECTOR_REFUSAL_NOT_TT_CRITICAL       1
#define QNTT_SECTOR_REFUSAL_HESSIAN_INDEFINITE    2
#define QNTT_SECTOR_REFUSAL_SYMPLECTIC_UNSTABLE   3
#define QNTT_SECTOR_REFUSAL_SOLVE_NOT_CONVERGED   4
#define QNTT_SECTOR_REFUSAL_UNCERTIFIED_NEGATIVE  5
/* 6 reserved (a multi-class flux -- reported as QNTT_INVALID_ARG, never here). */
#define QNTT_SECTOR_REFUSAL_NEGATIVE_OMEGA        7
#define QNTT_SECTOR_REFUSAL_NEAR_RESONANCE        8
#define QNTT_SECTOR_REFUSAL_INDEFINITE_ABOVE_POLE 9
/* 7..9 are written only by a frequency-dependent entry point, and none exists yet: the three are
   defined and mapped now so that when qntt_*_sector_alpha_omega lands its refusals are already
   DISCRIMINATED rather than crossing as a bare QNTT_NOT_CONVERGED. Do not renumber; a shipped
   caller switches on these. */

/* PD-probe verdicts. WITNESS_NEGATIVE is a PROOF (a fresh-apply Rayleigh quotient below zero,
 * valid however unconverged the Davidson was); LIKELY_PD is evidence with a margin; INCONCLUSIVE
 * means only "no negative direction found in this budget" and is NOT a positive-definiteness
 * claim. All three return QNTT_OK: the verdict is a diagnosis about the state, not a status of the
 * call. Read out_is_pd for the tri-state (1 PD, 0 indefinite, -1 inconclusive) rather than
 * collapsing INCONCLUSIVE into a boolean, and always read out_residual next to out_lambda -- a
 * LIKELY_PD whose residual is within an order of lambda says nothing. out_residual is UNSPECIFIED
 * (NaN) on WITNESS_NEGATIVE. */
#define QNTT_PD_WITNESS_NEGATIVE 0
#define QNTT_PD_LIKELY_PD        1
#define QNTT_PD_INCONCLUSIVE     2

/* Joint (TT + orbital) symplectic spectrum status, qntt_*_joint_rpa. Every one returns QNTT_OK:
 * they are diagnoses about the state. STABLE: both Hessian blocks PD, omega ascending. RESONANT:
 * exactly one block indefinite; omega carries sign(w^2)*sqrt|w^2|, a negative entry being an
 * imaginary pair (a resonance, not a failure). UNSTABLE: both blocks indefinite, no omega.
 * REFUSED: the point failed the joint context's own criticality / PD gate (message says which). */
#define QNTT_JOINT_RPA_STABLE   0
#define QNTT_JOINT_RPA_RESONANT 1
#define QNTT_JOINT_RPA_UNSTABLE 2
#define QNTT_JOINT_RPA_REFUSED  3

/* Per-iteration series of a joint run, qntt_casscf_diag_series. */
#define QNTT_CASSCF_SERIES_E_HISTORY     0
#define QNTT_CASSCF_SERIES_G_ORB_KEPT    1
#define QNTT_CASSCF_SERIES_G_ORB_FULL    2
#define QNTT_CASSCF_SERIES_G_ORB_DROPPED 3
#define QNTT_CASSCF_SERIES_TT_GRADNORMS  4
#define QNTT_CASSCF_SERIES_ORB_SPECTRUM  5
#define QNTT_CASSCF_SERIES_TR_RADII      6
#define QNTT_CASSCF_SERIES_RHO_HISTORY   7

/* GAD Weingarten-model arms (QnttGadOptions.w_model) and Sigma^-1 floor kinds (.floor_kind). */
#define QNTT_GAD_W_SWITCH  0   /* the default: term-A only above b_switch_tol, exact A+B below */
#define QNTT_GAD_W_DAMPED  1   /* A + B_s; w_damp_s > 0 fixes s, <= 0 selects the tracked rule */
#define QNTT_GAD_W_FULL    2   /* the exact A + B everywhere */
#define QNTT_GAD_FLOOR_DEFAULT  0   /* the context's own hard floor (1e-12) */
#define QNTT_GAD_FLOOR_HARD     1
#define QNTT_GAD_FLOOR_TIKHONOV 2
/* GAD rank adaptation (QnttGadOptions.rank_adapt). A launch-zero sector is never opened under
 * any mode, so a sector-pinned (CAS) grid stays pinned. */
#define QNTT_GAD_RANK_FIXED   0   /* the default: the declared grid, ranks <= declared */
#define QNTT_GAD_RANK_BOTH    1   /* grow (pooled total bond dimension <= chi, 0 = uncapped) and shrink */
#define QNTT_GAD_RANK_SHRINK  2   /* shrink only: the manifold-border arm */

/* Multi-state DMRG wire codes.
 * SsMode  (QnttStateSpecificOptions.mode, QnttExcitedOptions.mode): PROJECT leaves the local
 *   operator alone and searches the orthogonal complement; its `penalty` is recorded 0.0.
 * PenaltyRule travels as a (rule, value) PAIR. Under AUTO the value is a placeholder and is
 *   ignored; under FIXED the weight is UNGUARDED -- no floor, no escalation -- and a weight
 *   below the gap to the target has the lower state as its own minimizer. FIXED with value 0.0
 *   is legal and means the UNPENALIZED sweep, which converges back onto the lower states.
 * SsStart (QnttExcitedOptions.ss_start): SA (0) starts every rung from that root's
 *   state-average state. X0 (1) is reserved and refused with QNTT_INVALID_ARG. */
#define QNTT_SS_MODE_PENALTY  0
#define QNTT_SS_MODE_PROJECT  1
#define QNTT_PENALTY_AUTO     0
#define QNTT_PENALTY_FIXED    1
#define QNTT_SS_START_SA      0
#define QNTT_SS_START_X0      1

/* Abelian point-group kinds. The order is the core crate's PointGroupKind declaration order,
 * which also indexes its Molpro<->XOR and name tables: frozen, never renumbered.
 * QNTT_PG_AUTO is accepted ONLY by qntt_*_fcidump_read_pg and means "guess from max(ORBSYM)"
 * (1 -> C1, 2 -> Cs, 3..4 -> C2v, 5..8 -> D2h). The guess affects irrep NAMES only: all
 * same-order abelian groups share one product table, so no energy or selection rule depends on
 * it. */
#define QNTT_PG_AUTO (-1)
#define QNTT_PG_C1   0
#define QNTT_PG_CI   1
#define QNTT_PG_C2   2
#define QNTT_PG_CS   3
#define QNTT_PG_D2   4
#define QNTT_PG_C2V  5
#define QNTT_PG_C2H  6
#define QNTT_PG_D2H  7

/* --- option structs -------------------------------------------------------------------------- */
/* Fill these with the corresponding qntt_*_options_default() rather than by hand: the defaults
 * come from the Rust drivers. */

typedef struct {
  double eps;
  size_t max_iter;
  int32_t verbose;
} QnttAlsOptions;

typedef struct {
  double eps;
  size_t max_iter;
  int32_t verbose;
  uint64_t seed;   /* noise draws; irrelevant when the noise schedule is all-zero */
  int32_t noise_type; /* 0 = random, 1 = perturbative (White density-matrix noise, the default) */
  double conv_tol;    /* relative energy-improvement stop; 0 = use eps */
} QnttMalsOptions; /* the rank and noise schedules are passed alongside, being variable-length */

/* One outer iteration's diagnostics. `discarded_weight` is the relative dropped energy at the
 * binding bond — the standard DMRG rank-convergence diagnostic. */
typedef struct {
  double energy;
  double discarded_weight;
  size_t max_bond_dim; /* largest pooled total bond dimension reached in this sweep */
  size_t bond_dim;     /* the pooled cap in force this iteration (0 = uncapped) */
  double noise;
} QnttMalsSweep;

typedef struct {
  double dtau;
  double eps;
  double trunc;
  size_t max_sweeps;
  size_t chi; /* pooled total bond-dimension cap (SU(2): reduced multiplets); 0 = none */
  double exptol;
  int32_t verbose;
} QnttTdvpOptions;

typedef struct {
  double tol;
  size_t max_iter;
} QnttLanczosOptions;

typedef struct {
  double eps;
  size_t max_iter;
  double gradtol;
  uint64_t seed;
  int32_t variant; /* QNTT_CG_* */
} QnttRiemannianOpts;

/* The matrix-free RPA options of the signed entries. max_iter is the budget in PAIR-EQUIVALENTS
 * (the paired Davidson builds at most max_iter (A, B) apply pairs, the signed W-metric kernel
 * ceil(max_iter / 2) basis vectors of 2 applies each); tol is relative in omega units (pencil
 * residual |Hz - omega Sigma z| / |z| <= tol * omega_max, or tol * |omega| for an imaginary or
 * complex mode). Defaults: max_iter 800, tol 1e-8. */
typedef struct {
  size_t max_iter; /* budget in pair-equivalents ((A, B) apply pairs); default 800 */
  double tol;      /* relative, in omega units; default 1e-8 */
} QnttSymplecticLowestOpts;

/* Cross-sector (shifted-channel) readout options; see QNTT_DECLARE_SECTOR below. `seed` is NOT a
 * field: it is a call parameter of qntt_*_sector_rpa, the qntt_*_rpa_signed convention. Eight-byte
 * fields first, then an EVEN number of int32_t, so sizeof is the sum of the field sizes. */
typedef struct {
  /* TDA: outer block-Davidson iterations. RPA: the budget in pair-equivalents for either kernel
     ((A, B) apply pairs: the paired kernel's basis vectors, or ceil(max_iter/2) signed basis
     vectors of 2 Hessian-class applies each) -- the same accounting
     QnttSymplecticLowestOpts.max_iter carries. */
  size_t max_iter;
  double tol;
  /* RPA: thick-restart bound on the live basis; 0 = the kernel's auto max(8(k+b), 32) for start
     block b, clamped to dim T^phi. Too tight a bound limit-cycles on hard spectra. Ignored by the
     TDA readout. */
  size_t max_subspace;
  int32_t readout;  /* QNTT_SECTOR_READOUT_* */
  int32_t metric;   /* QNTT_RPA_METRIC_*: the f_W side under the PAIRED kernel, also the lane
                       under SIGNED (AUTO = II, no probe); ignored by the TDA readout */
  int32_t kernel;   /* QNTT_RPA_KERNEL_*: PAIRED (0, the default) or SIGNED; ignored by TDA */
  int32_t reserved; /* explicit, so sizeof == the sum of the field sizes; write 0 */
} QnttSectorOpts;

/* Cross-sector OBSERVABLE options; see QNTT_DECLARE_SECTOR_OBS below. TWO structs, one per entry
 * point, mirroring the library's own SectorStaticOpts / SectorMomentOpts one-to-one: no field on
 * either is inert on the entry point that receives it. Both lay their eight-byte fields out first
 * and then an EVEN number of int32_t, so sizeof equals the sum of the field sizes with NO implicit
 * padding and the Julia mirrors need no alignment guessing -- which is what the explicit
 * `reserved` tail of QnttSectorMomentOpts is for. */
typedef struct {
  double tol;                 /* PCG relative residual, default 1e-8 */
  size_t max_iter;            /* PCG cap, default 500 */
  double tt_grad_tol;         /* TT-criticality gate, RELATIVE to max(1,|lambda|), default 1e-6;
                                 +inf is the documented diagnosis-only escape */
  double sym_assert_rel;      /* read ONLY when has_sym_assert_rel != 0 */
  int32_t has_sym_assert_rel; /* 0 => the library's own 100*tol, RECOMPUTED from the actual tol */
  int32_t route;              /* QNTT_PROP_ROUTE_*; MATERIALIZED is the only legal value today */
} QnttSectorAlphaOpts;

typedef struct {
  double tt_grad_tol;    /* as above; the moments consume B^phi through the Grams */
  double degeneracy_tol; /* RELATIVE omega clustering window, default 1e-8 -- a statement about
                            which numbers EXIST, not a tolerance on the answer */
  double psd_tol;        /* Williamson PSD tolerance, default 1e-10 */
  int32_t route;         /* QNTT_PROP_ROUTE_*; MATERIALIZED is the only legal value today */
  int32_t reserved;      /* explicit, so sizeof == the sum of the field sizes; write 0 */
} QnttSectorMomentOpts;

/* Scalars of qntt_*_sector_alpha. The per-property numbers are the out_props table beside it. */
typedef struct {
  double alpha_defect;  /* max|raw[u,v] - raw[v,u]| BEFORE symmetrization; 0 for one property */
  double sigma_min;     /* min Schmidt value at the base point -- quote it with every number */
  size_t dim;           /* dim T^phi, NOT the target sector's dimension */
  int32_t phi;
  int32_t target;       /* Gamma0 XOR phi */
} QnttSectorAlphaInfo;

/* Scalars and SHAPE of qntt_*_sector_transition_moments. On the size-probe pass every one of
   these is written and no buffer is. n_modes/n_neg/n_clusters are the three lengths a caller
   cannot predict; all three are bounded by dim, which qntt_*_sector_rpa reports for free at
   k = 0. */
typedef struct {
  double orth_error;      /* max|<b_i,b_j> - delta_ij| over the channel basis, computed here */
  double sigma_min;
  size_t dim;
  size_t n_modes;         /* REAL modes only: the length of out_omega and of each moment row */
  size_t n_neg;           /* imaginary modes; > 0 forces has_alpha_sos = 0 */
  size_t n_clusters;      /* degenerate omega clusters; out_clusters holds 2*n_clusters entries */
  int32_t metric;         /* QNTT_RPA_METRIC_II / _RR; _AUTO on a structurally empty channel */
  int32_t has_alpha_sos;  /* 0 <=> n_neg > 0; out_alpha_sos is NOT written when 0 */
  int32_t complete_basis; /* 1 on the dense route: the sums are genuine, not partial */
  int32_t phi;
  int32_t target;
  int32_t reserved;       /* explicit, so sizeof == the sum of the field sizes; reads 0 */
} QnttSectorMomentsInfo;

/* --- joint (CASSCF-style) stack and GAD ------------------------------------------------------- */

/* core | active | virtual split of the orbital chain, in site order; must sum to d. */
typedef struct {
  size_t n_core;
  size_t n_active;
  size_t n_virt;
} QnttCasPartition;

/* The exposed subset of the joint driver's options; everything else stays at the Rust default
 * and the orbital pair mask is derived from the partition inside the library. */
typedef struct {
  size_t orb_max_iter;
  double orb_grad;          /* threshold on the KEPT orbital gradient norm */
  double tt_grad_tol;       /* TT criticality gate, relative to max(1, |lambda|) */
  double orb_floor;         /* keep-rule floor on the orbital metric N, relative */
  int32_t tt_preserve_grid; /* nonzero (THE DEFAULT HERE) keeps the pinned grid through TT legs */
  int32_t construction;     /* QNTT_CONSTRUCTION_*; 0 = layer default; ignored on SU(2) */
  int32_t verbose;
} QnttCasscfOptions;

/* Flat diagnostics of a joint run. Publication licence is converged && !rank_limited. */
typedef struct {
  int32_t converged;
  int32_t rank_limited;
  int32_t stalled_at_floor;
  size_t iters;
  size_t n_eff;             /* kept (non-redundant) orbital coordinates */
  size_t ls_build_count;
  size_t tt_legs;
  size_t rejections;
  double sigma_min;
  double u_orth_dev;
  double g_dropped_final;
} QnttCasscfInfo;

/* GAD options. w_model / w_damp_s / b_switch_tol spell the Weingarten model (QNTT_GAD_W_*);
 * floor_kind / floor the Sigma^-1 floor (QNTT_GAD_FLOOR_*); rank_adapt / chi / rank_eps /
 * rank_period the optional rank adaptation (QNTT_GAD_RANK_*), a base-point event between
 * accepted steps. */
typedef struct {
  double gradtol;
  size_t max_steps;         /* ACCEPTED steps; rejected trials do not consume it */
  double alpha0;
  double alpha_min;
  double alpha_max;
  size_t k_davidson;
  int32_t w_model;
  double w_damp_s;
  double b_switch_tol;
  int32_t floor_kind;
  double floor;
  int32_t certify;          /* run the two-probe index-1 certificate at gradtol */
  size_t probe_budget;      /* certificate AND cold-refresh Rayleigh-Ritz budget */
  double schmidt_floor;
  uint64_t seed;
  int32_t preserve_grid;    /* REQUIRED nonzero on a sector-pinned (CAS) grid */
  int32_t rank_adapt;       /* QNTT_GAD_RANK_* (0 = fixed grid, the default) */
  size_t chi;               /* RANK_BOTH: pooled total bond-dimension cap of a growth round
                                (SU(2): reduced multiplets); 0 = uncapped */
  double rank_eps;          /* adaptation energy budget (relative); positive, finite */
  size_t rank_period;       /* adapt every rank_period accepted steps (>= 1) */
  double border_gain_max;   /* manifold border: no certificate when the endpoint's undamped
                               Weingarten gain |B V|/|A V| exceeds this (default 1.0) */
} QnttGadOptions;

/* Flat GAD diagnostics. The two optional fields carry a has_* flag; the value is unspecified
 * when the flag is zero. lambda is ELECTRONIC. */
typedef struct {
  int32_t converged;        /* criticality AND (with certify) the index-1 certificate */
  int32_t index1_certified;
  int32_t has_index_estimate;
  size_t index_estimate;
  size_t steps;
  size_t applies;
  double gnorm;
  double lambda;
  int32_t has_eig1;
  double eig1;
  double v_overlap;
  double b_gain_max;
  double min_schmidt_min;
  int32_t schmidt_floor_hit;
  size_t rejected_steps;
  size_t relatches;
  size_t regime_switches;
  double radial_overlap_max;
  double branch_gap;
  int32_t has_witness_overlap;   /* certificate ran and found a negative direction */
  double witness_overlap;        /* |<w, V>|, probe-1 witness vs the tracked mode */
  int32_t tracked_mode_is_lowest;/* 0 with index1_certified: the saddle was reached on another
                                    negative branch (the certificate classifies the POINT) */
  size_t grid_changes;           /* accepted steps whose grid differed from the previous one */
  int32_t border_verdict;        /* gradtol reached on a manifold border: no certificate, no index */
  double b_gain_end;             /* undamped |B V|/|A V| at the last refresh (NaN before any) */
} QnttGadInfo;

/* --- multi-state DMRG option and diagnostic structs ------------------------------------------- */
/* The weight / bond-dimension / noise schedules are variable-length and travel alongside as
 * (ptr, len) pairs; 0 in maxsubspace or restart_keep means "the driver's derived default". */
typedef struct {
  QnttMalsOptions mals;
  int32_t homing;           /* 0 (default) labels roots by energy at every bond; 1 follows them */
  int32_t reorthonormalize; /* 1 (default): Loewdin-orthonormalize the truncated centres */
  double scale;             /* relative size of the start perturbation of centres 1..n (0.1) */
  size_t maxsubspace;
  size_t restart_keep;
} QnttStateAverageOptions;

/* One state-average outer iteration. The per-root energies and discarded weights ride the
 * companion buffers out_sweep_energies / out_sweep_discarded, indexed [s*nroots + i]. */
typedef struct {
  double energy;            /* sum_i w_i lambda_i -- what the sweep minimizes */
  double max_dev;           /* max_i |dlambda_i| / |lambda_i| over the roots */
  double overlap_dev;       /* drift from orthogonality before the Loewdin step */
  double discarded_weight;  /* POOLED; see the SU(2) caveat under QNTT_DECLARE_LAYER */
  size_t max_bond_dim;      /* largest pooled total bond dimension reached in this sweep */
  size_t bond_dim;          /* the pooled cap in force this iteration (0 = uncapped) */
  double noise;
  size_t numops;
} QnttSaSweep;

typedef struct {
  QnttMalsOptions mals;     /* unlike the state average, this path ACCEPTS noise_type = 0 */
  int32_t mode;             /* QNTT_SS_MODE_* */
  int32_t penalty_rule;     /* QNTT_PENALTY_* */
  double penalty_value;     /* read under QNTT_PENALTY_FIXED only */
  size_t roots;             /* local roots per bond; > 1 homes onto the incoming block */
  size_t maxsubspace;
  size_t restart_keep;
} QnttStateSpecificOptions;

/* One state-specific outer iteration. The per-lower-state overlaps |<psi_j, x>| ride the
 * companion buffer out_sweep_overlaps, indexed [s*nlower + j] in the caller's lower order. */
typedef struct {
  double energy;            /* the PLAIN Rayleigh quotient of H alone -- the reported energy */
  double penalized;         /* the penalized Ritz value the sweep actually minimizes */
  double discarded_weight;
  size_t max_bond_dim;      /* largest pooled total bond dimension reached in this sweep */
  size_t bond_dim;          /* the pooled cap in force this iteration (0 = uncapped) */
  double noise;
  double penalty;           /* the weight in force; 0.0 in project mode and on the delegate */
  size_t swaps;
  size_t numops;            /* 0 on the delegate path (nlower = 0 with roots = 1) */
  size_t skipped;
  int32_t converged;
} QnttSsSweep;

/* Excited-state ladder options. weights / bond_dims / noise / ss_noise travel alongside, and an
 * EMPTY schedule keeps the driver's own value -- which is NOT uniformly "off": empty weights are
 * 1/n and empty bond_dims the default ramp of `rank`, but empty noise is the built-in nine-entry
 * ramp and empty ss_noise is {0.0}. Pass {0.0} explicitly for a noise-free ladder. bond_dims is a
 * pooled total bond dimension schedule (SU(2): reduced multiplets), like `rank`. The polish
 * CG's adaptive-rounding block is not exposed: it is resolved from x0's own space and is inert on
 * this code path (the polish gradient is the exact environment contraction). */
typedef struct {
  size_t nroots;            /* roots returned, counting the ground state; >= 1 */
  size_t rank;              /* pooled total bond dimension (SU(2): reduced multiplets) cap of the
                                sweeps and the polish manifold; 0 = full rank */
  double eps;
  size_t max_iter;
  int32_t noise_type;       /* 0 = random (refused here), 1 = perturbative (the default) */
  uint64_t seed;
  int32_t mode;             /* QNTT_SS_MODE_* */
  int32_t penalty_rule;     /* QNTT_PENALTY_* */
  double penalty_value;     /* also fixes the POLISH weight */
  int32_t ss_start;         /* QNTT_SS_START_* */
  int32_t polish;           /* 0 returns the sweep's own state; gradnorm is then NaN */
  double lower_grad_tol;
  double trim_tol;
  size_t trim_passes;
  double cg_eps;            /* 1e-13: the pipeline override, not RiemannianOpts' own 1e-4 */
  size_t cg_max_iter;
  double cg_gradtol;        /* inert here; carried for API stability */
  uint64_t cg_seed;
  int32_t cg_variant;       /* QNTT_CG_* */
  size_t cg_restarts;
  double polish_tol;
  double grad_gate;
  double overlap_tol;
  int32_t keep_sa;          /* nonzero also fills out_sa_states */
  int32_t verbose;
} QnttExcitedOptions;

/* The ladder's scalars. The state-average and state-specific SWEEP TRACES are not exported here
 * (the state-specific ones are ragged): run qntt_*_state_average_mals / _state_specific_mals for
 * them. */
typedef struct {
  double overlap_dev;       /* the largest off-diagonal entry of the returned Gram */
  size_t swaps;
  int32_t enriched;         /* stage 0's enrichment iteration ran (a rank-1 start) */
  int32_t licence;          /* convergence AND orthogonality AND ordering -- NEVER completeness */
} QnttExcitedInfo;

/* One row of the ladder's per-root record (out_root_info holds nroots of these). */
typedef struct {
  double sa_energy;         /* the state-average stage's energy for this root */
  double ss_energy;         /* the state-specific sweep's own energy, BEFORE the polish */
  double j_ss;
  double j_pol;
  double gradnorm;          /* the exact penalized gate; NaN when polish == 0 */
  double penalty;           /* 0.0 when this root had no lower states */
  size_t passes;
  int32_t stalled;
  int32_t converged;
  int32_t swapped;
} QnttExcitedRoot;

/* --- global ---------------------------------------------------------------------------------- */

int32_t qntt_version(char *buf, size_t cap, size_t *out_len);
int32_t qntt_last_error(char *buf, size_t cap, size_t *out_len);
int32_t qntt_init(size_t nthreads); /* 0 = let the library choose; first call wins */
int32_t qntt_num_threads(size_t *out);

int32_t qntt_als_options_default(QnttAlsOptions *out);
int32_t qntt_mals_options_default(QnttMalsOptions *out);
int32_t qntt_tdvp_options_default(QnttTdvpOptions *out);
int32_t qntt_lanczos_options_default(QnttLanczosOptions *out);
int32_t qntt_riemannian_options_default(QnttRiemannianOpts *out);
int32_t qntt_symplectic_options_default(QnttSymplecticLowestOpts *out);
int32_t qntt_sector_options_default(QnttSectorOpts *out);
int32_t qntt_sector_alpha_options_default(QnttSectorAlphaOpts *out);
int32_t qntt_sector_moment_options_default(QnttSectorMomentOpts *out);
int32_t qntt_casscf_options_default(QnttCasscfOptions *out);
int32_t qntt_gad_options_default(QnttGadOptions *out);
int32_t qntt_state_average_options_default(QnttStateAverageOptions *out);
int32_t qntt_state_specific_options_default(QnttStateSpecificOptions *out);
int32_t qntt_excited_options_default(QnttExcitedOptions *out);

/* Diagnostics handle returned by qntt_*_casscf (layer-independent): the per-iteration series
 * (QNTT_CASSCF_SERIES_*, buffer-out) and the refusal message (string-out, empty if none). */
void qntt_casscf_diag_free(void *h);
int32_t qntt_casscf_diag_series(const void *h, int32_t which, double *out, size_t cap,
                                size_t *out_len);
int32_t qntt_casscf_diag_reason(const void *h, char *buf, size_t cap, size_t *out_len);

/* (Nup, Ndn) from an electron count and 2*Sz; abelian only. */
int32_t qntt_space_from_n_two_sz(size_t d, size_t n, int64_t two_sz, size_t *out_nup,
                                 size_t *out_ndn);

/* --- space queries, one family per layer ------------------------------------------------------ */

#define QNTT_DECLARE_SPACE(P)                                                                     \
  int32_t P##_check(size_t d, size_t a, size_t b);                                                \
  int32_t P##_grid_extent(size_t d, size_t a, size_t b, size_t *out_rows, size_t *out_cols);      \
  int32_t P##_bond_sectors(size_t d, size_t a, size_t b, size_t bond, size_t *out_a,              \
                           size_t *out_b, size_t cap, size_t *out_len);                           \
  /* max_ranks: per-sector path-count caps, one per bond sector, not a pooled bond dimension. */   \
  int32_t P##_max_ranks(size_t d, size_t a, size_t b, size_t *out, size_t cap, size_t *out_len);

QNTT_DECLARE_SPACE(qntt_space)
QNTT_DECLARE_SPACE(qntt_su2_space)

/* --- point-group tables (layer-independent) and abelian space queries -------------------------- */

int32_t qntt_pg_order(int32_t kind, size_t *out_order);          /* 1, 2, 4 or 8 */
int32_t qntt_pg_default_kind(size_t order, int32_t *out_kind);   /* the reader's per-order guess */
int32_t qntt_pg_kind_from_name(const char *name, int32_t *out_kind); /* ASCII case-insensitive */
int32_t qntt_pg_kind_name(int32_t kind, char *buf, size_t cap, size_t *out_len);
int32_t qntt_pg_irrep_name(int32_t kind, int32_t irrep, char *buf, size_t cap, size_t *out_len);
int32_t qntt_pg_irrep_from_molpro(int32_t kind, int32_t molpro_id, int32_t *out_irrep);
int32_t qntt_pg_molpro_from_irrep(int32_t kind, int32_t irrep, int32_t *out_molpro_id);
int32_t qntt_pg_orbsym_to_irreps(int32_t kind, const uint8_t *orbsym, size_t n, uint8_t *out_irreps,
                                 size_t cap, size_t *out_len);

/* The descriptor quadruple is (pg_kind, orbsym, orbsym_len, target); see the Conventions block. */
int32_t qntt_space_check_pg(size_t d, size_t a, size_t b, int32_t pg_kind, const uint8_t *orbsym,
                            size_t orbsym_len, int32_t target);
int32_t qntt_space_grid_extent_pg(size_t d, size_t a, size_t b, int32_t pg_kind,
                                  const uint8_t *orbsym, size_t orbsym_len, int32_t target,
                                  size_t *out_rows, size_t *out_cols, size_t *out_nirrep);
int32_t qntt_space_bond_sectors_pg(size_t d, size_t a, size_t b, int32_t pg_kind,
                                   const uint8_t *orbsym, size_t orbsym_len, int32_t target,
                                   size_t bond, size_t *out_a, size_t *out_b, uint8_t *out_irrep,
                                   size_t cap, size_t *out_len);
/* max_ranks_pg: per-sector path-count caps, one per bond sector, not a pooled bond dimension. */
int32_t qntt_space_max_ranks_pg(size_t d, size_t a, size_t b, int32_t pg_kind,
                                const uint8_t *orbsym, size_t orbsym_len, int32_t target,
                                size_t *out, size_t cap, size_t *out_len);

/* --- SU(2) point-group space queries ----------------------------------------------------------- */
/* Same descriptor quadruple; the space triple is (d, N, 2S) and a sector is (n, 2s, irrep). Rank
 * grids are REDUCED (multiplet) ranks, in the same flat layout as the abelian ones with
 * rows = cols = N+1 and 2s fastest. */
int32_t qntt_su2_space_check_pg(size_t d, size_t a, size_t b, int32_t pg_kind,
                                const uint8_t *orbsym, size_t orbsym_len, int32_t target);
int32_t qntt_su2_space_grid_extent_pg(size_t d, size_t a, size_t b, int32_t pg_kind,
                                      const uint8_t *orbsym, size_t orbsym_len, int32_t target,
                                      size_t *out_rows, size_t *out_cols, size_t *out_nirrep);
int32_t qntt_su2_space_bond_sectors_pg(size_t d, size_t a, size_t b, int32_t pg_kind,
                                       const uint8_t *orbsym, size_t orbsym_len, int32_t target,
                                       size_t bond, size_t *out_a, size_t *out_b,
                                       uint8_t *out_irrep, size_t cap, size_t *out_len);
/* max_ranks_pg: per-sector (REDUCED multiplet) path-count caps, not a pooled bond dimension. */
int32_t qntt_su2_space_max_ranks_pg(size_t d, size_t a, size_t b, int32_t pg_kind,
                                    const uint8_t *orbsym, size_t orbsym_len, int32_t target,
                                    size_t *out, size_t cap, size_t *out_len);

/* --- the shared per-(layer, scalar) surface --------------------------------------------------- */

#define QNTT_DECLARE_LAYER(SFX, T)                                                                \
  /* lifetime */                                                                                  \
  void qntt_##SFX##_vec_free(void *x);                                                            \
  void qntt_##SFX##_ham_free(void *h);                                                            \
                                                                                                  \
  /* construction */                                                                              \
  int32_t qntt_##SFX##_vec_zeros(size_t d, size_t a, size_t b, void **out);                       \
  int32_t qntt_##SFX##_vec_from_ranks(size_t d, size_t a, size_t b, const size_t *ranks,          \
                                      size_t ranks_len, void **out);                              \
  int32_t qntt_##SFX##_vec_randn(size_t d, size_t a, size_t b, const size_t *ranks,               \
                                 size_t ranks_len, uint64_t seed, void **out);                    \
  int32_t qntt_##SFX##_vec_clone(const void *x, void **out);                                      \
                                                                                                  \
  /* structure */                                                                                 \
  int32_t qntt_##SFX##_vec_space(const void *x, size_t *out_d, size_t *out_a, size_t *out_b);     \
  int32_t qntt_##SFX##_vec_ranks(const void *x, size_t *out, size_t cap, size_t *out_len);        \
  int32_t qntt_##SFX##_vec_ortho(const void *x, ptrdiff_t *out); /* -1 = not canonical */         \
  int32_t qntt_##SFX##_vec_check(const void *x, int32_t *out);                                    \
  /* out_max_sector_rank: largest single-sector rank over every bond. out_max_bond_rank: largest  \
   * pooled bond dimension over every bond (the TtState accessor, not a sweep-option field). */   \
  int32_t qntt_##SFX##_vec_sizes(const void *x, size_t *out_sites, size_t *out_max_sector_rank,   \
                                 size_t *out_max_bond_rank);                                      \
                                                                                                  \
  /* state diagnostics. Print all three at every stage of a calculation you want to certify,      \
   * not only at the end.                                                                         \
   *                                                                                              \
   * vec_dot_bilinear is the UNCONJUGATED (x,y), as against vec_dot's Hermitian <x,y>. The        \
   * quantity to watch is rho = |(x,x)| / <x,x>, which is 1 exactly when every coefficient        \
   * shares one global phase -- i.e. the state is real up to a phase. A "complex" run that        \
   * reports rho = 1 never left the real invariant subspace (complex conjugation is a symmetry    \
   * of the problem, so real states are invariant under MALS/CG with real environments) and       \
   * every number downstream of it is the real answer in disguise. ||Im x|| is NOT a substitute:  \
   * a global phase is unphysical. Over f64 this is identical to vec_dot.                         \
   *                                                                                              \
   * vec_schmidt returns the singular values at every INTERIOR bond 1..d, pooled over each        \
   * bond's sectors and sorted descending: `out` holds them concatenated, `out_weights` the       \
   * parallel multiplet weights (exactly *out_len of them; the length is shared), `out_counts`    \
   * how many belong to each bond (d-1 entries). Every buffer follows the (ptr, cap) size-query   \
   * protocol.                                                                                    \
   *                                                                                              \
   * SUM RULE, exact at every interior bond of a unit-norm state:                                 \
   *                                                                                              \
   *     sum_i  weight_i * sigma_i^2  =  ||x||^2                                                  \
   *                                                                                              \
   * On the abelian layer every weight is 1, so this is the ordinary sum sigma^2 = 1. On the      \
   * SU(2) layer weight = centerweight(2s_col, 2S_total) = (2s_col+1)/(2S_total+1): each REDUCED  \
   * multiplet contributes its genuine degeneracy; SU(2) truncation reads the same weighted       \
   * values. The UNWEIGHTED sum is NOT a normalization check on SU(2) and its direction           \
   * is not even fixed: below 1 at 2S_total = 0 (every weight >= 1), above 1 at 2S_total > 0      \
   * (sectors with 2s_col < 2S_total weigh less than 1).                                          \
   *                                                                                              \
   * sigma is REDUCED and never pre-weighted. min sigma over the table is the manifold-boundary   \
   * alarm:                                                                                       \
   * the Weingarten curvature scales like ||r_perp||/sigma_min, so a collapsing sigma_min means   \
   * the iterate is walking off the edge of the fixed-rank manifold, where the Hessian is not     \
   * defined and "the probe was inconclusive" and "there is no interior minimum" stop being       \
   * distinguishable.                                                                             \
   *                                                                                              \
   * grad_norm is ||P_T(Hx - lambda x)|| with lambda the Rayleigh quotient at x -- the            \
   * criticality check (the RPA reference criterion is < 1e-9). It is the gradient at x AS        \
   * GIVEN, so it scales with an unnormalized state. */                                           \
  int32_t qntt_##SFX##_vec_dot_bilinear(const void *x, const void *y, T *out);                    \
  int32_t qntt_##SFX##_vec_schmidt(const void *x, double *out, size_t cap, size_t *out_len,       \
                                   double *out_weights, size_t weights_cap,                       \
                                   size_t *out_counts, size_t counts_cap, size_t *out_nbonds);    \
  int32_t qntt_##SFX##_grad_norm(const void *h, const void *x, double *out);                      \
                                                                                                  \
  /* persistence: the QNTT-TT v1 state CACHE, not an archival format. A file                      \
   * records its layer and scalar and is refused by the other family's symbols; an unknown        \
   * format version is refused too -- delete stale caches, do not migrate them. Malformed input   \
   * is QNTT_IO carrying the reader's own message. Unaffected by RULE L: the format stores the    \
   * point group, so a |G| > 1 train round-trips exactly and there is no _pg twin. */             \
  int32_t qntt_##SFX##_vec_save(const void *x, const char *path);                                 \
  int32_t qntt_##SFX##_vec_load(const char *path, void **out);                                    \
                                                                                                  \
  /* blocks (column-major; an absent block reports rows = cols = 0) */                            \
  int32_t qntt_##SFX##_vec_block_get(const void *x, size_t site, int32_t family, size_t sa,       \
                                     size_t sb, T *out, size_t cap, size_t *out_rows,             \
                                     size_t *out_cols);                                           \
  int32_t qntt_##SFX##_vec_block_set(void *x, size_t site, int32_t family, size_t sa, size_t sb,  \
                                     const T *src, size_t rows, size_t cols);                     \
                                                                                                  \
  /* algebra */                                                                                   \
  int32_t qntt_##SFX##_vec_norm(const void *x, double *out);                                      \
  int32_t qntt_##SFX##_vec_dot(const void *x, const void *y, T *out);                             \
  int32_t qntt_##SFX##_vec_scaled(const void *x, const T *a, void **out);                         \
  int32_t qntt_##SFX##_vec_add(const void *x, const void *y, int32_t sign, void **out);           \
  /* chi: pooled total bond-dimension cap (SU(2): reduced multiplets); 0 = uncapped. */            \
  int32_t qntt_##SFX##_vec_round(const void *x, double eps, size_t chi, int32_t mode,             \
                                 void **out);                                                     \
  int32_t qntt_##SFX##_vec_round_mut(void *x, double eps, size_t chi, int32_t mode);              \
  int32_t qntt_##SFX##_vec_left_orthonormalize(void *x, int32_t keep_rank, double *out_norm);     \
  int32_t qntt_##SFX##_vec_round_sum(const T *alpha, const void *const *vecs, size_t n,           \
                                     double tol, void **out);                                     \
                                                                                                  \
  /* Hamiltonian */                                                                               \
  int32_t qntt_##SFX##_ham_new(size_t d, size_t a, size_t b, const T *t, size_t t_len,            \
                               const T *v, size_t v_len, const void *w, double eps,               \
                               int32_t construction, void **out);                                 \
  int32_t qntt_##SFX##_ham_space(const void *h, size_t *out_d, size_t *out_a, size_t *out_b);     \
  int32_t qntt_##SFX##_ham_bond_dims(const void *h, size_t *out, size_t cap, size_t *out_len);    \
  int32_t qntt_##SFX##_ham_apply(const void *h, const void *x, void **out);                       \
  /* NOTE: expectation is <x,Hx> WITHOUT dividing by <x,x> (the `rayleigh_quotient`              \
   * naming); ham_rayleigh does divide. */                                                        \
  int32_t qntt_##SFX##_ham_expectation(const void *h, const void *x, T *out);                     \
  int32_t qntt_##SFX##_ham_rayleigh(const void *h, const void *x, T *out);                        \
  int32_t qntt_##SFX##_ham_xthy(const void *x, const void *h, const void *y, T *out);             \
                                                                                                  \
  /* solvers */                                                                                   \
  int32_t qntt_##SFX##_als(const void *h, const void *x0, const QnttAlsOptions *opts,             \
                           T *out_lambda, void **out_x);                                          \
  /* `bond_dims` is a pooled total bond dimension schedule (SU(2): reduced multiplets); it must   \
   * be non-empty ([0] = uncapped). `noise` may be NULL/0-length, meaning noise-free. A schedule   \
   * MUST end at 0.0: a noisy sweep perturbs the state, so only the final noise-free sweeps carry  \
   * a variational energy. `out_sweeps` is optional (pass NULL/0); `out_nsweeps` always reports    \
   * the count. */                                                                                 \
  int32_t qntt_##SFX##_mals(const void *h, const void *x0, const QnttMalsOptions *opts,           \
                            const size_t *bond_dims, size_t bond_dims_len,                        \
                            const double *noise, size_t noise_len, double *out_lambda,            \
                            void **out_x, QnttMalsSweep *out_sweeps, size_t sweeps_cap,           \
                            size_t *out_nsweeps);                                                 \
  int32_t qntt_##SFX##_tdvp(const void *h, const void *x0, const QnttTdvpOptions *opts,           \
                            double *out_lambda, void **out_x);                                    \
  int32_t qntt_##SFX##_riemannian(const void *h, const void *x0,                                  \
                                  const QnttRiemannianOpts *opts, int32_t use_cg,                 \
                                  double *out_lambda, void **out_x);                              \
  int32_t qntt_##SFX##_lanczos(const void *h, const void *x0, const QnttLanczosOptions *opts,     \
                               double *out_lambda, void **out_x, int32_t *out_breakdown,          \
                               size_t *out_iters, double *hist_lambdas, double *hist_res,         \
                               size_t *hist_ranks, size_t *hist_rounded_ranks, size_t hist_cap);  \
  /* multi-state DMRG. Every energy is a double on every family, C64 included: the returned       \
   * levels are real. Schedules travel alongside as (ptr, len). The state-average and             \
   * state-specific bond_dims MUST be non-empty ([0] = uncapped) and their noise MAY be empty     \
   * (noise-free), exactly as qntt_*_mals; the LADDER inverts that for noise (see                 \
   * QnttExcitedOptions). out_states / out_roots / out_sa_states hold exactly nroots handles --   \
   * a shorter capacity is QNTT_INVALID_ARG before any work, never a truncation -- and are        \
   * emitted only after every other output has been written, so a refused call leaks nothing.     \
   * The sweep buffers keep qntt_*_mals's idiom instead: optional, truncating to sweeps_cap,      \
   * with out_nsweeps always reporting the full count; their per-root/per-lower companions hold   \
   * sweeps_cap*nroots (state average) or sweeps_cap*nlower (state specific) entries, sweep-major.\
   * A rank-1 determinant/CSF start is REFUSED by the two sweeps (QNTT_PANIC with the driver's    \
   * message) but ENRICHED by the ladder -- the same start, two documented outcomes.              \
   * excited_states' licence certifies convergence, orthogonality and ordering; it NEVER certifies\
   * completeness, so a licensed ladder can skip a level (rerun with nroots + 1 and compare the  \
   * sets, or from a second start). At a BINDING rank cap the penalized minimizer keeps an O(1/w) \
   * residual overlap: raise rank or the penalty, never overlap_tol. out_gram is                  \
   * (nlower + nroots)^2 row-major over [lower; roots], |<a,b>|, and follows the (ptr, cap,       \
   * out_len) protocol. SU(2): rank and every bond_dims entry are REDUCED MULTIPLET ranks, and at \
   * 2S > 0 a QnttSaSweep.discarded_weight at nroots = 1 is not the same quantity as at           \
   * nroots >= 2 (the single-state kernel's unweighted theta-Frobenius ratio vs the               \
   * centerweighted projection loss). */                                                          \
  int32_t qntt_##SFX##_state_average_mals(const void *h, const void *x0, size_t nroots,           \
                                          const QnttStateAverageOptions *opts,                    \
                                          const double *weights, size_t weights_len,              \
                                          const size_t *bond_dims, size_t bond_dims_len,          \
                                          const double *noise, size_t noise_len,                  \
                                          double *out_lambdas, void **out_states,                 \
                                          size_t states_cap, QnttSaSweep *out_sweeps,             \
                                          size_t sweeps_cap, size_t *out_nsweeps,                 \
                                          double *out_sweep_energies,                             \
                                          double *out_sweep_discarded);                           \
  int32_t qntt_##SFX##_state_specific_mals(const void *h, const void *x0,                         \
                                           const void *const *lower, size_t nlower,               \
                                           const QnttStateSpecificOptions *opts,                  \
                                           const size_t *bond_dims, size_t bond_dims_len,         \
                                           const double *noise, size_t noise_len,                 \
                                           const double *lower_energies,                          \
                                           size_t lower_energies_len, double *out_lambda,         \
                                           void **out_x, QnttSsSweep *out_sweeps,                 \
                                           size_t sweeps_cap, size_t *out_nsweeps,                \
                                           double *out_sweep_overlaps);                           \
  int32_t qntt_##SFX##_excited_states(const void *h, const void *x0, const void *const *lower,    \
                                      size_t nlower, const QnttExcitedOptions *opts,              \
                                      const double *weights, size_t weights_len,                  \
                                      const size_t *bond_dims, size_t bond_dims_len,              \
                                      const double *noise, size_t noise_len,                      \
                                      const double *ss_noise, size_t ss_noise_len,                \
                                      double *out_energies, void **out_roots, size_t roots_cap,   \
                                      QnttExcitedInfo *out_info, QnttExcitedRoot *out_root_info,  \
                                      double *out_lower_gradnorms, double *out_gram,              \
                                      size_t gram_cap, size_t *out_gram_len,                      \
                                      void **out_sa_states, size_t sa_states_cap);                \
                                                                                                  \
  /* The certified dense RPA oracle over the full tangent basis at x (one Hessian apply per tangent
     dimension: a small-system cross-check); out_stable = 0 with QNTT_NOT_CONVERGED at a point
     that is not a stable minimizer. The matrix-free solve is qntt_*_rpa_signed. Converge the GS
     to criticality and trim near-zero Schmidt directions first. */                               \
  int32_t qntt_##SFX##_rpa_dense(const void *h, const void *x, double *out_omega, size_t cap,     \
                                 size_t *out_len, int32_t *out_stable);                           \
                                                                                                  \
  /* PD probe of the radially-deflated Hessian part `part` (QNTT_HESS_*) at x, from a             \
   * seed-determined random tangent direction at x's own ranks. `budget` caps Davidson            \
   * iterations ~ Hessian applies and is NOT a formality: at a large tangent dimension a small    \
   * budget cannot converge, and a LIKELY_PD verdict then overstates its margin. Raise it until   \
   * the residual is small. out_verdict is required; out_lambda / out_residual / out_is_pd /      \
   * out_applies may be NULL. Returns QNTT_OK for every verdict -- see the QNTT_PD_* block. */    \
  int32_t qntt_##SFX##_pd_probe(const void *h, const void *x, int32_t part, size_t budget,        \
                                uint64_t seed, int32_t *out_verdict, double *out_lambda,          \
                                double *out_residual, int32_t *out_is_pd, size_t *out_applies);

QNTT_DECLARE_LAYER(f64, double)
QNTT_DECLARE_LAYER(c64, qntt_complex)
QNTT_DECLARE_LAYER(su2_f64, double)
QNTT_DECLARE_LAYER(su2_c64, qntt_complex)

/* --- scalar-only surface: integrals and three-body coefficients -------------------------------- */

#define QNTT_DECLARE_SCALAR(SFX, T)                                                               \
  void qntt_##SFX##_fcidump_free(void *h);                                                        \
  void qntt_##SFX##_w_free(void *w);                                                              \
                                                                                                  \
  int32_t qntt_##SFX##_fcidump_read(const char *path, int32_t symmetry, void **out);              \
  int32_t qntt_##SFX##_fcidump_info(const void *h, size_t *out_norb, size_t *out_nelec,           \
                                    int64_t *out_ms2, T *out_e0);                                 \
  int32_t qntt_##SFX##_fcidump_t(const void *h, T *out, size_t cap, size_t *out_len);             \
  int32_t qntt_##SFX##_fcidump_v(const void *h, T *out, size_t cap, size_t *out_len);             \
  int32_t qntt_##SFX##_tcdump_read(const char *path, size_t norb, int32_t symmetry, void **out);  \
                                                                                                  \
  int32_t qntt_##SFX##_w_new(size_t norb, void **out);                                            \
  int32_t qntt_##SFX##_w_set_coo(void *w, const size_t *idx, const T *vals, size_t n);            \
  int32_t qntt_##SFX##_w_nnz(const void *w, size_t *out);                                         \
                                                                                                  \
  /* abelian-only extras */                                                                       \
  int32_t qntt_##SFX##_tt_state(const int32_t *occ_up, const int32_t *occ_dn, size_t d,           \
                                void **out);                                                      \
  int32_t qntt_##SFX##_vec_to_dense(const void *x, T *out, size_t cap, size_t *out_len);          \
                                                                                                  \
  /* FCIDUMP point-group half. orbsym copies the RAW 1-based Molpro ids (out_len = 0 when the     \
   * header carried no ORBSYM); fcidump_point_group reports the RESOLVED kind (never AUTO), the   \
   * raw ISYM (0 when absent), and the 0-based XOR target. An explicit kind too small for the     \
   * file's ORBSYM surfaces as QNTT_IO, the reader's own error class. */                          \
  int32_t qntt_##SFX##_fcidump_read_pg(const char *path, int32_t symmetry, int32_t pg_kind,       \
                                       void **out);                                               \
  int32_t qntt_##SFX##_fcidump_orbsym(const void *h, uint8_t *out, size_t cap, size_t *out_len);  \
  int32_t qntt_##SFX##_fcidump_point_group(const void *h, int32_t *out_kind, int64_t *out_isym,   \
                                           int32_t *out_target);                                  \
  /* Project out (or reject) the integrals the dump's own ORBSYM forbids, IN PLACE. pg_kind is    \
   * QNTT_PG_AUTO or an explicit kind. REQUIRED before building a Hamiltonian over a              \
   * non-trivial-group space: a real symmetry-adapted dump carries ~1e-14 noise on forbidden      \
   * integrals, which survives eps = DBL_EPSILON and makes Hamiltonian construction fail. An      \
   * entry above 1e-8 is a genuine ORBSYM/kind mismatch and is reported as QNTT_IO. No-op at      \
   * |G| = 1; idempotent; requires the sole reference to the handle. */                           \
  int32_t qntt_##SFX##_fcidump_screen_pg(void *h, int32_t pg_kind);

QNTT_DECLARE_SCALAR(f64, double)
QNTT_DECLARE_SCALAR(c64, qntt_complex)

/* --- point-group surface, per scalar ----------------------------------------------------------- */
/* PG4 below is the descriptor quadruple: int32_t pg_kind, const uint8_t *orbsym,
 * size_t orbsym_len, int32_t target. The SU(2) counterpart is QNTT_DECLARE_SU2_PG below. */

#define QNTT_DECLARE_ABELIAN_PG(SFX, T)                                                           \
  int32_t qntt_##SFX##_vec_zeros_pg(size_t d, size_t a, size_t b, int32_t pg_kind,                \
                                    const uint8_t *orbsym, size_t orbsym_len, int32_t target,     \
                                    void **out);                                                  \
  int32_t qntt_##SFX##_vec_from_ranks_pg(size_t d, size_t a, size_t b, int32_t pg_kind,           \
                                         const uint8_t *orbsym, size_t orbsym_len,                \
                                         int32_t target, const size_t *ranks, size_t ranks_len,   \
                                         void **out);                                             \
  /* ranks at irrep-unreachable sectors are silently masked to 0 */                               \
  int32_t qntt_##SFX##_vec_randn_pg(size_t d, size_t a, size_t b, int32_t pg_kind,                \
                                    const uint8_t *orbsym, size_t orbsym_len, int32_t target,     \
                                    const size_t *ranks, size_t ranks_len, uint64_t seed,         \
                                    void **out);                                                  \
  /* the determinant's total irrep must equal target, else QNTT_INVALID_ARG */                    \
  int32_t qntt_##SFX##_tt_state_pg(const int32_t *occ_up, const int32_t *occ_dn, size_t d,        \
                                   int32_t pg_kind, const uint8_t *orbsym, size_t orbsym_len,     \
                                   int32_t target, void **out);                                   \
  /* the 0/1 indicator of the (Nup, Ndn, target) block; at C1 the plain (Nup, Ndn) indicator */   \
  int32_t qntt_##SFX##_tt_ones_pg(size_t d, size_t a, size_t b, int32_t pg_kind,                  \
                                  const uint8_t *orbsym, size_t orbsym_len, int32_t target,       \
                                  void **out);                                                    \
  int32_t qntt_##SFX##_vec_ranks_pg(const void *x, size_t *out, size_t cap, size_t *out_len);     \
  int32_t qntt_##SFX##_vec_block_get_pg(const void *x, size_t site, int32_t family, size_t sa,    \
                                        size_t sb, uint8_t irrep, T *out, size_t cap,             \
                                        size_t *out_rows, size_t *out_cols);                      \
  int32_t qntt_##SFX##_vec_block_set_pg(void *x, size_t site, int32_t family, size_t sa,          \
                                        size_t sb, uint8_t irrep, const T *src, size_t rows,      \
                                        size_t cols);                                             \
  /* out_irreps receives d 0-BASED XOR indices; the KIND is not recoverable (see Conventions) */  \
  int32_t qntt_##SFX##_vec_point_group(const void *x, int32_t *out_nbits, int32_t *out_target,    \
                                       uint8_t *out_irreps, size_t cap, size_t *out_len);         \
  int32_t qntt_##SFX##_ham_new_pg(size_t d, size_t a, size_t b, int32_t pg_kind,                  \
                                  const uint8_t *orbsym, size_t orbsym_len, int32_t target,       \
                                  const T *t, size_t t_len, const T *v, size_t v_len,             \
                                  const void *w, double eps, int32_t construction, void **out);   \
  int32_t qntt_##SFX##_ham_point_group(const void *h, int32_t *out_nbits, int32_t *out_target,    \
                                       uint8_t *out_irreps, size_t cap, size_t *out_len);

QNTT_DECLARE_ABELIAN_PG(f64, double)
QNTT_DECLARE_ABELIAN_PG(c64, qntt_complex)

/* SU(2)-only: a configuration state function from per-site genealogical branch codes. */
int32_t qntt_su2_f64_tt_csf(const int32_t *codes, size_t d, void **out);
int32_t qntt_su2_c64_tt_csf(const int32_t *codes, size_t d, void **out);

/* --- the f64 -> C64 half: the state cast, and the complex seed built from an H^ii witness ------ */
/* Only this direction exists. C64 -> f64 would have to discard an imaginary part, and there is no
 * complex-to-real seed to build. The output handle is a C64 vector, so free it with the C64
 * family's qntt_*_vec_free (qntt_c64_vec_free / qntt_su2_c64_vec_free), not the f64 one.
 *
 * vec_to_c64 copies the space, the rank grid, every coefficient (into the real part) AND the
 * orthogonality centre, which stays true because a real column-orthonormal frame is orthonormal
 * under the Hermitian inner product too.
 *
 * complex_seed probes H^ii at the REAL state x with `budget` Davidson iterations from a
 * seed-determined random tangent, takes the proven negative-curvature witness w, reprojects it
 * onto the complex tangent bases at x (so the fixed-rank retraction lands on x's own rank grid),
 * normalizes, retracts with the purely imaginary step i*eps and normalizes again -- the step that
 * leaves the real manifold. out_lambda (optional) receives the witness's Rayleigh quotient, the
 * lambda in the seed's predicted energy change lambda*eps^2 -- a DECREASE, since lambda < 0, and
 * with NO 1/2: along a g-orthonormal direction the operator is already the half-second-derivative.
 * eps is FIXED here; above a crossover near sigma_min/2 the retraction costs more than lambda*eps^2
 * buys and the seed lands ABOVE E(x). Rust callers should prefer the backtracking search in
 * qntt::solvers::hess::geometric_seed_search.
 *
 * It returns QNTT_NOT_CONVERGED, and no state, when the probe finds NO negative direction.
 * Seeding along an unproven direction is the null result: the complex polish returns to the real
 * state and every number downstream is the real answer in disguise. Verify with
 * rho = |(x,x)| / <x,x>: rho = 1 means the run silently stayed real. */
int32_t qntt_f64_vec_to_c64(const void *x, void **out);
int32_t qntt_su2_f64_vec_to_c64(const void *x, void **out);
int32_t qntt_f64_complex_seed(const void *h, const void *x, double eps, size_t budget,
                              uint64_t seed, double *out_lambda, void **out);
int32_t qntt_su2_f64_complex_seed(const void *h, const void *x, double eps, size_t budget,
                                  uint64_t seed, double *out_lambda, void **out);

/* --- SU(2) point-group surface, per scalar ----------------------------------------------------- */
/* Mirror of QNTT_DECLARE_ABELIAN_PG, same descriptor quadruple. Two deliberate asymmetries:
 * there is no tt_ones_pg (SU(2) has no tt_ones on any path), and tt_state_pg's counterpart is
 * tt_csf_pg, which INFERS (N, 2S) and DERIVES the total irrep from the genealogical walk and
 * returns QNTT_INVALID_ARG when the supplied target disagrees (or when a dn step is taken from a
 * 2s = 0 sector). ham_new_pg requires construction = 0: SU(2) has a single construction. */

#define QNTT_DECLARE_SU2_PG(SFX, T)                                                               \
  int32_t qntt_##SFX##_vec_zeros_pg(size_t d, size_t a, size_t b, int32_t pg_kind,                \
                                    const uint8_t *orbsym, size_t orbsym_len, int32_t target,     \
                                    void **out);                                                  \
  int32_t qntt_##SFX##_vec_from_ranks_pg(size_t d, size_t a, size_t b, int32_t pg_kind,           \
                                         const uint8_t *orbsym, size_t orbsym_len,                \
                                         int32_t target, const size_t *ranks, size_t ranks_len,   \
                                         void **out);                                             \
  /* ranks at irrep-unreachable sectors are silently masked to 0 */                               \
  int32_t qntt_##SFX##_vec_randn_pg(size_t d, size_t a, size_t b, int32_t pg_kind,                \
                                    const uint8_t *orbsym, size_t orbsym_len, int32_t target,     \
                                    const size_t *ranks, size_t ranks_len, uint64_t seed,         \
                                    void **out);                                                  \
  /* codes[k] is 0 empty, 1 up (S+1/2 BRANCH), 2 dn (S-1/2 BRANCH), 3 doubly occupied */          \
  int32_t qntt_##SFX##_tt_csf_pg(const int32_t *codes, size_t d, int32_t pg_kind,                 \
                                 const uint8_t *orbsym, size_t orbsym_len, int32_t target,        \
                                 void **out);                                                     \
  int32_t qntt_##SFX##_vec_ranks_pg(const void *x, size_t *out, size_t cap, size_t *out_len);     \
  int32_t qntt_##SFX##_vec_block_get_pg(const void *x, size_t site, int32_t family, size_t sa,    \
                                        size_t sb, uint8_t irrep, T *out, size_t cap,             \
                                        size_t *out_rows, size_t *out_cols);                      \
  int32_t qntt_##SFX##_vec_block_set_pg(void *x, size_t site, int32_t family, size_t sa,          \
                                        size_t sb, uint8_t irrep, const T *src, size_t rows,      \
                                        size_t cols);                                             \
  /* out_irreps receives d 0-BASED XOR indices; the KIND is not recoverable (see Conventions) */  \
  int32_t qntt_##SFX##_vec_point_group(const void *x, int32_t *out_nbits, int32_t *out_target,    \
                                       uint8_t *out_irreps, size_t cap, size_t *out_len);         \
  int32_t qntt_##SFX##_ham_new_pg(size_t d, size_t a, size_t b, int32_t pg_kind,                  \
                                  const uint8_t *orbsym, size_t orbsym_len, int32_t target,       \
                                  const T *t, size_t t_len, const T *v, size_t v_len,             \
                                  const void *w, double eps, int32_t construction, void **out);   \
  int32_t qntt_##SFX##_ham_point_group(const void *h, int32_t *out_nbits, int32_t *out_target,    \
                                       uint8_t *out_irreps, size_t cap, size_t *out_len);

QNTT_DECLARE_SU2_PG(su2_f64, double)
QNTT_DECLARE_SU2_PG(su2_c64, qntt_complex)

/* --- joint TT + orbital stack (CASSCF-style) and GAD, f64 layers only -------------------------- */
/* Only the sector-PINNED CAS mode is reachable: every call takes a QnttCasPartition and derives
 * the orbital pair mask from it. Three contracts a caller must not lose:
 *   - ROTATED BASIS. casscf returns (lambda, x, u, H_rot): x, H_rot and everything computed from
 *     them live in the one-particle basis u (row-major d x d), NOT the input orbitals. lambda is
 *     ELECTRONIC -- add the dump's E0 for a total.
 *   - NEVER MALS A PINNED STATE. ranks <= grid is the postcondition of every call here (never
 *     asserted equal); a two-site sweep regrows the pin. als and the joint driver with
 *     tt_preserve_grid are the sanctioned optimizers; gad needs preserve_grid on that grid.
 *   - converged && !rank_limited is the publication licence.
 * The space (point group included) is taken from the state handle. */
#define QNTT_DECLARE_JOINT(SFX)                                                                   \
  /* pinned-CAS reference state; m_active = 0 is full active rank */                              \
  int32_t qntt_##SFX##_cas_reference_state(const void *h, const QnttCasPartition *part,           \
                                           size_t m_active, uint64_t seed, double *out_lambda,    \
                                           void **out);                                           \
  /* ints is an f64 FCIDUMP handle in the INPUT basis; ints.norb must equal x0's d. u is         \
   * buffer-out (size-query with u_cap = 0, need d*d). out_diag: free with qntt_casscf_diag_free. */\
  int32_t qntt_##SFX##_casscf(const void *ints, const void *x0, const QnttCasPartition *part,     \
                              const QnttCasscfOptions *opts, double *out_lambda, void **out_x,    \
                              void **out_ham, double *out_u, size_t u_cap, size_t *out_u_len,     \
                              QnttCasscfInfo *out_info, void **out_diag);                         \
  /* joint symplectic spectrum at the joint-critical (h, x); out_status is QNTT_JOINT_RPA_*;      \
   * omega and fw (per-mode Weingarten fraction f_W = |<v,Bv>|/|<v,Av>|, NaN on imaginary        \
   * entries) are buffer-out of equal length; msg (optional) is TRUNCATED to msg_cap - 1 bytes,   \
   * never an error. */                                                                           \
  int32_t qntt_##SFX##_joint_rpa(const void *h, const void *x, const QnttCasPartition *part,      \
                                 double tt_grad_tol, double orb_floor, double *out_omega,         \
                                 size_t cap, size_t *out_len, double *out_fw, size_t fw_cap,      \
                                 size_t *out_fw_len, int32_t *out_status, size_t *out_n_modes,    \
                                 size_t *out_n_eff, char *msg, size_t msg_cap,                    \
                                 size_t *out_msg_len);                                            \
  /* SIGNED omega^2: every imaginary mode the solve found plus the k lowest real ones, ascending.  \
   * kernel is a QNTT_RPA_KERNEL_* request: PAIRED (0, the library default) or SIGNED (the       \
   * W-metric Davidson); any other code is QNTT_INVALID_ARG before any work. opts->max_iter is    \
   * the budget in pair-equivalents (PAIRED builds max_iter (A, B) apply pairs, SIGNED            \
   * ceil(max_iter / 2) basis vectors of 2 applies each), opts->tol is in omega units on both.    \
   * metric is a QNTT_RPA_METRIC_* request for the side q = X -+ Y the Weingarten fraction reads, \
   * and under SIGNED also the kernel's lane (AUTO = II under both, no probe); out_metric echoes  \
   * the side used -- II = 0, RR = 1, NOT QNTT_HESS_*. Decode order: kernel, metric, curvature    \
   * pair, then the handles.                                                                      \
   * curvature_kind / curvature_value are a QNTT_CURVATURE_REG_* request on the curvature term B  \
   * (see the defines): every output (omega^2, f_W, residuals, margins) is the regularized        \
   * pencil's; an unknown kind or a bad value (HARD/TIKHONOV need a finite value > 0; OFF        \
   * ignores it, NaN included) is QNTT_INVALID_ARG before any work.                               \
   * Negative entries are resonances, publishable only under out_negatives_certified. Per-mode   \
   * buffer-outs, each of out_omega2's length: out_fw the Weingarten fraction |<q,Bq>|/|<q,Aq>|,  \
   * out_residual the pencil residual |Hz - omega Sigma z| / |z| of the mode's amplitudes,       \
   * out_krein its Krein margin (|X|^2 - |Y|^2) / (|X|^2 + |Y|^2) (< 0 a de-excitation, 0 an     \
   * imaginary mode). out_complex: row-major n x 3, one row (Re omega^2, |Im omega^2|, residual)  \
   * per complex mode (one member of each conjugate pair). out_kernel echoes the kernel request;  \
   * out_report (5 entries) is [n-(A+B), n-(A-B) of the projected pencil, metric events,         \
   * signature -1 basis vectors held, general path], -1 where the kernel that ran has no such    \
   * number. Publish omega only when out_converged, every mode real with a positive margin, no   \
   * complex mode, and the counting offset (paired) or the metric events (signed) zero. The      \
   * length is known only after the solve; with n_max = max(opts->max_iter, k + 2) (the basis     \
   * bound), n_max entries per mode buffer and 3 * n_max complex doubles always suffice --        \
   * a smaller buffer fails AFTER the solve. */                                                   \
  int32_t qntt_##SFX##_rpa_signed(const void *h, const void *x, size_t k,                         \
                                  const QnttSymplecticLowestOpts *opts, int32_t kernel,           \
                                  int32_t metric, int32_t curvature_kind,                         \
                                  double curvature_value, uint64_t seed,                          \
                                  double *out_omega2, size_t cap, size_t *out_len,                \
                                  double *out_fw, size_t fw_cap, size_t *out_fw_len,              \
                                  double *out_residual, size_t res_cap, size_t *out_res_len,      \
                                  double *out_krein, size_t krein_cap, size_t *out_krein_len,     \
                                  double *out_complex, size_t cx_cap, size_t *out_cx_len,         \
                                  int32_t *out_metric, int32_t *out_negatives_certified,          \
                                  int32_t *out_converged, size_t *out_iters, size_t *out_applies, \
                                  int32_t *out_kernel, int64_t *out_report, size_t report_cap,    \
                                  size_t *out_report_len);                                        \
  /* the REAL saddle seed normalize(retract(x, eps * w_hat)) along the H^rr witness -- the GAD    \
   * launch point. A LikelyPd verdict is NOT a refusal (the softest mode is the kick wanted);     \
   * out_lambda (optional) is the witness's Rayleigh quotient. */                                 \
  int32_t qntt_##SFX##_saddle_seed(const void *h, const void *x, double eps, size_t budget,       \
                                   uint64_t seed, double *out_lambda, void **out);                \
  /* gentlest-ascent dynamics from x0 (no launch direction: the term-A initializer). */           \
  int32_t qntt_##SFX##_gad(const void *h, const void *x0, const QnttGadOptions *opts,             \
                           void **out_x, QnttGadInfo *out_info);

QNTT_DECLARE_JOINT(f64)
QNTT_DECLARE_JOINT(su2_f64)

/* --- signed RPA on the C64 layers ---------------------------------------------------------- */
/* The signed omega^2 of the REALIFIED C64 pencil at x (symplectic_lowest_signed_davidson_c64) by
 * the kernel `kernel` requests: QNTT_RPA_KERNEL_PAIRED (0, the default) or _SIGNED, which runs the
 * W-metric Davidson on RR, this layer's only lane; any other code is QNTT_INVALID_ARG before any
 * work. The outputs of the f64 layers' qntt_*_rpa_signed, with three differences: no metric
 * request, no curvature pair and no out_metric (A + B is the signed kernel's metric and A - B
 * the other factor; the two are orthogonally similar, so there is nothing to choose and f_W reads
 * q = X + Y); BOTH realified copies of every omega^2 come back, imaginary and complex ones
 * included, so k counts realified values (ask 2 * k_physical; Julia's rpa collapses the pairs
 * instead); out_fw is |Re<q,Bq>| / |Re<q,Aq>|. Buffer bound as there:
 * n_max = max(opts->max_iter, k + 2) entries per mode buffer, 3 * n_max complex doubles, 5 report
 * entries. */
#define QNTT_DECLARE_RPA_SIGNED_C64(SFX)                                                          \
  int32_t qntt_##SFX##_rpa_signed(const void *h, const void *x, size_t k,                         \
                                  const QnttSymplecticLowestOpts *opts, int32_t kernel,           \
                                  uint64_t seed, double *out_omega2, size_t cap, size_t *out_len, \
                                  double *out_fw, size_t fw_cap, size_t *out_fw_len,              \
                                  double *out_residual, size_t res_cap, size_t *out_res_len,      \
                                  double *out_krein, size_t krein_cap, size_t *out_krein_len,     \
                                  double *out_complex, size_t cx_cap, size_t *out_cx_len,         \
                                  int32_t *out_negatives_certified, int32_t *out_converged,       \
                                  size_t *out_iters, size_t *out_applies, int32_t *out_kernel,    \
                                  int64_t *out_report, size_t report_cap,                         \
                                  size_t *out_report_len);

QNTT_DECLARE_RPA_SIGNED_C64(c64)
QNTT_DECLARE_RPA_SIGNED_C64(su2_c64)

/* --- cross-sector (shifted-channel) linear response ------------------------------------------- */
/* Response INTO ANOTHER point-group irrep from one typed reference: the flux-`phi` block of the
 * Riemannian Hessian at `x`. A typed point is irrep-closed, so no ordinary tangent direction
 * reaches another channel; embed it in the point-group-UNTYPED fixed-rank manifold and its
 * tangent space splits as a DIRECT SUM over the fluxes, one independent response problem each,
 * evaluated at typed cost.
 *
 * f64 layers only, and only over a space with a non-trivial point group. Declared through its
 * own macro, NOT inside QNTT_DECLARE_LAYER: that one is instantiated for all four (layer,
 * scalar) pairs and would declare c64 symbols that do not and will not exist.
 *
 * Reading the numbers, in one place:
 *  - `h_ref` is the operator on x's OWN space (it supplies lambda and the Weingarten
 *    environments). `h_out` MUST have been BUILT -- not re-stamped -- on x's space with target
 *    `Gamma0 XOR phi`, from the SAME t, v, w and eps as h_ref: its construction depends on the
 *    target. Only the space is checked; identical integrals are a caller obligation.
 *  - `phi != 0` is required. phi = 0 is the in-sector problem and qntt_*_rpa_signed /
 *    _rpa_dense answer it; asking for it here is QNTT_INVALID_ARG, as is a phi
 *    outside the state's group, an h_out on the wrong space, or an out-of-range enum code.
 *    No caller error crosses as QNTT_PANIC.
 *  - `out_dim` is dim T^phi, which is NOT the dimension of the target irrep's sector: T^phi
 *    is built from x's own bases, so it is generally a PROPER subspace of that sector -- for
 *    GROUP reasons, even at full typed rank -- and can be 0 while the sector
 *    is large. The values are that sector's own levels only when out_dim equals its determinant
 *    count; compare the two before reading one as an excitation energy.
 *  - An EMPTY out_len is a real answer, not an error: a structurally unreachable channel (or
 *    k = 0) or a start block the projector annihilated, told apart by (out_converged, out_dim)
 *    -- see the Rust docs of qntt::solvers::hess::sector. Both return QNTT_OK; both RPA
 *    kernels answer an indefinite channel with a non-empty list.
 *  - qntt_*_sector_rpa decodes the curvature pair (QNTT_CURVATURE_REG_*) and the precond code
 *    (QNTT_SECTOR_PRECOND_*) FIRST, before any handle borrow: an unknown code or a bad curvature
 *    value is QNTT_INVALID_ARG naming the legal codes, before any work (precond_kappa is never
 *    refused). Statuses are otherwise unchanged; every diagnosis (empty, flagged, negative) is
 *    QNTT_OK.
 *  - RPA arm: opts->kernel selects the kernel (QNTT_RPA_KERNEL_PAIRED, the default, or _SIGNED;
 *    an unknown code is QNTT_INVALID_ARG); neither refuses on the metric; AUTO = II, no probe, no
 *    retry; every output is the (regularized) pencil's. out_fw[j] = |<q,Bq>|/|<q,Aq>| on the side
 *    q = X -+ Y of the mode's amplitudes that opts->metric names under PAIRED, the lane under
 *    SIGNED (two applies per mode, not in out_applies; +inf at <q,Aq> = 0), out_residual[j]
 *    its pencil residual |Hz - omega Sigma z| / |z|, out_krein[j] its Krein margin (< 0 on a
 *    positive omega^2 is a DE-EXCITATION to a lower state; 0 for an imaginary mode), out_complex
 *    row-major (Re omega^2, |Im omega^2|, residual) per WANTED complex mode -- Re omega^2 inside
 *    the reported window -- not every complex pair of the projected pencil. out_report: the
 *    5-entry report of qntt_*_rpa_signed for the kernel that ran (paired: the counting offset
 *    first, the last three -1; signed: the first two -1). SIZING: the list is every imaginary mode
 *    plus the k lowest real ones, known only after the solve; per-mode buffers of
 *    n_max = max(opts->max_iter, k + 2) entries (out_complex 3 * n_max doubles) always suffice,
 *    and a short buffer fails AFTER the solve (copy_out). Any anomaly (a nonzero counting offset,
 *    a metric event, a negative Krein margin, an imaginary or complex mode) => out_converged
 *    alone does not license a mode; out_negatives_certified = 1 is vacuous when nothing is
 *    negative.
 *  - TDA arm: at most k entries; the RPA-only buffers are written with length 0 and the report
 *    all -1; the curvature pair is validated and has no effect (A is never regularized). */
#define QNTT_DECLARE_SECTOR(SFX)                                                                  \
  /* Lowest `k` values of the flux-`phi` block at `x`. out_theta_or_omega2 carries eig(A^phi) --  \
     an ENERGY -- under QNTT_SECTOR_READOUT_TDA and the signed omega^2 under _RPA; the two are    \
     different quantities and the buffer's name says so. out_metric echoes the f_W side of the    \
     RPA solve (II for an II or AUTO request, RR for RR; the signed LANE; QNTT_RPA_METRIC_AUTO     \
     where a signed run returned before choosing one) and is written AUTO on the TDA arm.         \
     out_negatives_certified gates publication of a resonance: a negative value carries a        \
     fresh-apply witness only under it, and an i*kappa read off an uncertified negative is not    \
     publishable. Every out-parameter except opts may be NULL (a NULL buffer with cap 0 is a      \
     size query); k = 0 on a live channel reports its true out_dim with an empty answer and no    \
     applies. The contract above (selectors, flags, sizing) is not restated here. */              \
  int32_t qntt_##SFX##_sector_rpa(const void *h_ref, const void *h_out, const void *x,            \
                                  uint8_t phi, size_t k, const QnttSectorOpts *opts,              \
                                  int32_t curvature_kind, double curvature_value,                 \
                                  int32_t precond, double precond_kappa, uint64_t seed,           \
                                  double *out_theta_or_omega2, size_t cap, size_t *out_len,       \
                                  double *out_fw, size_t fw_cap, size_t *out_fw_len,              \
                                  double *out_residual, size_t res_cap, size_t *out_res_len,      \
                                  double *out_krein, size_t krein_cap, size_t *out_krein_len,     \
                                  double *out_complex, size_t cx_cap, size_t *out_cx_len,         \
                                  size_t *out_dim, int32_t *out_metric, int32_t *out_converged,   \
                                  size_t *out_iters, size_t *out_applies,                         \
                                  int32_t *out_negatives_certified, int64_t *out_report,          \
                                  size_t report_cap, size_t *out_report_len);                     \
  /* The DENSE certified oracle over the whole shifted tangent basis: one Hessian-class apply     \
     pair per tangent dimension, so a small-system cross-check, not a production path. The        \
     cap = 0 probe pass still pays that full solve (copy_out is at the END, the same bargain      \
     qntt_*_rpa_dense makes); size from out_dim on a first pass if that matters.                  \
     out_stable is a TRI-state, not a boolean: 1 = stable (both blocks PD, every omega^2 > 0),    \
     0 = resonant (exactly one block PD -- the signed omega^2 IS written, and a negative entry    \
     is a resonance omega = i*sqrt|omega^2|, physics, not a failure), -1 = unstable (both blocks  \
     indefinite; no real omega^2 exists, out_len = 0, and the call returns QNTT_NOT_CONVERGED     \
     with the diagnosis in qntt_last_error). A caller testing `out_stable != 1` must handle BOTH  \
     0 and -1. This is where the contract diverges from qntt_*_rpa_dense, which reports its own   \
     resonant arm as out_stable = 0 with NOTHING written and a non-OK status, because its buffer  \
     is declared to carry omega and a resonance's omega is imaginary; here the buffer carries     \
     omega^2, which is real on both arms, so the resonant arm is a QNTT_OK answer.                \
     out_stable is REQUIRED; the other out-parameters may be NULL. out_metric names the PD block  \
     on the resonant arm and is QNTT_RPA_METRIC_AUTO on the other two, where none was chosen.     \
     A structurally empty channel reports out_stable = 1 with out_len = 0 and spends no work. */  \
  int32_t qntt_##SFX##_sector_rpa_dense(const void *h_ref, const void *h_out, const void *x,      \
                                        uint8_t phi, double *out_omega2, size_t cap,              \
                                        size_t *out_len, size_t *out_dim, int32_t *out_stable,    \
                                        int32_t *out_metric);

QNTT_DECLARE_SECTOR(f64)

/* Cross-sector OBSERVABLES in one shifted irrep channel: the static polarizability and the dense
 * transition moments of the flux-`phi` block at the typed reference `x`. h_ref / h_out / phi carry
 * EXACTLY the contract stated above QNTT_DECLARE_SECTOR -- read it there, it is not restated.
 *
 * The property operator. `m` holds n_props contiguous norb x norb ROW-MAJOR blocks (first axis
 * most significant, the qntt_*_ham_new convention for t): entry (i,p,q) at
 * m[i*norb*norb + p*norb + q], and m_len must equal n_props*norb*norb exactly. `m` must be
 * readable for m_len doubles and 8-byte aligned -- the length is the caller's promise (see the
 * Contract block), and it is checked for exact equality BEFORE the buffer is read. mu_i = sum_{p,q}
 * m[i][p][q] * E_pq is a spin-summed one-body observable; each block must be a FULL SYMMETRIC
 * matrix (pass both triangles, not one representative per pair), every entry finite, norb equal
 * to the state's orbital count, and 1 <= n_props <= 3 (the oscillator-strength kernel is a
 * three-row kernel and its 2/3 is the isotropic average over x, y, z). Violations are
 * QNTT_INVALID_ARG naming the component; none of them crosses as QNTT_PANIC.
 *
 * The IRREP FLUX IS DERIVED FROM THE MATRIX, never from an axis table -- which irrep z carries
 * depends on the molecular orientation the generator chose, not on the group. A component whose
 * derived flux is not `phi` is a STRUCTURAL ZERO row and column of alpha (and an all-zero moment
 * row) with no work paid and a capture_defect of exactly 1, not an error: a caller looping the
 * same three components over every channel is the intended usage. A component whose matrix
 * carries MORE THAN ONE flux class has no single output space and IS an error
 * (QNTT_INVALID_ARG): split it by class and call once per channel.
 *
 * `labels` may be NULL (defaults p0, p1, p2) and individual entries may be NULL (that index
 * defaults). They are NUL-terminated UTF-8, at most 256 bytes, used in refusal messages only and
 * never returned -- the caller's component order indexes every row and column of every output.
 *
 * `x` must be NORMALIZED: alpha, T_n and lambda all scale with the reference, and a non-unit x is
 * QNTT_INVALID_ARG in every build profile (the library checks it in debug builds only).
 *
 * REFUSALS. A physics refusal returns QNTT_NOT_CONVERGED, writes NO out-buffer and NO out_len,
 * sets *out_refusal to a QNTT_SECTOR_REFUSAL_* code and leaves the full diagnosis in
 * qntt_last_error(). There is no length to report on that path: the out_len-always guarantee
 * belongs to copy_out's own failure path (a short buffer), not to a refusal that happened before
 * copy_out was reached. A refusal is a statement about the STATE, not a caller error: the
 * reference is not TT-critical, H^rr is indefinite on the channel (the reference is not the
 * lowest state of Gamma0 XOR phi), both Hessian blocks are indefinite (no real symplectic basis
 * exists at all), or a Krylov solve hit its cap (a partial CG answer is a lower bound of unknown
 * tightness). *out_refusal is QNTT_SECTOR_REFUSAL_NONE on QNTT_OK and on a caller error. A
 * resonance is a refusal of the static alpha only when H^rr is the indefinite block; the moment
 * entry publishes the REAL modes and reports the imaginary ones in out_unstable_omega2 -- it
 * never writes an imaginary omega into out_omega.
 *
 * A structurally empty channel (out_dim = 0) is a QNTT_OK answer, and only the MODE-INDEXED
 * lengths collapse: out_omega_len, out_moments_len, out_osc_len, out_unstable_len and
 * out_clusters_len are 0 (info.n_modes = n_neg = n_clusters = 0). The n_props-shaped outputs keep
 * their shapes -- out_alpha_len and out_sos_len are n_props*n_props, out_props_len is 3*n_props
 * (alpha) or 5*n_props (moments), out_iters_len is n_props -- with alpha and alpha_sos all zeros,
 * iters all 0 and capture_defect exactly 1 per component. info.has_alpha_sos is SET (an empty sum
 * is a sum), info.metric is QNTT_RPA_METRIC_AUTO, and no apply is spent. */
#define QNTT_DECLARE_SECTOR_OBS(SFX)                                                              \
  /* alpha^phi_uv = 2 <p_u, (H^rr,phi)^-1 p_v> on the channel: one PCG solve per IN-CHANNEL       \
     component against H^rr = A^phi + B^phi. out_alpha is the SYMMETRIZED n_props x n_props       \
     tensor, row-major in the caller's component order -- a 1x1 or 2x2 principal sub-block of the \
     Cartesian tensor embedded in the caller's shape, since only the components whose flux is     \
     `phi` are visible in this channel. Summing the channels, plus the in-sector block that       \
     qntt_*_rpa_* answers, is what gives the full tensor.                                         \
     out_props is a row-major n_props x 3 DIAGNOSTICS table: column 0 capture_defect =            \
     sqrt(1 - |P^phi(mu x)|^2 / |mu x|^2), how much of the property image the channel's           \
     borrowed-suffix tangent space MISSED (O(1) on a proper-subspace channel, exactly 1 on an     \
     out-of-flux component -- reported, never asserted zero); column 1 grad_norm = |p_u| (0 means \
     the component is invisible here and its whole row and column are exact zeros); column 2 the  \
     FRESH relative residual |p - H^rr xi|/|p|, one extra apply after the solve, never CG's       \
     recursive residual. out_iters holds the per-component CG iteration counts (0 on a structural \
     zero row and on an empty channel -- the no-apply proxy). Every output length is a function   \
     of n_props, so no size-probe pass is needed; a cap = 0 / NULL triple is still honoured as    \
     one. out_info and out_refusal may not be NULL; every buffer triple may. */                   \
  int32_t qntt_##SFX##_sector_alpha(                                                              \
      const void *h_ref, const void *h_out, const void *x, uint8_t phi,                           \
      const double *m, size_t m_len, size_t n_props, size_t norb,                                 \
      const char *const *labels, const QnttSectorAlphaOpts *opts,                                 \
      double *out_alpha, size_t alpha_cap, size_t *out_alpha_len,                                 \
      double *out_props, size_t props_cap, size_t *out_props_len,                                 \
      size_t *out_iters, size_t iters_cap, size_t *out_iters_len,                                 \
      QnttSectorAlphaInfo *out_info, int32_t *out_refusal);                                       \
  /* The channel's transition moments and oscillator strengths, DENSE route: the (A^phi, B^phi)   \
     Grams over the WHOLE shifted tangent basis then a signed reduced-Williamson reduction:       \
     a small-system path, not a production one, like qntt_*_sector_rpa_dense.                     \
     out_omega holds the info.n_modes REAL frequencies, ASCENDING. out_moments is a row-major     \
     n_props x n_modes table (entry (i,n) at out_moments[i*n_modes + n]); only |T_n|, same-n      \
     products across two components, and cluster sums are OBSERVABLE -- a bare signed T_n is      \
     gauge, and inside a degenerate cluster (out_clusters) only sum_{n in cluster} T_n T_n' is    \
     defined at all. out_oscillator holds f_n = (2/3) omega_n sum_c T_n(c)^2; with fewer than     \
     three components it is a PARTIAL sum and the 2/3 is NOT rescaled. out_alpha_sos is the       \
     n_props^2 sum-over-states reconstruction of qntt_*_sector_alpha's tensor -- an ALGEBRAIC     \
     identity at any rank, hence the mutual cross-check -- and is written ONLY when               \
     info.has_alpha_sos != 0. out_props is a row-major n_props x 5 diagnostics table: 0           \
     capture_defect, 1 completeness_defect = |p_u - sum_i gamma_i b_i|/|p_u| (how much of the     \
     covector the channel BASIS missed), 2 sum T_n^2, 3 |P^phi(mu x)|^2, 4 the S_i-side sum. The  \
     OPERATOR sum rule is columns 2 vs 3: EQUAL when B^phi = 0, and otherwise a defect of EITHER  \
     SIGN -- not an inequality. It is NOT Thomas-Reiche-Kuhn: sum f_n = N_e needs the true        \
     position operator in a complete basis and is a diagnostic here, never an oracle.             \
     out_clusters holds 2*info.n_clusters size_t as half-open [lo,hi) pairs partitioning          \
     0..n_modes, singletons included. out_unstable_omega2 holds the info.n_neg SIGNED omega^2 < 0 \
     of the imaginary modes, ascending; omega = i*sqrt|omega^2| and it is never a real omega.     \
     SIZING: n_modes, n_neg and n_clusters are all bounded by dim, and qntt_*_sector_rpa with     \
     k = 0 reports dim for free -- size from it and call once. The all-NULL probe pass writes the \
     whole out_info and no buffer, but pays the full dense solve a second time. out_info and      \
     out_refusal may not be NULL; every buffer triple may. */                                     \
  int32_t qntt_##SFX##_sector_transition_moments(                                                 \
      const void *h_ref, const void *h_out, const void *x, uint8_t phi,                           \
      const double *m, size_t m_len, size_t n_props, size_t norb,                                 \
      const char *const *labels, const QnttSectorMomentOpts *opts,                                \
      double *out_omega, size_t omega_cap, size_t *out_omega_len,                                 \
      double *out_moments, size_t moments_cap, size_t *out_moments_len,                           \
      double *out_oscillator, size_t osc_cap, size_t *out_osc_len,                                \
      double *out_alpha_sos, size_t sos_cap, size_t *out_sos_len,                                 \
      double *out_props, size_t props_cap, size_t *out_props_len,                                 \
      double *out_unstable_omega2, size_t unstable_cap, size_t *out_unstable_len,                 \
      size_t *out_clusters, size_t clusters_cap, size_t *out_clusters_len,                        \
      QnttSectorMomentsInfo *out_info, int32_t *out_refusal);

QNTT_DECLARE_SECTOR_OBS(f64)

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QNTT_H */
