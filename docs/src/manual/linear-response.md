# Linear response (RPA)

Excitation-energy estimates from the **symplectic eigenvalues of the Riemannian Hessian** at a
converged state on the fixed-rank TT manifold.

The framework is that of Grazioli, Hu & Cancès ([arXiv:2506.16420](https://arxiv.org/abs/2506.16420)):
a variational model is a Kähler manifold, the energy generates a Hamiltonian flow on it, and the
excitation energies of linear-response theory are the symplectic (Williamson) eigenvalues of the
Hessian at a critical point. They instantiate it on Grassmannians (Hartree–Fock, FCI); the
instantiation on the fixed-TT-rank manifold — tangent space, projector, Weingarten map — is this
library's.

## What it computes

At a critical point ``x_\star`` the Riemannian Hessian ``\mathfrak{h}`` on the tangent space, together
with the symplectic form ``J``, gives excitation estimates as the Williamson values ``\omega_j``
of ``\mathfrak{h}``. On the real layer this reduces to

```math
\omega_j^2 \in \mathrm{spec}\!\left( H^{rr} \circ H^{ii} \right),
\qquad H^{rr} = A + B, \quad H^{ii} = A - B,
```

with ``A``, ``B`` the usual linear-response blocks — the same structure as Casida's equations,
derived here without them.

## Using it

```julia
# 1. converge a ground state ON the manifold you want to linearize about
x0   = QNTT.tt_randn(space, QNTT.uniform_ranks(space, 4); seed = 1)
λ, x = QNTT.riemannian(H, x0; cg = true, eps = 1e-8, max_iter = 300)

# 2. the lowest k symplectic eigenvalues, matrix-free
ω, info = QNTT.rpa(H, x; k = 6, tol = 1e-8, max_iter = 600)

info.converged, info.iterations, info.applies
```

[`QNTT.rpa`](@ref) is a convenience over [`QNTT.rpa_signed`](@ref) (below) with its defaults; it
reads ``\omega = \sqrt{\omega^2}`` off the signed spectrum. `max_iter` counts `(A, B)` apply
pairs and `tol` is relative, in `ω` units. `info.converged == false` means the budget was reached
and the returned `ω` are the best Rayleigh–Ritz values so far.

For a cross-check on a small system, [`QNTT.rpa_dense`](@ref) builds the whole tangent basis and
solves the problem densely — a reference for small systems, not a production path:

```julia
ωd = QNTT.rpa_dense(H, x)          # must reproduce `ω` where they overlap
```

## What the numbers mean — and don't

!!! warning "Fixed-rank estimates"
    The ``\omega_j`` are linear-response estimates **on the manifold you converged on**. They
    approximate exact excitation energies; they *equal* them only when the manifold is the full
    space (`uniform_ranks(space, r)` at `r ≥ maxranks`, where TT is exact and the model is FCI).
    At reduced rank the gap between ``\omega_j`` and the exact ``E_k - E_0`` is a property of the
    ansatz, not an error in the calculation.

That is also what makes a good validation: at full rank on a small molecule, ``\omega_j`` should
reproduce the exact FCI excitation energies to solver tolerance. The
[notebooks](../notebooks.md) do exactly that for H₄.

Two further caveats, both structural:

- **Degeneracies are resolved.** A physically ``m``-fold degenerate ``\omega`` appears ``m``
  times, as in the dense driver.
- **`ω` needs a stable point.** At a saddle some ``\omega`` are imaginary or complex, which a list
  of real `ω` cannot hold: `rpa` and `rpa_dense` then throw with the diagnosis (the signed
  spectrum, for `rpa`) rather than return a meaningless number. [`QNTT.rpa_signed`](@ref) returns
  that spectrum as data. The usual cause is an under-converged ground state — tighten the
  ground-state solve first.

## Near the manifold border: regularizing the curvature term

[`QNTT.rpa_signed`](@ref) solves at a reference whatever its stability: it returns the signed
``\omega^2`` (a negative entry is a resonance) and never stops on an indefinite Hessian block. Per
mode it reports the pencil residual of the returned amplitudes ``(X, Y)`` and the Krein margin
``(\|X\|^2 - \|Y\|^2)/(\|X\|^2 + \|Y\|^2)`` (negative for a de-excitation, zero for an imaginary
mode); per solve, the kernel that ran and its report.

`kernel = :signed` runs a second solver, a W-metric Davidson on the block `metric` names (`:ii`,
the default, or `:rr`; `:auto` is `:ii`). It reports metric events and the signature of its basis
in `info.report`, and the same `max_iter` buys it about half as many basis vectors. Use it as a
second opinion, not the default:

```julia
ω², info = QNTT.rpa_signed(H, x; k = 6, kernel = :signed, metric = :ii)
info.kernel, info.report.metric_events
```

At a state with a Schmidt value close to zero — the border of the fixed-rank manifold — the
curvature term ``B`` grows like the inverse of that Schmidt value, and the bottom of the spectrum
fills with curvature modes that are not excitations: they make a metric block indefinite, or show
up as spurious low frequencies with a Weingarten fraction `f_w` close to 1. Trimming the state
moves the point. The curvature regularization instead leaves the point, its energy, ``A`` and the
tangent space alone and changes only ``B``, as a congruence:

- `curvature = :hard` replaces ``B`` by ``P B P``, where ``P`` removes from every tangent component
  the column directions along the right Schmidt vectors whose Schmidt value is below
  `curvature_scale`;
- `curvature = :tikhonov` replaces it by ``D B D``, with
  ``D = V\,\mathrm{diag}\big(s^2/(s^2+\varepsilon^2)\big)\,V^{\mathsf T}`` and
  ``\varepsilon`` = `curvature_scale`, which damps those directions instead of cutting them.

Either way the pencil ``(A + DBD,\ A - DBD)`` stays symmetric.

```julia
ω², info = QNTT.rpa_signed(H, x; k = 6, curvature = :hard, curvature_scale = 1e-4)
ω², info = QNTT.rpa_signed(H, x; k = 6, curvature = :tikhonov, curvature_scale = 1e-4)
```

In Rust the knob lives on the Hessian context, so every reader of its ``B`` sees the same
regularized operator:

```rust
use qntt::solvers::hess::{CurvatureReg, weingarten_operator};
use qntt::solvers::hess::symplectic_lowest_signed_davidson;

let ctx = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(1e-4));
// `readout.omega()` is `Some` only for a converged solve with no anomaly; `amplitudes` holds each
// mode's (X, Y).
let (readout, amplitudes) = symplectic_lowest_signed_davidson(&ctx, &seeds, 6, &opts);
```

- **The default is `:off`** (`CurvatureReg::Off`): the exact Riemannian Hessian.
- **Units.** The scale is an absolute Schmidt value, `QNTT.schmidt(x).sigma`, on both layers. On
  the spin-adapted layer that is the reduced `σ` of a multiplet, not the weighted value `√w·σ`, so
  at a singlet one scale cuts the same directions on both layers. `:hard` cuts the values strictly
  below the scale. The scale must be finite and positive; `:off` is the only way to switch the
  regularization off.
- **Every number is the regularized pencil's.** `ω²`, `f_w`, residuals, Krein margins and the
  per-mode checks all describe ``(A \pm DBD)``, not the Riemannian Hessian. The removed or damped
  directions keep their ``A`` content, so a larger scale biases the physical frequencies as well:
  scan the scale and report it beside every ``\omega``.
- **Symmetry channels.** [`QNTT.sector_rpa`](@ref) (below) takes the same `curvature` /
  `curvature_scale`, with the same meaning, on every channel.

## Tamm–Dancoff spectrum: ``A`` alone

The Tamm–Dancoff approximation keeps only the ``A`` block: the lowest eigenpairs of
``A = P_T (H - \lambda) P_T`` on the tangent space, with the radial direction (the state itself)
removed. Without ``B`` there are none of the curvature modes of the previous section, and the
spectrum does not depend on the curvature regularization. Three facts make its numbers easy to read:

- ``\theta`` is an **energy** relative to ``\lambda``, not an ``\omega^2``. For a unit tangent
  ``\xi`` orthogonal to the radial direction, ``\langle \xi, A \xi \rangle = E_\xi - \lambda``
  exactly, where ``E_\xi`` is the energy of the explicit rank-``2r`` state ``\iota(\xi)``, itself
  orthogonal to ``x``. So ``\lambda + \theta_j`` is the energy of a state, not an estimate of one.
- **Upper bounds.** At a critical point the sorted list ``\{\lambda, \lambda + \theta_1, \dots,
  \lambda + \theta_k\}`` bounds the ``k + 1`` lowest levels of ``H`` in the reference's symmetry
  sector from above, one by one (Hylleraas–Undheim–MacDonald). At full rank the tangent space is
  the whole sector and the ``\theta_j`` are the exact excitation energies.
- **Against the RPA.** At a stable minimizer (both ``A + B`` and ``A - B`` positive definite)
  ``\omega_j \le \theta_j`` for every ``j``: the min–max characterization of the RPA frequencies
  includes the subspace ``Y = 0``, on which it reduces to ``A``.

Without ``B`` it is cheaper than the RPA. The preconditioner is `TdaPrecond::Auto` by default (a
capped diagonal preconditioner, the same as `TdaPrecond::Diag { kappa: 10.0 }`); `TdaPrecond::Off`
turns it off. It is available from Rust for now, on both layers
(`qntt::solvers::hess::tda_lowest_davidson` and `qntt::su2::hess::tda_lowest_davidson`):

```rust
use qntt::core::tangent::TtTangent;
use qntt::solvers::hess::{TdaDavidsonOpts, tda_lowest_davidson, weingarten_operator};

let ctx = weingarten_operator(&h, &x);
let (bl, br) = ctx.bases();
let seeds: Vec<TtTangent<f64>> = (0..k + 2)
    .map(|_| TtTangent::project(bl, br, &qntt::tt_randn(x.space, &x.ranks, &mut rng)))
    .collect();
let modes = tda_lowest_davidson(&ctx, &seeds, k, &TdaDavidsonOpts::default());
// modes.theta (ascending, Ha, relative to λ), modes.vectors (unit, radial-free),
// modes.info: converged, fresh residuals and Rayleigh quotients, A applies, negative witnesses
```

A negative ``\theta`` comes with a certificate (`modes.info.negative`,
`modes.all_negatives_certified()`): a certified one is an explicit state below ``\lambda`` in the
tangent span, so the reference is not the lowest state the manifold reaches there. At a base point
whose declared rank grid its cores cannot fill — trim it first — the reachable tangent span is
smaller than the structural dimension, and asking for more roots than that span holds returns
what it holds, with `converged = false`.

## Other symmetry channels from one reference

A typed reference is irrep-**closed**: no tangent direction at an `A1` state carries `B1` character,
so a `B1` excitation is not "hard to reach at a larger `k`" — it is absent from the response space.
[step-11](../tutorial/step-11.md) teaches those walls.

There is a way through them, at one extra operator build per channel. Embed the typed state in the
point-group-**untyped** fixed-rank manifold — the same cores, bond ranks merged over the irreps.
Its tangent space splits into one block per irrep flux `φ`, and the Hessian is block-diagonal
across those blocks, so each other irrep becomes a separate, smaller response problem. Nothing is
re-converged: the gradient of a typed state is orthogonal to every other channel, so the reference
is a *critical point* of the untyped problem.

`B` does not vanish in these channels in general, so `:tda` (energies) and `:rpa` (signed `ω²`)
are different quantities in different units, not a refinement of one another.

```julia
Γ0 = s.target
for φ in 1:QNTT.nirrep(s) - 1
    s_out = QNTT.QnSpace(s.d, s.a, s.b; point_group = s.kind, orbsym = s.orbsym,
                         target = xor(Γ0, φ))
    H_out  = QNTT.Hamiltonian(s_out, data)      # built on the shifted target
    θ,  i1 = QNTT.sector_rpa(H, H_out, x, φ; k = 2, readout = :tda)   # energies
    ω², i2 = QNTT.sector_rpa(H, H_out, x, φ; k = 2, readout = :rpa)   # signed ω²
    @info "channel" φ i1.dim θ ω² i2.krein i2.report   # dim == 0 ⇒ unreachable
end
```

[`QNTT.sector_rpa`](@ref) reads out one channel; [`QNTT.sector_rpa_dense`](@ref) is its dense
reference over the whole shifted tangent basis (unregularized). The `:rpa` readout runs the same
solvers as [`QNTT.rpa_signed`](@ref) (`kernel = :paired`, the default, or `:signed`): it lists every
imaginary mode plus the `k` lowest real ones, and reports per entry the Weingarten fraction `f_w`,
the pencil residual and the Krein margin, beside the complex modes in the reported window and the
kernel's report. The same knobs apply — `metric` (the `f_w` side, and the block `:signed` uses),
`curvature` / `curvature_scale` — plus a preconditioner, `precond = :off | :d0 | :auto` (`:d0` a
diagonal preconditioner; `:auto`, the default, is currently `:off`):

```julia
ω², info = QNTT.sector_rpa(H, H_out, x, φ; k = 4, readout = :rpa,
                           curvature = :hard, curvature_scale = 1e-4, precond = :d0)
```

The `!!! warning "Fixed-rank estimates"` above applies verbatim, and five things are specific to a
shifted channel:

- **`info.dim` is not the sector's size.** The shifted tangent space is built from the reference's
  own bases, so it is generally a proper subspace of the target irrep's sector — a property of the
  group and the rank grid, not of the rank alone. Compare `info.dim` against that
  sector's determinant count before quoting a value as an excitation energy; where they differ the
  values are variational upper bounds on the channel's levels.
- **An empty channel is an answer.** `dim == 0` with an empty value list says the flux is unreachable
  from this reference's grid, even when the target sector is large.
- **`B ≡ 0` where the channel varies a single core.** The manifold then contains a straight line
  through the reference, the curvature term vanishes and `:rpa` collapses onto `:tda`. Geometry, not
  a defect.
- **Instability is information.** A negative value says the reference is not the lowest state of that
  irrep — criticality is not stability. A negative `ω²` is a resonance `ω = i√|ω²|`, never published
  as a real `ω`: a real spectrum comes back only when the channel produced values at all, every `ω²`
  is strictly positive and no negative witness is outstanding. Gate on `info.negatives_certified`.
- **An anomalous solve is read mode by mode.** Any anomaly — a nonzero counting offset in
  `info.report` (paired kernel), an indefinite metric block on the explored space
  (`info.metric_indefinite = true`, signed kernel), a negative Krein margin, an imaginary or complex
  mode — means `converged` alone does not license the values. A POSITIVE `ω²` with a negative
  Krein margin is not an excitation but a *de-excitation*: a state of the target irrep below `λ`
  can appear this way, with nothing negative in the list. Such entries are withheld from every
  published `ω` and level; in Rust, `SectorCtx::rpa_mode_checks` gives the per-mode acceptance
  numbers.

Each channel needs its own Hamiltonian built on the shifted target (re-labelling an existing one is
not enough). [step-18](../tutorial/step-18.md) runs the whole loop on LiH/STO-3G and prints
`dim T^φ` beside the target sector's determinant count.

### Observables in a channel

A cross-irrep one-body operator `μ̂ = Σ m_pq E_pq` lives in exactly **one** shifted channel: `μ̂x`
carries the flux class `Γ_μ` read off the matrix `m`, so `P^φ(μ̂x) = 0` for every `φ ≠ Γ_μ` by
space mismatch, not by cancellation. At a totally symmetric reference every dipole-allowed
transition is therefore in a shifted channel and the in-sector RPA sees only the forbidden ones.
The library derives the class from the matrix; it does not know which irrep `x`, `y`, `z` carry
(that depends on the orientation the integral generator chose), so the caller supplies the
component matrices.

Per channel, at a TT-critical reference, the library computes:

- the static polarizability `α^φ = 2⟨p, (H^rr)⁻¹p⟩` with `p = P^φ(μ̂x)`, refused when `H^rr` is
  indefinite (the reference is not the lowest state of that irrep);
- transition moments `T_n` and oscillator strengths `f_n = (2/3)ω_n Σ_c T_n(c)²`, densely over the
  whole channel or for the `k` lowest modes iteratively (a partial sum, refused unless the
  eigensolve converged);
- `α_SOS = 2Σ T_nT_n/ω_n`, equal to `α^φ` at any rank, withheld when the channel has imaginary
  modes;
- the sum rule `Σ_n T_n²` beside `‖P^φ(μ̂x)‖²`. It is exact only where `B^φ ≡ 0` and can miss on
  either side otherwise, so it is a diagnostic, not a check.

Two further defects are reported, never asserted zero: *capture*,
`√(1 − ‖P^φ(μ̂x)‖²/‖μ̂x‖²)`, how much of the property image the channel's tangent space misses,
and *completeness*, how much of the covector the channel basis misses. Every `α` is the response
within the fixed-rank TT manifold at the given orbitals, with no orbital relaxation. Spin-free
observables have zero singlet→triplet moments, so the spin-adapted channel returns exact zeros.
The Rust surface is `SectorCtx::polarizability_static`, `transition_moments_dense` /
`_iterative`, `linear_response` and `sector_response_all`; [step-18](../tutorial/step-18.md)
exercises it on synthetic operators.

**From C and Julia.** The static α and the dense moments are available as
`qntt_f64_sector_alpha` / `qntt_f64_sector_transition_moments` and [`QNTT.sector_alpha`](@ref) /
[`QNTT.sector_transition_moments`](@ref). You supply the operator components — in Julia as one to
three symmetric matrices, in C as one flat buffer of `n_props` contiguous `norb × norb` row-major
blocks (entry `(i,p,q)` at `m[i*norb*norb + p*norb + q]`, `m_len` exact). Each block must be the
full symmetric matrix, both triangles. A half triangle, a non-finite entry, a wrong `norb`, more
than three components, or a component carrying more than one flux class is a caller error
(`QNTT_INVALID_ARG` with a message); a component whose flux is not `φ` gives a zero row. The
frequency-dependent α(ω) and the iterative moments are Rust-only. Physics refusals return
`QNTT_NOT_CONVERGED` with a `QNTT_SECTOR_REFUSAL_*` code (`NOT_TT_CRITICAL`, `HESSIAN_INDEFINITE`,
`SYMPLECTIC_UNSTABLE`, `SOLVE_NOT_CONVERGED`) and write no output; Julia raises `QNTTError` with
the same diagnosis. Every output length is bounded by the channel dimension `dim`, which
`qntt_f64_sector_rpa` reports cheaply at `k = 0` (every buffer NULL), so size buffers from it and
call once.

## Critical points vs linear response

Grazioli–Hu–Cancès distinguish two routes to excited states: **critical-point search** (find other
critical points ``x_k`` of the energy; take ``E(x_k) - E(x_0)``) and **linear response** (this
one). For linear models — FCI, and therefore full-rank TT — the two coincide. For nonlinear
approximations they do not, and their paper shows on H₄ that some index-1 saddle points of the
UHF energy are *spurious*: artifacts of the nonlinearity rather than approximations of excited
states. Fixed-rank TT is likewise a nonlinear ansatz, so the same caution applies to any
critical-point interpretation here; the linear-response route implemented in this library does not
depend on locating those saddles.
