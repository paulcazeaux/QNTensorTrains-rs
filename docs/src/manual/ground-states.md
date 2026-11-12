# Ground states

The ground-state drivers return `(λ, x)` with `λ` the **electronic** energy; add `QNTT.e0(data)`
for the total. None of them mutates the starting train.

## Which one

| driver | manifold | when |
|---|---|---|
| [`QNTT.als`](@ref) | fixed — the ranks of `x0` | you know the rank you want; cheapest per sweep |
| [`QNTT.mals`](@ref) | adaptive, capped by a schedule | the usual choice: it discovers the ranks |
| [`QNTT.tdvp`](@ref) | adaptive, truncation-controlled | imaginary-time relaxation; robust from poor starts |
| [`QNTT.lanczos`](@ref) | rank grows with the Krylov space | small systems, or a reference |
| [`QNTT.riemannian`](@ref) | fixed | the geometric route; also what prepares a state for [linear response](linear-response.md) |
| [`QNTT.excited_states`](@ref) | adaptive (state average), then fixed (penalized polish) | not a ground state at all — the low-lying block as a certified ladder; see [Excited states](excited-states.md) |

## ALS and MALS

```julia
x0 = QNTT.tt_randn(space, QNTT.uniform_ranks(space, 8); seed = 1234)

λ, x          = QNTT.als(H, x0; eps = 1e-8, max_iter = 30)
λ, x, history = QNTT.mals(H, x0; bond_dims = [50, 100, 200], eps = 1e-8, max_iter = 20)
```

`bond_dims` is the per-outer-iteration POOLED rank cap schedule (see [Rank grids](spaces.md)); once
the sweep count exceeds its length the last entry is held. `[0]` means uncapped — every bond keeps
whatever rank the `eps`-truncated two-site SVD gives it.

The pattern that works well in practice is a ramp: a few cheap sweeps at low rank to get into the
right basin, then growth.

```julia
λ, x = QNTT.mals(H, x0; bond_dims = [20, 20, 50, 100, 200], eps = 1e-10, max_iter = 30)
```

Watch `QNTT.max_bond_rank(x)` afterwards: if it sits exactly at your cap, the cap — not `eps` —
determined the answer, and the energy is not converged in rank.

## Imaginary time

```julia
λ, x = QNTT.tdvp(H, x0; dtau = 0.1, eps = 1e-6, trunc = 1e-8, max_sweeps = 100)
```

Second-order symmetric sweeps of imaginary-time size `dtau`, renormalized after each. Slower than
MALS to high accuracy but less prone to sticking; useful as an independent check, or as a warm
start.

## Lanczos

```julia
λ, x, info = QNTT.lanczos(H, x0; tol = 1e-6, max_iter = 20)
info.lambdas       # Ritz value per iteration
info.ranks         # rank before rounding
info.rounded_ranks # after
info.breakdown
```

`breakdown` is not a failure on a small system — it means the Krylov space was exhausted, which is
the expected outcome once the space is spanned.

## Riemannian optimization

```julia
λ, x = QNTT.riemannian(H, x0; cg = true, eps = 1e-6, max_iter = 200)
```

Conjugate gradients (or steepest descent with `cg = false`) on the *fixed-rank manifold*. Unlike
the sweep drivers this optimizes the whole
train at once, and it leaves you at a genuine critical point of the energy restricted to the
manifold — the precondition for [linear response](linear-response.md).

## Checking convergence

```julia
using LinearAlgebra
norm(H * x - λ * x) / norm(x)          # residual of the eigenvalue equation
QNTT.rayleigh_quotient(H, x)           # should agree with λ
QNTT.max_bond_rank(x)                  # against your cap
```

On a full-rank manifold (`uniform_ranks(space, r)` with `r ≥ maxranks`) the TT ansatz is exact and
these drivers return the FCI energy, which is the cleanest possible validation on a small system.

Two things worth knowing about the pooled bond-dimension cap:

- `bond_dims` caps the **total** bond dimension, summed over symmetry sectors — the conventional
  DMRG `M`, so the numbers are directly comparable to the literature.
- **A flat rank curve is necessary but not sufficient.** Two-site DMRG started from a single
  determinant is stuck at a fixed point: the determinant already solves every local eigenproblem,
  so the energy stops moving in `M` while sitting far above the minimum. On N₂/cc-pVDZ this costs
  a factor of 2.5 in recovered correlation energy, silently.

### The noise schedule

The principled fix is `noise`, a per-iteration amplitude alongside `bond_dims`:

```julia
λ, x, history = QNTT.mals(H, hf;
                          bond_dims = [8, 16, 32, 64, 64],
                          noise     = [1e-4, 1e-4, 1e-5, 0.0, 0.0],
                          eps = 1e-9, max_iter = 20, seed = 3)
```

Before each truncation the kept basis is enriched, so a sector carrying no weight in the current
state can still survive — exactly the direction a determinant start has none of. On N₂/cc-pVDZ
this takes the run from about a third to nearly 90% of the correlation energy (notebook 02).

!!! warning "End the schedule at zero"
    The driver will not stop on a noisy iteration, so every schedule must end at `0.0`. Under
    `noise_type = :random` the noise-free tail is also the only part that carries a variational
    energy at all, because that flavour perturbs the state rather than just the choice of kept
    basis.

!!! tip "Amplitude matters, and 1e-4 is a good default"
    The noise amplitude is a sensitive knob. A typical schedule runs `1e-4` decaying to
    `1e-5` then `0`; an amplitude of `1e-2` is far too large and *degrades* an otherwise healthy
    run. Start around `1e-4`.

### Which noise flavour

`noise_type` picks how the enrichment is built. **The default is `:perturbative`** — White's
density-matrix noise, and the flavour to reach for unless you have a reason not to:

```julia
λ, x, history = QNTT.mals(H, hf;
                          bond_dims = [8, 16, 32, 64, 64],
                          noise     = [1e-4, 1e-4, 1e-5, 0.0, 0.0],
                          noise_type = :perturbative,   # the default; :random is the backup
                          eps = 1e-9, max_iter = 20)
```

It enriches the kept basis with the directions the *Hamiltonian* couples the state to. It is
**variational throughout** (even a noisy iteration gives an upper bound), **deterministic** (the
`seed` does nothing), and **exact at full rank** — sustained noise recovers the noise-free answer.
The amplitude is normalized internally, so one nominal `1e-4` works across bonds and molecules.

`:random` is the backup, for the case where it wins on the merits: from a **bare** determinant or
CSF start, the perturbative enrichment is built from the current state and
so cannot leave that state's subspace — MALS then stalls at a genuine critical point above the true
ground, at every rank cap. Fill the start's rank grid with random content first, or run that leg on
`:random`.

!!! note "The amplitude means different things to the two flavours"
    On `:perturbative` an entry of `noise` is a *nominal*, normalized amplitude. On `:random` it is
    the relative amplitude of the perturbation added to the two-site block itself. `~1e-4` is the
    right starting order either way, but a value tuned on one flavour does not transfer verbatim to
    the other.

`history` is what tells you whether the ramp converged:

```julia
for sw in history
    @printf("M = %-5d noise = %.0e   E = %.10f   discarded = %.2e\n",
            sw.bond_dim, sw.noise, sw.energy, sw.discarded_weight)
end
```

The discarded weight is the standard rank-convergence diagnostic — the quantity a production run
extrapolates to zero.
