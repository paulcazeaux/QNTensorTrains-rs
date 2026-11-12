# Excited states

Three routes reach an excited state, and they are not variations on one another: two are
variational DMRG sweeps, the third is a driver that composes them with a Riemannian polish and
gates the result. A fourth — [linear response](linear-response.md) — answers a different question
at a single state and is included here only so the comparison is on the table.

At full rank all of them agree with FCI; [step-17](../tutorial/step-17.md) runs the whole
comparison end to end on LiH/STO-3G, on both symmetry layers.

## Which route

| route | what it optimizes | when |
|---|---|---|
| [`QNTT.state_average_mals`](@ref) | `n` roots at once, one shared basis | you want the low-lying block together, at a common accuracy, cheaply |
| [`QNTT.state_specific_mals`](@ref) | one root against fixed lower states | you have the states below and want *this* one at its own rank |
| [`QNTT.excited_states`](@ref) | both, plus a penalized polish, plus gates | you want certified rungs and a diagnostic record |
| [`QNTT.rpa`](@ref) | excitation *frequencies* at one state | you already have a ground state and want the response spectrum |

The first two are ordinary two-site DMRG with one thing changed each. The state average changes
the **truncation** — one density matrix pooled over the roots, so every root lives in the same
basis. The state-specific sweep changes the **local operator** — a penalty that pushes a single
root off states you already hold.

`rpa` is a different approximation altogether: it linearizes at one state and returns each
distinct frequency once. It coincides with the ladder gaps at full rank and separates from them
under truncation — and because it reports distinct values, its rows stop lining up with the
ladder's the moment a level is skipped or degenerate.

## State average

```julia
data = QNTT.read_fcidump("testdata/h4.FCIDUMP")
s    = QNTT.QnSpace(data)
H    = QNTT.Hamiltonian(s, data)
x0   = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 4); seed = 2024)

λs, xs, hist = QNTT.state_average_mals(H, x0, 3; eps = 1e-10, bond_dims = [0], max_iter = 30)
```

One train carries all three roots: every core is shared except the orthogonality centre, and
the bond is truncated once against the weighted density matrix `ρ = Σᵢ wᵢ BᵢBᵢᴴ` of the roots.
The returned states are orthonormal by construction.

Four things follow from the shared basis:

- **Ask for one root more than you need.** The top root converges at the gap to the first
  *excluded* level, not to the one below it; a small gap there means slow convergence.
- **The ranks inflate.** The driver does **not** round the returned states — they carry the shared
  grid. `round(x; eps = 1e-10)` each one if you want its own rank.
- **The weights must be valid.** `weights` must be finite, strictly positive, **non-increasing**
  and sum to 1 (to `1e-12`); anything else is refused, never silently repaired.
- **`noise_type = :random` is refused.** The state average supports only the perturbative noise.

`nroots = 1` is the same calculation as [`QNTT.mals`](@ref).

Reading the history ([`QNTT.SaSweep`](@ref)):

```julia
for sw in hist
    @printf("M = %-4d noise = %.0e  Σwλ = %.10f  dev = %.1e  discarded = %.2e\n",
            sw.bond_dim, sw.noise, sw.energy, sw.max_dev, sw.discarded_weight)
end
```

`energy` is `Σᵢ wᵢ λᵢ`, the quantity the sweep actually minimizes — not any single root.
`max_dev` is the largest relative change of a root energy over the sweep, `discarded_weight` the
**pooled** truncation of the shared basis, and `overlap_dev` how far the truncated roots were from
orthonormal before they were re-orthonormalized.

## State specific

```julia
λ0, g, _  = QNTT.mals(H, x0; eps = 1e-10, bond_dims = [0])
λ1, x1, h1 = QNTT.state_specific_mals(H, x0, [g]; eps = 1e-10, bond_dims = [0], max_iter = 30)
abs(dot(x1, g))            # the residual overlap the penalty left behind
```

Each local eigenproblem sees `H_eff + w Σⱼ pⱼ⟨pⱼ, ·⟩`, where `pⱼ` is the lower state's image in
the current two-site space.

`penalty = :auto` (the default) recomputes the weight at the top of every iteration as
`w = max(10(λ⊥ − minⱼ Eⱼ), 1)`, where `λ⊥` is the energy of the component **orthogonal** to the
lower states. If the state ever ends an iteration more than halfway inside the penalized span
(`Σⱼ oⱼ² > ½`), the rule warns and escalates tenfold, and that iteration cannot satisfy the stop
test.

`mode = :project` searches the orthogonal complement instead of penalizing it, and records
`penalty == 0`. A bond whose local space cannot host the projected directions is *skipped*
(counted in `history[end].skipped`), never refused.

A number — `penalty = 5.0` — is used verbatim and is **unguarded**. A weight below the gap to the
target leaves the lower state as its own minimizer.

Two more contract points:

- Lower states need not be normalized (`1/‖ψⱼ‖` is folded into `pⱼ`, the reference energies and
  the recorded overlaps) and are never mutated. A zero-norm state is an error.
- If the lower states are themselves **approximate**, the target energy carries a
  **sign-indefinite** second-order error of order `gap·|⟨ψⱼ,x⟩|²`. Nothing in `history` can see
  it. That is precisely why [`QNTT.excited_states`](@ref) puts its supplied lower states through
  an exact gradient gate.

`λ` is the plain Rayleigh quotient of `H` alone; `history[end].penalized` is the penalized Ritz
value the sweep minimized, and is never below it. With `lower` empty **and** `roots = 1` (the
default) the driver is plain [`QNTT.mals`](@ref) and records `penalty = 0` and `numops = 0`.
Unlike the state average, this path **accepts** `noise_type = :random`.

## The ladder, and what its licence certifies

```julia
λs, xs, info = QNTT.excited_states(H, x0; nroots = 3, eps = 1e-10)
info.licence, info.overlap_dev, info.gradnorms
```

Stage 0 enriches a rank-1 start with one noisy iteration (`info.enriched`) — a determinant or CSF
start is fixed here rather than refused, unlike the two sweeps above. Stage 1 is a state average
for the starts. Stage 2 then runs **one root at a time, bottom-up**: a state-specific sweep
against everything already final below it, a penalized Riemannian polish to a critical point of
`J = (⟨x,Hx⟩ + w Σⱼ|⟨ψ̂ⱼ,x⟩|²)/⟨x,x⟩`, and rank-trim passes. Root `i` is finished before root
`i+1` starts, so each rung is built against polished states below it. Stage 3 checks the result.

```
licence = all(converged) && overlap_dev ≤ overlap_tol && swaps == 0
```

!!! warning "The licence never certifies completeness"
    Every clause above is a statement about the roots you got: that they are critical, mutually
    orthogonal, and in ascending order. None of them says a level was not **skipped**. The cause
    is truncation: the shared state-average basis can drop a sector that carries no weight, and
    enrichment noise cannot repopulate a sector `H` does not couple the current states to. A
    truncated ladder can then return a higher level in place of a missing one with the licence
    **true**; at full rank the ladder is complete.

Remedies, in increasing cost:

1. Ask for `nroots + 1` and compare the sets.
2. Run two starts and compare the sets.
3. Start from a state that already carries the suspect symmetry.

Refusals are warned and recorded, never repaired. A supplied `lower` state whose exact gradient
exceeds `lower_grad_tol` is named in the warning and its gate value kept in
`info.lower_gradnorms`; a rung that sits below the one before it is flagged in `info.swapped`.

`info` also carries the per-root record: `converged`, `stalled`, `gradnorms` (`NaN` when
`polish = false` — no critical point was sought, so there is no gate and no trim), `penalties`,
`passes`, `sa_energies`, `ss_energies`, `j_ss`, `j_pol`, and `overlaps`, the `(m+n)×(m+n)`
absolute Gram over `[lower; roots]`.

!!! note "Schedules: empty does not mean off"
    `weights = nothing` is `1/nroots`, `bond_dims = nothing` is the default ramp of `rank`, but
    `noise = nothing` is a built-in decaying ramp ending at `0.0`, and `ss_noise` defaults to
    `[0.0]`. Pass `noise = [0.0]` for a noise-free ladder. This is the opposite
    convention from [`QNTT.mals`](@ref) and the two sweeps, where an empty `noise` means noise-free.

## Pooled bond-dimension caps and the residual overlap

At a **binding** pooled bond-dimension cap the fixed-rank penalized minimizer does not become
exactly orthogonal to the lower states: it keeps a residual overlap of order `1/w`. That can exceed
the default `overlap_tol = 1e-6`, and the licence is then refused on orthogonality alone, every
root having converged.

The remedy is `rank` or the penalty, **never** the tolerance. Loosening `overlap_tol` does not
make the states more orthogonal; it makes the licence stop reporting that they are not.

[step-17](../tutorial/step-17.md) shows the routes side by side on LiH/STO-3G at full rank and
at two binding caps. Below full rank the polished energy is typically the lowest of the three
ladder stages, but that ordering is not a theorem — do not build a check on it.

## Spin adaptation

Everything above works on the SU(2) layer, with the usual reading:

- `rank` and every `bond_dims` entry are **reduced multiplet** ranks, not dimensions.
- Every root carries `2S` exactly, so the reference to compare against is the `S = 0`
  **sub**-spectrum — the `(nup, ndn)` levels with no `(nup+1, ndn−1)` partner.
- `trim_tol` reads the centerweighted Schmidt value, and on a saturated reduced grid a trim is
  lossy by construction.
- At `2S > 0` the pooled `discarded_weight` at `nroots ≥ 2` is not the same quantity as the
  single-state one at `nroots = 1` (the centerweighted projection loss, against the unweighted
  θ-Frobenius ratio). Compare like with like.

```julia
su2 = QNTT.Su2Space(data)
H2  = QNTT.Hamiltonian(su2, data)
y0  = QNTT.tt_randn(su2, QNTT.uniform_ranks(su2, 64); seed = 3)

λ2, y2, i2 = QNTT.excited_states(H2, y0; nroots = 3, eps = 1e-10)
```

A completeness example: on LiH from the aufbau CSF the ladder can skip the degenerate Π singlet
pair and be certified anyway, and whether it finds the pair is not monotone in `rank`. Compare
the sets as above.

## From C

The same three entry points exist in the C ABI for all four symbol families
(`qntt_f64_`, `qntt_c64_`, `qntt_su2_f64_`, `qntt_su2_c64_`); every energy is a `double` on every
family, including the complex ones. See [C ABI](../capi.md) for the wire codes and the buffer
rules.

```c
QnttExcitedOptions opts;
qntt_excited_options_default(&opts);
opts.nroots = 3;
opts.rank   = 16;
opts.eps    = 1e-10;

double  energies[3];
void   *roots[3] = {NULL, NULL, NULL};
QnttExcitedInfo info;
QnttExcitedRoot rows[3];
double  gram[9];                       /* (nlower + nroots)^2, row-major over [lower; roots] */
size_t  gram_len = 0;

int32_t st = qntt_f64_excited_states(H, x0, NULL, 0, &opts,
                                     NULL, 0,        /* weights   -> 1/n            */
                                     NULL, 0,        /* bond_dims -> ramp of rank   */
                                     NULL, 0,        /* noise     -> the default ramp */
                                     NULL, 0,        /* ss_noise  -> {0.0}          */
                                     energies, roots, 3,
                                     &info, rows, NULL,
                                     gram, 9, &gram_len,
                                     NULL, 0);       /* keep_sa == 0: no SA handles */
if (st == QNTT_OK && info.licence) {
    for (size_t i = 0; i < 3; ++i)
        printf("root %zu: E = %.12f  gate = %.2e\n", i, energies[i], rows[i].gradnorm);
}
for (size_t i = 0; i < 3; ++i) qntt_f64_vec_free(roots[i]);
```

A handle-array capacity below `nroots` is `QNTT_INVALID_ARG` **before the solver runs**, and a
refused call returns no handles. Free every returned root. See [C ABI](../capi.md) for the
sweep-history buffers.

## See also

- [step-17](../tutorial/step-17.md) — the state-average / state-specific / polish / RPA / FCI
  comparison on LiH, on both layers.
- [Ground states](ground-states.md) — the single-root drivers these are built on.
- [Linear response (RPA)](linear-response.md) — the response route, and what it shares with the
  ladder gaps.
