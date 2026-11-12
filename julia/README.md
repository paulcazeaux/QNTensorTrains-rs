# QNTT.jl

Julia bindings for [`qntt`](../README.md) — quantum-number-preserving block-sparse tensor trains
for fermionic many-body systems.

The whole Rust stack is callable: both symmetry layers (abelian `(Nup, Ndn)` and SU(2)
spin-adapted), both scalars (`Float64`, `ComplexF64`), the FCIDUMP readers, the sweep solvers
(ALS / MALS / TDVP / Lanczos / Riemannian), the RPA / linear-response drivers, and — on the
`Float64` layers — the joint TT ⊕ orbital stack (`cas_reference_state`, `casscf`, `joint_rpa`) and
gentlest-ascent dynamics (`saddle_seed`, `gad`). Calls cross the boundary once per *operation*, so
the wrapper adds no meaningful overhead.

## Setup

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release -p qntt-capi          # builds libqntt_capi.{dylib,so,dll}
julia --project=julia -e 'using QNTT; QNTT.version()'
```

The package finds the shared library automatically from the crate's cargo target directory
(`release`, then `debug`). To pin it — for a `QNTT_jll` artifact, or a build somewhere unusual:

```julia
QNTT.set_library_path!("/path/to/libqntt_capi.dylib")   # persistent, via Preferences
```

or set `ENV["QNTT_LIB"]` before `using QNTT` for a one-off. To use the package from another
project:

```julia
using Pkg; Pkg.develop(path = "/path/to/qntt/julia")
```

`capi/build_tarballs.jl` is a BinaryBuilder recipe that cross-compiles the library for every
supported platform and can emit an unregistered `QNTT_jll`:

```sh
julia --project=. capi/build_tarballs.jl --verbose --deploy=local
```

Run the test suite (needs the library built first):

```sh
julia --project=julia julia/test/runtests.jl
```

## Quick start

```julia
using QNTT, LinearAlgebra

data  = QNTT.read_fcidump("testdata/h4.FCIDUMP")   # norb, nelec, 2Sz, E_core, t, v
space = QNTT.QnSpace(data)                          # target sector from the dump's header
H     = QNTT.Hamiltonian(space, data)               # build the Hamiltonian

x0    = QNTT.tt_randn(space, QNTT.uniform_ranks(space, 8); seed = 1234)
λ, x  = QNTT.mals(H, x0; bond_dims = [0], eps = 1e-8)

E = λ + QNTT.e0(data)                               # electronic + core energy
```

`x` is an ordinary Julia value: `norm(x)`, `dot(x, y)`, `H * x`, `2x - y`, `round(x; eps = 1e-8)`,
`Array(x)` (dense, small systems), `QNTT.ranks(x)`, `QNTT.getblock(x, site, :up, (1, 0))`.

A complete driver is in [`examples/ground_state.jl`](examples/ground_state.jl):

```sh
julia --project=julia julia/examples/ground_state.jl testdata/h4.FCIDUMP 8 20
```

## Conventions

Everything is **0-based**, as in the Rust library: sites `0:d-1`, bonds `0:d`, and sector labels
`(a, b)` counting electrons strictly to the left of a bond. The exception is integral arrays —
`QNTT.onebody(data)` and `QNTT.twobody(data)` are ordinary 1-based Julia arrays, `t[i,j]` and
`v[i,j,k,l]` in the chemists' `(ij|kl)` convention, and `Hamiltonian(space, t, v)` takes them that
way. (The C ABI underneath is row-major; the wrapper transposes on the way in.)

Two energy accessors, deliberately distinct:

| call | value |
|---|---|
| `QNTT.expectation(H, x)` | `⟨x, H·x⟩` — **not** normalized (the Rust `RayleighQuotient`) |
| `QNTT.rayleigh_quotient(H, x)` | `⟨x, H·x⟩ / ⟨x, x⟩` |

**Point-group symmetry** is carried by **both** layers — the group itself is always abelian
(`:C1`…`:D2h`), the layer need not be. `orbsym` goes *in* as the 1-based Molpro ids an FCIDUMP
writes; irreps come *out* as 0-based XOR indices, since they are sector labels — convert with
`QNTT.irrep_from_molpro` / `QNTT.molpro_from_irrep`. `QNTT.read_fcidump` parses `ORBSYM`/`ISYM`
and **both** `QNTT.QnSpace(data)` and `QNTT.Su2Space(data)` inherit them, so a symmetric dump gets
its symmetry with no extra call; a C1 dump gives the trivial-group space. Naming a different same-order group changes irrep *names*
only — all same-order abelian groups share one product table.

Pass `point_group = nothing` to `Su2Space(data)` to drop the dump's symmetry. On a *genuinely*
symmetric dump, call `QNTT.screen_forbidden!(data)` before building a Hamiltonian over a
non-trivial-group space: symmetry-forbidden integrals carry ~1e-14 noise that otherwise makes the
Hamiltonian construction fail.

```julia
sc = QNTT.QnSpace(4, 2, 2; point_group = :C2v, orbsym = [2, 3, 4, 1], target = 2)
QNTT.nirrep(sc)                  # 4 — `maxranks`/`ranks` gain a third axis, g[a+1, b+1, ir+1]
QNTT.irrep_name(sc, 2)           # "B1"
```

## API

```julia
# spaces
QnSpace(d, nup, ndn); QnSpace(data); Su2Space(d, n, two_s)
Su2Space(data; two_s = nothing, point_group = :inherit)   # `nothing` drops the dump's group
QnSpace(d, nup, ndn; point_group = :C2v, orbsym = [...], target = 0)   # 1-based Molpro ids in
Su2Space(d, n, two_s; point_group = :C2v, orbsym = [...], target = 0)  # same on the SU(2) layer
bond_sectors(space, bond); grid_extent(space); maxranks(space); uniform_ranks(space, r)
istrivial(space); nirrep(space); point_group(x)

# point-group tables (layer-independent)
pg_order(kind); irrep_name(kind_or_space, ir)
irrep_from_molpro(kind, id); molpro_from_irrep(kind, ir); orbsym_to_irreps(kind, orbsym)

# integrals
read_fcidump([T,] path; symmetry = :eightfold, point_group = :auto)
norb, nelec, ms2, e0, onebody, twobody, orbsym, isym, pointgroup
screen_forbidden!(data; point_group = :auto)     # REQUIRED on a symmetric dump, in place
read_tcdump([T,] path, norb; symmetry = :full48)   # three-body / transcorrelated
threebody([T,] norb); setcoo!(w, idx, vals); nnz(w)

# states
tt_zeros, tt_from_ranks, tt_randn, tt_state(occ_up, occ_dn)     # abelian
tt_csf(codes); tt_csf(codes, space)                              # SU(2): :empty/:up/:dn/:occ
ranks, orthogonality_center, isvalid, nsites, max_sector_rank, max_bond_rank
getblock, setblock!, blocks, Array

# operators
Hamiltonian(space, t, v; w, eps, construction)   # :default|:median|:bipartite|:bipartite_per_sector
Hamiltonian(space, data::Integrals; ...)
H * x; expectation; rayleigh_quotient; xthy; automaton_bond_dims

# algebra
norm, dot, +, -, *, /, round, round!, round_sum, left_orthonormalize!

# state diagnostics (print all three at every stage of a certification narrative)
rho(x)                  # |(x,x)|/<x,x>; = 1 means a "complex" run never left the real manifold
bilinear_dot(x, y)      # the unconjugated (x,y), as against dot(x,y) = <x,y>
schmidt(x)              # -> (sigma, weight) per interior bond; sum(w .* s.^2) == 1 exactly, with
sigma_min(x)            #    w = 1 (abelian) or centerweight(2s,2S) (SU(2)). sigma stays REDUCED;
                        #    sigma_min is the manifold-boundary alarm
grad_norm(H, x)         # ||P(Hx - lambda x)||, the criticality gate (RPA reference: < 1e-9)

# persistence (a local CACHE, not an archive: no cross-version promise, no migration)
save(x, path); load(T, L, path); load(TTVector{T,L}, path); load(path)

# solvers                                        # als/tdvp/riemannian return (λ, x)
als(H, x0; eps, max_iter, verbose)
mals(H, x0; bond_dims, noise, noise_type, eps, max_iter, seed, verbose)   # -> (λ, x, history)
tdvp(H, x0; dtau, eps, trunc, max_sweeps, chi, exptol)
lanczos(H, x0; tol, max_iter)                    # returns (λ, x, info) with the sweep history
riemannian(H, x0; cg, eps, max_iter, gradtol, seed, variant)

# linear response
rpa(H, x; k, tol, max_iter, seed)   -> (ω, info)   # matrix-free
rpa_dense(H, x)                     -> ω           # dense reference, small systems only

# Hessian diagnosis and the complex manifold
pd_probe(H, x; part, budget, seed)  # part = :rr | :ii  (on ComplexF64 states :rr IS 𝔥, :ii throws)
    -> (verdict = :witness_negative | :likely_pd | :inconclusive,
        lambda, residual, is_pd = true|false|nothing, applies)
to_c64(x)                                       -> TTVector{ComplexF64}
complex_seed(H, x; eps, budget, seed)           -> (x_c::TTVector{ComplexF64}, lambda)

# misc
QNTT.init(nthreads); QNTT.nthreads(); QNTT.version()
```

Failed calls throw `QNTTError`, carrying the Rust library's own message — including the text of
core-crate assertions, which are caught at the boundary rather than aborting the process.

## Notes

- **Memory** is managed by finalizers. Handles are refcounted on the Rust side, so a handle that
  depends on another keeps it alive even if you drop your reference to it.
- **Threads**: the library runs its own rayon pool (`QNTT.init(n)` to size it). It never calls
  back into Julia, so there is no GC-safepoint interaction.
- **In-place operations** (`round!`, `setblock!`, `left_orthonormalize!`) require sole ownership
  of the object; an aliased handle is refused rather than silently copied.
- **A PD verdict is a diagnosis, not a status.** `pd_probe` returns normally for all three
  verdicts; `:witness_negative` is the *goal* of the `:ii` probe (it is the proven direction to
  complexify along), and `:inconclusive` means only "no negative direction found in this budget".
  Read `is_pd`'s three-way answer rather than collapsing `nothing` into a boolean, and always read
  `residual` next to `lambda` — a `:likely_pd` whose residual is within an order of `lambda` says
  nothing. Raise `budget` until it is small.
- **`save`/`load` are a cache.** There is no forward-compatibility promise and no migration path
  between library versions. A file records its layer and scalar, so the
  wrong family throws rather than reinterpreting bytes — but nothing in the file knows *which run*
  you meant, so check the space and rank grid after loading. A stale cache silently invalidates
  every number downstream of it.
- **`mals` takes a DMRG schedule.** `bond_dims` caps the *total* bond dimension summed over
  sectors (the conventional `M`); `noise` is the matching per-iteration noise amplitude, which lets
  an empty symmetry sector survive truncation. That is what breaks the fixed point a single
  determinant sits at — started there, a noise-free sweep converges, looks converged, and stops far
  above the minimum. **End every schedule at `0.0`**: a noisy sweep perturbs the state, so only the
  final noise-free sweeps carry a variational energy. The returned `history` records the per-sweep
  energy, discarded weight, bond dimension and noise. The amplitude is sensitive — a good default is
  `1e-4` decaying to `0`; `1e-2` is far too large and degrades a healthy run.
- **`round_sum` copies its summands.** At large rank, prefer accumulating with `+` and a single
  `round!` if it shows up in a profile.
- **`chi` is a pooled total bond dimension; on SU(2) it counts reduced multiplets.**
- The C ABI is documented in [`../capi/include/qntt.h`](../capi/include/qntt.h).
