# QNTensorTrains-rs

**Quantum-number-preserving block-sparse tensor trains for fermionic many-body systems, in Rust.**

`qntt` implements matrix-product-state / tensor-train (TT / MPS) methods for electronic-structure
and fermionic lattice problems. The spin-up and spin-down electron counts (`Nup`, `Ndn`) — or, on
the spin-adapted layer, the total spin — are held **exactly**, as block-sparse quantum numbers on
every tensor core: not enforced by penalty, not recovered by projection, but built into the data
structure. Second-quantized Hamiltonians are compiled to compact operator automata and applied by
block-sparse gather/scatter. It targets ground- and excited-state calculations at DMRG scale.

Linear algebra is pure Rust via [faer](https://crates.io/crates/faer) — no system BLAS/LAPACK
dependency.

## Features

- **Block-sparse quantum-number TT cores.** Every core is sparse over the sector grid; storage and
  every kernel operate per sector, so the symmetry is exact and free.
- **Two symmetry layers.** The abelian `(Nup, Ndn)`-conserving layer (`src/`) and a full **SU(2)
  spin-adapted** layer (`src/su2/`) with the same TT / Hamiltonian / solver stack over irreducible
  spin sectors.
- **Second-quantized Hamiltonians.** Read FCIDUMP integrals and compile the one-/two-body (and
  optional three-body / transcorrelated) operator to a compact, **exact-integer, cutoff-free**
  representation applied block-sparse.
- **Rounding.** Deterministic SVD / rank-revealing-QR truncation.
- **Solvers.** One- and two-site ALS / MALS (DMRG) with rank ramps and noise schedules, Lanczos,
  Davidson and PINVIT eigensolvers, matrix-exponential action, TDVP time evolution, and Riemannian
  optimization on the fixed-rank manifold.
- **Excited states.** State-averaged and state-specific MALS ladders, and gentlest-ascent saddle
  search.
- **Linear response.** Symplectic (RPA) eigenvalues of the Riemannian Hessian at a converged state,
  matrix-free or through a certified dense driver.
- **Joint (TT ⊕ orbital) optimization and response, on both layers.** A DMRG-SCF ground state in
  which the tensor-train coefficients *and* the one-particle basis are optimized together, the
  symplectic eigenvalues of the joint Hessian, and the response stack on the joint manifold —
  static polarizabilities, ω-dependent linear response, transition moments and oscillator
  strengths.
- **Point-group symmetry on both layers.** An optional abelian irrep quantum number on every bond.
  It is a trade-off rather than a free win, so measure it on your system. A symmetric FCIDUMP
  **must** be screened with `fcidump::screen_point_group_forbidden` before construction.
- **Real and complex** scalars (`f64`, `C64`), generic over a `Scalar` trait.
- **Threaded** with rayon; every threaded path is tested against its serial reference.

## Quick start

```sh
cargo test                 # full test suite
cargo build --release
```

Build the N₂ / cc-pVDZ Hamiltonian (28 orbitals) and run a MALS ground state on it:

```sh
cargo run --release --example step-06 -- mals 50 2
```

Library sketch — read integrals, set the target sector, and compile the Hamiltonian:

```rust
use qntt::fcidump::{read_fcidump, TwoBodySymmetry};
use qntt::{QnSpace, SparseHamiltonian};
use std::path::Path;

// Integrals and target sector (orbitals, electrons, 2·Sz).
let data = read_fcidump::<f64>(Path::new("system.FCIDUMP"), TwoBodySymmetry::Eightfold)?;
let space = QnSpace::from_n_two_sz(data.norb, data.nelec, data.ms2);

// Compile the second-quantized operator (default = the bipartite / vertex-cover construction).
let h = SparseHamiltonian::new(space, &data.t, &data.v, None, f64::EPSILON);

// `h.apply(&x)` and `rayleigh_quotient(&h, &x)` act on a block-sparse `TtVector` state `x`;
// see `examples/tutorial/step_06_scaling_up.rs` for building states and running the ALS / MALS solvers.
```

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Abelian `(Nup, Ndn)` layer: core TT types (`vector`, `core`), `hamiltonian/`, `solvers/`, `round`, `fcidump` |
| `src/su2/` | SU(2) spin-adapted layer — parallel TT / Hamiltonian / solver stack |
| `examples/` | The tutorial programs — indexed in [`examples/README.md`](examples/README.md) |
| `tests/` | Invariant-based integration tests |
| `capi/` | C ABI (`cdylib`) over both layers and both scalars — `include/qntt.h` is the contract; `build_tarballs.jl` is the BinaryBuilder recipe |
| `julia/` | `QNTT.jl`, the Julia driver package built on that ABI |
| `notebooks/` | Worked examples (ground states, linear response), generated from `mknotebooks.py` |
| `testdata/` | FCIDUMP integral fixtures used by the tests |
| `docs/` | Documenter.jl + Doxygen sources: manual, tutorial and API reference |

## Calling it from Julia

The full stack — both symmetry layers, both scalars, every solver — is callable from Julia through
the C ABI (`capi/`, `crate-type = ["cdylib", "staticlib"]`) and the `QNTT.jl` package in `julia/`.
Calls cross the boundary once per operation, never per matvec.

```sh
cargo build --release -p qntt-capi
julia --project=julia julia/examples/ground_state.jl testdata/h4.FCIDUMP
```

```julia
using QNTT

data  = QNTT.read_fcidump("testdata/h4.FCIDUMP")
space = QNTT.QnSpace(data)
H     = QNTT.Hamiltonian(space, data)
x0    = QNTT.tt_randn(space, QNTT.uniform_ranks(space, 8); seed = 1234)
λ, x  = QNTT.mals(H, x0; bond_dims = [0], eps = 1e-8)
ω, _  = QNTT.rpa(H, x; k = 5)          # fixed-rank linear-response spectrum
```

Worked examples live in [`notebooks/`](notebooks/) — ground states for H₄, LiH, H₂O and N₂ with ALS
and MALS, and excitation energies from the Riemannian Hessian checked against exact FCI gaps. See
[`julia/README.md`](julia/README.md) for the full API and
[`capi/include/qntt.h`](capi/include/qntt.h) for the ABI contract. Other languages can use the same
shared library: every function returns a status code, results travel through out-pointers, and
objects are opaque refcounted handles. [`capi/build_tarballs.jl`](capi/build_tarballs.jl)
cross-compiles it for every supported platform and can emit an unregistered `QNTT_jll`.

## Design notes

- **Symmetry is structural.** Quantum numbers index the block structure; they are never approximate
  and cannot drift. The operator is exact-integer and cutoff-free — no screening threshold decides
  the answer.
- **Correctness by oracle.** Kernels are certified against an independent dense (Jordan–Wigner)
  materialization at small system size, cross-checked between constructions, and pinned for
  run-to-run determinism; threaded paths must match their serial counterparts. Tests assert
  invariants and tolerances, never a particular factor layout.
- **Pure-Rust numerics.** All dense linear algebra goes through faer; there is no BLAS/LAPACK or
  other system dependency to install.
- **0-based throughout** — sites, bonds, and electron-count sectors.

## Status

Research library under active development — `v0.1`, API not yet stable. Both symmetry layers are
implemented and covered by an invariant-based suite of ~2,700 Rust tests plus the Julia binding
tests.

## Citing this work

If `qntt` contributes to published work, please cite it — see [`CITATION.cff`](CITATION.cff). A
methods paper is in preparation and will become the preferred reference.

## Acknowledgments

This work was supported by the U.S. National Science Foundation under Award No. 2608677
(Division of Mathematical Sciences).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE). Derivative works stay under the GPL when distributed;
results you compute with it are your own. For other terms, ask: <cazeaux@vt.edu>.
