# QNTensorTrains-rs

**Quantum-number-preserving block-sparse tensor trains for fermionic many-body systems.**

`qntt` implements matrix-product-state / tensor-train methods for electronic structure and
fermionic lattice problems, in Rust, with a C ABI and a Julia driver package on top. The total
spin-up and spin-down electron counts are held **exactly**, as block-sparse quantum numbers on
every tensor core — not enforced by penalty or recovered by projection, but built into the data
structure.

This site documents the **Julia interface** ([`QNTT.jl`](api.md)) and the [C ABI](capi.md)
underneath it. The Rust internals are documented by `cargo doc --open`.

## What is here

| Layer | Contents |
|---|---|
| Rust core (`src/`) | Block-sparse TT cores, Hamiltonians, solvers, both symmetry layers |
| C ABI (`capi/`) | `libqntt_capi` — exported functions over abelian/SU(2) × real/complex |
| Julia (`julia/`) | `QNTT.jl`, the driver package these pages describe |

## Capabilities

- **Two symmetry layers.** The abelian `(Nup, Ndn)`-conserving stack and a full **SU(2)
  spin-adapted** stack, with the same TT / Hamiltonian / solver surface over irreducible spin
  sectors.
- **Second-quantized Hamiltonians** from FCIDUMP integrals, built exactly with no screening
  cutoff. Optional three-body / transcorrelated terms.
- **Ground states** by one-site ALS and two-site rank-adaptive MALS (DMRG), imaginary-time TDVP,
  rounded Lanczos, and Riemannian optimization on the fixed-rank manifold.
- **Excited states** by state-average and state-specific two-site DMRG with a penalized
  Riemannian polish, gated for convergence, orthogonality and ordering.
- **Linear response.** Symplectic (RPA) eigenvalues of the Riemannian Hessian at a converged
  fixed-rank state, both matrix-free and via a dense driver for small systems.
- **Real and complex** scalars throughout.

## Getting started

```julia
using QNTT

data  = QNTT.read_fcidump("testdata/h4.FCIDUMP")
space = QNTT.QnSpace(data)
H     = QNTT.Hamiltonian(space, data)

x0    = QNTT.tt_randn(space, QNTT.uniform_ranks(space, 8); seed = 1234)
λ, x  = QNTT.mals(H, x0; bond_dims = [0], eps = 1e-8)

E     = λ + QNTT.e0(data)
ω, _  = QNTT.rpa(H, x; k = 5)
```

See [Getting started](getting-started.md) for installation, the [tutorial](tutorial/index.md) for the
numbered course of Rust examples, and the [notebooks](notebooks.md) for worked examples in Julia on
H₄, LiH, H₂O and N₂.

## Conventions, once

Everything is **0-based**: sites `0:d-1`, bonds `0:d`, and sector labels `(a, b)` counting
electrons strictly to the left of a bond. Integral arrays are the exception — `t[i,j]` and
`v[i,j,k,l]` are ordinary 1-based Julia arrays in the chemists' `(ij|kl)` convention.

Two energy accessors, deliberately named apart:

| call | value |
|---|---|
| [`QNTT.expectation`](@ref) | `⟨x, H·x⟩` — **not** normalized (the Rust `RayleighQuotient`) |
| [`QNTT.rayleigh_quotient`](@ref) | `⟨x, H·x⟩ / ⟨x, x⟩` |

Rank caps, deliberately named apart:

| name | notion | Rust type | SU(2) unit |
|---|---|---|---|
| `rmax` | PER-SECTOR — every sector rank `≤ n` | `PerSector` | reduced multiplets |
| `chi` | POOLED — `Σ` over sectors `≤ n` (the conventional DMRG `χ` / `M`) | `Pooled` | reduced multiplets |
| `bond_dims` | the DMRG schedule of POOLED caps, one entry per outer iteration | `Vec<usize>` | reduced multiplets |

`ExcitedOpts::rank` and the MALS `bond_dims` schedule stay plain integers and are pooled.

## References

- Ren, Li, Jiang & Shuai, *J. Chem. Phys.* **153**, 084118 (2020) — the bipartite operator
  construction.
- Grazioli, Hu & Cancès, *Critical point search and linear response theory for computing
  electronic excitation energies of molecular systems. Part I*,
  [arXiv:2506.16420](https://arxiv.org/abs/2506.16420) — the Kähler-manifold framework the
  linear-response driver instantiates on the TT manifold.
- Psenka & Boumal, *Second-order optimization for tensors with fixed tensor-train rank*,
  [arXiv:2011.13395](https://arxiv.org/abs/2011.13395) — the fixed-rank TT Hessian.

## Citing this work

If `qntt` contributes to published work, please cite it — see `CITATION.cff` in the repository
root. A methods paper is in preparation and will become the preferred reference.

## Acknowledgments

This work was supported by the U.S. National Science Foundation under Award No. 2608677
(Division of Mathematical Sciences).

## License

GPL-3.0-or-later — see `LICENSE` in the repository root. Derivative works stay under the GPL when
distributed; results you compute with it are your own. For other terms, ask
[cazeaux@vt.edu](mailto:cazeaux@vt.edu).
