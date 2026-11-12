# Examples

`tutorial/` holds the numbered course: each step is a runnable Rust program that builds on the
ones before it.

```sh
cargo run --release --example step-01
cargo run --release --example step-01 -- --help    # its arguments and defaults
```

Read in order. Each step names what it builds on.

| step | teaches | builds on |
|---|---|---|
| `step-01` a first ground state | space → Hamiltonian → one-site ALS → FCI, on LiH at full rank | — |
| `step-02` ranks and convergence | the rank ladder, pooled `M` vs per-sector rank, ALS sweep by sweep | 01 |
| `step-03` reading an FCIDUMP | real integrals off disk, and the conventions | 01 |
| `step-04` Hamiltonian construction | the construction options, and what the operator metrics mean | 03 |
| `step-05` MALS and noise | two-site MALS, the determinant trap, the noise flavours | 02, 03 |
| `step-06` scaling up | N₂/cc-pVDZ on the real integrals | 05 |
| `step-07` spin adaptation | SU(2), and why reduced multiplet ranks are not a pooled `M` | 06 |
| `step-08` linear response | RPA as symplectic eigenvalues of the Riemannian Hessian | 02, 05 |
| `step-09` spin-adapted response | the singlet-sector twin of step-08 | 07, 08 |
| `step-10` cone response | decoupling response-space size from ground-state rank | 08 |
| `step-11` channel response | per-channel composition | 10 |
| `step-12` the joint manifold | TT ⊕ orbital, joint against cone at one reference | 10 |
| `step-13` CASSCF response | the pinned CAS manifold, the `ω(m_active)` ladder | 12 |
| `step-14` spin-adapted CASSCF | transition moments, oscillator strengths | 09, 13 |
| `step-15` DMRG-SCF → RPA | the whole stack end to end | 13 |
| `step-16` PINVIT | another ground-state solver: preconditioned inverse iteration | 02 |
| `step-17` excited states | the ladder: state average → state specific → penalized polish, against RPA ω and the FCI gaps | 05, 07, 08 |
| `step-18` cross-sector response | linear response INTO another irrep from one typed reference | 08, 11 |

The same course is published, with each step's printed output, in the
[documentation](../docs/src/tutorial/index.md). The Julia notebooks in [`notebooks/`](../notebooks/)
teach the same physics through `QNTT.jl`.
