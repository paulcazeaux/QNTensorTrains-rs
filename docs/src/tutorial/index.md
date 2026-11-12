# Tutorial

A numbered course, one self-contained program per step, in the shape of the deal.II
tutorial: each page is a step's own header and commentary, split at comment boundaries,
followed by the output of a recorded run. Read in order, or follow the dependency graph.

```sh
cargo run --release --example step-01              # every step builds by default
cargo run --release --example step-01 -- --help    # its arguments, defaults and knobs
```

| step | lesson | builds on |
|---|---|---|
| [step-01](step-01.md) | A first ground state: load a molecule, build its Hamiltonian, run one-site ALS, compare to FCI | — |
| [step-02](step-02.md) | Ranks and convergence: what a per-sector rank cap costs in energy, and how one-site ALS approaches it | [step-01](step-01.md) |
| [step-03](step-03.md) | Reading a FCIDUMP: real integrals off disk, into a fixed-length ALS or MALS sweep | [step-01](step-01.md) |
| [step-04](step-04.md) | Inside the operator: the structure of the assembled Hamiltonian, counted bond by bond | [step-03](step-03.md) |
| [step-05](step-05.md) | Two-site MALS from a determinant start: what each noise flavour recovers of E_corr | [step-02](step-02.md), [step-03](step-03.md) |
| [step-06](step-06.md) | Scaling up: N₂/cc-pVDZ on the real integrals, where every step has a cost | [step-05](step-05.md) |
| [step-07](step-07.md) | Spin adaptation: the same N2/cc-pVDZ molecule as step-06, solved in the SU(2) singlet sector | [step-06](step-06.md) |
| [step-08](step-08.md) | Symplectic (RPA / linear-response) excitation spectrum of a small molecule on the fixed-rank TT manifold, end to end | [step-02](step-02.md), [step-05](step-05.md) |
| [step-09](step-09.md) | Spin-adapted response: singlet excitation energies on the fixed-rank SU(2) TT manifold | [step-07](step-07.md), [step-08](step-08.md) |
| [step-10](step-10.md) | Cone response: rank-increasing directions in the response ansatz at a fixed low-rank reference | [step-08](step-08.md) |
| [step-11](step-11.md) | Channel response: one linear-response solve per symmetry channel, merged into one spectrum | [step-10](step-10.md) |
| [step-12](step-12.md) | Joint (TT ⊕ orbital) response against cone-augmented response, at one shared reference | [step-10](step-10.md) |
| [step-13](step-13.md) | Linear response on a pinned CAS manifold: the CASSCF-RPA anchor, then the `ω(m_active)` ladder | [step-12](step-12.md) |
| [step-14](step-14.md) | Spin-adapted TT-CASSCF linear response: the pinned CAS manifold, the multiplet-rank ladder, and the Ŝ²-resolved labelling of the abelian spectrum | [step-09](step-09.md), [step-13](step-13.md) |
| [step-15](step-15.md) | The whole stack end to end: a DMRG-SCF ground state, then RPA response on the joint manifold | [step-13](step-13.md) |
| [step-16](step-16.md) | A second ground-state solver: preconditioned inverse iteration | [step-02](step-02.md) |
| [step-17](step-17.md) | Excited states as a ladder: state average, state-specific sweeps and a penalized polish, checked against linear response and the exact spectrum | [step-05](step-05.md), [step-07](step-07.md), [step-08](step-08.md) |
| [step-18](step-18.md) | Cross-sector response: linear response INTO another point-group irrep from one typed reference | [step-08](step-08.md), [step-11](step-11.md) |

## Dependency graph

```@raw html
<pre class="mermaid">
graph LR
  s01["step-01"]
  s02["step-02"]
  s03["step-03"]
  s04["step-04"]
  s05["step-05"]
  s06["step-06"]
  s07["step-07"]
  s08["step-08"]
  s09["step-09"]
  s10["step-10"]
  s11["step-11"]
  s12["step-12"]
  s13["step-13"]
  s14["step-14"]
  s15["step-15"]
  s16["step-16"]
  s17["step-17"]
  s18["step-18"]
  s01 --> s02
  s01 --> s03
  s03 --> s04
  s02 --> s05
  s03 --> s05
  s05 --> s06
  s06 --> s07
  s02 --> s08
  s05 --> s08
  s07 --> s09
  s08 --> s09
  s08 --> s10
  s10 --> s11
  s10 --> s12
  s12 --> s13
  s09 --> s14
  s13 --> s14
  s13 --> s15
  s02 --> s16
  s05 --> s17
  s07 --> s17
  s08 --> s17
  s08 --> s18
  s11 --> s18
</pre>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
mermaid.initialize({ startOnLoad: true });
</script>
```

The pages are generated from the sources by `scripts/mktutorial.py`; the gate fails when a
step changes and its page does not.
