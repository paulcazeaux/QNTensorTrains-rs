# Notebooks

Worked examples live in [`notebooks/`](https://github.com/paulcazeaux/QNTensorTrains-rs/tree/master/notebooks).

```sh
julia --project=notebooks -e 'using Pkg; Pkg.develop(path="julia"); Pkg.instantiate()'
julia --project=notebooks -e 'using IJulia; notebook(dir="notebooks")'
```

| | |
|---|---|
| **01 — Getting started** | Installation, a first ground state on H₄, the conventions, and what a tensor train looks like from the inside. |
| **02 — Ground states** | H₄, LiH, H₂O and N₂ with ALS and MALS: variational convergence in the rank, LiH's exact FCI energy reproduced, a real bond-dimension study on H₂O, and — on N₂ — a *silent* convergence failure (a determinant start is a fixed point of two-site DMRG) that costs a factor 2.5 in recovered correlation energy. |
| **03 — Linear response** | Excitation energies as symplectic eigenvalues of the Riemannian Hessian, checked against exact FCI gaps on H₄, plus the interaction-scaling family `H(η) = h + η·v`. |
| **04 — CASSCF vs DMRG (abelian)** | The full stack on three published systems — water CAS(8,6), formaldehyde CAS(4,3), ethylene CAS(2,2); singlets, 6-31G, Hartree-Fock canonical orbitals, the molecules of [arXiv:2604.13753](https://arxiv.org/abs/2604.13753). Per system: a CASSCF-equivalent joint TT ⊕ orbital ground state on the pinned CAS grid with joint RPA at that point; two-site MALS to the *full-space* ground state, on the untyped and on the point-group-typed space; a Riemannian CG polish of both; signed RPA at each polished state, tabulated against the paper with `f_W` beside every mode; and GAD from each. No parameter search and no tuning to the reference values. In the point-group-typed space, use the perturbative noise: random noise settles well above the untyped run. |
| **05 — CASSCF vs DMRG (SU(2))** | The spin-adapted twin of 04: same systems, same pipeline, sharing `notebooks/cache/`, so the two layers can be read side by side. Ranks here are *reduced multiplet* ranks and are not comparable to an abelian pooled `M`. |

The notebook sources are generated from
[`notebooks/mknotebooks.py`](https://github.com/paulcazeaux/QNTensorTrains-rs/blob/master/notebooks/mknotebooks.py)
so that reviewing a change means reading code rather than diffing JSON:

```sh
python3 notebooks/mknotebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

Notebooks 01–03 execute in minutes. **04 and 05 are full-molecule runs** that take hours; execute
them with no per-cell timeout:

```sh
cargo build --release -p qntt-capi
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  notebooks/04-casscf-vs-dmrg-abelian.ipynb notebooks/05-casscf-vs-dmrg-su2.ipynb
```
