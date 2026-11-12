# Notebooks

Worked examples driving the Rust library from Julia.

| | |
|---|---|
| [`01-getting-started.ipynb`](01-getting-started.ipynb) | Installation, a first ground state on H₄, the conventions, and what a tensor train looks like from the inside. |
| [`02-ground-states.ipynb`](02-ground-states.ipynb) | H₄, LiH, H₂O and N₂ with ALS and MALS: variational convergence in the rank, a reference FCI energy reproduced exactly on LiH, a real bond-dimension study on H₂O, and — on N₂ — a silent convergence failure worth knowing about. Ends with the **production pipeline** (MALS ramp + noise → adaptive sweeps → Riemannian polish → boundary trim), every knob a variable, which builds and caches the water 6-31G ground state notebook 03 consumes. |
| [`03-linear-response.ipynb`](03-linear-response.ipynb) | Excitation energies as symplectic eigenvalues of the Riemannian Hessian, checked against exact FCI gaps on H₄. Part II runs the **stability gate** on water 6-31G: `pd_probe` on both Hessian blocks at a stated budget → `complex_seed` declining (no negative direction to follow) → RPA frequencies at the real state against full-space FCI. |
| [`04-casscf-vs-dmrg-abelian.ipynb`](04-casscf-vs-dmrg-abelian.ipynb) | The three systems of arXiv:2604.13753 (water CAS(8,6), formaldehyde CAS(4,3), ethylene CAS(2,2), 6-31G) on the abelian layer: CASSCF-equivalent (joint solver on the pinned CAS grid + joint RPA), DMRG at a fixed truncation tolerance and pooled cap with and without the point group, Riemannian polish, signed TT-RPA at each polished state with `f_W` per mode in one table against the paper, GAD from each polished state. No tuning, no orbital rotation beyond the CASSCF. Cluster-sized. |
| [`05-casscf-vs-dmrg-su2.ipynb`](05-casscf-vs-dmrg-su2.ipynb) | The same pipeline on the SU(2) spin-adapted layer (singlet-only spectrum). Same generator as 04 (`paper_cells` in `mknotebooks.py`). |

## Running them

```sh
cargo build --release -p qntt-capi                       # the library the notebooks call
julia --project=notebooks -e 'using Pkg; Pkg.develop(path="julia"); Pkg.instantiate()'
julia --project=notebooks -e 'using IJulia; notebook(dir="notebooks")'
```

Each notebook activates `notebooks/Project.toml` in its first cell, so any Julia kernel will do.

### Runtimes and order

01 runs in seconds. 02 takes a few minutes and caches its water ground state under
`notebooks/cache/` (gitignored, built on first execution); a warm rerun is about twice as fast.
Run 02 before 03: 03 loads the ground state 02 caches and errors if it is absent.

04 and 05 are independent of 02/03 (and of each other) but share `notebooks/cache/`. They are
**cluster-sized**: DMRG at 22 and 26 orbitals under a cap of 256, then polish / RPA / GAD at the
resulting rank, on three systems and two layers. Build the library in release mode and execute them
with `jupyter nbconvert --to notebook --execute --inplace` on a large machine; for a laptop-sized
pass set `RUN = ["h2o_631g"]` in the second cell.

Delete `notebooks/cache/` after changing any solver knob — the cache keys on the physical setup
(fixture, `2S`, `M`), not on the schedule, so it will not notice.

## Editing them

The notebooks are generated from [`mknotebooks.py`](mknotebooks.py), so a change is reviewed as
code rather than as a JSON diff:

```sh
python3 notebooks/mknotebooks.py                                        # regenerate
cd notebooks && jupyter nbconvert --to notebook --execute --inplace *.ipynb   # re-run
```

Cell ids are content-derived, so regenerating an unchanged cell leaves it untouched.
