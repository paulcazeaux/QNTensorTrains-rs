#!/usr/bin/env python3
# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

"""Generate the QNTT example notebooks.

The notebooks are kept in this readable source form and rendered to .ipynb, so that reviewing a
change is reading Python/Julia rather than diffing JSON. Regenerate and re-execute with:

    python3 notebooks/mknotebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""

import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).parent


def _cell_id(text):
    """Stable per-cell id: derived from the content, so regenerating an unchanged cell keeps it."""
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def md(text):
    return {
        "cell_type": "markdown",
        "id": _cell_id(text),
        "metadata": {},
        "source": text.strip("\n").splitlines(True),
    }


def code(text):
    return {
        "cell_type": "code",
        "id": _cell_id(text),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


ACTIVATE = code(r"""
# Use the notebook environment (see notebooks/Project.toml) regardless of which kernel started us.
using Pkg
Pkg.activate(@__DIR__)
""")


def notebook(cells, name):
    # The activation cell goes after the title markdown so the notebook still reads top-down.
    cells = [cells[0], ACTIVATE] + cells[1:]

    # Preserve the outputs of code cells whose source is unchanged. Cell ids are content-derived,
    # so an unchanged cell keeps its id and therefore its recorded run — regenerating after a
    # prose edit no longer throws away an expensive execution.
    path = HERE / name
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except json.JSONDecodeError:
            old = {"cells": []}
        previous = {
            c["id"]: c
            for c in old.get("cells", [])
            if c.get("cell_type") == "code" and c.get("id") and c.get("outputs")
        }
        kept = 0
        for c in cells:
            prior = previous.get(c.get("id")) if c["cell_type"] == "code" else None
            if prior is not None:
                c["outputs"] = prior["outputs"]
                c["execution_count"] = prior.get("execution_count")
                kept += 1
        if kept:
            print(f"  ({kept} executed cells preserved in {name})")

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Julia 1.12",
                "language": "julia",
                "name": "julia-1.12",
            },
            "language_info": {
                "file_extension": ".jl",
                "mimetype": "application/julia",
                "name": "julia",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = HERE / name
    # nbformat's own serialization (unescaped UTF-8), so a regeneration after an executed run
    # does not rewrite every non-ASCII character as an escape.
    path.write_text(json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)


# =================================================================================================
# 01 — getting started
# =================================================================================================

FIXTURES = r'''
# Where the integral fixtures live, and a loader for the pyscf-exported `meta/h1/h2` format
# (used by the LiH and N2 fixtures that are not FCIDUMP files).
const ROOT = normpath(joinpath(@__DIR__, ".."))
fixture(parts...) = joinpath(ROOT, "testdata", parts...)

"""
Load a fixture directory: a keyed `meta.txt` plus the FCIDUMP its `dump_file` key names.

`e_ref` is *whatever reference the fixture was generated against*, and the meta says which by the
key it uses — `e_fci` where the determinant count allowed one, `e_ccsd_t` otherwise. It is never a
generic slot whose meaning varies per fixture (two fixtures used to record `e_ref == e_hf`, from a
skipped FCI silently falling back to the SCF energy).

These directories used to ship `h1.dat`/`h2.dat`: a dense positional dump with no write threshold,
which carried entries down to ~1e-22 where an FCIDUMP is written at 1e-15 — the same molecule with
materially different content (66% of `h2.dat` and 17.5% of the assembled operator at N₂/cc-pVDZ).
One format, so the question does not arise.
"""
function load_fixture(dir)
    meta = Dict{String,String}()
    for line in eachline(joinpath(dir, "meta.txt"))
        line = strip(line)
        (isempty(line) || startswith(line, "#")) && continue
        parts = split(line, limit = 2)
        length(parts) == 2 && (meta[parts[1]] = strip(parts[2]))
    end
    getf(k) = parse(Float64, meta[k])
    d      = parse(Int, meta["norb"])
    nelec  = parse(Int, meta["nelec"])
    ms2    = parse(Int, get(meta, "ms2", "0"))
    e_nuc  = getf("e_nuc")
    e_hf   = haskey(meta, "e_hf") ? getf("e_hf") : NaN
    refkey = findfirst(k -> haskey(meta, k), ["e_fci", "e_ccsd_t", "e_ccsd"])
    e_ref  = refkey === nothing ? NaN : getf(["e_fci", "e_ccsd_t", "e_ccsd"][refkey])

    ints = QNTT.read_fcidump(joinpath(dir, meta["dump_file"]))
    t, v = QNTT.onebody(ints), QNTT.twobody(ints)
    @assert size(t) == (d, d) "meta norb=$d disagrees with $(meta["dump_file"])"
    return (; t, v, d, nelec, ms2, e_nuc, e_hf, e_ref)
end
'''

CACHE = r'''
# --- the state cache ---------------------------------------------------------------------------
# The ground-state pipeline is deterministic and expensive (~2 minutes for water 6-31G at M = 64),
# while every downstream experiment repeats it verbatim. So: build once, save, and reuse. Cold on the first execution, warm in milliseconds
# thereafter.
#
# The cache is a local file, NOT an archival format: it records the build that wrote it and refuses
# anything else, and there is no migration path. If `load` ever complains, delete the file and let
# the cold path run again.
#
# It keys on the physical setup only (fixture, 2S, M and a tag). CHANGING A SOLVER KNOB DOES NOT
# INVALIDATE IT — delete `notebooks/cache/` by hand after editing the schedule below, or you will
# be reading a state built with the old one. The space is re-checked on load, so a change that
# alters the space is caught; one that only alters the path to the state is not.
const CACHE_DIR = joinpath(@__DIR__, "cache")
cache_file(tag) = joinpath(CACHE_DIR, tag * ".tt")

"""
    cached(T, L, tag, space, build) -> (x, :warm|:cold, seconds)

Return the cached state `tag`, or build it with `build()` and save it. The loaded state's space
must equal `space`, so a cache left over from a different physical setup is an error rather than a
silent wrong answer.
"""
function cached(::Type{T}, ::Type{L}, tag, space, build) where {T,L}
    path = cache_file(tag)
    if isfile(path)
        t = @elapsed x = QNTT.load(T, L, path)
        QNTT.space(x) == space ||
            error("cache $path holds $(QNTT.space(x)), but this run wants $space; delete it")
        return (x, :warm, t)
    end
    t = @elapsed x = build()
    mkpath(CACHE_DIR)
    QNTT.save(x, path)
    return (x, :cold, t)
end

report_cache(tag, how, t) = how === :warm ?
    @printf("cache HIT  %s — loaded in %.3f s (the pipeline below was skipped)\n", tag, t) :
    @printf("cache MISS %s — built in %.1f s and saved for next time\n", tag, t)
'''

cells01 = [
    md(r"""
# Getting started with QNTT.jl

`QNTT.jl` drives [`qntt`](https://github.com/paulcazeaux/QNTensorTrains-rs) — quantum-number-preserving
block-sparse tensor trains for fermionic many-body systems, written in Rust — from Julia.

This notebook covers installation, a first ground-state calculation, and the handful of
conventions worth knowing before anything else. The other notebooks build on it:

- **02 — Ground states**: H₄, LiH, H₂O and N₂ with ALS and MALS.
- **03 — Linear response**: excitation energies from the Riemannian Hessian.

## Setup

The package calls into a shared library that you build once from the Rust workspace:

```sh
cargo build --release -p qntt-capi
```

Then, from the repository root:

```julia
using Pkg
Pkg.activate("notebooks")
Pkg.develop(path = "julia")
Pkg.instantiate()
```
"""),
    code(r"""
using QNTT, LinearAlgebra, Printf

QNTT.init()          # size the library's thread pool (0 / no argument = let it choose)
@printf("qntt %s, %d threads\n", QNTT.version(), QNTT.nthreads())
"""),
    md(r"""
## Reading integrals

An FCIDUMP file carries the one- and two-electron integrals plus a header giving the orbital
count, electron count and `2·Sz`. The core (nuclear-repulsion) energy comes along with it: the
solvers all return the **electronic** energy, so a total energy is always `λ + e0`.
"""),
    code(r"""
data = QNTT.read_fcidump(joinpath(@__DIR__, "..", "testdata", "h4.FCIDUMP"))

@printf("norb   = %d\n", QNTT.norb(data))
@printf("nelec  = %d\n", QNTT.nelec(data))
@printf("2Sz    = %d\n", QNTT.ms2(data))
@printf("E_core = %.10f Ha\n", QNTT.e0(data))
"""),
    md(r"""
## The symmetry descriptor

A calculation lives in a *space*: the orbital count together with the conserved quantum numbers.
`QnSpace(data)` reads the target sector straight off the FCIDUMP header.

The abelian layer conserves the spin-up and spin-down electron counts separately, exactly — they
index the block structure of every tensor core, so they cannot drift the way a penalty term or a
projection would allow.
"""),
    code(r"""
space = QNTT.QnSpace(data)
"""),
    md(r"""
Bond dimensions are not a single number here: **every sector at every bond carries its own rank**.
`maxranks` gives the largest admissible rank at each — beyond that the manifold is the full space
and the tensor train is exact.

`uniform_ranks(space, r)` asks for rank `r` everywhere, capped where the symmetry allows less.
That is the usual way to pick a starting manifold.
"""),
    code(r"""
r = QNTT.uniform_ranks(space, 4)

@printf("bond   maximal ranks (per sector)   requested\n")
for b in 0:space.d
    @printf("  %d    %-28s %s\n", b, string(QNTT.maxranks(space)[b+1]), string(r[b+1]))
end
"""),
    md(r"""
## Compiling the Hamiltonian

The second-quantized operator is built exactly: no screening threshold decides the answer.

This is a one-time cost that grows with the orbital count. At scale, time it before assuming a
run is slow for numerical reasons.
"""),
    code(r"""
t0 = time()
H = QNTT.Hamiltonian(space, data)
@printf("built in %.3f s\n", time() - t0)
@printf("automaton bond dimensions: %s\n", string(QNTT.automaton_bond_dims(H)))
H
"""),
    md(r"""
## A ground state

`mals` is two-site DMRG: it sweeps, solving a local eigenproblem on each pair of neighbouring
sites and re-splitting by SVD, so the ranks adapt as it goes. `bond_dims = [0]` means "no cap" —
every bond keeps whatever rank the `eps`-truncated SVD gives it.
"""),
    code(r"""
x0 = QNTT.tt_randn(space, QNTT.uniform_ranks(space, 8); seed = 1234)

λ, x, _ = QNTT.mals(H, x0; bond_dims = [0], eps = 1e-10, max_iter = 30)

@printf("electronic energy  λ = %.10f Ha\n", λ)
@printf("total energy   λ + E_core = %.10f Ha\n", λ + QNTT.e0(data))
@printf("max bond dimension = %d\n", QNTT.max_bond_rank(x))
@printf("residual ‖Hx − λx‖/‖x‖ = %.2e\n", norm(H * x - λ * x) / norm(x))
"""),
    md(r"""
## The state is an ordinary Julia value

`x` supports the algebra you would expect. Operations that produce a train return a new one; the
solvers never mutate their input.
"""),
    code(r"""
@printf("norm(x)          = %.10f\n", norm(x))
@printf("dot(x, x)        = %.10f\n", dot(x, x))
@printf("rayleigh(H, x)   = %.10f\n", QNTT.rayleigh_quotient(H, x))
@printf("expectation(H,x) = %.10f   (⟨x,Hx⟩, NOT divided by ⟨x,x⟩)\n", QNTT.expectation(H, x))

y = 2x - x                       # exact TT algebra, unrounded
@printf("‖2x − x − x‖     = %.2e\n", norm(y - x))

z = round(y; eps = 1e-12)        # deterministic SVD rounding
@printf("ranks after rounding: max bond %d\n", QNTT.max_bond_rank(z))
"""),
    md(r"""
### Two energy accessors, deliberately named apart

| call | value |
|---|---|
| `QNTT.expectation(H, x)` | `⟨x, H·x⟩` — **not** normalized |
| `QNTT.rayleigh_quotient(H, x)` | `⟨x, H·x⟩ / ⟨x, x⟩` |

The unnormalized one is what the Rust library calls `RayleighQuotient`. They agree above only
because `mals` returns a normalized state.

## Looking inside a train

Cores are stored block-sparsely: for each site, one dense block per (family, sector) that the
quantum numbers admit. You can read and write those blocks directly.
"""),
    code(r"""
@printf("site 1 has %d non-empty blocks\n", length(QNTT.blocks(x, 1)))

for ((fam, sec), m) in sort(collect(QNTT.blocks(x, 1)); by = kv -> (string(kv[1][1]), kv[1][2]))
    @printf("  family %-6s sector %-6s  %d×%d\n", fam, sec, size(m, 1), size(m, 2))
end
"""),
    md(r"""
On a small system the train can also be materialized densely, in the `4^d` Jordan–Wigner basis —
useful as an oracle, useless beyond a dozen orbitals (the call refuses to allocate rather than
exhausting memory).
"""),
    code(r"""
xd = Array(x)
@printf("dense length 4^%d = %d, norm %.10f (vs %.10f)\n",
        space.d, length(xd), norm(xd), norm(x))
"""),
    md(r"""
## Conventions, once

- **Everything is 0-based**: sites `0:d-1`, bonds `0:d`, sector labels `(a, b)` counting electrons
  strictly to the *left* of a bond.
- Integral arrays are the exception: `QNTT.onebody(data)` and `QNTT.twobody(data)` are ordinary
  1-based Julia arrays, `t[i,j]` and `v[i,j,k,l]` in the chemists' `(ij|kl)` convention.
- Solvers return **electronic** energies; add `QNTT.e0(data)`.
- Failed calls throw `QNTTError` carrying the Rust library's own message.
"""),
    code(r"""
try
    QNTT.mals(H, x0; bond_dims = Int[])
catch e
    println(e)
end
"""),
]

notebook(cells01, "01-getting-started.ipynb")


# =================================================================================================
# 02 — ground states
# =================================================================================================

cells02 = [
    md(r"""
# Ground states: H₄, LiH, H₂O, N₂

Ground-state energies with the two DMRG drivers — one-site **ALS** on a fixed manifold, and
two-site **MALS**, which adapts the ranks as it sweeps — across four systems spanning three orders
of magnitude in Hilbert-space size.
"""),
    code(r"""
using QNTT, LinearAlgebra, Printf, Plots
gr()
QNTT.init()
@printf("qntt %s, %d threads\n", QNTT.version(), QNTT.nthreads())
""" + FIXTURES + CACHE),
    md(r"""
## H₄ — the whole Hilbert space

Four orbitals, four electrons: the `(Nup, Ndn) = (2, 2)` sector holds `C(4,2)² = 36` determinants.
At full rank the tensor train is *exact*, so DMRG here returns the FCI energy. Notebook 03 returns
to this system.
"""),
    code(r"""
h4 = QNTT.read_fcidump(fixture("h4.FCIDUMP"))
space4 = QNTT.QnSpace(h4)
H4 = QNTT.Hamiltonian(space4, h4)

@printf("%s, E_core = %.10f\n", string(space4), QNTT.e0(h4))
@printf("maximal bond dimensions per bond: %s\n",
        string([sum(g) for g in QNTT.maxranks(space4)]))
"""),
    md(r"""
**ALS** optimizes on a *fixed* manifold — the rank grid of the starting state, which it never
leaves. Its knob is the **per-sector** rank: below full rank it returns the best state of that rank
(variational, above FCI); on the full manifold, FCI.
"""),
    code(r"""
# ALS on manifolds of increasing per-sector rank. `total M` is the bond dimension summed over
# symmetry sectors — the number MALS caps below, and NOT the same number as the per-sector rank.
@printf("%-20s %8s %18s\n", "ALS per-sector rank", "total M", "λ")
for r in (1, 2, 4)
    x0 = QNTT.tt_randn(space4, QNTT.uniform_ranks(space4, r); seed = 7)
    λ, x = QNTT.als(H4, x0; eps = 1e-12, max_iter = 200)
    @printf("%-20d %8d %18.12f\n", r, QNTT.max_bond_rank(x), λ)
end
"""),
    md(r"""
**MALS** grows the ranks itself as it sweeps. Its knob is the **total** bond dimension `M` summed
over all sectors — the conventional DMRG *M*, a different number from ALS's per-sector rank. Sweep
`M` up to the full manifold (here `M = 16`) and the energy falls monotonically to FCI.
"""),
    code(r"""
Ms4 = [1, 2, 4, 8, 12, 16]
x0 = QNTT.tt_randn(space4, QNTT.uniform_ranks(space4, 1); seed = 7)
rows4 = map(Ms4) do M
    λ, x, _ = QNTT.mals(H4, x0; bond_dims = [M], eps = 1e-12, max_iter = 100)
    (; M, λ, bond = QNTT.max_bond_rank(x))
end

λ_exact = minimum(row.λ for row in rows4)
@printf("%-8s %18s %8s\n", "MALS M", "λ", "bond")
for row in rows4
    @printf("%-8d %18.12f %8d\n", row.M, row.λ, row.bond)
end
@printf("\nfull-rank (= FCI) electronic energy: %.12f Ha\n", λ_exact)
@printf("total energy: %.12f Ha\n", λ_exact + QNTT.e0(h4))
"""),
    code(r"""
err = [max(row.λ - λ_exact, 1e-16) for row in rows4]
plot(Ms4, err;
     yscale = :log10, marker = :circle, lw = 2, legend = false,
     xlabel = "total bond dimension M (summed over sectors)",
     ylabel = "λ − λ_exact  (Ha)",
     title = "H₄: MALS convergence in the bond dimension M")
"""),
    md(r"""
Both reach the exact energy once the manifold is large enough. Keep the two "rank" axes apart: ALS
reaches the full manifold at per-sector rank 4 (total `M = 16`), MALS at `M = 16`, and below full
rank ALS optimizes on the grid you hand it while MALS redistributes rank between sectors.

## LiH — a reference FCI energy to hit

Six orbitals, four electrons, STO-3G. pyscf's exact FCI energy ships with the fixture, so the
answer can be checked rather than merely watched to converge.
"""),
    code(r"""
lih = load_fixture(fixture("lih_sto3g"))
space_li = QNTT.QnSpace(lih.d, lih.nelec, lih.ms2, Val(:from_sz))
H_li = QNTT.Hamiltonian(space_li, lih.t, lih.v)

@printf("%s\n", string(space_li))
@printf("E_nuc = %.10f   E_HF = %.10f   E_FCI = %.10f  (exact, pyscf)\n",
        lih.e_nuc, lih.e_hf, lih.e_ref)
"""),
    code(r"""
x0 = QNTT.tt_randn(space_li, QNTT.uniform_ranks(space_li, 8); seed = 11)
λ, x_li, _ = QNTT.mals(H_li, x0; bond_dims = [0], eps = 1e-12, max_iter = 60)

E_total = λ + lih.e_nuc
@printf("MALS electronic λ   = %.12f Ha\n", λ)
@printf("total E = λ + E_nuc = %.12f Ha\n", E_total)
@printf("exact E_FCI         = %.12f Ha\n", lih.e_ref)
@printf("difference          = %.2e Ha\n", E_total - lih.e_ref)
@printf("correlation energy recovered: %.10f Ha\n", lih.e_hf - E_total)
"""),
    md(r"""
## H₂O — a real DMRG-scale problem

24 orbitals, 10 electrons; the answer now genuinely depends on the bond dimension. Two changes
matter at this size:

1. **Start from the Hartree–Fock determinant**, not from noise — a rank-1 train of the right order
   of magnitude that MALS grows rank around.
2. **Ramp the rank schedule.** `bond_dims` is read one entry per outer iteration, so `[8, 16, 32,
   …]` does the cheap sweeps first.
"""),
    code(r"""
h2o = QNTT.read_fcidump(fixture("h2o.FCIDUMP"))
space_w = QNTT.QnSpace(h2o)
t0 = time()
H_w = QNTT.Hamiltonian(space_w, h2o)
@printf("%s\n", string(space_w))
@printf("operator built in %.1f s; largest automaton bond dimension %d\n",
        time() - t0, maximum(QNTT.automaton_bond_dims(H_w)))

# The Hartree–Fock determinant: fill the lowest `Nup` orbitals of each spin.
hf_w = QNTT.tt_state([k < space_w.a for k in 0:space_w.d-1],
                     [k < space_w.b for k in 0:space_w.d-1])
E_hf_w = QNTT.rayleigh_quotient(H_w, hf_w) + QNTT.e0(h2o)
@printf("Hartree–Fock determinant: E = %.8f Ha\n", E_hf_w)
"""),
    code(r"""
ramp(r) = [min(r, 2^k) for k in 3:12]      # 8, 16, 32, … capped at r

rows_w = map([8, 16, 32, 64]) do r
    t = @elapsed λ, x, _ = QNTT.mals(H_w, hf_w; bond_dims = ramp(r), eps = 1e-9, max_iter = 14)
    E = λ + QNTT.e0(h2o)
    @printf("  ramp→%-4d  E = %.8f Ha   E_corr = %.6f   %5.1f s   bond = %d\n",
            r, E, E - E_hf_w, t, QNTT.max_bond_rank(x))
    (; r, E, bond = QNTT.max_bond_rank(x), t)
end
"""),
    code(r"""
E_best_w = minimum(row.E for row in rows_w)
plot([row.bond for row in rows_w], max.([row.E - E_best_w for row in rows_w], 1e-12);
     yscale = :log10, marker = :circle, lw = 2, legend = false,
     xlabel = "max bond dimension (summed over sectors)",
     ylabel = "E − E(best)  (Ha)",
     title = "H₂O / 24 orbitals: convergence in the bond dimension")
"""),
    md(r"""
The energy stops moving around bond dimension 32–64. Here the manifold never becomes exact, only
*sufficient*, so the curve flattens above the noise floor instead of collapsing as it did on H₄.

Diagnostic: if `max_bond_rank(x)` comes back sitting exactly on the cap, the cap — not the
truncation tolerance — decided the answer.

## N₂ / cc-pVDZ — with a reference to hit, and a trap to fall into

28 orbitals, 14 electrons, canonical RHF orbitals. The reference is **CCSD(T)**, not FCI, so
"correlation energy recovered" is measured against it. `bond_dims` caps the **total** bond
dimension summed over sectors — the conventional DMRG *M*. `M = 64` is small here; production runs
use 10³–10⁴.
"""),
    code(r"""
n2 = load_fixture(fixture("n2_ccpvdz"))
space_n = QNTT.QnSpace(n2.d, n2.nelec, n2.ms2, Val(:from_sz))

t0 = time()
H_n = QNTT.Hamiltonian(space_n, n2.t, n2.v)
@printf("%s\n", string(space_n))
@printf("operator built in %.1f s; largest automaton bond dimension %d\n",
        time() - t0, maximum(QNTT.automaton_bond_dims(H_n)))

hf_n = QNTT.tt_state([k < space_n.a for k in 0:space_n.d-1],
                     [k < space_n.b for k in 0:space_n.d-1])
E_hf_tt = QNTT.rayleigh_quotient(H_n, hf_n) + n2.e_nuc

@printf("\nHartree–Fock determinant : %.8f Ha\n", E_hf_tt)
@printf("fixture reference E_HF   : %.8f Ha   (difference %.1e)\n",
        n2.e_hf, E_hf_tt - n2.e_hf)
@printf("CCSD(T) reference        : %.8f Ha\n", n2.e_ref)
@printf("correlation energy to recover: %.6f Ha\n", n2.e_ref - n2.e_hf)
@printf("chemical accuracy (1 kcal/mol) = %.6f Ha\n", 1 / 627.509474)
"""),
    md(r"""
The Hartree–Fock determinant, evaluated as a rank-1 tensor train, reproduces the fixture's
reference HF energy to the last digit — a good check that the integrals, the orbital ordering and
the operator all line up before anything harder is attempted.

Now sweep the bond dimension, starting from that determinant.
"""),
    code(r"""
const KCAL = 627.509474                        # Hartree -> kcal/mol
recovered(E) = 100 * (n2.e_hf - E) / (n2.e_hf - n2.e_ref)
kcal(E) = KCAL * (E - n2.e_ref)                # distance from the reference, kcal/mol

rows_n = map([16, 32, 64, 128]) do r
    t = @elapsed λ, x = QNTT.mals(H_n, hf_n; bond_dims = ramp(r), eps = 1e-9, max_iter = 16)
    E = λ + n2.e_nuc
    @printf("  M→%-4d  E = %.8f   recovered %5.1f%%   %+7.1f kcal/mol   %5.1f s   M = %d\n",
            r, E, recovered(E), kcal(E), t, QNTT.max_bond_rank(x))
    (; r, E, bond = QNTT.max_bond_rank(x), t)
end
"""),
    md(r"""
The energy stops changing past bond dimension 32 and yet only about a third of the correlation
energy has been recovered: **converged in the bond dimension and nowhere near the minimum**.
Two-site DMRG started from a single determinant is stuck — the determinant already solves every
local eigenproblem the sweep poses — and nothing in the rank study reveals it.

The fix is a **noise schedule**: perturb the two-site block before each truncation, so the kept
basis can include a sector carrying no weight in the current state. `noise` is a per-iteration
amplitude alongside `bond_dims` and **must end at zero** — only the final noise-free sweeps carry a
variational energy.
"""),
    code(r"""
# Each operator channel is unit-trace normalized and alpha is globally rescaled by ||rho_noise||_F,
# so a single nominal amplitude is portable across bonds and molecules: ~1e-4 works well.
# The noise is very sensitive to it — 1e-2 is far too aggressive and costs
# sweeps to recover from (measured on this system).
noise_schedule = [1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 0.0, 0.0, 0.0, 0.0, 0.0]

λ, x, hist = QNTT.mals(H_n, hf_n; bond_dims = ramp(64), noise = noise_schedule,
                       eps = 1e-9, max_iter = 20, seed = 3)
E = λ + n2.e_nuc
@printf("from the SAME determinant, with noise: E = %.8f   recovered %5.2f%%   %+6.2f kcal/mol\n",
        E, recovered(E), kcal(E))
"""),
    code(r"""
# The history is what tells you whether a ramp converged — and it is also where you can watch the
# noise do its work.
@printf("%-4s %-8s %-8s %16s %14s\n", "it", "M cap", "noise", "energy", "discarded")
for (i, sw) in enumerate(hist)
    @printf("%-4d %-8d %-8.0e %16.8f %14.2e\n",
            i, sw.bond_dim, sw.noise, sw.energy + n2.e_nuc, sw.discarded_weight)
end
"""),
    md(r"""
From ~35% to ~88% of the correlation energy, same bond dimension, same starting determinant.

The energies in the noisy rows are *not* variational and may sit below the final answer; only the
tail, once `noise` reaches zero, is an upper bound — which is why the driver refuses to stop on a
noisy iteration.

## Orbital ordering: the Fiedler vector

A tensor train is one-dimensional: cheap when strongly coupled orbitals are neighbours along the
chain, expensive when they are far apart. Weight each pair by the exchange integral
`K[i,j] = |(ij|ji)|` and sort by the **Fiedler vector** of the graph Laplacian (Barcza, Legeza,
Marti & Reiher, *Phys. Rev. A* **83**, 012508 (2011)).
"""),
    code(r"""
# Fiedler ordering from the exchange-integral graph.
function fiedler_order(v)
    d = size(v, 1)
    K = [i == j ? 0.0 : abs(v[i, j, j, i]) for i in 1:d, j in 1:d]
    L = Diagonal(vec(sum(K, dims = 2))) - K            # graph Laplacian
    E = eigen(Symmetric(Matrix(L)))
    # Do not assume the eigenvalues come back sorted: take the second-smallest explicitly.
    return sortperm(E.vectors[:, sortperm(E.values)[2]])
end

perm = fiedler_order(n2.v)
@printf("Fiedler order: %s\n", string(perm))
"""),
    md(r"""
Reordering means permuting the integrals — `t[p,p]` and `v[p,p,p,p]` — and rebuilding the operator.
The Hartree–Fock determinant follows the permutation: the same *orbitals* are occupied, at different
*positions*, so its energy is unchanged.
"""),
    code(r"""
t_f = n2.t[perm, perm]
v_f = n2.v[perm, perm, perm, perm]
H_f = QNTT.Hamiltonian(space_n, t_f, v_f)

occ_f = [perm[k] <= space_n.a for k in 1:space_n.d]
hf_f = QNTT.tt_state(occ_f, occ_f)
@printf("HF energy, reordered vs canonical: %.8f vs %.8f\n",
        QNTT.rayleigh_quotient(H_f, hf_f) + n2.e_nuc, E_hf_tt)
"""),
    md(r"""
Now run both orderings from their Hartree–Fock determinant under the same noise schedule. These
`M` are small — production DMRG on this system runs `M` in the thousands — so read the numbers as a
demonstration of the machinery, not a converged verdict on the ordering.
"""),
    code(r"""
function sweep_M(H, x0, Ms)
    map(Ms) do M
        t = @elapsed λ, x, _ = QNTT.mals(H, x0; bond_dims = ramp(M), noise = noise_schedule,
                                         eps = 1e-10, max_iter = 20, seed = 3)
        E = λ + n2.e_nuc
        @printf("    M = %-5d E = %.8f   recovered %6.2f%%   %+7.2f kcal/mol   %6.1f s\n",
                M, E, recovered(E), kcal(E), t)
        (; M, E)
    end
end

Ms = [32, 64]
println("canonical order:"); can_rows  = sweep_M(H_n, hf_n, Ms)
println("Fiedler order:");   fied_rows = sweep_M(H_f, hf_f, Ms)
"""),
    code(r"""
p = plot(; xscale = :log2, yscale = :log10, xlabel = "bond dimension M",
         ylabel = "E − E_CCSD(T)  (Ha)", legend = :bottomleft,
         title = "N₂ / cc-pVDZ: approach to the reference")
floored(rows) = [max(r.E - n2.e_ref, 1e-12) for r in rows]
plot!(p, [r.M for r in can_rows], floored(can_rows); marker = :circle, lw = 2, label = "canonical")
plot!(p, [r.M for r in fied_rows], floored(fied_rows); marker = :square, lw = 2, label = "Fiedler")
hline!(p, [1 / KCAL]; ls = :dash, lc = :black, label = "chemical accuracy (1 kcal/mol)")
"""),
    md(r"""
## The SU(2) spin-adapted layer

The SU(2) layer conserves the **total spin** instead of the two electron counts separately: states
are built from irreducible multiplets, so a singlet is a singlet by construction.

- The space is `Su2Space(d, N, 2S)` — electrons and total spin — and the energy must agree with the
  abelian one for the same physical state.
- A multiplet is stored once instead of `2S+1` times, so the same accuracy needs a **smaller bond
  dimension**. That is the payoff.

The API is otherwise identical, including the schedule.
"""),
    code(r"""
lih_su2 = QNTT.Su2Space(lih.d, lih.nelec, abs(lih.ms2))     # N electrons, total spin 2S = 0
H_su2 = QNTT.Hamiltonian(lih_su2, lih.t, lih.v)
@printf("%s   vs abelian %s\n", string(lih_su2), string(space_li))

# The SU(2) analogue of a determinant is a configuration state function. Its per-site codes are
# genealogical COUPLING BRANCHES, not electron spins: :up raises the total spin, :dn lowers it, and
# both add exactly one electron.
csf = QNTT.tt_csf([:occ, :occ, :empty, :empty, :empty, :empty])
@printf("CSF norm %.10f, E = %.10f\n", norm(csf), QNTT.rayleigh_quotient(H_su2, csf) + lih.e_nuc)
"""),
    md(r"""
A CSF is a fixed point of the spin-adapted sweep for exactly the reason a determinant is one on the
abelian layer — so the same noise schedule is the remedy.
"""),
    code(r"""
λ_quiet, _, _ = QNTT.mals(H_su2, csf; bond_dims = [0], eps = 1e-12, max_iter = 40)
# The SU(2) driver takes either flavour (perturbative/White noise is the crate-wide default); this
# cell pins noise_type = :random, the path the printed energy below was recorded against.
λ_su2, x_su2, hist_su2 = QNTT.mals(H_su2, csf; bond_dims = [0],
                                   noise = [1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 0.0, 0.0],
                                   noise_type = :random, eps = 1e-12, max_iter = 40, seed = 3)

@printf("SU(2) from a CSF, noise-free : %.10f\n", λ_quiet + lih.e_nuc)
@printf("SU(2) from a CSF, with noise : %.10f\n", λ_su2 + lih.e_nuc)
@printf("exact FCI (pyscf)            : %.10f\n", lih.e_ref)
@printf("abelian answer               : %.10f\n", E_total)
"""),
    code(r"""
@printf("%-10s %18s %14s %14s\n", "layer", "energy", "max bond dim", "max sector rank")
@printf("%-10s %18.10f %14d %14d\n", "abelian", E_total,
        QNTT.max_bond_rank(x_li), QNTT.max_sector_rank(x_li))
@printf("%-10s %18.10f %14d %14d\n", "SU(2)", λ_su2 + lih.e_nuc,
        QNTT.max_bond_rank(x_su2), QNTT.max_sector_rank(x_su2))
@printf("\nenergies agree to %.1e Ha\n", abs((λ_su2 + lih.e_nuc) - E_total))
"""),
    md(r"""
Same energy on a smaller manifold; that ratio grows with the spin multiplicity involved. Everything
else transfers unchanged — `Su2Space(data)` reads an FCIDUMP header, and the TDVP, Riemannian
and `rpa` drivers all have SU(2) arms with the same keyword surface.
"""),
    md(r"""
## The production pipeline, with every knob in view

A real ground state — the kind notebook 03 does linear response *on* — is a **pipeline**:

1. **MALS stage A**, a fixed-`M` ramp *with noise* — noise escapes the determinant/CSF trap, and is
   only safe under a fixed pooled cap (uncapped ranks plus noise is the rank-explosion pathology).
2. **MALS stage B**, noise-free and rank-*adaptive*: a fixed grid cannot move rank between sectors.
3. **A Riemannian CG polish** to genuine criticality. Sweeps converge the *energy*; linear response
   needs `‖grad‖` small, which is stricter and costs the most.
4. **A boundary trim**, then re-polish: near-zero Schmidt directions cost ~`σ²` in energy but put
   ~`1/σ_min` into the Weingarten curvature and stall every downstream solver.

The parameters below are the whole pipeline, ordinary variables. Change one, delete
`notebooks/cache/`, and re-run — the cache is gitignored, rebuilt on demand, and each cached cell
prints whether it hit or missed.
"""),
    code(r"""
# ---- physical setup ---------------------------------------------------------------------------
# Switch FIXTURE to "lih_sto3g" for a seconds-long smoke run of this entire section. "h2o_631g"
# (13 orbitals, 10 electrons) is the DEMONSTRATED case: notebook 03's real-system section runs its
# stability gate and linear response on exactly this state, and it is why the cache exists at all.
const FIXTURE = "h2o_631g"
const TWO_S   = 0            # total spin 2S — 0 is a singlet. This is 2S, NOT the FCIDUMP's MS2.
const M       = 64           # pooled bond dimension (the conventional DMRG M, summed over sectors)

# ---- MALS stage A: fixed-M ramp WITH noise ------------------------------------------------------
const MALS_A_BOND_DIMS  = [max(M ÷ 4, 6), max(M ÷ 2, 10), max(3M ÷ 4, 14), M, M, M]
const MALS_A_NOISE      = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0]      # MUST end at 0.0
const MALS_A_NOISE_TYPE = :random
const MALS_A_EPS        = 1e-10
const MALS_A_MAX_ITER   = 30
const MALS_SEED         = 0x9E3779B9

# ---- MALS stage B: noise-free, rank-adaptive ----------------------------------------------------
# NOTE on the smoke fixture: on a system small enough that stage A already reaches FCI, this
# truncation tolerance becomes the BINDING constraint and stage B raises the energy again. That is
# the stage doing its job (it trades energy for a better-conditioned rank distribution), not a bug —
# but it is why the smoke run's numbers look worse than stage A's.
const MALS_B_EPS      = 1e-2       # truncation tolerance; the stage's real knob
const MALS_B_BOND_CAP = 4M         # a cap, not a target: uncapped drove water to rank 209
const MALS_B_MAX_ITER = 30

# ---- Riemannian CG polish -----------------------------------------------------------------------
const GRADTOL     = 1e-10   # THE criticality gate — the RPA reference criterion
const CG_MAX_ITER = 600     # inner CG iterations before the driver restarts
const CG_GRADTOL  = 1e-13   # the driver's own inner stop, deliberately below GRADTOL
const CG_RESTARTS = 30      # outer restarts allowed while chasing GRADTOL

# ---- boundary trim ------------------------------------------------------------------------------
const TRIM_EPS    = 1e-6
const TRIM_PASSES = 1       # was 3; passes 2-3 measurably buy nothing (the re-polish re-equilibrates)

@printf("fixture %s, 2S = %d, M = %d\n", FIXTURE, TWO_S, M)
"""),
    code(r"""
fx = load_fixture(fixture(FIXTURE))
space_gs = QNTT.Su2Space(fx.d, fx.nelec, TWO_S)      # the spin-adapted layer, as notebook 03 needs
t0 = time()
H_gs = QNTT.Hamiltonian(space_gs, fx.t, fx.v)
@printf("%s\n", string(space_gs))
@printf("operator built in %.1f s\n", time() - t0)
@printf("E_nuc = %.9f   E_HF = %.9f   E_ref = %.9f\n", fx.e_nuc, fx.e_hf, fx.e_ref)

# The closed-shell CSF: the spin-adapted analogue of the Hartree-Fock determinant, and — exactly
# like it — a FIXED POINT of the noise-free sweep. That is what stage A's noise is for.
csf0 = QNTT.tt_csf([k <= fx.nelec ÷ 2 ? :occ : :empty for k in 1:fx.d])
@printf("CSF start: E = %.9f Ha\n", QNTT.rayleigh_quotient(H_gs, csf0) + fx.e_nuc)
"""),
    md(r"""
The pipeline itself. `polish` re-enters `riemannian` until `grad_norm` is below `GRADTOL`: a single
call stops on its own inner criterion, which is not the gate that matters.
"""),
    code(r"""
"Riemannian CG to criticality: restart until ‖grad‖ < GRADTOL, or give up loudly."
function polish(H, x, λ_scale)
    inner_eps = GRADTOL / max(abs(λ_scale), 1)      # the inner test is RELATIVE to the energy
    _, y = QNTT.riemannian(H, x; cg = true, eps = inner_eps,
                           max_iter = CG_MAX_ITER, gradtol = CG_GRADTOL)
    ncg = 1
    for _ in 1:CG_RESTARTS
        QNTT.grad_norm(H, y) < GRADTOL && break
        _, y = QNTT.riemannian(H, y; cg = true, eps = inner_eps,
                               max_iter = CG_MAX_ITER, gradtol = CG_GRADTOL)
        ncg += 1
    end
    return (y / norm(y), ncg)
end

function ground_state_pipeline(H, x0, e_nuc)
    t_a = @elapsed λa, xa, _ = QNTT.mals(H, x0; bond_dims = MALS_A_BOND_DIMS,
                                         noise = MALS_A_NOISE, noise_type = MALS_A_NOISE_TYPE,
                                         eps = MALS_A_EPS, max_iter = MALS_A_MAX_ITER,
                                         seed = MALS_SEED)
    @printf("  MALS A (M = %d, noisy)      E = %.9f Ha   %5.1f s\n", M, λa + e_nuc, t_a)

    t_b = @elapsed λb, xb, _ = QNTT.mals(H, xa; bond_dims = [MALS_B_BOND_CAP], noise = [0.0],
                                         eps = MALS_B_EPS, max_iter = MALS_B_MAX_ITER)
    @printf("  MALS B (adaptive, eps=%.0e) E = %.9f Ha   %5.1f s\n", MALS_B_EPS, λb + e_nuc, t_b)

    t_p = @elapsed x, ncg = polish(H, xb, λb)
    @printf("  CG polish (%2d calls)       E = %.9f Ha   %5.1f s   ‖grad‖ = %.3e\n",
            ncg, QNTT.rayleigh_quotient(H, x) + e_nuc, t_p, QNTT.grad_norm(H, x))

    for pass in 1:TRIM_PASSES
        t_t = @elapsed begin
            QNTT.round!(x; eps = TRIM_EPS, mode = :rel)
            x = x / norm(x)
            x, nc = polish(H, x, QNTT.rayleigh_quotient(H, x))
        end
        @printf("  trim pass %d (eps=%.0e)      E = %.9f Ha   %5.1f s   σ_min = %.3e\n",
                pass, TRIM_EPS, QNTT.rayleigh_quotient(H, x) + e_nuc, t_t, QNTT.sigma_min(x))
    end
    return x
end

const GS_TAG = @sprintf("%s_2S%d_M%d_gs", FIXTURE, TWO_S, M)
x_gs, how, t_gs = cached(Float64, QNTT.SU2, GS_TAG, space_gs,
                         () -> ground_state_pipeline(H_gs, csf0, fx.e_nuc))
report_cache(GS_TAG, how, t_gs)
"""),
    md(r"""
### What "converged" means here

Five numbers, and the last three are the ones a linear-response calculation actually depends on.
An energy that has stopped moving says nothing about any of them.
"""),
    code(r"""
λ_gs = QNTT.rayleigh_quotient(H_gs, x_gs)
E_gs = λ_gs + fx.e_nuc
sp_gs = QNTT.schmidt(x_gs)

@printf("E                    = %.9f Ha\n", E_gs)
@printf("  vs fixture E_ref   = %.9f Ha   (Δ = %+.2e)\n", fx.e_ref, E_gs - fx.e_ref)
@printf("%% correlation energy = %.2f %%\n", 100 * (fx.e_hf - E_gs) / (fx.e_hf - fx.e_ref))
@printf("pooled bond dims     = %s\n", string([sum(g) for g in QNTT.ranks(x_gs)]))
@printf("max bond / sector    = %d / %d\n", QNTT.max_bond_rank(x_gs), QNTT.max_sector_rank(x_gs))
@printf("σ_min                = %.3e   (the manifold-boundary alarm)\n", QNTT.sigma_min(x_gs))
@printf("‖grad‖               = %.3e   (gate %.0e — %s)\n", QNTT.grad_norm(H_gs, x_gs), GRADTOL,
        QNTT.grad_norm(H_gs, x_gs) < GRADTOL ? "CRITICAL" : "NOT critical")
@printf("ρ = |(x,x)|/⟨x,x⟩    = %.15f   (a real state is 1 by construction)\n", QNTT.rho(x_gs))

# The Schmidt mass, bond by bond. Use the WEIGHTED sum: on the SU(2) layer each reduced multiplet
# carries a degeneracy weight, and the unweighted Σσ² reads ~0.99 and looks like a bug.
println("\nper-bond Schmidt spectrum:")
for (b, (σ, w)) in enumerate(zip(sp_gs.sigma, sp_gs.weight))
    @printf("  bond %2d: %3d values, σ ∈ [%.3e, %.3e], Σ w σ² = %.12f\n",
            b, length(σ), minimum(σ), maximum(σ), sum(w .* σ .^ 2))
end
"""),
    md(r"""
`σ_min` is a **conditioning** number, not an accuracy measure: the Weingarten curvature scales like
`‖r⊥‖/σ_min`, so a state on the manifold boundary has a Hessian that is undefined, not merely
ill-conditioned. When it collapses the answer is a *smaller* manifold, not a tighter tolerance.

The state is now cached; notebook 03 picks it up from here.
"""),
    md(r"""
## Optional: point-group and SU(2) symmetry

Two symmetries are switchable, and they switch differently.

**SU(2)** is the *space constructor*: `Su2Space` gives the spin-adapted layer, `QnSpace` the
`Nup`/`Ndn` one. Everything downstream — `mals`, `riemannian`, `rpa`, `pd_probe`, `save` — is
generic over the layer, so it is a one-line change. Note SU(2) ranks are **reduced multiplet**
ranks, so a given `M` is not the same object on the two layers, and the SU(2) singlet layer gives
the singlet spectrum directly where the abelian layer also returns triplets.

**Point group** is a keyword, and the *kind* is chosen once at read time — choosing it later would
silently reinterpret the `ISYM` target. It needs a fixture carrying `ORBSYM`, so it goes through
`read_fcidump`, not the bare `h1`/`h2` pair used above. `screen_forbidden!` is mandatory: a
symmetry-adapted dump carries a ~1e-14 forbidden residue that survives `eps = eps(Float64)` and
trips a hard assert during construction.

It is a **trade, not a free win**: point-group symmetry makes the Hamiltonian more expensive to
build and store in exchange for faster solves. Turn it on where the solver dominates, not by
reflex.
"""),
    code(r"""
# Symmetry-adapted fixtures in testdata/ (all self-describing FCIDUMPs):
#   h2o_c2v/H2O.FCIDUMP        NORB=7,  NELEC=10, MS2=0   C2v
#   ch2_triplet_c2v/*.FCIDUMP  NORB=7,  NELEC=8,  MS2=2   C2v
#   n2_d2h/*.FCIDUMP           NORB=28, NELEC=14, MS2=0   D2h   <- where the trade actually pays
PG_FIXTURE = fixture("h2o_c2v", "H2O.FCIDUMP")

data_pg = QNTT.read_fcidump(PG_FIXTURE; point_group = :auto)   # :auto, or pin :C1 … :D2h
pg = QNTT.pointgroup(data_pg)
@printf("%s: norb=%d nelec=%d ms2=%d — %s, target %s\n", basename(PG_FIXTURE),
        QNTT.norb(data_pg), QNTT.nelec(data_pg), QNTT.ms2(data_pg), pg.kind, pg.target)

QNTT.screen_forbidden!(data_pg)        # MANDATORY before constructing over a PG space

# All four combinations construct; pick one and the rest of the notebook is unchanged.
for (label, sp) in (("SU(2) + PG ", QNTT.Su2Space(data_pg; two_s = 0, point_group = :inherit)),
                    ("SU(2)      ", QNTT.Su2Space(data_pg; two_s = 0, point_group = nothing)),
                    ("abelian + PG", QNTT.QnSpace(data_pg)))
    H_pg = QNTT.Hamiltonian(sp, data_pg)          # the Hamiltonian is built over the chosen space
    @printf("  %-12s %s\n", label, sp)
end
"""),
]

notebook(cells02, "02-ground-states.ipynb")


# =================================================================================================
# 03 — linear response
# =================================================================================================

cells03 = [
    md(r"""
# Excitation energies from the Riemannian Hessian

The **symplectic (Williamson) eigenvalues of the Riemannian Hessian** at a critical point on the
fixed-rank tensor-train manifold, following Grazioli, Hu & Cancès
([arXiv:2506.16420](https://arxiv.org/abs/2506.16420)). At full rank the ω must reproduce the exact
FCI gaps; below it they are fixed-rank estimates whose error is a property of the ansatz.
"""),
    code(r"""
using QNTT, LinearAlgebra, Printf, Combinatorics, Plots
gr()
QNTT.init()
""" + FIXTURES + CACHE + r"""

h4 = QNTT.read_fcidump(fixture("h4.FCIDUMP"))
space = QNTT.QnSpace(h4)
H = QNTT.Hamiltonian(space, h4)
@printf("%s, E_core = %.10f\n", string(space), QNTT.e0(h4))
"""),
    md(r"""
## The exact answer, first

`C(4,2)² = 36` determinants — small enough to diagonalize exactly. `tt_state` builds each as a
rank-1 train, `xthy(x, H, y)` contracts `⟨x, H·y⟩`.
"""),
    code(r"""
d, nup, ndn = space.d, space.a, space.b
dets = [(u, v) for u in combinations(0:d-1, nup) for v in combinations(0:d-1, ndn)]
basis = [QNTT.tt_state([k in u for k in 0:d-1], [k in v for k in 0:d-1]) for (u, v) in dets]
n = length(basis)

Hmat = [QNTT.xthy(basis[i], H, basis[j]) for i in 1:n, j in 1:n]
@printf("%d determinants; ‖H − Hᵀ‖ = %.2e\n", n, norm(Hmat - Hmat'))

E = eigvals(Symmetric((Hmat + Hmat') / 2))
gaps = E[2:end] .- E[1]

@printf("\nFCI ground state      E₀ = %.12f Ha  (electronic)\n", E[1])
@printf("FCI excitation energies (Ha):\n")
for k in 1:8
    @printf("   E%d − E₀ = %.9f\n", k, gaps[k])
end
"""),
    md(r"""
## Linear response at full rank

Converge on a manifold large enough to be exact, then ask for the lowest symplectic eigenvalues.
`rpa` is matrix-free, a convenience over `rpa_signed`; `max_iter` counts `(A, B)` apply pairs.
"""),
    code(r"""
x0 = QNTT.tt_randn(space, QNTT.uniform_ranks(space, 8); seed = 7)
λ, x = QNTT.als(H, x0; eps = 1e-12, max_iter = 200)

@printf("ground state: λ = %.12f  (FCI E₀ = %.12f, difference %.1e)\n", λ, E[1], λ - E[1])
@printf("manifold: max bond dimension %d\n", QNTT.max_bond_rank(x))

ω, info = QNTT.rpa(H, x; k = 6, tol = 1e-10, max_iter = 800)
@printf("\nRPA: %d Krylov vectors, %d Hessian applies, converged = %s\n",
        info.iterations, info.applies, info.converged)
"""),
    code(r"""
@printf("%-4s %18s %18s %14s\n", "j", "ω_j (RPA)", "E_j − E₀ (FCI)", "difference")
for j in eachindex(ω)
    @printf("%-4d %18.12f %18.12f %14.2e\n", j, ω[j], gaps[j], ω[j] - gaps[j])
end
"""),
    md(r"""
They agree to solver tolerance — the strongest available test of the whole Hessian machinery.

## The certified dense driver

`rpa_dense` assembles the whole tangent basis and runs Williamson directly: a small-system
reference rather than a production path.
"""),
    code(r"""
ωd = QNTT.rpa_dense(H, x)

@printf("dense driver returned %d values; matrix-free returned %d\n", length(ωd), length(ω))
@printf("%-4s %18s %18s %14s\n", "j", "matrix-free", "dense", "difference")
for j in eachindex(ω)
    @printf("%-4d %18.12f %18.12f %14.2e\n", j, ω[j], ωd[j], ω[j] - ωd[j])
end
"""),
    md(r"""
Multiplicities are **resolved**: `k` counts physical values with multiplicity. Never de-duplicate
the output — a repeat is a degeneracy to be checked, never deleted.

## What happens below full rank

The ω become approximations — an ansatz bias, not noise.
"""),
    code(r"""
ranks = [1, 2, 3, 4, 6]
rows = map(ranks) do r
    x0 = QNTT.tt_randn(space, QNTT.uniform_ranks(space, r); seed = 7)
    λr, xr = QNTT.als(H, x0; eps = 1e-12, max_iter = 300)
    # Below some rank the reference is poor enough that its RPA spectrum is no longer real (an
    # imaginary or complex mode, a de-excitation), and there is no ω to report. `rpa` throws rather
    # than returning numbers (`rpa_signed` returns the signed spectrum as data) — catch that and
    # record it, because it is a real answer about the state, not a solver failure.
    ωr = try
        first(QNTT.rpa(H, xr; k = 4, tol = 1e-10, max_iter = 800))
    catch e
        e isa QNTTError || rethrow()
        nothing
    end
    (; r, λ = λr, bond = QNTT.max_bond_rank(xr), ω = ωr)
end

@printf("%-5s %-6s %16s   %s\n", "rank", "bond", "λ − E₀", "ω₁ … ω₄")
for row in rows
    @printf("%-5d %-6d %16.2e   %s\n", row.r, row.bond, row.λ - E[1],
            row.ω === nothing ? "— no PD metric, so no spectrum" :
            join([@sprintf("%.6f", w) for w in row.ω], "  "))
end
"""),
    code(r"""
ok = [row for row in rows if row.ω !== nothing]      # ranks that HAVE a symplectic spectrum
p = plot(; xlabel = "per-sector rank of the manifold", ylabel = "ω  (Ha)",
         title = "H₄: fixed-rank linear response vs exact FCI gaps",
         legend = :outerright)
for j in 1:4
    plot!(p, [row.r for row in ok], [row.ω[j] for row in ok];
          marker = :circle, lw = 2, label = "ω_$j")
    hline!(p, [gaps[j]]; ls = :dash, lc = :black, alpha = 0.4,
           label = j == 1 ? "exact FCI gaps" : "")
end
p
"""),
    md(r"""
## Caveats worth carrying

- The ω are **fixed-rank estimates**; quote the rank with the number.
- `x` must be a genuine **minimizer** — at a saddle `rpa` throws (no real ω); `rpa_signed` returns
  the signed spectrum there.
- Fixed-rank TT is nonlinear, so critical points need not be excited states.
"""),
    md(r"""
---

# Part II — a real system, and what it takes to trust the answer

Water 6-31G at `M = 64`, the ground state notebook 02 builds and caches. The order is fixed:

`pd_probe` on both blocks of the Hessian → a **stability verdict** → `rpa` for the excitation
energies → a comparison with full-space FCI.

`rpa` publishes ω only at a stable point; at an unstable one it throws and `rpa_signed` returns the
signed spectrum as data. The probes come first because a verdict is only as good as the budget behind
it, and that is a knob you set.
"""),
    md(r"""
### Knobs to play with

| knob | default | controls |
|---|---|---|
| `PROBE_BUDGET` | 600 | `pd_probe` Davidson iterations; at 200 both parts are still `:inconclusive` |
| `RPA_K` | 4 | physical ω returned, with multiplicity |
| `RPA_TOL` | 1e-4 | 3–4 digits, the target at this ansatz |
| `RPA_MAX_ITER` | 4000 | `(A, B)` apply pairs |
"""),
    code(r"""
# The ground state is the expensive part, and notebook 02 already built and cached it. This
# reloads that exact file; run 02 first if it is absent.
const FIXTURE = "h2o_631g"
const TWO_S   = 0
const M       = 64

fx = load_fixture(fixture(FIXTURE))
space_w = QNTT.Su2Space(fx.d, fx.nelec, TWO_S)
H_w = QNTT.Hamiltonian(space_w, fx.t, fx.v)

const GS_TAG = @sprintf("%s_2S%d_M%d_gs", FIXTURE, TWO_S, M)
if !isfile(cache_file(GS_TAG))
    error("no cached ground state at $(cache_file(GS_TAG)) — run notebook 02 first; " *
          "its final section builds and caches exactly this state.")
end
x_w, how, t_load = cached(Float64, QNTT.SU2, GS_TAG, space_w, () -> error("unreachable"))
report_cache(GS_TAG, how, t_load)

λ_w = QNTT.rayleigh_quotient(H_w, x_w)
E_w = λ_w + fx.e_nuc
@printf("\nE        = %.9f Ha   (%.2f%% of the correlation energy)\n",
        E_w, 100 * (fx.e_hf - E_w) / (fx.e_hf - fx.e_ref))
@printf("‖grad‖   = %.3e   %s\n", QNTT.grad_norm(H_w, x_w),
        QNTT.grad_norm(H_w, x_w) < 1e-9 ? "(critical)" : "(NOT critical — the probes below are then not about a minimizer)")
@printf("σ_min    = %.3e\n", QNTT.sigma_min(x_w))
@printf("bond dims = %s\n", string([sum(g) for g in QNTT.ranks(x_w)]))
"""),
    md(r"""
## The stability gate

`pd_probe` returns a Rayleigh quotient of the Hessian part, so a negative answer is a *proof* (a
variational upper bound on `λ_min`) and a non-negative one is evidence measured by its residual.

- **`:rr` = `A + B`**, the energy Hessian. Indefinite ⇒ the ground state is a *saddle*.
- **`:ii` = `A − B`**, imaginary-direction curvature. Indefinite ⇒ a *physical* instability no real
  optimizer ever looks at: the energy drops along a complex direction.

Both must be positive definite for real ω to exist.
"""),
    code(r"""
const PROBE_BUDGET = 600     # Davidson iterations ≈ Hessian applies
const PROBE_SEED   = 1234    # fixes the random starting tangent direction

for part in (:rr, :ii)
    t = @elapsed r = QNTT.pd_probe(H_w, x_w; part = part, budget = PROBE_BUDGET, seed = PROBE_SEED)
    @printf("pd_probe(:%s)  %-16s λ = %+.6e   residual = %-10s margin = %+.2e   %3d applies  %5.1f s\n",
            part, r.verdict, r.lambda,
            isnan(r.residual) ? "n/a" : @sprintf("%.1e", r.residual),
            r.lambda - r.residual, r.applies, t)
end
"""),
    md(r"""
Read each verdict as a pair: `λ_min` beside its residual. The **margin** `λ_min − residual` is the
number that decides whether `λ_min > 0` is real; a `:likely_pd` whose residual reaches `λ_min` says
nothing.

### The verdict depends on the budget

The same probe at a third of the budget, and at another seed:
"""),
    code(r"""
for (seed, budget) in ((PROBE_SEED, 200), (99, 200), (PROBE_SEED, PROBE_BUDGET))
    t = @elapsed r = QNTT.pd_probe(H_w, x_w; part = :ii, budget = budget, seed = seed)
    @printf("  seed %5d, budget %3d:  %-14s λ = %+.6e   residual = %.1e   %5.1f s\n",
            seed, budget, r.verdict, r.lambda, r.residual, t)
end
"""),
    md(r"""
At 200 iterations the bound has settled but the residual has not: `:inconclusive` is *not* a PD
claim, it is the probe declining to make one. Quote a verdict with its budget.

### Nothing to complexify along

At a state where `:ii` *is* indefinite, the proven negative direction `w` is the way out:
`complex_seed` steps to `retract(x, i·ε·ŵ)`, a complex polish finds the lower complex critical
point, and the RPA is done there. Here the gate says there is no such direction, and
`complex_seed` refuses to invent one:
"""),
    code(r"""
try
    QNTT.complex_seed(H_w, x_w; eps = 1e-2, budget = PROBE_BUDGET, seed = PROBE_SEED)
    println("complex_seed returned a seed: the probe found a negative H^ii direction after all")
catch e
    e isa QNTTError || rethrow()
    println(sprint(showerror, e))
end
"""),
    md(r"""
The refusal is the answer: the real ground state is stable against complex perturbations at this
rank, so the linear response is computed at the real state itself.

## Excitation energies

`tol ≈ 1e-4`: the model error is several percent, so more digits refine a number already wrong in the
second.
"""),
    code(r"""
const RPA_TOL      = 1e-4    # 3-4 digits: the physically meaningful target at this ansatz
const RPA_K        = 4       # physical values, with multiplicity
const RPA_MAX_ITER = 4000    # budget in (A, B) apply pairs (pair-equivalents)

t = @elapsed ω, info = QNTT.rpa(H_w, x_w; k = RPA_K, tol = RPA_TOL,
                                max_iter = RPA_MAX_ITER, seed = 1)
@printf("rpa: %d values, converged = %s, %d basis vectors, %d (A, B) apply pairs, %.1f s\n",
        length(ω), info.converged, info.iterations, info.applies, t)
for (j, w) in enumerate(ω)
    @printf("  ω[%d] = %.6f Ha\n", j, w)
end
info.converged ||
    println("\nNOT at tolerance in $RPA_MAX_ITER apply pairs. The lowest values settle first, " *
            "so ω[1] and ω[2] are trustworthy well before the upper ones; raise RPA_MAX_ITER to " *
            "tighten the rest. This is a budget statement, not a failure.")
"""),
    code(r"""
# Full-space FCI singlet gaps for this exact system, from the CASSCF-paper reference set. No JSON
# dependency: pull the one array we need out of the `6-31g` block textually, and assert on what
# came back rather than trusting the scrape.
refs = read(fixture("casscf_paper_references.json"), String)
blk = refs[findfirst("\"6-31g\"", refs)[1]:end]
span = blk[findfirst("\"fci_gaps\"", blk)[1]:end]
span = span[findfirst("[", span)[1]:findfirst("]", span)[1]]
fci_gaps = [parse(Float64, m.match) for m in eachmatch(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", span)]
length(fci_gaps) == 2 || error("expected 2 FCI gaps, scraped $(length(fci_gaps)): $fci_gaps")
@printf("full-space FCI singlet gaps: %s Ha  (%d determinants)\n",
        join([@sprintf("%.6f", g) for g in fci_gaps], ", "), 1656369)

# Pair each FCI gap with its NEAREST ω, not by index. The reference file carries three FCI roots,
# hence two gaps; the fixed-rank spectrum has its own density of states and need not line up
# one-for-one, so an index-by-index table would be comparing whatever happened to be in slot 2.
@printf("\n%-16s %14s %10s %12s\n", "FCI (full space)", "nearest ω", "index", "error")
for g in fci_gaps
    j = argmin(abs.(ω .- g))
    @printf("%-16.6f %14.6f %10d %11.2f%%\n", g, ω[j], j, 100 * (ω[j] - g) / g)
end
unmatched = [j for j in eachindex(ω) if !(j in [argmin(abs.(ω .- g)) for g in fci_gaps])]
isempty(unmatched) ||
    @printf("\nTT values with no counterpart among the reference's roots: %s\n",
            join([@sprintf("ω[%d] = %.6f", j, ω[j]) for j in unmatched], ", "))
"""),
    md(r"""
Every ω sits **above** both FCI gaps, by several percent to over ten, against a 1.66-million-determinant
full space at pooled bond dimension 64. Pairing by nearest value can send both gaps to the same `ω`
— the table prints the index so that is visible rather than hidden. Which fixed-rank mode
corresponds to which FCI state is not settled by energies alone; the ω are fixed-rank estimates,
quoted with the rank.
"""),
]

notebook(cells03, "03-linear-response.ipynb")


# =================================================================================================
# 04 / 05 — the CASSCF-paper systems: CASSCF baseline vs DMRG → polish → RPA → GAD, per layer
# =================================================================================================
#
# One generator, two notebooks: the abelian (Nup, Ndn) layer and the SU(2) spin-adapted layer run
# the identical pipeline on the identical fixtures, so the cells are templated on a handful of
# «PLACEHOLDERS» (space constructors, the start state, the layer tag) and a few layer-specific
# prose cells. Everything the two notebooks do differently is visible in `paper_cells` below.
#
# The ABI surface these cells call — `cas_reference_state`, `casscf`, `joint_rpa`, `saddle_seed`,
# `gad` — lives in capi/src/{joint,gad,probe}.rs and julia/src/highlevel/solvers.jl.

PAPER_HELPERS = r'''
# --- the systems of arXiv:2604.13753 (Part II, CASSCF) ------------------------------------------
# HF-canonical-MO fixtures with ORBSYM: h2o C2v, hcho C2v, c2h4 D2h. `cas` is the
# (n_core, n_active, n_virt) partition — cores first, virtuals last, as pyscf canonicalizes it.
const SYSTEMS = (
    (dir = "h2o_631g",  mol = "h2o",  label = "water CAS(8,6) / 6-31G",        cas = (1, 6, 6)),
    (dir = "hcho_631g", mol = "hcho", label = "formaldehyde CAS(4,3) / 6-31G", cas = (6, 3, 13)),
    (dir = "c2h4_631g", mol = "c2h4", label = "ethylene CAS(2,2) / 6-31G",     cas = (7, 2, 17)),
)
system(dir) = SYSTEMS[findfirst(s -> s.dir == dir, SYSTEMS)]

# Paper Table II (Hartree): rows ω₁, ω₂; columns SS / SA / LR. The LR column is what the
# CASSCF-equivalent below reproduces (certified in tests/joint_cas at ≤ 5e-7); SS and SA are
# different methods, printed for context only.
const PAPER = Dict(
    "h2o"  => [0.299801 0.279149 0.313274; 0.377632 0.357262 0.388773],
    "hcho" => [0.144960 0.147317 0.138848; 0.335188 0.464643 0.325269],
    "c2h4" => [0.384667 0.388940 0.354318; 0.567533 0.568822 0.562656],
)

"""
Scrape one value from the `6-31g` block of molecule `mol` in `testdata/casscf_paper_references.json`:
a scalar, a flat array, or `nothing` for a JSON `null`. Textual on purpose (no JSON dependency, as
notebook 03 does); every caller asserts on what comes back.
"""
function ref_value(mol, key)
    refs = read(fixture("casscf_paper_references.json"), String)
    function cut(s, pat)
        r = findfirst(pat, s)
        r === nothing && error("$pat not found in the reference file for $mol")
        return s[last(r)+1:end]
    end
    s = lstrip(cut(cut(cut(refs, "\"$mol\""), "\"6-31g\""), "\"$key\":"))
    startswith(s, "null") && return nothing
    num = r"-?\d+\.\d+(?:[eE][-+]?\d+)?"
    if startswith(s, "[")
        body = s[2:findfirst(']', s)-1]
        return [parse(Float64, m.match) for m in eachmatch(num, body)]
    end
    return parse(Float64, match(num, s).match)
end

"""
    cached_value(tag, build) -> (value, :warm|:cold, seconds)

`Serialization`-based sibling of `cached` for the small results (ω lists, energies, info tuples).
Same contract: keyed on the tag alone, so delete `notebooks/cache/` after changing a knob.
"""
function cached_value(tag, build)
    path = joinpath(CACHE_DIR, tag * ".jls")
    if isfile(path)
        t = @elapsed v = deserialize(path)
        return (v, :warm, t)
    end
    t = @elapsed v = build()
    mkpath(CACHE_DIR)
    serialize(path, v)
    return (v, :cold, t)
end

"Nearest entry of `ω` to `g`: (value, 1-based index, ω − g), or `nothing` on an empty spectrum."
function nearest(ω, g)
    isempty(ω) && return nothing
    j = argmin(abs.(ω .- g))
    return (ω[j], j, ω[j] - g)
end
fmt_near(n) = n === nothing ? "      —          " : @sprintf("%.6f [%2d] %+.1e", n[1], n[2], n[3])
"Nearest mode with its Weingarten fraction beside it: `ω [index] deviation  f_W`."
fmt_near(n, f_w) = n === nothing ? @sprintf("%-32s", "—") :
    @sprintf("%.6f [%2d] %+.1e f_W %s", n[1], n[2], n[3], isempty(f_w) ? "  ?  " : @sprintf("%5.2f", f_w[n[2]]))
fmt_fw(f_w, k) = isempty(f_w) ? "—" : join([@sprintf("%.2f", w) for w in f_w[1:min(k, length(f_w))]], "    ")

# `flush(stdout)` after each report: a no-op in Jupyter, and what makes a redirected `nbconvert`
# or script log stream while the long Rust calls run (Julia's I/O loop is blocked inside them).
fmt_ω(ω, k) = isempty(ω) ? "—" : join([@sprintf("%.5f", w) for w in ω[1:min(k, length(ω))]], " ")
'''


def paper_cells(layer):
    """Cells of the 04 (abelian) / 05 (SU(2)) notebook; `layer` is "abelian" or "su2"."""
    ab = layer == "abelian"
    sub = {
        "«LAYER»": layer,
        "«L»": "QNTT.Abelian" if ab else "QNTT.SU2",
        "«LAYERNAME»": "abelian `(Nup, Ndn)`" if ab else "SU(2) spin-adapted",
        # the untyped (C1) space of a loaded system `s` (fields fx, nup, ndn)
        "«UNTYPED»": "QNTT.QnSpace(s.fx.d, s.nup, s.ndn)" if ab
                     else "QNTT.Su2Space(s.fx.d, s.fx.nelec, 0)",
        # the point-group-typed space from a parsed FCIDUMP `data`
        "«TYPED»": "QNTT.QnSpace(data)" if ab
                   else "QNTT.Su2Space(data; two_s = 0, point_group = :inherit)",
        # the untyped space implied by a C1 FCIDUMP `data` (the `_cas` twins carry no ORBSYM)
        "«UNTYPED_FROM_DATA»": "QNTT.QnSpace(data)" if ab
                               else "QNTT.Su2Space(data; two_s = 0, point_group = nothing)",
        # the closed-shell start in space `sp` for a loaded system `s`
        "«START»": ("QNTT.tt_state([k <= s.nup for k in 1:s.fx.d], "
                    "[k <= s.ndn for k in 1:s.fx.d], sp)") if ab
                   else "QNTT.tt_csf([k <= s.fx.nelec ÷ 2 ? :occ : :empty for k in 1:s.fx.d], sp)",
        "«STARTNAME»": "Hartree-Fock determinant" if ab else "closed-shell CSF",
        "«RANKWORD»": "pooled bond dimension (summed over sectors)" if ab
                      else "reduced multiplet rank (NOT comparable to an abelian pooled M)",
        "«CAPWORD»": "pooled bond-dimension" if ab else "reduced-multiplet",
    }

    def T(text):
        for k, v in sub.items():
            text = text.replace(k, v)
        assert "«" not in text, text
        return text

    cells = [
        md(T(r"""
# CASSCF baseline, DMRG, Riemannian polish, RPA and GAD on the CASSCF-paper systems — «LAYERNAME» layer

Systems: the three molecules of arXiv:2604.13753 (Grazioli, Hu, Nottoli, Lipparini, Cancès,
*Critical point search and linear response theory … Part II. CASSCF*): water CAS(8,6),
formaldehyde CAS(4,3), ethylene CAS(2,2); singlets, 6-31G, fixture geometries, Hartree-Fock
canonical orbitals.

Per system:

1. CASSCF-equivalent: joint TT ⊕ orbital ground state on the pinned CAS grid, then joint RPA at
   that point.
2. DMRG (two-site MALS) to the full-space ground state at truncation tolerance `DMRG_EPS` under
   the «CAPWORD» cap `DMRG_CAP`, on the untyped space and on the point-group-typed
   space.
3. Riemannian CG polish of both DMRG states to `‖grad‖ < GRADTOL`.
4. Signed RPA at each polished state; one table against the paper, `f_W` beside every mode.
5. GAD from each polished state.

No parameter search, no tuning to reference values, no orbital rotation other than the CASSCF.
The ground state is real throughout. Notebook 04 is the abelian layer, 05 the SU(2) layer, from
one generator (`paper_cells` in `mknotebooks.py`); they share `notebooks/cache/`.
""")),
        code(T(r"""
using QNTT, LinearAlgebra, Printf, Serialization

# RAYON_THREADS is set by the cluster job; 0 lets rayon choose.
QNTT.init(parse(Int, get(ENV, "RAYON_THREADS", "0")))
@printf("qntt %s, %d threads, %s layer\n", QNTT.version(), QNTT.nthreads(), "«LAYER»")
""") + FIXTURES + CACHE + PAPER_HELPERS + T(r"""

const LAYER = "«LAYER»"
# Systems to run (h2o: d = 13; hcho: d = 22; c2h4: d = 26).
const RUN = ["h2o_631g", "hcho_631g", "c2h4_631g"]
""")),
        md(r"""
## Setup

Reads each FCIDUMP with its point group, screens the symmetry-forbidden entries, builds the
untyped space and the typed space and one operator on each. Prints `E_HF`, `E_ref`, the pyscf
CASSCF energy, the paper's LR values and both construction times.
"""),
        code(T(r"""
"Load one paper system: screened integrals, the two spaces, the two operators, the references."
function setup(dir)
    sys = system(dir)
    fx = load_fixture(fixture(dir))
    dump = only(filter(f -> endswith(f, ".FCIDUMP"), readdir(fixture(dir))))
    data = QNTT.read_fcidump(joinpath(fixture(dir), dump); point_group = :auto)
    QNTT.screen_forbidden!(data)                 # MANDATORY before a point-group-typed construction
    pg = QNTT.pointgroup(data)
    nup = (fx.nelec + fx.ms2) ÷ 2
    ndn = fx.nelec - nup
    s = (; sys, fx, data, pg, nup, ndn)
    sp_c1 = «UNTYPED»
    sp_pg = «TYPED»
    t_c1 = @elapsed H_c1 = QNTT.Hamiltonian(sp_c1, data)
    t_pg = @elapsed H_pg = QNTT.Hamiltonian(sp_pg, data)
    casscf_gs = ref_value(sys.mol, "casscf_gs")
    fci_gaps  = ref_value(sys.mol, "fci_gaps")        # `nothing` where no FCI exists (hcho, c2h4)
    return (; s..., sp_c1, sp_pg, H_c1, H_pg, t_c1, t_pg, casscf_gs, fci_gaps)
end

S = Dict(dir => setup(dir) for dir in RUN)
for dir in RUN
    s = S[dir]
    println(s.sys.label)
    @printf("  d = %d, N = %d, %s (target %s)   E_HF = %.9f   E_ref (%s) = %.9f\n",
            s.fx.d, s.fx.nelec, s.pg.kind, s.pg.target, s.fx.e_hf,
            isnan(s.fx.e_ref) ? "none" : (s.fci_gaps === nothing ? "CCSD(T)" : "FCI"), s.fx.e_ref)
    @printf("  pyscf CASSCF   = %.9f      paper LR ω₁, ω₂ = %.6f, %.6f\n",
            s.casscf_gs, PAPER[s.sys.mol][1, 3], PAPER[s.sys.mol][2, 3])
    @printf("  untyped space  %s   built in %6.1f s\n", string(s.sp_c1), s.t_c1)
    @printf("  typed space    %s   built in %6.1f s  (%.1f× the untyped build)\n",
            string(s.sp_pg), s.t_pg, s.t_pg / s.t_c1)
    flush(stdout)
end
""")),
        md(T(r"""
## 1. CASSCF-equivalent

`cas_reference_state` builds the pinned CAS(n, m) grid (core and virtual bonds carry one sector
at rank 1). `casscf` runs the joint TT ⊕ orbital solver from the fixture orbitals with the grid
held fixed. `joint_rpa` returns the symplectic eigenvalues of the joint Hessian at the converged
point and `f_W` per mode. The energy is compared with pyscf's CASSCF (`casscf_gs`), the two
lowest ω with the paper's LR column. ω are printed only under `converged && !rank_limited`.
`CASSCF_START = :cas` starts from the `*_cas` fixture twin (pyscf CASSCF orbitals) instead.
""")),
        code(T(r"""
const CASSCF_START    = :hf      # :hf = from the fixture (HF) orbitals; :cas = from the `*_cas` twin
const CASSCF_ORB_ITER = 40       # orbital iterations
const CASSCF_ORB_GRAD = 1e-6     # orbital-gradient gate
const CASSCF_TT_GRAD  = 1e-6     # TT-leg gradient tolerance
const CASSCF_SEED     = 0xCA5F0
const SHOW_K          = 6        # how many ω to print per spectrum

function casscf_baseline(s)
    sys = s.sys
    if CASSCF_START === :hf
        data, H = s.data, s.H_c1
    else
        dir_cas = fixture(sys.dir * "_cas")
        dump = only(filter(f -> endswith(f, ".FCIDUMP"), readdir(dir_cas)))
        data = QNTT.read_fcidump(joinpath(dir_cas, dump))          # C1 on purpose (see its meta.txt)
        H = QNTT.Hamiltonian(«UNTYPED_FROM_DATA», data)
    end
    t0 = time()
    λ0, x0 = QNTT.cas_reference_state(H, sys.cas; m_active = 0, seed = CASSCF_SEED)
    λ, x, u, H_rot, info = QNTT.casscf(data, x0, sys.cas;
                                       orb_max_iter = CASSCF_ORB_ITER, orb_grad = CASSCF_ORB_GRAD,
                                       tt_grad_tol = CASSCF_TT_GRAD, preserve_grid = true)
    t_scf = time() - t0
    e0 = QNTT.e0(data)
    certified = info.converged && !info.rank_limited          # the publication licence
    if certified
        ω, resp = QNTT.joint_rpa(H_rot, x, sys.cas)
    else
        ω, resp = Float64[], (status = :not_run, n_eff = info.n_eff, n_modes = 0, f_w = Float64[],
                              message = "the SCF did not certify — no ω published")
    end
    return (; E = λ + e0, E_start = λ0 + e0, converged = certified, iters = info.iters,
              refused = info.refused_reason, n_eff = info.n_eff,
              g_orb = isempty(info.g_orb_kept) ? NaN : last(info.g_orb_kept),
              ω, f_w = resp.f_w, status = resp.status, n_modes = resp.n_modes, message = resp.message,
              ranks = QNTT.ranks(x), t_scf, t_all = time() - t0)
end

B = Dict{String,Any}()
for dir in RUN
    s = S[dir]
    tag = @sprintf("%s_%s_casscf_%s", dir, LAYER, CASSCF_START)
    B[dir], how, t = cached_value(tag, () -> casscf_baseline(s))
    report_cache(tag, how, t); flush(stdout)
    b = B[dir]
    println(s.sys.label)
    @printf("  start (fixed orbitals)  E = %.9f\n", b.E_start)
    @printf("  CASSCF-equivalent       E = %.9f   Δ vs pyscf CASSCF = %+.2e   %d orbital iterations, \
              ‖g_orb‖ = %.2e, %.1f s\n", b.E, b.E - s.casscf_gs, b.iters, b.g_orb, b.t_scf)
    if !b.converged
        @printf("  DID NOT CERTIFY — no ω published. Refusal: %s\n", something(b.refused, "?"))
        continue
    end
    @printf("  joint RPA: %s, n_eff = %d non-redundant orbital pairs, %d modes, %.1f s total\n",
            b.status, b.n_eff, b.n_modes, b.t_all)
    b.status in (:stable, :resonant) || (println("  ", b.message); continue)
    b.status === :resonant && println("  RESONANT joint point: negative entries below are imaginary pairs (ω = i√|ω²|), not excitations")
    @printf("  ω[1:%d]   = %s\n", SHOW_K, fmt_ω(b.ω, SHOW_K))
    @printf("  f_W[1:%d] = %s\n", SHOW_K, fmt_fw(b.f_w, SHOW_K))
    for j in 1:2
        lr = PAPER[s.sys.mol][j, 3]
        @printf("  paper LR ω%d = %.6f   nearest mode: %s\n", j, lr, fmt_near(nearest(b.ω, lr), b.f_w))
    end
end
""")),
    ]

    if ab:
        cells.append(md(r"""
The abelian chart is an `Sz = 0` chart: the spectrum contains the `Sz = 0` components of
triplets. The paper's values are matched by nearest value, not by index.
"""))
    else:
        cells.append(md(r"""
The spectrum is singlet-only; the paper's two values are expected at indices 1 and 2. At water
the abelian chart has 272 modes, 152 of them singlets.
"""))

    cells += [
        md(T(r"""
## 2. DMRG

Start: the «STARTNAME». Stage A: fixed-`M` ramp `RAMP_BOND_DIMS` with perturbative noise
`RAMP_NOISE`. Stage B: noise-free rank-adaptive MALS at truncation tolerance `DMRG_EPS` under
the «CAPWORD» cap `DMRG_CAP`, re-entered one sweep at a time until the energy changes by less than
`DMRG_STOP`. Same start and schedule on both spaces. Ranks on this layer are the «RANKWORD».

`DMRG_CAP` bounds memory and run time, which grow quickly with the cap; even at the default the
notebook runs for many hours. The cap binds on all three systems, so the results are fixed-rank
results at that cap, with the truncation tolerance printed beside them.
""")),
        code(T(r"""
const DMRG_EPS       = 1e-3      # stage-B truncation tolerance
const DMRG_CAP       = 256       # stage-B «CAPWORD» cap (0 = uncapped)
const DMRG_STOP      = 1e-7      # stage-B energy stopping test (Ha), between our own re-entries
const DMRG_MAX_ITER  = 20        # stage-B sweep budget
const RAMP_BOND_DIMS = [16, 32, 64, 128, 128]           # stage A: fixed-M ramp WITH noise …
const RAMP_NOISE     = [1e-3, 1e-4, 1e-5, 1e-6, 0.0]    # … ending at 0.0 (only then is E variational)
const RAMP_EPS       = 1e-8
const RAMP_MAX_ITER  = 6
const RAMP_NOISE_TYPE = :perturbative   # White density-matrix noise, the library default (see below)
const MALS_SEED      = 0x9E3779B9

closed_shell(s, sp) = «START»

# Noise type, measured on water 6-31G from the closed-shell start: random noise leaves the typed
# run 0.035–0.045 Ha above the untyped one (E −76.076 / −76.086 for a 32- / 128-ramp vs −76.1212);
# perturbative noise with the 128-ramp brings both arms to −76.121230.

function dmrg(H, x0, e_nuc, label)
    t_a = @elapsed λa, xa, _ = QNTT.mals(H, x0; bond_dims = RAMP_BOND_DIMS, noise = RAMP_NOISE,
                                         noise_type = RAMP_NOISE_TYPE, eps = RAMP_EPS,
                                         max_iter = RAMP_MAX_ITER, seed = MALS_SEED)
    @printf("  %-8s stage A (M = %d, noisy)     E = %.9f   %7.1f s\n",
            label, last(RAMP_BOND_DIMS), λa + e_nuc, t_a)
    # `eps` is also the driver's energy stopping test; the stop used here is DMRG_STOP, applied
    # between re-entries of one sweep each.
    t_b = time()
    λb, xb, hist = QNTT.mals(H, xa; bond_dims = [DMRG_CAP], noise = [0.0], eps = DMRG_EPS, max_iter = 1)
    sweeps, dw = length(hist), last(hist).discarded_weight
    while sweeps < DMRG_MAX_ITER
        λn, xb, hist = QNTT.mals(H, xb; bond_dims = [DMRG_CAP], noise = [0.0], eps = DMRG_EPS, max_iter = 1)
        sweeps += length(hist); dw = last(hist).discarded_weight
        moved = abs(λn - λb); λb = λn
        moved < DMRG_STOP && break
    end
    @printf("  %-8s stage B (eps = %.0e, cap %d) E = %.9f   %7.1f s   %d sweeps, discarded weight %.2e, max bond %d%s\n",
            label, DMRG_EPS, DMRG_CAP, λb + e_nuc, time() - t_b, sweeps, dw, QNTT.max_bond_rank(xb),
            DMRG_CAP > 0 && QNTT.max_bond_rank(xb) >= DMRG_CAP ? "  (cap reached)" : "")
    flush(stdout)
    return xb
end

X = Dict{String,Any}()        # X[dir] = (c1 = state on the untyped space, pg = state on the typed space)
for dir in RUN
    s = S[dir]
    println(s.sys.label)
    tag_c1 = @sprintf("%s_%s_dmrg_c1_eps%.0e_cap%d", dir, LAYER, DMRG_EPS, DMRG_CAP)
    tag_pg = @sprintf("%s_%s_dmrg_pg_eps%.0e_cap%d", dir, LAYER, DMRG_EPS, DMRG_CAP)
    x_c1, how, t = cached(Float64, «L», tag_c1, s.sp_c1,
                          () -> dmrg(s.H_c1, closed_shell(s, s.sp_c1), s.fx.e_nuc, "untyped"))
    report_cache(tag_c1, how, t); flush(stdout)
    x_pg, how, t = cached(Float64, «L», tag_pg, s.sp_pg,
                          () -> dmrg(s.H_pg, closed_shell(s, s.sp_pg), s.fx.e_nuc, "typed"))
    report_cache(tag_pg, how, t); flush(stdout)
    X[dir] = (c1 = x_c1, pg = x_pg)
    for (label, H, x) in (("untyped", s.H_c1, x_c1), ("typed", s.H_pg, x_pg))
        E = QNTT.rayleigh_quotient(H, x) + s.fx.e_nuc
        @printf("  %-8s E = %.9f   Δ vs CASSCF %+.4e   Δ vs E_ref %+.4e   pooled ranks %s   max sector rank %d\n",
                label, E, E - s.casscf_gs, E - s.fx.e_ref, string([sum(g) for g in QNTT.ranks(x)]),
                QNTT.max_sector_rank(x))
    end
end
""")),
        md(r"""
## 3. Riemannian polish

One `round!` at `TRIM_EPS` (relative) first, then `polish` re-enters the Riemannian CG until
`‖grad‖ < GRADTOL` or `CG_RESTARTS` is exhausted, then a second trim and polish. Prints `E`,
`‖grad‖` with whether the gate or the budget stopped it, `σ_min` and the pooled ranks.

Stop the polish tight: on water at `DMRG_EPS = 1e-2`, a state stopped at `‖grad‖ < 1e-6` still
shows negative witnesses in both `pd_probe` blocks, and one carried to `1e-8` has none.
"""),
        code(T(r"""
const GRADTOL     = 1e-7    # criticality gate
const CG_MAX_ITER = 200
const CG_GRADTOL  = 1e-11   # inner CG stop
const CG_RESTARTS = 4
const TRIM_EPS    = 1e-6
const TRIM_PASSES = 1

"Riemannian CG to criticality: restart until ‖grad‖ < GRADTOL, or give up loudly."
function polish(H, x, λ_scale)
    inner_eps = GRADTOL / max(abs(λ_scale), 1)
    _, y = QNTT.riemannian(H, x; cg = true, eps = inner_eps, max_iter = CG_MAX_ITER, gradtol = CG_GRADTOL)
    ncg = 1
    for _ in 1:CG_RESTARTS
        QNTT.grad_norm(H, y) < GRADTOL && break
        _, y = QNTT.riemannian(H, y; cg = true, eps = inner_eps, max_iter = CG_MAX_ITER, gradtol = CG_GRADTOL)
        ncg += 1
    end
    return (y / norm(y), ncg)
end

function polish_and_trim(H, x, e_nuc, label)
    # Trim first: near-zero Schmidt directions put ~1/σ_min into the Weingarten curvature and
    # slow the CG; removing them before the polish is a conditioning step, not an accuracy one.
    QNTT.round!(x; eps = TRIM_EPS, mode = :rel)
    x = x / norm(x)
    t_p = @elapsed y, ncg = polish(H, x, QNTT.rayleigh_quotient(H, x))
    g = QNTT.grad_norm(H, y)
    @printf("  %-8s CG polish (%2d calls)   E = %.9f   ‖grad‖ = %.2e (%s)   %7.1f s\n",
            label, ncg, QNTT.rayleigh_quotient(H, y) + e_nuc, g,
            g < GRADTOL ? "at gate" : "budget exhausted above GRADTOL", t_p)
    for pass in 1:TRIM_PASSES
        t_t = @elapsed begin
            QNTT.round!(y; eps = TRIM_EPS, mode = :rel)
            y = y / norm(y)
            y, _ = polish(H, y, QNTT.rayleigh_quotient(H, y))
        end
        @printf("  %-8s trim pass %d (eps = %.0e)  E = %.9f   ‖grad‖ = %.2e   σ_min = %.2e   %7.1f s\n",
                label, pass, TRIM_EPS, QNTT.rayleigh_quotient(H, y) + e_nuc, QNTT.grad_norm(H, y),
                QNTT.sigma_min(y), t_t)
    end
    flush(stdout)
    return y
end

P = Dict{String,Any}()        # polished states, same shape as X
for dir in RUN
    s = S[dir]
    println(s.sys.label)
    tag_c1 = @sprintf("%s_%s_dmrg_c1_eps%.0e_cap%d_polished", dir, LAYER, DMRG_EPS, DMRG_CAP)
    tag_pg = @sprintf("%s_%s_dmrg_pg_eps%.0e_cap%d_polished", dir, LAYER, DMRG_EPS, DMRG_CAP)
    y_c1, how, t = cached(Float64, «L», tag_c1, s.sp_c1,
                          () -> polish_and_trim(s.H_c1, X[dir].c1, s.fx.e_nuc, "untyped"))
    report_cache(tag_c1, how, t); flush(stdout)
    y_pg, how, t = cached(Float64, «L», tag_pg, s.sp_pg,
                          () -> polish_and_trim(s.H_pg, X[dir].pg, s.fx.e_nuc, "typed"))
    report_cache(tag_pg, how, t); flush(stdout)
    P[dir] = (c1 = y_c1, pg = y_pg)
    for (label, H, y) in (("untyped", s.H_c1, y_c1), ("typed", s.H_pg, y_pg))
        g = QNTT.grad_norm(H, y)
        @printf("  %-8s E = %.9f   ‖grad‖ = %.2e (%s)   σ_min = %.2e   pooled ranks %s\n",
                label, QNTT.rayleigh_quotient(H, y) + s.fx.e_nuc, g,
                g < GRADTOL ? "critical" : "NOT critical at GRADTOL", QNTT.sigma_min(y),
                string([sum(g) for g in QNTT.ranks(y)]))
    end
end
""")),
        md(r"""
## 4. Linear response

At each of the three states (CASSCF point, polished untyped DMRG, polished typed DMRG):
`pd_probe(:rr)` and `pd_probe(:ii)` as diagnostics, then `rpa_signed` for the lowest `RPA_K` values.
`rpa_signed` returns signed ω²: positive entries are excitations, negative entries resonances `ω = iκ`, each
with a fresh-apply certificate. It never refuses: an indefinite point comes back solved, with a
nonzero counting offset in `info.report`, its complex modes in `info.complex` and each real mode's
Krein margin in `info.krein` (negative for a de-excitation) — its ω are then not licensed by
`converged` alone. Per mode it returns `f_W = |⟨q,Bq⟩| / |⟨q,Aq⟩|` on `q = X − Y`, with
`A = P_T(H−λ)P_T` and `B` the Weingarten term (`H^rr = A + B`, `H^ii = A − B`); on the `H^ii`
side `f_W > 1` is equivalent to `ω² < 0`.

Response at a point-group-typed reference is irrep-closed: the typed column contains only
totally-symmetric (A1 / Ag) excitations.
"""),
        code(T(r"""
const PROBE_BUDGET = 100     # pd_probe Davidson iterations per part
const RPA_K        = 4       # physical ω returned, with multiplicity
const RPA_TOL      = 1e-6
const RPA_MAX_ITER = 1200    # budget in (A, B) apply pairs (pair-equivalents)
const RPA_SEED     = 1

# Linear response at a REAL reference, whatever its stability. `rpa_signed` runs the paired
# Davidson — it never refuses — and returns SIGNED ω²: positive entries are excitations, negative
# ones resonances `ω = iκ`, each certified by fresh applies. Beside every mode it returns the
# Weingarten fraction `f_W = |⟨q,Bq⟩| / |⟨q,Aq⟩|` (see the prose below the cell).
function response(H, x, label)
    t0 = time()
    pr = QNTT.pd_probe(H, x; part = :rr, budget = PROBE_BUDGET, seed = RPA_SEED)
    pi_ = QNTT.pd_probe(H, x; part = :ii, budget = PROBE_BUDGET, seed = RPA_SEED)
    for (part, p) in ((":rr  (A + B)", pr), (":ii  (A − B)", pi_))
        @printf("  %-8s pd_probe %s  %-17s λ = %+.3e   residual = %-9s  %3d applies\n",
                label, part, p.verdict, p.lambda, isnan(p.residual) ? "n/a" : @sprintf("%.1e", p.residual), p.applies)
    end
    pr.is_pd === false && println("  $label   ‖ A + B has a negative witness: this is NOT a minimizer of the real energy")
    pi_.is_pd === false && println("  $label   ‖ A − B has a negative witness: complex-unstable real state — " *
                                   "the solve below reports it as a counting offset")
    empty = (; ω = Float64[], f_w = Float64[], resonances = Float64[], metric = :none, converged = false,
               iterations = 0, applies = 0, negatives_certified = true, pr, pi = pi_)
    res = nothing
    message = ""
    try
        res = QNTT.rpa_signed(H, x; k = RPA_K, tol = RPA_TOL, max_iter = RPA_MAX_ITER,
                              seed = RPA_SEED)
    catch err
        message = sprint(showerror, err)
        println("  $label   rpa_signed refused: $message")
    end
    res === nothing && return (; empty..., message, wall = time() - t0)
    ω2, info = res
    pos = ω2 .> 0
    ω, f_w, resonances = sqrt.(ω2[pos]), info.f_w[pos], sqrt.(-ω2[.!pos])
    @printf("  %-8s rpa_signed (%s kernel) — %d excitations, %d resonances%s; converged = %s, max residual %.1e, %d basis vectors, %d applies, %.1f s\n",
            label, info.kernel,
            length(ω), length(resonances),
            isempty(resonances) ? "" : (info.negatives_certified ? " (all certified)" : " (NOT all certified — do not read the iκ)"),
            info.converged, maximum(info.residuals; init = 0.0), info.iterations, info.applies, time() - t0)
    offset = something(info.report.neg_apb, 0) + something(info.report.neg_amb, 0)
    events = something(info.report.metric_events, 0)
    (offset > 0 || events > 0) && @printf("  %-8s ‖ indefinite point (counting offset %d, %d metric events, %d de-excitations, %d complex modes): `converged` does not license these ω\n",
                                          label, offset, events, count(<(0), info.krein), length(info.complex))
    isempty(resonances) || @printf("  %-8s resonances iκ, κ = %s\n", label, fmt_ω(resonances, RPA_K))
    flush(stdout)
    return (; ω, f_w, resonances, metric = info.metric, info.converged, info.iterations, info.applies,
              info.negatives_certified, pr, pi = pi_, message, wall = time() - t0)
end

R = Dict{String,Any}()
for dir in RUN
    s = S[dir]
    println(s.sys.label)
    # `rpasigned`: results of the never-refusing signed solver; a cache entry from the refusing
    # one (tag `rpa`) is not reused.
    tag_c1 = @sprintf("%s_%s_rpasigned_c1_eps%.0e_cap%d_k%d", dir, LAYER, DMRG_EPS, DMRG_CAP, RPA_K)
    tag_pg = @sprintf("%s_%s_rpasigned_pg_eps%.0e_cap%d_k%d", dir, LAYER, DMRG_EPS, DMRG_CAP, RPA_K)
    r_c1, how, t = cached_value(tag_c1, () -> response(s.H_c1, P[dir].c1, "untyped"))
    report_cache(tag_c1, how, t); flush(stdout)
    r_pg, how, t = cached_value(tag_pg, () -> response(s.H_pg, P[dir].pg, "typed"))
    report_cache(tag_pg, how, t); flush(stdout)
    R[dir] = (c1 = r_c1, pg = r_pg)
    for (label, r) in (("untyped", r_c1), ("typed A1", r_pg))
        @printf("  %-9s ω[1:%d]   = %s%s\n", label, RPA_K, fmt_ω(r.ω, RPA_K),
                isempty(r.ω) ? "" : (r.converged ? "" : "   (NOT at tolerance — lowest values settle first)"))
        @printf("  %-9s f_W[1:%d] = %s\n", label, RPA_K, fmt_fw(r.f_w, RPA_K))
    end
end
""")),
        md(r"""
### Table

Rows: the paper's two LR values (SS and SA printed alongside) and, for water, the FCI gaps.
Columns: the nearest mode of each method with its index, deviation and `f_W`. `—` where no
spectrum was published.
"""),
        code(T(r"""
for dir in RUN
    s, b, r = S[dir], B[dir], R[dir]
    println("\n", s.sys.label, "   (ω in Hartree; entry = nearest mode [index] deviation)")
    @printf("  %-26s %-34s %-34s %-34s\n", "reference", "CASSCF-LR (ours)", "DMRG untyped TT-RPA",
            "DMRG typed TT-RPA (A1 only)")
    keep = b.converged && b.status in (:stable, :resonant) ? (b.ω .> 0) : falses(length(b.ω))
    ω_cas, fw_cas = b.ω[keep], b.f_w[keep]
    rows = [(@sprintf("paper LR ω%d = %.6f", j, PAPER[s.sys.mol][j, 3]), PAPER[s.sys.mol][j, 3]) for j in 1:2]
    if s.fci_gaps !== nothing
        append!(rows, [(@sprintf("FCI gap %d   = %.6f", j, g), g) for (j, g) in enumerate(s.fci_gaps)])
    end
    for (name, g) in rows
        @printf("  %-26s %s  %s  %s\n", name, fmt_near(nearest(ω_cas, g), fw_cas),
                fmt_near(nearest(r.c1.ω, g), r.c1.f_w), fmt_near(nearest(r.pg.ω, g), r.pg.f_w))
    end
    @printf("  paper SS / SA for context: ω₁ = %.6f / %.6f   ω₂ = %.6f / %.6f\n",
            PAPER[s.sys.mol][1, 1], PAPER[s.sys.mol][1, 2], PAPER[s.sys.mol][2, 1], PAPER[s.sys.mol][2, 2])
    @printf("  full lists: CASSCF-LR %s | untyped %s | typed %s\n",
            fmt_ω(ω_cas, RPA_K), fmt_ω(r.c1.ω, RPA_K), fmt_ω(r.pg.ω, RPA_K))
    for (label, r) in (("untyped", r.c1), ("typed", r.pg))
        r.metric === :rr && @printf("  %s: H^ii indefinite at this real state — metric swapped to H^rr; resonances iκ, κ = %s (%s)\n",
                                    label, fmt_ω(r.resonances, RPA_K), r.negatives_certified ? "certified" : "NOT certified")
    end
end
""")),
        md(r"""
## 5. GAD

`saddle_seed` kicks the polished state by `GAD_KICK` along the softest `A + B` mode (radial mode
deflated). `gad` runs gentlest-ascent dynamics with the damped Weingarten model, grid preserved,
and certifies the index at the endpoint with the exact `A + B`. Reported: endpoint energy above
the ground state beside the nearest RPA ω of the same arm, `‖grad‖`, the index certificate,
steps, rejections, applies, `eig1`, `b_gain_max`, `min_schmidt_min`, wall time. On the typed
state only A1 / Ag saddles are reachable. Long GAD trajectories can end at different saddles on
different CPU architectures. `GAD_ON_CASSCF = true` also launches from the CASSCF point.
"""),
        code(T(r"""
const GAD_KICK         = 5e-2    # step along the softest A+B mode
const GAD_KICK_BUDGET  = 100     # Davidson budget for that mode
const GAD_STEPS        = 100     # accepted-step budget
const GAD_GRADTOL      = 1e-7
const GAD_PROBE_BUDGET = 100     # the index-1 certificate's Davidson budget
const GAD_SEED         = 0x6AD
const GAD_ON_CASSCF    = false   # also launch from the CASSCF point (re-runs the SCF; uncached)

function gad_demo(H, x, label)
    t0 = time()
    λ_gs = QNTT.rayleigh_quotient(H, x)
    xk, λ_witness = QNTT.saddle_seed(H, x; eps = GAD_KICK, budget = GAD_KICK_BUDGET, seed = GAD_SEED)
    g0 = QNTT.grad_norm(H, xk)
    @printf("  %-8s kick: softest A+B eigenvalue %.4f, ‖grad‖ after kick %.2e\n", label, λ_witness, g0)
    xe, info = QNTT.gad(H, xk; gradtol = GAD_GRADTOL, max_steps = GAD_STEPS, k_davidson = 8,
                        w_model = :damped, certify = true, probe_budget = GAD_PROBE_BUDGET,
                        preserve_grid = true, seed = GAD_SEED)
    λe = QNTT.rayleigh_quotient(H, xe)
    flush(stdout)
    return (; λ_gs, λ_witness, g0, λe, gap = λe - λ_gs, info..., wall = time() - t0)
end

function report_gad(g, label, ω_ref, paper_ω1)
    status = g.converged ? (g.index1_certified ? "converged, index-1 CERTIFIED" :
                            @sprintf("converged, index estimate %s (not certified)", string(g.index_estimate))) :
                           @sprintf("NOT converged in %d steps (‖grad‖ = %.2e)", g.steps, g.gnorm)
    @printf("  %-8s %s\n", label, status)
    @printf("  %-8s endpoint λ − λ_gs = %.6f   nearest RPA ω of this arm: %s   paper LR ω₁ = %.6f\n",
            label, g.gap, fmt_near(nearest(ω_ref, g.gap)), paper_ω1)
    @printf("  %-8s %d accepted / %d rejected steps, %d relatches, %d Hessian applies, eig1 = %s, b_gain_max = %.2e, min σ = %.2e, %.1f s\n",
            label, g.steps, g.rejected_steps, g.relatches, g.applies,
            g.eig1 === nothing ? "n/a" : @sprintf("%.4f", g.eig1), g.b_gain_max, g.min_schmidt_min, g.wall)
end

G = Dict{String,Any}()
for dir in RUN
    s = S[dir]
    println(s.sys.label)
    tag_c1 = @sprintf("%s_%s_gad_c1_eps%.0e_cap%d", dir, LAYER, DMRG_EPS, DMRG_CAP)
    tag_pg = @sprintf("%s_%s_gad_pg_eps%.0e_cap%d", dir, LAYER, DMRG_EPS, DMRG_CAP)
    g_c1, how, t = cached_value(tag_c1, () -> gad_demo(s.H_c1, P[dir].c1, "untyped"))
    report_cache(tag_c1, how, t); flush(stdout)
    g_pg, how, t = cached_value(tag_pg, () -> gad_demo(s.H_pg, P[dir].pg, "typed"))
    report_cache(tag_pg, how, t); flush(stdout)
    G[dir] = (c1 = g_c1, pg = g_pg)
    report_gad(g_c1, "untyped", R[dir].c1.ω, PAPER[s.sys.mol][1, 3])
    report_gad(g_pg, "typed", R[dir].pg.ω, PAPER[s.sys.mol][1, 3])
    if GAD_ON_CASSCF
        # The CASSCF point re-converged here (the baseline cell cached numbers, not the rotated operator).
        λ0, x0 = QNTT.cas_reference_state(s.H_c1, s.sys.cas; m_active = 0, seed = CASSCF_SEED)
        λ, x, u, H_rot, info = QNTT.casscf(s.data, x0, s.sys.cas; orb_max_iter = CASSCF_ORB_ITER,
                                           orb_grad = CASSCF_ORB_GRAD, tt_grad_tol = CASSCF_TT_GRAD,
                                           preserve_grid = true)
        info.converged || (println("  casscf   SCF did not certify — GAD skipped"); continue)
        g_cas = gad_demo(H_rot, x, "casscf")
        report_gad(g_cas, "casscf", B[dir].ω, PAPER[s.sys.mol][1, 3])
    end
end
""")),
        md(r"""
## Summary
"""),
        code(T(r"""
for dir in RUN
    s, b = S[dir], B[dir]
    println("\n", s.sys.label)
    @printf("  %-34s %.9f\n", "E_HF (fixture)", s.fx.e_hf)
    @printf("  %-34s %.9f\n", "CASSCF (pyscf, reference file)", s.casscf_gs)
    @printf("  %-34s %.9f   %s\n", "CASSCF-equivalent (ours)", b.E, b.converged ? "certified" : "NOT certified")
    for (label, H, y) in (("DMRG untyped, polished", s.H_c1, P[dir].c1), ("DMRG typed, polished", s.H_pg, P[dir].pg))
        @printf("  %-34s %.9f   pooled ranks %s\n", label, QNTT.rayleigh_quotient(H, y) + s.fx.e_nuc,
                string([sum(g) for g in QNTT.ranks(y)]))
    end
    isnan(s.fx.e_ref) || @printf("  %-34s %.9f\n", s.fci_gaps === nothing ? "E_ref (CCSD(T))" : "E_ref (FCI)", s.fx.e_ref)
    @printf("  %-34s %s\n", "ω: CASSCF-LR", fmt_ω(b.converged && b.status in (:stable, :resonant) ? b.ω : Float64[], RPA_K))
    @printf("  %-34s %s\n", "ω: DMRG untyped TT-RPA", fmt_ω(R[dir].c1.ω, RPA_K))
    @printf("  %-34s %s\n", "ω: DMRG typed TT-RPA (A1)", fmt_ω(R[dir].pg.ω, RPA_K))
    @printf("  %-34s %.6f / %.6f\n", "GAD gap untyped / typed", G[dir].c1.gap, G[dir].pg.gap)
end
""")),
        md(T(r"""
- ω are fixed-rank values at the ranks printed beside them, in the fixture orbitals.
- The typed column is A1 / Ag-only.
- Ranks on this layer are the «RANKWORD».
""")),
    ]
    return cells


notebook(paper_cells("abelian"), "04-casscf-vs-dmrg-abelian.ipynb")
notebook(paper_cells("su2"), "05-casscf-vs-dmrg-su2.ipynb")
