# Hamiltonians

## From an FCIDUMP

```julia
data  = QNTT.read_fcidump("system.FCIDUMP")     # symmetry = :eightfold by default
space = QNTT.QnSpace(data)
H     = QNTT.Hamiltonian(space, data)
```

[`QNTT.read_fcidump`](@ref) parses the header (`NORB`, `NELEC`, `MS2`) and the integral records,
completing the two-body block under the requested permutation group: `:eightfold` for ordinary
real chemists' integrals, `:pairswap` for transcorrelated dumps (where hermiticity is broken but
the pair-swap relation still holds), `:none` to keep only explicit entries.

```julia
QNTT.norb(data), QNTT.nelec(data), QNTT.ms2(data), QNTT.e0(data)
t = QNTT.onebody(data)      # Matrix, t[i,j]
v = QNTT.twobody(data)      # Array{T,4}, v[i,j,k,l], chemists' (ij|kl)
```

`QNTT.e0(data)` is the core (nuclear-repulsion) energy: the solvers return the **electronic**
energy, so the total is `λ + e0`.

## From arrays

Integrals computed elsewhere — PySCF, a model Hamiltonian, a modified `v` — go in directly:

```julia
H = QNTT.Hamiltonian(space, t, v)
```

`t` and `v` are ordinary 1-based Julia arrays: `t[1,2]` is the coefficient of a†₀a₁ on 0-based
sites. The wrapper handles the C ABI's row-major layout.

## Operator construction

The Hamiltonian is built exactly: no screening threshold decides the answer.

```julia
H = QNTT.Hamiltonian(space, data; construction = :bipartite)
QNTT.automaton_bond_dims(H)     # summed-over-sector operator bond dimensions
```

| `construction` | |
|---|---|
| `:default` | the layer's default (`:bipartite` on the abelian layer) |
| `:median` | the median construction |
| `:bipartite` | the bipartite construction (Ren et al., 2020) — lower peak memory |
| `:bipartite_per_sector` | a slightly smaller operator, at a higher construction cost |

All of them build the *same* operator; they differ only in its internal layout and in
construction cost. The SU(2) layer has a single construction and only accepts `:default`.

Building the operator is a one-time cost that grows with the orbital count; at scale, time it
before assuming a run is slow for numerical reasons.

## Three-body / transcorrelated terms

```julia
w = QNTT.read_tcdump("system.TCDUMP", QNTT.norb(data))
H = QNTT.Hamiltonian(space, data; w = w)
```

or build one directly from a COO list (0-based indices, creation/annihilation slot order):

```julia
w = QNTT.threebody(norb)
QNTT.setcoo!(w, idx, vals)      # idx is 6×n
```

## Acting with it

```julia
Hx = H * x                              # exact matvec, unrounded
QNTT.rayleigh_quotient(H, x)            # ⟨x,Hx⟩ / ⟨x,x⟩
QNTT.expectation(H, x)                  # ⟨x,Hx⟩, NOT normalized
QNTT.xthy(x, H, y)                      # ⟨x,Hy⟩, without materializing Hy
```

`H * x` has ranks equal to the product of the state's and the operator's, so in an iterative loop
you round it with `round(y; eps = …)`.
