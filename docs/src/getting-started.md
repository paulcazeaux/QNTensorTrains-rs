# Getting started

## Build the library

The Julia package calls into a shared library built from the Rust workspace. You need a Rust
toolchain (≥ 1.85 — the crate is edition 2024):

```sh
cargo build --release -p qntt-capi
```

That produces `libqntt_capi.{dylib,so,dll}` in the cargo target directory. Nothing else is
required: the numerics are pure Rust, so there is no BLAS/LAPACK to install.

## Install the Julia package

```julia
using Pkg
Pkg.develop(path = "/path/to/qntt/julia")
using QNTT
QNTT.version()
```

`QNTT.jl` finds the library automatically from the cargo target directory. To pin it — for a
`QNTT_jll` artifact, or a build somewhere unusual:

```julia
QNTT.set_library_path!("/path/to/libqntt_capi.dylib")   # persistent, via Preferences
```

or set `ENV["QNTT_LIB"]` before `using QNTT` for a one-off. See [`QNTT.find_library`](@ref) for
the full search order.

### Distribution

`capi/build_tarballs.jl` is a BinaryBuilder recipe that cross-compiles the library for every
supported platform and can emit an unregistered `QNTT_jll`:

```sh
julia --project=. capi/build_tarballs.jl --verbose --deploy=local
```

Point the package at it with `QNTT.set_library_path!(QNTT_jll.libqntt_capi_path)`.

## Threads

The library runs its own thread pool, independent of Julia's:

```julia
QNTT.init(8)        # 0, or no argument, lets it choose
QNTT.nthreads()
```

Only the first call in a session has an effect. The library never calls back into Julia, so there
is no interaction with the Julia GC or its scheduler.

## A first calculation

```julia
using QNTT, LinearAlgebra

data  = QNTT.read_fcidump("testdata/h4.FCIDUMP")     # integrals + header
space = QNTT.QnSpace(data)                            # target sector from the header
H     = QNTT.Hamiltonian(space, data)                 # build the Hamiltonian

x0    = QNTT.tt_randn(space, QNTT.uniform_ranks(space, 8); seed = 1234)
λ, x  = QNTT.mals(H, x0; bond_dims = [0], eps = 1e-8)

@show λ + QNTT.e0(data)          # total electronic + core energy
@show norm(H * x - λ * x) / norm(x)
```

`x` is an ordinary Julia value. `norm(x)`, `dot(x, y)`, `H * x`, `2x - y`,
`round(x; eps = 1e-8)`, `QNTT.ranks(x)`, `Array(x)` (dense, small systems only) all work as you
would expect.

## Errors

Failed calls throw [`QNTTError`](@ref), carrying the Rust library's own message — including the
text of core-crate assertions, which are caught at the ABI boundary rather than aborting the
process:

```julia
julia> QNTT.mals(H, x0; bond_dims = Int[])
ERROR: QNTTError(2): mals bond_dims must be non-empty (use [0] for uncapped)
```

## Where to go next

- [Spaces, sectors and ranks](manual/spaces.md) — the symmetry bookkeeping, and how to choose a
  manifold.
- [Hamiltonians](manual/hamiltonians.md) — integrals, conventions, operator construction.
- [Ground states](manual/ground-states.md) — which solver to reach for.
- [Excited states](manual/excited-states.md) — state-average and state-specific DMRG, and the
  certified ladder.
- [Linear response](manual/linear-response.md) — excitation-energy estimates.
- [Tutorial](tutorial/index.md) — the numbered course of Rust examples, from a first ground state up.
- [Notebooks](notebooks.md) — worked examples on H₄, LiH, H₂O and N₂.
