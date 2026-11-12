# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

"""
    QNTT

Julia bindings for [`qntt`](https://github.com/paulcazeaux/QNTensorTrains-rs) — quantum-number-
preserving block-sparse tensor trains for fermionic many-body systems.

Two symmetry layers are exposed, each for real and complex scalars: the abelian `(Nup, Ndn)`
stack ([`QnSpace`](@ref)) and the SU(2) spin-adapted stack ([`Su2Space`](@ref)).

```julia
using QNTT

data  = QNTT.read_fcidump("testdata/h4.FCIDUMP")
space = QNTT.QnSpace(data)
H     = QNTT.Hamiltonian(space, data)
x0    = QNTT.tt_randn(space, QNTT.uniform_ranks(space, 8); seed = 1)
λ, x  = QNTT.mals(H, x0; bond_dims = [0], eps = 1e-8)
E     = λ + QNTT.e0(data)          # + core energy
```

The shared library is located, in order, from `ENV["QNTT_LIB"]`, then a `release` and a `debug`
build under the crate's cargo target directory. Build it with

```sh
cargo build --release -p qntt-capi
```

Everything is 0-based, matching the Rust library: sites `0:d-1`, bonds `0:d`, sector labels
`(a, b)` counting electrons strictly to the left of a bond. Integral arrays are the exception —
`t[i,j]` and `v[i,j,k,l]` are ordinary 1-based Julia arrays.

Abelian point-group symmetry is available on both layers (`QnSpace(...; point_group)` and
`Su2Space(...; point_group)`; the group is always abelian, the layer need not be). `orbsym` goes **in** as the 1-based Molpro ids an FCIDUMP writes; irreps come
**out** as 0-based XOR indices, since they are sector labels — convert with
[`QNTT.irrep_from_molpro`](@ref) / [`QNTT.molpro_from_irrep`](@ref). `QNTT.read_fcidump` picks up
`ORBSYM`/`ISYM` automatically and `QNTT.QnSpace(data)` inherits them; which same-order group is
named affects irrep names only.
"""
module QNTT

using Libdl
using LinearAlgebra
import Preferences

export QNTTError

"""Supertype of the symmetry-layer tags [`Abelian`](@ref) and [`SU2`](@ref)."""
abstract type Layer end

# --- error type ---------------------------------------------------------------------------------

"""
    QNTTError(code, msg)
    QNTTError(code, msg, refusal)

A failed library call. `code` is the C status (1 null argument, 2 invalid argument, 3 a caught
Rust panic — typically a core-crate assertion, 4 I/O, 5 did not converge).

`refusal` DISCRIMINATES a `code == 5` physics refusal of the cross-sector observables
([`sector_alpha`](@ref), [`sector_transition_moments`](@ref)): the wire code
[`QNTT.SECTOR_REFUSALS`](@ref) names, so a caller can branch on *which* refusal happened without
parsing `msg`. It is `0` (`:none`) everywhere else — every other call, and every CALLER error,
which is a statement about the arguments and not about the state. The two-argument constructor is
the one every other call site uses.
"""
struct QNTTError <: Exception
    code::Int
    msg::String
    refusal::Int32
end
QNTTError(code, msg) = QNTTError(code, msg, Int32(0))

function Base.showerror(io::IO, e::QNTTError)
    print(io, "QNTTError(", e.code, "): ", e.msg)
    e.refusal == 0 || print(io, " [refusal ", e.refusal, " = ",
                            get(SECTOR_REFUSALS, Cint(e.refusal), :unknown), "]")
    return nothing
end

const _STATUS = Dict(
    1 => "null argument",
    2 => "invalid argument",
    3 => "panic in the Rust library",
    4 => "I/O error",
    5 => "did not converge",
)

# `check(status)`: throw a `QNTTError` carrying the library's own message unless `status` is
# success. Internal: every wrapper routes its C status through it.
function check(status::Integer)
    status == 0 && return nothing
    msg = last_error()
    isempty(msg) && (msg = get(_STATUS, Int(status), "unknown status $status"))
    throw(QNTTError(Int(status), msg))
end

# `check(status, refusal)`: `check` for the cross-sector observables, whose physics refusals
# cross as `QNTT_NOT_CONVERGED` (5) plus a machine-readable discriminant in the C `out_refusal`.
# The code travels on the thrown `QNTTError`'s `refusal` field, so the caller does not have to
# parse the message. The C entries write `out_refusal` on every path INCLUDING caller errors
# (`:none` there), and only status 5 is a statement about the state, so only status 5 carries the
# code on.
function check(status::Integer, refusal::Integer)
    status == 0 && return nothing
    msg = last_error()
    isempty(msg) && (msg = get(_STATUS, Int(status), "unknown status $status"))
    throw(QNTTError(Int(status), msg, Int32(status == 5 ? refusal : 0)))
end

# --- library discovery --------------------------------------------------------------------------

"""
    QNTT.set_library_path!(path)
    QNTT.set_library_path!(nothing)

Pin the shared library QNTT loads, persistently, via `Preferences`. Changing it invalidates the
precompilation cache, so the new path takes effect on the next `using QNTT` — which is exactly
why this is preferred over `ENV["QNTT_LIB"]` for a permanent choice.

Point it at a `QNTT_jll` build:

```julia
using QNTT_jll
QNTT.set_library_path!(QNTT_jll.libqntt_capi_path)
```

Pass `nothing` to clear it and go back to automatic discovery.
"""
function set_library_path!(path::Union{AbstractString,Nothing})
    if path === nothing
        Preferences.delete_preferences!(@__MODULE__, "library_path"; force = true)
    else
        isfile(path) || throw(ArgumentError("not a file: $path"))
        Preferences.set_preferences!(@__MODULE__, "library_path" => abspath(path);
                                     force = true)
    end
    @info "QNTT library preference updated; restart Julia for it to take effect."
    return nothing
end

"""
    QNTT.find_library() -> String

Absolute path of the `qntt` shared library, searched in this order:

1. the `library_path` preference ([`set_library_path!`](@ref)) — this is where a `QNTT_jll`
   artifact goes;
2. `ENV["QNTT_LIB"]` (a file, or a directory containing the library);
3. the cargo target directory reported by `cargo metadata` (`release`, then `debug`);
4. the conventional `target/` directory next to the crate.
"""
function find_library()
    ext = Sys.isapple() ? "dylib" : (Sys.iswindows() ? "dll" : "so")
    stem = Sys.iswindows() ? "qntt_capi" : "libqntt_capi"
    name = "$stem.$ext"

    pref = Preferences.@load_preference("library_path", nothing)
    if pref !== nothing
        isfile(pref) && return abspath(pref)
        error("""
              the QNTT `library_path` preference points at "$pref", which does not exist.
              Fix it with QNTT.set_library_path!(newpath), or clear it with
              QNTT.set_library_path!(nothing).
              """)
    end

    if haskey(ENV, "QNTT_LIB")
        p = ENV["QNTT_LIB"]
        isfile(p) && return abspath(p)
        isfile(joinpath(p, name)) && return abspath(joinpath(p, name))
        error("QNTT_LIB is set to \"$p\", which is neither $name nor a directory containing it")
    end

    root = normpath(joinpath(@__DIR__, "..", ".."))
    candidates = String[]
    # Honour a repo-level `target-dir` override (this project keeps artifacts outside the source
    # tree), falling back to the conventional location.
    try
        meta = read(Cmd(`cargo metadata --no-deps --format-version 1`; dir = root), String)
        m = match(r"\"target_directory\":\"(.*?)\"", meta)
        m === nothing || push!(candidates, String(m.captures[1]))
    catch
        # cargo not on PATH: fall through to the conventional path only.
    end
    push!(candidates, joinpath(root, "target"))

    for dir in candidates, profile in ("release", "debug")
        p = joinpath(dir, profile, name)
        isfile(p) && return p
    end

    error("""
          could not find $name.

          Build it with

              cargo build --release -p qntt-capi

          from $root, or point QNTT_LIB at the library (or its directory).
          """)
end

const libqntt = find_library()

include("options.jl")
include("lowlevel.jl")
include("highlevel.jl")

# The library path is resolved at precompile time and baked into the cache. If the build moves
# afterwards (a rebuild into a different target directory, an artifact garbage-collected), say so
# once and clearly instead of failing later inside a `ccall`.
function __init__()
    if !isfile(libqntt)
        @warn """
              the qntt shared library recorded at precompile time is gone:
                  $libqntt
              Rebuild it (`cargo build --release -p qntt-capi`), then force a recompile with
              `Pkg.precompile()`, or pin a path with `QNTT.set_library_path!(...)`.
              """
    end
    return nothing
end

end # module
