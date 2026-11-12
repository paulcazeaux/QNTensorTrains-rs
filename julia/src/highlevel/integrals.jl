# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

# =================================================================================================
# Integrals
# =================================================================================================

"""
    read_fcidump([T,] path; symmetry = :eightfold, point_group = :auto) -> Integrals

Parse an FCIDUMP file. `symmetry` completes the two-body block: `:eightfold` (the usual real
chemists' integrals), `:pairswap` (transcorrelated dumps, hermiticity broken) or `:none`.

`point_group` picks the group `ORBSYM`/`ISYM` are read in: `:auto` (the default) guesses it from
`max(ORBSYM)` — `1 → :C1`, `2 → :Cs`, `3..4 → :C2v`, `5..8 → :D2h` — and anything else forces
that kind. **The guess affects irrep names only**: all same-order abelian groups share one
product table, so no integral, energy or selection rule depends on which one is named. The
integrals themselves are byte-identical either way.

On a genuinely symmetric dump, follow this with [`screen_forbidden!`](@ref) before building a
Hamiltonian over a point-group space.
"""
function read_fcidump(::Type{T}, path::AbstractString;
                      symmetry::Symbol = :eightfold,
                      point_group::Symbol = :auto) where {T}
    isfile(path) || throw(ArgumentError("no such FCIDUMP file: $path"))
    kind = point_group === :auto ? PG_AUTO : _code(POINT_GROUPS, point_group, "point group")
    return Integrals{T}(_fcidump_read_pg(T, path,
                                         _code(TWO_BODY_SYMMETRIES, symmetry, "symmetry"), kind))
end
read_fcidump(path::AbstractString; kwargs...) = read_fcidump(Float64, path; kwargs...)

"""
    orbsym(data) -> Vector{UInt8}

The dump's raw `ORBSYM`: **1-based Molpro ids**, one per orbital, in file order. Empty when the
header carried none.
"""
orbsym(data::Integrals{T}) where {T} = _fcidump_orbsym(T, _ptr(data))

"""
    isym(data) -> Int

The dump's raw `ISYM` (a 1-based Molpro id), `0` when absent or literally `ISYM=0` — both of
which mean "totally symmetric".
"""
isym(data::Integrals{T}) where {T} = _fcidump_point_group(T, _ptr(data))[2]

"""
    pointgroup(data) -> (kind = Symbol, isym = Int, target = Int)

The dump's resolved point group: the kind its ids were read in (explicit or guessed), the raw
`ISYM`, and the target irrep as a **0-based XOR index**.
"""
function pointgroup(data::Integrals{T}) where {T}
    kind, raw, target = _fcidump_point_group(T, _ptr(data))
    return (kind = POINT_GROUP_NAMES[kind], isym = raw, target = target)
end

for (name, i) in ((:norb, 1), (:nelec, 2), (:ms2, 3), (:e0, 4))
    @eval $name(d::Integrals{T}) where {T} = _fcidump_info(T, _ptr(d))[$i]
end

@doc "Orbital count of a parsed FCIDUMP." norb
@doc "Electron count of a parsed FCIDUMP." nelec
@doc "`2·Sz` of a parsed FCIDUMP." ms2
@doc "Core (nuclear-repulsion) energy of a parsed FCIDUMP." e0

"""
    onebody(data) -> Matrix

One-body integrals as a Julia array, `t[i,j]`, 1-based orbital indices. (Internally the library
keeps them row-major; this accessor transposes, so what you get back indexes the way Julia
expects.)
"""
function onebody(data::Integrals{T}) where {T}
    n = norb(data)
    return permutedims(reshape(_fcidump_t(T, _ptr(data)), n, n), (2, 1))
end

"""
    twobody(data) -> Array{T,4}

Two-body integrals as a Julia array, `v[i,j,k,l]`, 1-based orbital indices in the chemists'
`(ij|kl)` convention. As with [`onebody`](@ref), the row-major/column-major conversion is done
here.
"""
function twobody(data::Integrals{T}) where {T}
    n = norb(data)
    return permutedims(reshape(_fcidump_v(T, _ptr(data)), n, n, n, n), (4, 3, 2, 1))
end

"""
    QnSpace(data::Integrals) -> Space{Abelian}
    Su2Space(data::Integrals; two_s = nothing, point_group = :inherit) -> Space{SU2}

The target space implied by a parsed FCIDUMP's electron count and `2·Sz`. Both **inherit the
dump's `ORBSYM`/`ISYM`** — for a file with no point-group information the result is exactly the
trivial-group space it always was.

!!! warning "Behaviour change"
    `Su2Space(data)` now inherits the dump's `ORBSYM`/`ISYM`, as `QnSpace(data)` always has.
    Before this release the spin-adapted layer had no irrep axis and the symmetry was silently
    dropped. Pass `point_group = nothing` to restore the old behaviour. For a dump with no
    point-group information the two are identical.

`two_s` is the TOTAL `2S`, which an FCIDUMP does not carry (`MS2` is `2·Sz`, a projection); the
default `abs(ms2(data))` is the minimal total spin consistent with it — the value this
constructor has always used.

`point_group` accepts only `:inherit` (the default) or `nothing` **by design**: naming a
different same-order kind here would renumber `ORBSYM` without renumbering the already-resolved
`ISYM` target, which is a silent wrong answer. Choose the kind once, at
`read_fcidump(path; point_group = ...)`.

Screen a symmetry-adapted dump with [`screen_forbidden!`](@ref) before building a Hamiltonian
over the resulting space.
"""
function QnSpace(data::Integrals)
    pg = pointgroup(data)
    ob = orbsym(data)
    isempty(ob) && (ob = fill(0x01, norb(data)))
    return QnSpace(norb(data), nelec(data), ms2(data), Val(:from_sz);
                   point_group = pg.kind, orbsym = ob, target = pg.target)
end

function Su2Space(data::Integrals; two_s = nothing, point_group = :inherit)
    ts = two_s === nothing ? abs(ms2(data)) : Int(two_s)
    point_group === nothing && return Space{SU2}(norb(data), nelec(data), ts)
    point_group === :inherit ||
        throw(ArgumentError("point_group must be :inherit or nothing; choose the KIND at read " *
                            "time via read_fcidump(path; point_group = ...)"))
    pg = pointgroup(data)
    ob = orbsym(data)
    isempty(ob) && (ob = fill(0x01, norb(data)))
    return Space{SU2}(norb(data), nelec(data), ts;
                      point_group = pg.kind, orbsym = ob, target = pg.target)
end

"""
    screen_forbidden!(data; point_group = :auto) -> data

Project out the symmetry-forbidden `t`/`v` entries of a symmetry-adapted dump, in place.

A real symmetry-adapted FCIDUMP carries floating-point noise (~1e-14) on integrals its own
`ORBSYM` forbids. **This is REQUIRED before building a Hamiltonian over a non-trivial-group
space**: unscreened, that noise survives the construction tolerance, desynchronizes the operator
automaton's irrep bookkeeping, and the first `H*x` fails deep inside the kernel.

A no-op without a point group, and idempotent. An entry above `1e-8` is no longer plausibly noise
— that is a genuine `ORBSYM`/kind mismatch and throws rather than having your data quietly
altered.
"""
function screen_forbidden!(data::Integrals{T}; point_group::Symbol = :auto) where {T}
    kind = point_group === :auto ? PG_AUTO : _code(POINT_GROUPS, point_group, "point group")
    _fcidump_screen_pg(T, _ptr(data), kind)
    return data
end

"""
    read_tcdump([T,] path, norb; symmetry = :full48) -> ThreeBody

Parse a TCDUMP three-body (transcorrelated) file.
"""
function read_tcdump(::Type{T}, path::AbstractString, n::Integer;
                     symmetry::Symbol = :full48) where {T}
    isfile(path) || throw(ArgumentError("no such TCDUMP file: $path"))
    return ThreeBody{T}(_tcdump_read(T, path, n,
                                     _code(THREE_BODY_SYMMETRIES, symmetry, "symmetry")))
end
read_tcdump(path::AbstractString, n::Integer; kwargs...) =
    read_tcdump(Float64, path, n; kwargs...)

"""
    threebody([T,] norb) -> ThreeBody

Empty sparse three-body coefficient tensor; fill it with [`setcoo!`](@ref).
"""
threebody(::Type{T}, n::Integer) where {T} = ThreeBody{T}(_w_new(T, n))
threebody(n::Integer) = threebody(Float64, n)

"""
    setcoo!(w, idx, vals)

Bulk fill: `idx` is `6×n` of **0-based** orbital indices in creation/annihilation slot order,
`vals` the matching coefficients.
"""
function setcoo!(w::ThreeBody{T}, idx::AbstractMatrix{<:Integer},
                 vals::AbstractVector) where {T}
    _w_set_coo!(T, _ptr(w), Matrix{Csize_t}(idx), T[convert(T, v) for v in vals])
    return w
end

nnz(w::ThreeBody{T}) where {T} = _w_nnz(T, _ptr(w))

