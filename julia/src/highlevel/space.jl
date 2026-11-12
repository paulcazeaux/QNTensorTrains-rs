# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

# =================================================================================================
# Spaces
# =================================================================================================

"""
    Space{L}(d, a, b; point_group = :C1, orbsym = nothing, target = 0)

Symmetry descriptor. Build one with [`QnSpace`](@ref) (abelian, `a = Nup`, `b = Ndn`) or
[`Su2Space`](@ref) (spin-adapted, `a = N` electrons, `b = 2S`). Everything is 0-based, matching
the Rust library: sites `0:d-1`, bonds `0:d`, sector labels count electrons strictly to the left
of a bond.

`point_group`/`orbsym`/`target` add point-group symmetry (see [`QnSpace`](@ref)); the default is
the trivial group, in which case every call routes to the non-`_pg` C symbols and the behaviour is
exactly what it was before point groups existed. **Both layers accept a point group** — the group
itself is always abelian (`:C1`…`:D2h`), the layer need not be.
"""
struct Space{L<:Layer}
    d::Int
    a::Int
    b::Int
    kind::Symbol            # point-group LABEL: numerics depend on |G|, not on which group
    orbsym::Vector{UInt8}   # 1-based Molpro ids, one per site (all ones under :C1)
    target::Int             # target irrep, 0-based XOR index
    function Space{L}(d::Integer, a::Integer, b::Integer;
                      point_group::Symbol = :C1, orbsym = nothing,
                      target::Integer = 0) where {L<:Layer}
        haskey(POINT_GROUPS, point_group) ||
            throw(ArgumentError("point_group must be one of $(sort(collect(keys(POINT_GROUPS)))), got :$point_group"))
        ob = orbsym === nothing ? fill(0x01, Int(d)) : Vector{UInt8}(orbsym)
        length(ob) == d ||
            throw(ArgumentError("orbsym must have d = $d entries, got $(length(ob))"))
        trivial = point_group === :C1 && all(==(0x01), ob) && target == 0
        if trivial
            check(_space_check(L, d, a, b))
        else
            check(_space_check_pg(L, d, a, b, POINT_GROUPS[point_group], ob, target))
        end
        return new{L}(Int(d), Int(a), Int(b), point_group, ob, Int(target))
    end
end

"""
    istrivial(space) -> Bool

Whether the space carries no point-group symmetry — the case in which every high-level call
routes to the legacy (non-`_pg`) C symbols.
"""
istrivial(s::Space) = s.kind === :C1 && all(==(0x01), s.orbsym) && s.target == 0

"""
    nirrep(space) -> Int

Order `|G|` of the space's point group (`1` without one). The rank grids gain a third axis when
this exceeds 1.
"""
nirrep(s::Space) = istrivial(s) ? 1 : _pg_order(POINT_GROUPS[s.kind])

"""
    QnSpace(d, nup, ndn; point_group = :C1, orbsym = nothing, target = 0)
    QnSpace(d, nelec, ms2, Val(:from_sz); ...)
    QnSpace(data::Integrals)

Abelian `(Nup, Ndn)`-conserving space over `d` orbitals. The second form splits a total electron
count and `2·Sz` the way the Rust `QnSpace::from_n_two_sz` does; the third inherits the parsed
FCIDUMP's `ORBSYM`/`ISYM`.

For point-group symmetry pass `point_group` (`:C1`…`:D2h`), `orbsym` as `d` **1-based Molpro
ids** exactly as an FCIDUMP writes them, and `target` as a **0-based XOR** irrep index — convert
an `ISYM` with [`irrep_from_molpro`](@ref). Which same-order group you name affects irrep NAMES
only; all same-order abelian groups share one product table.
"""
QnSpace(d::Integer, nup::Integer, ndn::Integer; kwargs...) =
    Space{Abelian}(d, nup, ndn; kwargs...)

function QnSpace(d::Integer, nelec::Integer, ms2::Integer, ::Val{:from_sz}; kwargs...)
    nup, ndn = from_n_two_sz(d, nelec, ms2)
    return Space{Abelian}(d, nup, ndn; kwargs...)
end

"""
    Su2Space(d, n, two_s; point_group = :C1, orbsym = nothing, target = 0)

SU(2) spin-adapted space: `n` electrons coupled to total spin `S = two_s/2` over `d` orbitals.

Point-group symmetry is carried exactly as on the abelian layer (SPG6): pass `point_group`
(`:C1`…`:D2h`), `orbsym` as `d` **1-based Molpro ids** exactly as an FCIDUMP writes them, and
`target` as a **0-based XOR** irrep index — convert an `ISYM` with
[`irrep_from_molpro`](@ref). Which same-order group you name affects irrep NAMES only. A bond
sector is then `(n, 2s, ir)`, and `2S` is a TOTAL spin, not the `MS2` projection an FCIDUMP
carries.
"""
Su2Space(d::Integer, n::Integer, two_s::Integer; kwargs...) = Space{SU2}(d, n, two_s; kwargs...)

function Base.show(io::IO, s::Space{Abelian})
    print(io, "QnSpace(d=", s.d, ", Nup=", s.a, ", Ndn=", s.b)
    istrivial(s) || print(io, ", ", s.kind, ", target=", irrep_name(s, s.target))
    print(io, ")")
end
function Base.show(io::IO, s::Space{SU2})
    print(io, "Su2Space(d=", s.d, ", N=", s.a, ", 2S=", s.b)
    istrivial(s) || print(io, ", ", s.kind, ", target=", irrep_name(s, s.target))
    print(io, ")")
end

# A `Space` is a VALUE — two descriptors built separately for the same symmetry are the same
# space. This has to be spelled out: the point-group fields made the struct non-`isbits` (the
# `orbsym::Vector{UInt8}`), and Julia's default `==` on such a struct is object egality, so two
# `QnSpace(4, 2, 2)` would compare unequal purely because their `orbsym` vectors are distinct
# objects. `hash` matches field-for-field so a `Space` can key a `Dict`.
function Base.:(==)(x::Space{L}, y::Space{L}) where {L<:Layer}
    return x.d == y.d && x.a == y.a && x.b == y.b &&
           x.kind === y.kind && x.orbsym == y.orbsym && x.target == y.target
end
Base.:(==)(::Space, ::Space) = false      # abelian and spin-adapted are never the same space
Base.hash(s::Space{L}, h::UInt) where {L<:Layer} =
    hash(L, hash(s.d, hash(s.a, hash(s.b, hash(s.kind, hash(s.orbsym, hash(s.target, h)))))))

"""
    bond_sectors(space, bond) -> Vector{Tuple{Int,Int}}
                             -> Vector{Tuple{Int,Int,Int}}   (with a point group)

Admissible sector labels at `bond` (`0:d`), in the library's canonical order — the order block
iteration and every rank grid use. With a point group each label gains its 0-based XOR irrep.
"""
bond_sectors(s::Space{L}, bond::Integer) where {L} =
    istrivial(s) ? _bond_sectors(L, s.d, s.a, s.b, bond) :
    _bond_sectors_pg(L, s.d, s.a, s.b, POINT_GROUPS[s.kind], s.orbsym, s.target, bond)

"""
    grid_extent(space) -> (rows, cols)

Shape of one per-bond rank grid: `(Nup+1, Ndn+1)` abelian, `(N+1, N+1)` SU(2). The irrep axis, if
any, is reported by [`nirrep`](@ref).
"""
grid_extent(s::Space{L}) where {L} = _grid_extent(L, s.d, s.a, s.b)

"""
    maxranks(space) -> Vector{Matrix{Int}}
                    -> Vector{Array{Int,3}}   (with a point group)

Maximal admissible rank at every `(bond, sector)` — the full-rank manifold. Index a grid as
`g[a+1, b+1]` for sector `(a, b)`, or `g[a+1, b+1, ir+1]` when the space has a point group.
"""
maxranks(s::Space{L}) where {L} =
    istrivial(s) ? _unflatten_ranks(s, _max_ranks_flat(L, s.d, s.a, s.b)) :
    _unflatten_ranks(s, _max_ranks_flat_pg(L, s.d, s.a, s.b, POINT_GROUPS[s.kind],
                                           s.orbsym, s.target))

"""
    uniform_ranks(space, r) -> Vector{Matrix{Int}}

Per-bond rank grids capped at `r`, i.e. `min(r, maxranks)`. The usual way to pick a starting
manifold: `tt_randn(space, uniform_ranks(space, 20))`.
"""
uniform_ranks(s::Space, r::Integer) = [min.(g, r) for g in maxranks(s)]

function _unflatten_ranks(s::Space, flat::Vector{Csize_t})
    rows, cols = grid_extent(s)
    nir = nirrep(s)
    nb = length(flat) ÷ (rows * cols * nir)
    # The flat layout is bond-major, then irrep, with the SECOND sector axis fastest — so a
    # `(cols, rows, nir, nb)` reshape lands transposed in the first two axes; permute back to
    # `[a+1, b+1(, ir+1)]` indexing. At `|G| = 1` the irrep axis is dropped entirely, so the
    # return type is exactly what it was before point groups existed.
    a = permutedims(reshape(Int.(flat), cols, rows, nir, nb), (2, 1, 3, 4))
    return nir == 1 ? [a[:, :, 1, k] for k in 1:nb] : [a[:, :, :, k] for k in 1:nb]
end

function _flatten_ranks(s::Space, grids::AbstractVector)
    rows, cols = grid_extent(s)
    nir = nirrep(s)
    length(grids) == s.d + 1 ||
        throw(ArgumentError("need d+1 = $(s.d + 1) rank grids, got $(length(grids))"))
    # A plain matrix at `|G| = 1`, an `(rows, cols, |G|)` array beyond — the shapes
    # `_unflatten_ranks` hands back.
    want = nir == 1 ? (rows, cols) : (rows, cols, nir)
    flat = Vector{Csize_t}(undef, (s.d + 1) * rows * cols * nir)
    i = 1
    for g in grids
        size(g) == want ||
            throw(ArgumentError("rank grid must be $(join(want, "×")), got $(size(g))"))
        gg = reshape(g, rows, cols, nir)
        for ir in 1:nir, r in 1:rows, c in 1:cols
            flat[i] = Csize_t(gg[r, c, ir])
            i += 1
        end
    end
    return flat
end

# =================================================================================================
# Point-group tables
# =================================================================================================

_pg_code(kind) = kind isa Symbol ? _code(POINT_GROUPS, kind, "point group") : Cint(kind)

"""
    pg_order(kind) -> Int

Number of irreps of a point-group kind (`:C1`…`:D2h`): 1, 2, 4 or 8.
"""
pg_order(kind) = _pg_order(_pg_code(kind))

"""
    irrep_name(kind_or_space, irrep) -> String

Name of an irrep given by its **0-based XOR index**, e.g. `irrep_name(:C2v, 3) == "B2"`.
"""
irrep_name(kind, irrep::Integer) = _pg_irrep_name(_pg_code(kind), irrep)
irrep_name(s::Space, irrep::Integer) = _pg_irrep_name(_pg_code(s.kind), irrep)

"""
    irrep_from_molpro(kind, id) -> Int

The **0-based XOR index** of the irrep an FCIDUMP calls `id` (a **1-based Molpro id**, as in
`ORBSYM` and `ISYM`). This is the single conversion between the two numberings; never shift by
hand.
"""
irrep_from_molpro(kind, id::Integer) = _pg_irrep_from_molpro(_pg_code(kind), id)

"""
    molpro_from_irrep(kind, irrep) -> Int

The **1-based Molpro id** of a **0-based XOR** irrep index — the inverse of
[`irrep_from_molpro`](@ref).
"""
molpro_from_irrep(kind, irrep::Integer) = _pg_molpro_from_irrep(_pg_code(kind), irrep)

"""
    orbsym_to_irreps(kind, orbsym) -> Vector{UInt8}

Map a whole `ORBSYM` run of 1-based Molpro ids to 0-based XOR irreps, one per site.
"""
orbsym_to_irreps(kind, orbsym) = _pg_orbsym_to_irreps(_pg_code(kind), orbsym)

