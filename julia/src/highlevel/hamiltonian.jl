# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

# =================================================================================================
# Hamiltonians
# =================================================================================================

# The C API takes row-major flat buffers; Julia arrays are column-major, so a reversed
# `permutedims` produces the required order. (For real 8-fold-symmetric `v` the reversal happens
# to be the identity — do not rely on it: a transcorrelated `w`, or a complex/`:pairswap` `v`,
# is not symmetric under it.)
_rowmajor(t::AbstractMatrix{T}) where {T} = vec(permutedims(t, (2, 1)))
_rowmajor(v::AbstractArray{T,4}) where {T} = vec(permutedims(v, (4, 3, 2, 1)))

"""
    Hamiltonian(space, t, v; w = nothing, eps = eps(Float64), construction = :default)
    Hamiltonian(space, data::Integrals; ...)

Compile the second-quantized operator into its action automaton.

`t[i,j]` and `v[i,j,k,l]` are ordinary 1-based Julia arrays (chemists' `(ij|kl)`); `w` is an
optional [`ThreeBody`](@ref) transcorrelated term. `construction` selects the automaton
construction on the abelian layer — `:default`, `:median`, `:bipartite`, `:bipartite_per_sector`
— and must be `:default` on the SU(2) layer, which has only one.

The `Integrals` form passes the parsed buffers straight through without a layout round trip. On a
symmetric dump over a non-trivial-group `space`, call [`screen_forbidden!`](@ref) on the
`Integrals` first.
"""
function Hamiltonian(s::Space{L}, t::AbstractMatrix{T}, v::AbstractArray{T,4};
                     w::Union{Nothing,ThreeBody{T}} = nothing,
                     eps::Real = eps(Float64),
                     construction::Symbol = :default) where {T,L}
    size(t) == (s.d, s.d) ||
        throw(ArgumentError("one-body tensor must be $(s.d)×$(s.d), got $(size(t))"))
    all(==(s.d), size(v)) ||
        throw(ArgumentError("two-body tensor must be $(s.d)^4, got $(size(v))"))
    wp = w === nothing ? C_NULL : _ptr(w)
    return Hamiltonian{T,L}(_ham_new_routed(s, T, _rowmajor(t), _rowmajor(v), wp, Float64(eps),
                                            _code(CONSTRUCTIONS, construction, "construction")))
end

# The one place the point-group routing for operators lives: a trivial space goes to the legacy
# symbol verbatim (so the default path is byte-for-byte what it was), anything else to `_pg`.
function _ham_new_routed(s::Space{L}, ::Type{T}, tflat, vflat, wp, eps, cons) where {T,L}
    istrivial(s) && return _ham_new(L, T, s.d, s.a, s.b, tflat, vflat, wp, eps, cons)
    return _ham_new_pg(L, T, s.d, s.a, s.b, POINT_GROUPS[s.kind], s.orbsym, s.target,
                       tflat, vflat, wp, eps, cons)
end

function Hamiltonian(s::Space{L}, data::Integrals{T};
                     w::Union{Nothing,ThreeBody{T}} = nothing,
                     eps::Real = eps(Float64),
                     construction::Symbol = :default) where {T,L}
    wp = w === nothing ? C_NULL : _ptr(w)
    return Hamiltonian{T,L}(_ham_new_routed(s, T, _fcidump_t(T, _ptr(data)),
                                            _fcidump_v(T, _ptr(data)), wp, Float64(eps),
                                            _code(CONSTRUCTIONS, construction, "construction")))
end

function Base.show(io::IO, h::Hamiltonian{T,L}) where {T,L}
    print(io, "Hamiltonian{", T, ",", nameof(L), "}(", space(h), ")")
end

"""
    H * x -> TTVector

Exact (unrounded) `H·x`. Its ranks are the product of `x`'s and the operator's, so round it in an
iterative loop.
"""
Base.:*(h::Hamiltonian{T,L}, x::TTVector{T,L}) where {T,L} =
    TTVector{T,L}(_ham_apply(L, T, _ptr(h), _ptr(x)))

"""
    expectation(H, x) -> T

`⟨x, H·x⟩`, **not** divided by `⟨x, x⟩`. Use [`rayleigh_quotient`](@ref) unless `x` is known to
be unit-norm.
"""
expectation(h::Hamiltonian{T,L}, x::TTVector{T,L}) where {T,L} =
    _ham_expectation(L, T, _ptr(h), _ptr(x))

"""
    rayleigh_quotient(H, x) -> T

`⟨x, H·x⟩ / ⟨x, x⟩`, safe on an unnormalized train.
"""
rayleigh_quotient(h::Hamiltonian{T,L}, x::TTVector{T,L}) where {T,L} =
    _ham_rayleigh(L, T, _ptr(h), _ptr(x))

"""
    xthy(x, H, y) -> T

`⟨x, H·y⟩`, contracted without materializing `H·y`.
"""
xthy(x::TTVector{T,L}, h::Hamiltonian{T,L}, y::TTVector{T,L}) where {T,L} =
    _ham_xthy(L, T, _ptr(x), _ptr(h), _ptr(y))

"""
    automaton_bond_dims(H) -> Vector{Int}

Summed-over-sector operator bond dimensions, one per bond (abelian layer).
"""
automaton_bond_dims(h::Hamiltonian{T,L}) where {T,L} = _ham_bond_dims(L, T, _ptr(h))

