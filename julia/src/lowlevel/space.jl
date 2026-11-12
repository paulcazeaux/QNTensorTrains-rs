# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

"""Symmetry layer tag: the abelian `(Nup, Ndn)`-conserving stack."""
struct Abelian <: Layer end

"""Symmetry layer tag: the SU(2) spin-adapted stack."""
struct SU2 <: Layer end

const _LAYERS = ((Abelian, ""), (SU2, "su2_"))
const _SCALARS = ((Float64, "f64"), (ComplexF64, "c64"))

# --- global / scalar-independent -------------------------------------------------------------

_last_error_len() = (len = Ref{Csize_t}(0);
    ccall((:qntt_last_error, libqntt), Cint, (Ptr{UInt8}, Csize_t, Ptr{Csize_t}), C_NULL, 0, len);
    Int(len[]))

function last_error()
    len = _last_error_len()
    len == 0 && return ""
    buf = Vector{UInt8}(undef, len + 1)
    ccall((:qntt_last_error, libqntt), Cint, (Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
          buf, length(buf), Ref{Csize_t}(0))
    return String(buf[1:len])
end

"""
    QNTT.version() -> VersionNumber

Version of the loaded `qntt` shared library.
"""
function version()
    len = Ref{Csize_t}(0)
    ccall((:qntt_version, libqntt), Cint, (Ptr{UInt8}, Csize_t, Ptr{Csize_t}), C_NULL, 0, len)
    buf = Vector{UInt8}(undef, Int(len[]) + 1)
    ccall((:qntt_version, libqntt), Cint, (Ptr{UInt8}, Csize_t, Ptr{Csize_t}), buf, length(buf), len)
    return VersionNumber(String(buf[1:Int(len[])]))
end

"""
    QNTT.init(nthreads = 0)

Size the library's global thread pool (`0` lets it choose). Only the first call in a session has
an effect.
"""
init(nthreads::Integer = 0) =
    check(ccall((:qntt_init, libqntt), Cint, (Csize_t,), nthreads))

"""
    QNTT.nthreads() -> Int

Threads in the library's global pool.
"""
function nthreads()
    out = Ref{Csize_t}(0)
    check(ccall((:qntt_num_threads, libqntt), Cint, (Ptr{Csize_t},), out))
    return Int(out[])
end

# --- space queries (per layer) ----------------------------------------------------------------

for (L, lp) in _LAYERS
    p = "qntt_$(lp)space_"
    @eval begin
        _space_check(::Type{$L}, d, a, b) = ccall(
            ($(QuoteNode(Symbol(p * "check"))), libqntt), Cint,
            (Csize_t, Csize_t, Csize_t), d, a, b)

        function _grid_extent(::Type{$L}, d, a, b)
            rows, cols = Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(QuoteNode(Symbol(p * "grid_extent"))), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Ptr{Csize_t}, Ptr{Csize_t}),
                        d, a, b, rows, cols))
            return (Int(rows[]), Int(cols[]))
        end

        function _bond_sectors(::Type{$L}, d, a, b, bond)
            len = Ref{Csize_t}(0)
            check(ccall(($(QuoteNode(Symbol(p * "bond_sectors"))), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Csize_t, Ptr{Csize_t}, Ptr{Csize_t},
                         Csize_t, Ptr{Csize_t}),
                        d, a, b, bond, C_NULL, C_NULL, 0, len))
            n = Int(len[])
            xa, xb = Vector{Csize_t}(undef, n), Vector{Csize_t}(undef, n)
            check(ccall(($(QuoteNode(Symbol(p * "bond_sectors"))), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Csize_t, Ptr{Csize_t}, Ptr{Csize_t},
                         Csize_t, Ptr{Csize_t}),
                        d, a, b, bond, xa, xb, n, len))
            return [(Int(xa[i]), Int(xb[i])) for i in 1:n]
        end

        function _max_ranks_flat(::Type{$L}, d, a, b)
            len = Ref{Csize_t}(0)
            check(ccall(($(QuoteNode(Symbol(p * "max_ranks"))), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
                        d, a, b, C_NULL, 0, len))
            flat = Vector{Csize_t}(undef, Int(len[]))
            check(ccall(($(QuoteNode(Symbol(p * "max_ranks"))), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
                        d, a, b, flat, length(flat), len))
            return flat
        end
    end
end

# --- point group: tables and the `_pg` space queries -------------------------------------------
#
# The GROUP is always abelian (`:C1`…`:D2h`); the LAYER need not be — both layers carry it (SPG6).
# 1:1 with `capi/include/qntt.h`'s point-group block. `orbsym` is passed as a plain
# `Vector{UInt8}` of 1-BASED MOLPRO ids: `ccall` roots array arguments for the duration of the
# call, so no `GC.@preserve` is needed anywhere here (the descriptor is loose arguments, never a
# struct holding a raw pointer). Irreps returned by the library are 0-BASED XOR indices.

_pg_order(kind) = (out = Ref{Csize_t}(0);
    check(ccall((:qntt_pg_order, libqntt), Cint, (Cint, Ptr{Csize_t}), Cint(kind), out));
    Int(out[]))

_pg_default_kind(order) = (out = Ref{Cint}(0);
    check(ccall((:qntt_pg_default_kind, libqntt), Cint, (Csize_t, Ptr{Cint}), order, out));
    out[])

function _pg_kind_from_name(name::AbstractString)
    out = Ref{Cint}(0)
    check(ccall((:qntt_pg_kind_from_name, libqntt), Cint, (Cstring, Ptr{Cint}), name, out))
    return out[]
end

# The `*_name` entry points copy exactly `out_len` bytes and NUL-terminate only when the buffer
# has room, so `out_len + 1` is always safe.
function _pg_kind_name(kind)
    len = Ref{Csize_t}(0)
    check(ccall((:qntt_pg_kind_name, libqntt), Cint,
                (Cint, Ptr{UInt8}, Csize_t, Ptr{Csize_t}), Cint(kind), C_NULL, 0, len))
    buf = Vector{UInt8}(undef, Int(len[]) + 1)
    check(ccall((:qntt_pg_kind_name, libqntt), Cint,
                (Cint, Ptr{UInt8}, Csize_t, Ptr{Csize_t}), Cint(kind), buf, length(buf), len))
    return String(buf[1:Int(len[])])
end

function _pg_irrep_name(kind, irrep)
    len = Ref{Csize_t}(0)
    check(ccall((:qntt_pg_irrep_name, libqntt), Cint,
                (Cint, Cint, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
                Cint(kind), Cint(irrep), C_NULL, 0, len))
    buf = Vector{UInt8}(undef, Int(len[]) + 1)
    check(ccall((:qntt_pg_irrep_name, libqntt), Cint,
                (Cint, Cint, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
                Cint(kind), Cint(irrep), buf, length(buf), len))
    return String(buf[1:Int(len[])])
end

_pg_irrep_from_molpro(kind, id) = (out = Ref{Cint}(0);
    check(ccall((:qntt_pg_irrep_from_molpro, libqntt), Cint, (Cint, Cint, Ptr{Cint}),
                Cint(kind), Cint(id), out));
    Int(out[]))

_pg_molpro_from_irrep(kind, irrep) = (out = Ref{Cint}(0);
    check(ccall((:qntt_pg_molpro_from_irrep, libqntt), Cint, (Cint, Cint, Ptr{Cint}),
                Cint(kind), Cint(irrep), out));
    Int(out[]))

function _pg_orbsym_to_irreps(kind, orbsym)
    ob = Vector{UInt8}(orbsym)
    out = Vector{UInt8}(undef, length(ob))
    len = Ref{Csize_t}(0)
    check(ccall((:qntt_pg_orbsym_to_irreps, libqntt), Cint,
                (Cint, Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
                Cint(kind), ob, length(ob), out, length(out), len))
    return out
end

# The `_pg` space queries, one family per layer: `qntt_space_*_pg` (abelian) and
# `qntt_su2_space_*_pg` (SU(2), SPG6). Same descriptor quadruple on both; the space triple is
# `(d, Nup, Ndn)` abelian and `(d, N, 2S)` on SU(2), and a sector label gains its 0-based XOR irrep.
for (L, lp) in _LAYERS
    p = "qntt_$(lp)space_"
    @eval begin
        _space_check_pg(::Type{$L}, d, a, b, kind, orbsym, target) = ccall(
            ($(QuoteNode(Symbol(p * "check_pg"))), libqntt), Cint,
            (Csize_t, Csize_t, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint),
            d, a, b, Cint(kind), Vector{UInt8}(orbsym), length(orbsym), Cint(target))

        function _grid_extent_pg(::Type{$L}, d, a, b, kind, orbsym, target)
            ob = Vector{UInt8}(orbsym)
            rows, cols, nir = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(QuoteNode(Symbol(p * "grid_extent_pg"))), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint,
                         Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}),
                        d, a, b, Cint(kind), ob, length(ob), Cint(target), rows, cols, nir))
            return (Int(rows[]), Int(cols[]), Int(nir[]))
        end

        function _bond_sectors_pg(::Type{$L}, d, a, b, kind, orbsym, target, bond)
            ob = Vector{UInt8}(orbsym)
            len = Ref{Csize_t}(0)
            check(ccall(($(QuoteNode(Symbol(p * "bond_sectors_pg"))), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint, Csize_t,
                         Ptr{Csize_t}, Ptr{Csize_t}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
                        d, a, b, Cint(kind), ob, length(ob), Cint(target), bond,
                        C_NULL, C_NULL, C_NULL, 0, len))
            n = Int(len[])
            xa, xb = Vector{Csize_t}(undef, n), Vector{Csize_t}(undef, n)
            xi = Vector{UInt8}(undef, n)
            check(ccall(($(QuoteNode(Symbol(p * "bond_sectors_pg"))), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint, Csize_t,
                         Ptr{Csize_t}, Ptr{Csize_t}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
                        d, a, b, Cint(kind), ob, length(ob), Cint(target), bond, xa, xb, xi, n, len))
            return [(Int(xa[i]), Int(xb[i]), Int(xi[i])) for i in 1:n]
        end

        function _max_ranks_flat_pg(::Type{$L}, d, a, b, kind, orbsym, target)
            ob = Vector{UInt8}(orbsym)
            len = Ref{Csize_t}(0)
            check(ccall(($(QuoteNode(Symbol(p * "max_ranks_pg"))), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint,
                         Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
                        d, a, b, Cint(kind), ob, length(ob), Cint(target), C_NULL, 0, len))
            flat = Vector{Csize_t}(undef, Int(len[]))
            check(ccall(($(QuoteNode(Symbol(p * "max_ranks_pg"))), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint,
                         Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
                        d, a, b, Cint(kind), ob, length(ob), Cint(target), flat, length(flat), len))
            return flat
        end
    end
end

"""
    QNTT.from_n_two_sz(d, nelec, ms2) -> (Nup, Ndn)

Split an electron count and `2·Sz` into the abelian layer's per-spin counts.
"""
function from_n_two_sz(d::Integer, n::Integer, two_sz::Integer)
    nup, ndn = Ref{Csize_t}(0), Ref{Csize_t}(0)
    check(ccall((:qntt_space_from_n_two_sz, libqntt), Cint,
                (Csize_t, Csize_t, Int64, Ptr{Csize_t}, Ptr{Csize_t}),
                d, n, two_sz, nup, ndn))
    return (Int(nup[]), Int(ndn[]))
end
