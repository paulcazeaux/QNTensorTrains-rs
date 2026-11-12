# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

# --- the f64 -> C64 half: the state cast and the complex seed ------------------------------------
#
# Per LAYER only, not per (layer, scalar): the reverse direction would have to discard an imaginary
# part, and there is no complex-to-real seed to build. Both return a ComplexF64 handle, so the
# caller must wrap them in `TTVector{ComplexF64,L}`.

for (L, lp) in _LAYERS
    p = "qntt_$(lp)f64_"
    sym(op) = QuoteNode(Symbol(p * op))
    @eval begin
        function _vec_to_c64(::Type{$L}, x)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("vec_to_c64")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), x, out))
            return out[]
        end

        # Probes H^ii at the REAL state `x` and retracts `i*eps*ŵ` along the witness. Throws
        # (QNTT_NOT_CONVERGED) when the probe finds no negative direction — seeding along an
        # unproven direction is the null result, not a finding. Also returns the witness λ.
        function _complex_seed(::Type{$L}, h, x, eps, budget, seed)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            lambda = Ref{Float64}(NaN)
            check(ccall(($(sym("complex_seed")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Csize_t, UInt64, Ptr{Float64},
                         Ptr{Ptr{Cvoid}}),
                        h, x, Float64(eps), budget, seed, lambda, out))
            return (out[], lambda[])
        end
    end
end

# --- abelian-only extras -------------------------------------------------------------------------

for (T, ss) in _SCALARS
    p = "qntt_$(ss)_"
    sym(op) = QuoteNode(Symbol(p * op))
    @eval begin
        function _tt_state(::Type{$T}, occ_up, occ_dn)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            up = Cint[b ? 1 : 0 for b in occ_up]
            dn = Cint[b ? 1 : 0 for b in occ_dn]
            length(up) == length(dn) ||
                throw(ArgumentError("occ_up and occ_dn must have the same length"))
            check(ccall(($(sym("tt_state")), libqntt), Cint,
                        (Ptr{Cint}, Ptr{Cint}, Csize_t, Ptr{Ptr{Cvoid}}),
                        up, dn, length(up), out))
            return out[]
        end

        # ---- abelian-only point-group twins: a DETERMINANT (`tt_state_pg`) and the
        # `(Nup, Ndn, target)` block indicator (`tt_ones_pg`). The SU(2) counterpart of the first
        # is `_tt_csf_pg` below; the second has none (see `capi/src/su2_pointgroup.rs`).
        function _tt_state_pg(::Type{$T}, occ_up, occ_dn, kind, orbsym, target)
            up = Cint[b ? 1 : 0 for b in occ_up]
            dn = Cint[b ? 1 : 0 for b in occ_dn]
            length(up) == length(dn) ||
                throw(ArgumentError("occ_up and occ_dn must have the same length"))
            ob = Vector{UInt8}(orbsym)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("tt_state_pg")), libqntt), Cint,
                        (Ptr{Cint}, Ptr{Cint}, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint,
                         Ptr{Ptr{Cvoid}}),
                        up, dn, length(up), Cint(kind), ob, length(ob), Cint(target), out))
            return out[]
        end

        function _tt_ones_pg(::Type{$T}, d, a, b, kind, orbsym, target)
            ob = Vector{UInt8}(orbsym)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("tt_ones_pg")), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint,
                         Ptr{Ptr{Cvoid}}),
                        d, a, b, Cint(kind), ob, length(ob), Cint(target), out))
            return out[]
        end

        function _to_dense(::Type{$T}, x)
            len = Ref{Csize_t}(0)
            check(ccall(($(sym("vec_to_dense")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{$T}, Csize_t, Ptr{Csize_t}), x, C_NULL, 0, len))
            v = Vector{$T}(undef, Int(len[]))
            check(ccall(($(sym("vec_to_dense")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{$T}, Csize_t, Ptr{Csize_t}), x, v, length(v), len))
            return v
        end

        # ---- integrals ----
        _fcidump_free(::Type{$T}, ptr) =
            ccall(($(sym("fcidump_free")), libqntt), Cvoid, (Ptr{Cvoid},), ptr)
        _w_free(::Type{$T}, ptr) =
            ccall(($(sym("w_free")), libqntt), Cvoid, (Ptr{Cvoid},), ptr)

        function _fcidump_read(::Type{$T}, path, symmetry)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("fcidump_read")), libqntt), Cint,
                        (Cstring, Cint, Ptr{Ptr{Cvoid}}), path, symmetry, out))
            return out[]
        end

        # `pg_kind` is a wire code, or `PG_AUTO` (-1) to guess from `max(ORBSYM)`. The integrals
        # are byte-identical to `_fcidump_read`'s on the same file; only the kind the handle
        # remembers for `_fcidump_point_group` differs.
        function _fcidump_read_pg(::Type{$T}, path, symmetry, pg_kind)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("fcidump_read_pg")), libqntt), Cint,
                        (Cstring, Cint, Cint, Ptr{Ptr{Cvoid}}),
                        path, symmetry, Cint(pg_kind), out))
            return out[]
        end

        # Project out (or reject) the integrals the dump's own ORBSYM forbids, IN PLACE.
        # `pg_kind` is a wire code or `PG_AUTO`. No-op at |G| = 1; idempotent; a forbidden entry
        # above 1e-8 is a genuine ORBSYM/kind mismatch and throws.
        _fcidump_screen_pg(::Type{$T}, h, pg_kind) =
            check(ccall(($(sym("fcidump_screen_pg")), libqntt), Cint,
                        (Ptr{Cvoid}, Cint), h, Cint(pg_kind)))

        # RAW 1-based Molpro ids as parsed; empty when the header carried no ORBSYM.
        function _fcidump_orbsym(::Type{$T}, h)
            len = Ref{Csize_t}(0)
            check(ccall(($(sym("fcidump_orbsym")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}), h, C_NULL, 0, len))
            buf = Vector{UInt8}(undef, Int(len[]))
            check(ccall(($(sym("fcidump_orbsym")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
                        h, buf, length(buf), len))
            return buf
        end

        # (resolved kind code, RAW ISYM, 0-based XOR target).
        function _fcidump_point_group(::Type{$T}, h)
            kind, target = Ref{Cint}(0), Ref{Cint}(0)
            isym = Ref{Int64}(0)
            check(ccall(($(sym("fcidump_point_group")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cint}, Ptr{Int64}, Ptr{Cint}), h, kind, isym, target))
            return (kind[], Int(isym[]), Int(target[]))
        end

        function _fcidump_info(::Type{$T}, h)
            norb, nelec = Ref{Csize_t}(0), Ref{Csize_t}(0)
            ms2, e0 = Ref{Int64}(0), Ref{$T}(zero($T))
            check(ccall(($(sym("fcidump_info")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Int64}, Ptr{$T}),
                        h, norb, nelec, ms2, e0))
            return (Int(norb[]), Int(nelec[]), Int(ms2[]), e0[])
        end

        function _fcidump_t(::Type{$T}, h)
            len = Ref{Csize_t}(0)
            check(ccall(($(sym("fcidump_t")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{$T}, Csize_t, Ptr{Csize_t}), h, C_NULL, 0, len))
            buf = Vector{$T}(undef, Int(len[]))
            check(ccall(($(sym("fcidump_t")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{$T}, Csize_t, Ptr{Csize_t}), h, buf, length(buf), len))
            return buf
        end

        function _fcidump_v(::Type{$T}, h)
            len = Ref{Csize_t}(0)
            check(ccall(($(sym("fcidump_v")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{$T}, Csize_t, Ptr{Csize_t}), h, C_NULL, 0, len))
            buf = Vector{$T}(undef, Int(len[]))
            check(ccall(($(sym("fcidump_v")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{$T}, Csize_t, Ptr{Csize_t}), h, buf, length(buf), len))
            return buf
        end

        function _tcdump_read(::Type{$T}, path, norb, symmetry)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("tcdump_read")), libqntt), Cint,
                        (Cstring, Csize_t, Cint, Ptr{Ptr{Cvoid}}), path, norb, symmetry, out))
            return out[]
        end

        function _w_new(::Type{$T}, norb)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("w_new")), libqntt), Cint,
                        (Csize_t, Ptr{Ptr{Cvoid}}), norb, out))
            return out[]
        end

        function _w_set_coo!(::Type{$T}, w, idx::Matrix{Csize_t}, vals::Vector{$T})
            size(idx, 1) == 6 || throw(ArgumentError("idx must be 6×n"))
            size(idx, 2) == length(vals) || throw(ArgumentError("idx and vals disagree"))
            check(ccall(($(sym("w_set_coo")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{$T}, Csize_t),
                        w, idx, vals, length(vals)))
            return nothing
        end

        function _w_nnz(::Type{$T}, w)
            out = Ref{Csize_t}(0)
            check(ccall(($(sym("w_nnz")), libqntt), Cint, (Ptr{Cvoid}, Ptr{Csize_t}), w, out))
            return Int(out[])
        end
    end
end

# --- SU(2)-only extras ---------------------------------------------------------------------------

for (T, ss) in _SCALARS
    p = "qntt_su2_$(ss)_"
    @eval begin
        function _tt_csf(::Type{$T}, codes)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            c = Cint[Cint(v) for v in codes]
            check(ccall(($(QuoteNode(Symbol(p * "tt_csf"))), libqntt), Cint,
                        (Ptr{Cint}, Csize_t, Ptr{Ptr{Cvoid}}), c, length(c), out))
            return out[]
        end

        # `tt_state_pg`'s SU(2) counterpart: `(N, 2S)` are inferred from the genealogical walk and
        # the total irrep is DERIVED from it — `target` is still passed, and a mismatch throws.
        function _tt_csf_pg(::Type{$T}, codes, kind, orbsym, target)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            c = Cint[Cint(v) for v in codes]
            ob = Vector{UInt8}(orbsym)
            check(ccall(($(QuoteNode(Symbol(p * "tt_csf_pg"))), libqntt), Cint,
                        (Ptr{Cint}, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint, Ptr{Ptr{Cvoid}}),
                        c, length(c), Cint(kind), ob, length(ob), Cint(target), out))
            return out[]
        end
    end
end

# --- the joint (CASSCF-style) stack, the saddle seed and GAD: per LAYER, f64 only ---------------

function _casscf_diag_series(diag, which::Cint)
    len = Ref{Csize_t}(0)
    check(ccall((:qntt_casscf_diag_series, libqntt), Cint,
                (Ptr{Cvoid}, Cint, Ptr{Float64}, Csize_t, Ptr{Csize_t}), diag, which, C_NULL, 0, len))
    out = Vector{Float64}(undef, Int(len[]))
    isempty(out) && return out
    check(ccall((:qntt_casscf_diag_series, libqntt), Cint,
                (Ptr{Cvoid}, Cint, Ptr{Float64}, Csize_t, Ptr{Csize_t}),
                diag, which, out, length(out), len))
    return out
end

function _casscf_diag_reason(diag)
    len = Ref{Csize_t}(0)
    check(ccall((:qntt_casscf_diag_reason, libqntt), Cint,
                (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}), diag, C_NULL, 0, len))
    len[] == 0 && return nothing
    buf = Vector{UInt8}(undef, Int(len[]) + 1)
    check(ccall((:qntt_casscf_diag_reason, libqntt), Cint,
                (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}), diag, buf, length(buf), len))
    return String(buf[1:Int(len[])])
end

_casscf_diag_free(diag) = ccall((:qntt_casscf_diag_free, libqntt), Cvoid, (Ptr{Cvoid},), diag)

# --- label marshalling ---------------------------------------------------------------------------
#
# `Vector{Cstring}` of `pointer(b)` values holds NO reference to the byte arrays it points into, so
# BOTH halves must be returned and BOTH must sit inside the caller's `GC.@preserve`: a collection
# between building the vector and the ccall would hand the library freed bytes. That is the classic
# Julia FFI bug and it is why this is a named helper rather than an inline `pointer.(labels)`.
#
# `nothing` is the documented default (`p0`, `p1`, `p2` on the C side); an individual `nothing`
# entry defaults that one index, so a caller who wants to name only the interesting component does
# not have to name all three.
function _cstring_vec(labels, n::Integer)
    labels === nothing && return (C_NULL, nothing)
    length(labels) == n ||
        throw(ArgumentError("labels must have one entry per property component ($n), got \
                             $(length(labels))"))
    keep = [s === nothing ? UInt8[] : push!(Vector{UInt8}(codeunits(String(s))), 0x00)
            for s in labels]
    ptrs = Cstring[isempty(b) ? Cstring(C_NULL) : Cstring(pointer(b)) for b in keep]
    return (ptrs, keep)
end

for (L, lp) in _LAYERS
    p = "qntt_$(lp)f64_"
    sym(op) = QuoteNode(Symbol(p * op))
    @eval begin
        function _cas_reference_state(::Type{$L}, h, part::CasPartition, m_active, seed)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            lambda = Ref{Float64}(NaN)
            check(ccall(($(sym("cas_reference_state")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{CasPartition}, Csize_t, UInt64, Ptr{Float64},
                         Ptr{Ptr{Cvoid}}),
                        h, Ref(part), m_active, seed, lambda, out))
            return (lambda[], out[])
        end

        # Returns (λ, x_ptr, h_ptr, u_flat_rowmajor, info, diag_ptr). The caller owns all three
        # handles; `diag` is layer-independent and freed with `_casscf_diag_free`.
        function _casscf(::Type{$L}, ints, x0, part::CasPartition, opts::CasscfOptions, d)
            lambda = Ref{Float64}(NaN)
            x, h, diag = Ref{Ptr{Cvoid}}(C_NULL), Ref{Ptr{Cvoid}}(C_NULL), Ref{Ptr{Cvoid}}(C_NULL)
            u = Vector{Float64}(undef, d * d)
            ulen = Ref{Csize_t}(0)
            info = Ref{CasscfInfo}()
            check(ccall(($(sym("casscf")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{CasPartition}, Ptr{CasscfOptions},
                         Ptr{Float64}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Float64}, Csize_t,
                         Ptr{Csize_t}, Ptr{CasscfInfo}, Ptr{Ptr{Cvoid}}),
                        ints, x0, Ref(part), Ref(opts), lambda, x, h, u, length(u), ulen, info,
                        diag))
            return (lambda[], x[], h[], u, info[], diag[])
        end

        function _joint_rpa(::Type{$L}, h, x, part::CasPartition, tt_grad_tol, orb_floor)
            len, fwlen, status = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Cint}(-1)
            n_modes, n_eff = Ref{Csize_t}(0), Ref{Csize_t}(0)
            msg = Vector{UInt8}(undef, 1024)
            mlen = Ref{Csize_t}(0)
            check(ccall(($(sym("joint_rpa")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{CasPartition}, Float64, Float64, Ptr{Float64},
                         Csize_t, Ptr{Csize_t}, Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Cint},
                         Ptr{Csize_t}, Ptr{Csize_t}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
                        h, x, Ref(part), tt_grad_tol, orb_floor, C_NULL, 0, len, C_NULL, 0, fwlen,
                        status, n_modes, n_eff, msg, length(msg), mlen))
            omega = Vector{Float64}(undef, Int(len[]))
            fw = Vector{Float64}(undef, Int(fwlen[]))
            if !isempty(omega) || !isempty(fw)
                check(ccall(($(sym("joint_rpa")), libqntt), Cint,
                            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{CasPartition}, Float64, Float64, Ptr{Float64},
                             Csize_t, Ptr{Csize_t}, Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Cint},
                             Ptr{Csize_t}, Ptr{Csize_t}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
                            h, x, Ref(part), tt_grad_tol, orb_floor, omega, length(omega), len,
                            fw, length(fw), fwlen, status, n_modes, n_eff, msg, length(msg), mlen))
            end
            n = min(Int(mlen[]), length(msg) - 1)
            return (omega, fw, status[], Int(n_modes[]), Int(n_eff[]), String(msg[1:n]))
        end

        # The returned length is (#negatives + k), known only after the solve, so every buffer
        # is sized by the header's basis bound n_max = max(max_iter, k + 2) (both kernels, in
        # pair-equivalents): no call can fail on room after paying for the solve. `kcode` is a
        # QNTT_RPA_KERNEL_* request code (RPA_KERNEL_CODES); `kernel` is the echo.
        function _rpa_signed(::Type{$L}, h, x, k, opts::SymplecticOptions, kcode, metric,
                             curvature_kind, curvature_value, seed)
            nmax = max(Int(opts.max_iter), Int(k) + 2)
            omega2 = Vector{Float64}(undef, nmax)
            fw = Vector{Float64}(undef, nmax)
            res = Vector{Float64}(undef, nmax)
            krein = Vector{Float64}(undef, nmax)
            cx = Vector{Float64}(undef, 3nmax)
            report = Vector{Int64}(undef, 5)
            len, fwlen, reslen = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
            kreinlen, cxlen, replen = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
            met, cert, conv, kernel = Ref{Cint}(0), Ref{Cint}(0), Ref{Cint}(0), Ref{Cint}(0)
            iters, applies = Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(sym("rpa_signed")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Ptr{SymplecticOptions}, Cint, Cint, Cint,
                         Float64, UInt64, Ptr{Float64}, Csize_t, Ptr{Csize_t},
                         Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Float64}, Csize_t, Ptr{Csize_t},
                         Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Float64}, Csize_t, Ptr{Csize_t},
                         Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Cint},
                         Ptr{Int64}, Csize_t, Ptr{Csize_t}),
                        h, x, k, Ref(opts), kcode, metric, curvature_kind, curvature_value, seed,
                        omega2, nmax, len, fw, nmax, fwlen, res, nmax, reslen, krein, nmax,
                        kreinlen, cx, 3nmax, cxlen, met, cert, conv, iters, applies, kernel,
                        report, 5, replen))
            ncx = Int(cxlen[]) ÷ 3
            cplx = [(re = cx[3j - 2], im = cx[3j - 1], residual = cx[3j]) for j in 1:ncx]
            return (omega2[1:Int(len[])], fw[1:Int(fwlen[])], res[1:Int(reslen[])],
                    krein[1:Int(kreinlen[])], cplx, met[], cert[] != 0, conv[] != 0,
                    Int(iters[]), Int(applies[]), Int(kernel[]), report[1:Int(replen[])])
        end

        # The same-sector Tamm-Dancoff readout. At most `k` pairs come back and the C side checks
        # every buffer against `k` before the solve, so `k` sizes them all. `states = true` asks
        # for the explicit Ritz states (new handles, owned by the caller); the slots past the
        # returned count stay NULL.
        function _tda(::Type{$L}, h, x, k, opts::TdaOptions, seed, states::Bool)
            n = Int(k)
            theta = Vector{Float64}(undef, n)
            res = Vector{Float64}(undef, n)
            slots = fill(Ptr{Cvoid}(C_NULL), n)
            len, reslen, slen = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
            conv, cert, pre = Ref{Cint}(0), Ref{Cint}(0), Ref{Cint}(0)
            iters, applies, dim = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(sym("tda")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Ptr{TdaOptions}, UInt64,
                         Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Float64}, Csize_t, Ptr{Csize_t},
                         Ptr{Ptr{Cvoid}}, Csize_t, Ptr{Csize_t}, Ptr{Cint}, Ptr{Cint},
                         Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Cint}),
                        h, x, k, Ref(opts), seed, theta, n, len, res, n, reslen,
                        states ? slots : Ptr{Ptr{Cvoid}}(C_NULL), n, slen, conv, cert, iters,
                        applies, dim, pre))
            return (theta[1:Int(len[])], res[1:Int(reslen[])], slots[1:Int(slen[])], conv[] != 0,
                    cert[] != 0, Int(iters[]), Int(applies[]), Int(dim[]), pre[])
        end

        # --- cross-sector (shifted-channel) response ------------------------------------------
        # No probe pass. On `:tda` the output is at most `k` values and the per-mode (RPA-only)
        # buffers come back empty, so `k` sizes the value buffer. On `:rpa` the signed list is
        # every negative omega^2 plus the `k` lowest non-negative ones — known only after the
        # solve — so every per-mode buffer is sized by the header's basis bound
        # n_max = max(max_iter, k + 2) (`3 n_max` for the complex rows): no call can fail on room
        # after paying for the solve. An empty `vals` is a real answer, told apart from a failure
        # by `(conv, dim)` — see the `sector_rpa` docstring.
        function _sector_rpa(::Type{$L}, h_ref, h_out, x, phi, k, opts::SectorOptions,
                             curvature_kind, curvature_value, precond, precond_kappa, seed)
            tda = opts.readout == SECTOR_READOUTS[:tda]
            nmax = max(Int(opts.max_iter), Int(k) + 2)
            nval = tda ? Int(k) : nmax
            nmode = tda ? 0 : nmax
            vals = Vector{Float64}(undef, nval)
            fw = Vector{Float64}(undef, nmode)
            res = Vector{Float64}(undef, nmode)
            krein = Vector{Float64}(undef, nmode)
            cx = Vector{Float64}(undef, 3nmode)
            report = Vector{Int64}(undef, 5)
            len, fwlen, reslen = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
            kreinlen, cxlen, dim = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
            replen = Ref{Csize_t}(0)
            metric, cert, conv = Ref{Cint}(0), Ref{Cint}(0), Ref{Cint}(0)
            iters, applies = Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(sym("sector_rpa")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt8, Csize_t, Ptr{SectorOptions},
                         Cint, Float64, Cint, Float64, UInt64,
                         Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Float64}, Csize_t, Ptr{Csize_t},
                         Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Float64}, Csize_t, Ptr{Csize_t},
                         Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Cint}, Ptr{Cint},
                         Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Cint}, Ptr{Int64}, Csize_t,
                         Ptr{Csize_t}),
                        h_ref, h_out, x, UInt8(phi), k, Ref(opts), curvature_kind,
                        curvature_value, precond, precond_kappa, seed,
                        vals, length(vals), len, fw, length(fw), fwlen, res, length(res), reslen,
                        krein, length(krein), kreinlen, cx, length(cx), cxlen, dim, metric, conv,
                        iters, applies, cert, report, 5, replen))
            n = min(Int(len[]), length(vals))
            ncx = min(Int(cxlen[]), length(cx)) ÷ 3
            cplx = [(re = cx[3j - 2], im = cx[3j - 1], residual = cx[3j]) for j in 1:ncx]
            return (vals[1:n], Int(dim[]), metric[], conv[] != 0, Int(iters[]), Int(applies[]),
                    cert[] != 0, fw[1:min(Int(fwlen[]), nmode)], res[1:min(Int(reslen[]), nmode)],
                    krein[1:min(Int(kreinlen[]), nmode)], cplx, report[1:Int(replen[])])
        end

        # The dense oracle PROBES, as `_rpa_dense` does — and the probe pass pays the full dense
        # solve, so this is two solves. Fine at the scale a dense oracle is for.
        function _sector_rpa_dense(::Type{$L}, h_ref, h_out, x, phi)
            len, dim = Ref{Csize_t}(0), Ref{Csize_t}(0)
            stable, metric = Ref{Cint}(0), Ref{Cint}(0)
            check(ccall(($(sym("sector_rpa_dense")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt8, Ptr{Float64}, Csize_t,
                         Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Cint}, Ptr{Cint}),
                        h_ref, h_out, x, UInt8(phi), C_NULL, 0, len, dim, stable, metric))
            w2 = Vector{Float64}(undef, Int(len[]))
            if !isempty(w2)
                check(ccall(($(sym("sector_rpa_dense")), libqntt), Cint,
                            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt8, Ptr{Float64}, Csize_t,
                             Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Cint}, Ptr{Cint}),
                            h_ref, h_out, x, UInt8(phi), w2, length(w2), len, dim, stable, metric))
            end
            return (w2, Int(dim[]), Int(stable[]), metric[])
        end

        # --- cross-sector OBSERVABLES ---------------------------------------------------------
        # `m` is a flat row-major n_props x norb x norb buffer built by the caller (the high level
        # uses `_rowmajor`); `labels` is `nothing` or a vector of strings. GC.@preserve keeps the
        # matrix buffer AND every label's own byte array alive across the ccall -- a
        # Vector{Cstring} of `pointer(s)` values holds NO reference to the strings themselves,
        # which is the one way this binding can be silently wrong (the label-lifetime testset is
        # what would notice).
        function _sector_alpha(::Type{$L}, h_ref, h_out, x, phi, m::Vector{Float64},
                               n_props::Integer, norb::Integer, labels,
                               opts::SectorAlphaOptions)
            np = Int(n_props)
            alpha = Vector{Float64}(undef, np * np)
            props = Vector{Float64}(undef, np * 3)
            iters = Vector{Csize_t}(undef, np)
            alen, plen, ilen = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
            info = Ref{SectorAlphaInfo}()
            refusal = Ref{Cint}(-99)  # sentinel: the ABI writes NONE (0) on every non-NULL path
            lab, keep = _cstring_vec(labels, np)
            # The status is CAPTURED and checked AFTER the preserve block, not inside it: a
            # physics refusal carries its discriminant in `refusal[]`, and `check` must see both.
            # Throwing from inside the block would also unwind through `GC.@preserve`.
            st = GC.@preserve m alpha props iters lab keep begin
                ccall(($(sym("sector_alpha")), libqntt), Cint,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt8,
                       Ptr{Float64}, Csize_t, Csize_t, Csize_t,
                       Ptr{Cstring}, Ptr{SectorAlphaOptions},
                       Ptr{Float64}, Csize_t, Ptr{Csize_t},
                       Ptr{Float64}, Csize_t, Ptr{Csize_t},
                       Ptr{Csize_t}, Csize_t, Ptr{Csize_t},
                       Ptr{SectorAlphaInfo}, Ptr{Cint}),
                      h_ref, h_out, x, UInt8(phi),
                      m, length(m), np, norb, lab, Ref(opts),
                      alpha, length(alpha), alen,
                      props, length(props), plen,
                      iters, length(iters), ilen,
                      info, refusal)
            end
            check(st, refusal[])
            return (alpha, props, Int.(iters), info[], refusal[])
        end

        # `cap === nothing` — and ONLY that — takes the TWO-PASS route: the first call has every
        # buffer C_NULL, fills `info` and pays the full dense solve, the second fills
        # exactly-sized buffers. A caller who already holds `dim` (free from `sector_rpa` at
        # k = 0, and an upper bound on n_modes, n_neg and n_clusters alike) passes it as `cap` and
        # pays ONE solve. `cap = 0` is a KNOWN dimension — a structurally empty channel — not an
        # absent one, so it takes the one-solve route with zero-length mode buffers.
        function _sector_transition_moments(::Type{$L}, h_ref, h_out, x, phi, m::Vector{Float64},
                                            n_props::Integer, norb::Integer, labels,
                                            opts::SectorMomentOptions,
                                            cap::Union{Integer,Nothing})
            np = Int(n_props)
            info = Ref{SectorMomentsInfo}()
            refusal = Ref{Cint}(-99)  # sentinel: the ABI writes NONE (0) on every non-NULL path
            lab, keep = _cstring_vec(labels, np)
            l = [Ref{Csize_t}(0) for _ in 1:7]
            # The 33-slot signature is spelled out at BOTH call sites: `ccall`'s argument-type
            # tuple must be a literal, so it cannot be hoisted into a local.
            if cap === nothing
                st = GC.@preserve m lab keep begin
                    ccall(($(sym("sector_transition_moments")), libqntt), Cint,
                          (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt8,
                           Ptr{Float64}, Csize_t, Csize_t, Csize_t,
                           Ptr{Cstring}, Ptr{SectorMomentOptions},
                           Ptr{Float64}, Csize_t, Ptr{Csize_t},
                           Ptr{Float64}, Csize_t, Ptr{Csize_t},
                           Ptr{Float64}, Csize_t, Ptr{Csize_t},
                           Ptr{Float64}, Csize_t, Ptr{Csize_t},
                           Ptr{Float64}, Csize_t, Ptr{Csize_t},
                           Ptr{Float64}, Csize_t, Ptr{Csize_t},
                           Ptr{Csize_t}, Csize_t, Ptr{Csize_t},
                           Ptr{SectorMomentsInfo}, Ptr{Cint}),
                          h_ref, h_out, x, UInt8(phi),
                          m, length(m), np, norb, lab, Ref(opts),
                          C_NULL, 0, l[1], C_NULL, 0, l[2], C_NULL, 0, l[3],
                          C_NULL, 0, l[4], C_NULL, 0, l[5], C_NULL, 0, l[6],
                          C_NULL, 0, l[7], info, refusal)
                end
                check(st, refusal[])
                nm, nn, nc = Int(info[].n_modes), Int(info[].n_neg), Int(info[].n_clusters)
            else
                cap < 0 && throw(ArgumentError("dim must be non-negative, got $cap"))
                nm = nn = nc = Int(cap)
            end
            omega = Vector{Float64}(undef, nm)
            moments = Vector{Float64}(undef, np * nm)
            osc = Vector{Float64}(undef, nm)
            sos = Vector{Float64}(undef, np * np)
            props = Vector{Float64}(undef, np * 5)
            unstable = Vector{Float64}(undef, nn)
            clusters = Vector{Csize_t}(undef, 2 * nc)
            st = GC.@preserve m lab keep omega moments osc sos props unstable clusters begin
                ccall(($(sym("sector_transition_moments")), libqntt), Cint,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt8,
                       Ptr{Float64}, Csize_t, Csize_t, Csize_t,
                       Ptr{Cstring}, Ptr{SectorMomentOptions},
                       Ptr{Float64}, Csize_t, Ptr{Csize_t},
                       Ptr{Float64}, Csize_t, Ptr{Csize_t},
                       Ptr{Float64}, Csize_t, Ptr{Csize_t},
                       Ptr{Float64}, Csize_t, Ptr{Csize_t},
                       Ptr{Float64}, Csize_t, Ptr{Csize_t},
                       Ptr{Float64}, Csize_t, Ptr{Csize_t},
                       Ptr{Csize_t}, Csize_t, Ptr{Csize_t},
                       Ptr{SectorMomentsInfo}, Ptr{Cint}),
                      h_ref, h_out, x, UInt8(phi),
                      m, length(m), np, norb, lab, Ref(opts),
                      omega, length(omega), l[1],
                      moments, length(moments), l[2],
                      osc, length(osc), l[3],
                      sos, length(sos), l[4],
                      props, length(props), l[5],
                      unstable, length(unstable), l[6],
                      clusters, length(clusters), l[7],
                      info, refusal)
            end
            check(st, refusal[])
            n = Tuple(min(Int(r[]), c) for (r, c) in zip(l, (length(omega), length(moments),
                                                             length(osc), length(sos),
                                                             length(props), length(unstable),
                                                             length(clusters))))
            return (omega[1:n[1]], moments[1:n[2]], osc[1:n[3]], sos[1:n[4]], props[1:n[5]],
                    unstable[1:n[6]], Int.(clusters[1:n[7]]), info[], refusal[])
        end

        function _saddle_seed(::Type{$L}, h, x, eps, budget, seed)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            lambda = Ref{Float64}(NaN)
            check(ccall(($(sym("saddle_seed")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Csize_t, UInt64, Ptr{Float64},
                         Ptr{Ptr{Cvoid}}),
                        h, x, Float64(eps), budget, seed, lambda, out))
            return (out[], lambda[])
        end

        function _gad(::Type{$L}, h, x0, opts::GadOptions)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            info = Ref{GadInfo}()
            check(ccall(($(sym("gad")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{GadOptions}, Ptr{Ptr{Cvoid}}, Ptr{GadInfo}),
                        h, x0, Ref(opts), out, info))
            return (out[], info[])
        end
    end
end

# The C64 layers' signed solve (qntt_{c64,su2_c64}_rpa_signed): a kernel request (SIGNED runs the
# W-metric Davidson on RR, the layer's only lane) but no metric request, no curvature pair and no
# out_metric; BOTH realified copies of every omega^2 come back, so k counts realified values.
# Buffers are sized by the same basis bound n_max = max(max_iter, k + 2) as `_rpa_signed`.
for (L, lp) in _LAYERS
    p = "qntt_$(lp)c64_"
    sym(op) = QuoteNode(Symbol(p * op))
    @eval function _rpa_signed_c64(::Type{$L}, h, x, k, opts::SymplecticOptions, kcode, seed)
        nmax = max(Int(opts.max_iter), Int(k) + 2)
        omega2 = Vector{Float64}(undef, nmax)
        fw = Vector{Float64}(undef, nmax)
        res = Vector{Float64}(undef, nmax)
        krein = Vector{Float64}(undef, nmax)
        cx = Vector{Float64}(undef, 3nmax)
        report = Vector{Int64}(undef, 5)
        len, fwlen, reslen = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
        kreinlen, cxlen, replen = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
        cert, conv, kernel = Ref{Cint}(0), Ref{Cint}(0), Ref{Cint}(0)
        iters, applies = Ref{Csize_t}(0), Ref{Csize_t}(0)
        check(ccall(($(sym("rpa_signed")), libqntt), Cint,
                    (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Ptr{SymplecticOptions}, Cint, UInt64,
                     Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Float64}, Csize_t, Ptr{Csize_t},
                     Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Float64}, Csize_t, Ptr{Csize_t},
                     Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Csize_t},
                     Ptr{Csize_t}, Ptr{Cint}, Ptr{Int64}, Csize_t, Ptr{Csize_t}),
                    h, x, k, Ref(opts), kcode, seed, omega2, nmax, len, fw, nmax, fwlen, res, nmax,
                    reslen, krein, nmax, kreinlen, cx, 3nmax, cxlen, cert, conv, iters, applies,
                    kernel, report, 5, replen))
        ncx = Int(cxlen[]) ÷ 3
        cplx = [(re = cx[3j - 2], im = cx[3j - 1], residual = cx[3j]) for j in 1:ncx]
        return (omega2[1:Int(len[])], fw[1:Int(fwlen[])], res[1:Int(reslen[])],
                krein[1:Int(kreinlen[])], cplx, cert[] != 0, conv[] != 0, Int(iters[]),
                Int(applies[]), Int(kernel[]), report[1:Int(replen[])])
    end
end
