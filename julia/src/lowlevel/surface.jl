# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

# --- per-(layer, scalar) surface ----------------------------------------------------------------

for (L, lp) in _LAYERS, (T, ss) in _SCALARS
    p = "qntt_$(lp)$(ss)_"
    sym(op) = QuoteNode(Symbol(p * op))
    @eval begin
        # ---- lifetime ----
        _vec_free(::Type{$L}, ::Type{$T}, ptr) =
            ccall(($(sym("vec_free")), libqntt), Cvoid, (Ptr{Cvoid},), ptr)
        _ham_free(::Type{$L}, ::Type{$T}, ptr) =
            ccall(($(sym("ham_free")), libqntt), Cvoid, (Ptr{Cvoid},), ptr)

        # ---- construction ----
        function _vec_zeros(::Type{$L}, ::Type{$T}, d, a, b)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("vec_zeros")), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Ptr{Ptr{Cvoid}}), d, a, b, out))
            return out[]
        end

        function _vec_from_ranks(::Type{$L}, ::Type{$T}, d, a, b, flat)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("vec_from_ranks")), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Ptr{Csize_t}, Csize_t, Ptr{Ptr{Cvoid}}),
                        d, a, b, flat, length(flat), out))
            return out[]
        end

        function _vec_randn(::Type{$L}, ::Type{$T}, d, a, b, flat, seed)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("vec_randn")), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Ptr{Csize_t}, Csize_t, UInt64,
                         Ptr{Ptr{Cvoid}}),
                        d, a, b, flat, length(flat), seed, out))
            return out[]
        end

        function _vec_clone(::Type{$L}, ::Type{$T}, x)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("vec_clone")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), x, out))
            return out[]
        end

        # ---- structure ----
        function _vec_space(::Type{$L}, ::Type{$T}, x)
            d, a, b = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(sym("vec_space")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}), x, d, a, b))
            return (Int(d[]), Int(a[]), Int(b[]))
        end

        function _vec_ranks_flat(::Type{$L}, ::Type{$T}, x)
            len = Ref{Csize_t}(0)
            check(ccall(($(sym("vec_ranks")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}), x, C_NULL, 0, len))
            flat = Vector{Csize_t}(undef, Int(len[]))
            check(ccall(($(sym("vec_ranks")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
                        x, flat, length(flat), len))
            return flat
        end

        function _vec_ortho(::Type{$L}, ::Type{$T}, x)
            out = Ref{Cssize_t}(0)
            check(ccall(($(sym("vec_ortho")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cssize_t}), x, out))
            return Int(out[])
        end

        function _vec_check(::Type{$L}, ::Type{$T}, x)
            out = Ref{Cint}(0)
            check(ccall(($(sym("vec_check")), libqntt), Cint, (Ptr{Cvoid}, Ptr{Cint}), x, out))
            return out[] != 0
        end

        function _vec_sizes(::Type{$L}, ::Type{$T}, x)
            s, r, br = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(sym("vec_sizes")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}), x, s, r, br))
            return (Int(s[]), Int(r[]), Int(br[]))
        end

        # ---- state diagnostics ----
        # The BILINEAR (unconjugated) contraction, as against `_vec_dot`'s Hermitian one; over
        # Float64 the two coincide.
        function _vec_dot_bilinear(::Type{$L}, ::Type{$T}, x, y)
            out = Ref{$T}(0)
            check(ccall(($(sym("vec_dot_bilinear")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{$T}), x, y, out))
            return out[]
        end

        # Schmidt values at every INTERIOR bond, concatenated, with their parallel multiplet
        # weights and the per-bond counts that partition both. One call: the whole table costs
        # one canonicalization. The weights array shares `out_len` with the values array.
        function _vec_schmidt(::Type{$L}, ::Type{$T}, x)
            # NB: `ccall` needs the argument-type tuple spelled out at each call site — it is
            # syntax, not a value, so hoisting it into a local is a load-time syntax error.
            len, nb = Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(sym("vec_schmidt")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Float64}, Csize_t,
                         Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
                        x, C_NULL, 0, len, C_NULL, 0, C_NULL, 0, nb))
            vals = Vector{Float64}(undef, Int(len[]))
            wts = Vector{Float64}(undef, Int(len[]))
            counts = Vector{Csize_t}(undef, Int(nb[]))
            check(ccall(($(sym("vec_schmidt")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Float64}, Csize_t,
                         Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
                        x, vals, length(vals), len, wts, length(wts),
                        counts, length(counts), nb))
            σ = Vector{Vector{Float64}}(undef, length(counts))
            w = Vector{Vector{Float64}}(undef, length(counts))
            off = 0
            for (i, c) in enumerate(counts)
                σ[i] = vals[(off + 1):(off + Int(c))]
                w[i] = wts[(off + 1):(off + Int(c))]
                off += Int(c)
            end
            return (sigma = σ, weight = w)
        end

        function _grad_norm(::Type{$L}, ::Type{$T}, h, x)
            out = Ref{Float64}(0)
            check(ccall(($(sym("grad_norm")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}), h, x, out))
            return out[]
        end

        # ---- persistence (the QNTT-TT v1 state cache) ----
        # These are vector operations, so they ride the same generated family as the rest. The
        # file records its layer and scalar; loading through the wrong family is a clean
        # QNTT_IO, never a reinterpretation. Unaffected by RULE L — the format stores the point
        # group, so a |G| > 1 train round-trips exactly.
        _vec_save(::Type{$L}, ::Type{$T}, x, path) =
            check(ccall(($(sym("vec_save")), libqntt), Cint, (Ptr{Cvoid}, Cstring), x, path))

        function _vec_load(::Type{$L}, ::Type{$T}, path)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("vec_load")), libqntt), Cint,
                        (Cstring, Ptr{Ptr{Cvoid}}), path, out))
            return out[]
        end

        # ---- blocks ----
        function _block_dims(::Type{$L}, ::Type{$T}, x, site, family, sa, sb)
            rows, cols = Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(sym("vec_block_get")), libqntt), Cint,
                        (Ptr{Cvoid}, Csize_t, Cint, Csize_t, Csize_t, Ptr{$T}, Csize_t,
                         Ptr{Csize_t}, Ptr{Csize_t}),
                        x, site, family, sa, sb, C_NULL, 0, rows, cols))
            return (Int(rows[]), Int(cols[]))
        end

        function _block_get(::Type{$L}, ::Type{$T}, x, site, family, sa, sb)
            rows, cols = _block_dims($L, $T, x, site, family, sa, sb)
            m = Matrix{$T}(undef, rows, cols)
            (rows * cols == 0) && return m
            r2, c2 = Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(sym("vec_block_get")), libqntt), Cint,
                        (Ptr{Cvoid}, Csize_t, Cint, Csize_t, Csize_t, Ptr{$T}, Csize_t,
                         Ptr{Csize_t}, Ptr{Csize_t}),
                        x, site, family, sa, sb, m, length(m), r2, c2))
            return m
        end

        function _block_set!(::Type{$L}, ::Type{$T}, x, site, family, sa, sb, m::AbstractMatrix)
            src = convert(Matrix{$T}, m)
            check(ccall(($(sym("vec_block_set")), libqntt), Cint,
                        (Ptr{Cvoid}, Csize_t, Cint, Csize_t, Csize_t, Ptr{$T}, Csize_t, Csize_t),
                        x, site, family, sa, sb, src, size(src, 1), size(src, 2)))
            return nothing
        end

        # ---- algebra ----
        function _vec_norm(::Type{$L}, ::Type{$T}, x)
            out = Ref{Float64}(0)
            check(ccall(($(sym("vec_norm")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Float64}), x, out))
            return out[]
        end

        function _vec_dot(::Type{$L}, ::Type{$T}, x, y)
            out = Ref{$T}(zero($T))
            check(ccall(($(sym("vec_dot")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{$T}), x, y, out))
            return out[]
        end

        function _vec_scaled(::Type{$L}, ::Type{$T}, x, a)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            av = Ref{$T}(convert($T, a))
            check(ccall(($(sym("vec_scaled")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{$T}, Ptr{Ptr{Cvoid}}), x, av, out))
            return out[]
        end

        function _vec_add(::Type{$L}, ::Type{$T}, x, y, sign)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("vec_add")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{Ptr{Cvoid}}), x, y, sign, out))
            return out[]
        end

        function _vec_round(::Type{$L}, ::Type{$T}, x, eps, chi, mode)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("vec_round")), libqntt), Cint,
                        (Ptr{Cvoid}, Float64, Csize_t, Cint, Ptr{Ptr{Cvoid}}),
                        x, eps, chi, mode, out))
            return out[]
        end

        _vec_round!(::Type{$L}, ::Type{$T}, x, eps, chi, mode) =
            check(ccall(($(sym("vec_round_mut")), libqntt), Cint,
                        (Ptr{Cvoid}, Float64, Csize_t, Cint), x, eps, chi, mode))

        function _vec_left_orthonormalize!(::Type{$L}, ::Type{$T}, x, keep_rank)
            out = Ref{Float64}(0)
            check(ccall(($(sym("vec_left_orthonormalize")), libqntt), Cint,
                        (Ptr{Cvoid}, Cint, Ptr{Float64}), x, keep_rank, out))
            return out[]
        end

        function _vec_round_sum(::Type{$L}, ::Type{$T}, alpha, ptrs, tol)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("vec_round_sum")), libqntt), Cint,
                        (Ptr{$T}, Ptr{Ptr{Cvoid}}, Csize_t, Float64, Ptr{Ptr{Cvoid}}),
                        alpha, ptrs, length(ptrs), tol, out))
            return out[]
        end

        # ---- Hamiltonian ----
        function _ham_new(::Type{$L}, ::Type{$T}, d, a, b, t, v, w, eps, construction)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("ham_new")), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Ptr{$T}, Csize_t, Ptr{$T}, Csize_t,
                         Ptr{Cvoid}, Float64, Cint, Ptr{Ptr{Cvoid}}),
                        d, a, b, t, length(t), v, length(v), w, eps, construction, out))
            return out[]
        end

        function _ham_space(::Type{$L}, ::Type{$T}, h)
            d, a, b = Ref{Csize_t}(0), Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(sym("ham_space")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}), h, d, a, b))
            return (Int(d[]), Int(a[]), Int(b[]))
        end

        function _ham_bond_dims(::Type{$L}, ::Type{$T}, h)
            len = Ref{Csize_t}(0)
            check(ccall(($(sym("ham_bond_dims")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}), h, C_NULL, 0, len))
            dims = Vector{Csize_t}(undef, Int(len[]))
            check(ccall(($(sym("ham_bond_dims")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
                        h, dims, length(dims), len))
            return Int.(dims)
        end

        function _ham_apply(::Type{$L}, ::Type{$T}, h, x)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("ham_apply")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), h, x, out))
            return out[]
        end

        function _ham_expectation(::Type{$L}, ::Type{$T}, h, x)
            out = Ref{$T}(zero($T))
            check(ccall(($(sym("ham_expectation")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{$T}), h, x, out))
            return out[]
        end

        function _ham_rayleigh(::Type{$L}, ::Type{$T}, h, x)
            out = Ref{$T}(zero($T))
            check(ccall(($(sym("ham_rayleigh")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{$T}), h, x, out))
            return out[]
        end

        function _ham_xthy(::Type{$L}, ::Type{$T}, x, h, y)
            out = Ref{$T}(zero($T))
            check(ccall(($(sym("ham_xthy")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{$T}), x, h, y, out))
            return out[]
        end

        # ---- solvers ----
        function _als(::Type{$L}, ::Type{$T}, h, x0, opts::AlsOptions)
            lam = Ref{$T}(zero($T))
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("als")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{AlsOptions}, Ptr{$T}, Ptr{Ptr{Cvoid}}),
                        h, x0, Ref(opts), lam, out))
            return (lam[], out[])
        end

        function _mals(::Type{$L}, ::Type{$T}, h, x0, opts::MalsOptions, bond_dims, noise)
            lam = Ref{Float64}(0)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            bd = Csize_t[b for b in bond_dims]
            ns = Float64[a for a in noise]
            cap = Int(opts.max_iter)
            sweeps = Vector{MalsSweep}(undef, cap)
            nsw = Ref{Csize_t}(0)
            check(ccall(($(sym("mals")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{MalsOptions}, Ptr{Csize_t}, Csize_t,
                         Ptr{Float64}, Csize_t, Ptr{Float64}, Ptr{Ptr{Cvoid}},
                         Ptr{MalsSweep}, Csize_t, Ptr{Csize_t}),
                        h, x0, Ref(opts), bd, length(bd), ns, length(ns), lam, out,
                        sweeps, cap, nsw))
            return (lam[], out[], sweeps[1:min(Int(nsw[]), cap)])
        end

        function _tdvp(::Type{$L}, ::Type{$T}, h, x0, opts::TdvpOptions)
            lam = Ref{Float64}(0)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("tdvp")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{TdvpOptions}, Ptr{Float64}, Ptr{Ptr{Cvoid}}),
                        h, x0, Ref(opts), lam, out))
            return (lam[], out[])
        end

        function _riemannian(::Type{$L}, ::Type{$T}, h, x0, opts::RiemannianOptions, cg)
            lam = Ref{Float64}(0)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("riemannian")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{RiemannianOptions}, Cint, Ptr{Float64},
                         Ptr{Ptr{Cvoid}}),
                        h, x0, Ref(opts), cg, lam, out))
            return (lam[], out[])
        end

        function _lanczos(::Type{$L}, ::Type{$T}, h, x0, opts::LanczosOptions)
            lam = Ref{Float64}(0)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            brk, iters = Ref{Cint}(0), Ref{Csize_t}(0)
            cap = Int(opts.max_iter)
            lambdas = Vector{Float64}(undef, cap)
            res = Vector{Float64}(undef, cap)
            ranks = Vector{Csize_t}(undef, cap)
            rranks = Vector{Csize_t}(undef, cap)
            check(ccall(($(sym("lanczos")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{LanczosOptions}, Ptr{Float64},
                         Ptr{Ptr{Cvoid}}, Ptr{Cint}, Ptr{Csize_t}, Ptr{Float64}, Ptr{Float64},
                         Ptr{Csize_t}, Ptr{Csize_t}, Csize_t),
                        h, x0, Ref(opts), lam, out, brk, iters,
                        lambdas, res, ranks, rranks, cap))
            n = min(Int(iters[]), cap)
            return (lam[], out[], brk[] != 0,
                    lambdas[1:n], res[1:n], Int.(ranks[1:n]), Int.(rranks[1:n]))
        end

        # ---- multi-state DMRG ----
        # Handle arrays are pre-validated by the library (a short capacity is an error, never a
        # truncation), so every one below is sized to `nroots` exactly. The sweep buffers keep
        # `_mals`' truncating idiom and are sized to `max_iter`.
        #
        # KNOWN GAP: the per-root / per-lower-state companion buffers are passed `C_NULL`, so
        # `SaSweep.energies`, `SaSweep.discarded_weights` and `SsSweep.overlaps` — all reachable
        # from Rust and from C — are NOT reachable from Julia. They are not recoverable from the
        # pooled row either: `max_dev` and the pooled `discarded_weight` are aggregates over the
        # roots, and the per-lower overlap trace (the collapse signal a state-specific run is
        # watched with) has no pooled counterpart at all. Surfacing them means a fourth element
        # in the returned tuple or a keyword, i.e. a change to this surface's contract; until
        # that is decided the three-tuple stands and the gap is stated here rather than papered
        # over. `SaSweep`/`SsSweep` are `#[repr(C)]` mirrors, so the traces cannot ride along as
        # extra struct fields.
        function _state_average_mals(::Type{$L}, ::Type{$T}, h, x0, nroots,
                                     opts::StateAverageOptions, weights, bond_dims, noise)
            n = Int(nroots)
            bd = Csize_t[b for b in bond_dims]
            ns = Float64[a for a in noise]
            ws = Float64[w for w in weights]
            lambdas = Vector{Float64}(undef, n)
            states = fill(Ptr{Cvoid}(C_NULL), n)
            cap = Int(opts.mals.max_iter)
            sweeps = Vector{SaSweep}(undef, cap)
            nsw = Ref{Csize_t}(0)
            check(ccall(($(sym("state_average_mals")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Ptr{StateAverageOptions},
                         Ptr{Float64}, Csize_t, Ptr{Csize_t}, Csize_t, Ptr{Float64}, Csize_t,
                         Ptr{Float64}, Ptr{Ptr{Cvoid}}, Csize_t, Ptr{SaSweep}, Csize_t,
                         Ptr{Csize_t}, Ptr{Float64}, Ptr{Float64}),
                        h, x0, n, Ref(opts), ws, length(ws), bd, length(bd), ns, length(ns),
                        lambdas, states, n, sweeps, cap, nsw, C_NULL, C_NULL))
            return (lambdas, states, sweeps[1:min(Int(nsw[]), cap)])
        end

        function _state_specific_mals(::Type{$L}, ::Type{$T}, h, x0, lower::Vector{Ptr{Cvoid}},
                                      opts::StateSpecificOptions, bond_dims, noise,
                                      lower_energies)
            bd = Csize_t[b for b in bond_dims]
            ns = Float64[a for a in noise]
            le = Float64[e for e in lower_energies]
            lam = Ref{Float64}(0)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            cap = Int(opts.mals.max_iter)
            sweeps = Vector{SsSweep}(undef, cap)
            nsw = Ref{Csize_t}(0)
            check(ccall(($(sym("state_specific_mals")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t,
                         Ptr{StateSpecificOptions}, Ptr{Csize_t}, Csize_t, Ptr{Float64}, Csize_t,
                         Ptr{Float64}, Csize_t, Ptr{Float64}, Ptr{Ptr{Cvoid}}, Ptr{SsSweep},
                         Csize_t, Ptr{Csize_t}, Ptr{Float64}),
                        h, x0, lower, length(lower), Ref(opts), bd, length(bd), ns, length(ns),
                        le, length(le), lam, out, sweeps, cap, nsw, C_NULL))
            return (lam[], out[], sweeps[1:min(Int(nsw[]), cap)])
        end

        # The Gram is `(nlower + nroots)^2` row-major over `[lower; roots]`, so it is sized here
        # rather than size-queried; `keep_sa` is honoured by allocating the second handle array
        # (the library refuses a short one before it runs).
        function _excited_states(::Type{$L}, ::Type{$T}, h, x0, lower::Vector{Ptr{Cvoid}},
                                 opts::ExcitedOptions, weights, bond_dims, noise, ss_noise)
            n = Int(opts.nroots)
            m = length(lower)
            ws = Float64[w for w in weights]
            bd = Csize_t[b for b in bond_dims]
            ns = Float64[a for a in noise]
            sn = Float64[a for a in ss_noise]
            energies = Vector{Float64}(undef, n)
            roots = fill(Ptr{Cvoid}(C_NULL), n)
            info = Ref{ExcitedInfo}()
            rows = Vector{ExcitedRoot}(undef, n)
            lg = Vector{Float64}(undef, m)
            gram = Vector{Float64}(undef, (m + n)^2)
            glen = Ref{Csize_t}(0)
            keep = opts.keep_sa != 0
            sa = fill(Ptr{Cvoid}(C_NULL), keep ? n : 0)
            check(ccall(($(sym("excited_states")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t, Ptr{ExcitedOptions},
                         Ptr{Float64}, Csize_t, Ptr{Csize_t}, Csize_t, Ptr{Float64}, Csize_t,
                         Ptr{Float64}, Csize_t, Ptr{Float64}, Ptr{Ptr{Cvoid}}, Csize_t,
                         Ptr{ExcitedInfo}, Ptr{ExcitedRoot}, Ptr{Float64}, Ptr{Float64}, Csize_t,
                         Ptr{Csize_t}, Ptr{Ptr{Cvoid}}, Csize_t),
                        h, x0, lower, m, Ref(opts), ws, length(ws), bd, length(bd), ns, length(ns),
                        sn, length(sn), energies, roots, n, info, rows, lg, gram, length(gram),
                        glen, sa, length(sa)))
            return (energies, roots, info[], rows, lg, gram, sa)
        end

        # ---- point-group twins (both layers, SPG6) ----
        #
        # Each takes the descriptor quadruple (kind, orbsym, orbsym_len, target) right after the
        # space triple. `orbsym` holds 1-BASED MOLPRO ids; `target` and every returned irrep are
        # 0-BASED XOR indices. Rank grids gain the irrep axis.
        function _vec_zeros_pg(::Type{$L}, ::Type{$T}, d, a, b, kind, orbsym, target)
            ob = Vector{UInt8}(orbsym)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("vec_zeros_pg")), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint,
                         Ptr{Ptr{Cvoid}}),
                        d, a, b, Cint(kind), ob, length(ob), Cint(target), out))
            return out[]
        end

        function _vec_from_ranks_pg(::Type{$L}, ::Type{$T}, d, a, b, kind, orbsym, target, flat)
            ob = Vector{UInt8}(orbsym)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("vec_from_ranks_pg")), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint,
                         Ptr{Csize_t}, Csize_t, Ptr{Ptr{Cvoid}}),
                        d, a, b, Cint(kind), ob, length(ob), Cint(target),
                        flat, length(flat), out))
            return out[]
        end

        function _vec_randn_pg(::Type{$L}, ::Type{$T}, d, a, b, kind, orbsym, target, flat, seed)
            ob = Vector{UInt8}(orbsym)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("vec_randn_pg")), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint,
                         Ptr{Csize_t}, Csize_t, UInt64, Ptr{Ptr{Cvoid}}),
                        d, a, b, Cint(kind), ob, length(ob), Cint(target),
                        flat, length(flat), seed, out))
            return out[]
        end

        function _vec_ranks_flat_pg(::Type{$L}, ::Type{$T}, x)
            len = Ref{Csize_t}(0)
            check(ccall(($(sym("vec_ranks_pg")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}), x, C_NULL, 0, len))
            flat = Vector{Csize_t}(undef, Int(len[]))
            check(ccall(($(sym("vec_ranks_pg")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
                        x, flat, length(flat), len))
            return flat
        end

        function _block_dims_pg(::Type{$L}, ::Type{$T}, x, site, family, sa, sb, ir)
            rows, cols = Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(sym("vec_block_get_pg")), libqntt), Cint,
                        (Ptr{Cvoid}, Csize_t, Cint, Csize_t, Csize_t, UInt8, Ptr{$T}, Csize_t,
                         Ptr{Csize_t}, Ptr{Csize_t}),
                        x, site, family, sa, sb, UInt8(ir), C_NULL, 0, rows, cols))
            return (Int(rows[]), Int(cols[]))
        end

        function _block_get_pg(::Type{$L}, ::Type{$T}, x, site, family, sa, sb, ir)
            rows, cols = _block_dims_pg($L, $T, x, site, family, sa, sb, ir)
            m = Matrix{$T}(undef, rows, cols)
            (rows * cols == 0) && return m
            r2, c2 = Ref{Csize_t}(0), Ref{Csize_t}(0)
            check(ccall(($(sym("vec_block_get_pg")), libqntt), Cint,
                        (Ptr{Cvoid}, Csize_t, Cint, Csize_t, Csize_t, UInt8, Ptr{$T}, Csize_t,
                         Ptr{Csize_t}, Ptr{Csize_t}),
                        x, site, family, sa, sb, UInt8(ir), m, length(m), r2, c2))
            return m
        end

        function _block_set_pg!(::Type{$L}, ::Type{$T}, x, site, family, sa, sb, ir, m::AbstractMatrix)
            src = convert(Matrix{$T}, m)
            check(ccall(($(sym("vec_block_set_pg")), libqntt), Cint,
                        (Ptr{Cvoid}, Csize_t, Cint, Csize_t, Csize_t, UInt8, Ptr{$T},
                         Csize_t, Csize_t),
                        x, site, family, sa, sb, UInt8(ir), src, size(src, 1), size(src, 2)))
            return nothing
        end

        # Reports what the SPACE stores: nbits, the XOR target, and the per-site XOR irreps. The
        # point-group KIND is deliberately NOT recoverable (nbits does not determine it), so the
        # high level re-labels with the conventional kind of that order.
        function _vec_point_group(::Type{$L}, ::Type{$T}, x)
            nbits, target = Ref{Cint}(0), Ref{Cint}(0)
            len = Ref{Csize_t}(0)
            check(ccall(($(sym("vec_point_group")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
                        x, nbits, target, C_NULL, 0, len))
            irreps = Vector{UInt8}(undef, Int(len[]))
            check(ccall(($(sym("vec_point_group")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
                        x, nbits, target, irreps, length(irreps), len))
            return (Int(nbits[]), Int(target[]), irreps)
        end

        function _ham_point_group(::Type{$L}, ::Type{$T}, h)
            nbits, target = Ref{Cint}(0), Ref{Cint}(0)
            len = Ref{Csize_t}(0)
            check(ccall(($(sym("ham_point_group")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
                        h, nbits, target, C_NULL, 0, len))
            irreps = Vector{UInt8}(undef, Int(len[]))
            check(ccall(($(sym("ham_point_group")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{UInt8}, Csize_t, Ptr{Csize_t}),
                        h, nbits, target, irreps, length(irreps), len))
            return (Int(nbits[]), Int(target[]), irreps)
        end

        function _ham_new_pg(::Type{$L}, ::Type{$T}, d, a, b, kind, orbsym, target,
                             t, v, w, eps, construction)
            ob = Vector{UInt8}(orbsym)
            out = Ref{Ptr{Cvoid}}(C_NULL)
            check(ccall(($(sym("ham_new_pg")), libqntt), Cint,
                        (Csize_t, Csize_t, Csize_t, Cint, Ptr{UInt8}, Csize_t, Cint,
                         Ptr{$T}, Csize_t, Ptr{$T}, Csize_t, Ptr{Cvoid}, Float64, Cint,
                         Ptr{Ptr{Cvoid}}),
                        d, a, b, Cint(kind), ob, length(ob), Cint(target),
                        t, length(t), v, length(v), w, eps, construction, out))
            return out[]
        end

        # ---- PD probe ----
        # `part` is a HESS_PARTS wire code. Every verdict returns QNTT_OK — the verdict is a
        # diagnosis about the state, not a status of the call — so this does NOT throw on
        # WitnessNegative, which is the goal of the `:ii` probe.
        function _pd_probe(::Type{$L}, ::Type{$T}, h, x, part, budget, seed)
            verdict, is_pd = Ref{Cint}(0), Ref{Cint}(0)
            lambda, residual = Ref{Float64}(NaN), Ref{Float64}(NaN)
            applies = Ref{Csize_t}(0)
            check(ccall(($(sym("pd_probe")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Csize_t, UInt64, Ptr{Cint}, Ptr{Float64},
                         Ptr{Float64}, Ptr{Cint}, Ptr{Csize_t}),
                        h, x, Cint(part), budget, seed, verdict, lambda, residual, is_pd,
                        applies))
            return (Int(verdict[]), lambda[], residual[], Int(is_pd[]), Int(applies[]))
        end

        # ---- RPA ----
        function _rpa_dense(::Type{$L}, ::Type{$T}, h, x)
            len, stable = Ref{Csize_t}(0), Ref{Cint}(0)
            check(ccall(($(sym("rpa_dense")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Cint}),
                        h, x, C_NULL, 0, len, stable))
            omega = Vector{Float64}(undef, Int(len[]))
            check(ccall(($(sym("rpa_dense")), libqntt), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ptr{Csize_t}, Ptr{Cint}),
                        h, x, omega, length(omega), len, stable))
            return omega
        end
    end
end
