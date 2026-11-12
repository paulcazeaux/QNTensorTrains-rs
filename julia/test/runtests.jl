# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

using Test
using LinearAlgebra
using QNTT

# The fixtures live in the Rust crate's `testdata/`, two levels up from `julia/test/`.
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
h4() = joinpath(ROOT, "testdata", "h4.FCIDUMP")
h2o() = joinpath(ROOT, "testdata", "h2o.FCIDUMP")

@testset "QNTT" begin

@testset "library" begin
    @test QNTT.version() isa VersionNumber
    @test QNTT.nthreads() ≥ 1
    QNTT.init(0)                     # idempotent; must not throw on a second call
    @test QNTT.nthreads() ≥ 1
end

@testset "spaces and rank grids" begin
    s = QNTT.QnSpace(6, 3, 3)
    @test (s.d, s.a, s.b) == (6, 3, 3)
    @test QNTT.grid_extent(s) == (4, 4)
    @test_throws QNTTError QNTT.QnSpace(3, 4, 0)          # Nup > d

    # Sector enumeration is the library's canonical order and matches the admissible bounds.
    for b in 0:s.d
        secs = QNTT.bond_sectors(s, b)
        @test !isempty(secs)
        @test all(0 ≤ a ≤ min(b, 3) for (a, _) in secs)
    end

    mx = QNTT.maxranks(s)
    @test length(mx) == s.d + 1
    @test all(size(g) == (4, 4) for g in mx)
    @test mx[1][1, 1] == 1 && mx[end][end, end] == 1        # rank-1 boundaries

    # Rank grids survive a full round trip through a train.
    r = QNTT.uniform_ranks(s, 3)
    x = QNTT.tt_from_ranks(s, r)
    @test QNTT.ranks(x) == r
    @test isvalid(x)

    su2 = QNTT.Su2Space(4, 4, 0)
    @test QNTT.grid_extent(su2) == (5, 5)
    @test_throws QNTTError QNTT.Su2Space(4, 4, 6)          # 2S > N
    @test_throws QNTTError QNTT.Su2Space(4, 3, 0)          # N − 2S odd
end

@testset "FCIDUMP" begin
    data = QNTT.read_fcidump(h4())
    @test QNTT.norb(data) == 4
    @test QNTT.nelec(data) == 4
    @test QNTT.ms2(data) == 0
    @test QNTT.e0(data) ≈ 2.29310124732 atol = 1e-10

    t = QNTT.onebody(data)
    v = QNTT.twobody(data)
    @test size(t) == (4, 4)
    @test size(v) == (4, 4, 4, 4)
    @test t ≈ t'                                            # real one-body integrals are symmetric
    @test v ≈ permutedims(v, (3, 4, 1, 2))                  # chemists' pair swap

    @test_throws ArgumentError QNTT.read_fcidump("no/such/file.FCIDUMP")
end

@testset "integral layout convention" begin
    # The C API takes row-major buffers and Julia is column-major, so this is the one place a
    # silent transposition could hide. `t[1,2] = 1` is a†₀a₁ in the library's 0-based convention:
    # it must move an electron from orbital 1 to orbital 0, and not the reverse.
    t = zeros(2, 2); t[1, 2] = 1.0
    v = zeros(2, 2, 2, 2)
    H = QNTT.Hamiltonian(QNTT.QnSpace(2, 1, 0), t, v)
    x = QNTT.tt_state([false, true], [false, false])
    y = QNTT.tt_state([true, false], [false, false])
    @test QNTT.xthy(y, H, x) ≈ 1.0
    @test QNTT.xthy(x, H, y) ≈ 0.0

    # And the two Hamiltonian constructors agree: parsed buffers passed straight through vs.
    # Julia arrays permuted on the way in.
    data = QNTT.read_fcidump(h4())
    s = QNTT.QnSpace(data)
    H1 = QNTT.Hamiltonian(s, data)
    H2 = QNTT.Hamiltonian(s, QNTT.onebody(data), QNTT.twobody(data))
    z = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 3); seed = 8)
    @test QNTT.rayleigh_quotient(H1, z) ≈ QNTT.rayleigh_quotient(H2, z) rtol = 1e-14
end

@testset "algebra" begin
    s = QNTT.QnSpace(4, 2, 2)
    x = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 2); seed = 1)
    y = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 2); seed = 2)

    @test norm(x) > 0
    @test dot(x, x) ≈ norm(x)^2 rtol = 1e-12
    @test norm(2x) ≈ 2norm(x) rtol = 1e-12
    @test norm(x - x) ≤ 1e-14 * norm(x)
    @test dot(x, y) ≈ conj(dot(y, x)) rtol = 1e-12

    # The dense materialization agrees with the TT inner products.
    xd, yd = Array(x), Array(y)
    @test length(xd) == 4^4
    @test norm(xd) ≈ norm(x) rtol = 1e-12
    @test dot(xd, yd) ≈ dot(x, y) rtol = 1e-10

    # Rounding is a contraction that keeps the vector within its budget. NOTE the floor on these
    # difference norms: ‖a − b‖ for two nearly-equal trains is evaluated by contracting the
    # difference, so cancellation limits its accuracy to ≈ √eps · ‖a‖ ≈ 1e-8 · ‖a‖ no matter how
    # exact the trains are. Tolerances below that would test the contraction, not the rounding.
    DIFF_TOL = 1e-7
    xr = round(x; eps = 1e-12)
    @test norm(x - xr) ≤ DIFF_TOL * norm(x)

    # Deterministic weighted sum.
    z = QNTT.round_sum([2.0, -1.0], [x, y]; tol = 1e-12)
    @test norm(z - (2x - y)) ≤ DIFF_TOL * norm(z)

    # Reproducibility: the same seed gives the same train.
    @test norm(QNTT.tt_randn(s, QNTT.uniform_ranks(s, 2); seed = 1) - x) == 0
end

@testset "blocks" begin
    s = QNTT.QnSpace(4, 2, 2)
    src = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 2); seed = 42)
    dst = QNTT.tt_from_ranks(s, QNTT.ranks(src))

    copied = 0
    for site in 0:(s.d - 1), fam in (:empty, :up, :dn, :occ), sec in QNTT.bond_sectors(s, site)
        m = QNTT.getblock(src, site, fam, sec)
        isempty(m) && continue
        QNTT.setblock!(dst, site, fam, sec, m)
        copied += 1
    end
    @test copied > 0
    @test norm(src - dst) ≤ 1e-14 * norm(src)

    @test_throws QNTTError QNTT.setblock!(dst, 0, :empty, (0, 0), zeros(99, 99))
    @test_throws ArgumentError QNTT.getblock(dst, 0, :nonsense, (0, 0))

    d = QNTT.blocks(src, 1)
    @test !isempty(d)
    @test all(!isempty(m) for m in values(d))
end

@testset "ground state (abelian)" begin
    data = QNTT.read_fcidump(h4())
    s = QNTT.QnSpace(data)
    H = QNTT.Hamiltonian(s, data)

    @test QNTT.space(H) == s
    bd = QNTT.automaton_bond_dims(H)
    @test length(bd) == s.d + 1 && first(bd) == 1 && last(bd) == 1

    x0 = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 4); seed = 2024)
    λ, x = QNTT.als(H, x0; eps = 1e-8, max_iter = 30)

    # H4/STO-3G at full rank: the electronic ground state, pinned against the Rust suite.
    @test λ ≈ -4.459488695954761 rtol = 1e-9
    @test QNTT.rayleigh_quotient(H, x) ≈ λ rtol = 1e-8
    @test QNTT.expectation(H, x) ≈ λ * dot(x, x) rtol = 1e-8

    # `H * x` is the exact matvec.
    Hx = H * x
    @test dot(x, Hx) / dot(x, x) ≈ λ rtol = 1e-8
    @test QNTT.xthy(x, H, x) ≈ dot(x, Hx) rtol = 1e-10

    λm, _ = QNTT.mals(H, x0; eps = 1e-8, max_iter = 20, bond_dims = [0])
    @test λm ≈ λ rtol = 1e-8

    # Imaginary-time TDVP: variational, and a clear improvement on the random start. It is not
    # asked to reach the ALS minimum here — 40 sweeps at Δτ = 0.1 is a smoke run, not a converged
    # calculation.
    rq0 = QNTT.rayleigh_quotient(H, x0)
    λt, _ = QNTT.tdvp(H, x0; eps = 1e-6, max_sweeps = 40)
    @test λt ≥ λ - 1e-6
    @test λt < rq0

    # Rounded Lanczos on a system this small exhausts its Krylov space, so `breakdown` is the
    # expected outcome, not a failure; what matters is that the reported λ stays variational.
    λl, _, info = QNTT.lanczos(H, x0; tol = 1e-6, max_iter = 20)
    @test λl ≥ λ - 1e-6
    @test length(info.lambdas) ≥ 1
    @test length(info.ranks) == length(info.lambdas)
end

@testset "DMRG schedule: ramp + noise" begin
    data = QNTT.read_fcidump(h4())
    s = QNTT.QnSpace(data)
    H = QNTT.Hamiltonian(s, data)

    # A single determinant is a fixed point of the noise-free two-site sweep: it already solves
    # every local eigenproblem, so the run converges — variational, flat in the bond dimension —
    # far above the minimum. Noise is the escape.
    occ = [k < s.a for k in 0:(s.d - 1)]
    hf = QNTT.tt_state(occ, occ)

    λ_quiet, _, hist_quiet = QNTT.mals(H, hf; eps = 1e-12, max_iter = 30, bond_dims = [0])
    λ_noisy, x, hist = QNTT.mals(H, hf; eps = 1e-12, max_iter = 30, bond_dims = [0],
                                 noise = [1e-1, 1e-2, 1e-3, 1e-4, 0.0, 0.0, 0.0], seed = 3,
                                 noise_type = :random)

    @test λ_quiet > λ_noisy + 1e-6                     # the trap is real
    @test λ_noisy ≈ -4.459488695954761 rtol = 1e-9     # and noise reaches the exact energy

    # The history describes the run that produced it.
    @test !isempty(hist)
    @test hist[end].energy ≈ λ_noisy
    @test hist[end].noise == 0.0                        # a schedule must end noise-free
    @test all(0.0 ≤ sw.discarded_weight ≤ 1.0 for sw in hist)
    @test all(sw.max_bond_dim > 0 for sw in hist)
    @test length(hist_quiet) ≥ 1

    # Reproducible in the seed, and an all-zero schedule is exactly the noise-free driver.
    a, _, _ = QNTT.mals(H, hf; eps = 1e-12, max_iter = 10, bond_dims = [0],
                        noise = [1e-2, 0.0], seed = 77, noise_type = :random)
    b, _, _ = QNTT.mals(H, hf; eps = 1e-12, max_iter = 10, bond_dims = [0],
                        noise = [1e-2, 0.0], seed = 77, noise_type = :random)
    @test a == b
    z, _, _ = QNTT.mals(H, hf; eps = 1e-12, max_iter = 30, bond_dims = [0], noise = [0.0])
    @test z == λ_quiet

    # A negative amplitude is refused rather than reaching the driver.
    @test_throws QNTTError QNTT.mals(H, hf; bond_dims = [0], noise = [-1.0])

    # Perturbative (White) noise — the recorded default, all the way from the Rust
    # `MalsOptions::default()` through `qntt_mals_options_default`, so a call that omits
    # `noise_type` runs this flavour. Pin that here: the wire code is 1, and omitting the keyword
    # must agree with naming it.
    @test QNTT.defaults(QNTT.MalsOptions).noise_type == 1
    # Uncapped on this small system it reaches the exact energy, and it is deterministic — no RNG
    # on that path, so the seed is irrelevant.
    λ_pert, _, hp = QNTT.mals(H, hf; eps = 1e-12, max_iter = 30, bond_dims = [0],
                              noise = [1e-4, 1e-4, 1e-4, 1e-5, 0.0, 0.0, 0.0],
                              noise_type = :perturbative)
    @test λ_quiet > λ_pert + 1e-6
    @test λ_pert ≈ -4.459488695954761 rtol = 1e-8
    @test hp[end].noise == 0.0
    λ_pert2, _, _ = QNTT.mals(H, hf; eps = 1e-12, max_iter = 30, bond_dims = [0],
                              noise = [1e-4, 1e-4, 1e-4, 1e-5, 0.0, 0.0, 0.0], seed = 999,
                              noise_type = :perturbative)
    @test λ_pert == λ_pert2      # perturbative ignores the seed
    λ_dflt, _, _ = QNTT.mals(H, hf; eps = 1e-12, max_iter = 30, bond_dims = [0],
                             noise = [1e-4, 1e-4, 1e-4, 1e-5, 0.0, 0.0, 0.0])
    @test λ_dflt == λ_pert       # omitting noise_type IS :perturbative
end

@testset "ground state (SU(2))" begin
    data = QNTT.read_fcidump(h4())
    s = QNTT.Su2Space(data)
    H = QNTT.Hamiltonian(s, data)

    x0 = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 8); seed = 3)
    λ, x = QNTT.mals(H, x0; eps = 1e-10, max_iter = 40, bond_dims = [0])

    # The spin-adapted layer solves the same singlet problem as the abelian one.
    @test λ ≈ -4.459488695954761 rtol = 1e-7
    @test QNTT.rayleigh_quotient(H, x) ≈ λ rtol = 1e-7

    # A closed-shell CSF is a unit-norm state of the right space.
    csf = QNTT.tt_csf([:occ, :occ, :empty, :empty])
    @test norm(csf) ≈ 1.0 rtol = 1e-12
    @test QNTT.space(csf) == s

    # The schedule works on the spin-adapted layer too, and a CSF traps it exactly as a
    # determinant traps the abelian driver.
    λ_quiet, _, _ = QNTT.mals(H, csf; eps = 1e-12, max_iter = 30, bond_dims = [0])
    λ_noisy, _, hist = QNTT.mals(H, csf; eps = 1e-12, max_iter = 30, bond_dims = [0],
                                 noise = [1e-1, 1e-2, 1e-3, 1e-4, 0.0, 0.0, 0.0], seed = 3,
                                 noise_type = :random)
    @test λ_quiet > λ_noisy + 1e-6
    @test λ_noisy ≈ -4.459488695954761 rtol = 1e-7
    @test hist[end].noise == 0.0

    # Both flavours reach the SU(2) driver — perturbative (the crate-wide default) escapes the CSF
    # the same way, through the θ-scaled enrichment.
    λ_pert, _, hist_p = QNTT.mals(H, csf; eps = 1e-12, max_iter = 30, bond_dims = [0],
                                  noise = [1e-3, 1e-3, 1e-4, 0.0, 0.0, 0.0],
                                  noise_type = :perturbative)
    @test λ_quiet > λ_pert + 1e-6
    @test λ_pert ≈ -4.459488695954761 rtol = 1e-7
    @test hist_p[end].noise == 0.0
end

@testset "complex scalars" begin
    data = QNTT.read_fcidump(ComplexF64, h4())
    s = QNTT.QnSpace(data)
    H = QNTT.Hamiltonian(s, data)

    x0 = QNTT.tt_randn(ComplexF64, s, QNTT.uniform_ranks(s, 4); seed = 5)
    @test eltype(x0) == ComplexF64
    λ, x = QNTT.als(H, x0; eps = 1e-10, max_iter = 60)

    @test abs(imag(λ)) ≤ 1e-10 * max(abs(real(λ)), 1)     # Hermitian ⇒ real energy
    @test real(λ) ≈ -4.459488695954761 rtol = 1e-6
    @test dot(x, x) isa ComplexF64

    # The signed C64 solve at the ground state: both realified copies of each ω², agreeing with
    # `rpa` (one ω per physical value) and the dense driver.
    ω2, info = QNTT.rpa_signed(H, x; k = 4, tol = 1e-8, max_iter = 400, seed = 1)
    @test info.converged && info.kernel === :paired && info.metric == :rr
    @test info.report.neg_apb == 0 && info.report.neg_amb == 0 && info.report.metric_events === nothing
    @test length(ω2) == 4 && all(ω2 .> 0) && issorted(ω2)
    @test ω2[1] ≈ ω2[2] rtol = 1e-8
    @test ω2[3] ≈ ω2[4] rtol = 1e-8
    @test length(info.f_w) == length(info.residuals) == length(info.krein) == 4
    @test all(>(0), info.krein) && isempty(info.complex)
    ω, _ = QNTT.rpa(H, x; k = 2, tol = 1e-8, max_iter = 400, seed = 1)
    @test sqrt.(ω2[1:2:end]) ≈ ω rtol = 1e-6
    # The signed W-metric kernel on request (always on `:rr` here): the same spectrum.
    ω2s, is = QNTT.rpa_signed(H, x; k = 4, kernel = :signed, tol = 1e-8, max_iter = 400, seed = 1)
    @test is.kernel === :signed && is.metric === :rr && is.converged
    @test is.report.neg_apb === nothing && is.report.metric_events == 0
    @test ω2s ≈ ω2 rtol = 1e-6
    @test_throws ArgumentError QNTT.rpa_signed(H, x; k = 4, kernel = :both)
    @test_throws QNTTError QNTT.rpa_signed(H, x; k = 4, kernel = 7)
    ωd = QNTT.rpa_dense(H, x)
    @test sqrt.(ω2[1:2:end]) ≈ ωd[1:2] rtol = 1e-6
end

@testset "RPA / linear response" begin
    data = QNTT.read_fcidump(h4())
    s = QNTT.QnSpace(data)
    H = QNTT.Hamiltonian(s, data)

    # A converged fixed-rank ground state on a small manifold.
    x0 = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 2); seed = 7)
    _, x = QNTT.als(H, x0; eps = 1e-10, max_iter = 50)

    ω, info = QNTT.rpa(H, x; k = 3, tol = 1e-8, max_iter = 400, seed = 1)
    @test length(ω) ≥ 1
    @test all(ω .> 0)                                       # a minimizer has positive ω
    @test issorted(ω)
    @test info.applies ≥ info.iterations

    # The certified dense driver must reproduce the iterative one at this size.
    ωd = QNTT.rpa_dense(H, x)
    @test length(ωd) ≥ length(ω)
    for j in eachindex(ω)
        @test ω[j] ≈ ωd[j] rtol = 1e-5
    end

    # `rpa` owns the refusal the retired C `rpa_lowest` made: at the normalized random rank-2 start
    # (the C ABI's indefinite fixture) the signed spectrum is anomalous — premise pinned on
    # `rpa_signed`'s own fields — and `rpa` throws code 5 (did not converge) naming `rpa_signed`.
    xr = x0 * (1 / norm(x0))
    ω2r, ir = QNTT.rpa_signed(H, xr; k = 3, tol = 1e-8, max_iter = 800, seed = 1)
    @test any(<(0), ω2r) || !isempty(ir.complex) || any(<=(0), ir.krein) ||
          ir.report.neg_apb + ir.report.neg_amb > 0
    err = try
        QNTT.rpa(H, xr; k = 3, tol = 1e-8, max_iter = 800, seed = 1)
        nothing
    catch e
        e
    end
    @test err isa QNTTError && err.code == 5
    @test occursin("indefinite point", err.msg) && occursin("rpa_signed", err.msg)
end

@testset "same-sector Tamm-Dancoff (tda)" begin
    # H4 at rank cap 2 on both layers: a capped critical point, so θ is a tangent-space Ritz
    # spectrum, not FCI. The exact statement available is ⟨ι(ξ), H ι(ξ)⟩ = λ + ⟨ξ, Aξ⟩, which the
    # returned states must carry: E(sⱼ) = λ + θⱼ, ‖sⱼ‖ = 1, ⟨x, sⱼ⟩ = 0.
    data = QNTT.read_fcidump(h4())
    for (s, layer) in ((QNTT.QnSpace(data), "abelian"), (QNTT.Su2Space(data), "su2"))
        H = QNTT.Hamiltonian(s, data)
        x0 = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 2); seed = 7)
        λ, x = QNTT.als(H, x0; eps = 1e-10, max_iter = 50)
        θ, info = QNTT.tda(H, x; k = 3, seed = 11, states = true)
        @test length(θ) == 3
        @test issorted(θ)
        @test info.converged
        @test info.negatives_certified
        @test info.precond === :diag                      # :auto runs as :diag
        @test all(r -> r ≤ 1e-7, info.residuals)
        @test info.applies ≥ info.iterations
        @test info.space_dim ≥ 3
        λx = QNTT.rayleigh_quotient(H, x)
        @test length(info.states) == 3
        for (j, sj) in enumerate(info.states)
            @test norm(sj) ≈ 1 atol = 1e-12
            @test abs(dot(x, sj)) < 1e-8
            @test QNTT.rayleigh_quotient(H, sj) ≈ λx + θ[j] atol = 1e-8
        end
        # Same seed ⇒ the same solve; without states nothing is converted.
        θ2, info2 = QNTT.tda(H, x; k = 3, seed = 11)
        @test θ2 == θ && isempty(info2.states)
        # Explicit arms echo themselves and agree to the tolerance.
        θo, io = QNTT.tda(H, x; k = 3, seed = 11, precond = :off)
        @test io.precond === :off && io.converged
        @test θo ≈ θ atol = 1e-7
        # Caller errors: an unknown Symbol before the call, a bad integer code and a negative
        # tol on the C side.
        @test_throws ArgumentError QNTT.tda(H, x; k = 3, precond = :bogus)
        err = try
            QNTT.tda(H, x; k = 3, precond = 9)
            nothing
        catch e
            e
        end
        @test err isa QNTTError && err.code == 2 && occursin("TDA precond 9", err.msg)
        @test_throws QNTTError QNTT.tda(H, x; k = 3, tol = -1.0)
    end
    d = QNTT.defaults(QNTT.TdaOptions)
    @test d.tol == 1e-8 && d.max_iter == 300 && d.precond == QNTT.TDA_PRECOND_CODES[:auto]
    @test d.precond_kappa == 10.0
    @test QNTT.setfields(d; precond = :shifted).precond == QNTT.TDA_PRECOND_CODES[:shifted]
end

@testset "cross-sector response" begin
    # The `d = 4` order-4 table [A, B, C, B] at (Nup, Ndn) = (2, 1), Γ0 = 0 — the smallest
    # point-group fixture with a live shifted channel. `orbsym` goes in as 1-based Molpro ids, so
    # the site irreps are converted through the library's own table. `t`/`v` are symmetry-allowed
    # by construction, with EXACT zeros on the forbidden entries, so no screening pass is needed.
    d, nup, ndn = 4, 2, 1
    site_irreps = [0, 1, 2, 1]
    ob = UInt8[QNTT.molpro_from_irrep(:C2v, ir) for ir in site_irreps]
    t = zeros(d, d)
    for p in 1:d
        t[p, p] = 0.17 * (p - 1 - 0.5d)
    end
    t[2, 4] = t[4, 2] = -0.85
    v = zeros(d, d, d, d)
    for p in 1:d, q in 1:d
        p == q && continue
        w = 1.0 + abs(p - q)
        v[p, p, q, q] = 0.5 / w
        v[p, q, q, p] = 0.23 / w
    end
    for idx in ((2, 4, 1, 1), (4, 2, 1, 1), (1, 1, 2, 4), (1, 1, 4, 2))
        v[idx...] = 0.19
    end

    space(target) = QNTT.QnSpace(d, nup, ndn; point_group = :C2v, orbsym = ob, target = target)
    ham(target) = QNTT.Hamiltonian(space(target), t, v)
    H = ham(0)
    _, x = QNTT.als(H, QNTT.tt_randn(space(0), QNTT.uniform_ranks(space(0), 2); seed = 0x7505);
                    eps = 1e-12, max_iter = 60)

    # `defaults(SectorOptions)` comes from the library, and the keyword Symbols convert.
    dflt = QNTT.defaults(QNTT.SectorOptions)
    @test dflt.readout == QNTT.SECTOR_READOUTS[:tda]
    @test dflt.metric == QNTT.RPA_METRIC_CODES[:auto]
    @test dflt.kernel == QNTT.RPA_KERNEL_CODES[:paired] && dflt.reserved == 0
    o = QNTT.setfields(dflt; readout = :rpa, metric = :rr, tol = 1e-9, kernel = :signed)
    @test (o.readout, o.metric, o.tol, o.kernel) == (Cint(1), Cint(1), 1e-9, Cint(1))
    @test_throws ArgumentError QNTT.setfields(dflt; readout = :nope)

    # φ = 0 is the in-sector problem, and a flux outside the group is refused. Neither may crash.
    @test_throws QNTTError QNTT.sector_rpa(H, ham(1), x, 0)
    @test_throws QNTTError QNTT.sector_rpa(H, ham(1), x, 7)
    # H_out re-stamped (here: H itself, on the wrong target) is refused on the space stamp.
    @test_throws QNTTError QNTT.sector_rpa(H, H, x, 1)

    # `info.dim` is dim T^φ. Which fluxes are live is a property of the GROUP TABLE and the rank
    # grid, and on this table (measured 2026-09-08 through the Rust `SectorCtx::dim` at every rank
    # cap tried) φ = 1 and φ = 2 are structurally EMPTY at any grid while φ = 3 is live. The exact
    # `dim` of the live one is a function of the endpoint's ranks and is printed, not pinned. An
    # empty channel is an answer with `converged = true` and no applies, not an error.
    dims = Dict{Int,Int}()
    for phi in 1:3
        vals, info = QNTT.sector_rpa(H, ham(xor(0, phi)), x, phi; k = 2)
        dims[phi] = info.dim
        if info.dim == 0
            @test isempty(vals)
            @test info.converged
            @test info.applies == 0
        end
    end
    @info "cross-sector channel dims" dims
    @test dims[1] == 0 && dims[2] == 0
    @test dims[3] > 0

    phi = 3
    Ho = ham(xor(0, phi))

    # :tda reports eig(A^φ), an ENERGY, ascending. Its `metric` field is meaningless and reads
    # :auto — never a resolved block.
    θ, itda = QNTT.sector_rpa(H, Ho, x, phi; k = 3, readout = :tda, max_iter = 200)
    @test itda.readout === :tda
    @test itda.metric === :auto
    @test length(θ) ≤ 3 && issorted(θ)
    @test itda.applies > 0
    @test itda.negatives_certified isa Bool

    # :rpa reports the signed ω² of the pair — a different quantity in different units.
    ω2, irpa = QNTT.sector_rpa(H, Ho, x, phi; k = 3, readout = :rpa, max_iter = 400)
    @test irpa.readout === :rpa
    @test issorted(ω2)
    @test irpa.metric ∈ (:ii, :rr, :auto)
    irpa.applies > 0 && @test irpa.metric ∈ (:ii, :rr)   # a resolved run names its block

    # The dense oracle must reproduce the iterative ω² on the same channel.
    ω2d, idense = QNTT.sector_rpa_dense(H, Ho, x, phi)
    @test idense.dim == dims[phi]
    @test idense.status ∈ (:stable, :resonant)
    @test length(ω2d) ≥ length(ω2)
    for j in eachindex(ω2)
        @test ω2[j] ≈ ω2d[j] rtol = 1e-5
    end
    idense.status === :stable && @test all(ω2d .> 0)

    # k = 0 on a LIVE channel: nothing requested, nothing outstanding — the one place
    # `converged` accompanies a positive `dim` with an empty answer.
    vals0, info0 = QNTT.sector_rpa(H, Ho, x, phi; k = 0)
    @test isempty(vals0) && info0.dim == dims[phi] && info0.converged && info0.applies == 0

    @test dflt.max_subspace == 0
    @test QNTT.setfields(dflt; max_subspace = 40).max_subspace == Csize_t(40)

    # --- rpa-floor-selection P2: the curvature/precond selectors and the signed readout's fields ---
    # The preconditioner keyword's wire table mirrors `QNTT_SECTOR_PRECOND_*`.
    @test QNTT.SECTOR_PRECOND_CODES == Dict(:off => Cint(0), :d0 => Cint(1), :auto => Cint(2))
    # J1 — the default call IS the explicit `curvature = :off, precond = :off` call (values, applies,
    # iterations), `:off` ignores its scale (NaN included) and `precond = :auto` is the sector
    # default (`:off`); the signed readout's per-mode fields are sized by the values on `:rpa` and
    # EMPTY on `:tda`.
    # rpa-followup C1: the :rpa arm under both kernels (`kernel = :paired` the default).
    for (ro, kern) in ((:rpa, :paired), (:rpa, :signed), (:tda, :paired))
        mi = ro === :rpa ? 400 : 200   # :rpa counts pair-equivalents: the same 200 vectors
        v0, i0 = QNTT.sector_rpa(H, Ho, x, phi; k = 3, readout = ro, kernel = kern, max_iter = mi)
        v1, i1 = QNTT.sector_rpa(H, Ho, x, phi; k = 3, readout = ro, kernel = kern, max_iter = mi,
                                 curvature = :off, precond = :off)
        v2, i2 = QNTT.sector_rpa(H, Ho, x, phi; k = 3, readout = ro, kernel = kern, max_iter = mi,
                                 curvature = :off, curvature_scale = NaN, precond = :auto)
        @test v0 == v1 && v1 == v2
        @test i0.applies == i1.applies == i2.applies
        @test i0.iterations == i1.iterations == i2.iterations
        n = ro === :rpa ? length(v0) : 0
        @test length(i0.f_w) == n && length(i0.residuals) == n && length(i0.krein) == n
        @test i0.metric_indefinite isa Bool && i0.complex isa AbstractVector
        @test i0.kernel === kern
        if ro === :tda
            @test isempty(i0.complex) && !i0.metric_indefinite
            @test all(isnothing, values(i0.report))
        elseif kern === :paired
            # The paired kernel: its counting offset, no signed entry, never metric-flagged.
            @test i0.report.neg_apb isa Integer && i0.report.neg_amb isa Integer
            @test i0.report.metric_events === nothing && i0.report.general_path === nothing
            @test !i0.metric_indefinite
            @test all(m -> isfinite(m) && abs(m) ≤ 1 + 1e-12, i0.krein)
            @test all(r -> isfinite(r) && r ≥ 0, i0.residuals) && all(f -> f ≥ 0, i0.f_w)
        else
            # The signed kernel: no paired entry in its report.
            @test i0.report.neg_apb === nothing && i0.report.neg_amb === nothing
            @test i0.report.metric_events isa Integer && i0.report.general_path isa Bool
            @test all(m -> isfinite(m) && abs(m) ≤ 1 + 1e-12, i0.krein)
            @test all(r -> isfinite(r) && r ≥ 0, i0.residuals) && all(f -> f ≥ 0, i0.f_w)
            if !i0.metric_indefinite
                @test i0.report.n_negative_basis == 0
                @test !i0.report.general_path && isempty(i0.complex)
                @test all(((w, m),) -> w < 0 || m > 0, zip(v0, i0.krein))
            end
        end
    end
    # J2 — a hard floor below every Schmidt value of `QNTT.schmidt(x)` (and a Tikhonov scale so small
    # that every weight is 1 to 1e-15) is the identity projector: `:off`, bitwise. `precond = :d0`
    # reaches the same omega^2 (rtol 1e-5).
    sch = QNTT.schmidt(x)
    ŵ = sort(reduce(vcat, [sqrt.(w) .* σ for (σ, w) in zip(sch.sigma, sch.weight)]))
    ω2o, io = QNTT.sector_rpa(H, Ho, x, phi; k = 3, readout = :rpa, max_iter = 400)
    for (curv, sc) in ((:hard, 0.5 * first(ŵ)), (:tikhonov, 1e-9 * first(ŵ)))
        ω2c, ic = QNTT.sector_rpa(H, Ho, x, phi; k = 3, readout = :rpa, max_iter = 400,
                                  curvature = curv, curvature_scale = sc)
        @test ω2c == ω2o && ic.applies == io.applies && ic.f_w == io.f_w
    end
    ω2p, ip = QNTT.sector_rpa(H, Ho, x, phi; k = 3, readout = :rpa, max_iter = 400,
                              precond = :d0)
    @test ip.converged && length(ω2p) == length(ω2o)
    for j in eachindex(ω2o)
        @test ω2p[j] ≈ ω2o[j] rtol = 1e-5
    end
    # J3 — caller errors are `ArgumentError` BEFORE the ccall, raised by the selector validation
    # itself (never the generic "has no field" of an unknown keyword reaching `setfields`), on both
    # readouts; integer codes pass through and the C side refuses a bad one (`QNTTError`).
    sec_err(; kw...) = try
        QNTT.sector_rpa(H, Ho, x, phi; k = 2, max_iter = 200, kw...)
        nothing
    catch e
        e
    end
    for kw in ((readout = :rpa, curvature = :both, curvature_scale = 1e-3),
               (readout = :rpa, curvature = :hard, curvature_scale = 0.0),
               (readout = :rpa, curvature = :hard, curvature_scale = NaN),
               (readout = :tda, curvature = :both, curvature_scale = 1e-3))
        e = sec_err(; kw...)
        @test e isa ArgumentError && occursin("curvature", e.msg) &&
              !occursin("has no field", e.msg)
    end
    for ro in (:rpa, :tda)
        e = sec_err(; readout = ro, precond = :nope)
        @test e isa ArgumentError && occursin("precond", e.msg) && !occursin("has no field", e.msg)
    end
    @test sec_err(; readout = :rpa, curvature = 7, curvature_scale = 1e-3) isa QNTTError
    @test sec_err(; readout = :rpa, precond = 7) isa QNTTError
    # The kernel selector: an unknown Symbol is an ArgumentError naming `kernel` before the call,
    # an unknown integer code the library's refusal.
    e = sec_err(; readout = :rpa, kernel = :nope)
    @test e isa ArgumentError && occursin("kernel", e.msg)
    @test sec_err(; readout = :rpa, kernel = 7) isa QNTTError
    # J4 — the binding allocates n_max = max(max_iter, k + 2) per-mode entries (+ 3 n_max complex):
    # a signed list LONGER than k (every negative plus the k lowest) never errors. The `pg4c`
    # table [Γ3, Γ2, Γ1, A] at (2, 2) — the smallest CURVED shifted channel (φ = 1), here at a
    # normalized random rank-1 point whose dense pencil is Unstable with one negative ω² and one
    # complex pair (measured through the Rust twin `sector_util::pg4c_random_reference(7)`).
    ob4 = UInt8[QNTT.molpro_from_irrep(:C2v, ir) for ir in (3, 2, 1, 0)]
    t4 = zeros(4, 4)
    for p in 1:4
        t4[p, p] = 0.17 * (p - 1 - 2.0)
    end
    v4 = zeros(4, 4, 4, 4)
    for p in 1:4, q in 1:4
        p == q && continue
        w = 1.0 + abs(p - q)
        v4[p, p, q, q] = 0.5 / w
        v4[p, q, q, p] = 0.23 / w
    end
    for idx in ((1, 2, 3, 4), (2, 1, 3, 4), (1, 2, 4, 3), (2, 1, 4, 3),
                (3, 4, 1, 2), (4, 3, 1, 2), (3, 4, 2, 1), (4, 3, 2, 1))
        v4[idx...] = 0.19
    end
    s4(target) = QNTT.QnSpace(4, 2, 2; point_group = :C2v, orbsym = ob4, target = target)
    H4 = QNTT.Hamiltonian(s4(0), t4, v4)
    H4o = QNTT.Hamiltonian(s4(1), t4, v4)
    x4 = QNTT.tt_randn(s4(0), QNTT.uniform_ranks(s4(0), 1); seed = 7)
    x4 = x4 * (1 / norm(x4))
    ω4, i4 = QNTT.sector_rpa(H4, H4o, x4, 1; k = 2, readout = :rpa, kernel = :signed, metric = :ii,
                             max_iter = 40)
    @test i4.dim > 0
    @test length(ω4) > 2            # the premise that makes the n_max sizing necessary
    @test any(<(0), ω4)
    @test length(i4.f_w) == length(ω4) && length(i4.residuals) == length(ω4)
    @test length(i4.krein) == length(ω4)
    @test all(c -> isfinite(c.re) && isfinite(c.im) && c.residual ≥ 0, i4.complex)
    @test i4.metric_indefinite isa Bool && i4.report.n_negative_basis ≥ 0
    # J2 non-vacuity on the CURVED channel: a hard floor between the two largest Schmidt values
    # changes omega^2, so a Symbol path that always sent kind 0 cannot pass the bitwise arms above.
    sch4 = QNTT.schmidt(x4)
    ŵ4 = sort(reduce(vcat, [sqrt.(w) .* σ for (σ, w) in zip(sch4.sigma, sch4.weight)]))
    f4 = sqrt(ŵ4[end] * ŵ4[end - 1])
    ω4o, _ = QNTT.sector_rpa(H4, H4o, x4, 1; k = 2, readout = :rpa, metric = :ii, max_iter = 400)
    for curv in (:hard, :tikhonov)
        ω4c, _ = QNTT.sector_rpa(H4, H4o, x4, 1; k = 2, readout = :rpa, metric = :ii,
                                 max_iter = 400, curvature = curv, curvature_scale = f4)
        @test ω4c != ω4o
    end
    # J2 PREMISE (fixer): `precond = :d0` must REACH the solve. On the small channels above `:d0`
    # exhausts before the warm-up and is bitwise `:off`, so a binding that dropped the keyword
    # would pass them. On a d = 6 table ([Γ2, Γ1, Γ0, Γ2, Γ1, Γ0], twin of the Rust/C `d6`
    # fixtures) at a normalized random rank-1 point, the first channel with dim ≥ 20 iterates past
    # the warm-up and `:d0` changes the answer (values, iterations or applies) on BOTH readouts.
    ob6 = UInt8[QNTT.molpro_from_irrep(:C2v, ir) for ir in (2, 1, 0, 2, 1, 0)]
    t6 = zeros(6, 6)
    for p in 1:6
        t6[p, p] = 0.13 * (p - 1 - 3.0)
    end
    for (p, q) in ((1, 4), (2, 5), (3, 6))
        t6[p, q] = t6[q, p] = -0.8 + 0.1 * (p - 1)
    end
    v6 = zeros(6, 6, 6, 6)
    for p in 1:6, q in 1:6
        p == q && continue
        w = 1.0 + abs(p - q)
        v6[p, p, q, q] = 0.5 / w
        v6[p, q, q, p] = 0.23 / w
    end
    for idx in ((1, 4, 2, 5), (2, 5, 1, 4), (4, 1, 5, 2), (5, 2, 4, 1))
        v6[idx...] = 0.19
    end
    s6(target) = QNTT.QnSpace(6, 2, 2; point_group = :C2v, orbsym = ob6, target = target)
    H6 = QNTT.Hamiltonian(s6(0), t6, v6)
    x6 = QNTT.tt_randn(s6(0), QNTT.uniform_ranks(s6(0), 1); seed = 0x9002)
    x6 = x6 * (1 / norm(x6))
    d0_hit = false
    for phi6 in 1:3
        H6o = QNTT.Hamiltonian(s6(phi6), t6, v6)
        _, i60 = QNTT.sector_rpa(H6, H6o, x6, phi6; k = 0)
        i60.dim < 20 && continue
        for ro in (:rpa, :tda)
            mi = ro === :rpa ? 120 : 60   # :rpa counts pair-equivalents: the same 60 vectors
            vo, io6 = QNTT.sector_rpa(H6, H6o, x6, phi6; k = 2, readout = ro, metric = :ii,
                                      max_iter = mi, precond = :off)
            vd, id6 = QNTT.sector_rpa(H6, H6o, x6, phi6; k = 2, readout = ro, metric = :ii,
                                      max_iter = mi, precond = :d0)
            @info "J2 d6 premise" phi6 ro i60.dim io6.iterations id6.iterations io6.applies id6.applies
            @test vo != vd || io6.iterations != id6.iterations || io6.applies != id6.applies
        end
        d0_hit = true
        break
    end
    @test d0_hit
    # rpa-floor-selection P3: `max_subspace` reaches the :rpa solve, on the same d = 6 channel
    # (dim ≥ 20; the small channels above exhaust below any bound). The kernel's auto bound
    # min(max(8(k+b), 32), dim) passed explicitly is the default (0) call; the tightest admitted
    # bound 2(k+b), b = k + 2, restarts the basis — a dropped field would reproduce the default
    # call — and the C ABI test pins that call to the Rust readout bit for bit.
    ms_hit = false
    for phi6 in 1:3
        H6o = QNTT.Hamiltonian(s6(phi6), t6, v6)
        _, i60 = QNTT.sector_rpa(H6, H6o, x6, phi6; k = 0)
        i60.dim < 20 && continue
        k1 = 1
        auto = min(max(8 * (k1 + k1 + 2), 32), i60.dim)
        tight = 2 * (k1 + k1 + 2)
        call(ms) = QNTT.sector_rpa(H6, H6o, x6, phi6; k = k1, readout = :rpa, metric = :ii,
                                   max_iter = 120, max_subspace = ms)
        w0, j0 = call(0)
        we, je = call(auto)
        wt, jt = call(tight)
        @info "P3 d6 max_subspace" phi6 i60.dim auto tight j0.iterations jt.iterations
        @test we == w0 && je.iterations == j0.iterations && je.applies == j0.applies
        @test wt != w0 || jt.iterations != j0.iterations || jt.applies != j0.applies
        ms_hit = true
        break
    end
    @test ms_hit

    # --- cross-sector OBSERVABLES (Phase F) -------------------------------------------------------
    # Two options structs, not one: `tol`/`max_iter`/`sym_assert_rel` are read by `sector_alpha`
    # only and `degeneracy_tol`/`psd_tol` by `sector_transition_moments` only, so no field is inert
    # on the entry point that receives it. Both come from the library's own defaults.
    adflt = QNTT.defaults(QNTT.SectorAlphaOptions)
    mdflt = QNTT.defaults(QNTT.SectorMomentOptions)
    # EVERY field, pinned by VALUE against the Rust defaults (`SectorStaticOpts::default` /
    # `SectorMomentOpts::default`, which the C `*_options_default` exporters build field by
    # field). The values are pairwise distinct within each struct, so this is the layout check
    # these bindings rely on: swap two same-typed fields in the Julia mirror — `tol` with
    # `tt_grad_tol`, `degeneracy_tol` with `psd_tol`, `route` with `reserved` — and it fails.
    @test (adflt.tol, adflt.max_iter, adflt.tt_grad_tol) == (1e-8, Csize_t(500), 1e-6)
    @test adflt.sym_assert_rel == 0.0           # unread while the flag is clear, hence 0.0
    @test adflt.has_sym_assert_rel == Cint(0)   # => the library recomputes 100*tol from the tol
    @test adflt.route == QNTT.PROP_ROUTES[:materialized]
    @test (mdflt.tt_grad_tol, mdflt.degeneracy_tol, mdflt.psd_tol) == (1e-6, 1e-8, 1e-10)
    @test mdflt.route == QNTT.PROP_ROUTES[:materialized]
    @test mdflt.reserved == Cint(0)             # explicit tail padding: must stay 0
    # And the field ORDER itself, so a mirror reordered with the same types is caught too.
    @test fieldnames(QNTT.SectorAlphaOptions) ==
          (:tol, :max_iter, :tt_grad_tol, :sym_assert_rel, :has_sym_assert_rel, :route)
    @test fieldnames(QNTT.SectorMomentOptions) ==
          (:tt_grad_tol, :degeneracy_tol, :psd_tol, :route, :reserved)
    ao = QNTT.setfields(adflt; route = :materialized, tol = 1e-9)
    @test (ao.route, ao.tol) == (Cint(1), 1e-9)
    @test_throws ArgumentError QNTT.setfields(adflt; route = :nope)
    @test_throws ArgumentError QNTT.setfields(mdflt; route = :nope)

    # The IRREP FLUX IS DERIVED FROM THE MATRIX, never from an axis table: on this table the pairs
    # (p, q) with Γ_p ⊗ Γ_q = 3 are (2, 3) and (3, 4) in 1-based site order, so a symmetric matrix
    # supported there — and with a ZERO DIAGONAL, since a diagonal cell is same-irrep — is a
    # representable flux-3 observable. Synthetic: no shipped fixture with a nontrivial point group
    # carries a dipole matrix, and the docstrings say "supply your own".
    mu = zeros(d, d)
    mu[2, 3] = mu[3, 2] = 0.75
    mu[3, 4] = mu[4, 3] = -0.4375
    @test mu == mu'

    α, ia = QNTT.sector_alpha(H, Ho, x, phi, mu)
    @test size(α) == (1, 1)
    @test α ≈ α'
    @test ia.dim == dims[phi]
    @test ia.phi == phi && ia.target == xor(0, phi)
    @test 0 ≤ ia.capture_defect[1] ≤ 1
    @test ia.iterations[1] > 0          # anti-vacuity: an in-channel component really solved
    @test ia.grad_norm[1] > 0
    @test isfinite(ia.alpha_defect) && isfinite(ia.sigma_min)
    @info "SYNTHETIC channel alpha" phi α = α[1, 1] ia.capture_defect ia.sigma_min ia.dim

    ω, T, im_ = QNTT.sector_transition_moments(H, Ho, x, phi, mu)
    @test length(ω) == size(T, 2)
    @test size(T, 1) == 1
    @test issorted(ω)
    @test im_.complete_basis            # the DENSE route: the sums are genuine, not partial
    @test im_.metric ∈ (:ii, :rr, :auto)
    @test im_.dim == dims[phi]
    @test all(isfinite, im_.oscillator)
    # `clusters` is converted to 1-BASED Julia ranges (the one index-convention conversion in the
    # bindings) and must partition 1:length(ω), singletons included.
    @test im_.clusters isa Vector{UnitRange{Int}}
    @test vcat(collect.(im_.clusters)...) == collect(1:length(ω))
    # An imaginary mode is never a real ω, and it depletes the resolution of the identity.
    @test all(<(0), im_.unstable_omega2)
    @test (im_.n_neg > 0) == (im_.alpha_sos === nothing)
    if im_.alpha_sos !== nothing
        # α_SOS is an ALGEBRAIC identity at any rank, hence the MUTUAL cross-check; the budget is
        # the completeness and orthogonality defects, printed rather than guessed.
        budget = 1e-6 + 100 * (im_.completeness_defect[1] + im_.orth_error)
        @info "alpha_sos vs alpha" budget im_.completeness_defect im_.orth_error
        @test im_.alpha_sos ≈ α rtol = budget
    end

    # The two-pass route (`dim = nothing`) and the one-solve route (`dim` from sector_rpa at k = 0)
    # are the same answer exactly, not to a tolerance.
    ω1, T1, i1 = QNTT.sector_transition_moments(H, Ho, x, phi, mu; dim = dims[phi])
    @test ω1 == ω && abs.(T1) == abs.(T)
    @test i1.n_modes == im_.n_modes && i1.clusters == im_.clusters

    # A component whose SINGLE derived flux is not `phi` is a STRUCTURAL ZERO row and column with
    # no work paid — not an error. A caller looping the same components over every channel is the
    # intended usage.
    mu1 = zeros(d, d)
    mu1[1, 2] = mu1[2, 1] = 0.6      # xor(Γ_1, Γ_2) = 1, not phi = 3
    α2, ia2 = QNTT.sector_alpha(H, Ho, x, phi, [mu, mu1])
    @test size(α2) == (2, 2)
    @test α2[1, 2] == 0.0 && α2[2, 1] == 0.0 && α2[2, 2] == 0.0
    @test α2[1, 1] != 0.0
    @test ia2.capture_defect[2] == 1.0 && ia2.grad_norm[2] == 0.0 && ia2.iterations[2] == 0

    # Refusals throw QNTTError, never a Julia-level crash and never a partial number.
    @test_throws QNTTError QNTT.sector_alpha(H, Ho, x, 0, mu)          # phi = 0 is in-sector
    @test_throws QNTTError QNTT.sector_alpha(H, H, x, phi, mu)         # H_out re-stamped
    @test_throws QNTTError QNTT.sector_alpha(H, Ho, x, phi, [mu, mu, mu, mu])  # 4 components
    @test_throws QNTTError QNTT.sector_alpha(H, Ho, x, phi, mu; route = :shifted)
    @test_throws QNTTError QNTT.sector_transition_moments(H, Ho, x, phi, mu; route = :shifted)
    # A matrix carrying MORE THAN ONE flux class has no single output space: split it by class.
    mumulti = zeros(d, d)
    mumulti[1, 2] = mumulti[2, 1] = 0.75    # flux 1
    mumulti[1, 3] = mumulti[3, 1] = -0.5    # flux 2
    @test_throws QNTTError QNTT.sector_alpha(H, Ho, x, phi, mumulti)
    @test_throws QNTTError QNTT.sector_transition_moments(H, Ho, x, phi, mumulti)

    # The refusal DISCRIMINANT reaches the caller, not just the message: `err.refusal` is the
    # wire code, `0` on a caller error. The gate is driven by TIGHTENING `tt_grad_tol` against the
    # ALS endpoint's own gradient (the capi arm's own construction), so the refusal is a property
    # of the OPTION and not of an optimizer trajectory.
    for f in (() -> QNTT.sector_alpha(H, Ho, x, phi, mu; tt_grad_tol = 1e-300),
              () -> QNTT.sector_transition_moments(H, Ho, x, phi, mu; tt_grad_tol = 1e-300))
        err = try
            f()
            nothing
        catch e
            e
        end
        @test err isa QNTTError
        @test err.code == 5                    # QNTT_NOT_CONVERGED, never a new status
        @test err.refusal == 1
        @test QNTT.SECTOR_REFUSALS[Cint(err.refusal)] === :not_tt_critical
        @test occursin("not TT-critical", err.msg)
        @test occursin("not_tt_critical", sprint(showerror, err))
    end
    # A CALLER error is not a physics refusal: the field stays 0.
    let err = try
            QNTT.sector_alpha(H, Ho, x, phi, [mu, mu, mu, mu])
            nothing
        catch e
            e
        end
        @test err isa QNTTError && err.code == 2 && err.refusal == 0
        @test QNTT.SECTOR_REFUSALS[Cint(err.refusal)] === :none
    end
    # The nine wire codes, mirrored from qntt.h — `6` is deliberately absent (a multi-class flux
    # is a caller error and arrives as status 2 with `:none`).
    @test sort(collect(keys(QNTT.SECTOR_REFUSALS))) == Cint[0, 1, 2, 3, 4, 5, 7, 8, 9]

    # `dim = 0` is a KNOWN dimension, not an absent one: on the LIVE channel it is simply a wrong
    # size, and the library refuses it rather than silently probing (the `nothing` route is the
    # only probe trigger).
    @test_throws QNTTError QNTT.sector_transition_moments(H, Ho, x, phi, mu; dim = 0)

    # A LABEL-LIFETIME arm, and the only way to notice a lost `GC.@preserve`: a `Vector{Cstring}`
    # of `pointer(s)` values holds NO reference to the strings themselves, so a collection between
    # building the vector and the ccall would hand the library freed bytes. The refusal message of a
    # deliberately multi-class NAMED component is what proves the name survived the round trip.
    let caught = nothing
        GC.gc()
        try
            QNTT.sector_alpha(H, Ho, x, phi, mumulti; labels = ["dipole_z"])
        catch e
            caught = e
        end
        GC.gc()
        @test caught isa QNTTError
        @test occursin("dipole_z", caught.msg)
    end
end

@testset "state diagnostics (rho, Schmidt, gradient)" begin
    data = QNTT.read_fcidump(h4())
    s = QNTT.QnSpace(data)
    H = QNTT.Hamiltonian(s, data)
    x0 = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 4); seed = 31)
    λ, x = QNTT.als(H, x0; eps = 1e-10, max_iter = 50)

    # --- rho: the vacuity guard -----------------------------------------------------------------
    @test QNTT.rho(x) ≈ 1.0 rtol = 1e-12          # a Float64 state is real by construction
    @test QNTT.bilinear_dot(x, x) == dot(x, x)    # over Float64 the two contractions are one map

    xc = QNTT.to_c64(x)
    @test QNTT.rho(xc) ≈ 1.0 rtol = 1e-12         # ... and so is its real embedding
    # A GLOBAL PHASE must not move rho. This is exactly why rho is used instead of ‖Im x‖, which
    # this same rotation sends from 0 to ‖x‖.
    @test QNTT.rho((0.6 + 0.8im) * xc) ≈ 1.0 rtol = 1e-12
    @test norm(imag.(Array((0.6 + 0.8im) * xc))) > 0.1 * norm(xc)   # the substitute would lie here

    # A genuinely complex state reads strictly below 1.
    y = QNTT.to_c64(QNTT.tt_randn(s, QNTT.uniform_ranks(s, 4); seed = 32))
    z = xc + im * y
    @test QNTT.rho(z) < 1 - 1e-6

    # --- Schmidt spectrum and its sum rule ------------------------------------------------------
    # THE identity, at every interior bond of a unit-norm state:  sum(w .* σ.^2) == 1.
    # Abelian is the w ≡ 1 special case, not a separate fact.
    xn = x / norm(x)
    sp = QNTT.schmidt(xn)
    @test length(sp.sigma) == s.d - 1             # interior bonds only
    @test length(sp.weight) == length(sp.sigma)
    for (σ, w) in zip(sp.sigma, sp.weight)
        @test !isempty(σ)
        @test issorted(σ; rev = true)
        @test all(>(0), σ)
        @test all(==(1.0), w)                     # no multiplet degeneracy on the abelian layer
        @test sum(w .* σ .^ 2) ≈ 1.0 rtol = 1e-12
        @test sum(abs2, σ) ≈ 1.0 rtol = 1e-12     # hence the ordinary Σσ² = 1 here
    end
    @test QNTT.sigma_min(xn) == minimum(minimum.(sp.sigma))
    @test 0 < QNTT.sigma_min(xn) ≤ 1 + 1e-10

    # The SAME rule on SU(2), where the weight is a genuine multiplet degeneracy
    # centerweight(2s_col, 2S_total) = (2s_col+1)/(2S_total+1). A SINGLET ALONE CANNOT TEST THIS:
    # at 2S = 0 the denominator is 1, so (2s+1) and (2s+1)/(2S+1) are indistinguishable and a
    # wrong formula would pass. Hence the non-singlet arms, and the explicit check that undoing
    # the denominator gives 2S+1 rather than 1.
    for (n, two_s) in ((4, 0), (4, 2), (6, 4))
        su2 = QNTT.Su2Space(6, n, two_s)
        xs = QNTT.tt_randn(su2, QNTT.uniform_ranks(su2, 4); seed = 5)
        xsn = xs / norm(xs)
        sps = QNTT.schmidt(xsn)
        @test length(sps.sigma) == su2.d - 1
        for (σ, w) in zip(sps.sigma, sps.weight)
            @test issorted(σ; rev = true) && all(>(0), σ)
            @test sum(w .* σ .^ 2) ≈ 1.0 rtol = 1e-12          # the identity
            @test sum(w .* (two_s + 1) .* σ .^ 2) ≈ two_s + 1.0 rtol = 1e-12  # denominator pinned
        end
        two_s == 0 || @test any(any(!=(1.0), w) for w in sps.weight)  # the weighting is exercised
        # The UNWEIGHTED sum is not a normalization check, and its direction is not even fixed.
        for σ in sps.sigma
            two_s == 0 ? (@test sum(abs2, σ) ≤ 1 + 1e-12) : (@test sum(abs2, σ) > 1)
        end
        @test 0 < QNTT.sigma_min(xsn) ≤ 1 + 1e-10
    end

    # --- gradient norm --------------------------------------------------------------------------
    g_converged = QNTT.grad_norm(H, x)
    g_random = QNTT.grad_norm(H, x0)
    @test g_random > g_converged
    @test g_converged < 1e-6                      # full rank on H4: ALS reaches the eigenvector
    @test QNTT.rayleigh_quotient(H, x) ≈ λ rtol = 1e-8
end

@testset "state cache (save/load)" begin
    dir = mktempdir()
    try
        # Abelian Float64: the round trip must reproduce the state, its space and its rank grid.
        s = QNTT.QnSpace(6, 3, 3)
        x = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 3); seed = 2026)
        path = joinpath(dir, "abelian.tt")
        QNTT.save(x, path)
        @test isfile(path)

        y = QNTT.load(Float64, QNTT.Abelian, path)
        @test QNTT.space(y) == s
        @test QNTT.ranks(y) == QNTT.ranks(x)
        # Coefficients round-trip bit-for-bit, so the only slack is the difference contraction.
        @test norm(x - y) ≤ 1e-13 * norm(x)
        @test QNTT.orthogonality_center(y) == QNTT.orthogonality_center(x)

        # Both spellings of the type-directed form agree.
        z = QNTT.load(QNTT.TTVector{Float64,QNTT.Abelian}, path)
        @test norm(x - z) ≤ 1e-13 * norm(x)
        @test norm(x - QNTT.load(path)) ≤ 1e-13 * norm(x)      # the Float64/abelian default

        # The file records its layer and scalar; the wrong family is an error, never a
        # reinterpretation.
        @test_throws QNTTError QNTT.load(Float64, QNTT.SU2, path)
        @test_throws QNTTError QNTT.load(ComplexF64, QNTT.Abelian, path)

        # SU(2) × ComplexF64 rides the same generated family.
        su2 = QNTT.Su2Space(4, 4, 0)
        c = QNTT.tt_randn(ComplexF64, su2, QNTT.uniform_ranks(su2, 4); seed = 11)
        cpath = joinpath(dir, "su2.tt")
        QNTT.save(c, cpath)
        c2 = QNTT.load(ComplexF64, QNTT.SU2, cpath)
        @test eltype(c2) == ComplexF64
        @test QNTT.space(c2) == su2
        @test norm(c - c2) ≤ 1e-13 * norm(c)

        # A file that is not a state file at all is refused by the magic check, not decoded.
        junk = joinpath(dir, "junk.tt")
        write(junk, "definitely not a qntt state file")
        @test_throws QNTTError QNTT.load(Float64, QNTT.Abelian, junk)
        @test_throws QNTTError QNTT.load(Float64, QNTT.Abelian, joinpath(dir, "absent.tt"))
    finally
        rm(dir; recursive = true, force = true)
    end
end

@testset "PD probe and the complex seed" begin
    data = QNTT.read_fcidump(h4())
    s = QNTT.QnSpace(data)
    H = QNTT.Hamiltonian(s, data)
    x0 = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 2); seed = 7)
    _, x = QNTT.als(H, x0; eps = 1e-10, max_iter = 50)

    for part in (:rr, :ii)
        r = QNTT.pd_probe(H, x; part = part, budget = 40, seed = 1234)
        # A verdict is a diagnosis about the state, not a status of the call: none of the three
        # throws, and `:witness_negative` in particular is the GOAL of the `:ii` probe.
        @test r.verdict in (:witness_negative, :likely_pd, :inconclusive)
        @test r.applies > 0
        # The tri-state must not collapse.
        @test r.is_pd === (r.verdict === :likely_pd ? true :
                           r.verdict === :witness_negative ? false : nothing)
        if r.verdict === :witness_negative
            @test r.lambda < 0
            @test isnan(r.residual)                 # unspecified on a witness
        else
            @test r.lambda ≥ 0
            @test isfinite(r.residual) && r.residual ≥ 0
        end
    end
    @test_throws ArgumentError QNTT.pd_probe(H, x; part = :nonsense)

    # The f64 → C64 embedding touches nothing but the field.
    xc = QNTT.to_c64(x)
    @test eltype(xc) == ComplexF64
    @test QNTT.ranks(xc) == QNTT.ranks(x)
    @test norm(xc) == norm(x)
    @test Array(xc) == complex.(Array(x))
    # The orthogonality centre crosses too. It stays true (a real column-orthonormal frame is
    # orthonormal under the Hermitian inner product), and it is load-bearing: with it set, `norm`
    # is the centre-core shortcut rather than a self-contraction.
    @test QNTT.orthogonality_center(xc) == QNTT.orthogonality_center(x)
    @test QNTT.orthogonality_center(x) !== nothing        # `als` returns a canonical state

    # The complex seed either leaves the real manifold or says why it cannot. On a small stable
    # fixture the refusal is the expected, honest branch — assert whichever happens, not one.
    res = try
        QNTT.complex_seed(H, x; eps = 1e-3, budget = 60, seed = 99)
    catch e
        e
    end
    if res isa QNTTError
        @test res.code == 5                                   # QNTT_NOT_CONVERGED
        @test occursin("H^ii", res.msg)
    else
        y, λ = res
        @test eltype(y) == ComplexF64
        @test λ < 0                                           # a seed implies a proven witness
        @test norm(y) ≈ 1.0 rtol = 1e-10
        @test isvalid(y)
        # ρ = |(x,x)| / ⟨x,x⟩ < 1 is the proof it actually left the real manifold.
        @test abs(sum(Array(y) .^ 2)) / sum(abs2, Array(y)) < 1 - 1e-9
    end
    @test_throws QNTTError QNTT.complex_seed(H, x; eps = 0.0)
end

@testset "MALS conv_tol separates the stop from the truncation" begin
    data = QNTT.read_fcidump(h4())
    s = QNTT.QnSpace(data)
    H = QNTT.Hamiltonian(s, data)
    x0 = QNTT.tt_state([true, true, false, false], [true, true, false, false], s)
    # Loose eps alone stops almost at once; the same eps swept under a tight conv_tol keeps going
    # and ends at least as low. Same seed and schedule, so the only difference is the stop test.
    λa, _, ha = QNTT.mals(H, x0; bond_dims = [0], noise = [1e-3, 0.0], eps = 1e-2, max_iter = 30)
    λb, _, hb = QNTT.mals(H, x0; bond_dims = [0], noise = [1e-3, 0.0], eps = 1e-2,
                          conv_tol = 1e-12, max_iter = 30)
    @test length(hb) ≥ length(ha)
    @test λb ≤ λa + 1e-10
    @test QNTT.defaults(QNTT.MalsOptions).conv_tol == 0.0
    @test_throws QNTTError QNTT.mals(H, x0; conv_tol = -1.0)
end

@testset "CASSCF stack, saddle seed and GAD" begin
    data = QNTT.read_fcidump(h4())
    cas = (1, 2, 1)
    # The curvature keyword's wire table mirrors `QNTT_CURVATURE_REG_*` (rpa-floor-selection P1).
    @test QNTT.CURVATURE_REG_CODES == Dict(:off => Cint(0), :hard => Cint(1), :tikhonov => Cint(2))
    for (mk, L) in ((QNTT.QnSpace, QNTT.Abelian), (d -> QNTT.Su2Space(d), QNTT.SU2))
        s = mk(data)
        H = QNTT.Hamiltonian(s, data)
        λ0, x0 = QNTT.cas_reference_state(H, cas; m_active = 0, seed = 7)
        @test QNTT.layer(x0) === L
        @test isfinite(λ0)
        @test_throws QNTTError QNTT.cas_reference_state(H, (1, 1, 1))     # does not cover d

        λ, x, u, H_rot, info = QNTT.casscf(data, x0, cas; orb_max_iter = 20, preserve_grid = true)
        @test λ ≤ λ0 + 1e-12
        @test size(u) == (4, 4)
        @test u' * u ≈ I atol = 1e-10
        @test info.converged isa Bool && info.iters isa Int
        @test info.refused_reason === nothing || info.refused_reason isa String
        @test length(info.g_orb_kept) == length(info.e_history)
        @test !haskey(info, :wall)
        @test all(all(r .≤ r0) for (r, r0) in zip(QNTT.ranks(x), QNTT.ranks(x0)))
        @test QNTT.rayleigh_quotient(H_rot, x) ≈ λ rtol = 1e-8

        ω, resp = QNTT.joint_rpa(H_rot, x, cas)
        @test resp.status in (:stable, :resonant, :unstable, :refused)
        @test resp.message isa String && !isempty(resp.message)
        @test length(resp.f_w) == length(ω)
        if info.converged
            @test resp.status !== :refused
            @test resp.n_eff == info.n_eff && resp.n_modes ≥ resp.n_eff
            if resp.status === :stable
                @test issorted(ω) && first(ω) > 0
                @test all(f -> f ≥ 0, resp.f_w)          # a ratio of magnitudes
            end
        end

        # The signed matrix-free solver at the same point, under every metric (side) request: it
        # never refuses, so each call returns; `k` counts the NON-NEGATIVE entries; only the side
        # the f_W ratio reads moves with the request.
        for metric in (:auto, :ii, :rr)
            ω2, ri = QNTT.rpa_signed(H_rot, x; k = 3, metric, tol = 1e-6, max_iter = 800,
                                     seed = 1)
            n = length(ω2)
            @test n > 0 && issorted(ω2) && count(≥(0), ω2) ≤ 3
            @test ri.metric === (metric === :rr ? :rr : :ii)
            @test ri.negatives_certified isa Bool && ri.converged isa Bool
            @test length(ri.f_w) == n && all(f -> f ≥ 0, ri.f_w)
            @test length(ri.residuals) == n && all(r -> isfinite(r) && r ≥ 0, ri.residuals)
            @test length(ri.krein) == n && all(m -> isfinite(m) && abs(m) ≤ 1 + 1e-12, ri.krein)
            @test ri.kernel === :paired
            @test ri.report.metric_events === nothing && ri.report.neg_apb isa Integer
            if ri.report.neg_apb + ri.report.neg_amb == 0
                # No counting offset: every real mode an excitation, no complex mode.
                @test all(((w, m),) -> w < 0 || m > 0, zip(ω2, ri.krein))
                @test isempty(ri.complex)
            end
            @test ri.applies > 0
        end
        # The default request is `:ii` with no probe (2026-10-18): bitwise the explicit `:ii` call.
        # Tells the default apart from `:rr` at every point, from `:auto` only where its probe swaps.
        ω2d, rd = QNTT.rpa_signed(H_rot, x; k = 3, tol = 1e-6, max_iter = 800, seed = 1)
        ω2i, rii = QNTT.rpa_signed(H_rot, x; k = 3, metric = :ii, tol = 1e-6, max_iter = 800,
                                   seed = 1)
        @test rd.metric === :ii && ω2d == ω2i && rd.applies == rii.applies
        @test_throws ArgumentError QNTT.rpa_signed(H_rot, x; k = 3, metric = :both)

        # The kernel selector (rpa-followup B3): `kernel = :signed` runs the W-metric Davidson on
        # the lane `metric` names (`:auto` = `:ii`, no probe: bitwise the `:ii` call); the default
        # is `:paired`; a bad Symbol is an `ArgumentError`, a bad integer code the C side's refusal.
        sig = Dict{Symbol,Any}()
        for metric in (:auto, :ii, :rr)
            ω2s, rs = QNTT.rpa_signed(H_rot, x; k = 3, kernel = :signed, metric, tol = 1e-6,
                                      max_iter = 800, seed = 1)
            @test rs.kernel === :signed
            @test rs.metric === (metric === :rr ? :rr : :ii)
            @test rs.report.neg_apb === nothing && rs.report.neg_amb === nothing
            @test rs.report.metric_events isa Integer && rs.report.general_path isa Bool
            @test length(rs.f_w) == length(ω2s) && rs.applies > 0
            sig[metric] = (ω2s, rs)
        end
        @test sig[:auto][1] == sig[:ii][1] && sig[:auto][2].applies == sig[:ii][2].applies
        ω2p, rp = QNTT.rpa_signed(H_rot, x; k = 3, kernel = :paired, tol = 1e-6, max_iter = 800,
                                  seed = 1)
        @test rp.kernel === :paired && ω2p == ω2d && rp.applies == rd.applies
        @test_throws ArgumentError QNTT.rpa_signed(H_rot, x; k = 3, kernel = :both)
        @test_throws QNTTError QNTT.rpa_signed(H_rot, x; k = 3, kernel = 7)

        # Curvature regularization (rpa-floor-selection P1 §1.3): `curvature`/`curvature_scale`
        # mirror the C pair. J1 — the default is `:off`, bitwise, and `:off` ignores its scale
        # (NaN included, the C contract).
        ω2o, ro = QNTT.rpa_signed(H_rot, x; k = 3, curvature = :off, tol = 1e-6, max_iter = 800,
                                  seed = 1)
        @test ω2o == ω2d && ro.applies == rd.applies
        ω2n, rn = QNTT.rpa_signed(H_rot, x; k = 3, curvature = :off, curvature_scale = NaN,
                                  tol = 1e-6, max_iter = 800, seed = 1)
        @test ω2n == ω2d && rn.applies == rd.applies
        # J2 — `:hard`/`:tikhonov` run and return the structural contract; an `:auto` request on a
        # regularized context runs `:ii` with no probe (bitwise the `:ii` call); a hard floor below
        # every RAW σ of `QNTT.schmidt` (the scale's unit on this layer, not √weight·σ) is the
        # identity projector, i.e. `:off` bitwise.
        sch = QNTT.schmidt(x)
        σs = sort(reduce(vcat, sch.sigma))
        s_mid = σs[cld(length(σs), 2)]
        for curv in (:hard, :tikhonov)
            ω2c, rc = QNTT.rpa_signed(H_rot, x; k = 3, curvature = curv, curvature_scale = s_mid,
                                      tol = 1e-6, max_iter = 800, seed = 1)
            n = length(ω2c)
            @test n > 0 && issorted(ω2c) && count(≥(0), ω2c) ≤ 3
            @test rc.metric === :ii
            @test length(rc.f_w) == n && all(f -> f ≥ 0, rc.f_w)
            @test length(rc.residuals) == n && length(rc.krein) == n && rc.applies > 0
            ω2a, ra = QNTT.rpa_signed(H_rot, x; k = 3, metric = :auto, curvature = curv,
                                      curvature_scale = s_mid, tol = 1e-6, max_iter = 800, seed = 1)
            @test ra.metric === :ii && ω2a == ω2c && ra.applies == rc.applies
        end
        ω2b, rb = QNTT.rpa_signed(H_rot, x; k = 3, curvature = :hard,
                                  curvature_scale = 0.5 * first(σs), tol = 1e-6, max_iter = 800,
                                  seed = 1)
        @test ω2b == ω2d && rb.applies == rd.applies && rb.f_w == rd.f_w
        # J2 non-vacuity (fix round 2026-10-19): at this pinned CAS point the regularization was
        # measured INERT (ω² bitwise `:off` for both kinds at the then-ŵ median — the touched columns
        # carry no tangent room; not re-measured at the raw median since P4b), so a Symbol path that
        # always sent kind 0 could pass every arm above.
        # Pinned on a normalized RANDOM rank-2 point of the same H (the C ABI C2 fixture's shape):
        # a floor between the two largest raw σ changes ω² for both kinds.
        xr = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 2); seed = 11)
        xr = xr * (1 / norm(xr))
        schr = QNTT.schmidt(xr)
        σr = sort(reduce(vcat, schr.sigma))
        f_top = sqrt(σr[end] * σr[end - 1])
        ω2ro, _ = QNTT.rpa_signed(H, xr; k = 3, tol = 1e-6, max_iter = 800, seed = 1)
        for curv in (:hard, :tikhonov)
            ω2rc, _ = QNTT.rpa_signed(H, xr; k = 3, curvature = curv, curvature_scale = f_top,
                                      tol = 1e-6, max_iter = 800, seed = 1)
            @test ω2rc != ω2ro
        end
        # J3 — caller errors are `ArgumentError` before the ccall, raised by the curvature
        # validation itself (never the generic "has no field" of an unknown keyword reaching
        # `setfields`); integer codes pass through and the C side refuses them (`QNTTError`).
        curv_err(; kw...) = try
            QNTT.rpa_signed(H_rot, x; k = 3, tol = 1e-6, max_iter = 800, seed = 1, kw...)
            nothing
        catch e
            e
        end
        for kw in ((curvature = :both, curvature_scale = 1e-3),
                   (curvature = :hard, curvature_scale = 0.0),
                   (curvature = :hard, curvature_scale = NaN),
                   (curvature = :tikhonov, curvature_scale = -1.0))
            e = curv_err(; kw...)
            @test e isa ArgumentError && occursin("curvature", e.msg) &&
                  !occursin("has no field", e.msg)
        end
        @test curv_err(; curvature = 7, curvature_scale = 1e-3) isa QNTTError
        @test curv_err(; curvature = 1, curvature_scale = 0.0) isa QNTTError

        xk, λw = QNTT.saddle_seed(H_rot, x; eps = 5e-2, budget = 40, seed = 3)
        @test isfinite(λw) && QNTT.grad_norm(H_rot, xk) > 1e-6
        @test_throws QNTTError QNTT.saddle_seed(H_rot, x; eps = 0.0)

        xe, g = QNTT.gad(H_rot, xk; max_steps = 40, w_model = :damped, certify = true,
                         probe_budget = 40, preserve_grid = true, seed = 3)
        @test g.steps ≤ 40 && isfinite(g.gnorm) && isfinite(g.lambda)
        @test g.index_estimate === nothing || g.index_estimate isa Int
        @test g.eig1 === nothing || g.eig1 isa Float64
        @test !haskey(g, :wall)
        g.converged && @test g.index1_certified && g.index_estimate == 1
        @test all(all(r .≤ r0) for (r, r0) in zip(QNTT.ranks(xe), QNTT.ranks(xk)))
        @test g.grid_changes == 0 && g.tracked_mode_is_lowest isa Bool
        @test g.witness_overlap === nothing || 0 ≤ g.witness_overlap ≤ 1 + 1e-12
        @test_throws ArgumentError QNTT.gad(H_rot, xk; w_model = :nonsense)

        # Rank adaptation over the ABI: :both never opens a launch-zero (pinned CAS) sector,
        # :shrink never exceeds the launch grid, bad knobs are refused.
        xb, gb = QNTT.gad(H_rot, xk; max_steps = 20, certify = false, preserve_grid = true,
                          rank_adapt = :both, rank_eps = 1e-8, seed = 3)
        @test gb.steps ≥ 1 && gb.grid_changes isa Int
        @test all(all((r0 .> 0) .| (r .== 0)) for (r, r0) in zip(QNTT.ranks(xb), QNTT.ranks(xk)))
        xs, gs = QNTT.gad(H_rot, xk; max_steps = 20, certify = false, preserve_grid = true,
                          rank_adapt = :shrink, rank_eps = 1e-6, seed = 3)
        @test all(all(r .≤ r0) for (r, r0) in zip(QNTT.ranks(xs), QNTT.ranks(xk)))
        @test_throws ArgumentError QNTT.gad(H_rot, xk; rank_adapt = :nonsense)
        @test_throws QNTTError QNTT.gad(H_rot, xk; rank_adapt = :both, rank_eps = 0.0)

        # Border policy: the default refuses at the border and reports it; an explicit gain
        # ceiling runs; a non-positive one is refused.
        @test g.border_verdict isa Bool && (g.b_gain_end === nothing || g.b_gain_end ≥ 0)
        xt, gt = QNTT.gad(H_rot, xk; max_steps = 20, certify = false, preserve_grid = true,
                          border_gain_max = 0.5, seed = 3)
        @test gt.steps ≥ 1 && gt.border_verdict isa Bool
        @test_throws QNTTError QNTT.gad(H_rot, xk; border_gain_max = 0.0)
    end
end

@testset "excited states" begin
    data = QNTT.read_fcidump(h4())
    s = QNTT.QnSpace(data)
    H = QNTT.Hamiltonian(s, data)
    x0 = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 4); seed = 2024)

    # --- the dense oracle -----------------------------------------------------------------------
    # Every determinant of a particle sector, the exact matrix in that basis (one matvec per
    # column), and its spectrum. This is the reference every root-resolved claim below is made
    # against; it is self-checked against the suite's pinned ground energy, and each arm asserts
    # the FIXTURE GAP it needs, so a fixture swap fails loudly rather than making a convergence
    # claim vacuous.
    occs(d, n) = [[((m >> (i - 1)) & 1) == 1 for i in 1:d]
                  for m in 0:(2^d - 1) if count_ones(m) == n]
    function sector_levels(nup, ndn)
        sp = QNTT.QnSpace(QNTT.norb(data), nup, ndn)
        Hs = QNTT.Hamiltonian(sp, data)
        dets = vec([QNTT.tt_state(u, dn)
                    for u in occs(QNTT.norb(data), nup), dn in occs(QNTT.norb(data), ndn)])
        M = zeros(length(dets), length(dets))
        for (j, ej) in enumerate(dets)
            hej = Hs * ej
            for (i, ei) in enumerate(dets)
                M[i, j] = dot(ei, hej)
            end
        end
        @test maximum(abs, M - M') ≤ 1e-9              # the reference is Hermitian
        return eigvals(Symmetric((M + M') / 2))
    end

    E = sector_levels(2, 2)
    E31 = sector_levels(3, 1)
    # S = 0 = the (2,2) multiset minus the (3,1) multiset (the `tests/gad/` pattern).
    s0 = let rest = copy(E31), out = Float64[]
        for lvl in E
            k = findfirst(y -> abs(y - lvl) < 1e-8, rest)
            k === nothing ? push!(out, lvl) : deleteat!(rest, k)
        end
        @test isempty(rest)                            # every higher-Sz level has a partner
        out
    end

    @test E[1] ≈ -4.459488695954761 atol = 1e-10       # the oracle reproduces the suite's pin
    @test length(E) == 36 && length(s0) == 20
    @test E[3] - E[2] ≥ 0.05                           # the abelian nroots = 2 excluded gap
    @test E[4] - E[3] ≥ 0.05                           # ... and the nroots = 3 one
    @test s0[4] - s0[3] ≥ 0.2                          # the SU(2) nroots = 3 excluded gap

    # --- defaults are the C fillers, which are the Rust defaults --------------------------------
    dsa = QNTT.defaults(QNTT.StateAverageOptions)
    @test dsa.mals == QNTT.defaults(QNTT.MalsOptions)
    @test dsa.homing == 0                                # label by energy at every bond
    @test dsa.reorthonormalize == 1                      # Löwdin the truncated centres
    @test dsa.scale == 0.1
    @test dsa.maxsubspace == 0 && dsa.restart_keep == 0    # 0 is the ABI spelling of "the default"

    dss = QNTT.defaults(QNTT.StateSpecificOptions)
    @test dss.mals == QNTT.defaults(QNTT.MalsOptions)
    @test dss.mode == 0                                  # :penalty
    @test dss.penalty_rule == 0                          # :auto
    @test dss.roots == 1
    @test dss.maxsubspace == 0 && dss.restart_keep == 0

    dex = QNTT.defaults(QNTT.ExcitedOptions)
    @test dex.nroots == 2
    @test dex.rank == 0
    @test dex.eps == 1e-10
    @test dex.max_iter == 30
    @test dex.noise_type == 1                            # :perturbative, the recorded default
    @test dex.seed == 1234
    @test dex.mode == 0 && dex.penalty_rule == 0 && dex.ss_start == 0
    @test dex.polish == 1
    @test dex.lower_grad_tol == 1e-6
    @test dex.trim_tol == 1e-6 && dex.trim_passes == 3
    # The polish CG runs at the response pipeline's stop, not `RiemannianOpts`' own 1e-4.
    @test dex.cg_eps == 1e-13 && dex.cg_gradtol == 1e-12 && dex.cg_max_iter == 600
    @test dex.cg_variant == 1                            # Polak-Ribière
    @test dex.cg_restarts == 10
    @test dex.polish_tol == 1e-10 && dex.grad_gate == 1e-9 && dex.overlap_tol == 1e-6
    @test dex.verbose == 0

    # Symbol keywords reach the wire codes through `setfields`.
    @test QNTT.setfields(dss; mode = :project).mode == 1
    @test QNTT.setfields(dex; ss_start = :x0).ss_start == 1
    @test_throws ArgumentError QNTT.setfields(dss; mode = :nonsense)

    # --- nroots = 1 IS `mals`, bit for bit ------------------------------------------------------
    for (tag, noise) in (("noise-free", Float64[]), ("perturbative", [1e-4, 1e-4, 0.0]))
        λ, x, hm = QNTT.mals(H, x0; eps = 1e-10, max_iter = 20, bond_dims = [0], noise = noise)
        λs, xs, hs = QNTT.state_average_mals(H, x0, 1; eps = 1e-10, max_iter = 20,
                                             bond_dims = [0], noise = noise)
        @test length(λs) == 1 && length(xs) == 1
        @test λs[1] == λ                                # exact equality, not ≈
        @test Array(xs[1]) == Array(x)                  # and the state, coefficient for coefficient
        @test length(hs) == length(hm)
        for (a, b) in zip(hs, hm)
            @test a.energy == b.energy
            @test a.discarded_weight == b.discarded_weight
            @test a.max_bond_dim == b.max_bond_dim
            @test a.bond_dim == b.bond_dim
            @test a.noise == b.noise
        end
        @info "state average nroots = 1 ≡ mals ($tag)" λ sweeps=length(hm)

        # ... and so is a state-specific sweep with no lower states (the delegate path).
        μ, y, hp = QNTT.state_specific_mals(H, x0, QNTT.TTVector{Float64,QNTT.Abelian}[];
                                            eps = 1e-10, max_iter = 20, bond_dims = [0],
                                            noise = noise)
        @test μ == λ
        @test Array(y) == Array(x)
        @test length(hp) == length(hm)
        for (a, b) in zip(hp, hm)
            @test a.energy == b.energy
            @test a.penalized == a.energy               # no penalty ⇒ θ = λ
            @test a.penalty == 0.0
            @test a.numops == 0                         # the delegate path records no applies
            @test a.discarded_weight == b.discarded_weight
            @test a.noise == b.noise
        end
    end

    # --- 2-root state average against the dense levels -------------------------------------------
    λs, xs, hist = QNTT.state_average_mals(H, x0, 2; eps = 1e-10, max_iter = 30, bond_dims = [0])
    @info "state average" λs dense=E[1:2] dev=abs.(λs .- E[1:2])
    @test length(λs) == 2 && length(xs) == 2
    @test all(abs(λs[i] - E[i]) ≤ 1e-6 for i in 1:2)
    @test issorted(λs)                                  # `homing = false` labels by energy
    @test all(isapprox(norm(x), 1; rtol = 1e-8) for x in xs)
    @test abs(dot(xs[1], xs[2])) ≤ 1e-6
    @test all(isapprox(QNTT.rayleigh_quotient(H, xs[i]), λs[i]; rtol = 1e-8) for i in 1:2)
    @test !isempty(hist)
    @test hist[end].noise == 0.0                        # a schedule ends noise-free
    @test hist[end].numops > 0                          # the state-average path does work
    @test all(0.0 ≤ sw.discarded_weight ≤ 1.0 for sw in hist)
    @test all(sw.max_bond_dim ≥ 2 for sw in hist)       # the shared basis hosts both roots
    @test hist[end].overlap_dev ≥ 0.0
    @test hist[end].energy ≈ (λs[1] + λs[2]) / 2 rtol = 1e-10   # Σwᵢλᵢ at the 1/n default

    # Explicit weights run; the four load-bearing clauses are pointed refusals.
    λw, _, _ = QNTT.state_average_mals(H, x0, 2; weights = [0.6, 0.4], eps = 1e-10,
                                       max_iter = 30, bond_dims = [0])
    @test all(abs(λw[i] - E[i]) ≤ 1e-6 for i in 1:2)
    for w in ([0.4, 0.6], [0.5, 0.6], [1.0])
        err = try
            QNTT.state_average_mals(H, x0, 2; weights = w, bond_dims = [0])
            nothing
        catch ex
            ex
        end
        @test err isa QNTTError
        @test occursin("weight", lowercase(err.msg))
    end
    # The state-average truncation defines only perturbative noise.
    @test_throws QNTTError QNTT.state_average_mals(H, x0, 2; bond_dims = [0],
                                                   noise = [1e-3, 0.0], noise_type = :random)
    @test_throws Exception QNTT.state_average_mals(H, x0, 0; bond_dims = [0])
    # A determinant start cannot seed a state average (hazard 5: the two-site local space is
    # one-dimensional). The ladder enriches the same start — see below.
    occ = [k < s.a for k in 0:(s.d - 1)]
    @test_throws QNTTError QNTT.state_average_mals(H, QNTT.tt_state(occ, occ), 2; bond_dims = [0])


    # The subspace keywords, routed end to end rather than only checked as struct defaults:
    # `homing` labels by overlap (so the claim is the SET), and `maxsubspace`/`restart_keep` reach
    # the block solver's restart rule — a value below `2·nroots` is refused there, not here.
    λh, xh, hh = QNTT.state_average_mals(H, x0, 2; homing = true, reorthonormalize = false,
                                         maxsubspace = 24, restart_keep = 4, eps = 1e-10,
                                         max_iter = 30, bond_dims = [0])
    @info "state average (homing, restarted)" λh dense=E[1:2]
    @test all(abs(sort(λh)[i] - E[i]) ≤ 1e-6 for i in 1:2)
    @test abs(dot(xh[1], xh[2])) ≤ 1e-6
    @test hh[end].numops > 0
    @test_throws QNTTError QNTT.state_average_mals(H, x0, 2; maxsubspace = 3, bond_dims = [0])

    # --- state-specific polish of root 2 against root 1 -------------------------------------------
    λ1, x1, sh = QNTT.state_specific_mals(H, x0, [xs[1]]; eps = 1e-10, max_iter = 30,
                                          bond_dims = [0])
    @info "state specific" λ1 dense=E[2] overlap=abs(dot(x1, xs[1]))
    @test abs(λ1 - E[2]) ≤ 1e-6
    @test abs(dot(x1, xs[1])) ≤ 1e-6                    # orthogonal to the fixed lower state
    @test sh[end].converged != 0
    @test sh[end].energy ≈ λ1 rtol = 1e-12              # `λ` is the PLAIN energy
    @test sh[end].penalized ≥ sh[end].energy - 1e-12    # θ ≥ λ
    @test sh[end].penalty ≥ 1.0                         # the Auto rule's floor
    @test sh[end].numops > 0                            # the sweep path, not the delegate
    @test sh[end].skipped == 0

    # Projection reaches the same level and records no weight at all.
    λp, xp, ph = QNTT.state_specific_mals(H, x0, [xs[1]]; mode = :project, eps = 1e-10,
                                          max_iter = 30, bond_dims = [0])
    @test abs(λp - E[2]) ≤ 1e-6
    @test ph[end].penalty == 0.0
    @test abs(dot(xp, xs[1])) ≤ 1e-8

    # An explicit weight is used verbatim (and is UNGUARDED — a `w` below the gap would make the
    # lower state its own minimizer; 5.0 is far above it here).
    _, _, fh = QNTT.state_specific_mals(H, x0, [xs[1]]; penalty = 5.0, eps = 1e-10,
                                        max_iter = 30, bond_dims = [0])
    @test fh[end].penalty == 5.0

    # Supplied reference energies agree with the ones computed at entry.
    λe, _, _ = QNTT.state_specific_mals(H, x0, [xs[1]]; lower_energies = [λs[1]], eps = 1e-10,
                                        max_iter = 30, bond_dims = [0])
    @test abs(λe - λ1) ≤ 1e-8
    @test_throws ArgumentError QNTT.state_specific_mals(H, x0, [xs[1]];
                                                        lower_energies = [1.0, 2.0])
    @test_throws ArgumentError QNTT.state_specific_mals(H, x0, [xs[1]]; penalty = :nonsense)
    @test_throws ArgumentError QNTT.state_specific_mals(H, x0, [xs[1]]; mode = :nonsense)
    # A negative explicit weight is refused — by the Julia keyword translation if it pre-checks,
    # by the library otherwise; the contract is the refusal, not which layer makes it.
    @test_throws Exception QNTT.state_specific_mals(H, x0, [xs[1]]; penalty = -1.0,
                                                    bond_dims = [0])
    # A state-specific sweep DOES accept random noise (its truncation is single-state).
    λr, _, _ = QNTT.state_specific_mals(H, x0, [xs[1]]; eps = 1e-10, max_iter = 30,
                                        bond_dims = [0], noise = [1e-3, 0.0],
                                        noise_type = :random, seed = 5)
    @test abs(λr - E[2]) ≤ 1e-6


    # Two lower states at once. At `nlower = 1` the per-lower stride is 1 and nothing can tell a
    # transposed or reversed companion from the right one; at 2 the caller's order is testable, and
    # both rungs must be pushed off.
    λ2s, x2s, h2s = QNTT.state_specific_mals(H, x0, [xs[1], x1]; eps = 1e-10, max_iter = 30,
                                             bond_dims = [0])
    @info "state specific (2 lower)" λ2s dense=E[3] o1=abs(dot(x2s, xs[1])) o2=abs(dot(x2s, x1))
    @test abs(λ2s - E[3]) ≤ 1e-6
    @test abs(dot(x2s, xs[1])) ≤ 1e-6
    @test abs(dot(x2s, x1)) ≤ 1e-6
    @test h2s[end].penalty ≥ 1.0
    @test h2s[end].numops > 0

    # `roots > 1` is the block sweep, and it is NOT the delegate path even with no lower states:
    # the bit-identity with `mals` and the `numops = 0` ledger hold only at `roots = 1`.
    λbk, xbk, hbk = QNTT.state_specific_mals(H, x0, QNTT.TTVector{Float64,QNTT.Abelian}[];
                                             roots = 2, eps = 1e-10, max_iter = 30,
                                             bond_dims = [0])
    @test abs(λbk - E[1]) ≤ 1e-6
    @test hbk[end].numops > 0                           # the block sweep did work
    @test isapprox(norm(xbk), 1; rtol = 1e-8)
    # ... and with a lower state it stays orthogonal to that rung and lands ON a dense level —
    # but WHICH one is the block's own continuation, not the first level above the rung: measured
    # here it returns E[3], not E[2]. `roots > 1` is a local device for a hard bond, not a request
    # for a particular root.
    λr2, xr2, _ = QNTT.state_specific_mals(H, x0, [xs[1]]; roots = 2, eps = 1e-10,
                                           max_iter = 30, bond_dims = [0])
    @info "state specific (roots = 2)" λr2 level=findfirst(l -> abs(l - λr2) ≤ 1e-6, E)
    @test any(abs(λr2 - l) ≤ 1e-6 for l in E)
    @test λr2 ≥ E[2] - 1e-6
    @test abs(dot(xr2, xs[1])) ≤ 1e-6

    # --- the ladder --------------------------------------------------------------------------------
    λl, xl, info = QNTT.excited_states(H, x0; nroots = 2, eps = 1e-10)
    @info "ladder" λl dense=E[1:2] licence=info.licence gradnorms=info.gradnorms
    @test length(λl) == 2 && length(xl) == 2
    @test all(abs(λl[i] - E[i]) ≤ 1e-6 for i in 1:2)
    @test info.licence
    @test all(info.converged)
    @test !any(info.stalled)
    @test info.swaps == 0
    @test info.overlap_dev ≤ 1e-6
    @test !info.enriched                                # a rank-4 random start hosts two roots
    @test size(info.overlaps) == (2, 2)
    @test all(abs(info.overlaps[i, i] - 1) ≤ 1e-10 for i in 1:2)
    @test info.penalties[1] == 0.0                      # root 1 has nothing to be pushed off
    @test info.penalties[2] ≥ 1.0
    @test all(g ≤ 1e-9 for g in info.gradnorms)         # the exact gate, per root
    @test all(abs(info.ss_energies[i] - E[i]) ≤ 1e-6 for i in 1:2)
    @test all(abs(info.sa_energies[i] - E[i]) ≤ 1e-6 for i in 1:2)
    @test isempty(info.lower_gradnorms)
    @test abs(dot(xl[1], xl[2])) ≤ 1e-6

    # A ladder built on a supplied lower state: one new root, a 2 × 2 Gram over [lower; roots].
    λb, xb, ib = QNTT.excited_states(H, x0; nroots = 1, lower = [xl[1]], eps = 1e-10)
    @test abs(λb[1] - E[2]) ≤ 1e-6
    @test length(ib.lower_gradnorms) == 1 && ib.lower_gradnorms[1] ≤ 1e-6
    @test size(ib.overlaps) == (2, 2)
    @test ib.overlap_dev ≤ 1e-6

    # `polish = false` is a bookkeeping contract: no critical point, hence no gate and no trim.
    _, _, inp = QNTT.excited_states(H, x0; nroots = 2, eps = 1e-10, polish = false)
    @test all(isnan, inp.gradnorms)
    @test !any(inp.stalled)
    @test all(p -> p == 0, inp.passes)

    # The reserved `ss_start = :x0` code is refused with a message, never a crash.
    err = try
        QNTT.excited_states(H, x0; nroots = 2, eps = 1e-10, ss_start = :x0)
        nothing
    catch e
        e
    end
    @test err !== nothing
    @test occursin("not available", sprint(showerror, err))

    # An explicit penalty is used verbatim on every penalized rung.
    _, _, ifx = QNTT.excited_states(H, x0; nroots = 2, eps = 1e-10, penalty = 2.0)
    @test all(==(2.0), ifx.penalties[2:end])


    # Both variable-length schedules non-empty AT ONCE, with different lengths. Julia converts a
    # `Vector{Csize_t}` to a `Ptr{Float64}` without complaint, so with either array empty the four
    # `(ptr, len)` pairs of the ccall could be ordered wrongly and nothing would notice; here a
    # swap makes the weights a pair of denormals (refused) and the bond dims astronomical.
    λwb, _, iwb = QNTT.excited_states(H, x0; nroots = 2, weights = [0.6, 0.4],
                                      bond_dims = [8, 8, 0], noise = [0.0], eps = 1e-10)
    @test all(abs(λwb[i] - E[i]) ≤ 1e-6 for i in 1:2)
    @test iwb.licence

    # The other CG variant reaches the same rungs (the polish is a means, not a contract).
    λfr, _, ifr = QNTT.excited_states(H, x0; nroots = 2, eps = 1e-10, cg_variant = :fr)
    @test all(abs(λfr[i] - E[i]) ≤ 1e-6 for i in 1:2)
    @test ifr.licence

    # `keep_sa` hands back the state-average stage's own states, as handles of this layer.
    λk, xk, ik = QNTT.excited_states(H, x0; nroots = 2, eps = 1e-10, keep_sa = true)
    @test length(ik.sa_states) == 2
    @test all(isapprox(norm(x), 1; rtol = 1e-8) for x in ik.sa_states)
    @test all(isapprox(QNTT.rayleigh_quotient(H, ik.sa_states[i]), ik.sa_energies[i]; atol = 1e-8)
              for i in 1:2)
    @test all(abs(λk[i] - E[i]) ≤ 1e-6 for i in 1:2)
    # The default leaves it empty rather than allocating states nobody asked for.
    @test isempty(info.sa_states)

    # The ladder refuses random noise (its state-average stage does), and enriches a determinant.
    @test_throws QNTTError QNTT.excited_states(H, x0; nroots = 2, noise_type = :random)
    λd, _, idet = QNTT.excited_states(H, QNTT.tt_state(occ, occ); nroots = 2, eps = 1e-10)
    @test idet.enriched
    @test abs(λd[1] - E[1]) ≤ 1e-6

    # --- SU(2): the ladder walks the S = 0 sub-spectrum --------------------------------------------
    su2 = QNTT.Su2Space(data)
    H2 = QNTT.Hamiltonian(su2, data)
    y0 = QNTT.tt_randn(su2, QNTT.uniform_ranks(su2, 64); seed = 3)
    λ2, y2, i2 = QNTT.excited_states(H2, y0; nroots = 3, eps = 1e-10)
    @info "SU(2) ladder" λ2 singlets=s0[1:3] licence=i2.licence
    # Every returned energy IS a singlet level (2S is carried exactly) ...
    @test all(any(abs(λ2[i] - l) ≤ 1e-6 for l in s0) for i in 1:3)
    # ... and at full rank the SET is the three lowest (the completeness loss is a truncation
    # phenomenon, not a full-rank one).
    @test all(abs(sort(λ2)[i] - s0[i]) ≤ 1e-6 for i in 1:3)
    @test i2.licence
    @test i2.overlap_dev ≤ 1e-6
    @test all(i2.converged)
    # The state average alone reaches the same three levels (its excluded gap is 0.29 here).
    λ2s, _, _ = QNTT.state_average_mals(H2, y0, 3; eps = 1e-10, max_iter = 40, bond_dims = [0])
    @test all(abs(sort(λ2s)[i] - s0[i]) ≤ 1e-6 for i in 1:3)

    # --- C64: identical surface, real energies, the same spectrum ----------------------------------
    datac = QNTT.read_fcidump(ComplexF64, h4())
    Hc = QNTT.Hamiltonian(QNTT.QnSpace(datac), datac)
    x0c = QNTT.tt_randn(ComplexF64, s, QNTT.uniform_ranks(s, 4); seed = 5)
    λc, xc, _ = QNTT.state_average_mals(Hc, x0c, 2; eps = 1e-10, max_iter = 30, bond_dims = [0])
    @test λc isa Vector{Float64}                        # every energy is a Float64, on every family
    @test all(abs(λc[i] - E[i]) ≤ 1e-6 for i in 1:2)
    @test eltype(xc[1]) == ComplexF64
    @test abs(dot(xc[1], xc[2])) ≤ 1e-6

    # --- handle lifetime ----------------------------------------------------------------------------
    GC.gc(); GC.gc()
    @test all(norm(x) > 0 for x in xl)
    @test all(norm(y) > 0 for y in y2)
    @test norm(x1) > 0
end

@testset "C header matches the shared library" begin
    # The header is hand-written; this keeps it from drifting away from the symbols the library
    # actually exports (and, since the bindings are generated from the same naming scheme, from
    # what `ccall` will look up at run time).
    hdr = read(joinpath(ROOT, "capi", "include", "qntt.h"), String)

    # Expand the four declaration macros the way the preprocessor would.
    function macro_body(name)
        i = findfirst("#define " * name, hdr)
        i === nothing && error("macro $name not found in qntt.h")
        lines = String[]
        for line in split(hdr[first(i):end], "\n")
            push!(lines, line)
            endswith(rstrip(line), "\\") || break
        end
        return join(lines, "\n")
    end

    function instantiate(body, subs)
        b = body
        for (name, val) in subs
            b = replace(b, "##" * name * "##" => val, name * "##" => val, "##" * name => val)
        end
        return Set(m.captures[1] for m in eachmatch(r"\b(qntt_[A-Za-z0-9_]+)\s*\(", b))
    end

    declared = Set{String}()
    for p in ("qntt_space", "qntt_su2_space")
        union!(declared, instantiate(macro_body("QNTT_DECLARE_SPACE(P)"), ["P" => p]))
    end
    for sfx in ("f64", "c64", "su2_f64", "su2_c64")
        union!(declared, instantiate(macro_body("QNTT_DECLARE_LAYER(SFX, T)"), ["SFX" => sfx]))
    end
    for sfx in ("f64", "c64")
        union!(declared, instantiate(macro_body("QNTT_DECLARE_SCALAR(SFX, T)"), ["SFX" => sfx]))
    end
    for sfx in ("f64", "su2_f64")
        union!(declared, instantiate(macro_body("QNTT_DECLARE_JOINT(SFX)"), ["SFX" => sfx]))
    end
    for sfx in ("f64", "su2_f64")
        union!(declared, instantiate(macro_body("QNTT_DECLARE_TDA(SFX)"), ["SFX" => sfx]))
    end
    # The cross-sector family is f64-only today; the SU(2) instantiation is a later phase's, and
    # listing it here before the symbol exists would fail this very check.
    for sfx in ("f64",)
        union!(declared, instantiate(macro_body("QNTT_DECLARE_SECTOR(SFX)"), ["SFX" => sfx]))
    end
    # The cross-sector OBSERVABLES are f64-only for the same reason: the SU(2) twin (E6) is
    # unbuilt, so `QNTT_DECLARE_SECTOR_OBS` is invoked once in the header and once here.
    for sfx in ("f64",)
        union!(declared, instantiate(macro_body("QNTT_DECLARE_SECTOR_OBS(SFX)"), ["SFX" => sfx]))
    end
    for m in eachmatch(r"^\s*(?:int32_t|void)\s+(qntt_[A-Za-z0-9_]+)\s*\("m, hdr)
        push!(declared, m.captures[1])
    end
    filter!(s -> !occursin("##", s), declared)

    @test length(declared) > 200
    lib = QNTT.Libdl.dlopen(QNTT.libqntt)
    try
        missing = [s for s in declared if QNTT.Libdl.dlsym(lib, s; throw_error = false) === nothing]
        @test isempty(missing)
        isempty(missing) || @info "declared but not exported" missing
    finally
        QNTT.Libdl.dlclose(lib)
    end
end

@testset "errors and handle lifetime" begin
    data = QNTT.read_fcidump(h4())
    s = QNTT.QnSpace(data)
    H = QNTT.Hamiltonian(s, data)
    x = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 2); seed = 4)

    # A core-crate guard becomes a Julia exception carrying the library's own wording.
    err = try
        QNTT.mals(H, x; bond_dims = Int[])
        nothing
    catch e
        e
    end
    @test err isa QNTTError
    @test occursin("bond_dims", err.msg)

    @test_throws ArgumentError QNTT.Hamiltonian(s, zeros(3, 3), zeros(3, 3, 3, 3))

    # Churn handles with the GC running: finalizers must not double-free or crash.
    for i in 1:200
        y = QNTT.tt_randn(s, QNTT.uniform_ranks(s, 2); seed = i)
        z = H * y
        _ = norm(z)
        i % 50 == 0 && GC.gc()
    end
    GC.gc(); GC.gc()
    @test norm(x) > 0                                       # still alive and usable
end

end
