#!/usr/bin/env julia
# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

#
# Ground state of a molecule from an FCIDUMP, driven entirely from Julia.
#
#   julia --project=julia julia/examples/ground_state.jl [FCIDUMP] [rank] [sweeps]
#
# Defaults to the H4 fixture shipped with the crate. This is the shape a research driver takes:
# read integrals, pick a manifold, sweep, report — with no Rust recompile between runs.

using Printf
using LinearAlgebra
using QNTT

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

path   = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(ROOT, "testdata", "h4.FCIDUMP")
rank   = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 8
sweeps = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 20

QNTT.init()   # size the library thread pool once
@printf("qntt %s, %d threads\n", QNTT.version(), QNTT.nthreads())

data = QNTT.read_fcidump(path)
@printf("\n%s\n  norb = %d, nelec = %d, 2Sz = %d, E_core = %.10f\n",
        basename(path), QNTT.norb(data), QNTT.nelec(data), QNTT.ms2(data), QNTT.e0(data))

space = QNTT.QnSpace(data)
println("  ", space)

t0 = time()
H = QNTT.Hamiltonian(space, data)
@printf("\noperator built in %.2f s; automaton bond dims %s\n",
        time() - t0, string(QNTT.automaton_bond_dims(H)))

x0 = QNTT.tt_randn(space, QNTT.uniform_ranks(space, rank); seed = 1234)
@printf("start: max bond dim %d, ⟨H⟩ = %.10f\n",
        QNTT.max_bond_rank(x0), QNTT.rayleigh_quotient(H, x0))

t0 = time()
λ_als, x_als = QNTT.als(H, x0; eps = 1e-8, max_iter = sweeps)
t_als = time() - t0

t0 = time()
λ_mals, x_mals = QNTT.mals(H, x0; eps = 1e-8, max_iter = sweeps, bond_dims = [0])
t_mals = time() - t0

e0 = QNTT.e0(data)
println("\n", "-"^64)
@printf("%-8s %16s %16s %8s %8s\n", "method", "λ (electronic)", "E = λ + E_core", "bond", "time")
@printf("%-8s %16.10f %16.10f %8d %7.2fs\n",
        "ALS", λ_als, λ_als + e0, QNTT.max_bond_rank(x_als), t_als)
@printf("%-8s %16.10f %16.10f %8d %7.2fs\n",
        "MALS", λ_mals, λ_mals + e0, QNTT.max_bond_rank(x_mals), t_mals)
println("-"^64)

# The converged state is an ordinary Julia value: inspect it, or feed it to linear response.
x = λ_mals ≤ λ_als ? x_mals : x_als
@printf("\nresidual ‖Hx − λx‖ / ‖x‖ = %.3e\n",
        norm(H * x - λ_mals * x) / norm(x))

if QNTT.max_sector_rank(x) ≤ 8 && space.d ≤ 8
    ω, info = QNTT.rpa(H, x; k = min(5, space.d))
    @printf("\nfixed-rank RPA excitation estimates (Hartree), %d applies, converged=%s:\n",
            info.applies, info.converged)
    for (j, w) in enumerate(ω)
        @printf("  ω_%d = %10.6f   (%8.3f eV)\n", j, w, w * 27.211386245988)
    end
    println("\n  NOTE: these are fixed-rank linear-response estimates, not exact excitation")
    println("  energies — they coincide only when the manifold is the full space.")
end
