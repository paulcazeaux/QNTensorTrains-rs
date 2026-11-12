# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

# The user-facing layer: handle types with finalizers, keyword-argument solvers, and the
# array-layout conversions. Every type is parameterized by scalar `T` and symmetry layer `L`, so
# one method set covers all four ABI families.
#
# Split into section files under highlevel/, included below in original definition order (Julia
# is load-order-sensitive, so this order must not change):
include("highlevel/space.jl")        # Spaces, point-group tables
include("highlevel/trains.jl")       # Handles, building/inspecting trains, algebra, state
                                      # diagnostics, persistence
include("highlevel/integrals.jl")    # FCIDUMP integrals, three-body coefficients
include("highlevel/hamiltonian.jl")  # Hamiltonians
include("highlevel/solvers.jl")      # Solvers, RPA, PD probes, complex seed
