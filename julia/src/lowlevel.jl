# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

# Raw `ccall` layer: one method family per (layer, scalar), generated so that every `ccall` site
# sees a literal symbol. Everything here is 1:1 with `capi/include/qntt.h` — no argument massaging,
# no allocation policy, no convenience. The user-facing layer lives in `highlevel.jl`.
#
# Naming: `qntt_<layer><scalar>_<op>`, layer ∈ {"", "su2_"}, scalar ∈ {"f64", "c64"}.
#
# Split into julia/src/lowlevel/ section files, included below in original top-to-bottom
# definition order (Julia is definition-order-sensitive at load time).

include("lowlevel/space.jl")
include("lowlevel/surface.jl")
include("lowlevel/extras.jl")
