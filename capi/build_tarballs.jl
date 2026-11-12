# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

# BinaryBuilder recipe for the qntt C ABI.
#
#   julia --project=. build_tarballs.jl --verbose                 # all platforms
#   julia --project=. build_tarballs.jl --verbose aarch64-apple-darwin   # just one
#   julia --project=. build_tarballs.jl --deploy=local            # + an unregistered JLL
#
# The last form writes a complete `QNTT_jll` package to your Julia depot, which you can then
# `Pkg.develop` — that is the distribution path while this repository is private. When it becomes
# public, the same recipe is what goes into a Yggdrasil PR: only the `sources` commit hash and the
# deploy flags change.
#
# REQUIREMENT: the crate is Rust edition 2024, so it needs rustc >= 1.85. Older BinaryBuilder
# toolchains ship 1.7x and will fail at parse time with an "edition 2024 is unstable" error; the
# script below checks the version up front and says so plainly rather than dumping a parse error.
# If your BinaryBuilder is too old, either update it or (as a fallback) build the tarballs
# natively per platform with `cargo build --release -p qntt-capi` and package the artifacts by
# hand — the products list below is the full contents.

using BinaryBuilder, Pkg

name = "QNTT"
version = v"0.1.0"

# Pin an exact commit for reproducibility. Update this (and `version`) on every release.
sources = [
    GitSource(
        "https://github.com/paulcazeaux/QNTensorTrains-rs.git",
        # `git rev-parse HEAD` of the tagged release commit.
        "0000000000000000000000000000000000000000",
    ),
]

script = raw"""
cd $WORKSPACE/srcdir/QNTensorTrains-rs

# The working checkout may carry a .cargo/config.toml that redirects the target directory outside
# the build tree (this repo keeps artifacts out of a synced folder). Pin it explicitly so the
# recipe never depends on that file's presence or contents.
export CARGO_TARGET_DIR="$PWD/target"

# Edition-2024 guard: fail with a readable message instead of a parse error.
rustc --version
rustc --version | awk '{split($2, v, "."); if (v[1] < 1 || (v[1] == 1 && v[2] < 85)) {
    print "ERROR: qntt needs rustc >= 1.85 (Rust edition 2024); this toolchain is " $2;
    exit 1 } }'

cargo build --release -p qntt-capi --target ${rust_target}

install -Dvm 755 "target/${rust_target}/release/"*"qntt_capi."*"${dlext}" \
                 "${libdir}/libqntt_capi.${dlext}"
install -Dvm 644 capi/include/qntt.h "${includedir}/qntt.h"
install_license LICENSE
"""

# Rust cross-compiles cleanly to every tier-1 BinaryBuilder platform. The library has no system
# dependencies at all — the numerics are pure Rust (faer), so there is no BLAS/LAPACK to link.
platforms = supported_platforms()

products = [
    LibraryProduct("libqntt_capi", :libqntt_capi),
    FileProduct("include/qntt.h", :qntt_header),
]

dependencies = Dependency[]

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat = "1.6", compilers = [:c, :rust])
