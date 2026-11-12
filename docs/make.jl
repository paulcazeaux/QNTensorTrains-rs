#!/usr/bin/env julia
# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

#
# Build the documentation:
#
#     julia --project=docs -e 'using Pkg; Pkg.develop(path="julia"); Pkg.instantiate()'
#     julia --project=docs docs/make.jl
#
# Output lands in `docs/build/`. The C ABI reference is produced by Doxygen as a sub-site under
# `docs/build/capi/` — run `docs/doxygen.sh` first (or pass `--doxygen`, which shells out to it)
# if you want that section populated. Doxygen is optional: without it the C ABI page still links
# to the header, which carries the full contract in its comments.

using Documenter
using QNTT

const DOCS = @__DIR__
const ROOT = normpath(joinpath(DOCS, ".."))

if "--doxygen" in ARGS
    run(`bash $(joinpath(DOCS, "doxygen.sh"))`)
end

makedocs(;
    sitename = "QNTensorTrains-rs",
    authors = "Paul Cazeaux",
    modules = [QNTT],
    # Name the remote explicitly, so source links point at the canonical repository wherever the
    # site is built (Documenter otherwise infers it from the checkout's `origin`).
    repo = Remotes.GitHub("paulcazeaux", "QNTensorTrains-rs"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://paulcazeaux.github.io/QNTensorTrains-rs",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Manual" => [
            "Spaces, sectors and ranks" => "manual/spaces.md",
            "Hamiltonians" => "manual/hamiltonians.md",
            "Ground states" => "manual/ground-states.md",
            "Excited states" => "manual/excited-states.md",
            "Linear response (RPA)" => "manual/linear-response.md",
        ],
        "Tutorial" => [
            "Overview" => "tutorial/index.md",
            "step-01" => "tutorial/step-01.md",
            "step-02" => "tutorial/step-02.md",
            "step-03" => "tutorial/step-03.md",
            "step-04" => "tutorial/step-04.md",
            "step-05" => "tutorial/step-05.md",
            "step-06" => "tutorial/step-06.md",
            "step-07" => "tutorial/step-07.md",
            "step-08" => "tutorial/step-08.md",
            "step-09" => "tutorial/step-09.md",
            "step-10" => "tutorial/step-10.md",
            "step-11" => "tutorial/step-11.md",
            "step-12" => "tutorial/step-12.md",
            "step-13" => "tutorial/step-13.md",
            "step-14" => "tutorial/step-14.md",
            "step-15" => "tutorial/step-15.md",
            "step-16" => "tutorial/step-16.md",
            "step-17" => "tutorial/step-17.md",
            "step-18" => "tutorial/step-18.md",
        ],
        "Notebooks" => "notebooks.md",
        "Julia API" => [
            "Core reference" => "api.md",
            "Response, diagnostics and saddles" => "api-response.md",
        ],
        "C ABI" => "capi.md",
    ],
    # The Julia package is a thin wrapper: doctests would exercise the Rust library, which is a
    # build artifact and not present on a docs-only CI runner.
    doctest = false,
    warnonly = [:missing_docs],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(; repo = "github.com/paulcazeaux/QNTensorTrains-rs.git", devbranch = "master")
end
