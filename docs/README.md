# Documentation

Two generators, one output tree:

| | source | output |
|---|---|---|
| **Documenter.jl** | `docs/src/*.md` + the `QNTT.jl` docstrings | `docs/build/` |
| **Doxygen** | `capi/include/qntt.h` | `docs/build/capi/` |

The Rust internals are documented separately by `cargo doc --open`.

## Building

```sh
# once
julia --project=docs -e 'using Pkg; Pkg.develop(path="julia"); Pkg.instantiate()'

# every time
julia --project=docs docs/make.jl            # Julia + guides
bash docs/doxygen.sh                         # C ABI reference (optional)
```

or `julia --project=docs docs/make.jl --doxygen` to do both in one go. `make docs-site` builds the
C library first and then the Julia site.

Doxygen is optional — install it with `brew install doxygen` (macOS) or
`apt-get install doxygen` (Debian). Without it the C ABI page still stands on its own; the header
carries the complete contract in its comments.

`docs/make.jl` deploys to GitHub Pages only when `CI=true`, so local builds never publish.

## Adding a page

Write it into `docs/src/`, then list it in the `pages` argument of `makedocs` in `docs/make.jl`.
Docstrings are pulled in by `@docs` blocks in `docs/src/api.md`; a docstring that exists but is
listed nowhere shows up as a `missing_docs` warning at build time, which is the reminder to add
it.
