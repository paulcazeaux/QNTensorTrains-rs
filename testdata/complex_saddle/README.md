# complex_saddle — a real critical point with a complex minimizer (H2O / 6-31G)

Saved TT states (`qntt::tt_io` format) used by the saddle-search and Hessian tests:

| file | what |
|---|---|
| `epsonly_real.tt`        | a fixed-rank real critical point `x` (SU(2) singlet, f64) |
| `epsonly_c64_witness.tt` | a negative-curvature direction at `x`, embedded over C64 |
| `complex_min.tt`         | the complex minimizer reached from `x` along that direction (C64), same rank as `x` |
