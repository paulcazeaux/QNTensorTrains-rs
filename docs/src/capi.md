# C ABI

The Julia package is a thin wrapper over a C shared library, `libqntt_capi`, which is equally
usable from C, C++, Python (`ctypes`/`cffi`) or anything else with an FFI. The complete contract
lives in [`capi/include/qntt.h`](https://github.com/paulcazeaux/QNTensorTrains-rs/blob/master/capi/include/qntt.h);
this page summarizes it. Run `bash docs/doxygen.sh` to generate the full per-function reference
into `docs/build/capi/`.

## Contract

- Every function returns `int32_t`; `QNTT_OK` (0) is success. Results are written through
  out-pointers, never returned by value.
- On a non-`OK` return, output memory is unspecified and `qntt_last_error()` holds a message for
  the **calling thread**.
- Handles are opaque, refcounted and type-tagged. Passing the wrong handle type — or the right
  type with the wrong scalar — is reported as `QNTT_INVALID_ARG`, not undefined behaviour.
- Rust panics (typically core-crate assertions on bad input) are caught at the boundary and
  returned as `QNTT_PANIC` with the assertion's own message. Nothing aborts the host process.
- Buffer-out functions take `(ptr, cap, out_len)`. Call with `cap = 0`, `ptr = NULL` to query the
  length; a non-zero `cap` smaller than needed is an error, never a truncation.

## Status codes

| | |
|---|---|
| `QNTT_OK` (0) | success |
| `QNTT_NULL_ARG` (1) | a required pointer was null |
| `QNTT_INVALID_ARG` (2) | bad enum value, wrong handle type, short buffer, … |
| `QNTT_PANIC` (3) | a Rust panic was caught |
| `QNTT_IO` (4) | filesystem / parse failure |
| `QNTT_NOT_CONVERGED` (5) | an iterative driver failed to converge or broke down |

## Symbol families

The exported functions are four instantiations of one surface, plus scalar- and
layer-specific extras:

| prefix | layer | scalar |
|---|---|---|
| `qntt_f64_*` | abelian `(Nup, Ndn)` | `double` |
| `qntt_c64_*` | abelian `(Nup, Ndn)` | `double _Complex` |
| `qntt_su2_f64_*` | SU(2) spin-adapted | `double` |
| `qntt_su2_c64_*` | SU(2) spin-adapted | `double _Complex` |

Space queries are scalar-independent: `qntt_space_*` and `qntt_su2_space_*`.

## Data layout

- **Everything is 0-based**: sites `0..d-1`, bonds `0..d`, sector labels counting electrons
  strictly to the left of a bond.
- A space is `(d, a, b)`: abelian `(d, Nup, Ndn)`, SU(2) `(d, N, 2S)`. A sector is the pair
  `(a, b)`.
- **Rank grids** cross as dense `rows × cols` blocks of `size_t` with the *second* axis fastest; a
  per-bond stack is `d+1` such blocks back to back. Query `(rows, cols)` with `*_grid_extent`.
- **Coefficient tensors** `t`, `v` are flat **row-major** (first axis most significant), `d²` and
  `d⁴` entries.
- **Core blocks** are copied **column-major** (Fortran / faer / Julia order).
- **Block families** are `0` empty, `1` up, `2` down, `3` doubly occupied — with the SU(2) caveat
  that `1`/`2` are coupling branches, not spins.
- **Two rank notions.** Option-struct cap fields are either a `chi` (or `rank`/`bond_dims`) —
  a POOLED total bond dimension, SU(2) counting reduced multiplets — or a `rmax` — a PER-SECTOR
  cap (every sector's own rank stays under it); each field's comment in `qntt.h` states which.

## Excited states

Three per-layer entry points, in all four symbol families. Every energy is a `double` on every
family, the complex ones included — the returned levels are real.

```c
int32_t qntt_f64_state_average_mals (const void *h, const void *x0, size_t nroots, …);
int32_t qntt_f64_state_specific_mals(const void *h, const void *x0,
                                     const void *const *lower, size_t nlower, …);
int32_t qntt_f64_excited_states     (const void *h, const void *x0,
                                     const void *const *lower, size_t nlower, …);
```

Their options come from `qntt_state_average_options_default`,
`qntt_state_specific_options_default` and `qntt_excited_options_default`, which fill in the
library's defaults.

| wire code | values |
|---|---|
| `mode` (`SsMode`) | `QNTT_SS_MODE_PENALTY` 0, `QNTT_SS_MODE_PROJECT` 1 |
| `penalty_rule` + `penalty_value` (`PenaltyRule`) | `QNTT_PENALTY_AUTO` 0 (value ignored), `QNTT_PENALTY_FIXED` 1 (value used verbatim, unguarded) |
| `ss_start` (`SsStart`) | `QNTT_SS_START_SA` 0 (each rung starts from its state-average state); `1` is reserved and refused with `QNTT_INVALID_ARG` |

The penalty rule travels as a **pair**, not a sentinel: under `AUTO` the value is a placeholder,
under `FIXED` it is the weight — and `FIXED` with `0.0` is legal and means the unpenalized sweep,
which converges back onto the lower states.

The weight, rank and noise schedules travel alongside as `(ptr, len)` pairs, and an **empty
schedule does not uniformly mean "off"**:

| schedule | empty means |
|---|---|
| state average / state specific `bond_dims` | *not allowed* — pass `{0}` for uncapped |
| state average / state specific `noise` | noise-free, exactly as `qntt_*_mals` |
| `excited_states` `weights` | `1/nroots` each |
| `excited_states` `bond_dims` | the default ramp of `opts->rank` |
| `excited_states` `noise` | a built-in decaying ramp — pass `{0.0}` for noise-free |
| `excited_states` `ss_noise` | `{0.0}` |

Handle arrays (`out_states`, `out_roots`, `out_sa_states`) are **stricter** than the general
buffer contract: they hold exactly `nroots` handles, a shorter capacity is `QNTT_INVALID_ARG`
*before any work*, and a refused call returns no handles. Free each returned handle with `qntt_*_vec_free`. The sweep-history
buffers keep `qntt_*_mals`'s idiom instead — optional and truncating to `sweeps_cap`, with
`out_nsweeps` always reporting the full count; their per-root (`sweeps_cap · nroots`) and
per-lower-state (`sweeps_cap · nlower`) companions are sweep-major.

`qntt_*_excited_states` returns its diagnostics flat: `QnttExcitedInfo` for the scalars, an array
of `nroots` `QnttExcitedRoot` rows for the per-root record, `out_lower_gradnorms` for the supplied
lower states' gates, and `out_gram` — `(nlower + nroots)²` values, row-major over `[lower; roots]`,
absolute — under the usual `(ptr, cap, out_len)` protocol. The state-average and state-specific
*sweep traces* are not exported here; call those two entry points directly for them.

A rank-1 determinant or CSF start is refused by the two sweeps (`QNTT_PANIC`, carrying the
driver's own message) but **enriched** by the ladder, which reports it in `info.enriched`. On the
SU(2) families `rank` and every `bond_dims` entry are reduced multiplet ranks. See
[Excited states](manual/excited-states.md) for the routes, the licence and its blindness to
completeness, and a worked C example.

## Example

```c
#include "qntt.h"

void *data = NULL, *H = NULL, *x0 = NULL, *x = NULL;
if (qntt_f64_fcidump_read("h4.FCIDUMP", QNTT_SYM2_EIGHTFOLD, &data) != QNTT_OK) { /* … */ }

size_t norb, nelec; int64_t ms2; double e0;
qntt_f64_fcidump_info(data, &norb, &nelec, &ms2, &e0);

size_t nup, ndn;
qntt_space_from_n_two_sz(norb, nelec, ms2, &nup, &ndn);

size_t t_len = 0, v_len = 0;
qntt_f64_fcidump_t(data, NULL, 0, &t_len);
qntt_f64_fcidump_v(data, NULL, 0, &v_len);
double *t = malloc(t_len * sizeof *t), *v = malloc(v_len * sizeof *v);
qntt_f64_fcidump_t(data, t, t_len, &t_len);
qntt_f64_fcidump_v(data, v, v_len, &v_len);

qntt_f64_ham_new(norb, nup, ndn, t, t_len, v, v_len, NULL, 2.22e-16,
                 QNTT_CONSTRUCTION_DEFAULT, &H);

/* … build x0 from a rank grid, then: */
QnttMalsOptions opts; qntt_mals_options_default(&opts);
size_t schedule[] = {0};
double lambda;
qntt_f64_mals(H, x0, &opts, schedule, 1, &lambda, &x);

qntt_f64_vec_free(x); qntt_f64_vec_free(x0);
qntt_f64_ham_free(H); qntt_f64_fcidump_free(data);
```

Reading an error:

```c
if (status != QNTT_OK) {
    size_t n = 0;
    qntt_last_error(NULL, 0, &n);
    char *msg = malloc(n + 1);
    qntt_last_error(msg, n + 1, &n);
    fprintf(stderr, "qntt: %s\n", msg);
}
```
