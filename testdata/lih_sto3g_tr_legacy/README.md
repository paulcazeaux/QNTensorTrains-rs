# lih_sto3g_tr_legacy — a saved intermediate point of a joint ground-state run on LiH / STO-3G

| file | what |
|---|---|
| `LIH_ROT.FCIDUMP` | the `testdata/lih_sto3g` integrals rotated by the accumulated orbital rotation `u` |
| `meta.txt`        | keyed metadata |
| `x.tt`            | the TT state at that point; with the rotated integrals the point is `(x, u = I)` |

Used by the fixture round-trip tests of `tests/io_integrals/`.
