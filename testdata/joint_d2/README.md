# joint_d2 — a saved endpoint of the joint (TT ⊕ orbital) ground state on a small synthetic problem

| file | what |
|---|---|
| `D2_ROT.FCIDUMP` | the problem's integrals, rotated by the run's accumulated orbital rotation `u` |
| `meta.txt`       | keyed metadata (`norb`, `nelec`, energies, iteration counts) |
| `x.tt`           | the TT state at the endpoint; with the rotated integrals the point is `(x, u = I)` |

Saved so the joint-response tests start from the same point on every CPU architecture.
