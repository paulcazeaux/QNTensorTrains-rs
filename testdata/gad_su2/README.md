# gad_su2 — saved SU(2) states for the molecular saddle-search tests

`qntt::tt_io::save_su2_tt` files used by `tests/gad/`, LiH / STO-3G at reduced multiplet cap 4:

| file | what |
|---|---|
| `lih_sto3g_m4_launch.tt` | a launch point displaced from the ground state along the softest Hessian direction |
| `lih_sto3g_m4_e1s.tt`    | the index-1 saddle (first excited singlet) gentlest ascent reaches from it |

They are saved rather than recomputed so that the tests start from the same point on every CPU
architecture: a long non-convex trajectory can end in different basins on different hardware.
