# Julia API: response, diagnostics and saddles

```@meta
CurrentModule = QNTT
```

The option structs and wire-code tables these entry points take are on the [main reference page](@ref Options).

## Linear response

```@docs
QNTT.rpa
QNTT.rpa_dense
QNTT.rpa_signed
QNTT.sector_rpa
QNTT.sector_rpa_dense
QNTT.sector_alpha
QNTT.sector_transition_moments
```

## Diagnostics

State and curvature diagnostics: the complex-manifold guards, the Schmidt data that measures the
distance to the border of the fixed-rank manifold, criticality, and the positive-definiteness
probe with the seed it hands to the complex polish.

```@docs
QNTT.rho
QNTT.bilinear_dot
QNTT.schmidt
QNTT.sigma_min
QNTT.grad_norm
QNTT.pd_probe
QNTT.HESS_PARTS
QNTT.PD_VERDICTS
QNTT.complex_seed
```

## CASSCF and joint response

```@docs
QNTT.CasPartition
QNTT.cas_reference_state
QNTT.casscf
QNTT.joint_rpa
QNTT.JOINT_RPA_STATUS
QNTT.CasscfOptions
QNTT.CasscfInfo
```

## Saddle search (GAD)

```@docs
QNTT.saddle_seed
QNTT.gad
QNTT.GadOptions
QNTT.GadInfo
```
