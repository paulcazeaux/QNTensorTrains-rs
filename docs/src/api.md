# Julia API

```@meta
CurrentModule = QNTT
```

The linear-response solvers, the state and curvature diagnostics, the CASSCF / joint stack and the
saddle search are on the [second reference page](api-response.md); their option structs and wire
codes are under [Options](@ref) below.

## Library

```@docs
QNTT
QNTT.version
QNTT.init
QNTT.nthreads
QNTT.find_library
QNTT.set_library_path!
QNTTError
```

## Layers

Every space, train and operator is parametrized by its symmetry layer tag as well as its scalar
type.

```@docs
QNTT.Layer
QNTT.Abelian
QNTT.SU2
```

## Spaces, sectors and ranks

```@docs
QNTT.Space
QNTT.QnSpace
QNTT.Su2Space
QNTT.bond_sectors
QNTT.grid_extent
QNTT.maxranks
QNTT.uniform_ranks
QNTT.from_n_two_sz
QNTT.istrivial
QNTT.nirrep
```

## Point-group symmetry

`orbsym` goes in as 1-based Molpro ids; irreps come out as 0-based XOR indices. **Both layers
accept a point group** (the group itself is always abelian, `:C1`…`:D2h`), and `Su2Space(data)`
inherits the dump's `ORBSYM`/`ISYM` — pass `point_group = nothing` to drop it. Screen a
symmetry-adapted dump with [`QNTT.screen_forbidden!`](@ref) before building a Hamiltonian over a
non-trivial-group space.

```@docs
QNTT.pg_order
QNTT.irrep_name
QNTT.irrep_from_molpro
QNTT.molpro_from_irrep
QNTT.orbsym_to_irreps
QNTT.point_group
QNTT.PG_AUTO
```

## Integrals

```@docs
QNTT.Integrals
QNTT.read_fcidump
QNTT.norb
QNTT.nelec
QNTT.ms2
QNTT.e0
QNTT.onebody
QNTT.twobody
QNTT.orbsym
QNTT.isym
QNTT.pointgroup
QNTT.screen_forbidden!
QNTT.ThreeBody
QNTT.read_tcdump
QNTT.threebody
QNTT.setcoo!
```

## Tensor trains

```@docs
QNTT.TTVector
QNTT.tt_zeros
QNTT.tt_from_ranks
QNTT.tt_randn
QNTT.tt_state
QNTT.tt_ones
QNTT.tt_csf
QNTT.space
QNTT.ranks
QNTT.orthogonality_center
QNTT.nsites
QNTT.getblock
QNTT.setblock!
QNTT.blocks
Base.round(::QNTT.TTVector)
QNTT.round!
QNTT.round_sum
QNTT.left_orthonormalize!
QNTT.to_c64
Base.isvalid(::QNTT.TTVector)
Base.Array(::QNTT.TTVector{T,QNTT.Abelian}) where {T}
```

## Saving and loading

```@docs
QNTT.save
QNTT.load
```

## Hamiltonians

```@docs
QNTT.Hamiltonian
Base.:*(::QNTT.Hamiltonian{T,L}, ::QNTT.TTVector{T,L}) where {T,L}
QNTT.expectation
QNTT.rayleigh_quotient
QNTT.xthy
QNTT.automaton_bond_dims
```

## Solvers

```@docs
QNTT.als
QNTT.mals
QNTT.tdvp
QNTT.lanczos
QNTT.riemannian
```

## Excited states

```@docs
QNTT.state_average_mals
QNTT.state_specific_mals
QNTT.excited_states
```

## Options

Every solver takes its parameters as keywords; the structs below are the ABI-level mirrors, and
`QNTT.setfields` is how a keyword set is applied on top of the library's own defaults.

```@docs
QNTT.AlsOptions
QNTT.MalsOptions
QNTT.MalsSweep
QNTT.TdvpOptions
QNTT.LanczosOptions
QNTT.RiemannianOptions
QNTT.SymplecticOptions
QNTT.SectorOptions
QNTT.SECTOR_READOUTS
QNTT.RPA_METRIC_CODES
QNTT.RPA_KERNEL_CODES
QNTT.RPA_METRICS
QNTT.RPA_KERNELS
QNTT.CURVATURE_REG_CODES
QNTT.SECTOR_PRECOND_CODES
QNTT.SECTOR_STATUS
QNTT.SectorAlphaOptions
QNTT.SectorMomentOptions
QNTT.PROP_ROUTES
QNTT.SECTOR_REFUSALS
QNTT.StateAverageOptions
QNTT.SaSweep
QNTT.StateSpecificOptions
QNTT.SsSweep
QNTT.ExcitedOptions
QNTT.ExcitedInfo
QNTT.ExcitedRoot
QNTT.setfields
```

## Index

```@index
```
