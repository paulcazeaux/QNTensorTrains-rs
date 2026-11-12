# Spaces, sectors and ranks

## The symmetry descriptor

A calculation is set in a [`Space`](@ref QNTT.Space): the orbital count together with the
conserved quantum numbers.

```julia
space = QNTT.QnSpace(4, 2, 2)          # d = 4 orbitals, Nup = 2, Ndn = 2
space = QNTT.QnSpace(data)             # from a parsed FCIDUMP header
space = QNTT.Su2Space(4, 4, 0)         # d = 4, N = 4 electrons, total spin S = 0
```

The abelian layer conserves `(Nup, Ndn)` separately; the SU(2) layer conserves the total electron
count `N` and couples to a definite total spin `2S`. Both are exact — they index the block
structure of every core, so they cannot drift.

Sites are `0:d-1` and bonds are `0:d`, bond `b` sitting to the left of site `b`. A **sector** at a
bond is the pair `(a, b)` counting what is accumulated strictly to the left of it: `(nup, ndn)`
abelian, `(n, 2s)` SU(2).

```julia
QNTT.bond_sectors(space, 2)     # admissible sectors at bond 2, canonical order
```

Rank grids are laid out in that order.

## Rank grids

A tensor train's bond dimension is not a single number here — each *sector* at each *bond* carries
its own rank. [`QNTT.ranks`](@ref) returns one matrix per bond, indexed `g[a+1, b+1]`:

```julia
r = QNTT.ranks(x)              # length d+1
r[3][2, 2]                     # rank of sector (1, 1) at bond 2
```

Because of this, "a rank cap of `n`" means one of two different things, named apart on purpose (see
"Conventions, once" on the [front page](../index.md)): a PER-SECTOR cap (`rmax`, Rust type
`PerSector`) bounds every sector's own rank by `n`, while a POOLED cap (`chi`, Rust type `Pooled`)
bounds the TOTAL bond dimension `Σ` over sectors by `n` — the conventional DMRG `χ` / `M`, and what
the `bond_dims` schedule ramps. On the SU(2) layer every rank, under either cap, counts REDUCED
multiplets and is never comparable to an abelian pooled bond dimension.

Sectors have a maximum admissible rank (beyond which the manifold is the full space):

```julia
QNTT.maxranks(space)           # per-bond grids of the maximum
QNTT.uniform_ranks(space, 20)  # min.(20, maxranks) — the usual starting manifold
```

`uniform_ranks(space, r)` is what you normally hand to a constructor: it requests rank `r`
everywhere and is silently capped where the symmetry allows less. Taking `r` large enough
(`maxranks`) gives the *full* space, where TT is exact and DMRG returns the FCI energy — useful
for validation on small systems, and the regime where linear response reproduces exact excitation
energies.

## Building trains

```julia
QNTT.tt_zeros(space)                                   # every rank zero
QNTT.tt_from_ranks(space, r)                           # zero blocks at prescribed ranks
QNTT.tt_randn(space, r; seed = 1234)                   # Gaussian, reproducible
QNTT.tt_state([true,false,…], [true,false,…])          # a determinant (abelian)
QNTT.tt_csf([:occ, :occ, :empty, :empty])              # a CSF (SU(2))
```

`tt_randn` is fully determined by its seed — the same seed gives a bit-identical train, which is
what makes solver comparisons reproducible.

!!! warning "SU(2) `:up` and `:dn` are not electron spins"
    On the SU(2) layer the four block families are the *genealogical coupling branches* of adding
    one electron: `:up` raises the total spin (`2s → 2s+1`), `:dn` lowers it. Both add exactly one
    electron. Only on the abelian layer do they mean spin-up and spin-down.

## Looking inside

```julia
QNTT.nsites(x)              # d
QNTT.max_sector_rank(x)     # largest single-sector rank
QNTT.max_bond_rank(x)       # largest summed-over-sectors bond dimension
QNTT.orthogonality_center(x)
isvalid(x)                  # structural self-check

QNTT.getblock(x, 1, :up, (1, 0))    # one core block, as a Matrix
QNTT.blocks(x, 1)                   # every present block of core 1
QNTT.setblock!(x, 1, :up, (1, 0), M)
Array(x)                            # dense 4^d vector — small systems only
```

`max_bond_rank` is the number to compare against a conventional DMRG bond dimension: it sums the
per-sector ranks at a bond, which is what a symmetry-agnostic code would have to store.
