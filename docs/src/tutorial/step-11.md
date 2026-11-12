# step-11: Channel response: one linear-response solve per symmetry channel, merged into one spectrum

Molecular demonstration of the PER-CHANNEL composition driver
(`solvers::hess::channel_response`).

The point being demonstrated: linear response at a fixed-rank reference is irrep- and
spin-CLOSED. Every tangent direction, and every cone direction, built at a reference in one
symmetry channel stays in that channel — and `SparseHamiltonian` hard-asserts space equality,
so there is not even a cross-space matrix element to compute. A `B1` excitation of an `A1`
ground state, or a triplet excitation of a singlet, is not "hard to reach" at any budget `k`:
it is *structurally* absent from the response space. For a symmetry-CONSERVING `H` the
cross-blocks between channels vanish identically, so the fix is not a bigger eigenproblem —
it is channel-local solves plus a scalar merge, which is what this driver does.

Arms:
  - the per-channel table: one row per enumerated channel (irrep, and optionally the spin-raised
    partner sector), with its fixed-rank ground energy, gradient, status, and the cone readout's
    `E_0^CI`;
  - the merged spectrum: every channel's levels sorted by absolute energy, each labeled with
    the channel that produced it and its gap against the REFERENCE channel's `E_0^CI`. Levels
    marked `dup` are degenerate COPIES of states another channel also reports — on the abelian
    layer `H` is spin-free, so the `(nup+δ, ndn−δ)` sector holds exactly the `S ≥ Sz+δ`
    multiplets, all of which have a same-energy member in the reference sector too. They are
    printed for their spin LABEL and must be filtered before the column is read as a spectrum;
  - the exact cross-check: dense configuration enumeration in the trivial-point-group sector,
    filtered per irrep by the determinant fuse — an independent, representation-free channel
    assignment;
  - the nonvacuity line: the lowest merged level that another IRREP channel produced and that
    lies below the reference channel's highest reported level. That is an excitation no linear
    response at the reference could have produced at any budget. Spin-raised channels are
    excluded from that search on purpose — they are not witnesses to it, since their levels are
    degenerate copies of reference-sector states that a purely tangent direction does reach.

Character fractions ride along: `r/t/c` per level is the Ritz vector's weight in the
mixed basis's reference / tangent / cone blocks. A cone-dominated level (`*`) is one the tangent
space at that channel's own reference could not represent either.

**Screening.** The fixture is loaded through `load_fixture_dir` with
`PointGroupPolicy::Resolve(None)`, whose `screen: true` default runs
`fcidump::screen_point_group_forbidden` BEFORE the space is built. That is not optional
bookkeeping: a symmetric dump's ~1e-14 symmetry-forbidden residue survives `eps = f64::EPSILON`
and makes the typed Hamiltonian's construction fail a symmetry assertion.
`channel_response_from_load` is the entry that is screened by construction.

## Builds on

- [step-10](step-10.md): Cone response: rank-increasing directions in the response ansatz at a fixed low-rank reference

## Usage

```text
cargo run --release --example step-11                                  # lih_sto3g, all irreps
cargo run --release --example step-11 -- testdata/lih_sto3g m=3 k=16 targets=4
cargo run --release --example step-11 -- testdata/h2o_sto3g m=4 k=8 spin=1
cargo run --release --example step-11 -- --help
```

- `dir` — the fixture directory (default `testdata/lih_sto3g`);
- `m=` — the per-channel pooled bond-dimension cap `chi` (also seeds the per-sector start rank), `0` = uncapped, the library default (default `0`);
- `k=` — the per-channel cone budget (default `16`);
- `targets=` — the number of gaps per channel (default `4`);
- `spin=` — the number of spin-RAISED partner channels, `0` = none, `1` = the `(nup+1, ndn−1)`
  ladder (default `0`);
- `noexact` — skip the dense cross-check (a flag, off by default);
- `--format text|tsv` — output format (default `text`).

## The program

The full source is [`examples/tutorial/step_11_channel_response.rs`](https://github.com/paulcazeaux/QNTensorTrains-rs/blob/master/examples/tutorial/step_11_channel_response.rs); the commentary below is the file's own.

```rust
use qntt::hamiltonian::assembly::Construction;
use qntt::integrals::{IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::pointgroup::{Irrep, PointGroup};
use qntt::solvers::hess::{
    ChannelGsOpts, ChannelOpts, ChannelSet, ChannelSpectrum, ChannelStatus, ConeOpts, IrrepSel,
    SpinSel, channel_response_from_load,
};
use qntt::{QnSpace, SparseHamiltonian, TtVector, dot, tt_state};
use qntt_exkit::args::{Def, Spec, Val};
use std::time::Instant;
```

### exact reference: configuration enumeration, optionally irrep-filtered

All `n`-element subsets of `0..d`, lexicographic.

```rust
fn combinations(d: usize, n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if n > d {
        return out;
    }
    let mut cur: Vec<usize> = (0..n).collect();
    if n == 0 {
        return vec![vec![]];
    }
    loop {
        out.push(cur.clone());
        let mut i = n;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if cur[i] != i + d - n {
                cur[i] += 1;
                for j in i + 1..n {
                    cur[j] = cur[j - 1] + 1;
                }
                break;
            }
        }
    }
}
```

Symmetric-matrix eigenvalues by cyclic Jacobi, ascending. Enough for the `d ≤ 7` sector dims
this example enumerates (225/441).

```rust
#[allow(clippy::needless_range_loop)]
fn sym_eigvals_jacobi(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _ in 0..100 {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off += a[i][j] * a[i][j];
                }
            }
        }
        if off.sqrt() < 1e-12 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut ev: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    ev.sort_by(|x, y| x.partial_cmp(y).unwrap());
    ev
}
```

Exact spectrum by dense determinant enumeration, optionally restricted to determinants whose
fused occupied-spin-orbital irrep equals `target` — a channel assignment that owes nothing to
the tensor-train representation.

```rust
fn exact_spectrum(
    space: QnSpace,
    ham: &SparseHamiltonian<f64>,
    filter: Option<(PointGroup, Irrep)>,
) -> Vec<f64> {
    let d = space.d;
    let states: Vec<TtVector<f64>> = combinations(d, space.nup)
        .into_iter()
        .flat_map(|up| {
            combinations(d, space.ndn)
                .into_iter()
                .map(move |dn| (up.clone(), dn))
        })
        .filter(|(up, dn)| match filter {
            None => true,
            Some((pg, tgt)) => {
                up.iter()
                    .chain(dn.iter())
                    .fold(Irrep::IDENTITY, |ir, &s| ir.fuse(pg.irrep_of(s)))
                    == tgt
            }
        })
        .map(|(up, dn)| {
            let mut occ_up = vec![false; d];
            let mut occ_dn = vec![false; d];
            for &s in &up {
                occ_up[s] = true;
            }
            for &s in &dn {
                occ_dn[s] = true;
            }
            tt_state(&occ_up, &occ_dn)
        })
        .collect();
    let n = states.len();
    if n == 0 {
        return Vec::new();
    }
    let mut hmat = vec![vec![0.0f64; n]; n];
    for (j, sj) in states.iter().enumerate() {
        let hs = ham * sj;
        for (i, si) in states.iter().enumerate() {
            hmat[i][j] = dot(si, &hs);
        }
    }
    sym_eigvals_jacobi(hmat)
}
```

### printing

The SPUR marker threshold on a level's CONE fraction. A printed marker only.

```rust
const SPUR_CONE_FRACTION: f64 = 0.5;

fn channel_tag(spec: &ChannelSpectrum<QnSpace>, ci: usize) -> String {
    let c = &spec.channels[ci];
    let ir = c
        .label
        .name
        .clone()
        .unwrap_or_else(|| format!("irrep {}", c.label.irrep.index()));
    match (c.label.spin_delta, c.space) {
        (0, _) => ir,
```

An infeasible offset has NO space (`ChannelResult::space` is `None`), so the tag falls
back to the offset itself rather than borrowing some live channel's sector.

```rust
        (delta, None) => format!("{ir}/+{delta}"),
        (_, Some(sp)) => format!("{ir}/({},{})", sp.nup, sp.ndn),
    }
}
```

The channel's sector column, or `—` when the parameters were infeasible and no space exists.

```rust
fn sector_tag(c: &qntt::solvers::hess::ChannelResult<QnSpace>) -> String {
    match c.space {
        Some(sp) => format!("({},{})", sp.nup, sp.ndn),
        None => "—".to_string(),
    }
}

fn main() {
    let m = Spec::new("step-11")
        .about("per-channel composition driver on a molecular fixture: irrep- and spin-closed linear response")
        .pos(
            "dir",
            "fixture directory",
            Val::Path,
            Def::Str("testdata/lih_sto3g"),
        )
        .key(
            "m",
            "per-channel pooled bond-dimension cap chi (also seeds the per-sector start rank), 0 = uncapped, the library default",
            Val::Usize,
            Def::Usize(0),
        )
        .key("k", "per-channel cone budget", Val::Usize, Def::Usize(16))
        .key(
            "targets",
            "number of gaps per channel",
            Val::Usize,
            Def::Usize(4),
        )
        .key(
            "spin",
            "number of spin-RAISED partner channels (0 = none, 1 = the (nup+1, ndn−1) ladder)",
            Val::Usize,
            Def::Usize(0),
        )
        .flag("noexact", "skip the dense cross-check")
        .parse();
    let dir = m.str("dir").to_string();
    let (mcap, k, targets, spin) = (
        m.usize("m"),
        m.usize("k"),
        m.usize("targets"),
        m.usize("spin"),
    );
    let do_exact = !m.flag("noexact");
```

SCREENED BY CONSTRUCTION: `Resolve` + the default `screen: true` runs
`screen_point_group_forbidden` before the typed space is built.

```rust
    let opts_load = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        ..IntegralsOptions::default()
    };
    let load = load_fixture_dir::<f64>(std::path::Path::new(&dir), &opts_load)
        .unwrap_or_else(|e| panic!("{dir}: {e}"));
    let (d, nelec) = (load.ints.norb, load.ints.nelec);
    let e_core = load.ints.e0;
    let space = load.space;
    println!("\n================================================================================");
    println!(
        "== {dir}: d={d}, N={nelec} — sector ({},{}), group {:?} (order {}), target irrep {}",
        space.nup,
        space.ndn,
        load.pg.kind,
        load.pg.pg.order(),
        space.target.index()
    );
    println!(
        "== per-channel composition: mcap {} ({}), cone k={k}, targets={targets}, spin channels {spin}",
        mcap,
        if mcap == 0 { "uncapped" } else { "capped" }
    );

    let opts = ChannelOpts {
        channels: ChannelSet {
            irreps: IrrepSel::All,
            spin: if spin == 0 {
                SpinSel::RefOnly
            } else {
                SpinSel::Raised(spin)
            },
        },
        kind: Some(load.pg.kind),
        gs: ChannelGsOpts {
            chi: qntt::Pooled(mcap),
            ..ChannelGsOpts::default()
        },
        cone: ConeOpts {
            targets,
            max_added: k,
            ..ConeOpts::default()
        },
        eps_h: f64::EPSILON,
    };

    let t0 = Instant::now();
    let spec = channel_response_from_load(&load, &opts);
    let elapsed = t0.elapsed().as_secs_f64();

    println!("\n-- channels ({:.1}s total) --", elapsed);
    println!(
        "{:<14} {:>6} {:>18} {:>10} {:>18} {:>7}  status",
        "channel", "sector", "E_gs + E_core", "‖grad‖", "E_0^CI + E_core", "gs c"
    );
    for (ci, c) in spec.channels.iter().enumerate() {
        let sector = sector_tag(c);
        let egs = c
            .e_gs
            .map(|e| format!("{:18.9}", e + e_core))
            .unwrap_or_else(|| format!("{:>18}", "—"));
        let g = c
            .grad_norm
            .map(|g| format!("{g:10.1e}"))
            .unwrap_or_else(|| format!("{:>10}", "—"));
        let e0 = c
            .cone
            .as_ref()
            .map(|r| format!("{:18.9}", r.e0 + e_core))
            .unwrap_or_else(|| format!("{:>18}", "—"));
        let gsc = c
            .cone
            .as_ref()
            .and_then(|r| r.character.first().copied())
            .map(|f| format!("{:7.1e}", f[2]))
            .unwrap_or_else(|| format!("{:>7}", "—"));
        println!(
            "{:<14} {sector:>6} {egs} {g} {e0} {gsc}  {:?}",
            channel_tag(&spec, ci),
            c.status
        );
    }
```

The merged spectrum. `dup` marks a level that is a degenerate COPY of a state another
channel also holds — on this (abelian) layer every spin-raised channel's levels are, because
a spin-free H puts each S ≥ Sz+δ multiplet in the reference sector too. They are printed for
their spin LABEL, and must be filtered out before the column is read as a spectrum.

```rust
    println!("\n-- merged spectrum (gaps vs the reference channel's E_0^CI) --");
    println!(
        "{:>4} {:<14} {:>3} {:>18} {:>12} {:>4}  r/t/c",
        "#", "channel", "k", "E + E_core", "gap", "dup"
    );
    for (i, l) in spec.levels.iter().enumerate() {
        let ch = &spec.channels[l.channel];
        let chr = ch
            .cone
            .as_ref()
            .and_then(|r| r.character.get(l.k).copied())
            .map(|f| {
                let mark = if f[2] > SPUR_CONE_FRACTION { " *" } else { "" };
                format!("{:.4}/{:.4}/{:.4}{mark}", f[0], f[1], f[2])
            })
            .unwrap_or_default();
        let gap = l
            .gap
            .map(|g| format!("{g:12.6}"))
            .unwrap_or_else(|| format!("{:>12}", "—"));
        println!(
            "{i:>4} {:<14} {:>3} {:18.9} {gap} {:>4}  {chr}",
            channel_tag(&spec, l.channel),
            l.k,
            l.energy + e_core,
            if l.redundant { "dup" } else { "" }
        );
    }
    println!("  * = cone fraction > {SPUR_CONE_FRACTION} (descriptive, never asserted)");
    println!(
        "  dup = degenerate copy of a state another channel also reports (abelian spin channels; \
         see ChannelSpectrum::levels)"
    );
```

Nonvacuity: a level LR at the reference structurally cannot see, interleaved with the ones
it can.

IRREP channels only. A different total irrep really is unreachable from the reference: the
tangent and cone constructions are irrep-closed and there is no cross-space matrix element.
A spin-RAISED channel on this layer is NOT such a witness — its levels are degenerate copies
of reference-sector states, and a purely tangent direction at the reference can and does
produce them (on LiH with `spin=1`, the lowest (3,1) level coincides with a reference-channel
level of character 0.0000/1.0000/0.0000, i.e. pure tangent).

```rust
    let ref_top = spec
        .levels
        .iter()
        .filter(|l| l.channel == spec.reference)
        .map(|l| l.energy)
        .fold(f64::NEG_INFINITY, f64::max);
    match spec
        .levels
        .iter()
        .find(|l| !l.redundant && l.channel != spec.reference && l.energy < ref_top)
    {
        Some(l) => println!(
            "\nnonvacuity: level {:.9} from irrep channel `{}` lies BELOW the reference channel's\n\
             highest reported level {:.9} — a different total irrep, so no tangent or cone direction\n\
             at the reference could produce it at any budget.",
            l.energy + e_core,
            channel_tag(&spec, l.channel),
            ref_top + e_core
        ),
        None => println!(
            "\nnonvacuity: no level from another IRREP channel lies below the reference channel's \
             top level ({:.9}) at this budget",
            ref_top + e_core
        ),
    }

    if !do_exact {
        return;
    }
```

Independent cross-check: dense enumeration in the trivial-PG sector, filtered per irrep by
the determinant fuse. Skipped above a sector-dimension cap: the enumeration materializes
`dim` product states and a dim² matrix, which runs out of memory at h2o_631g scale (dim
1.66e6). The cross-check is for small fixtures.

```rust
    let dim_plain = combinations(d, space.nup).len() * combinations(d, space.ndn).len();
    const EXACT_DIM_CAP: usize = 50_000;
    if dim_plain > EXACT_DIM_CAP {
        println!(
            "\n-- exact cross-check SKIPPED: sector dim {dim_plain} > cap {EXACT_DIM_CAP} \
             (dense enumeration would OOM; per-channel convergence + cross-layer checks are \
             the certificates at this scale) --"
        );
        return;
    }
    let te = Instant::now();
    let plain = QnSpace::new(d, space.nup, space.ndn);
    let hplain = SparseHamiltonian::new_with(
        plain,
        &load.ints.t,
        &load.ints.v,
        None,
        f64::EPSILON,
        Construction::Median,
    );
    println!(
        "\n-- exact cross-check (dense enumeration, dim {}) --",
        combinations(d, space.nup).len() * combinations(d, space.ndn).len()
    );
    for (ci, c) in spec.channels.iter().enumerate() {
        if c.status == ChannelStatus::Empty {
            continue;
        }
        let Some(cone) = c.cone.as_ref() else {
            continue;
        };
        let Some(cspace) = c.space else {
            continue;
        };
```

The enumeration runs in the TRIVIAL-PG sector and the irrep restriction is applied by
the determinant fuse — so the channel assignment being checked owes nothing to the
typed space the driver solved in.

```rust
        let (sp, ham) = if c.label.spin_delta == 0 {
            (plain, None)
        } else {
            let sp = QnSpace::new(d, cspace.nup, cspace.ndn);
            let h = SparseHamiltonian::new_with(
                sp,
                &load.ints.t,
                &load.ints.v,
                None,
                f64::EPSILON,
                Construction::Median,
            );
            (sp, Some(h))
        };
        let ex = exact_spectrum(
            sp,
            ham.as_ref().unwrap_or(&hplain),
            Some((load.pg.pg, c.label.irrep)),
        );
        let worst = cone
            .energies
            .iter()
            .zip(ex.iter())
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f64, f64::max);
        println!(
            "  {:<14} sector ({},{}) dim {:>4}, E0 exact {:18.9}, worst |E_j^CI − E_j^exact| over {} levels: {:.2e}",
            channel_tag(&spec, ci),
            cspace.nup,
            cspace.ndn,
            ex.len(),
            ex.first().copied().unwrap_or(f64::NAN) + e_core,
            cone.energies.len().min(ex.len()),
            worst
        );
    }
    println!("  ({:.1}s)", te.elapsed().as_secs_f64());
}
```

## Results

The default run on `testdata/lih_sto3g` (`d = 6`, `N = 4`, sector `(2,2)`, group `C2v` of order
4, target irrep 0) enumerates the four irreps and solves all of them. Every channel
comes back `Ok` with `‖grad‖` at ~1e-13, and since the default `m=0` leaves each channel's
reference uncapped, its `E_0^CI` coincides with its own `E_gs` — `A1` at −7.882401932, the
degenerate `B1`/`B2` pair at −7.716454011, `A2` at −7.249389104 — and the ground level's cone
fraction is zero: with nothing truncated there is no normal direction left to add.

The merged spectrum interleaves the four ladders: `A1` at gaps 0, 0.115983 and 0.133186, then
the `B1`/`B2` pair at 0.165948, on out to 2.291089 in `A2`. With `spin=0` no level is marked
`dup`. The nonvacuity line then reports the `B2` level at −7.716454011 lying below the reference
channel's highest reported level −7.318494520 — a different total irrep, so no tangent or cone
direction at the `A1` reference could produce it at any budget. That single line is the point of
the driver.

The dense cross-check enumerates the sector of dimension 225, which the determinant fuse splits
69/44/56/56 over the irreps, and reproduces every channel's five reported CI levels to at worst
2.84e-13.

Recorded output (capture release-0.1):

```text
================================================================================
== testdata/lih_sto3g: d=6, N=4 — sector (2,2), group C2v (order 4), target irrep 0
== per-channel composition: mcap 0 (uncapped), cone k=16, targets=4, spin channels 0

-- channels (10.7s total) --
channel        sector      E_gs + E_core     ‖grad‖    E_0^CI + E_core    gs c  status
A1              (2,2)       -7.882401932    4.5e-13       -7.882401932   0.0e0  Ok
A2              (2,2)       -7.249389104    7.6e-13       -7.249389104 9.5e-29  Ok
B1              (2,2)       -7.716454011    4.1e-13       -7.716454011 1.9e-29  Ok
B2              (2,2)       -7.716454011    3.3e-13       -7.716454011 2.7e-30  Ok

-- merged spectrum (gaps vs the reference channel's E_0^CI) --
   # channel          k         E + E_core          gap  dup  r/t/c
   0 A1               0       -7.882401932     0.000000       1.0000/0.0000/0.0000
   1 A1               1       -7.766418475     0.115983       0.0000/1.0000/0.0000
   2 A1               2       -7.749216187     0.133186       0.0000/1.0000/0.0000
   3 B2               0       -7.716454011     0.165948       1.0000/0.0000/0.0000
   4 B1               0       -7.716454011     0.165948       1.0000/0.0000/0.0000
   5 B1               1       -7.696951988     0.185450       0.0000/1.0000/0.0000
   6 B2               1       -7.696951988     0.185450       0.0000/1.0000/0.0000
   7 A1               3       -7.482612400     0.399790       0.0000/1.0000/0.0000
   8 A1               4       -7.318494520     0.563907       0.0000/1.0000/0.0000
   9 B2               2       -7.297966922     0.584435       0.0000/1.0000/0.0000
  10 B1               2       -7.297966922     0.584435       0.0000/1.0000/0.0000
  11 A2               0       -7.249389104     0.633013       1.0000/0.0000/0.0000
  12 A2               1       -7.215230496     0.667171       0.0000/1.0000/0.0000
  13 B2               3       -7.185953855     0.696448       0.0000/1.0000/0.0000
  14 B1               3       -7.185953855     0.696448       0.0000/1.0000/0.0000
  15 B1               4       -7.086645046     0.795757       0.0000/1.0000/0.0000
  16 B2               4       -7.086645046     0.795757       0.0000/1.0000/0.0000
  17 A2               2       -5.635018060     2.247384       0.0000/1.0000/0.0000
  18 A2               3       -5.603048408     2.279354       0.0000/1.0000/0.0000
  19 A2               4       -5.591313188     2.291089       0.0000/1.0000/0.0000
  * = cone fraction > 0.5 (descriptive, never asserted)
  dup = degenerate copy of a state another channel also reports (abelian spin channels; see ChannelSpectrum::levels)

nonvacuity: level -7.716454011 from irrep channel `B2` lies BELOW the reference channel's
highest reported level -7.318494520 — a different total irrep, so no tangent or cone direction
at the reference could produce it at any budget.

-- exact cross-check (dense enumeration, dim 225) --
  A1             sector (2,2) dim   69, E0 exact       -7.882401932, worst |E_j^CI − E_j^exact| over 5 levels: 1.76e-13
  A2             sector (2,2) dim   44, E0 exact       -7.249389104, worst |E_j^CI − E_j^exact| over 5 levels: 2.63e-13
  B1             sector (2,2) dim   56, E0 exact       -7.716454011, worst |E_j^CI − E_j^exact| over 5 levels: 9.24e-14
  B2             sector (2,2) dim   56, E0 exact       -7.716454011, worst |E_j^CI − E_j^exact| over 5 levels: 1.07e-13
  (0.5s)
```

## Possibilities for extensions

- `spin=1` adds the `(nup+1, ndn−1)` partner channels. Their levels come back marked `dup`,
  which is what that flag is for: filter on it before reading the column as a spectrum, and keep
  it when what you want is the spin label.
- `m=3` caps the per-channel reference so it is no longer ambient-exact. The cone stage then has
  work to do, and the `gs c` column and the `*` cone-dominated marks stop being zero.
- Other symmetric fixtures under `testdata/`: `h2o_sto3g`, `h2o_c2v`, or `ch2_triplet_c2v` for
  an open-shell reference. Past a sector dimension of 50 000 the dense cross-check skips
  itself; `noexact` turns it off by hand.
- Raise `targets=` and `k=` together to lengthen every channel's ladder, and watch whether the
  nonvacuity witness moves down.
