# step-02: Ranks and convergence: what a per-sector rank cap costs in energy, and how one-site ALS approaches it

Step-01 ran ALS at full rank, where the train is exact. Real molecules are never that small, and
the whole point of a tensor train is to keep far fewer parameters than the sector dimension.
This step measures what that costs on the same molecule, from two angles:

**The rank ladder.** For a ladder of per-sector rank caps `R`, start from a random train at that
rank, run ALS to tolerance, and tabulate: the largest pooled bond dimension `M` (the sum of the
sector ranks at the fattest bond, the number another code calls "bond dimension"), the number of
stored coefficients, the energy, its gap to FCI, and the residual `‖Hx − λx‖`. Two facts to look
for: the energy is **variational** (every row sits above FCI, and the gap shrinks monotonically
as `R` grows), and the gap closes **fast** — LiH's Schmidt spectrum decays quickly, so a handful
of ranks buys micro-hartree accuracy at a small fraction of the full parameter count.

**Convergence at fixed rank.** Take a rank cap that binds (`conv=2` by default) and drive ALS
one sweep at a time, printing the energy after each. ALS is a monotone descent on a fixed-rank
manifold: the energy decreases every sweep and the decrease shrinks geometrically, roughly a
decade per sweep on LiH. The residual `‖Hx − λx‖` tells a different story: it measures the
distance to a true eigenvector, so at a binding cap it plateaus at the truncation error and
stays there however many sweeps run. Convergence *of the sweep* and accuracy *of the rank* are
two separate questions, and the two columns answer one each.

What ALS **cannot** do is change the ranks it was given. A rank chosen too small stays too
small, and the sector-by-sector distribution of rank stays whatever the random start drew.
Adaptive rank is the job of the two-site sweep, which [step-05](step-05.md) introduces together with the trap
that a determinant start sets for it.

## Builds on

- [step-01](step-01.md): A first ground state: load a molecule, build its Hamiltonian, run one-site ALS, compare to FCI

## Usage

```text
cargo run --release --example step-02                          # LiH/STO-3G, ranks 1,2,3,4,6,8
cargo run --release --example step-02 -- h2o_sto3g ranks=2,4,8 conv=4
cargo run --release --example step-02 -- --help
```

A final `full` row is always appended to the ladder: the rank cap large enough that no sector is
truncated, which is the exact reference the other rows are read against.

## The program

The full source is [`examples/tutorial/step_02_ranks_and_convergence.rs`](https://github.com/paulcazeaux/QNTensorTrains-rs/blob/master/examples/tutorial/step_02_ranks_and_convergence.rs); the commentary below is the file's own.

```rust
use qntt::solvers::als::{AlsOptions, als, als_mut};
use qntt::solvers::riemannian::normalize_mut;
use qntt::{QnSpace, SparseHamiltonian, TtVector, rayleigh_quotient, tt_randn};
use qntt_exkit::prelude::*;
use qntt_exkit::ranks::abelian::{per_bond_sum, uniform};
use qntt_exkit::stats::time;
```

Stored coefficients of a train: the parameter count a rank cap is trading against.

```rust
fn params(x: &TtVector<f64>) -> usize {
    (0..x.space.d).map(|k| x.core(k).num_coeffs()).sum()
}
```

The residual of the eigen-equation, `‖Hx − λx‖`, evaluated in the TT format.

```rust
fn residual(h: &SparseHamiltonian<f64>, x: &TtVector<f64>, lambda: f64) -> f64 {
    (&h.apply(x) - &(x * lambda)).norm()
}

fn main() {
    let m = Spec::new("step-02")
        .about("ranks and convergence: the energy cost of a rank cap, and how ALS approaches it")
        .pos(
            "fixture",
            "a directory under testdata/ (or a path) with integrals and a meta.txt",
            Val::Str,
            Def::Str("lih_sto3g"),
        )
        .key(
            "ranks",
            "per-sector rank caps to ladder through; a `full` row is always appended",
            Val::List(&Val::Usize),
            Def::List(&["1", "2", "3", "4", "6", "8"]),
        )
        .key(
            "eps",
            "ALS stops when the Rayleigh quotient moves by less than this between sweeps",
            Val::F64,
            Def::F64(1e-9),
        )
        .key(
            "sweeps",
            "maximum ALS sweeps per ladder row, and the length of the convergence table",
            Val::Usize,
            Def::Usize(30),
        )
        .key(
            "conv",
            "per-sector rank cap for the sweep-by-sweep convergence table",
            Val::Usize,
            Def::Usize(2),
        )
        .key(
            "seed",
            "seed of every random start",
            Val::Usize,
            Def::Usize(7),
        )
        .parse();
    let fixture = m.str("fixture").to_string();
    let ranks: Vec<usize> = m
        .strs("ranks")
        .iter()
        .map(|s| s.parse().expect("validated at parse"))
        .collect();
    let eps = m.f64("eps");
    let sweeps = m.usize("sweeps");
    let r_conv = m.usize("conv");
    let seed = m.usize("seed") as u64;
```

### the objects of [step-01](step-01.md), in three lines

```rust
    let fx = qntt_exkit::repo!().load(&fixture);
    let space: QnSpace = fx.space();
    let h: SparseHamiltonian<f64> = fx.ham();
    let (e_core, e_fci) = (fx.e_core(), fx.e_ref());
    banner_kv(
        "step-02",
        &[
            ("fixture", fixture.clone()),
            ("d", fx.d().to_string()),
            ("nelec", fx.nelec().to_string()),
            ("E_fci", format!("{e_fci:.10}")),
        ],
    );
```

The rank that no sector can exceed: the largest achievable per-sector rank over all bonds.

```rust
    let r_full = (0..=space.d)
        .flat_map(|b| {
            space
                .bond_sectors(b)
                .into_iter()
                .map(move |s| space.max_rank(b, s) as usize)
        })
        .max()
        .unwrap_or(1);
```

### A. the rank ladder

One row per cap. Every row starts from the SAME seed, so the rows differ only by how much of
the random draw the cap keeps; `M` is the pooled bond dimension at the fattest bond.

```rust
    flush("");
    flush(format!(
        "A. rank ladder — ALS from a seeded random start at each per-sector cap (full = {r_full})"
    ));
    let mut ladder = Table::new(vec![
        Col::text("R", 5),
        Col::int("M", 5).header("M max"),
        Col::int("params", 8),
        Col::float("E", 16, 10),
        Col::sci("dE", 10, 2).header("E - E_fci"),
        Col::sci("residual", 9, 1),
        Col::ms("wall", 10, 3).header("wall (ms)"),
    ])
    .start();
    let opts = AlsOptions {
        eps,
        max_iter: sweeps,
        ..AlsOptions::default()
    };
    let mut rows: Vec<(usize, TtVector<f64>)> = Vec::new();
    for (i, &r) in ranks.iter().chain(std::iter::once(&r_full)).enumerate() {
        let label = if i == ranks.len() {
            "full".to_string()
        } else {
            r.to_string()
        };
        let mut x0 = tt_randn(space, &uniform(space, r), &mut qntt_exkit::seeded(seed));
        normalize_mut(&mut x0);
        let ((lambda, x), wall) = time(|| als(&h, &x0, &opts));
        let e = lambda + e_core;
        let m_max = per_bond_sum(space, &x).into_iter().max().unwrap_or(0);
        ladder.row(row![
            label,
            m_max,
            params(&x),
            e,
            e - e_fci,
            residual(&h, &x, lambda),
            wall,
        ]);
        rows.push((r, x));
    }
    ladder.finish();
```

### B. convergence at a fixed rank

Drive ALS one sweep at a time at a BINDING cap (default R = 2). `als_mut` keeps the train and
returns the Rayleigh quotient after the sweep. The energy decrease per sweep measures
stationarity on the fixed-rank manifold and shrinks geometrically; the residual measures the
distance to a true eigenvector and plateaus at the truncation error, which no number of sweeps
can lower. Both columns are needed to read a run correctly.

```rust
    flush("");
    flush(format!(
        "B. convergence — one ALS sweep at a time at R = {r_conv}, stopping when |ΔE| < {eps:.0e}"
    ));
    let mut conv = Table::new(vec![
        Col::int("sweep", 5),
        Col::float("E", 16, 10),
        Col::sci("dE", 10, 2).header("E - E_fci"),
        Col::sci("step", 9, 1).header("|E_k - E_k-1|"),
        Col::sci("residual", 9, 1),
    ])
    .start();
    let one = AlsOptions {
        eps,
        max_iter: 1,
        ..AlsOptions::default()
    };
    let mut x = tt_randn(
        space,
        &uniform(space, r_conv),
        &mut qntt_exkit::seeded(seed),
    );
    normalize_mut(&mut x);
    let mut e_prev = rayleigh_quotient(&h, &x) + e_core;
    let mut sweeps_used = sweeps;
    for k in 1..=sweeps {
        let lambda = als_mut(&h, &mut x, &one);
        let e = lambda + e_core;
        let step = (e - e_prev).abs();
        conv.row(row![k, e, e - e_fci, step, residual(&h, &x, lambda)]);
        e_prev = e;
        if step < eps {
            sweeps_used = k;
            break;
        }
    }
    conv.finish();

    let (_, x_full) = rows.last().expect("the full row is always present");
    let e_full = rayleigh_quotient(&h, x_full) + e_core;
    Summary::new("step-02")
        .s("fixture", &fixture)
        .s(
            "ranks",
            ranks
                .iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join(","),
        )
        .i("R_full", r_full as i64)
        .f("E_full", e_full, 10)
        .e("dE_full", e_full - e_fci, 2)
        .i("R_conv", r_conv as i64)
        .i("sweeps_to_eps", sweeps_used as i64)
        .emit();
}
```

## Results

On LiH the ladder closes from 1.8e-4 hartree above FCI at `R = 1` (a rank-1 train per sector is
still richer than a determinant, because every sector carries its own block) to machine
precision at `R = 4`, where the pooled bond dimension is 24 of the exact 29. The parameter count
is the sobering column: the exact train stores 442 numbers against a sector dimension of
`C(6,2)² = 225`. On a molecule this small the format is pure overhead; the course's point here
is correctness, and the compression only appears once `d` is large enough that the sector
dimension dwarfs `d · R²` — [step-06](step-06.md). On water in the same basis (`h2o_sto3g`) the picture is
already different: `R = 1` sits 0.38 hartree high, `R = 4` reaches 2e-7, and the residual at the
`full` row is 1e-6 rather than 1e-7 because the inner solves are looser on a harder operator.

The convergence table at `R = 2` settles to `eps` in five sweeps, each cutting the step by about
a decade, while the residual stays at 1.3e-2 throughout: that plateau is the rank cap, not the
solver.

Recorded output (capture release-0.1):

```text
== step-02 | fixture=lih_sto3g | d=6 | nelec=4 | E_fci=-7.8824019323 ==

A. rank ladder — ALS from a seeded random start at each per-sector cap (full = 9)
R     M max   params                E  E - E_fci  residual  wall (ms)
---------------------------------------------------------------------
1         9       90    -7.8822222562    1.80e-4    2.6e-2    270.986
2        16      212    -7.8823614103    4.05e-5    1.3e-2    333.758
3        23      306    -7.8824006136    1.32e-6    1.8e-3    268.519
4        24      352    -7.8824019323  -1.78e-15    2.4e-7    202.916
6        26      388    -7.8824019323   1.78e-15    1.4e-7    193.889
8        28      424    -7.8824019323   5.33e-15    2.1e-7    213.289
full     29      442    -7.8824019323  -1.78e-15    3.9e-7    183.705

B. convergence — one ALS sweep at a time at R = 2, stopping when |ΔE| < 1e-9
sweep                E  E - E_fci |E_k - E_k-1|  residual
---------------------------------------------------------
    1    -7.8823590381    4.29e-5     5.6e0    1.4e-2
    2    -7.8823611870    4.07e-5    2.1e-6    1.3e-2
    3    -7.8823613831    4.05e-5    2.0e-7    1.3e-2
    4    -7.8823614004    4.05e-5    1.7e-8    1.3e-2
    5    -7.8823614004    4.05e-5   1.8e-15    1.3e-2
SUMMARY step-02 fixture=lih_sto3g ranks=1,2,3,4,6,8 R_full=9 E_full=-7.8824019323 dE_full=5.33e-15 R_conv=2 sweeps_to_eps=5
```

## Possibilities for extensions

- Run `h2o_sto3g` (`d = 7`, ten electrons): the ladder needs more ranks and the residual at a
  binding cap is visibly larger — a first look at a genuinely correlated system.
- Change `seed=`: at a binding cap the converged energy can move with the start, because a
  fixed-rank manifold has more than one stationary point. At full rank it does not move at all.
- Set `conv=8` and watch the convergence table collapse to one sweep: at a cap that does not
  bind, ALS is exact diagonalization and the "convergence" is Davidson's, not the sweep's.
- Print the sector ranks of the `full` row and compare them to `space.max_rank(b, s)`; then
  compare the `R = 4` row's ranks to the same cap to see which sectors were truncated.
