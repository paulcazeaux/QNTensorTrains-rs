# step-03: Reading a FCIDUMP: real integrals off disk, into a fixed-length ALS or MALS sweep

Steps 01–02 loaded fixtures through a helper. This step reads a FCIDUMP directly with
`load_fcidump`, builds the Hamiltonian, and runs ALS (one-site) or MALS (two-site) from a
random start at per-sector rank `R` for a fixed number of sweeps. It reports the operator's bond
dimension, the pooled bond dimension `M` of the start (Σ_sector rank at the fattest bond, the
number other DMRG codes call "bond dimension"), and the cost per sweep and per local
application of `H`. Every output line is labelled, so a run can be put beside another code's run
on the same dump at the same `M`, sweep count and symmetry (abelian `Nup`/`Ndn`, i.e. SZ).
Accuracy is not the goal here.

`--pg [KIND]` (optional, order-independent) resolves the dump's `ORBSYM`/`ISYM` and uses the
point-group-symmetric space instead: bare `--pg` guesses the conventional group of the implied
order, `--pg C2v` forces one. Without the flag no point-group path is entered at all.

## Builds on

- [step-01](step-01.md): A first ground state: load a molecule, build its Hamiltonian, run one-site ALS, compare to FCI

## Usage

```text
cargo run --release --example step-03                       # LiH/STO-3G, ALS at rank 20
cargo run --release --example step-03 -- testdata/lih_sto3g/LIH.FCIDUMP
cargo run --release --example step-03 -- testdata/n2_ccpvdz/N2.FCIDUMP mals 20 5 400
cargo run --release --example step-03 -- testdata/n2_d2h/N2.FCIDUMP als 20 3 0 --pg
cargo run --release --example step-03 -- --help
```

Positionals, in this order (`<fcidump> [method] [R] [sweeps] [Mcap]`):

- `fcidump` `<str>` — a FCIDUMP path, or a `testdata/` fixture name whose directory holds one
  (default `lih_sto3g`).
- `method` `<str>` — `mals` selects two-site MALS; any other value is one-site ALS
  (default `als`).
- `R` `<usize>` — per-sector rank of the random start (default 20).
- `sweeps` `<usize>` — number of sweeps (default 3).
- `Mcap` `<usize>` — total bond-dimension cap for MALS, 0 = uncapped (default 0).

Flags:

- `--pg [kind]` — resolve the fixture's point group; a bare `--pg` means `auto`, otherwise
  name one (`c1`, `ci`, `c2`, `cs`, `d2`, `c2v`, …).
- `--format <text|tsv>` — output format (default `text`).
- `-h`, `--help` — the full option list.

Environment:

- `RAYON_NUM_THREADS=<usize>` — rayon worker threads used by the sweep.

## The program

The full source is [`examples/tutorial/step_03_reading_an_fcidump.rs`](https://github.com/paulcazeaux/QNTensorTrains-rs/blob/master/examples/tutorial/step_03_reading_an_fcidump.rs); the commentary below is the file's own.

```rust
use std::path::Path;
use std::time::Instant;

use qntt::hamiltonian::matvec::onesite_workspace;
use qntt::integrals::{IntegralsOptions, PointGroupPolicy, load_fcidump};
use qntt::solvers::als::{AlsOptions, als, left_to_right_sketching, right_to_left_sketching};
use qntt::solvers::mals::{MalsOptions, mals};
use qntt::{SparseHamiltonian, TtVector, rayleigh_quotient, tt_randn};
use qntt_exkit::prelude::*;
use qntt_exkit::ranks;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
```

A FCIDUMP path, or a fixture name under `testdata/` whose directory holds exactly one
`*.FCIDUMP` — so the step runs bare on LiH and still accepts any dump on disk.

```rust
fn resolve_dump(arg: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(arg);
    if p.is_file() {
        return p.to_path_buf();
    }
    let dir = qntt_exkit::repo!().fixture_dir(arg);
    let mut dumps: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{arg}: neither a file nor a fixture directory ({e})"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "FCIDUMP"))
        .collect();
    match dumps.len() {
        1 => dumps.pop().unwrap(),
        n => panic!(
            "{}: expected exactly one *.FCIDUMP, found {n}",
            dir.display()
        ),
    }
}

fn main() {
    let m_args = Spec::new("step-03")
        .about("head-to-head DMRG timing harness on a FCIDUMP: ALS or MALS at a fixed rank")
        .pos(
            "fcidump",
            "a FCIDUMP path, or a testdata/ fixture name holding one",
            Val::Str,
            Def::Str("lih_sto3g"),
        )
```

`method` is a plain string positional, so the order is always
`<fcidump> <method> <R> …`: only `mals` is recognized, anything else runs ALS.

```rust
        .pos(
            "method",
            "`mals` selects two-site MALS; any other value is one-site ALS",
            Val::Str,
            Def::Str("als"),
        )
        .pos("R", "per-sector rank", Val::Usize, Def::Usize(20))
        .pos("sweeps", "number of sweeps", Val::Usize, Def::Usize(3))
        .pos(
            "Mcap",
            "total bond-dimension cap for MALS (0 = uncapped)",
            Val::Usize,
            Def::Usize(0),
        )
        .pg()
        .env(
            "RAYON_NUM_THREADS",
            "<usize>",
            "rayon worker threads used by the sweep",
        )
        .parse();

    let dump = resolve_dump(m_args.str("fcidump"));
    let path = dump.to_str().expect("utf-8 path");
    let method = m_args.str("method");
    let r = m_args.usize("R");
    let sweeps = m_args.usize("sweeps");
    let mcap = m_args.usize("Mcap");
    let pg_flag = match m_args.pg() {
        PgChoice::Unset => None,
        PgChoice::Auto => Some(None),
        PgChoice::Kind(k) => Some(Some(k)),
    };
    let threads = rayon::current_num_threads();
```

### 1. reading the dump

`load_fcidump` parses the packed, 1-based dump, completes the two-body permutational symmetry
(eightfold by default), applies the point-group policy below, and returns a space plus
chemists' `(ij|kl)` integrals with no factor applied; the core energy is returned separately,
for the caller to add. Read dumps through it rather than a hand-written loader.

With `--pg` the space carries the ORBSYM table and ISYM target, and `load_fcidump` also
removes the ~1e-14 residue a real dump leaves on integrals its own point group forbids, which
would otherwise break the symmetric operator's construction. Build a point-group space this
way. Absent the flag, no point-group path is entered.

```rust
    let opts = IntegralsOptions {
        point_group: match pg_flag {
            Some(kind) => PointGroupPolicy::Resolve(kind),
            None => PointGroupPolicy::Ignore,
        },
        ..IntegralsOptions::default()
    };
    let load = load_fcidump::<f64>(Path::new(path), None, &opts).unwrap_or_else(|e| panic!("{e}"));
    let pg_label = match pg_flag {
        Some(_) => format!(
            " | pg={} ({}) target={}",
            load.pg.kind.name(),
            if load.pg.guessed {
                "guessed"
            } else {
                "explicit"
            },
            load.pg
                .kind
                .irrep_name(load.pg.target)
                .expect("target in range")
        ),
        None => String::new(),
    };
    let (space, data) = (load.space, load.ints);
    banner(format_args!(
        "qntt {method} | {path} | norb={} nelec={} ms2={} | R={r} sweeps={sweeps} threads={threads}{pg_label}",
        data.norb, data.nelec, data.ms2
    ));
```

### 2. the Hamiltonian

Assembly at `eps = f64::EPSILON` drops nothing the reader produced. The core energy is not
part of the operator, which is why every λ printed below is electronic.

```rust
    let t0 = Instant::now();
    let h = SparseHamiltonian::new(space, &data.t, &data.v, None, f64::EPSILON);
    let t_build = t0.elapsed();
```

The operator's bond dimension at each cut.

```rust
    let abd = h.automaton_bond_dims();
    let abd_max = abd.iter().copied().max().unwrap_or(0);
    let abd_mean = abd.iter().sum::<usize>() as f64 / abd.len() as f64;
    println!("AUTOMATON_BOND_DIM max={abd_max} mean={abd_mean:.1}  profile={abd:?}");
```

### 3. a random start at per-sector rank R

`R` is a PER-SECTOR rank. `per_bond_sum` pools it over a bond's sectors, and the largest of
those sums is `M`, the single total bond dimension another code reports. `tt_randn` neither
orthogonalizes nor normalizes the train, and `rayleigh_quotient` assumes a unit-norm
argument, so the `λ0` printed just below is a raw `⟨x|H|x⟩`: a sanity number, not an energy.

```rust
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB10C_2BEE);
    let x0: TtVector<f64> = tt_randn(space, &ranks::abelian::uniform(space, r), &mut rng);
    let m = ranks::abelian::per_bond_sum(space, &x0)
        .into_iter()
        .max()
        .unwrap();
    let lam0 = rayleigh_quotient(&h, &x0);
    flush(format_args!(
        "H build {:.3} ms | max total bond rank M={m} | λ0={lam0:.6} (e0={:.6})",
        t_build.as_secs_f64() * 1e3,
        data.e0
    ));
```

### 4. the isolated local apply

Build the left and right environments of the middle site once, then time repeated local
applications of `H` to the middle core, with no sweep around them.

```rust
    {
        let k = data.norb / 2;
        let mut xo = x0.clone();
        xo.right_orthogonalize(true);
        let fr = right_to_left_sketching(&h, &xo);
        let fl_stack = left_to_right_sketching(&h, &{
            let mut xl = x0.clone();
            xl.right_orthogonalize(true);
            xl.move_core(data.norb - 1, true);
            xl
        });
        let mut xm = x0.clone();
        xm.right_orthogonalize(true);
        xm.move_core(k, true);
        let ws = onesite_workspace(&h, xm.core(k));
        let reps = 20usize;
        let t0 = Instant::now();
        let mut sink = 0.0f64;
        for _ in 0..reps {
            let y = h.sketched_hamiltonian(xm.core(k), &fl_stack[k], &fr[k], &ws);
            sink += y.norm();
        }
        let per = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "PURE_MATVEC {:.4}ms/apply  (mid-site k={k}, {reps} reps, M={m}, sink {sink:.3e}) — vs Block2 Tdav/Ndav",
            per * 1e3
        );
    }
```

### 5. the sweep leg

`eps = 1e-14` is set below what this leg is expected to reach, so it runs the requested
number of sweeps: what is compared across codes is cost per sweep and cost per local apply at
a fixed sweep count, not a converged energy. Reset the apply counter first so `PER_APPLY`
charges only this leg, and not the 20 warm-up applies timed above.

```rust
    qntt::hamiltonian::matvec::reset_local_matvec_applies();
    let t0 = Instant::now();
    let lam = if method == "mals" {
        let opts = MalsOptions {
            eps: 1e-14,
            bond_dims: if mcap > 0 { vec![mcap] } else { vec![0] },
            max_iter: sweeps,
            verbose: false,
            ..Default::default()
        };
        mals(&h, &x0, &opts).0
    } else {
        let opts = AlsOptions {
            eps: 1e-14,
            max_iter: sweeps,
            verbose: false,
        };
        als(&h, &x0, &opts).0
    };
    let dt = t0.elapsed();
    let applies = qntt::hamiltonian::matvec::local_matvec_applies();
    let per_apply_ms = if applies > 0 {
        1e3 * dt.as_secs_f64() / applies as f64
    } else {
        0.0
    };
    println!(
        "SWEEP_TIME_TOTAL {:.4}s  PER_SWEEP {:.4}s  ({sweeps} sweeps)  λ={lam:.6}  M={m}",
        dt.as_secs_f64(),
        dt.as_secs_f64() / sweeps as f64
    );
    println!(
        "LOCAL_APPLIES {applies}  PER_APPLY {per_apply_ms:.4}ms  (time per local H·x matvec — the fair cross-code metric)"
    );
}
```

## Results

The header echoes what was read — `norb`, `nelec`, `ms2`, the requested `R` and `sweeps`, the
thread count, and, under `--pg`, the resolved group, whether it was guessed or explicit, and the
target irrep. Then, in order:

- `AUTOMATON_BOND_DIM` — the bond dimension of the assembled operator at each cut (max, mean,
  and the whole profile). It is a property of the Hamiltonian alone and does not move with `R`.
- `H build … | max total bond rank M=… | λ0=… (e0=…)` — assembly wall time, the pooled bond
  dimension `M` the random start realizes, and `λ0 = ⟨x₀|H|x₀⟩` on the raw random start.
  `tt_randn` does not normalize, so `λ0` carries the start's scale and is a sanity number rather
  than an energy; `e0` beside it is the dump's core energy, which nothing here adds in.
- `PURE_MATVEC` — one local `H·x` at the middle site, averaged over 20 applications, with no
  sweep around it.
- `SWEEP_TIME_TOTAL … PER_SWEEP … λ=… M=…` — the whole sweep leg, and its cost per sweep.
- `LOCAL_APPLIES … PER_APPLY …` — how many local matvecs the sweep issued and the mean time per
  one. Dividing out the apply count is what makes ALS and MALS, or two codes with different
  iteration counts, comparable: `PER_SWEEP` alone rewards a solver that iterates less.

`λ` out of the sweep is a genuine electronic Rayleigh quotient, but `eps` is fixed at 1e-14
and the sweep count is whatever was asked for, so read it as a check that the run went
downhill, not as a converged energy — and add `e0` to it before comparing with any total
energy. Accuracy is [step-02](step-02.md)'s subject, not this one's.

Recorded output (capture release-0.1):

```text
== qntt als | testdata/lih_sto3g/LIH.FCIDUMP | norb=6 nelec=4 ms2=0 | R=20 sweeps=3 threads=32 ==
AUTOMATON_BOND_DIM max=214 mean=87.7  profile=[1, 25, 214, 186, 160, 27, 1]
H build 5.122 ms | max total bond rank M=29 | λ0=-1896.373094 (e0=0.995318)
PURE_MATVEC 0.5945ms/apply  (mid-site k=3, 20 reps, M=29, sink 2.208e3) — vs Block2 Tdav/Ndav
SWEEP_TIME_TOTAL 0.2027s  PER_SWEEP 0.0676s  (3 sweeps)  λ=-8.877720  M=29
LOCAL_APPLIES 158  PER_APPLY 1.2826ms  (time per local H·x matvec — the fair cross-code metric)
```

## Possibilities for extensions

- Run the dumps that ship with the fixtures — `testdata/lih_sto3g/LIH.FCIDUMP` (6 orbitals),
  `testdata/h2o_sto3g/H2O.FCIDUMP` (7), `testdata/n2_ccpvdz/N2.FCIDUMP` (28) — and watch
  `AUTOMATON_BOND_DIM` grow with `norb` while `PER_APPLY` grows with both it and `M`.
- Give the same dump to `als` and to `mals` at one `R`: `PER_SWEEP` favours the one-site sweep,
  `PER_APPLY` is the fair comparison, and `Mcap` is what keeps two-site growth bounded.
- Add `--pg` on a symmetry-adapted dump (`testdata/n2_d2h/N2.FCIDUMP`,
  `testdata/h2o_c2v/H2O.FCIDUMP`) and compare the operator profile and the sweep time against
  the same dump without the flag — the point-group space is a trade, not a free win.
- Vary `RAYON_NUM_THREADS` (1, 2, 4, …) on one dump and compare `PER_APPLY`.
