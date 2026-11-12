# step-14: Spin-adapted TT-CASSCF linear response: the pinned CAS manifold, the multiplet-rank ladder, and the Ŝ²-resolved labelling of the abelian spectrum

**Spin-adapted TT-CASSCF linear response**: the pinned CAS manifold at a CAS-MO fixture, the
truncated multiplet-rank ladder `ω(m_active)`, transition moments, oscillator strengths and
`α(0)` at every rung, plus the **Ŝ²-resolved labelling** of the abelian spectrum, which only
the spin-adapted layer can produce.

```text
  load testdata/<fixture>/              CASSCF-converged MOs + fingerprinted dipole .props
    ├─ su2_cas_reference_state          pinned CAS(n,m) reduced-multiplet TT at each rung
    ├─ su2_joint_ground_state           re-converge on the rung's manifold
    ├─ joint_symplectic_eigenvalues     must be Stable, or no ω is printed
    ├─ su2_transition_moments           T_n from the dipoles, rotated into the joint basis
    └─ su2_polarizability_static        α(0) diagonal
  full rung only, with the abelian twin: the singlet / non-singlet split of the abelian spectrum
```

## Two points to keep in mind

**The ladder is its own curve.** `m_active` caps reduced multiplet ranks. It is not an abelian
pooled `m`: a truncated SU(2) manifold and a truncated abelian manifold at "the same" rank
number are different manifolds. Only the full rung is comparable across layers, and there the
SU(2) spectrum is a subset of the abelian one, never equal to it.

**No ω without stability.** Criticality is not stability: a converged, joint-critical SU(2)
endpoint can still have no real ω spectrum. Every rung requires `info.converged` and a
`Stable` result from `joint_symplectic_eigenvalues`; a rung that fails either prints its
refusal and no spectrum.

Every SU(2) response call here passes `PropRoute::Materialized`, the only property route on
this layer (the default `Auto` resolves to it; an explicit `Shifted` panics).

## Runtime

A full rung at `h2o_631g_cas` (d=13, 152 modes) is expensive. Use `h2o_sto3g_cas` (d=7) for
the ladder; at `h2o_631g_cas` prefer a single full rung (`rungs=`) and add `nolabel` unless the
Ŝ² column is wanted, since the labelling also converges the abelian twin.

## Builds on

- [step-09](step-09.md): Spin-adapted response: singlet excitation energies on the fixed-rank SU(2) TT manifold
- [step-13](step-13.md): Linear response on a pinned CAS manifold: the CASSCF-RPA anchor, then the `ω(m_active)` ladder

## Usage

```text
cargo run --release --example step-14                            # h2o_631g_cas, rungs 1,2,3
cargo run --release --example step-14 -- h2o_sto3g_cas           # the fast fixture
cargo run --release --example step-14 -- h2o_631g_cas nolabel    # no abelian twin
cargo run --release --example step-14 -- --help
```

* `fixture` — CAS-MO fixture directory, a name under `testdata/` or a path (default
  `h2o_631g_cas`; `h2o_sto3g_cas` is the fast one);
* `rungs=` — comma-separated truncated `m_active` values, run before the full-rank rung
  (default `1,2,3`); `0` means full and is always run last, so it never rides in this list;
* `k=` — how many ω to tabulate (default `8`);
* `nolabel` — flag, off by default: skip the abelian twin, i.e. no Ŝ² labelling and no
  cross-layer α column;
* `--format text|tsv` — output format of the tables (default `text`).

## The program

The full source is [`examples/tutorial/step_14_spin_adapted_casscf.rs`](https://github.com/paulcazeaux/QNTensorTrains-rs/blob/master/examples/tutorial/step_14_spin_adapted_casscf.rs); the commentary below is the file's own.

```rust
use std::path::Path;
use std::time::Instant;

use qntt::CoeffTensor;
use qntt::hamiltonian::SparseHamiltonian;
use qntt::integrals::{Integrals, PropertyOperator, load_property_matching, read_meta};
use qntt::qn::QnSpace;
use qntt::rotation::rotate_integrals;
use qntt::solvers::hess::SymplecticSpectrum;
use qntt::solvers::joint::cas::{CasPartition, cas_pairs, cas_reference_state};
use qntt::solvers::joint::{
    JointOpts, JointScfOpts, PropRoute, StaticResponseOpts, TransitionMomentOpts, ZRoute,
    joint_context, joint_ground_state, oscillator_strengths,
};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::joint::cas::{su2_cas_rank_grid, su2_cas_reference_state};
use qntt::su2::joint::{
    su2_joint_context, su2_joint_ground_state, su2_polarizability_static, su2_transition_moments,
};
use qntt::su2::qn::Su2Space;
use qntt_exkit::out::{Col, Table, flush};
use qntt_exkit::prelude::{Def, Spec, Val};
```

Orbital-criticality tolerance of the response calls (static α, transition moments). The SCF
stops at `orb_grad = 1e-6`, but the response entry points re-check the orbital gradient at the
endpoint, and that value can differ slightly between CPU architectures; 3e-5 keeps a
roundoff-level excess from being refused as `NotJointCritical` after a converged SCF.

```rust
const ORB_GATE: f64 = 3e-5;
```

The pinned SCF settings. `tt_preserve_grid` is required on a pinned manifold: without it the
TT leg can leave the declared grid mid-SCF.

```rust
fn scf_opts(part: &CasPartition, d: usize) -> JointScfOpts {
    JointScfOpts {
        pairs: Some(cas_pairs(part, d)),
        orb_max_iter: 40,
        orb_grad: 1e-6,
        orb_grad_full: Some(1e-6),
        tt_grad_tol: 1e-6,
        orb_floor: 1e-8,
        tt_preserve_grid: true,
        ..JointScfOpts::default()
    }
}

fn props_in(dir: &Path, d: usize, u: &[f64]) -> Vec<PropertyOperator<f64>> {
    let meta = read_meta(&dir.join("meta.txt")).expect("meta.txt");
    ["x", "y", "z"]
        .iter()
        .map(|c| {
            let p = load_property_matching::<f64>(&dir.join(format!("dipole_{c}.prop")), d, &meta)
                .unwrap_or_else(|e| panic!("dipole_{c}: {e}"));
```

★ Basis identity: the fixture states the dipole in the dump's MOs; the optimum lives
at `u` away from them.

```rust
            let wrapped = Integrals::<f64> {
                t: p.m.clone(),
                v: CoeffTensor::<f64>::zeros(d, 4),
                e0: 0.0,
                norb: d,
                nelec: 0,
                ms2: 0,
                orbsym: Vec::new(),
                isym: None,
            };
            PropertyOperator::new(p.label.clone(), rotate_integrals(&wrapped, u).t)
        })
        .collect()
}

fn main() {
    let m = Spec::new("step-14")
        .about("spin-adapted TT-CASSCF linear response: the truncated multiplet-rank ladder")
        .pos(
            "fixture",
            "CAS-MO fixture dir (`h2o_sto3g_cas` is the fast one)",
            Val::Path,
            Def::Str("h2o_631g_cas"),
        )
        .key(
            "rungs",
            "truncated `m_active` values, run before the full-rank rung",
            Val::List(&Val::Usize),
            Def::List(&["1", "2", "3"]),
        )
        .key("k", "how many ω to tabulate", Val::Usize, Def::Usize(8))
        .flag(
            "nolabel",
            "skip the abelian twin: no Ŝ² labelling, no cross-layer α column",
        )
        .parse();
    let fixture = m.str("fixture");
```

`0` means full and is always run last, so it never rides in the truncated list.

```rust
    let rungs: Vec<usize> = m
        .strs("rungs")
        .iter()
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .filter(|m| *m > 0)
        .collect();
    let k = m.usize("k");
    let label = !m.flag("nolabel");

    let repo = qntt_exkit::repo!();
    let fx = repo.load(fixture);
    let dir = fx.dir.clone();
    let (part, casscf_gs): (CasPartition, Option<f64>) = {
        let meta = fx.meta();
        let get = |key: &str| -> usize {
            meta.keys
                .get(key)
                .unwrap_or_else(|| panic!("{fixture}: meta missing `{key}`"))
                .parse()
                .unwrap()
        };
        (
            CasPartition {
                n_core: get("n_core"),
                n_active: get("n_active"),
                n_virt: get("n_virt"),
            },
            meta.keys.get("casscf_gs").and_then(|v| v.parse().ok()),
        )
    };
    let ints = fx.into_ints();
    let d = ints.norb;
    let space = Su2Space::new(d, ints.nelec, 0);

    println!("== SU(2) TT-CASSCF linear response — {fixture}");
    println!(
        "d = {d}, N = {}, 2S = 0, CAS({} e⁻, {} orb), core/active/virt = {}/{}/{}, \
         non-redundant pairs = {}",
        space.n,
        space.n - 2 * part.n_core,
        part.n_active,
        part.n_core,
        part.n_active,
        part.n_virt,
        cas_pairs(&part, d).keep.len()
    );

    let ham = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
```

### the ladder: truncated multiplet rungs first, full rank last

```rust
    let mut ladder: Vec<usize> = rungs.clone();
    ladder.push(0);
    println!(
        "\n★ The ladder below is its OWN curve. `m_active` caps REDUCED MULTIPLET ranks — it is \
         NOT an abelian pooled `m`, and the two layers' truncated ladders are not comparable \
         objects. Only the full rung is cross-layer comparable.\n\
         ★ A `*` after `m_active` marks a rung the SCF stopped at the √ε line-search floor \
         (converged + stalled_at_floor): its ω carry a first-order-in-‖g_kept‖ bias, so read \
         `|g_kept|` beside them. Not a refusal — a diagnosis endpoint."
    );
    flush("");
    let mut ladder_t = Table::new(vec![
        Col::tag("m_active", 8),
        Col::int("tan_dim", 9),
        Col::int("modes", 7),
        Col::float("lambda+e0", 16, 9),
        Col::sci("|g_kept|", 11, 2),
        Col::int("n_eff", 7),
        Col::float("omega_1", 11, 6),
        Col::float("alpha_xx", 11, 6),
        Col::float("alpha_zz", 11, 6),
    ])
    .no_rule()
    .start();

    let mut full_su2_omega: Option<Vec<f64>> = None;
    let mut full_su2_alpha: Option<Vec<f64>> = None;
    for &m in &ladder {
        let t0 = Instant::now();
        let grid = su2_cas_rank_grid(&space, &part, qntt::PerSector(m));
```

One reference state per rung, used both for `tan_dim` and as the SCF seed.

```rust
        let (_l0, x0) = su2_cas_reference_state(&ham, &space, &part, m, 0xCA5D);
        let tan_dim: usize = {
            let (bl, _) = qntt::su2::riemannian::tangent_bases(&x0);
            qntt::su2::weingarten::tangent_dim(&bl)
        };
        let out = su2_joint_ground_state(&ints, space, &x0, &scf_opts(&part, d));
```

A rung the SCF stopped at the √ε floor (`converged` + `stalled_at_floor`) has its `‖gk‖`
at the attainable floor rather than below the tolerance, so its ω carry a
first-order-in-`‖gk‖` bias. It is still printed, but marked with `*`; read `|g_kept|`
beside it.

```rust
        let mtag = {
            let base = if m == 0 {
                "full".to_string()
            } else {
                m.to_string()
            };
            if out.info.stalled_at_floor {
                format!("{base}*")
            } else {
                base
            }
        };
```

No ω from an unconverged endpoint. A floor-limited one (converged + stalled_at_floor)
passes here and is marked above rather than dropped.

```rust
        if !out.info.converged {
            println!(
                "{mtag:>8} {tan_dim:>9} {:>7} {:>16} — SCF REFUSED: {:?}",
                "—", "—", out.info.refused_reason
            );
            continue;
        }
        let ctx = match su2_joint_context(
            &out.h,
            &out.x,
            &JointOpts {
                tt_grad_tol: 1e-6,
                orb_floor: 1e-8,
                pairs: Some(cas_pairs(&part, d)),
                z_route: ZRoute::Split,
                b_general: false,
            },
        ) {
            Ok(c) => c,
            Err(e) => {
                println!("{mtag:>8} {tan_dim:>9} — context REFUSED: {e:?}");
                continue;
            }
        };
        let (basis, _rep) = ctx.joint_basis();
```

Criticality is not stability: only a `Stable` spectrum has real ω.

```rust
        let omega = match ctx.joint_symplectic_eigenvalues(&basis) {
            SymplecticSpectrum::Stable { omega } => omega,
            SymplecticSpectrum::Resonant { signed } => {
                println!(
                    "{mtag:>8} {tan_dim:>9} {:>7} — RESONANT: {} unstable pair(s), lowest \
                     ω² = {:.3e} (no real ω spectrum here)",
                    basis.len(),
                    signed.n_neg,
                    signed.omega2.first().copied().unwrap_or(0.0)
                );
                continue;
            }
            SymplecticSpectrum::Unstable { reason, .. } => {
                println!(
                    "{mtag:>8} {tan_dim:>9} {:>7} — UNSTABLE: {reason:?} (no real ω spectrum here)",
                    basis.len()
                );
                continue;
            }
        };
        let props = props_in(&dir, d, &out.u);
        let alpha = su2_polarizability_static(
            &ctx,
            &props,
```

The default tolerance (1e-8): a much tighter static solve is very slow at this scale.

```rust
            &StaticResponseOpts {
                route: PropRoute::Materialized,
                orb_grad_tol: ORB_GATE,
                orb_grad_tol_full: Some(ORB_GATE),
                ..StaticResponseOpts::default()
            },
        )
        .expect("su2 static α");
        ladder_t.row(qntt_exkit::row![
            &mtag,
            tan_dim,
            omega.len(),
            out.lambda + ints.e0,
            out.info.g_orb_kept.last().copied().unwrap_or(f64::NAN),
            ctx.n_eff(),
            omega[0],
            alpha.alpha[0],
            alpha.alpha[8]
        ]);
        let _ = grid;

        if m == 0 {
```

### the full rung's detail: ω table, moments, oscillator strengths

```rust
            let tm = su2_transition_moments(
                &ctx,
                &basis,
                &props,
                &TransitionMomentOpts {
                    route: PropRoute::Materialized,
                    orb_grad_tol: ORB_GATE,
                    orb_grad_tol_full: Some(ORB_GATE),
                    ..TransitionMomentOpts::default()
                },
            )
            .expect("su2 transition moments");
            let f = oscillator_strengths(&tm.omega, &tm.moments);
            println!(
                "\n-- full rank: the {k} lowest singlet excitations (iso_defect = {:.2e}, \
                 orth_error = {:.2e})",
                ctx.iso_defect(),
                ctx.orth_error()
            );
            let mut exc_t = Table::new(vec![
                Col::int("n", 4),
                Col::float("omega", 12, 6),
                Col::sci("|T_x|", 12, 3),
                Col::sci("|T_y|", 12, 3),
                Col::sci("|T_z|", 12, 3),
                Col::sci("f", 12, 4),
            ])
            .no_rule()
            .start();
            for (n, fn_) in f.iter().enumerate().take(k.min(tm.omega.len())) {
                exc_t.row(qntt_exkit::row![
                    n,
                    tm.omega[n],
                    tm.moments[0][n].abs(),
                    tm.moments[1][n].abs(),
                    tm.moments[2][n].abs(),
                    *fn_
                ]);
            }
            exc_t.finish();
            println!(
                "\nα(0) diagonal = ({:.9}, {:.9}, {:.9})   SOS − solve = ({:.2e}, {:.2e}, {:.2e})",
                alpha.alpha[0],
                alpha.alpha[4],
                alpha.alpha[8],
                tm.alpha_sos[0] - alpha.alpha[0],
                tm.alpha_sos[4] - alpha.alpha[4],
                tm.alpha_sos[8] - alpha.alpha[8]
            );
            if let Some(gs) = casscf_gs {
                println!(
                    "λ + e0 = {:.12}   fixture casscf_gs = {gs:.12}   Δ = {:.2e}",
                    out.lambda + ints.e0,
                    out.lambda + ints.e0 - gs
                );
            }
            full_su2_omega = Some(omega.clone());
            full_su2_alpha = Some(alpha.alpha.clone());
        }
        eprintln!("  [rung {mtag} took {:.1} s]", t0.elapsed().as_secs_f64());
    }
    ladder_t.finish();
```

### the Ŝ² labelling of the abelian spectrum

```rust
    let (Some(su2_w), true) = (full_su2_omega.as_ref(), label) else {
        return;
    };
    println!(
        "\n== ★ Ŝ²-resolved labelling of the abelian spectrum (the abelian twin at the same \
         orbitals)"
    );
    let t0 = Instant::now();
    let nup = (ints.nelec + ints.ms2.unsigned_abs() as usize) / 2;
    let ab_space = QnSpace::new(d, nup, ints.nelec - nup);
    let ab_ham = SparseHamiltonian::new(ab_space, &ints.t, &ints.v, None, f64::EPSILON);
    let (_l, abx0) = cas_reference_state(&ab_ham, &ab_space, &part, 0, 0xCA5D);
    let ab_out = joint_ground_state(&ints, ab_space, &abx0, &scf_opts(&part, d));
    if !ab_out.info.converged {
        println!(
            "abelian twin did not converge: {:?}",
            ab_out.info.refused_reason
        );
        return;
    }
    let ab_ctx = joint_context(
        &ab_out.h,
        &ab_out.x,
        &JointOpts {
            pairs: Some(cas_pairs(&part, d)),
            ..JointOpts::default()
        },
    )
    .expect("abelian context");
    let (ab_basis, _) = ab_ctx.joint_basis();
    let SymplecticSpectrum::Stable { omega: ab_w } = ab_ctx.joint_symplectic_eigenvalues(&ab_basis)
    else {
        println!("the abelian twin is symplectically unstable; no labelling possible");
        return;
    };

    let is_singlet = |a: f64| -> bool {
        su2_w
            .iter()
            .map(|w| (a - w).abs())
            .fold(f64::INFINITY, f64::min)
            <= 1e-6
    };
    let n_non = ab_w.iter().filter(|a| !is_singlet(**a)).count();
    println!(
        "su2 {} modes ⊂ abelian {} modes; residue {} non-singlet (S ≥ 1). ★ An INCLUSION, never an \
         equality — and only at full active rank.",
        su2_w.len(),
        ab_w.len(),
        n_non
    );
    flush("");
    let mut lab_t = Table::new(vec![
        Col::int("n", 4),
        Col::float("omega(abelian)", 12, 6),
        Col::tag("S label", 12),
    ])
    .no_rule()
    .start();
    for (n, a) in ab_w.iter().enumerate().take(k) {
        lab_t.row(qntt_exkit::row![
            n,
            *a,
            if is_singlet(*a) {
                "SINGLET"
            } else {
                "non-singlet"
            }
        ]);
    }
    lab_t.finish();
    if let Some(a_su2) = full_su2_alpha {
        let a_ab = qntt::solvers::joint::polarizability_static(
            &ab_ctx,
            &props_in(&dir, d, &ab_out.u),
            &StaticResponseOpts {
                orb_grad_tol: ORB_GATE,
                orb_grad_tol_full: Some(ORB_GATE),
                ..StaticResponseOpts::default()
            },
        )
        .expect("abelian static α");
        println!(
            "\nα is layer-independent (the dipole is spin-free ⇒ non-singlet moments vanish): \
             max relative Δ over the diagonal = {:.2e}",
            (0..3)
                .map(|c| (a_su2[c * 3 + c] - a_ab.alpha[c * 3 + c]).abs()
                    / a_ab.alpha[c * 3 + c].abs().max(1.0))
                .fold(0.0f64, f64::max)
        );
    }
    eprintln!("  [labelling took {:.1} s]", t0.elapsed().as_secs_f64());
}
```

## Results

The run opens with the fixture banner — at the default `h2o_631g_cas` it reads `d = 13, N = 10,
2S = 0, CAS(8 e⁻, 6 orb), core/active/virt = 1/6/6, non-redundant pairs = 48` — then the two ★
warnings, then one ladder row per rung: `m_active` (`full` for the last rung, with a trailing
`*` when the SCF stopped at the √ε line-search floor), the tangent dimension, the mode count,
`lambda+e0`, `|g_kept|`, `n_eff`, the lowest ω, and `alpha_xx`, `alpha_zz`. A rung that fails a
gate prints its refusal in place of the row — `SCF REFUSED`, `context REFUSED`, `RESONANT`
(with the number of unstable pairs and the lowest ω²) or `UNSTABLE` (with the reason) — and no
spectrum. Per-rung wall times go to stderr.

The full rung then prints its detail: the `k` lowest singlet excitations with `|T_x|`, `|T_y|`,
`|T_z|` and the oscillator strength `f`, titled with `iso_defect` and `orth_error`; the α(0)
diagonal beside its SOS − solve differences; and `λ + e0` against the fixture's `casscf_gs`.
Unless `nolabel`, the abelian twin is converged at the same orbitals, and each abelian ω within
1e-6 of an SU(2) ω is labelled `SINGLET`, the rest `non-singlet` — the inclusion above, never
an equality. The two layers' α diagonals are then compared: they agree because a spin-free
dipole gives the non-singlet modes vanishing moments.

Recorded output (capture capped-default):

```text
== SU(2) TT-CASSCF linear response — h2o_631g_cas
d = 13, N = 10, 2S = 0, CAS(8 e⁻, 6 orb), core/active/virt = 1/6/6, non-redundant pairs = 48

★ The ladder below is its OWN curve. `m_active` caps REDUCED MULTIPLET ranks — it is NOT an abelian pooled `m`, and the two layers' truncated ladders are not comparable objects. Only the full rung is cross-layer comparable.
★ A `*` after `m_active` marks a rung the SCF stopped at the √ε line-search floor (converged + stalled_at_floor): its ω carry a first-order-in-‖g_kept‖ bias, so read `|g_kept|` beside them. Not a refusal — a diagnosis endpoint.

m_active   tan_dim   modes        lambda+e0    |g_kept|   n_eff     omega_1    alpha_xx    alpha_zz
      1*        32      79    -76.015091843     2.17e-6      48    0.329266    9.997776    4.166495
      2*        70     117    -76.034647087     1.89e-6      48    0.319485    7.517236    5.177571
      3*        96     143    -76.039557092     5.61e-6      48    0.313377    7.707219    5.030583
   full*       105     152    -76.039557119     5.54e-6      48    0.313274    7.707232    5.030588

-- full rank: the 8 lowest singlet excitations (iso_defect = 9.99e-16, orth_error = 6.88e-13)
   n        omega        |T_x|        |T_y|        |T_z|            f
   0     0.313274    1.171e-14     2.105e-1    1.135e-14    9.2512e-3
   1     0.388773    2.437e-14    1.010e-11    1.492e-14   2.6456e-23
   2     0.412951    1.816e-11    1.089e-15     6.881e-1    1.3034e-1
   3     0.492897     6.818e-1    1.037e-15    3.930e-12    1.5275e-1
   4     0.549441      1.110e0    1.406e-15    2.241e-12    4.5157e-1
   5     0.681980    1.701e-11    4.819e-16     7.860e-1    2.8090e-1
   6     1.059512    2.810e-15    1.837e-10    8.165e-15   2.3825e-20
   7     1.092350    1.707e-14     3.446e-1    1.462e-15    8.6483e-2

α(0) diagonal = (7.707231547, 1.327016911, 5.030587856)   SOS − solve = (1.18e-12, -3.02e-13, -1.15e-13)
λ + e0 = -76.039557118528   fixture casscf_gs = -76.039557117417   Δ = -1.11e-9

== ★ Ŝ²-resolved labelling of the abelian spectrum (the abelian twin at the same orbitals)
su2 152 modes ⊂ abelian 272 modes; residue 120 non-singlet (S ≥ 1). ★ An INCLUSION, never an equality — and only at full active rank.

   n omega(abelian)      S label
   0     0.313274      SINGLET
   1     0.388773      SINGLET
   2     0.412951      SINGLET
   3     0.492897      SINGLET
   4     0.549441      SINGLET
   5     0.573869  non-singlet
   6     0.589661  non-singlet
   7     0.643614  non-singlet

α is layer-independent (the dipole is spin-free ⇒ non-singlet moments vanish): max relative Δ over the diagonal = 4.43e-8
```

## Possibilities for extensions

- Run the whole ladder at `h2o_sto3g_cas` (d = 7): the same CAS(8 e⁻, 6 orb), no virtuals, and
  the rungs finish — the fixture to explore `rungs=` with.
- `hcho_631g_cas` and `c2h4_631g_cas` also ship the three dipole `.prop` files and a
  `casscf_gs`; their active spaces are smaller (CAS(4,3), CAS(2,2)) inside a much larger
  orbital set (22 and 26 orbitals), so watch the non-redundant pair count in the banner.
- Raise `k=`: it sets both the length of the excitation table and the number of abelian modes
  that get an Ŝ² label.
- Print the full 3×3 `alpha.alpha` instead of its diagonal, or `tm.alpha_sos` beside it, to see
  the off-diagonal components the α(0) line drops.
