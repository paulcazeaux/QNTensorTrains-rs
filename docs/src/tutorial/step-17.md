# step-17: Excited states as a ladder: state average, state-specific sweeps and a penalized polish, checked against linear response and the exact spectrum

Every earlier step targeted ONE state. Excited states need two standard DMRG routes, and this
step runs both on the same molecule and prints the exact answer beside them. **State average**
carries `n` states in one train that shares every core but the orthogonality centre, solving
for the `n` lowest roots at each bond and truncating to one shared basis weighted over all of
them. It converges every root together, at the price of larger ranks than a single state needs.
**State-specific** sweeps then refine each averaged root on its own, against the roots already
final below it, by adding a penalty `w·Σⱼ |⟨ψⱼ,·⟩|²` on the lower states; the weight `w` is
chosen automatically and raised if a rung collapses onto a lower state. A **penalized
Riemannian polish** (minimizing `J = ⟨x,Hx⟩ + w·Σⱼ|⟨ψ̂ⱼ,x⟩|²` on the fixed-rank manifold)
finishes each rung before the next one starts, so every rung is orthogonalized against the
states it will be published next to.

Beside the ladder the program prints the linear-response `ω` of [step-08](step-08.md) at the polished ground
state and the gaps of the determinant-basis FCI spectrum. The two estimates are different
approximations — `ω` is a response quantity at one state, the ladder gaps are variational
differences — and at full rank (`--rank 0`) both are exact, which is what the closing
`ACCEPTANCE` line checks. At a truncated rank they separate, and the line checks only the
licences.

**What the licence does and does not certify.** `info.licence` is true when every rung
converged (its sweep stopped and its penalized gradient is under tolerance), the ladder is
orthogonal to `overlap_tol`, and no rung sits below the one before it. It never certifies
completeness: the shared truncation of the state-average stage can drop a symmetry sector that
carries no weight, and the enrichment noise — built from `H` acting on the current states —
cannot repopulate a sector `H` does not couple to. The SU(2) arm from the aufbau CSF is an
example: on LiH it skips the degenerate Π singlet pair and the licence is true anyway, so
which singlet level each root landed on is reported, never asserted. Remedies: ask for one root
more than you need, compare two starts, or seed the state average from a state that already
carries weight in the suspect sector. Which level a root lands on can also differ slightly
between CPU architectures.

The SU(2) reference is the `S = 0` SUB-spectrum: the `(nup, ndn)` levels that have no
`(nup+1, ndn−1)` partner. A ladder that lands on a genuine singlet a few levels up is not wrong,
only incomplete, and the deviation line says which.

## Builds on

- [step-05](step-05.md): Two-site MALS from a determinant start: what each noise flavour recovers of E_corr
- [step-07](step-07.md): Spin adaptation: the same N2/cc-pVDZ molecule as step-06, solved in the SU(2) singlet sector
- [step-08](step-08.md): Symplectic (RPA / linear-response) excitation spectrum of a small molecule on the fixed-rank TT manifold, end to end

## Usage

```text
cargo run --release --example step-17                              # LiH/STO-3G, full rank, 3 roots, both layers
cargo run --release --example step-17 -- lih_sto3g --rank 16       # a binding cap: the estimates separate
cargo run --release --example step-17 -- lih_sto3g --rank 8        # the overlap check refuses the licence
cargo run --release --example step-17 -- --layer su2 --nroots 4    # the SU(2) ladder alone
cargo run --release --example step-17 -- --help
```

- `fixture` — subdir of `testdata/` [default `lih_sto3g`].
- `--rank <usize>` — bond-rank cap of the ladder, `0` = full rank [default `0`].
- `--nroots <usize>` — roots in the ladder, ground state included [default `3`].
- `--layer <abelian|su2|both>` — which symmetry layer(s) to run [default `both`].
- `--format <text|tsv>` — output format [default `text`].

## The program

The full source is [`examples/tutorial/step_17_excited_states.rs`](https://github.com/paulcazeaux/QNTensorTrains-rs/blob/master/examples/tutorial/step_17_excited_states.rs); the commentary below is the file's own.

```rust
use qntt::core::tangent::TtTangent;
use qntt::solvers::excited::{ExcitedInfo, ExcitedOpts, excited_states};
use qntt::solvers::hess::{
    RpaReadout, SymplecticDavidsonOpts, symplectic_lowest_signed_davidson, weingarten_operator,
};
use qntt::su2::core::Family;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::Su2Space;
use qntt::su2::solvers::excited::{
    ExcitedOpts as Su2ExcitedOpts, excited_states as su2_excited_states,
};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::{Su2TtVector, tt_csf, tt_randn as su2_tt_randn};
use qntt::{CoeffTensor, TtVector, tt_randn};
use qntt_exkit::prelude::*;
use qntt_testkit::dense::{dense_eigen, dense_hamiltonian_matrix, dets_for};
use qntt_testkit::gad_mol::singlet_indices;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
```

The acceptance tolerance at full rank: every stage's gap must equal the FCI gap to this.

```rust
const ACCEPT_TOL: f64 = 1e-8;
```

The dense spectrum of one `(nup, ndn)` sector of `(t, v)`.

```rust
fn sector_levels(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    nup: usize,
    ndn: usize,
) -> Vec<f64> {
    let dets = dets_for(d, nup, ndn);
    let mat = dense_hamiltonian_matrix(d, t, v, &dets);
    dense_eigen(&mat).0
}
```

The index of the level nearest `e`, and the distance to it.

```rust
fn nearest(levels: &[f64], e: f64) -> (usize, f64) {
    let mut best = (0usize, f64::INFINITY);
    for (j, &l) in levels.iter().enumerate() {
        let dv = (e - l).abs();
        if dv < best.1 {
            best = (j, dv);
        }
    }
    best
}
```

The largest deviation of each stage's gaps from the reference gaps, plus the gates the
acceptance line reads.

```rust
struct LayerDevs {
    sa: f64,
    ss: f64,
    pol: f64,
    omega: f64,
    licence: bool,
    overlap_dev: f64,
}
```

The RPA ω of one solve, published only through the readout's gate (a converged solve that met
no anomaly); the run line is printed under `label` either way.

```rust
fn rpa_omega(label: &str, so: &RpaReadout) -> Option<Vec<f64>> {
    let omega = so.omega();
    match &omega {
        Some(_) => println!(
            "{label} {} basis vectors, {} applies, converged=true",
            so.iterations, so.applies
        ),
        None => println!(
            "{label} no stable spectrum (converged={}, report {:?}) — the ω column is left empty",
            so.converged, so.report
        ),
    }
    omega
}
```

One layer's ladder, rendered: the per-root energies of each stage and then the gaps against
the reference levels. `omega` is `None` when the RPA solve published no spectrum.

```rust
#[allow(clippy::too_many_arguments)]
fn report<V>(
    layer: &str,
    e: &[f64],
    info: &ExcitedInfo<V>,
    levels: &[f64],
    idx: &[usize],
    omega: Option<&[f64]>,
    e_core: f64,
) -> LayerDevs {
    println!("\n{layer}: ladder energies (Ha, total = electronic + e_core)");
    let mut t = Table::new(vec![
        Col::int("root", 6),
        Col::float("e_sa", 17, 9).header("E (state avg)"),
        Col::float("e_ss", 17, 9).header("E (state spec)"),
        Col::float("e_pol", 17, 9).header("E (polished)"),
        Col::sci("w", 10, 2).header("penalty w"),
        Col::sci("gate", 10, 2).header("‖grad J‖"),
        Col::int("passes", 8),
        Col::tag("conv", 6),
    ])
    .no_rule()
    .start();
    for (i, &ei) in e.iter().enumerate() {
        t.row(row![
            i,
            info.sa_history.lambdas[i] + e_core,
            info.ss_energies[i] + e_core,
            ei + e_core,
            info.penalties[i],
            info.gradnorms[i],
            info.passes[i],
            if info.converged[i] { "yes" } else { "NO" }
        ]);
    }
    t.finish();

    println!("\n{layer}: gaps to root 0 (Ha) against the dense reference spectrum");
    let mut g = Table::new(vec![
        Col::int("j", 5),
        Col::int("level", 7).header("ref lvl"),
        Col::float("gap_sa", 15, 9).header("gap (SA)"),
        Col::float("gap_ss", 15, 9).header("gap (SS)"),
        Col::float("gap_pol", 15, 9).header("gap (pol)"),
        Col::float("omega", 15, 9).header("ω (RPA)"),
        Col::float("gap_ref", 15, 9).header("gap (dense)"),
        Col::sci("dev", 11, 3).header("pol − dense"),
    ])
    .no_rule()
    .start();
    let (mut dev_sa, mut dev_ss, mut dev_pol, mut dev_om) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for j in 1..e.len() {
        let gap_ref = levels[idx[j]] - levels[idx[0]];
        let gap_sa = info.sa_history.lambdas[j] - info.sa_history.lambdas[0];
        let gap_ss = info.ss_energies[j] - info.ss_energies[0];
        let gap_pol = e[j] - e[0];
        let om = omega.and_then(|o| o.get(j - 1).copied());
        dev_sa = dev_sa.max((gap_sa - gap_ref).abs());
        dev_ss = dev_ss.max((gap_ss - gap_ref).abs());
        dev_pol = dev_pol.max((gap_pol - gap_ref).abs());
        match om {
            Some(w) => dev_om = dev_om.max((w - gap_ref).abs()),
```

A row with no ω — the RPA solve met an anomaly, or it returned fewer
distinct frequencies than the ladder has gaps — is NOT evidence that the RPA gap
equals the dense one. It must fail the rank-0 acceptance claim, never pass it
vacuously.

```rust
            None => dev_om = f64::INFINITY,
        }
        g.row(row![
            j,
            idx[j],
            gap_sa,
            gap_ss,
            gap_pol,
            om.map(|w| format!("{w:>14.9}"))
                .unwrap_or_else(|| format!("{:>14}", "—")),
            gap_ref,
            gap_pol - gap_ref
        ]);
    }
    g.finish();
    println!(
        "{layer}: the ω column is the RPA spectrum in ASCENDING order at root 0's manifold, so \
         when the ladder skips a level the two columns stop lining up row for row."
    );
    println!(
        "{layer}: licence={} converged={:?} swaps={} swapped={:?} overlap_dev={:.3e} \
         passes={:?} enriched={}",
        info.licence,
        info.converged,
        info.swaps,
        info.swapped,
        info.overlap_dev,
        info.passes,
        info.enriched
    );
    println!(
        "{layer}: state-average stage — {} sweeps, max bond rank {}, {} local applies",
        info.sa_history.sweeps.len(),
        info.sa_history
            .sweeps
            .last()
            .map(|s| s.max_bond_dim)
            .unwrap_or(0),
        info.sa_history
            .sweeps
            .iter()
            .map(|s| s.numops)
            .sum::<usize>()
    );
    LayerDevs {
        sa: dev_sa,
        ss: dev_ss,
        pol: dev_pol,
        omega: dev_om,
        licence: info.licence,
        overlap_dev: info.overlap_dev,
    }
}

fn main() {
    let m = Spec::new("step-17")
        .about(
            "excited-state ladder (state average → state specific → penalized polish) against RPA \
             ω and the FCI gaps, on both symmetry layers",
        )
        .pos(
            "fixture",
            "subdir of `testdata/`",
            Val::Str,
            Def::Str("lih_sto3g"),
        )
        .opt(
            "rank",
            "bond-rank cap of the ladder (0 = full rank)",
            Val::Usize,
            Def::Usize(0),
        )
        .opt(
            "nroots",
            "roots in the ladder, ground state included",
            Val::Usize,
            Def::Usize(3),
        )
        .opt(
            "layer",
            "which symmetry layer(s) to run",
            Val::Enum(&["abelian", "su2", "both"]),
            Def::Str("both"),
        )
        .parse();
    let fixture = m.str("fixture").to_string();
    let rank: usize = m.usize("rank");
    let nroots: usize = m.usize("nroots").max(1);
    let layer = m.str("layer").to_string();
    let want_ab = layer != "su2";
    let want_su2 = layer != "abelian";

    let fx = qntt_exkit::repo!().load(&fixture);
    let (d, nup, ndn, nelec) = (fx.d(), fx.nup(), fx.ndn(), fx.nelec());
    let e_core = fx.e_core();
    banner(format_args!(
        "excited-state ladder — {fixture} (d={d}, N↑={nup} N↓={ndn}, rank cap {rank}, \
         nroots {nroots})"
    ));
    println!(
        "HF {:.9} Ha, FCI reference {:.9} Ha (meta); every table below is TOTAL energy \
         (electronic + e_core = {e_core:.9})",
        fx.e_hf(),
        fx.e_ref()
    );
```

### the dense reference spectra

```rust
    let (t, v) = (fx.t().clone(), fx.v().clone());
    let ev = sector_levels(d, &t, &v, nup, ndn);
    println!(
        "dense oracle:  {} determinants in ({nup},{ndn}), E[0..{}] = {:?}",
        dets_for(d, nup, ndn).len(),
        nroots.min(ev.len()),
        &ev[..nroots.min(ev.len())]
    );
```

The `S = 0` sub-spectrum: the levels of this sector with no partner one unit of Sz up.

```rust
    let singlets: Option<(Vec<f64>, Vec<usize>)> = if nup == ndn && nup < d {
        let ev_up = sector_levels(d, &t, &v, nup + 1, ndn - 1);
        let idx = singlet_indices(&ev, &ev_up);
        let lv: Vec<f64> = idx.iter().map(|&i| ev[i]).collect();
        println!(
            "               {} of them are S = 0 levels, s0[0..{}] = {:?}",
            lv.len(),
            nroots.min(lv.len()),
            &lv[..nroots.min(lv.len())]
        );
        Some((lv, idx))
    } else {
        None
    };

    let mut ab: Option<LayerDevs> = None;
    let mut su2: Option<(LayerDevs, f64, f64)> = None;
```

### abelian arm

```rust
    if want_ab {
        let space = fx.space();
        let ham = fx.ham();
        let x0: TtVector<f64> = fx.hf();
        let mut opts = ExcitedOpts::new(&space);
        opts.nroots = nroots;
        opts.rank = rank;
        opts.keep_sa = true;
        let (e, roots, info) = excited_states(&ham, &x0, &[], &opts);
```

RPA at the polished ground state: the same construction as [step-08](step-08.md). At full rank the
manifold is the whole sector, `B ≡ 0`, and ω is the exact gap; at a cap it is a
fixed-rank linear-response estimate and the two columns separate.

```rust
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x17_A1);
        let ctx = weingarten_operator(&ham, &roots[0]);
        let (base_l, base_r) = {
            let (l, r) = ctx.bases();
            (l.clone(), r.clone())
        };
        let k = nroots.saturating_sub(1).max(1);
        let v0s: Vec<TtTangent<f64>> = (0..k + 2)
            .map(|_| {
                let dirv: TtVector<f64> = tt_randn(space, &roots[0].ranks, &mut rng);
                TtTangent::project(&base_l, &base_r, &dirv)
            })
            .collect();
        let (so, _) =
            symplectic_lowest_signed_davidson(&ctx, &v0s, k, &SymplecticDavidsonOpts::default());
        let omega = rpa_omega("abelian RPA:  ", &so);
        let idx: Vec<usize> = (0..e.len()).collect();
        ab = Some(report(
            "abelian",
            &e,
            &info,
            &ev,
            &idx,
            omega.as_deref(),
            e_core,
        ));
    }
```

### SU(2) arm

```rust
    if want_su2 {
        match &singlets {
            None => println!(
                "\nsu2: skipped — the SU(2) singlet ladder needs a closed-shell fixture \
                 (N↑ = N↓); this one is ({nup},{ndn})."
            ),
            Some((levels, _)) => {
                let space = Su2Space::new(d, nelec, 0);
                let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
```

The aufbau CSF: the lowest N/2 spatial orbitals doubly occupied.

```rust
                let codes: Vec<Family> = (0..d)
                    .map(|k| {
                        if k < nelec / 2 {
                            Family::Occ
                        } else {
                            Family::Empty
                        }
                    })
                    .collect();
                let x0: Su2TtVector<f64> = tt_csf(&codes);
                let mut opts = Su2ExcitedOpts::new(&space);
                opts.nroots = nroots;
                opts.rank = rank;
                opts.keep_sa = true;
                let (e, roots, info) = su2_excited_states(&ham, &x0, &[], &opts);

                let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x17_52);
                let ctx = qntt::su2::hess::weingarten_operator(&ham, &roots[0]);
                let (base_l, base_r) = {
                    let (l, r) = ctx.bases();
                    (l.clone(), r.clone())
                };
                let k = nroots.saturating_sub(1).max(1);
                let v0s: Vec<Su2TtTangent<f64>> = (0..k + 2)
                    .map(|_| {
                        let dirv: Su2TtVector<f64> = su2_tt_randn(space, &roots[0].ranks, &mut rng);
                        Su2TtTangent::project(&base_l, &base_r, &dirv)
                    })
                    .collect();
                let (so, _) = qntt::su2::hess::symplectic_lowest_signed_davidson(
                    &ctx,
                    &v0s,
                    k,
                    &SymplecticDavidsonOpts::default(),
                );
                let omega = rpa_omega("su2 RPA:      ", &so);
```

Which S = 0 level did each root land on? The ladder is not guaranteed to land
on levels 0, 1, 2 — the licence does not certify completeness — so the reference
gap of root `j` is read off the level it actually hit.

```rust
                let hit: Vec<(usize, f64)> = e.iter().map(|&x| nearest(levels, x)).collect();
                let subset_dev = hit.iter().map(|&(_, dv)| dv).fold(0.0f64, f64::max);
                let singlet_dev = e
                    .iter()
                    .enumerate()
                    .map(|(k, &x)| (x - levels[k.min(levels.len() - 1)]).abs())
                    .fold(0.0f64, f64::max);
                let idx: Vec<usize> = hit.iter().map(|&(j, _)| j).collect();
                let devs = report("su2", &e, &info, levels, &idx, omega.as_deref(), e_core);
                println!(
                    "su2: the ladder reached S = 0 levels {idx:?} (distance to them \
                     {subset_dev:.3e}); against the levels 0..{} in order the deviation is \
                     {singlet_dev:.3e}.",
                    e.len() - 1
                );
                if idx.iter().enumerate().any(|(k, &j)| j != k) {
                    println!(
                        "su2: the aufbau CSF start SKIPPED at least one S = 0 level — the \
                         completeness caveat (a degenerate pair carries no weight in the shared \
                         state-average basis and H cannot repopulate it). REPORTED, not an error: \
                         `licence` never certifies completeness. Remedies: one more root, or a \
                         start that already carries that symmetry."
                    );
                }
                su2 = Some((devs, subset_dev, singlet_dev));
            }
        }
    }
```

### summary + acceptance

```rust
    let mut s = Summary::new("step-17")
        .s("fixture", &fixture)
        .i("rank", rank as i64)
        .i("nroots", nroots as i64)
        .s("layer", &layer);
    if let Some(a) = &ab {
        s = s
            .e("ab_gap_dev_sa", a.sa, 3)
            .e("ab_gap_dev_ss", a.ss, 3)
            .e("ab_gap_dev_pol", a.pol, 3)
            .e("ab_omega_dev", a.omega, 3)
            .b("ab_licence", a.licence);
    }
    if let Some((devs, sub, sing)) = &su2 {
        s = s
            .e("su2_subset_dev", *sub, 3)
            .e("su2_singlet_dev", *sing, 3)
            .b("su2_licence", devs.licence);
    }
    s.emit();
```

At full rank the manifold is the whole sector: every stage is an exact diagonalization and
ω is the exact gap, so the acceptance line is a hard numeric claim. At a cap it is not —
the truncation error is the point of running there — so only the licences are checked.

```rust
    let ab_ok = ab.as_ref().is_none_or(|a| {
        a.licence
            && (rank > 0
                || (a.sa <= ACCEPT_TOL
                    && a.ss <= ACCEPT_TOL
                    && a.pol <= ACCEPT_TOL
                    && a.omega <= ACCEPT_TOL))
    });
    let su2_ok = su2
        .as_ref()
        .is_none_or(|(devs, sub, _)| devs.licence && (rank > 0 || *sub <= ACCEPT_TOL));
    let pass = ab_ok && su2_ok;
    let criterion = if rank == 0 {
        format!(
            "criterion: every SA/SS/polish/RPA gap within {ACCEPT_TOL:.0e} of the dense gap and \
             every licence true"
        )
    } else {
        "criterion: every licence true (a capped ladder does not reproduce the dense gaps by \
         construction)"
            .to_string()
    };
    println!(
        "\nACCEPTANCE {fixture} rank {rank}: {} ({criterion})",
        if pass { "PASS" } else { "FAIL" }
    );
```

A capped run whose only refused clause is the overlap check is expected behaviour, not a
defect: the fixed-rank penalized minimizer keeps an O(1/w) residual overlap with the rungs
below it, so `overlap_tol = 1e-6` is out of reach at a binding cap.

```rust
    if !pass && rank > 0 {
        let ab_gram = ab
            .as_ref()
            .map(|a| !a.licence && a.overlap_dev > 1e-6)
            .unwrap_or(false);
        let su2_gram = su2
            .as_ref()
            .map(|(d, _, _)| !d.licence && d.overlap_dev > 1e-6)
            .unwrap_or(false);
        if ab_gram || su2_gram {
            println!(
                "  the refused licence is the Gram gate at a BINDING cap (abelian overlap_dev \
                 {:.3e}, su2 {:.3e}): the fixed-rank penalized minimizer keeps an O(1/w) residual \
                 overlap with the rungs below it, so 1e-6 is not reachable at this rank. Raise \
                 --rank (16 clears it on this fixture) or fix a larger penalty; do NOT loosen \
                 `overlap_tol`.",
                ab.as_ref().map(|a| a.overlap_dev).unwrap_or(f64::NAN),
                su2.as_ref()
                    .map(|(d, _, _)| d.overlap_dev)
                    .unwrap_or(f64::NAN)
            );
        }
    }
}
```

## Results

At full rank on LiH/STO-3G every column agrees to the printed digits: the abelian ladder
`−7.882401932 / −7.766418475 / −7.749216187` Ha is the dense `E[0..3]`, the state-average,
state-specific and polished gaps `0.115983457` and `0.133185746` Ha equal both the RPA `ω` and
the dense gaps to `1e-13`, the penalty weights settle at ten times each gap (`1.16`, `1.33`),
and the licence is true with an overlap deviation of `2e-14`. The SU(2) ladder reaches the
singlet levels `[0, 1, 4]`: its rungs are exact singlet eigenvalues, but the degenerate Π pair
(levels 2 and 3, gap `0.185`) is skipped and the licence is true regardless — the completeness
caveat above, printed as a deviation of `3.785e-1` against levels `0..2` in order. The `ω`
column follows the ladder there, since the response manifold at the aufbau-derived ground state
carries no weight in that sector either.

At `--rank 16` the state-average energies sit `2e-6`–`7e-6` Ha above the dense levels while
the state-specific and polished rungs recover them to `1e-13` — the cap binds only on the pooled
basis. At `--rank 8` every route is off (state average `1.7e-4`–`5.4e-4`, polished `1.6e-5`–
`1.5e-4`, RPA `1.4e-4`–`1.8e-4` on the gaps), the polished rung is the closest on every root,
and the licence is refused by the overlap check alone: the fixed-rank penalized minimizer keeps a
residual overlap of about `5e-5` with the rungs below it (`O(1/w)`), so `overlap_tol = 1e-6` is
not reachable at that rank. The printed remedy is to raise the rank or the penalty — never to
loosen the tolerance.

Recorded output (capture release-0.1):

```text
== excited-state ladder — lih_sto3g (d=6, N↑=2 N↓=2, rank cap 0, nroots 3) ==
HF -7.862023860 Ha, FCI reference -7.882401932 Ha (meta); every table below is TOTAL energy (electronic + e_core = 0.995317638)
dense oracle:  225 determinants in (2,2), E[0..3] = [-8.877719570384329, -8.76173611320165, -8.74453382460122]
               105 of them are S = 0 levels, s0[0..3] = [-8.877719570384329, -8.74453382460122, -8.692269625852175]
abelian RPA:   56 basis vectors, 56 applies, converged=true

abelian: ladder energies (Ha, total = electronic + e_core)
  root     E (state avg)    E (state spec)      E (polished)  penalty w   ‖grad J‖   passes   conv
     0      -7.882401932      -7.882401932      -7.882401932     0.00e0   7.99e-13        0    yes
     1      -7.766418475      -7.766418475      -7.766418475     1.16e0   7.88e-13        0    yes
     2      -7.749216187      -7.749216187      -7.749216187     1.33e0   7.63e-13        0    yes

abelian: gaps to root 0 (Ha) against the dense reference spectrum
    j ref lvl        gap (SA)        gap (SS)       gap (pol)         ω (RPA)     gap (dense) pol − dense
    1       1     0.115983457     0.115983457     0.115983457     0.115983457     0.115983457  -5.684e-14
    2       2     0.133185746     0.133185746     0.133185746     0.133185746     0.133185746   1.119e-13
abelian: the ω column is the RPA spectrum in ASCENDING order at root 0's manifold, so when the ladder skips a level the two columns stop lining up row for row.
abelian: licence=true converged=[true, true, true] swaps=0 swapped=[] overlap_dev=2.427e-14 passes=[0, 0, 0] enriched=true
abelian: state-average stage — 9 sweeps, max bond rank 23, 392 local applies
su2 RPA:       44 basis vectors, 44 applies, converged=true

su2: ladder energies (Ha, total = electronic + e_core)
  root     E (state avg)    E (state spec)      E (polished)  penalty w   ‖grad J‖   passes   conv
     0      -7.882401932      -7.882401932      -7.882401932     0.00e0   4.96e-13        0    yes
     1      -7.749216187      -7.749216187      -7.749216187     1.33e0   5.03e-13        0    yes
     2      -7.318494520      -7.318494520      -7.318494520     5.64e0   6.36e-13        0    yes

su2: gaps to root 0 (Ha) against the dense reference spectrum
    j ref lvl        gap (SA)        gap (SS)       gap (pol)         ω (RPA)     gap (dense) pol − dense
    1       1     0.133185746     0.133185746     0.133185746     0.133185746     0.133185746   1.084e-13
    2       4     0.563907413     0.563907413     0.563907413     0.563907413     0.563907413  -4.974e-14
su2: the ω column is the RPA spectrum in ASCENDING order at root 0's manifold, so when the ladder skips a level the two columns stop lining up row for row.
su2: licence=true converged=[true, true, true] swaps=0 swapped=[] overlap_dev=1.686e-14 passes=[0, 0, 0] enriched=true
su2: state-average stage — 9 sweeps, max bond rank 16, 376 local applies
su2: the ladder reached S = 0 levels [0, 1, 4] (distance to them 1.705e-13); against the levels 0..2 in order the deviation is 3.785e-1.
su2: the aufbau CSF start SKIPPED at least one S = 0 level — the completeness caveat (a degenerate pair carries no weight in the shared state-average basis and H cannot repopulate it). REPORTED, not an error: `licence` never certifies completeness. Remedies: one more root, or a start that already carries that symmetry.
SUMMARY step-17 fixture=lih_sto3g rank=0 nroots=3 layer=both ab_gap_dev_sa=1.030e-13 ab_gap_dev_ss=1.048e-13 ab_gap_dev_pol=1.119e-13 ab_omega_dev=1.166e-13 ab_licence=true su2_subset_dev=1.705e-13 su2_singlet_dev=3.785e-1 su2_licence=true

ACCEPTANCE lih_sto3g rank 0: PASS (criterion: every SA/SS/polish/RPA gap within 1e-8 of the dense gap and every licence true)
```

## Possibilities for extensions

- Run `--layer su2 --rank 16` and compare the singlet levels reached with the full-rank run:
  the capped state-average trajectory finds the Π pair the full-rank one skips, and more rank
  does not restore it monotonically. The program prints the levels it reached.
- Ask for `--nroots 4` or `5` on the abelian layer and watch which level the top root converges
  to: the top state-average root converges only at the gap to the first excluded level.
- Change the start: the abelian ladder begins at the HF determinant and needs one noisy sweep
  to host three roots (`info.enriched`); a random start of larger rank skips that stage.
- Print `info.penalties` against `10·(E_i − E_0)` from the dense spectrum: the weight rule
  tracks whatever level a rung actually sits on.
