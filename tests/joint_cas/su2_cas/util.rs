// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;
use std::sync::OnceLock;

use qntt::integrals::{Integrals, IntegralsOptions, load_fixture_dir};
use qntt::rotation::orbital_pairs;
use qntt::solvers::joint::cas::{CasPartition, cas_pairs};
use qntt::solvers::joint::{JointOpts, ZRoute};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::hess::{normal_residual, weingarten_operator};
use qntt::su2::joint::cas::{su2_cas_rank_grid, su2_cas_reference_state};
use qntt::su2::joint::{Su2JointContext, Su2JointGateError, su2_joint_context};
use qntt::su2::orbital::orbital_gradient;
use qntt::su2::qn::{Su2RankGrid, Su2Sector, Su2Space, max_ranks};
use qntt::su2::riemannian::tangent_bases;
use qntt::su2::vector::Su2TtVector;

pub(crate) const TT_GRAD_TOL: f64 = 1e-4;

pub(crate) struct Su2CasFixture {
    pub(crate) label: &'static str,
    pub(crate) space: Su2Space,
    pub(crate) ints: Integrals<f64>,
    pub(crate) ham: Su2SparseHamiltonian<f64>,
    pub(crate) part: CasPartition,
    pub(crate) m_active: usize,
    pub(crate) grid: Vec<Su2RankGrid>,
    pub(crate) lambda: f64,
    pub(crate) x: Su2TtVector<f64>,
    pub(crate) casscf_gs: Option<f64>,
}

impl Su2CasFixture {
    pub(crate) fn d(&self) -> usize {
        self.space.d
    }

    pub(crate) fn frames(&self) -> (Su2TtVector<f64>, Su2TtVector<f64>) {
        tangent_bases(&self.x)
    }

    pub(crate) fn tt_gradnorm(&self) -> f64 {
        weingarten_operator(&self.ham, &self.x).gradient().norm()
    }

    pub(crate) fn z_norm(&self) -> f64 {
        let (bl, br) = self.frames();
        let z = normal_residual(&self.ham, &bl, &br, &self.x, self.lambda);
        qntt::su2::ops::dot(&z, &z).sqrt()
    }

    pub(crate) fn orb_grad_norm(&self, pairs: &[(usize, usize)]) -> f64 {
        let g = orbital_gradient(&self.ham, &self.x);
        pairs
            .iter()
            .map(|&(p, q)| g.get(&[p, q]).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    pub(crate) fn cas_mask_pairs(&self) -> Vec<(usize, usize)> {
        cas_pairs(&self.part, self.d()).keep
    }

    pub(crate) fn full_pairs(&self) -> Vec<(usize, usize)> {
        orbital_pairs(self.space.pg, self.d())
    }

    pub(crate) fn joint_ctx(
        &self,
        mask: bool,
        orb_floor: f64,
    ) -> Result<Su2JointContext<'_>, Su2JointGateError> {
        let opts = JointOpts {
            tt_grad_tol: TT_GRAD_TOL,
            orb_floor,
            pairs: mask.then(|| cas_pairs(&self.part, self.d())),
            z_route: ZRoute::Explicit,
            b_general: false,
        };
        su2_joint_context(&self.ham, &self.x, &opts)
    }

    pub(crate) fn window(&self, i: usize) -> usize {
        if i < self.part.n_core {
            0
        } else if i < self.part.n_core + self.part.n_active {
            1
        } else {
            2
        }
    }

    pub(crate) fn pair_class(&self, p: usize, q: usize) -> &'static str {
        let (a, b) = (
            self.window(p).min(self.window(q)),
            self.window(p).max(self.window(q)),
        );
        match (a, b) {
            (0, 0) => "CC",
            (0, 1) => "CA",
            (0, 2) => "CV",
            (1, 1) => "AA",
            (1, 2) => "AV",
            _ => "VV",
        }
    }
}

fn fixture_dir(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name)
}

fn build(
    label: &'static str,
    dirname: &str,
    part: CasPartition,
    m_active: usize,
    seed: u64,
    n_override: Option<(usize, usize)>,
) -> Su2CasFixture {
    let load = load_fixture_dir::<f64>(&fixture_dir(dirname), &IntegralsOptions::default())
        .unwrap_or_else(|e| panic!("{label}: loading {dirname}: {e:?}"));
    let casscf_gs = load
        .meta
        .as_ref()
        .and_then(|m| m.keys.get("casscf_gs"))
        .and_then(|v| v.parse().ok());
    let ints = load.ints;
    let (n, two_s) = n_override.unwrap_or((ints.nelec, 0));
    let space = Su2Space::new(ints.norb, n, two_s);
    assert_eq!(
        part.d(),
        space.d,
        "{label}: partition covers {} orbitals but the dump has d = {}",
        part.d(),
        space.d
    );
    let ham = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
    let grid = su2_cas_rank_grid(&space, &part, qntt::PerSector(m_active));
    let (lambda, x) = su2_cas_reference_state(&ham, &space, &part, m_active, seed);
    Su2CasFixture {
        label,
        space,
        ints,
        ham,
        part,
        m_active,
        grid,
        lambda,
        x,
        casscf_gs,
    }
}

pub(crate) const LIH_PART: CasPartition = CasPartition {
    n_core: 1,
    n_active: 2,
    n_virt: 3,
};
pub(crate) const H2O_PART: CasPartition = CasPartition {
    n_core: 1,
    n_active: 6,
    n_virt: 0,
};

pub(crate) fn lih_cas22() -> &'static Su2CasFixture {
    static F: OnceLock<Su2CasFixture> = OnceLock::new();
    F.get_or_init(|| build("lih_cas22", "lih_sto3g", LIH_PART, 0, 0xCA22, None))
}

pub(crate) fn h2o_cas86() -> &'static Su2CasFixture {
    static F: OnceLock<Su2CasFixture> = OnceLock::new();
    F.get_or_init(|| build("h2o_cas86", "h2o_sto3g_cas", H2O_PART, 0, 0xCA86, None))
}

pub(crate) fn h2o_cas86_m2() -> &'static Su2CasFixture {
    static F: OnceLock<Su2CasFixture> = OnceLock::new();
    F.get_or_init(|| build("h2o_cas86_m2", "h2o_sto3g_cas", H2O_PART, 2, 0xCA82, None))
}

pub(crate) fn lih_doublet() -> &'static Su2CasFixture {
    static F: OnceLock<Su2CasFixture> = OnceLock::new();
    F.get_or_init(|| {
        build(
            "lih_doublet",
            "lih_sto3g",
            LIH_PART,
            0,
            0xD00B,
            Some((3, 1)),
        )
    })
}

pub(crate) fn grid_totals(grid: &[Su2RankGrid]) -> Vec<usize> {
    grid.iter().map(|g| g.total()).collect()
}

pub(crate) fn grid_diff(
    space: &Su2Space,
    got: &[Su2RankGrid],
    want: &[Su2RankGrid],
) -> Vec<(usize, Su2Sector, usize, usize)> {
    assert_eq!(got.len(), want.len(), "grid lengths differ");
    (0..=space.d)
        .flat_map(|b| {
            space
                .bond_sectors(b)
                .into_iter()
                .filter(move |&s| got[b][s] != want[b][s])
                .map(move |s| (b, s, got[b][s], want[b][s]))
        })
        .collect()
}

pub(crate) fn fmt_grid_diff(diff: &[(usize, Su2Sector, usize, usize)]) -> String {
    if diff.is_empty() {
        return "(identical)".to_string();
    }
    diff.iter()
        .take(12)
        .map(|(b, s, got, want)| {
            format!(
                "bond {b} (n={},2s={},g={}): got {got} want {want}",
                s.n,
                s.two_s,
                s.irrep.index()
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

pub(crate) fn grid_le(space: &Su2Space, got: &[Su2RankGrid], want: &[Su2RankGrid]) -> bool {
    got.len() == want.len()
        && (0..=space.d).all(|b| {
            space
                .bond_sectors(b)
                .into_iter()
                .all(|s| got[b][s] <= want[b][s])
        })
}

pub(crate) fn regrown_zero_sectors(
    space: &Su2Space,
    got: &[Su2RankGrid],
    want: &[Su2RankGrid],
) -> Vec<(usize, Su2Sector, usize)> {
    (0..=space.d)
        .flat_map(|b| {
            space
                .bond_sectors(b)
                .into_iter()
                .filter(move |&s| want[b][s] == 0 && got[b][s] > 0)
                .map(move |s| (b, s, got[b][s]))
        })
        .collect()
}

pub(crate) fn full_ranks(space: &Su2Space) -> Vec<Su2RankGrid> {
    max_ranks(space)
}

fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    (1..=k).fold(1usize, |acc, i| acc * (n - k + i) / i)
}

pub(crate) fn weyl_paldus(n_orb: usize, n_elec: usize, two_s: usize) -> usize {
    if two_s > n_elec || !(n_elec - two_s).is_multiple_of(2) {
        return 0;
    }
    let lo = (n_elec - two_s) / 2;
    let hi = (n_elec + two_s) / 2 + 1;
    (two_s + 1) * binom(n_orb + 1, lo) * binom(n_orb + 1, hi) / (n_orb + 1)
}

pub(crate) fn weyl_paldus_sum(n_orb: usize, n_elec: usize) -> usize {
    (0..=n_elec)
        .step_by(2)
        .map(|two_s| weyl_paldus(n_orb, n_elec, two_s))
        .sum()
}

pub(crate) fn expected_n_eff(part: &CasPartition) -> usize {
    let (ni, na, ne) = (part.n_core, part.n_active, part.n_virt);
    na * ni + ni * ne + na * ne
}
