// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::TtTangent;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::Su2Space;
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::Su2TtVector;
use qntt::{QnSpace, Scalar, SparseHamiltonian, TtVector, dot, tt_state};

pub fn grad_exact_at<T: Scalar>(ham: &SparseHamiltonian<T>, y: &TtVector<T>) -> TtTangent<T> {
    let lam = qntt::rayleigh_quotient(ham, y).re_s();
    let hy = ham.apply(y);
    let res = &hy - &(y * T::from_f64(lam));
    let (bl, br) = qntt::solvers::riemannian::tangent_bases(y);
    TtTangent::project(&bl, &br, &res)
}

pub fn su2_grad_exact_at<T: Scalar>(
    ham: &Su2SparseHamiltonian<T>,
    y: &Su2TtVector<T>,
) -> Su2TtTangent<T> {
    let lam = qntt::su2::hamiltonian::rayleigh_quotient(ham, y).re_s();
    let hy = ham.apply(y);
    let res = &hy - &(y * T::from_f64(lam));
    let (bl, br) = qntt::su2::riemannian::tangent_bases(y);
    Su2TtTangent::project(&bl, &br, &res)
}

pub fn su2_radial_mode_at<T: Scalar>(
    bl: &Su2TtVector<T>,
    br: &Su2TtVector<T>,
    x: &Su2TtVector<T>,
) -> Su2TtTangent<T> {
    let mut r = Su2TtTangent::project(bl, br, x);
    let n = r.norm();
    r.scale_mut(T::from_f64(1.0 / n));
    r
}

pub fn su2_deflate_radial_span<T: Scalar>(radial: &Su2TtTangent<T>, v: &mut Su2TtTangent<T>) {
    for _ in 0..2 {
        let c = radial.dot(v);
        v.axpy(-c, radial);
    }
}

pub fn su2_kick_along<T: Scalar>(v0: &Su2TtTangent<T>, eps: f64) -> Su2TtVector<T> {
    let mut x = v0.retract_hosvd(T::from_f64(eps));
    qntt::su2::riemannian::normalize_mut(&mut x);
    x.check();
    x
}

pub fn combinations(d: usize, n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if n == 0 {
        return vec![vec![]];
    }
    if n > d {
        return out;
    }
    let mut cur: Vec<usize> = (0..n).collect();
    loop {
        out.push(cur.clone());
        let mut i = n;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if cur[i] != i + d - n {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        cur[i] += 1;
        for j in i + 1..n {
            cur[j] = cur[j - 1] + 1;
        }
    }
}

pub fn det_states<T: Scalar>(d: usize, nup: usize, ndn: usize) -> Vec<TtVector<T>> {
    let ups = combinations(d, nup);
    let dns = combinations(d, ndn);
    let mut out = Vec::with_capacity(ups.len() * dns.len());
    for up in &ups {
        for dn in &dns {
            let mut occ_up = vec![false; d];
            let mut occ_dn = vec![false; d];
            for &s in up {
                occ_up[s] = true;
            }
            for &s in dn {
                occ_dn[s] = true;
            }
            out.push(tt_state::<T>(&occ_up, &occ_dn));
        }
    }
    out
}

pub fn dense_h_in_sector<T: Scalar>(states: &[TtVector<T>], ham: &SparseHamiltonian<T>) -> Vec<T> {
    let n = states.len();
    let mut mat = vec![T::zero(); n * n];
    for (j, sj) in states.iter().enumerate() {
        let hs = ham.apply(sj);
        for (i, si) in states.iter().enumerate() {
            mat[i * n + j] = dot(si, &hs);
        }
    }
    mat
}

#[allow(clippy::needless_range_loop)]
pub fn dense_sector_eigen<T: Scalar>(mat: &[T], n: usize) -> (Vec<f64>, Vec<Vec<T>>) {
    let mut m = faer::Mat::<T>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            *m.get_mut(i, j) = mat[i * n + j];
        }
    }
    let (evals, u) = qntt::core::factor::dense_self_adjoint_eigen(m.as_ref());
    let evecs: Vec<Vec<T>> = (0..n)
        .map(|c| (0..n).map(|r| *u.get(r, c)).collect())
        .collect();
    (evals, evecs)
}

pub fn coeff_vector<T: Scalar>(states: &[TtVector<T>], x: &TtVector<T>) -> Vec<T> {
    states.iter().map(|s| dot(s, x)).collect()
}

pub fn overlap_modulus<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut acc = T::zero();
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        acc += a[i].conj_s() * b[i];
        na += a[i].abs_s().powi(2);
        nb += b[i].abs_s().powi(2);
    }
    acc.abs_s() / (na.sqrt() * nb.sqrt()).max(1e-300)
}

pub fn singlet_indices(ev22: &[f64], ev31: &[f64]) -> Vec<usize> {
    ev22.iter()
        .enumerate()
        .filter(|(_, e)| {
            ev31.iter()
                .all(|p| (*e - p).abs() > 1e-8 * e.abs().max(1.0))
        })
        .map(|(i, _)| i)
        .collect()
}

pub struct Su2MolecularArm {
    pub ham: Su2SparseHamiltonian<f64>,
    pub dets: Vec<TtVector<f64>>,
    pub ev: Vec<f64>,
    pub evecs: Vec<Vec<f64>>,
    pub singlets: Vec<usize>,
    pub xgs: Su2TtVector<f64>,
    pub gs_gnorm: f64,
    pub x0: Su2TtVector<f64>,
    pub rank_max: usize,
}

pub fn su2_molecular_ground_state(
    space: Su2Space,
    ham: &Su2SparseHamiltonian<f64>,
    m: usize,
) -> Su2TtVector<f64> {
    use qntt::solvers::mals::NoiseType;
    use qntt::su2::core::Family;
    use qntt::su2::riemannian::{
        RiemannianOpts as Su2RiemannianOpts, normalize_mut as su2_normalize_mut,
        riemannian_cg as su2_cg,
    };
    use qntt::su2::solvers::mals::{MalsOptions, mals};
    use qntt::su2::vector::tt_csf;

    let nocc = space.n / 2;
    let codes: Vec<Family> = (0..space.d)
        .map(|i| if i < nocc { Family::Occ } else { Family::Empty })
        .collect();
    let mut x0: Su2TtVector<f64> = tt_csf(&codes);
    x0.right_orthogonalize(true);
    let mopts = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![
            (m / 4).max(6),
            (m / 2).max(10),
            (3 * m / 4).max(14),
            m,
            m,
            m,
        ],
        noise: vec![
            3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        seed: 0x9E37_79B9,
        noise_type: NoiseType::Random,
        max_iter: 30,
        ..Default::default()
    };
    let (_lmals, xa) = mals(ham, &x0, &mopts);
    let mut opts = Su2RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    let (_l, mut x) = su2_cg(ham, &xa, &opts);
    for _ in 0..30 {
        if su2_grad_exact_at(ham, &x).norm() < 1e-10 {
            break;
        }
        let (_l2, x2) = su2_cg(ham, &x, &opts);
        x = x2;
    }
    su2_normalize_mut(&mut x);
    x.check();
    x
}

pub fn su2_molecular_arm(dir: &str, m: usize) -> Su2MolecularArm {
    use qntt::hamiltonian::assembly::Construction;
    use qntt::integrals::{IntegralsOptions, load_fixture_dir};
    use qntt::solvers::hess::HessPart;

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("qntt-testkit lives one level below the repo root")
        .join(dir);
    let load = load_fixture_dir::<f64>(&path, &IntegralsOptions::default())
        .unwrap_or_else(|e| panic!("{dir}: {e}"));
    let ints = load.ints;
    let (d, nelec) = (ints.norb, ints.nelec);
    let (nup, ndn) = (nelec / 2, nelec / 2);
    let (t, v) = (ints.t, ints.v);

    let sp = QnSpace::new(d, nup, ndn);
    let h22 = SparseHamiltonian::new_with(sp, &t, &v, None, f64::EPSILON, Construction::Median);
    let dets = det_states::<f64>(d, nup, ndn);
    let (ev, evecs) = dense_sector_eigen(&dense_h_in_sector(&dets, &h22), dets.len());
    let sp31 = QnSpace::new(d, nup + 1, ndn - 1);
    let h31 = SparseHamiltonian::new_with(sp31, &t, &v, None, f64::EPSILON, Construction::Median);
    let dets31 = det_states::<f64>(d, nup + 1, ndn - 1);
    let (ev31, _) = dense_sector_eigen(&dense_h_in_sector(&dets31, &h31), dets31.len());
    let singlets = singlet_indices(&ev, &ev31);
    assert!(
        singlets.len() >= 3,
        "{dir}: sector-degeneracy spin labelling found only {} singlets — the reference this arm \
         is measured against does not exist",
        singlets.len()
    );

    let space = Su2Space::new(d, nelec, 0);
    let ham = Su2SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let xgs = su2_molecular_ground_state(space, &ham, m);
    let gs_gnorm = su2_grad_exact_at(&ham, &xgs).norm();
    let rank_max = (0..=space.d)
        .flat_map(|b| space.bond_sectors(b).into_iter().map(move |s| (b, s)))
        .map(|(b, s)| xgs.ranks[b][s])
        .max()
        .unwrap_or(0);

    let v0 = {
        let ctx = qntt::su2::hess::weingarten_operator(&ham, &xgs);
        let (bl, br) = ctx.bases();
        let diag = ctx.a_diagonal();
        let seed = Su2TtTangent::project(bl, br, &diag.to_vector());
        let (_verdict, w, _a) =
            qntt::su2::hess::hessian_pd_probe_witness(&ctx, HessPart::Rr, &seed, 300);
        let radial = su2_radial_mode_at(bl, br, bl);
        let mut vv = w;
        su2_deflate_radial_span(&radial, &mut vv);
        let n = vv.norm();
        vv.scale_mut(1.0 / n);
        vv
    };
    let x0 = su2_kick_along(&v0, 5e-2);
    let g0 = su2_grad_exact_at(&ham, &x0).norm();
    assert!(
        g0 > 1e-5,
        "{dir}: the kick left the state critical (‖grad‖ = {g0:.3e}) — vacuous arm"
    );

    Su2MolecularArm {
        ham,
        dets,
        ev,
        evecs,
        singlets,
        xgs,
        gs_gnorm,
        x0,
        rank_max,
    }
}

pub fn mol_opts(max_steps: usize, probe_budget: usize) -> qntt::solvers::gad::GadOpts {
    use qntt::solvers::gad::{GadOpts, WModel};
    use qntt::solvers::joint::WDampRule;
    GadOpts {
        gradtol: 1e-7,
        max_steps,
        k_davidson: 8,
        w_model: WModel::Damped(WDampRule::GradTracked),
        certify: true,
        probe_budget,
        preserve_grid: true,
        ..GadOpts::default()
    }
}
