use super::*;
use qntt_testkit::dense_apply_string;
pub(crate) use qntt_testkit::randomized_state;

#[allow(dead_code)]
fn det_index(d: usize, up_site: usize, dn_site: usize) -> usize {
    let mut idx = 0usize;
    for k in (0..d).rev() {
        let fam = ((k == up_site) as usize) | (((k == dn_site) as usize) << 1);
        idx = (idx << 2) | fam;
    }
    idx
}

pub(crate) fn dense_apply_hamiltonian_12<T: Scalar>(
    d: usize,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    vec: &[T],
) -> Vec<T> {
    let len = vec.len();
    let mut out = vec![T::zero(); len];
    let spins = [Spin::Up, Spin::Dn];
    let accumulate = |ops: &[(Orbital, bool)], coeff: T, out: &mut [T]| {
        let term = dense_apply_string(d, ops, vec);
        for (o, tm) in out.iter_mut().zip(term) {
            *o += tm * coeff;
        }
    };
    for i in 0..d {
        for j in 0..d {
            let c = t.get(&[i, j]);
            if c == T::zero() {
                continue;
            }
            for &s in &spins {
                accumulate(
                    &[(Orbital::new(i, s), true), (Orbital::new(j, s), false)],
                    c,
                    &mut out,
                );
            }
        }
    }
    let half = T::from_f64(0.5);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let c = v.get(&[i, j, k, l]);
                    if c == T::zero() {
                        continue;
                    }
                    for &s1 in &spins {
                        for &s2 in &spins {
                            let (oi, oj) = (Orbital::new(i, s1), Orbital::new(j, s1));
                            let (ok, ol) = (Orbital::new(k, s2), Orbital::new(l, s2));
                            if oi != ok && oj != ol {
                                accumulate(
                                    &[(oi, true), (ok, true), (ol, false), (oj, false)],
                                    c * half,
                                    &mut out,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

pub(crate) fn random_coeffs(
    d: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            t.set(&[i, j], rng.sample::<f64, _>(StandardNormal));
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if v.get(&[i, j, k, l]) != 0.0 {
                        continue;
                    }
                    let x: f64 = rng.sample(StandardNormal);
                    v.set(&[i, j, k, l], x);
                    v.set(&[k, l, i, j], x);
                }
            }
        }
    }
    (t, v)
}

pub(crate) fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in a.iter().zip(b) {
        num += (x - y) * (x - y);
        den += y * y;
    }
    (num / den.max(1e-300)).sqrt()
}

pub(crate) fn build_bipartite(
    space: QnSpace,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
) -> SparseHamiltonian<f64> {
    SparseHamiltonian::new_with(space, t, v, None, f64::EPSILON, Construction::Bipartite)
}

pub(crate) fn build_median(
    space: QnSpace,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
) -> SparseHamiltonian<f64> {
    SparseHamiltonian::new_with(space, t, v, None, f64::EPSILON, Construction::Median)
}

pub(crate) fn build_persector(
    space: QnSpace,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
) -> SparseHamiltonian<f64> {
    SparseHamiltonian::new_with(
        space,
        t,
        v,
        None,
        f64::EPSILON,
        Construction::BipartitePerSector,
    )
}

pub(crate) fn hash_assembled(a: &qntt::hamiltonian::assembly::Assembled<f64>) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let nirrep = a.state_qns[0].nirrep();
    a.coo.len().hash(&mut h);
    for (k, nest) in a.coo.iter().enumerate() {
        for (yf, ysec, xf, xsec, cell) in nest.iter_grid_order() {
            (
                k, yf as u8, ysec.nup, ysec.ndn, xf as u8, xsec.nup, xsec.ndn,
            )
                .hash(&mut h);
            if nirrep > 1 {
                (ysec.irrep.index(), xsec.irrep.index()).hash(&mut h);
            }
            cell.rows.len().hash(&mut h);
            for ((&r, &c), &val) in cell.rows.iter().zip(&cell.cols).zip(&cell.vals) {
                r.hash(&mut h);
                c.hash(&mut h);
                val.to_bits().hash(&mut h);
            }
        }
    }
    a.state_qns.len().hash(&mut h);
    for (b, grid) in a.state_qns.iter().enumerate() {
        for (s, list) in grid.iter() {
            if list.is_empty() {
                continue;
            }
            (b, s.nup, s.ndn).hash(&mut h);
            if nirrep > 1 {
                s.irrep.index().hash(&mut h);
            }
            for sec in list {
                (sec.nup, sec.ndn).hash(&mut h);
                if nirrep > 1 {
                    sec.irrep.index().hash(&mut h);
                }
            }
        }
    }
    h.finish()
}

pub(crate) fn assert_cores_bit_identical(y1: &TtVector<f64>, y2: &TtVector<f64>, ctx: &str) {
    assert_eq!(y1.ranks, y2.ranks, "{ctx}: rank grids differ");
    let d = y1.space.d;
    for k in 0..d {
        for f in Family::ALL {
            for s in y1.cores[k].transitions(f) {
                let a = y1.cores[k].block(f, s).expect("block a present");
                let b = y2.cores[k].block(f, s).expect("block b present");
                assert_eq!(a.nrows(), b.nrows(), "{ctx}: core {k} {f:?} {s:?} nrows");
                assert_eq!(a.ncols(), b.ncols(), "{ctx}: core {k} {f:?} {s:?} ncols");
                for col in 0..a.ncols() {
                    for row in 0..a.nrows() {
                        assert_eq!(
                            a.get(row, col).to_bits(),
                            b.get(row, col).to_bits(),
                            "{ctx}: core {k} family {f:?} sector {s:?} entry ({row},{col}) differs"
                        );
                    }
                }
            }
        }
    }
}
