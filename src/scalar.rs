// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Scalar field abstraction: `f64` (real) and `Complex<f64>` (Hermitian problems).

use faer::traits::ComplexField;
use num_complex::Complex;
use rand::{Rng, RngExt};
use rand_distr::StandardNormal;

/// Complex double-precision scalar (same type as `faer::c64`).
pub type C64 = Complex<f64>;

mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for super::C64 {}
}

/// Sealed scalar trait for TT entries. Implemented exactly for `f64` and `Complex<f64>`.
pub trait Scalar:
    ComplexField<Real = f64>
    + Copy
    + Send
    + Sync
    + 'static
    + PartialEq
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + private::Sealed
{
    /// `true` for `Complex<f64>`.
    const IS_COMPLEX: bool;

    /// Additive identity.
    #[inline]
    fn zero() -> Self {
        Self::zero_impl()
    }

    /// Multiplicative identity.
    #[inline]
    fn one() -> Self {
        Self::one_impl()
    }

    /// Embed a real number.
    #[inline]
    fn from_f64(x: f64) -> Self {
        Self::from_f64_impl(x)
    }

    /// Complex conjugate (identity for `f64`).
    #[inline]
    fn conj_s(self) -> Self {
        Self::conj_impl(&self)
    }

    /// Modulus `|x|`.
    #[inline]
    fn abs_s(self) -> f64 {
        Self::abs_impl(&self)
    }

    /// Real part.
    #[inline]
    fn re_s(self) -> f64 {
        Self::real_part_impl(&self)
    }

    /// Imaginary part (`0` for `f64`). Used by the real/imag embedding.
    #[inline]
    fn im_s(self) -> f64 {
        Self::imag_part_impl(&self)
    }

    /// Multiplicative inverse `1/x`. Implemented per-field with a robustly scaled
    /// algorithm for complex so it does not overflow/underflow `|x|²` for extreme magnitudes
    /// (the naive `conj(x)/|x|²` loses the answer for `|x| ≳ 1.3e154` or `≲ 1e-154`).
    fn recip(self) -> Self;

    /// Field exponential `exp(x)` (`f64::exp` for real, `Complex::exp` for complex). Used by the
    /// tridiagonal-exponential bridge [`crate::core::factor::tridiag_exp_e1`] backing the
    /// Lanczos `expmv`.
    fn exp_s(self) -> Self;

    /// Construct from an explicit `(re, im)` pair. For `f64` the imaginary part is dropped
    /// (callers that need a genuine imaginary coefficient — e.g. real-time TDVP's `a = -i·dt/2` —
    /// gate on [`Scalar::IS_COMPLEX`] first). For `Complex<f64>` returns `re + i·im`.
    fn from_complex_f64(re: f64, im: f64) -> Self;

    /// Draw from the standard normal distribution of the field: `N(0,1)` for `f64`,
    /// circularly-symmetric `CN(0,1)` for complex (components have variance 1/2 each, each part
    /// `N(0, 1/2)` so the modulus has unit variance).
    fn sample_standard<R: Rng + ?Sized>(rng: &mut R) -> Self;
}

impl Scalar for f64 {
    const IS_COMPLEX: bool = false;

    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }

    #[inline]
    fn exp_s(self) -> Self {
        self.exp()
    }

    #[inline]
    fn from_complex_f64(re: f64, _im: f64) -> Self {
        re
    }

    #[inline]
    fn sample_standard<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.sample(StandardNormal)
    }
}

impl Scalar for C64 {
    const IS_COMPLEX: bool = true;

    #[inline]
    fn recip(self) -> Self {
        let (a, b) = (self.re, self.im);
        if a.abs() >= b.abs() {
            let r = b / a;
            let den = a + b * r;
            C64::new(1.0 / den, -r / den)
        } else {
            let r = a / b;
            let den = a * r + b;
            C64::new(r / den, -1.0 / den)
        }
    }

    #[inline]
    fn exp_s(self) -> Self {
        self.exp()
    }

    #[inline]
    fn from_complex_f64(re: f64, im: f64) -> Self {
        C64::new(re, im)
    }

    #[inline]
    fn sample_standard<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let re: f64 = rng.sample(StandardNormal);
        let im: f64 = rng.sample(StandardNormal);
        C64::new(re, im) * std::f64::consts::FRAC_1_SQRT_2
    }
}

pub(crate) const GEMM_PAR_MIN_WORK: usize = 1 << 21;

pub(crate) const GEMM_PAR_WIDTH: usize = 4;

#[inline]
pub(crate) fn gemm_par(m: usize, n: usize, k: usize) -> faer::Par {
    let work = m.saturating_mul(n).saturating_mul(k);
    #[cfg(test)]
    let min_work =
        gemm_par_hook::min_work().unwrap_or_else(|| crate::knobs::GEMM_PAR_MIN_WORK.get());
    #[cfg(not(test))]
    let min_work = crate::knobs::GEMM_PAR_MIN_WORK.get();
    let par = if work < min_work {
        faer::Par::Seq
    } else {
        gemm_par_at(
            m,
            n,
            k,
            min_work,
            crate::knobs::GEMM_PAR_WIDTH.get(),
            rayon::current_num_threads(),
        )
    };
    #[cfg(test)]
    gemm_par_hook::record(par);
    par
}

pub(crate) fn gemm_par_at(
    m: usize,
    n: usize,
    k: usize,
    min_work: usize,
    width: usize,
    pool: usize,
) -> faer::Par {
    let work = m.saturating_mul(n).saturating_mul(k);
    if min_work == usize::MAX || work < min_work || pool <= 1 || m.min(n).min(k) <= 1 {
        return faer::Par::Seq;
    }
    let width = if width == 0 { pool } else { width.min(pool) };
    if width <= 1 {
        faer::Par::Seq
    } else {
        faer::Par::rayon(width)
    }
}

pub(crate) fn gram_colmajor(a: &[f64], m: usize, n: usize, par: faer::Par) -> Vec<f64> {
    let am = faer::MatRef::from_column_major_slice(a, m, n);
    let mut t = faer::Mat::<f64>::zeros(n, n);
    faer::linalg::matmul::matmul(t.as_mut(), faer::Accum::Replace, am.adjoint(), am, 1.0, par);
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v = *t.get(i, j);
            out[i * n + j] = v;
            out[j * n + i] = v;
        }
    }
    out
}

pub(crate) fn matmul_colmajor_rowmajor(
    a: &[f64],
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
    par: faer::Par,
) -> Vec<f64> {
    let am = faer::MatRef::from_column_major_slice(a, m, k);
    let bm = faer::MatRef::from_row_major_slice(b, k, n);
    let mut out = vec![0.0f64; m * n];
    faer::linalg::matmul::matmul(
        faer::MatMut::from_column_major_slice_mut(&mut out, m, n),
        faer::Accum::Replace,
        am,
        bm,
        1.0,
        par,
    );
    out
}

#[cfg(test)]
pub(crate) mod gemm_par_hook {
    use std::cell::RefCell;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Default)]
    pub(crate) struct Hook {
        pub(crate) min_work: Option<usize>,
        pub(crate) taken: AtomicUsize,
        pub(crate) calls: AtomicUsize,
    }

    impl Hook {
        pub(crate) fn forced() -> Arc<Hook> {
            Arc::new(Hook {
                min_work: Some(0),
                ..Hook::default()
            })
        }
        pub(crate) fn observed() -> Arc<Hook> {
            Arc::new(Hook {
                min_work: Some(super::GEMM_PAR_MIN_WORK),
                ..Hook::default()
            })
        }
        pub(crate) fn taken(&self) -> usize {
            self.taken.load(Ordering::Relaxed)
        }
        pub(crate) fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    thread_local! {
        static HOOK: RefCell<Option<Arc<Hook>>> = const { RefCell::new(None) };
    }

    pub(super) fn min_work() -> Option<usize> {
        HOOK.with(|h| h.borrow().as_ref().and_then(|h| h.min_work))
    }

    pub(super) fn record(par: faer::Par) {
        HOOK.with(|h| {
            if let Some(h) = h.borrow().as_ref() {
                h.calls.fetch_add(1, Ordering::Relaxed);
                if matches!(par, faer::Par::Rayon(_)) {
                    h.taken.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
    }

    pub(crate) fn pool(nt: usize, hook: &Arc<Hook>) -> rayon::ThreadPool {
        let hook = Arc::clone(hook);
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .start_handler(move |_| HOOK.with(|h| *h.borrow_mut() = Some(Arc::clone(&hook))))
            .build()
            .unwrap()
    }
}

#[cfg(test)]
mod gemm_par_tests;
