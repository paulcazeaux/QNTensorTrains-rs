// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

fn b_delta(rho: f64, delta: f64) -> Option<f64> {
    if !rho.is_finite() {
        return None;
    }
    let cond = 4.0 * delta * (delta - 1.0) * rho * rho;
    if cond >= 1.0 {
        return None;
    }
    Some(2.0 * delta / (1.0 + 2.0 * delta * rho * rho + (1.0 - cond).sqrt()))
}

pub(super) fn eigenvector(rho: f64, delta: f64) -> f64 {
    let Some(bdelta) = b_delta(rho, delta) else {
        return f64::INFINITY;
    };
    let arg = (delta * bdelta).sqrt() * rho;
    if arg > 1.0 {
        return f64::INFINITY;
    }
    2.0 * (0.5 * arg.asin()).sin()
}

pub(super) fn eigenvalue(rho: f64, delta: f64) -> f64 {
    let Some(bdelta) = b_delta(rho, delta) else {
        return f64::INFINITY;
    };
    let denom = 1.0 - bdelta * rho * rho;
    if denom <= 0.0 {
        return f64::INFINITY;
    }
    bdelta / denom * rho * rho
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_small_rho_asymptotics() {
        let delta = 50.0;
        assert!((b_delta(0.0, delta).unwrap() - delta).abs() < 1e-12);
        for &rho in &[1e-6, 1e-7, 1e-8] {
            let ratio_vec = eigenvector(rho, delta) / (delta * rho);
            let ratio_val = eigenvalue(rho, delta) / (delta * rho * rho);
            assert!(
                (ratio_vec - 1.0).abs() < 1e-6,
                "eigenvector bound / δρ = {ratio_vec} at ρ={rho:e}"
            );
            assert!(
                (ratio_val - 1.0).abs() < 1e-6,
                "eigenvalue bound / δρ² = {ratio_val} at ρ={rho:e}"
            );
        }
    }

    #[test]
    fn eigenvalue_bound_is_below_eigenvector_bound() {
        let delta = 50.0;
        for k in 1..200 {
            let rho = 1e-9 * 1.15f64.powi(k);
            let (v, e) = (eigenvector(rho, delta), eigenvalue(rho, delta));
            if v.is_finite() && e.is_finite() {
                assert!(e <= v, "eq-5.18 {e:e} > Thm-5.7 {v:e} at ρ={rho:e}");
            }
        }
    }

    #[test]
    fn gap_condition_refuses_together() {
        let delta = 50.0f64;
        let rho_crit = 1.0 / (4.0 * delta * (delta - 1.0)).sqrt();
        assert!(eigenvector(rho_crit * 1.01, delta).is_infinite());
        assert!(eigenvalue(rho_crit * 1.01, delta).is_infinite());
        assert!(eigenvector(rho_crit * 0.5, delta).is_finite());
        assert!(eigenvalue(rho_crit * 0.5, delta).is_finite());
        for &rho in &[f64::INFINITY, f64::NAN] {
            assert!(eigenvector(rho, delta).is_infinite());
            assert!(eigenvalue(rho, delta).is_infinite());
        }
    }
}
