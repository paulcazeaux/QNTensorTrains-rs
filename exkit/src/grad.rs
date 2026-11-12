// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Weingarten,
    Projected,
    Env,
}

impl Route {
    pub const NAMES: &'static [&'static str] = &["weingarten", "projected", "env"];

    pub fn as_str(self) -> &'static str {
        match self {
            Route::Weingarten => "weingarten",
            Route::Projected => "projected",
            Route::Env => "env",
        }
    }
}

impl fmt::Display for Route {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Route {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "weingarten" => Ok(Route::Weingarten),
            "projected" => Ok(Route::Projected),
            "env" => Ok(Route::Env),
            other => Err(format!(
                "`{other}` is not a gradient route (expected one of: {})",
                Route::NAMES.join(", ")
            )),
        }
    }
}

pub mod abelian {
    use super::Route;
    use qntt::core::tangent::TtTangent;
    use qntt::solvers::hess::{riemannian_gradient_env, weingarten_operator};
    use qntt::solvers::riemannian::tangent_bases;
    use qntt::{Scalar, SparseHamiltonian, TtVector, rayleigh_quotient};

    pub fn norm_weingarten<T: Scalar>(ham: &SparseHamiltonian<T>, y: &TtVector<T>) -> f64 {
        weingarten_operator(ham, y).gradient().norm()
    }

    pub fn norm_projected<T: Scalar>(ham: &SparseHamiltonian<T>, y: &TtVector<T>) -> f64 {
        let lam = rayleigh_quotient(ham, y);
        let hy = ham.apply(y);
        let res = &hy - &(y * lam);
        let (bl, br) = tangent_bases(y);
        TtTangent::project(&bl, &br, &res).norm()
    }

    pub fn norm_env<T: Scalar>(ham: &SparseHamiltonian<T>, y: &TtVector<T>) -> f64 {
        let (bl, br) = tangent_bases(y);
        let lambda = rayleigh_quotient(ham, y).re_s();
        riemannian_gradient_env(ham, &bl, &br, lambda).norm()
    }

    pub fn norm_via<T: Scalar>(r: Route, ham: &SparseHamiltonian<T>, y: &TtVector<T>) -> f64 {
        match r {
            Route::Weingarten => norm_weingarten(ham, y),
            Route::Projected => norm_projected(ham, y),
            Route::Env => norm_env(ham, y),
        }
    }
}

pub mod su2 {
    use super::Route;
    use qntt::Scalar;
    use qntt::su2::hamiltonian::Su2SparseHamiltonian;
    use qntt::su2::hamiltonian::rayleigh_quotient;
    use qntt::su2::hess::{riemannian_gradient_env, weingarten_operator};
    use qntt::su2::riemannian::tangent_bases;
    use qntt::su2::tangent::Su2TtTangent;
    use qntt::su2::vector::Su2TtVector;

    pub fn norm_weingarten<T: Scalar>(ham: &Su2SparseHamiltonian<T>, y: &Su2TtVector<T>) -> f64 {
        weingarten_operator(ham, y).gradient().norm()
    }

    pub fn norm_projected<T: Scalar>(ham: &Su2SparseHamiltonian<T>, y: &Su2TtVector<T>) -> f64 {
        let lam = rayleigh_quotient(ham, y);
        let hy = ham.apply(y);
        let res = &hy - &(y * lam);
        let (bl, br) = tangent_bases(y);
        Su2TtTangent::project(&bl, &br, &res).norm()
    }

    pub fn norm_env<T: Scalar>(ham: &Su2SparseHamiltonian<T>, y: &Su2TtVector<T>) -> f64 {
        let (bl, br) = tangent_bases(y);
        let lambda = rayleigh_quotient(ham, y).re_s();
        riemannian_gradient_env(ham, &bl, &br, lambda).norm()
    }

    pub fn norm_via<T: Scalar>(r: Route, ham: &Su2SparseHamiltonian<T>, y: &Su2TtVector<T>) -> f64 {
        match r {
            Route::Weingarten => norm_weingarten(ham, y),
            Route::Projected => norm_projected(ham, y),
            Route::Env => norm_env(ham, y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_route_roundtrips_through_its_name() {
        for r in [Route::Weingarten, Route::Projected, Route::Env] {
            assert_eq!(r.as_str().parse::<Route>().unwrap(), r);
        }
    }

    #[test]
    fn t_route_rejects_an_unknown_name_by_listing_the_set() {
        let e = "davidson".parse::<Route>().unwrap_err();
        assert!(e.contains("weingarten"), "{e}");
    }
}
