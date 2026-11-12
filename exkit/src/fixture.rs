// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::PathBuf;

use qntt::CoeffTensor;
use qntt::fcidump::PointGroupData;
use qntt::integrals::{FixtureMeta, Integrals, IntegralsLoad};
use qntt::vector::tt_state_in;
use qntt::{QnSpace, Scalar, SparseHamiltonian, TtVector};

pub struct Fixture<T: Scalar = f64> {
    pub name: String,
    pub dir: PathBuf,
    pub load: IntegralsLoad<T>,
}

impl<T: Scalar> Fixture<T> {
    pub fn d(&self) -> usize {
        self.load.ints.norb
    }

    pub fn nelec(&self) -> usize {
        self.load.ints.nelec
    }

    pub fn ms2(&self) -> i64 {
        self.load.ints.ms2
    }

    pub fn nup(&self) -> usize {
        self.load.space.nup
    }

    pub fn ndn(&self) -> usize {
        self.load.space.ndn
    }

    pub fn space(&self) -> QnSpace {
        self.load.space
    }

    pub fn pg(&self) -> &PointGroupData {
        &self.load.pg
    }

    pub fn ints(&self) -> &Integrals<T> {
        &self.load.ints
    }

    pub fn t(&self) -> &CoeffTensor<T> {
        &self.load.ints.t
    }

    pub fn v(&self) -> &CoeffTensor<T> {
        &self.load.ints.v
    }

    pub fn into_tv(self) -> (CoeffTensor<T>, CoeffTensor<T>) {
        (self.load.ints.t, self.load.ints.v)
    }

    pub fn into_ints(self) -> Integrals<T> {
        self.load.ints
    }

    pub fn e_core(&self) -> f64 {
        self.meta().e_core
    }

    pub fn e_hf(&self) -> f64 {
        self.try_e_hf()
            .unwrap_or_else(|| panic!("{}: meta.txt carries no e_hf", self.name))
    }

    pub fn try_e_hf(&self) -> Option<f64> {
        self.meta().e_hf
    }

    pub fn e_ref(&self) -> f64 {
        self.try_e_ref()
            .unwrap_or_else(|| panic!("{}: meta.txt carries no reference energy", self.name))
    }

    pub fn try_e_ref(&self) -> Option<f64> {
        self.meta().e_ref
    }

    pub fn meta(&self) -> &FixtureMeta {
        self.load
            .meta
            .as_ref()
            .unwrap_or_else(|| panic!("{}: loaded without a meta.txt", self.name))
    }

    pub fn hamiltonian(&self, eps: f64) -> SparseHamiltonian<T> {
        self.load.hamiltonian(eps)
    }

    pub fn ham(&self) -> SparseHamiltonian<T> {
        self.hamiltonian(f64::EPSILON)
    }

    pub fn hf(&self) -> TtVector<T> {
        let space = self.space();
        let occ_up: Vec<bool> = (0..space.d).map(|i| i < space.nup).collect();
        let occ_dn: Vec<bool> = (0..space.d).map(|i| i < space.ndn).collect();
        tt_state_in(space, &occ_up, &occ_dn)
    }

    pub fn require_singlet(&self) -> &Self {
        assert_eq!(
            self.ms2(),
            0,
            "{}: this driver needs a closed-shell fixture, but MS2 = {}",
            self.name,
            self.ms2()
        );
        self
    }

    pub fn require_norb_at_most(&self, max: usize) -> &Self {
        assert!(
            self.d() <= max,
            "{}: d = {} exceeds this driver's rank-uncapped limit of {max}; \
             the iterates grow until the process is killed",
            self.name,
            self.d()
        );
        self
    }
}
