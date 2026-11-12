// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::{Path, PathBuf};

use qntt::Scalar;
use qntt::fcidump::FcidumpError;
use qntt::integrals::{IntegralsOptions, load_fixture_dir};

use crate::fixture::Fixture;

#[derive(Clone, Debug)]
pub struct Repo {
    root: PathBuf,
}

impl Repo {
    #[doc(hidden)]
    pub fn from_manifest_dir(manifest_dir: &'static str) -> Repo {
        let root = std::env::var_os("QNTT_REPO_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(manifest_dir));
        Repo { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn testdata(&self) -> PathBuf {
        self.root.join("testdata")
    }

    pub fn fixture_dir(&self, name_or_path: &str) -> PathBuf {
        let looks_like_path = name_or_path.contains('/')
            || name_or_path.contains(std::path::MAIN_SEPARATOR)
            || name_or_path.starts_with('.');
        if looks_like_path {
            PathBuf::from(name_or_path)
        } else {
            self.testdata().join(name_or_path)
        }
    }

    pub fn load(&self, name_or_path: &str) -> Fixture<f64> {
        self.load_with(name_or_path, &IntegralsOptions::default())
    }

    pub fn load_with(&self, name_or_path: &str, opts: &IntegralsOptions) -> Fixture<f64> {
        match self.try_load_with(name_or_path, opts) {
            Ok(f) => f,
            Err(e) => panic!("{name_or_path}: {e}"),
        }
    }

    pub fn try_load(&self, name_or_path: &str) -> Result<Fixture<f64>, FcidumpError> {
        self.try_load_with(name_or_path, &IntegralsOptions::default())
    }

    pub fn try_load_with(
        &self,
        name_or_path: &str,
        opts: &IntegralsOptions,
    ) -> Result<Fixture<f64>, FcidumpError> {
        self.try_load_as::<f64>(name_or_path, opts)
    }

    pub fn try_load_as<T: Scalar>(
        &self,
        name_or_path: &str,
        opts: &IntegralsOptions,
    ) -> Result<Fixture<T>, FcidumpError> {
        let dir = self.fixture_dir(name_or_path);
        let load = load_fixture_dir::<T>(&dir, opts)?;
        Ok(Fixture {
            name: name_or_path.to_string(),
            dir,
            load,
        })
    }

    pub fn n2(&self) -> Fixture<f64> {
        self.load("n2_ccpvdz")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> Repo {
        Repo {
            root: PathBuf::from("/checkout"),
        }
    }

    #[test]
    fn t_fixture_dir_bare_name_goes_under_testdata() {
        assert_eq!(
            repo().fixture_dir("h2o_631g"),
            PathBuf::from("/checkout/testdata/h2o_631g")
        );
    }

    #[test]
    fn t_fixture_dir_takes_a_path_verbatim() {
        for p in ["../other/testdata/h2o", "./local", "/abs/path"] {
            assert_eq!(repo().fixture_dir(p), PathBuf::from(p), "{p}");
        }
    }
}
