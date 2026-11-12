// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::joint::cas::CasPartition;

pub(crate) const H2O_PART: CasPartition = CasPartition {
    n_core: 1,
    n_active: 6,
    n_virt: 0,
};
