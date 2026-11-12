// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::s1::*;

#[test]
#[should_panic(expected = "inadmissible at site")]
fn csf_step_refuses_inadmissible_step() {
    let space = Su2Space::new(4, 2, 0);
    let top = Su2Sector::new(2, 0);
    let _: Su2Core<f64> = Su2Core::csf_step(space, 0, Su2Family::Occ, top);
}
