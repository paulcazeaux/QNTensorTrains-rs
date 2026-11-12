// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

pub(crate) fn clamp_hi_first(raw: f64, lo: f64, hi: f64) -> f64 {
    if raw > hi {
        hi
    } else if raw < lo {
        lo
    } else {
        raw
    }
}
