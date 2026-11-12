// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::sync::{Mutex, MutexGuard};

static SWITCH: Mutex<()> = Mutex::new(());

pub(crate) fn lock() -> MutexGuard<'static, ()> {
    SWITCH.lock().unwrap_or_else(|e| e.into_inner())
}

mod block_fork;
mod pairgram_par;
mod probe_batch;
mod storage_pins_par;
mod tda_batch;
mod vector_algebra_par;
