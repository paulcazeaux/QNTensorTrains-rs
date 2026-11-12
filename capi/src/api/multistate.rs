// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::api::vector::borrow_many;
use crate::err::{CResult, Error, copy_out, ptr_mut, ptr_ref, ptr_slice, ptr_slice_mut};
use crate::handle::{Handle, borrow, emit, into_raw};
use crate::layer::Layer;
use crate::opts::{
    ExcitedSchedules, QnttExcitedInfo, QnttExcitedOptions, QnttExcitedRoot, QnttSaSweep,
    QnttSsSweep, QnttStateAverageOptions, QnttStateSpecificOptions,
};

unsafe fn check_handle_out<T>(
    buf: *mut *mut T,
    cap: usize,
    need: usize,
    what: &str,
) -> Result<(), Error> {
    if buf.is_null() {
        return Err(Error::null(what));
    }
    if cap < need {
        return Err(Error::invalid(format!(
            "{what}: holds {cap} handles, need {need}"
        )));
    }
    Ok(())
}

/// State-average two-site MALS: the `nroots` lowest states sharing one train.
///
/// # Safety
/// Every non-null pointer must hold its stated length, and `out_states` at least `states_cap`
/// handle slots.
#[allow(clippy::too_many_arguments)]
pub unsafe fn state_average_mals<L: Layer>(
    h: *const Handle<L::Ham>,
    x0: *const Handle<L::Vector>,
    nroots: usize,
    opts: *const QnttStateAverageOptions,
    weights: *const f64,
    weights_len: usize,
    bond_dims: *const usize,
    bond_dims_len: usize,
    noise: *const f64,
    noise_len: usize,
    out_lambdas: *mut f64,
    out_states: *mut *mut Handle<L::Vector>,
    states_cap: usize,
    out_sweeps: *mut QnttSaSweep,
    sweeps_cap: usize,
    out_nsweeps: *mut usize,
    out_sweep_energies: *mut f64,
    out_sweep_discarded: *mut f64,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x0 = unsafe { borrow(x0)? };
    let w = unsafe { ptr_slice(weights, weights_len, "weights")? };
    let bd = unsafe { ptr_slice(bond_dims, bond_dims_len, "bond_dims")? };
    let ns = unsafe { ptr_slice(noise, noise_len, "noise")? };
    let o = unsafe { ptr_ref(opts, "opts")? }.to_core(nroots, w, bd, ns)?;
    let lambdas = unsafe { ptr_slice_mut(out_lambdas, nroots, "out_lambdas")? };
    unsafe { check_handle_out(out_states, states_cap, nroots, "out_states")? };

    let (hist, xs) = L::state_average_mals(h, x0, nroots, &o);

    debug_assert_eq!(hist.lambdas.len(), nroots);
    debug_assert_eq!(xs.len(), nroots);
    lambdas.copy_from_slice(&hist.lambdas);
    unsafe {
        if !out_nsweeps.is_null() {
            *out_nsweeps = hist.sweeps.len();
        }
        let n = sweeps_cap.min(hist.sweeps.len());
        if n > 0 {
            if !out_sweeps.is_null() {
                let dst = ptr_slice_mut(out_sweeps, n, "out_sweeps")?;
                for (d, s) in dst.iter_mut().zip(hist.sweeps.iter()) {
                    *d = QnttSaSweep::from_core(s);
                }
            }
            if !out_sweep_energies.is_null() {
                let dst = ptr_slice_mut(out_sweep_energies, n * nroots, "out_sweep_energies")?;
                for (s, sweep) in hist.sweeps.iter().take(n).enumerate() {
                    debug_assert_eq!(sweep.energies.len(), nroots);
                    dst[s * nroots..(s + 1) * nroots].copy_from_slice(&sweep.energies);
                }
            }
            if !out_sweep_discarded.is_null() {
                let dst = ptr_slice_mut(out_sweep_discarded, n * nroots, "out_sweep_discarded")?;
                for (s, sweep) in hist.sweeps.iter().take(n).enumerate() {
                    debug_assert_eq!(sweep.discarded_weights.len(), nroots);
                    dst[s * nroots..(s + 1) * nroots].copy_from_slice(&sweep.discarded_weights);
                }
            }
        }
        let slots = ptr_slice_mut(out_states, nroots, "out_states")?;
        for (slot, x) in slots.iter_mut().zip(xs) {
            *slot = into_raw(x);
        }
    }
    Ok(())
}

/// State-specific two-site MALS: one state against fixed `lower` states.
///
/// # Safety
/// Every non-null pointer must hold its stated length; `lower` must hold `nlower` live handles.
#[allow(clippy::too_many_arguments)]
pub unsafe fn state_specific_mals<L: Layer>(
    h: *const Handle<L::Ham>,
    x0: *const Handle<L::Vector>,
    lower: *const *const Handle<L::Vector>,
    nlower: usize,
    opts: *const QnttStateSpecificOptions,
    bond_dims: *const usize,
    bond_dims_len: usize,
    noise: *const f64,
    noise_len: usize,
    lower_energies: *const f64,
    lower_energies_len: usize,
    out_lambda: *mut f64,
    out_x: *mut *mut Handle<L::Vector>,
    out_sweeps: *mut QnttSsSweep,
    sweeps_cap: usize,
    out_nsweeps: *mut usize,
    out_sweep_overlaps: *mut f64,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x0 = unsafe { borrow(x0)? };
    let refs = unsafe { borrow_many::<L>(lower, nlower)? };
    let bd = unsafe { ptr_slice(bond_dims, bond_dims_len, "bond_dims")? };
    let ns = unsafe { ptr_slice(noise, noise_len, "noise")? };
    let le = unsafe { ptr_slice(lower_energies, lower_energies_len, "lower_energies")? };
    let o = unsafe { ptr_ref(opts, "opts")? }.to_core(nlower, le, bd, ns)?;
    let lambda = unsafe { ptr_mut(out_lambda, "out_lambda")? };
    if out_x.is_null() {
        return Err(Error::null("out_x"));
    }
    let owned: Vec<L::Vector> = refs.into_iter().cloned().collect();

    let (hist, x) = L::state_specific_mals(h, x0, &owned, &o);

    *lambda = hist.lambda;
    unsafe {
        if !out_nsweeps.is_null() {
            *out_nsweeps = hist.sweeps.len();
        }
        let n = sweeps_cap.min(hist.sweeps.len());
        if n > 0 {
            if !out_sweeps.is_null() {
                let dst = ptr_slice_mut(out_sweeps, n, "out_sweeps")?;
                for (d, s) in dst.iter_mut().zip(hist.sweeps.iter()) {
                    *d = QnttSsSweep::from_core(s);
                }
            }
            if nlower > 0 && !out_sweep_overlaps.is_null() {
                let dst = ptr_slice_mut(out_sweep_overlaps, n * nlower, "out_sweep_overlaps")?;
                for (s, sweep) in hist.sweeps.iter().take(n).enumerate() {
                    debug_assert_eq!(sweep.overlaps.len(), nlower);
                    dst[s * nlower..(s + 1) * nlower].copy_from_slice(&sweep.overlaps);
                }
            }
        }
        emit(x, out_x)
    }
}

/// The excited-state ladder: a state average, then one state-specific sweep and one penalized
/// Riemannian polish per root, bottom-up, with convergence / orthogonality / ordering gates.
///
/// # Safety
/// Every non-null pointer must hold its stated length; `lower` must hold `nlower` live handles.
#[allow(clippy::too_many_arguments)]
pub unsafe fn excited_states<L: Layer>(
    h: *const Handle<L::Ham>,
    x0: *const Handle<L::Vector>,
    lower: *const *const Handle<L::Vector>,
    nlower: usize,
    opts: *const QnttExcitedOptions,
    weights: *const f64,
    weights_len: usize,
    bond_dims: *const usize,
    bond_dims_len: usize,
    noise: *const f64,
    noise_len: usize,
    ss_noise: *const f64,
    ss_noise_len: usize,
    out_energies: *mut f64,
    out_roots: *mut *mut Handle<L::Vector>,
    roots_cap: usize,
    out_info: *mut QnttExcitedInfo,
    out_root_info: *mut QnttExcitedRoot,
    out_lower_gradnorms: *mut f64,
    out_gram: *mut f64,
    gram_cap: usize,
    out_gram_len: *mut usize,
    out_sa_states: *mut *mut Handle<L::Vector>,
    sa_states_cap: usize,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x0 = unsafe { borrow(x0)? };
    let refs = unsafe { borrow_many::<L>(lower, nlower)? };
    let o = unsafe { ptr_ref(opts, "opts")? };
    let sched = ExcitedSchedules {
        weights: unsafe { ptr_slice(weights, weights_len, "weights")? },
        bond_dims: unsafe { ptr_slice(bond_dims, bond_dims_len, "bond_dims")? },
        noise: unsafe { ptr_slice(noise, noise_len, "noise")? },
        ss_noise: unsafe { ptr_slice(ss_noise, ss_noise_len, "ss_noise")? },
    };
    let nroots = o.nroots;
    let energies = unsafe { ptr_slice_mut(out_energies, nroots, "out_energies")? };
    unsafe { check_handle_out(out_roots, roots_cap, nroots, "out_roots")? };
    if o.keep_sa != 0 {
        unsafe { check_handle_out(out_sa_states, sa_states_cap, nroots, "out_sa_states")? };
    }
    let owned: Vec<L::Vector> = refs.into_iter().cloned().collect();

    let (e, roots, info) = L::excited_states(h, x0, &owned, o, &sched)?;

    debug_assert_eq!(e.len(), nroots);
    energies.copy_from_slice(&e);
    unsafe {
        if !out_info.is_null() {
            *ptr_mut(out_info, "out_info")? = QnttExcitedInfo::from_core(&info);
        }
        if !out_root_info.is_null() {
            let dst = ptr_slice_mut(out_root_info, nroots, "out_root_info")?;
            for (i, d) in dst.iter_mut().enumerate() {
                *d = QnttExcitedRoot {
                    sa_energy: info.sa_history.lambdas[i],
                    ss_energy: info.ss_energies[i],
                    j_ss: info.j_ss[i],
                    j_pol: info.j_pol[i],
                    gradnorm: info.gradnorms[i],
                    penalty: info.penalties[i],
                    passes: info.passes[i],
                    stalled: info.stalled[i] as std::os::raw::c_int,
                    converged: info.converged[i] as std::os::raw::c_int,
                    swapped: info.swapped.contains(&(nlower + i)) as std::os::raw::c_int,
                };
            }
        }
        if !out_lower_gradnorms.is_null() && nlower > 0 {
            let dst = ptr_slice_mut(out_lower_gradnorms, nlower, "out_lower_gradnorms")?;
            dst.copy_from_slice(&info.lower_gradnorms);
        }
        let m = nlower + nroots;
        let flat: Vec<f64> = info.overlaps.iter().flatten().copied().collect();
        debug_assert_eq!(flat.len(), m * m);
        copy_out(&flat, out_gram, gram_cap, out_gram_len, "out_gram")?;

        let slots = ptr_slice_mut(out_roots, nroots, "out_roots")?;
        for (slot, x) in slots.iter_mut().zip(roots) {
            *slot = into_raw(x);
        }
        if o.keep_sa != 0 {
            let sa = info
                .sa_states
                .expect("keep_sa != 0 always fills ExcitedInfo::sa_states");
            let slots = ptr_slice_mut(out_sa_states, nroots, "out_sa_states")?;
            for (slot, x) in slots.iter_mut().zip(sa) {
                *slot = into_raw(x);
            }
        }
    }
    Ok(())
}
