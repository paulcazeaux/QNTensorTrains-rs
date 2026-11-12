// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

pub struct Counting;

#[inline(always)]
fn account_alloc(bytes: u64) {
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
    ALLOCS.fetch_add(1, Relaxed);
    ALLOC_BYTES.fetch_add(bytes, Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            account_alloc(l.size() as u64);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            account_alloc(l.size() as u64);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size() as u64, Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new_size) };
        if !q.is_null() {
            let old = l.size() as u64;
            let new = new_size as u64;
            if new >= old {
                let live = LIVE.fetch_add(new - old, Relaxed) + (new - old);
                PEAK.fetch_max(live, Relaxed);
            } else {
                LIVE.fetch_sub(old - new, Relaxed);
            }
            ALLOCS.fetch_add(1, Relaxed);
            ALLOC_BYTES.fetch_add(new, Relaxed);
        }
        q
    }
}

impl Counting {
    pub fn live() -> u64 {
        LIVE.load(Relaxed)
    }
    pub fn peak() -> u64 {
        PEAK.load(Relaxed)
    }
    pub fn allocs() -> u64 {
        ALLOCS.load(Relaxed)
    }
    pub fn alloc_bytes() -> u64 {
        ALLOC_BYTES.load(Relaxed)
    }
    pub fn reset_peak() {
        PEAK.store(LIVE.load(Relaxed), Relaxed);
    }
    pub fn reset_counts() {
        ALLOCS.store(0, Relaxed);
        ALLOC_BYTES.store(0, Relaxed);
    }
    pub fn installed() -> bool {
        let before = ALLOCS.load(Relaxed);
        let b = Box::new(0u64);
        std::hint::black_box(&b);
        ALLOCS.load(Relaxed) != before
    }
}

pub fn rss_now() -> u64 {
    #[cfg(target_os = "linux")]
    {
        proc_status_kb("VmRSS:") * 1024
    }
    #[cfg(target_os = "macos")]
    {
        let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
        let sz = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
        let n = unsafe {
            libc::proc_pidinfo(
                libc::getpid(),
                libc::PROC_PIDTASKINFO,
                0,
                (&mut info as *mut libc::proc_taskinfo).cast::<libc::c_void>(),
                sz,
            )
        };
        if n == sz { info.pti_resident_size } else { 0 }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

pub fn rss_peak() -> u64 {
    #[cfg(target_os = "linux")]
    {
        proc_status_kb("VmHWM:") * 1024
    }
    #[cfg(target_os = "macos")]
    {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
        if rc == 0 { ru.ru_maxrss as u64 } else { 0 }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

pub fn vm_size() -> u64 {
    #[cfg(target_os = "linux")]
    {
        proc_status_kb("VmSize:") * 1024
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

pub fn malloc_in_use() -> u64 {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        let mi = unsafe { libc::mallinfo2() };
        (mi.uordblks as u64) + (mi.hblkhd as u64)
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        0
    }
}

#[cfg(target_os = "linux")]
fn proc_status_kb(key: &str) -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    s.lines()
        .find_map(|l| l.strip_prefix(key))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap_or(0)
}

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub t_s: f64,
    pub rss: u64,
    pub live: u64,
}

pub struct Sampler {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<Vec<Sample>>>,
    t0: Instant,
}

impl Sampler {
    pub fn start(period: Duration) -> Sampler {
        let stop = Arc::new(AtomicBool::new(false));
        let t0 = Instant::now();
        let flag = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("mem-sampler".into())
            .spawn(move || {
                let mut out = Vec::new();
                loop {
                    out.push(Sample {
                        t_s: t0.elapsed().as_secs_f64(),
                        rss: rss_now(),
                        live: Counting::live(),
                    });
                    if flag.load(Relaxed) {
                        break;
                    }
                    std::thread::sleep(period);
                }
                out
            })
            .expect("spawn mem-sampler");
        Sampler {
            stop,
            handle: Some(handle),
            t0,
        }
    }

    pub fn elapsed_s(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }

    pub fn stop(mut self) -> Vec<Sample> {
        self.stop.store(true, Relaxed);
        self.handle
            .take()
            .map(|h| h.join().expect("mem-sampler panicked"))
            .unwrap_or_default()
    }

    pub fn window_max(samples: &[Sample], t_from: f64, t_to: f64) -> (u64, u64) {
        samples
            .iter()
            .filter(|s| s.t_s >= t_from && s.t_s <= t_to)
            .fold((0, 0), |(r, l), s| (r.max(s.rss), l.max(s.live)))
    }

    pub fn write_csv(samples: &[Sample], path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "t_s,rss_bytes,live_bytes")?;
        for s in samples {
            writeln!(f, "{:.3},{},{}", s.t_s, s.rss, s.live)?;
        }
        Ok(())
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop.store(true, Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
