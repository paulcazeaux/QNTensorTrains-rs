// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

const NIL: usize = usize::MAX;

pub(super) fn hopcroft_karp(
    nu: usize,
    nv: usize,
    adj: &[Vec<usize>],
) -> (usize, Vec<usize>, Vec<usize>) {
    let mut mu = vec![NIL; nu];
    let mut mv = vec![NIL; nv];
    let mut matching = 0usize;
    let mut dist = vec![usize::MAX; nu];
    let mut queue = std::collections::VecDeque::new();
    loop {
        queue.clear();
        for u in 0..nu {
            if mu[u] == NIL {
                dist[u] = 0;
                queue.push_back(u);
            } else {
                dist[u] = usize::MAX;
            }
        }
        let mut found = false;
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                match mv[v] {
                    NIL => found = true,
                    w if dist[w] == usize::MAX => {
                        dist[w] = dist[u] + 1;
                        queue.push_back(w);
                    }
                    _ => {}
                }
            }
        }
        if !found {
            break;
        }
        fn dfs(
            u: usize,
            adj: &[Vec<usize>],
            mu: &mut [usize],
            mv: &mut [usize],
            dist: &mut [usize],
        ) -> bool {
            for i in 0..adj[u].len() {
                let v = adj[u][i];
                let w = mv[v];
                if w == NIL || (dist[w] == dist[u] + 1 && dfs(w, adj, mu, mv, dist)) {
                    mu[u] = v;
                    mv[v] = u;
                    return true;
                }
            }
            dist[u] = usize::MAX;
            false
        }
        for u in 0..nu {
            if mu[u] == NIL && dfs(u, adj, &mut mu, &mut mv, &mut dist) {
                matching += 1;
            }
        }
    }
    (matching, mu, mv)
}

pub(super) fn koenig_cover(
    nu: usize,
    nv: usize,
    adj: &[Vec<usize>],
    mu: &[usize],
    mv: &[usize],
) -> (Vec<bool>, Vec<bool>) {
    let mut zu = vec![false; nu];
    let mut zv = vec![false; nv];
    let mut stack: Vec<usize> = (0..nu).filter(|&u| mu[u] == NIL).collect();
    for &u in &stack {
        zu[u] = true;
    }
    while let Some(u) = stack.pop() {
        for &v in &adj[u] {
            if !zv[v] {
                zv[v] = true;
                let w = mv[v];
                if w != NIL && !zu[w] {
                    zu[w] = true;
                    stack.push(w);
                }
            }
        }
    }
    let cover_u: Vec<bool> = zu.iter().map(|&z| !z).collect();
    (cover_u, zv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hk_koenig_pins() {
        let adj = vec![vec![0, 1, 2]];
        let (m, mu, mv) = hopcroft_karp(1, 3, &adj);
        assert_eq!(m, 1);
        let (cu, cv) = koenig_cover(1, 3, &adj, &mu, &mv);
        assert_eq!(cu, vec![true]);
        assert_eq!(cv, vec![false, false, false]);
        let adj = vec![vec![0, 1], vec![0, 1]];
        let (m, mu, mv) = hopcroft_karp(2, 2, &adj);
        assert_eq!(m, 2);
        let (cu, cv) = koenig_cover(2, 2, &adj, &mu, &mv);
        assert_eq!(
            cu.iter().filter(|&&c| c).count() + cv.iter().filter(|&&c| c).count(),
            2
        );
        let adj = vec![vec![0], vec![0, 1]];
        let (m, _, _) = hopcroft_karp(2, 2, &adj);
        assert_eq!(m, 2);
    }
}
