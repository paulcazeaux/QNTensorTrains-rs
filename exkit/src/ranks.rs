// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

pub mod abelian {
    use qntt::qn::{RankGrid, Sector};
    use qntt::{QnSpace, Scalar, TtVector};

    pub fn uniform(space: QnSpace, r: usize) -> Vec<RankGrid> {
        let d = space.d;
        (0..=d)
            .map(|b| {
                let mut g = RankGrid::zeros(&space);
                for s in space.bond_sectors(b) {
                    g[s] = if b == 0 || b == d {
                        1
                    } else {
                        r.min(space.max_rank(b, s) as usize)
                    };
                }
                g
            })
            .collect()
    }

    pub fn per_bond_max<T: Scalar>(space: QnSpace, x: &TtVector<T>) -> Vec<usize> {
        (0..=space.d)
            .map(|b| {
                space
                    .bond_sectors(b)
                    .into_iter()
                    .map(|s: Sector| x.rank(b, s))
                    .max()
                    .unwrap_or(0)
            })
            .collect()
    }

    pub fn per_bond_sum<T: Scalar>(space: QnSpace, x: &TtVector<T>) -> Vec<usize> {
        (0..=space.d)
            .map(|b| {
                space
                    .bond_sectors(b)
                    .into_iter()
                    .map(|s: Sector| x.rank(b, s))
                    .sum()
            })
            .collect()
    }
}

pub mod su2 {
    use qntt::Scalar;
    use qntt::su2::qn::Su2Space;
    use qntt::su2::qn::{Su2RankGrid, max_ranks};
    use qntt::su2::vector::Su2TtVector;

    pub fn uniform(space: &Su2Space, r: usize) -> Vec<Su2RankGrid> {
        let maxg = max_ranks(space);
        let d = space.d;
        (0..=d)
            .map(|b| {
                let mut g = Su2RankGrid::zeros(space);
                for s in space.bond_sectors(b) {
                    g[s] = if b == 0 || b == d {
                        1
                    } else {
                        r.min(maxg[b][s])
                    };
                }
                g
            })
            .collect()
    }

    pub fn per_bond_max<T: Scalar>(space: &Su2Space, y: &Su2TtVector<T>) -> Vec<usize> {
        (0..=space.d)
            .map(|b| {
                space
                    .bond_sectors(b)
                    .into_iter()
                    .map(|s| y.rank(b, s))
                    .max()
                    .unwrap_or(0)
            })
            .collect()
    }

    pub fn per_bond_sum<T: Scalar>(space: &Su2Space, y: &Su2TtVector<T>) -> Vec<usize> {
        (0..=space.d)
            .map(|b| {
                space
                    .bond_sectors(b)
                    .into_iter()
                    .map(|s| y.rank(b, s))
                    .sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use qntt::QnSpace;
    use qntt::qn::max_ranks;

    #[test]
    fn t_max_rank_agrees_with_max_ranks_on_every_bond_and_sector() {
        for (d, nup, ndn) in [(4, 2, 2), (6, 3, 3), (6, 3, 1), (8, 4, 4), (10, 5, 5)] {
            let space = QnSpace::new(d, nup, ndn);
            let table = max_ranks(&space);
            for (b, row) in table.iter().enumerate() {
                for s in space.bond_sectors(b) {
                    assert_eq!(
                        space.max_rank(b, s) as usize,
                        row[s],
                        "d={d} nup={nup} ndn={ndn} bond={b} sector={s:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn t_uniform_caps_the_interior_and_pins_the_boundaries() {
        let space = QnSpace::new(6, 3, 3);
        let g = super::abelian::uniform(space, 1000);
        for s in space.bond_sectors(0) {
            assert_eq!(g[0][s], 1);
        }
        for s in space.bond_sectors(6) {
            assert_eq!(g[6][s], 1);
        }
        for (b, gb) in g.iter().enumerate().take(6).skip(1) {
            for s in space.bond_sectors(b) {
                assert_eq!(gb[s], space.max_rank(b, s) as usize, "bond {b}");
            }
        }
    }
}
