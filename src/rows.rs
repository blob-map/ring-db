//! Adaptive sets of rows within one segment.
//!
//! Filters and deletions evaluate to a [`RowSet`]. Its representation is chosen
//! by density so that kernels consume it without conversion (see [`Mask`]).
use std::cmp::Ordering;

use crate::backend::Mask;

/// A set of rows in `0..universe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RowSet {
    universe: usize,
    repr: Repr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Repr {
    All,
    Empty,
    /// Sorted, unique rows.
    Sparse(Vec<u32>),
    /// One bit per row, least-significant bit first; unused tail bits are zero.
    Dense {
        words: Vec<u64>,
        len: usize,
    },
}

impl RowSet {
    pub(crate) fn all(universe: usize) -> Self {
        let repr = if universe == 0 {
            Repr::Empty
        } else {
            Repr::All
        };
        Self { universe, repr }
    }

    pub(crate) fn empty(universe: usize) -> Self {
        Self {
            universe,
            repr: Repr::Empty,
        }
    }

    /// `rows` must be sorted, unique, and below `universe`.
    pub(crate) fn from_sorted(universe: usize, rows: Vec<u32>) -> Self {
        debug_assert!(rows.windows(2).all(|w| w[0] < w[1]));
        debug_assert!(rows.last().is_none_or(|&row| (row as usize) < universe));
        if rows.is_empty() {
            return Self::empty(universe);
        }
        if rows.len() == universe {
            return Self::all(universe);
        }
        if prefer_dense(universe, rows.len()) {
            let mut words = vec![0; word_count(universe)];
            for &row in &rows {
                words[row as usize / 64] |= 1 << (row % 64);
            }
            return Self::from_words(universe, words);
        }
        Self {
            universe,
            repr: Repr::Sparse(rows),
        }
    }

    /// `words` must have one bit per row with zero tail bits.
    pub(crate) fn from_words(universe: usize, words: Vec<u64>) -> Self {
        debug_assert_eq!(words.len(), word_count(universe));
        let len = words.iter().map(|word| word.count_ones() as usize).sum();
        if len == 0 {
            return Self::empty(universe);
        }
        if len == universe {
            return Self::all(universe);
        }
        if !prefer_dense(universe, len) {
            return Self::from_sorted(universe, sparse_from_words(&words));
        }
        Self {
            universe,
            repr: Repr::Dense { words, len },
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.repr {
            Repr::All => self.universe,
            Repr::Empty => 0,
            Repr::Sparse(rows) => rows.len(),
            Repr::Dense { len, .. } => *len,
        }
    }

    #[cfg(test)]
    fn is_all(&self) -> bool {
        matches!(self.repr, Repr::All)
    }

    #[cfg(test)]
    fn contains(&self, row: u32) -> bool {
        match &self.repr {
            Repr::All => (row as usize) < self.universe,
            Repr::Empty => false,
            Repr::Sparse(rows) => rows.binary_search(&row).is_ok(),
            Repr::Dense { words, .. } => words[row as usize / 64] & (1 << (row % 64)) != 0,
        }
    }

    /// The kernel view of this set, or `None` when it is empty.
    pub(crate) fn mask(&self) -> Option<Mask<'_>> {
        match &self.repr {
            Repr::All => Some(Mask::All),
            Repr::Empty => None,
            Repr::Sparse(rows) => Some(Mask::Sparse(rows)),
            Repr::Dense { words, .. } => Some(Mask::Dense(words)),
        }
    }

    /// Rows in both sets. Only `self` is consumed, so `other` can be shared.
    pub(crate) fn intersection(self, other: &Self) -> Self {
        debug_assert_eq!(self.universe, other.universe);
        let universe = self.universe;
        let bit = |words: &[u64], row: u32| words[row as usize / 64] & (1 << (row % 64)) != 0;
        match (self.repr, &other.repr) {
            (Repr::Empty, _) | (_, Repr::Empty) => Self::empty(universe),
            (repr, Repr::All) => Self { universe, repr },
            (Repr::All, _) => other.clone(),
            (Repr::Sparse(left), Repr::Sparse(right)) => {
                Self::from_sorted(universe, merge(&left, right, true))
            }
            (Repr::Sparse(mut rows), Repr::Dense { words, .. }) => {
                rows.retain(|&row| bit(words, row));
                Self::from_sorted(universe, rows)
            }
            (Repr::Dense { words, .. }, Repr::Sparse(rows)) => {
                let kept = rows
                    .iter()
                    .copied()
                    .filter(|&row| bit(&words, row))
                    .collect();
                Self::from_sorted(universe, kept)
            }
            (Repr::Dense { mut words, .. }, Repr::Dense { words: right, .. }) => {
                words.iter_mut().zip(right).for_each(|(a, b)| *a &= b);
                Self::from_words(universe, words)
            }
        }
    }

    pub(crate) fn union(self, other: Self) -> Self {
        debug_assert_eq!(self.universe, other.universe);
        let universe = self.universe;
        match (self.repr, other.repr) {
            (Repr::All, _) | (_, Repr::All) => Self::all(universe),
            (Repr::Empty, repr) | (repr, Repr::Empty) => Self { universe, repr },
            (Repr::Sparse(left), Repr::Sparse(right)) => {
                Self::from_sorted(universe, merge(&left, &right, false))
            }
            (Repr::Sparse(rows), Repr::Dense { mut words, .. })
            | (Repr::Dense { mut words, .. }, Repr::Sparse(rows)) => {
                for row in rows {
                    words[row as usize / 64] |= 1 << (row % 64);
                }
                Self::from_words(universe, words)
            }
            (
                Repr::Dense {
                    words: mut left, ..
                },
                Repr::Dense { words: right, .. },
            ) => {
                left.iter_mut().zip(&right).for_each(|(a, b)| *a |= b);
                Self::from_words(universe, left)
            }
        }
    }

    pub(crate) fn complement(self) -> Self {
        let universe = self.universe;
        match self.repr {
            Repr::All => Self::empty(universe),
            Repr::Empty => Self::all(universe),
            Repr::Sparse(rows) => {
                let mut words = full_words(universe);
                for row in rows {
                    words[row as usize / 64] &= !(1 << (row % 64));
                }
                Self::from_words(universe, words)
            }
            Repr::Dense { mut words, .. } => {
                words.iter_mut().for_each(|word| *word = !*word);
                clear_tail(&mut words, universe);
                Self::from_words(universe, words)
            }
        }
    }
}

pub(crate) fn word_count(universe: usize) -> usize {
    universe.div_ceil(64)
}

pub(crate) fn prefer_dense(universe: usize, len: usize) -> bool {
    // One bitmap word occupies the same space as two sparse rows.
    len > word_count(universe).saturating_mul(2)
}

fn full_words(universe: usize) -> Vec<u64> {
    let mut words = vec![u64::MAX; word_count(universe)];
    clear_tail(&mut words, universe);
    words
}

pub(crate) fn clear_tail(words: &mut [u64], universe: usize) {
    let used = universe % 64;
    if let (Some(last), true) = (words.last_mut(), used != 0) {
        *last &= (1 << used) - 1;
    }
}

fn sparse_from_words(words: &[u64]) -> Vec<u32> {
    let mut rows = Vec::new();
    for (index, &word) in words.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            rows.push((index * 64) as u32 + remaining.trailing_zeros());
            remaining &= remaining - 1;
        }
    }
    rows
}

/// Sorted intersection (`keep_common_only`) or union of two sorted lists.
fn merge(left: &[u32], right: &[u32], keep_common_only: bool) -> Vec<u32> {
    let mut out = Vec::with_capacity(if keep_common_only {
        left.len().min(right.len())
    } else {
        left.len() + right.len()
    });
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            Ordering::Less => {
                if !keep_common_only {
                    out.push(left[i]);
                }
                i += 1;
            }
            Ordering::Greater => {
                if !keep_common_only {
                    out.push(right[j]);
                }
                j += 1;
            }
            Ordering::Equal => {
                out.push(left[i]);
                i += 1;
                j += 1;
            }
        }
    }
    if !keep_common_only {
        out.extend_from_slice(&left[i..]);
        out.extend_from_slice(&right[j..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(set: &RowSet) -> Vec<u32> {
        (0..set.universe as u32)
            .filter(|&row| set.contains(row))
            .collect()
    }

    #[test]
    fn representations_adapt_to_density() {
        let n = 1000;
        assert!(matches!(RowSet::all(0).repr, Repr::Empty));
        assert!(matches!(
            RowSet::from_sorted(n, vec![1, 5]).repr,
            Repr::Sparse(_)
        ));
        let dense = RowSet::from_sorted(n, (0..n as u32).step_by(2).collect());
        assert!(matches!(dense.repr, Repr::Dense { .. }));
        assert_eq!(dense.len(), 500);
        assert!(RowSet::from_sorted(n, (0..n as u32).collect()).is_all());
        // A dense set that thins out becomes sparse.
        let thin = dense.intersection(&RowSet::from_sorted(n, vec![0, 1, 2]));
        assert!(matches!(thin.repr, Repr::Sparse(_)));
        assert_eq!(rows(&thin), [0, 2]);
    }

    #[test]
    fn boolean_operations_match_their_definitions() {
        let n = 200;
        let a: Vec<u32> = (0..n as u32).filter(|i| i % 3 == 0).collect();
        let b: Vec<u32> = vec![0, 1, 3, 64, 199];
        let sets = || {
            [
                RowSet::from_sorted(n, a.clone()),
                RowSet::from_sorted(n, b.clone()),
                RowSet::all(n),
                RowSet::empty(n),
            ]
        };
        let members = |set: &RowSet| rows(set);
        for (i, set) in sets().into_iter().enumerate() {
            for (j, right) in sets().into_iter().enumerate() {
                let (l, r) = (members(&set), members(&right));
                let and: Vec<u32> = l.iter().copied().filter(|x| r.contains(x)).collect();
                let mut or: Vec<u32> = l.iter().chain(&r).copied().collect();
                or.sort_unstable();
                or.dedup();
                assert_eq!(rows(&set.clone().intersection(&right)), and, "{i}&{j}");
                assert_eq!(rows(&set.clone().union(right)), or, "{i}|{j}");
            }
            let expected: Vec<u32> = (0..n as u32).filter(|x| !set.contains(*x)).collect();
            assert_eq!(rows(&set.complement()), expected);
        }
    }
}
