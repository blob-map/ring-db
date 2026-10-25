//! Typed field storage of one segment, and its incremental builders.
//!
//! Every column has one entry per row. Missing scalar values are tracked by a
//! presence bitmap, which is omitted when every row has a value. Numeric
//! columns keep their rows sorted by value for range filters. Keyword columns
//! keep a sorted per-segment dictionary, the term ids of each row in 1, 2 or 4
//! bytes, and an inverted index whose frequent terms are bitmaps.
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::{Bound, Range, RangeBounds};

use bytemuck::Pod;
use rayon::prelude::*;

use crate::codec::{Decoder, Encoder, ToLe, corrupt};
use crate::error::Result;
use crate::rows::{RowSet, prefer_dense, word_count};
use crate::schema::FieldType;
use crate::value::Value;

pub(crate) enum Column {
    U32(Scalars<u32>),
    U64(Scalars<u64>),
    I64(Scalars<i64>),
    F32(Scalars<f32>),
    Bool(Scalars<bool>),
    /// `Keyword` and `Keywords` fields.
    Keywords(Keywords),
    /// `Text` and `Bytes` fields.
    Blobs(Blobs),
    TextList(Lists),
}

impl Column {
    /// The value of `row`, as the variant of `ty`.
    pub(crate) fn value(&self, row: usize, ty: FieldType) -> Option<Value> {
        match self {
            Self::U32(c) => c.get(row).map(Value::U32),
            Self::U64(c) => c.get(row).map(Value::U64),
            Self::I64(c) => c.get(row).map(Value::I64),
            Self::F32(c) => c.get(row).map(Value::F32),
            Self::Bool(c) => c.get(row).map(Value::Bool),
            Self::Keywords(c) => {
                let mut terms = c.terms(row).map(str::to_string);
                match ty {
                    FieldType::Keyword => terms.next().map(Value::Text),
                    _ => Some(terms.collect::<Vec<_>>())
                        .filter(|terms| !terms.is_empty())
                        .map(Value::TextList),
                }
            }
            Self::Blobs(c) => {
                let bytes = c.get(row)?;
                Some(match ty {
                    FieldType::Text => Value::Text(utf8(bytes).to_string()),
                    _ => Value::Bytes(bytes.to_vec()),
                })
            }
            Self::TextList(c) => {
                let items: Vec<String> = c.get(row).map(|s| utf8(s).to_string()).collect();
                (!items.is_empty()).then_some(Value::TextList(items))
            }
        }
    }
}

fn utf8(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).expect("text columns hold validated UTF-8")
}

/// Bits of rows that have a value; `None` when every row has one.
type Presence = Option<Vec<u64>>;

fn present(presence: &Presence, row: usize) -> bool {
    presence
        .as_ref()
        .is_none_or(|words| words[row / 64] & (1 << (row % 64)) != 0)
}

pub(crate) struct Scalars<T> {
    values: Vec<T>,
    present: Presence,
    /// Rows with a value, sorted by value; empty for bool columns. Built, not
    /// stored, so segment files do not change.
    order: Vec<u32>,
}

/// A range filter uses `order` while it matches at most one row in this many:
/// setting one bit per match costs about as much as testing 32 values in a
/// parallel scan.
const RANGE_SCAN_RATIO: usize = 32;

impl<T: Copy + Send + Sync> Scalars<T> {
    fn get(&self, row: usize) -> Option<T> {
        present(&self.present, row).then(|| self.values[row])
    }

    /// Rows with a value for which `pred` holds.
    pub(crate) fn select(&self, pred: impl Fn(T) -> bool + Sync) -> RowSet {
        let mut words: Vec<u64> = self
            .values
            .par_chunks(64)
            .with_min_len(64)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0, |word, (bit, &value)| {
                    word | (u64::from(pred(value)) << bit)
                })
            })
            .collect();
        if let Some(present) = &self.present {
            words
                .iter_mut()
                .zip(present)
                .for_each(|(word, p)| *word &= p);
        }
        RowSet::from_words(self.values.len(), words)
    }
}

impl<T: Copy + PartialOrd + Send + Sync> Scalars<T> {
    /// Build `order`. Values must not be NaN.
    fn indexed(mut self) -> Self {
        let mut pairs: Vec<(T, u32)> = (0..self.values.len())
            .filter(|&row| present(&self.present, row))
            .map(|row| (self.values[row], row as u32))
            .collect();
        pairs.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        self.order = pairs.into_iter().map(|(_, row)| row).collect();
        self
    }

    /// Rows with a value in `low..high`.
    pub(crate) fn select_range(&self, low: Bound<T>, high: Bound<T>) -> RowSet {
        let n = self.values.len();
        let below = |v: T| match low {
            Bound::Included(l) => v < l,
            Bound::Excluded(l) => v <= l,
            Bound::Unbounded => false,
        };
        let above = |v: T| match high {
            Bound::Included(h) => v > h,
            Bound::Excluded(h) => v >= h,
            Bound::Unbounded => false,
        };
        let value = |&row: &u32| self.values[row as usize];
        let start = self.order.partition_point(|row| below(value(row)));
        let len = self.order[start..].partition_point(|row| !above(value(row)));
        let rows = &self.order[start..start + len];
        if len * RANGE_SCAN_RATIO > n {
            return self.select(|v| (low, high).contains(&v));
        }
        if len * 64 < n {
            // Fewer rows than bitmap words: sorting them is cheaper.
            let mut rows = rows.to_vec();
            rows.sort_unstable();
            return RowSet::from_sorted(n, rows);
        }
        let mut words = vec![0; word_count(n)];
        for &row in rows {
            words[row as usize / 64] |= 1 << (row % 64);
        }
        RowSet::from_words(n, words)
    }
}

/// Non-decreasing positions, held as `u32` while they fit.
pub(crate) enum Offsets {
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl Offsets {
    /// The offsets of no range: a single zero.
    fn new() -> Self {
        Self::U32(vec![0])
    }

    fn push(&mut self, offset: usize) {
        match self {
            Self::U32(offsets) => match u32::try_from(offset) {
                Ok(offset) => offsets.push(offset),
                Err(_) => {
                    let mut wide: Vec<u64> = offsets.iter().map(|&o| u64::from(o)).collect();
                    wide.push(offset as u64);
                    *self = Self::U64(wide);
                }
            },
            Self::U64(offsets) => offsets.push(offset as u64),
        }
    }

    /// Number of ranges, one less than the number of offsets.
    fn count(&self) -> usize {
        match self {
            Self::U32(offsets) => offsets.len() - 1,
            Self::U64(offsets) => offsets.len() - 1,
        }
    }

    fn get(&self, index: usize) -> usize {
        match self {
            Self::U32(offsets) => offsets[index] as usize,
            Self::U64(offsets) => offsets[index] as usize,
        }
    }

    fn range(&self, index: usize) -> Range<usize> {
        self.get(index)..self.get(index + 1)
    }
}

/// Unsigned integers in the narrowest of 1, 2 and 4 bytes that holds them.
pub(crate) enum Ids {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl Ids {
    /// `values`, none of them above `max`.
    fn new(values: Vec<u32>, max: u32) -> Self {
        if u8::try_from(max).is_ok() {
            Self::U8(values.iter().map(|&v| v as u8).collect())
        } else if u16::try_from(max).is_ok() {
            Self::U16(values.iter().map(|&v| v as u16).collect())
        } else {
            Self::U32(values)
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::U8(ids) => ids.len(),
            Self::U16(ids) => ids.len(),
            Self::U32(ids) => ids.len(),
        }
    }

    fn get(&self, index: usize) -> u32 {
        match self {
            Self::U8(ids) => u32::from(ids[index]),
            Self::U16(ids) => u32::from(ids[index]),
            Self::U32(ids) => ids[index],
        }
    }
}

/// Byte strings packed end to end.
pub(crate) struct Strings {
    /// `len() + 1` offsets into `data`, starting at zero.
    offsets: Offsets,
    data: Vec<u8>,
}

impl Strings {
    fn new() -> Self {
        Self {
            offsets: Offsets::new(),
            data: Vec::new(),
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
        self.offsets.push(self.data.len());
    }

    fn len(&self) -> usize {
        self.offsets.count()
    }

    fn get(&self, index: usize) -> &[u8] {
        &self.data[self.offsets.range(index)]
    }

    /// Position of `item` in a sorted list.
    fn find(&self, item: &[u8]) -> Option<usize> {
        let (mut low, mut high) = (0, self.len());
        while low < high {
            let mid = (low + high) / 2;
            match self.get(mid).cmp(item) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }
}

pub(crate) struct Keywords {
    rows: usize,
    /// Sorted, unique terms of this segment.
    dict: Strings,
    /// Term ids. With `offsets`, a sorted range of them per row. Without, one
    /// per row: zero for no term, and otherwise the term id plus one.
    ids: Ids,
    offsets: Option<Offsets>,
    postings: Postings,
}

impl Keywords {
    /// Index the sorted, unique term ids of each row, which refer to `dict`.
    fn new(dict: Strings, offsets: Offsets, ids: Vec<u32>) -> Self {
        let rows = offsets.count();
        let terms = dict.len() as u32;
        let postings = Postings::new(rows, dict.len(), &offsets, &ids);
        let (ids, offsets) = if (0..rows).all(|row| offsets.range(row).len() <= 1) {
            let single = (0..rows)
                .map(|row| offsets.range(row).next().map_or(0, |index| ids[index] + 1))
                .collect();
            (Ids::new(single, terms), None)
        } else {
            (Ids::new(ids, terms.saturating_sub(1)), Some(offsets))
        };
        Self {
            rows,
            dict,
            ids,
            offsets,
            postings,
        }
    }

    fn term_ids(&self, row: usize) -> impl Iterator<Item = u32> {
        let (range, shift) = match &self.offsets {
            Some(offsets) => (offsets.range(row), 0),
            None => (row..row + usize::from(self.ids.get(row) != 0), 1),
        };
        range.map(move |index| self.ids.get(index) - shift)
    }

    fn terms(&self, row: usize) -> impl Iterator<Item = &str> {
        self.term_ids(row)
            .map(|id| utf8(self.dict.get(id as usize)))
    }

    /// Rows holding at least one of `terms`.
    pub(crate) fn select(&self, terms: &[String]) -> RowSet {
        let universe = self.rows;
        let postings: Vec<Posting<'_>> = terms
            .iter()
            .filter_map(|term| self.dict.find(term.as_bytes()))
            .map(|id| self.postings.get(id, universe))
            .collect();
        match postings[..] {
            [] => RowSet::empty(universe),
            [Posting::Sparse(rows)] => RowSet::from_sorted(universe, rows.to_vec()),
            [Posting::Dense(words)] => RowSet::from_words(universe, words.to_vec()),
            _ => {
                let sparse: usize = postings
                    .iter()
                    .map(|posting| match posting {
                        Posting::Sparse(rows) => rows.len(),
                        Posting::Dense(_) => 0,
                    })
                    .sum();
                let any_dense = postings.iter().any(|p| matches!(p, Posting::Dense(_)));
                if any_dense || prefer_dense(universe, sparse) {
                    let mut words = vec![0; word_count(universe)];
                    for posting in postings {
                        match posting {
                            Posting::Sparse(rows) => rows
                                .iter()
                                .for_each(|&row| words[row as usize / 64] |= 1 << (row % 64)),
                            Posting::Dense(bits) => {
                                words.iter_mut().zip(bits).for_each(|(a, b)| *a |= b);
                            }
                        }
                    }
                    RowSet::from_words(universe, words)
                } else {
                    let mut rows: Vec<u32> = postings
                        .iter()
                        .flat_map(|posting| match posting {
                            Posting::Sparse(rows) => rows.iter().copied(),
                            Posting::Dense(_) => [].iter().copied(),
                        })
                        .collect();
                    rows.sort_unstable();
                    rows.dedup();
                    RowSet::from_sorted(universe, rows)
                }
            }
        }
    }
}

/// The sorted rows of each term: a list, or a bitmap for a term in more than
/// one row in 32, where a bitmap is smaller.
struct Postings {
    /// Per term, a range of `rows`; empty for the terms in `dense`.
    offsets: Offsets,
    rows: Vec<u32>,
    /// Sorted ids of the terms held as bitmaps, with `word_count(rows)` words
    /// each in `words`.
    dense: Vec<u32>,
    words: Vec<u64>,
}

#[derive(Clone, Copy)]
enum Posting<'a> {
    Sparse(&'a [u32]),
    Dense(&'a [u64]),
}

impl Postings {
    fn new(rows: usize, terms: usize, offsets: &Offsets, ids: &[u32]) -> Self {
        let mut counts = vec![0usize; terms];
        ids.iter().for_each(|&id| counts[id as usize] += 1);
        let width = word_count(rows);
        // Per term, its next position in the list, or the index of its bitmap.
        let mut slots = Vec::with_capacity(terms);
        let mut dense = Vec::new();
        let mut posting_offsets = Offsets::new();
        let mut listed = 0;
        for (id, &count) in counts.iter().enumerate() {
            if prefer_dense(rows, count) {
                slots.push(Err(dense.len()));
                dense.push(id as u32);
            } else {
                slots.push(Ok(listed));
                listed += count;
            }
            posting_offsets.push(listed);
        }
        let mut list = vec![0u32; listed];
        let mut words = vec![0u64; dense.len() * width];
        for row in 0..rows {
            for &id in &ids[offsets.range(row)] {
                match &mut slots[id as usize] {
                    Ok(next) => {
                        list[*next] = row as u32;
                        *next += 1;
                    }
                    Err(bitmap) => words[*bitmap * width + row / 64] |= 1 << (row % 64),
                }
            }
        }
        Self {
            offsets: posting_offsets,
            rows: list,
            dense,
            words,
        }
    }

    fn get(&self, id: usize, rows: usize) -> Posting<'_> {
        match self.dense.binary_search(&(id as u32)) {
            Ok(bitmap) => {
                let width = word_count(rows);
                Posting::Dense(&self.words[bitmap * width..(bitmap + 1) * width])
            }
            Err(_) => Posting::Sparse(&self.rows[self.offsets.range(id)]),
        }
    }
}

pub(crate) struct Blobs {
    items: Strings,
    present: Presence,
}

impl Blobs {
    fn get(&self, row: usize) -> Option<&[u8]> {
        present(&self.present, row).then(|| self.items.get(row))
    }
}

pub(crate) struct Lists {
    /// Per row, a range of `items`.
    row_offsets: Offsets,
    items: Strings,
}

impl Lists {
    fn get(&self, row: usize) -> impl Iterator<Item = &[u8]> {
        self.row_offsets
            .range(row)
            .map(|index| self.items.get(index))
    }
}

/// Appends one value per row, then freezes into a [`Column`].
pub(crate) enum ColumnBuilder {
    U32(ScalarsBuilder<u32>),
    U64(ScalarsBuilder<u64>),
    I64(ScalarsBuilder<i64>),
    F32(ScalarsBuilder<f32>),
    Bool(ScalarsBuilder<bool>),
    Keywords {
        local: HashMap<String, u32>,
        row_offsets: Offsets,
        ids: Vec<u32>,
    },
    Blobs {
        items: Strings,
        present: PresenceBuilder,
    },
    TextList {
        row_offsets: Offsets,
        items: Strings,
    },
}

impl ColumnBuilder {
    pub(crate) fn new(ty: FieldType) -> Self {
        match ty {
            FieldType::U32 => Self::U32(ScalarsBuilder::default()),
            FieldType::U64 => Self::U64(ScalarsBuilder::default()),
            FieldType::I64 => Self::I64(ScalarsBuilder::default()),
            FieldType::F32 => Self::F32(ScalarsBuilder::default()),
            FieldType::Bool => Self::Bool(ScalarsBuilder::default()),
            FieldType::Keyword | FieldType::Keywords => Self::Keywords {
                local: HashMap::new(),
                row_offsets: Offsets::new(),
                ids: Vec::new(),
            },
            FieldType::Text | FieldType::Bytes => Self::Blobs {
                items: Strings::new(),
                present: PresenceBuilder::default(),
            },
            FieldType::TextList => Self::TextList {
                row_offsets: Offsets::new(),
                items: Strings::new(),
            },
        }
    }

    /// Append the value of the next row. `value` already has this column's type.
    pub(crate) fn push(&mut self, value: Option<Value>) {
        match (self, value) {
            (Self::U32(b), v) => b.push(v.and_then(|v| v.as_u32())),
            (Self::U64(b), v) => b.push(v.and_then(|v| v.as_u64())),
            (Self::I64(b), v) => b.push(v.and_then(|v| v.as_i64())),
            (Self::F32(b), v) => b.push(v.and_then(|v| v.as_f32())),
            (Self::Bool(b), v) => b.push(v.and_then(|v| v.as_bool())),
            (
                Self::Keywords {
                    local,
                    row_offsets,
                    ids,
                },
                value,
            ) => {
                let terms = match value {
                    Some(Value::Text(term)) => vec![term],
                    Some(Value::TextList(terms)) => terms,
                    _ => Vec::new(),
                };
                let mut row: Vec<u32> = terms
                    .into_iter()
                    .map(|term| {
                        let next = local.len() as u32;
                        *local.entry(term).or_insert(next)
                    })
                    .collect();
                row.sort_unstable();
                row.dedup();
                ids.extend(row);
                row_offsets.push(ids.len());
            }
            (Self::Blobs { items, present }, value) => {
                let bytes = match &value {
                    Some(Value::Text(text)) => text.as_bytes(),
                    Some(Value::Bytes(bytes)) => bytes,
                    _ => &[],
                };
                present.push(value.is_some());
                items.push(bytes);
            }
            (Self::TextList { row_offsets, items }, value) => {
                if let Some(Value::TextList(list)) = value {
                    list.iter().for_each(|item| items.push(item.as_bytes()));
                }
                row_offsets.push(items.len());
            }
        }
    }

    pub(crate) fn finish(self) -> Column {
        match self {
            Self::U32(b) => Column::U32(b.finish().indexed()),
            Self::U64(b) => Column::U64(b.finish().indexed()),
            Self::I64(b) => Column::I64(b.finish().indexed()),
            Self::F32(b) => Column::F32(b.finish().indexed()),
            Self::Bool(b) => Column::Bool(b.finish()),
            Self::Keywords {
                local,
                row_offsets,
                ids,
            } => Column::Keywords(finish_keywords(local, row_offsets, ids)),
            Self::Blobs { items, present } => Column::Blobs(Blobs {
                present: present.finish(items.len()),
                items,
            }),
            Self::TextList { row_offsets, items } => Column::TextList(Lists { row_offsets, items }),
        }
    }
}

/// Sort the dictionary, renumber the ids, and build the inverted index.
fn finish_keywords(
    local: HashMap<String, u32>,
    row_offsets: Offsets,
    mut ids: Vec<u32>,
) -> Keywords {
    let mut terms: Vec<(String, u32)> = local.into_iter().collect();
    terms.sort_unstable();
    let mut renumber = vec![0u32; terms.len()];
    let mut dict = Strings::new();
    for (sorted, (term, id)) in terms.iter().enumerate() {
        renumber[*id as usize] = sorted as u32;
        dict.push(term.as_bytes());
    }
    for row in 0..row_offsets.count() {
        let row_ids = &mut ids[row_offsets.range(row)];
        row_ids
            .iter_mut()
            .for_each(|id| *id = renumber[*id as usize]);
        row_ids.sort_unstable();
    }
    Keywords::new(dict, row_offsets, ids)
}

#[derive(Default)]
pub(crate) struct PresenceBuilder {
    words: Vec<u64>,
    rows: usize,
    missing: usize,
}

impl PresenceBuilder {
    fn push(&mut self, present: bool) {
        if self.rows.is_multiple_of(64) {
            self.words.push(0);
        }
        if present {
            *self.words.last_mut().expect("a word for this row") |= 1 << (self.rows % 64);
        } else {
            self.missing += 1;
        }
        self.rows += 1;
    }

    fn finish(self, rows: usize) -> Presence {
        debug_assert_eq!(self.rows, rows);
        (self.missing > 0).then_some(self.words)
    }
}

pub(crate) struct ScalarsBuilder<T> {
    values: Vec<T>,
    present: PresenceBuilder,
}

impl<T> Default for ScalarsBuilder<T> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            present: PresenceBuilder::default(),
        }
    }
}

impl<T: Default> ScalarsBuilder<T> {
    fn push(&mut self, value: Option<T>) {
        self.present.push(value.is_some());
        self.values.push(value.unwrap_or_default());
    }

    fn finish(self) -> Scalars<T> {
        Scalars {
            present: self.present.finish(self.values.len()),
            values: self.values,
            order: Vec::new(),
        }
    }
}

fn encode_presence(e: &mut Encoder, presence: &Presence) {
    match presence {
        None => e.u8(0),
        Some(words) => {
            e.u8(1);
            e.slice(words);
        }
    }
}

fn decode_presence(d: &mut Decoder<'_>, rows: usize) -> Result<Presence> {
    match d.u8()? {
        0 => Ok(None),
        1 => {
            let words = d.slice::<u64>()?;
            if words.len() != word_count(rows) {
                return Err(corrupt("presence bitmap has a wrong length"));
            }
            Ok(Some(words))
        }
        _ => Err(corrupt("unknown presence flag")),
    }
}

/// `offsets` must start at zero, never decrease, and end at `end`.
fn valid_offsets<T: Copy + Into<u64>>(offsets: &[T], count: usize, end: usize) -> bool {
    offsets.len().checked_sub(1) == Some(count)
        && offsets[0].into() == 0
        && offsets.windows(2).all(|w| w[0].into() <= w[1].into())
        && offsets[count].into() == end as u64
}

impl Offsets {
    fn encode(&self, e: &mut Encoder) {
        match self {
            Self::U32(offsets) => {
                e.u8(4);
                e.slice(offsets);
            }
            Self::U64(offsets) => {
                e.u8(8);
                e.slice(offsets);
            }
        }
    }

    /// Read offsets, then check them with [`Offsets::checked`]. Version 1 files
    /// hold `u64` offsets only.
    fn read(d: &mut Decoder<'_>, version: u32) -> Result<Self> {
        let width = if version == 1 { 8 } else { d.u8()? };
        match width {
            4 => Ok(Self::U32(d.slice()?)),
            8 => Ok(Self::U64(d.slice()?)),
            _ => Err(corrupt("unknown offset width")),
        }
    }

    /// These offsets if they hold `count` ranges ending at `end`, as `u32`
    /// when they fit.
    fn checked(self, count: usize, end: usize) -> Result<Self> {
        let valid = match &self {
            Self::U32(offsets) => valid_offsets(offsets, count, end),
            Self::U64(offsets) => valid_offsets(offsets, count, end),
        };
        match self {
            _ if !valid => Err(corrupt("offsets are out of order or out of bounds")),
            Self::U64(offsets) if u32::try_from(end).is_ok() => {
                Ok(Self::U32(offsets.into_iter().map(|o| o as u32).collect()))
            }
            offsets => Ok(offsets),
        }
    }
}

impl Ids {
    fn encode(&self, e: &mut Encoder) {
        match self {
            Self::U8(ids) => {
                e.u8(1);
                e.slice(ids);
            }
            Self::U16(ids) => {
                e.u8(2);
                e.slice(ids);
            }
            Self::U32(ids) => {
                e.u8(4);
                e.slice(ids);
            }
        }
    }

    /// Decode ids that are all below `limit`.
    fn decode(d: &mut Decoder<'_>, limit: u64) -> Result<Self> {
        let ids = match d.u8()? {
            1 => Self::U8(d.slice()?),
            2 => Self::U16(d.slice()?),
            4 => Self::U32(d.slice()?),
            _ => return Err(corrupt("unknown id width")),
        };
        if (0..ids.len()).any(|index| u64::from(ids.get(index)) >= limit) {
            return Err(corrupt("keyword index refers outside the segment"));
        }
        Ok(ids)
    }
}

impl<T: Pod + ToLe> Scalars<T> {
    fn encode(&self, e: &mut Encoder) {
        e.slice(&self.values);
        encode_presence(e, &self.present);
    }

    fn decode(d: &mut Decoder<'_>, rows: usize) -> Result<Self> {
        let values = d.slice::<T>()?;
        if values.len() != rows {
            return Err(corrupt("column has a wrong length"));
        }
        Ok(Self {
            values,
            present: decode_presence(d, rows)?,
            order: Vec::new(),
        })
    }
}

impl Strings {
    fn encode(&self, e: &mut Encoder) {
        self.offsets.encode(e);
        e.slice(&self.data);
    }

    fn decode(d: &mut Decoder<'_>, version: u32, count: usize, utf8: bool) -> Result<Self> {
        let offsets = Offsets::read(d, version)?;
        let data = d.slice::<u8>()?;
        let strings = Self {
            offsets: offsets.checked(count, data.len())?,
            data,
        };
        if utf8 && (0..count).any(|i| std::str::from_utf8(strings.get(i)).is_err()) {
            return Err(corrupt("text is not UTF-8"));
        }
        Ok(strings)
    }
}

impl Keywords {
    fn encode(&self, e: &mut Encoder) {
        e.u64(self.dict.len() as u64);
        self.dict.encode(e);
        e.u8(u8::from(self.offsets.is_some()));
        self.ids.encode(e);
        if let Some(offsets) = &self.offsets {
            offsets.encode(e);
        }
        self.postings.encode(e);
    }

    fn decode(d: &mut Decoder<'_>, version: u32, rows: usize) -> Result<Self> {
        let terms = usize::try_from(d.u64()?).map_err(|_| corrupt("too many terms"))?;
        let dict = Strings::decode(d, version, terms, true)?;
        if (1..terms).any(|i| dict.get(i - 1) >= dict.get(i)) {
            return Err(corrupt("keyword dictionary is not sorted"));
        }
        if version == 1 {
            return Self::decode_v1(d, dict, rows);
        }
        let multi = match d.u8()? {
            0 => false,
            1 => true,
            _ => return Err(corrupt("unknown keyword layout")),
        };
        let ids = Ids::decode(d, terms as u64 + u64::from(!multi))?;
        let offsets = if multi {
            let offsets = Offsets::read(d, version)?.checked(rows, ids.len())?;
            let sorted = (0..rows).all(|row| {
                let range = offsets.range(row);
                (range.start + 1..range.end).all(|i| ids.get(i - 1) < ids.get(i))
            });
            if !sorted {
                return Err(corrupt("keyword ids of a row are not sorted"));
            }
            Some(offsets)
        } else if ids.len() == rows {
            None
        } else {
            return Err(corrupt("keyword column has a wrong length"));
        };
        Ok(Self {
            rows,
            dict,
            ids,
            offsets,
            postings: Postings::decode(d, rows, terms)?,
        })
    }

    /// The rest of a version 1 column, whose postings are rebuilt.
    fn decode_v1(d: &mut Decoder<'_>, dict: Strings, rows: usize) -> Result<Self> {
        let offsets = Offsets::read(d, 1)?;
        let ids = d.slice::<u32>()?;
        d.slice::<u64>()?;
        d.slice::<u32>()?;
        let offsets = offsets.checked(rows, ids.len())?;
        let valid = (0..rows).all(|row| {
            let row_ids = &ids[offsets.range(row)];
            row_ids.windows(2).all(|w| w[0] < w[1])
                && row_ids.last().is_none_or(|&id| (id as usize) < dict.len())
        });
        if !valid {
            return Err(corrupt("keyword index refers outside the segment"));
        }
        Ok(Self::new(dict, offsets, ids))
    }
}

impl Postings {
    fn encode(&self, e: &mut Encoder) {
        e.slice(&self.rows);
        self.offsets.encode(e);
        e.slice(&self.dense);
        e.slice(&self.words);
    }

    fn decode(d: &mut Decoder<'_>, rows: usize, terms: usize) -> Result<Self> {
        let list = d.slice::<u32>()?;
        let offsets = Offsets::read(d, 2)?.checked(terms, list.len())?;
        let dense = d.slice::<u32>()?;
        let words = d.slice::<u64>()?;
        let width = word_count(rows);
        let used = rows % 64;
        let valid = list.iter().all(|&row| (row as usize) < rows)
            && (0..terms).all(|id| list[offsets.range(id)].windows(2).all(|w| w[0] < w[1]))
            && dense.windows(2).all(|w| w[0] < w[1])
            && dense
                .iter()
                .all(|&id| (id as usize) < terms && offsets.range(id as usize).is_empty())
            && dense.len().checked_mul(width) == Some(words.len())
            && (used == 0 || words.chunks(width).all(|bits| bits[width - 1] >> used == 0));
        if !valid {
            return Err(corrupt(
                "keyword postings are out of order or out of bounds",
            ));
        }
        Ok(Self {
            offsets,
            rows: list,
            dense,
            words,
        })
    }
}

impl Column {
    pub(crate) fn encode(&self, e: &mut Encoder) {
        match self {
            Self::U32(c) => c.encode(e),
            Self::U64(c) => c.encode(e),
            Self::I64(c) => c.encode(e),
            Self::F32(c) => c.encode(e),
            Self::Bool(c) => {
                let bytes: Vec<u8> = c.values.iter().map(|&v| u8::from(v)).collect();
                e.slice(&bytes);
                encode_presence(e, &c.present);
            }
            Self::Keywords(c) => c.encode(e),
            Self::Blobs(c) => {
                c.items.encode(e);
                encode_presence(e, &c.present);
            }
            Self::TextList(c) => {
                c.row_offsets.encode(e);
                e.u64(c.items.len() as u64);
                c.items.encode(e);
            }
        }
    }

    /// Decode a column of `rows` rows, written in segment format `version`,
    /// and check its internal references.
    pub(crate) fn decode(
        d: &mut Decoder<'_>,
        ty: FieldType,
        rows: usize,
        version: u32,
    ) -> Result<Self> {
        Ok(match ty {
            FieldType::U32 => Self::U32(Scalars::decode(d, rows)?.indexed()),
            FieldType::U64 => Self::U64(Scalars::decode(d, rows)?.indexed()),
            FieldType::I64 => Self::I64(Scalars::decode(d, rows)?.indexed()),
            FieldType::F32 => {
                let column = Scalars::<f32>::decode(d, rows)?;
                if column.values.iter().any(|v| !v.is_finite()) {
                    return Err(corrupt("f32 column holds a value that is not finite"));
                }
                Self::F32(column.indexed())
            }
            FieldType::Bool => {
                let bytes = Scalars::<u8>::decode(d, rows)?;
                if bytes.values.iter().any(|&b| b > 1) {
                    return Err(corrupt("bool column holds a value other than 0 or 1"));
                }
                Self::Bool(Scalars {
                    values: bytes.values.iter().map(|&b| b == 1).collect(),
                    present: bytes.present,
                    order: Vec::new(),
                })
            }
            FieldType::Keyword | FieldType::Keywords => {
                Self::Keywords(Keywords::decode(d, version, rows)?)
            }
            FieldType::Text | FieldType::Bytes => {
                let items = Strings::decode(d, version, rows, ty == FieldType::Text)?;
                Self::Blobs(Blobs {
                    items,
                    present: decode_presence(d, rows)?,
                })
            }
            FieldType::TextList => {
                let row_offsets = Offsets::read(d, version)?;
                let count = usize::try_from(d.u64()?).map_err(|_| corrupt("too many items"))?;
                Self::TextList(Lists {
                    row_offsets: row_offsets.checked(rows, count)?,
                    items: Strings::decode(d, version, count, true)?,
                })
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected<T: Copy + PartialOrd>(values: &[Option<T>], range: (Bound<T>, Bound<T>)) -> RowSet {
        let rows = (0..values.len() as u32)
            .filter(|&row| values[row as usize].is_some_and(|v| range.contains(&v)))
            .collect();
        RowSet::from_sorted(values.len(), rows)
    }

    #[test]
    fn range_selection_matches_a_scan_on_every_path() {
        let mut x = 7u64;
        let mut next = |n: u64| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) % n
        };
        let n = 10_000;
        let values: Vec<Option<i64>> = (0..n)
            .map(|_| (next(10) != 0).then(|| next(2000) as i64 - 1000))
            .collect();
        let mut builder = ScalarsBuilder::default();
        values.iter().for_each(|&v| builder.push(v));
        let column = builder.finish().indexed();
        // Widths from a few rows (sorted rows) to most rows (scan).
        for width in [0, 1, 5, 20, 40, 60, 100, 2000] {
            for _ in 0..20 {
                let low = next(2200) as i64 - 1100;
                for range in [
                    (Bound::Included(low), Bound::Included(low + width)),
                    (Bound::Excluded(low), Bound::Excluded(low + width)),
                    (Bound::Unbounded, Bound::Excluded(low)),
                    (Bound::Included(low), Bound::Unbounded),
                ] {
                    assert_eq!(
                        column.select_range(range.0, range.1),
                        expected(&values, range)
                    );
                }
            }
        }

        let floats = [
            Some(-0.0f32),
            Some(0.0),
            None,
            Some(1.5),
            Some(-2.0),
            Some(0.0),
        ];
        let mut builder = ScalarsBuilder::default();
        floats.iter().for_each(|&v| builder.push(v));
        let column = builder.finish().indexed();
        for range in [
            (Bound::Included(0.0), Bound::Included(0.0)),
            (Bound::Excluded(-0.0), Bound::Unbounded),
            (Bound::Unbounded, Bound::Excluded(0.0)),
            (Bound::Included(-2.0), Bound::Excluded(1.5)),
        ] {
            assert_eq!(
                column.select_range(range.0, range.1),
                expected(&floats, range)
            );
        }
    }

    /// `rows` rows, each holding the terms of `per_row` draws of `term`.
    fn keyword_rows(
        rows: usize,
        per_row: u64,
        mut term: impl FnMut(usize) -> Option<String>,
    ) -> Vec<Vec<String>> {
        (0..rows)
            .map(|row| (0..per_row).filter_map(|_| term(row)).collect())
            .collect()
    }

    #[test]
    fn keyword_columns_match_a_scan_in_every_layout() {
        let mut x = 11u64;
        let mut next = move |n: u64| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) % n
        };
        // About half of the values are `t0`, so `t0` is held as a bitmap.
        let skewed = |terms: u64, missing: u64| {
            move |_: usize| match next(10) {
                n if n < missing => None,
                n if n < 5 => Some("t0".to_string()),
                _ => Some(format!("t{}", next(terms))),
            }
        };
        let cases = [
            (
                FieldType::Keyword,
                keyword_rows(1000, 1, skewed(3, 1)),
                1,
                false,
            ),
            (
                FieldType::Keyword,
                keyword_rows(5000, 1, skewed(400, 1)),
                2,
                false,
            ),
            (
                FieldType::Keyword,
                keyword_rows(70_000, 1, |row| Some(format!("r{row}"))),
                4,
                false,
            ),
            // A Keywords field with at most one term per row has one id per row.
            (
                FieldType::Keywords,
                keyword_rows(1000, 1, skewed(5, 3)),
                1,
                false,
            ),
            (
                FieldType::Keywords,
                keyword_rows(3000, 4, skewed(40, 3)),
                1,
                true,
            ),
            (
                FieldType::Keywords,
                keyword_rows(2000, 3, skewed(400, 2)),
                2,
                true,
            ),
        ];
        for (ty, values, width, multi) in cases {
            let mut builder = ColumnBuilder::new(ty);
            for terms in &values {
                builder.push(match (ty, terms.first()) {
                    (_, None) => None,
                    (FieldType::Keyword, Some(term)) => Some(Value::Text(term.clone())),
                    _ => Some(Value::TextList(terms.clone())),
                });
            }
            let built = builder.finish();
            let mut e = Encoder::default();
            built.encode(&mut e);
            let mut d = Decoder::new(e.as_bytes());
            let decoded = Column::decode(&mut d, ty, values.len(), 2).unwrap();
            assert!(d.is_empty());
            let rows: Vec<Vec<String>> = values
                .iter()
                .map(|terms| {
                    let mut terms = terms.clone();
                    terms.sort();
                    terms.dedup();
                    terms
                })
                .collect();
            let distinct = {
                let mut all: Vec<&String> = rows.iter().flatten().collect();
                all.sort();
                all.dedup();
                all
            };
            let queries: Vec<Vec<String>> = (0..40)
                .map(|i| {
                    // Many rare terms make a bitmap without a bitmap posting.
                    let n = [1, 2, 3, 200][i % 4];
                    (0..n)
                        .map(|_| match next(8) {
                            0 if n < 200 => "t0".to_string(),
                            1 => "absent".to_string(),
                            _ => distinct[next(distinct.len() as u64) as usize].clone(),
                        })
                        .collect()
                })
                .collect();
            for column in [&built, &decoded] {
                let Column::Keywords(c) = column else {
                    panic!("a keyword column")
                };
                let actual_width = match c.ids {
                    Ids::U8(_) => 1,
                    Ids::U16(_) => 2,
                    Ids::U32(_) => 4,
                };
                assert_eq!((actual_width, c.offsets.is_some()), (width, multi));
                assert_eq!(c.postings.dense.is_empty(), width == 4);
                for (row, terms) in rows.iter().enumerate() {
                    assert!(c.terms(row).eq(terms.iter().map(String::as_str)));
                }
                for query in &queries {
                    let expected = (0..rows.len() as u32)
                        .filter(|&row| rows[row as usize].iter().any(|t| query.contains(t)))
                        .collect();
                    assert_eq!(c.select(query), RowSet::from_sorted(rows.len(), expected));
                }
            }
        }
    }

    #[test]
    fn offsets_widen_past_u32_and_narrow_when_read() {
        let mut offsets = Offsets::new();
        offsets.push(5);
        offsets.push(1 << 33);
        assert!(matches!(offsets, Offsets::U64(_)));
        assert_eq!((offsets.count(), offsets.range(1)), (2, 5..1 << 33));
        assert!(matches!(offsets.checked(2, 1 << 33), Ok(Offsets::U64(_))));
        let narrow = Offsets::U64(vec![0, 3, 7]).checked(2, 7);
        assert!(matches!(narrow, Ok(Offsets::U32(o)) if o == [0, 3, 7]));
        assert!(Offsets::U64(vec![0, 8, 7]).checked(2, 7).is_err());
        assert!(Offsets::U32(vec![0, 7]).checked(2, 7).is_err());
    }
}
