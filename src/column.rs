//! Typed field storage of one segment, and its incremental builders.
//!
//! Every column has one entry per row. Missing scalar values are tracked by a
//! presence bitmap, which is omitted when every row has a value. Keyword
//! columns keep a sorted per-segment dictionary and an inverted index.
use std::collections::HashMap;

use bytemuck::Pod;
use rayon::prelude::*;

use crate::codec::{Decoder, Encoder, ToLe, corrupt};
use crate::error::Result;
use crate::rows::RowSet;
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
}

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

/// Byte strings packed end to end.
pub(crate) struct Strings {
    /// `len() + 1` offsets into `data`, starting at zero.
    offsets: Vec<u64>,
    data: Vec<u8>,
}

impl Strings {
    fn new() -> Self {
        Self {
            offsets: vec![0],
            data: Vec::new(),
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
        self.offsets.push(self.data.len() as u64);
    }

    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn get(&self, index: usize) -> &[u8] {
        &self.data[self.offsets[index] as usize..self.offsets[index + 1] as usize]
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
    /// Sorted, unique terms of this segment.
    dict: Strings,
    /// Per row, a sorted range of `ids`.
    row_offsets: Vec<u64>,
    ids: Vec<u32>,
    /// Per term, a range of `postings` holding its sorted rows.
    posting_offsets: Vec<u64>,
    postings: Vec<u32>,
}

impl Keywords {
    fn terms(&self, row: usize) -> impl Iterator<Item = &str> {
        let ids = &self.ids[self.row_offsets[row] as usize..self.row_offsets[row + 1] as usize];
        ids.iter().map(|&id| utf8(self.dict.get(id as usize)))
    }

    /// Rows holding at least one of `terms`.
    pub(crate) fn select(&self, terms: &[String]) -> RowSet {
        let universe = self.row_offsets.len() - 1;
        terms
            .iter()
            .filter_map(|term| self.dict.find(term.as_bytes()))
            .map(|id| {
                let range =
                    self.posting_offsets[id] as usize..self.posting_offsets[id + 1] as usize;
                RowSet::from_sorted(universe, self.postings[range].to_vec())
            })
            .reduce(RowSet::union)
            .unwrap_or_else(|| RowSet::empty(universe))
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
    row_offsets: Vec<u64>,
    items: Strings,
}

impl Lists {
    fn get(&self, row: usize) -> impl Iterator<Item = &[u8]> {
        (self.row_offsets[row] as usize..self.row_offsets[row + 1] as usize)
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
        row_offsets: Vec<u64>,
        ids: Vec<u32>,
    },
    Blobs {
        items: Strings,
        present: PresenceBuilder,
    },
    TextList {
        row_offsets: Vec<u64>,
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
                row_offsets: vec![0],
                ids: Vec::new(),
            },
            FieldType::Text | FieldType::Bytes => Self::Blobs {
                items: Strings::new(),
                present: PresenceBuilder::default(),
            },
            FieldType::TextList => Self::TextList {
                row_offsets: vec![0],
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
                row_offsets.push(ids.len() as u64);
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
                row_offsets.push(items.len() as u64);
            }
        }
    }

    pub(crate) fn finish(self) -> Column {
        match self {
            Self::U32(b) => Column::U32(b.finish()),
            Self::U64(b) => Column::U64(b.finish()),
            Self::I64(b) => Column::I64(b.finish()),
            Self::F32(b) => Column::F32(b.finish()),
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
    row_offsets: Vec<u64>,
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
    let mut counts = vec![0u64; terms.len() + 1];
    for range in row_offsets.windows(2) {
        let row_ids = &mut ids[range[0] as usize..range[1] as usize];
        row_ids
            .iter_mut()
            .for_each(|id| *id = renumber[*id as usize]);
        row_ids.sort_unstable();
        row_ids.iter().for_each(|&id| counts[id as usize + 1] += 1);
    }
    for index in 1..counts.len() {
        counts[index] += counts[index - 1];
    }
    let posting_offsets = counts.clone();
    let mut postings = vec![0u32; ids.len()];
    for (row, range) in row_offsets.windows(2).enumerate() {
        for &id in &ids[range[0] as usize..range[1] as usize] {
            let slot = &mut counts[id as usize];
            postings[*slot as usize] = row as u32;
            *slot += 1;
        }
    }
    Keywords {
        dict,
        row_offsets,
        ids,
        posting_offsets,
        postings,
    }
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
            if words.len() != crate::rows::word_count(rows) {
                return Err(corrupt("presence bitmap has a wrong length"));
            }
            Ok(Some(words))
        }
        _ => Err(corrupt("unknown presence flag")),
    }
}

/// `offsets` must start at zero, never decrease, and end at `end`.
fn check_offsets(offsets: &[u64], count: usize, end: usize) -> Result<()> {
    let valid = offsets.len() == count + 1
        && offsets[0] == 0
        && offsets.windows(2).all(|w| w[0] <= w[1])
        && offsets[count] == end as u64;
    if valid {
        Ok(())
    } else {
        Err(corrupt("offsets are out of order or out of bounds"))
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
        })
    }
}

impl Strings {
    fn encode(&self, e: &mut Encoder) {
        e.slice(&self.offsets);
        e.slice(&self.data);
    }

    fn decode(d: &mut Decoder<'_>, count: usize, utf8: bool) -> Result<Self> {
        let offsets = d.slice::<u64>()?;
        let data = d.slice::<u8>()?;
        check_offsets(&offsets, count, data.len())?;
        let strings = Self { offsets, data };
        if utf8 && (0..count).any(|i| std::str::from_utf8(strings.get(i)).is_err()) {
            return Err(corrupt("text is not UTF-8"));
        }
        Ok(strings)
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
            Self::Keywords(c) => {
                e.u64(c.dict.len() as u64);
                c.dict.encode(e);
                e.slice(&c.row_offsets);
                e.slice(&c.ids);
                e.slice(&c.posting_offsets);
                e.slice(&c.postings);
            }
            Self::Blobs(c) => {
                c.items.encode(e);
                encode_presence(e, &c.present);
            }
            Self::TextList(c) => {
                e.slice(&c.row_offsets);
                e.u64(c.items.len() as u64);
                c.items.encode(e);
            }
        }
    }

    /// Decode a column of `rows` rows and check its internal references.
    pub(crate) fn decode(d: &mut Decoder<'_>, ty: FieldType, rows: usize) -> Result<Self> {
        Ok(match ty {
            FieldType::U32 => Self::U32(Scalars::decode(d, rows)?),
            FieldType::U64 => Self::U64(Scalars::decode(d, rows)?),
            FieldType::I64 => Self::I64(Scalars::decode(d, rows)?),
            FieldType::F32 => Self::F32(Scalars::decode(d, rows)?),
            FieldType::Bool => {
                let bytes = Scalars::<u8>::decode(d, rows)?;
                if bytes.values.iter().any(|&b| b > 1) {
                    return Err(corrupt("bool column holds a value other than 0 or 1"));
                }
                Self::Bool(Scalars {
                    values: bytes.values.iter().map(|&b| b == 1).collect(),
                    present: bytes.present,
                })
            }
            FieldType::Keyword | FieldType::Keywords => {
                let terms = usize::try_from(d.u64()?).map_err(|_| corrupt("too many terms"))?;
                let dict = Strings::decode(d, terms, true)?;
                if (1..terms).any(|i| dict.get(i - 1) >= dict.get(i)) {
                    return Err(corrupt("keyword dictionary is not sorted"));
                }
                let row_offsets = d.slice::<u64>()?;
                let ids = d.slice::<u32>()?;
                let posting_offsets = d.slice::<u64>()?;
                let postings = d.slice::<u32>()?;
                check_offsets(&row_offsets, rows, ids.len())?;
                check_offsets(&posting_offsets, terms, postings.len())?;
                if ids.iter().any(|&id| id as usize >= terms)
                    || postings.iter().any(|&row| row as usize >= rows)
                {
                    return Err(corrupt("keyword index refers outside the segment"));
                }
                Self::Keywords(Keywords {
                    dict,
                    row_offsets,
                    ids,
                    posting_offsets,
                    postings,
                })
            }
            FieldType::Text | FieldType::Bytes => {
                let items = Strings::decode(d, rows, ty == FieldType::Text)?;
                Self::Blobs(Blobs {
                    items,
                    present: decode_presence(d, rows)?,
                })
            }
            FieldType::TextList => {
                let row_offsets = d.slice::<u64>()?;
                let count = usize::try_from(d.u64()?).map_err(|_| corrupt("too many items"))?;
                check_offsets(&row_offsets, rows, count)?;
                Self::TextList(Lists {
                    row_offsets,
                    items: Strings::decode(d, count, true)?,
                })
            }
        })
    }
}
