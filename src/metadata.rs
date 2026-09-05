//! Integer metadata tags and candidate sets used to restrict vector queries.
//!
//! Payload records remain opaque to the query engine. At insertion time, callers
//! may associate `(key, value)` integer tags with a vector. The resulting index
//! maps each tag to a reusable [`CandidateSet`].

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use crate::error::{Result, RingDbError};

const METADATA_MAGIC: [u8; 8] = *b"RDBMETA\0";
const METADATA_VERSION: u32 = 1;
const HEADER_LEN: usize = 8 + 4 + 8 + 8;
const ENTRY_HEADER_LEN: usize = 4 + 1 + 3 + 8 + 8 + 8;

const ENCODING_ALL: u8 = 0;
const ENCODING_SPARSE: u8 = 1;
const ENCODING_DENSE: u8 = 2;

/// Application-defined metadata field identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetadataKey(u32);

impl MetadataKey {
    /// Construct a metadata key from its stable integer identifier.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the underlying integer identifier.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Application-defined metadata value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetadataValue(u64);

impl MetadataValue {
    /// Construct a metadata value from its stable integer representation.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the underlying integer representation.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// One equality-filterable metadata tag attached to a vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetadataTag {
    /// Field identifier.
    pub key: MetadataKey,
    /// Exact value indexed for the field.
    pub value: MetadataValue,
}

impl MetadataTag {
    /// Construct a tag from a key and value.
    pub const fn new(key: MetadataKey, value: MetadataValue) -> Self {
        Self { key, value }
    }
}

/// A set of vector IDs bound to a fixed universe length.
///
/// The representation is selected automatically from the set's density. Clones
/// share their immutable storage and are therefore constant-time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateSet {
    universe_len: usize,
    repr: CandidateRepr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CandidateRepr {
    All,
    Empty,
    Sparse(Arc<[u32]>),
    Dense { words: Arc<[u64]>, len: usize },
}

/// Borrowed candidate representation consumed by compute backends.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CandidateSetView<'a> {
    All,
    Empty,
    Sparse(&'a [u32]),
    Dense(&'a [u64]),
}

impl CandidateSet {
    /// A set containing every ID in a universe of `universe_len` vectors.
    pub(crate) fn all(universe_len: usize) -> Self {
        let repr = if universe_len == 0 {
            CandidateRepr::Empty
        } else {
            CandidateRepr::All
        };
        Self { universe_len, repr }
    }

    /// An empty set bound to a universe of `universe_len` vectors.
    pub(crate) fn empty(universe_len: usize) -> Self {
        Self {
            universe_len,
            repr: CandidateRepr::Empty,
        }
    }

    /// Build a set from vector IDs, sorting and deduplicating them.
    pub(crate) fn from_ids(
        universe_len: usize,
        ids: impl IntoIterator<Item = u32>,
    ) -> Result<Self> {
        let mut ids: Vec<u32> = ids.into_iter().collect();
        for &id in &ids {
            validate_id(id, universe_len)?;
        }
        ids.sort_unstable();
        ids.dedup();
        Ok(Self::from_sorted_ids(universe_len, ids))
    }

    /// Number of vectors in the universe to which this set belongs.
    pub const fn universe_len(&self) -> usize {
        self.universe_len
    }

    /// Number of eligible vector IDs.
    pub fn len(&self) -> usize {
        match &self.repr {
            CandidateRepr::All => self.universe_len,
            CandidateRepr::Empty => 0,
            CandidateRepr::Sparse(ids) => ids.len(),
            CandidateRepr::Dense { len, .. } => *len,
        }
    }

    /// Whether this candidate set contains no IDs.
    pub fn is_empty(&self) -> bool {
        matches!(self.repr, CandidateRepr::Empty)
    }

    /// Test membership, returning an error when `id` is outside the universe.
    pub fn contains(&self, id: u32) -> Result<bool> {
        validate_id(id, self.universe_len)?;
        Ok(self.contains_valid(id))
    }

    /// Return the intersection of two sets from the same universe.
    pub fn intersection(&self, other: &Self) -> Result<Self> {
        self.ensure_compatible(other)?;

        let result = match (&self.repr, &other.repr) {
            (CandidateRepr::Empty, _) | (_, CandidateRepr::Empty) => Self::empty(self.universe_len),
            (CandidateRepr::All, _) => other.clone(),
            (_, CandidateRepr::All) => self.clone(),
            (CandidateRepr::Sparse(left), CandidateRepr::Sparse(right)) => {
                Self::from_sorted_ids(self.universe_len, intersect_sparse(left, right))
            }
            (CandidateRepr::Sparse(ids), CandidateRepr::Dense { words, .. })
            | (CandidateRepr::Dense { words, .. }, CandidateRepr::Sparse(ids)) => {
                let kept = ids
                    .iter()
                    .copied()
                    .filter(|&id| dense_contains(words, id))
                    .collect();
                Self::from_sorted_ids(self.universe_len, kept)
            }
            (
                CandidateRepr::Dense { words: left, .. },
                CandidateRepr::Dense { words: right, .. },
            ) => Self::combine_dense(self.universe_len, left, right, |a, b| a & b),
        };
        Ok(result)
    }

    /// Return the union of two sets from the same universe.
    pub fn union(&self, other: &Self) -> Result<Self> {
        self.ensure_compatible(other)?;

        let result = match (&self.repr, &other.repr) {
            (CandidateRepr::All, _) | (_, CandidateRepr::All) => Self::all(self.universe_len),
            (CandidateRepr::Empty, _) => other.clone(),
            (_, CandidateRepr::Empty) => self.clone(),
            (CandidateRepr::Sparse(left), CandidateRepr::Sparse(right)) => {
                Self::from_sorted_ids(self.universe_len, union_sparse(left, right))
            }
            (CandidateRepr::Sparse(ids), CandidateRepr::Dense { words, .. })
            | (CandidateRepr::Dense { words, .. }, CandidateRepr::Sparse(ids)) => {
                let mut combined = words.to_vec();
                for &id in ids.iter() {
                    combined[id as usize / 64] |= 1_u64 << (id % 64);
                }
                Self::from_dense_words(self.universe_len, combined)
            }
            (
                CandidateRepr::Dense { words: left, .. },
                CandidateRepr::Dense { words: right, .. },
            ) => Self::combine_dense(self.universe_len, left, right, |a, b| a | b),
        };
        Ok(result)
    }

    /// Return every ID in the same universe that is not in this set.
    pub fn complement(&self) -> Self {
        match &self.repr {
            CandidateRepr::All => Self::empty(self.universe_len),
            CandidateRepr::Empty => Self::all(self.universe_len),
            CandidateRepr::Sparse(ids) => {
                let mut words = all_dense_words(self.universe_len);
                for &id in ids.iter() {
                    words[id as usize / 64] &= !(1_u64 << (id % 64));
                }
                Self::from_dense_words(self.universe_len, words)
            }
            CandidateRepr::Dense { words, .. } => {
                let mut complement: Vec<u64> = words.iter().map(|word| !word).collect();
                mask_unused_bits(&mut complement, self.universe_len);
                Self::from_dense_words(self.universe_len, complement)
            }
        }
    }

    /// Representation selected for backend execution.
    pub(crate) fn view(&self) -> CandidateSetView<'_> {
        match &self.repr {
            CandidateRepr::All => CandidateSetView::All,
            CandidateRepr::Empty => CandidateSetView::Empty,
            CandidateRepr::Sparse(ids) => CandidateSetView::Sparse(ids),
            CandidateRepr::Dense { words, .. } => CandidateSetView::Dense(words),
        }
    }

    fn contains_valid(&self, id: u32) -> bool {
        match &self.repr {
            CandidateRepr::All => true,
            CandidateRepr::Empty => false,
            CandidateRepr::Sparse(ids) => ids.binary_search(&id).is_ok(),
            CandidateRepr::Dense { words, .. } => dense_contains(words, id),
        }
    }

    fn ensure_compatible(&self, other: &Self) -> Result<()> {
        if self.universe_len == other.universe_len {
            return Ok(());
        }
        Err(RingDbError::CandidateSetMismatch {
            expected: self.universe_len,
            got: other.universe_len,
        })
    }

    fn from_sorted_ids(universe_len: usize, ids: Vec<u32>) -> Self {
        if ids.is_empty() {
            return Self::empty(universe_len);
        }
        if ids.len() == universe_len {
            return Self::all(universe_len);
        }
        if prefer_dense(universe_len, ids.len()) {
            return Self::from_dense_words(universe_len, dense_from_ids(universe_len, &ids));
        }
        Self {
            universe_len,
            repr: CandidateRepr::Sparse(ids.into()),
        }
    }

    fn from_dense_words(universe_len: usize, words: Vec<u64>) -> Self {
        let len = words.iter().map(|word| word.count_ones() as usize).sum();
        if len == 0 {
            return Self::empty(universe_len);
        }
        if len == universe_len {
            return Self::all(universe_len);
        }
        if !prefer_dense(universe_len, len) {
            return Self::from_sorted_ids(universe_len, sparse_from_dense(&words, universe_len));
        }
        Self {
            universe_len,
            repr: CandidateRepr::Dense {
                words: words.into(),
                len,
            },
        }
    }

    fn combine_dense(
        universe_len: usize,
        left: &[u64],
        right: &[u64],
        combine: fn(u64, u64) -> u64,
    ) -> Self {
        let words = left
            .iter()
            .zip(right)
            .map(|(&a, &b)| combine(a, b))
            .collect();
        Self::from_dense_words(universe_len, words)
    }
}

fn validate_id(id: u32, universe_len: usize) -> Result<()> {
    if (id as usize) < universe_len {
        return Ok(());
    }
    Err(RingDbError::InvalidVectorId {
        id,
        len: universe_len,
    })
}

fn dense_word_count(universe_len: usize) -> usize {
    universe_len / 64 + usize::from(!universe_len.is_multiple_of(64))
}

fn prefer_dense(universe_len: usize, len: usize) -> bool {
    // One bitmap word occupies the same space as two sparse u32 IDs.
    len > dense_word_count(universe_len).saturating_mul(2)
}

fn dense_from_ids(universe_len: usize, ids: &[u32]) -> Vec<u64> {
    let mut words = vec![0; dense_word_count(universe_len)];
    for &id in ids {
        words[id as usize / 64] |= 1_u64 << (id % 64);
    }
    words
}

fn all_dense_words(universe_len: usize) -> Vec<u64> {
    let mut words = vec![u64::MAX; dense_word_count(universe_len)];
    mask_unused_bits(&mut words, universe_len);
    words
}

fn mask_unused_bits(words: &mut [u64], universe_len: usize) {
    let used_bits = universe_len % 64;
    if used_bits == 0 {
        return;
    }
    if let Some(last) = words.last_mut() {
        *last &= (1_u64 << used_bits) - 1;
    }
}

fn dense_contains(words: &[u64], id: u32) -> bool {
    words[id as usize / 64] & (1_u64 << (id % 64)) != 0
}

fn sparse_from_dense(words: &[u64], universe_len: usize) -> Vec<u32> {
    let mut ids = Vec::new();
    for (word_index, &word) in words.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let id = word_index * 64 + bit;
            if id < universe_len {
                ids.push(id as u32);
            }
            remaining &= remaining - 1;
        }
    }
    ids
}

fn intersect_sparse(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(left.len().min(right.len()));
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                result.push(left[left_index]);
                left_index += 1;
                right_index += 1;
            }
        }
    }
    result
}

fn union_sparse(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(left.len().saturating_add(right.len()));
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => {
                result.push(left[left_index]);
                left_index += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(right[right_index]);
                right_index += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(left[left_index]);
                left_index += 1;
                right_index += 1;
            }
        }
    }
    result.extend_from_slice(&left[left_index..]);
    result.extend_from_slice(&right[right_index..]);
    result
}

/// Mutable metadata accumulator. Each call to `push_vector` advances the ID.
#[derive(Debug, Default)]
pub(crate) struct MetadataIndexBuilder {
    universe_len: usize,
    postings: BTreeMap<MetadataTag, Vec<u32>>,
}

impl MetadataIndexBuilder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Attach tags to the next sequential vector ID.
    pub(crate) fn push_vector(&mut self, tags: &[MetadataTag]) {
        debug_assert!(self.universe_len < u32::MAX as usize);
        let id = self.universe_len as u32;

        for (index, &tag) in tags.iter().enumerate() {
            if tags[..index].contains(&tag) {
                continue;
            }
            self.postings.entry(tag).or_default().push(id);
        }
        self.universe_len += 1;
    }

    pub(crate) fn finish(self) -> MetadataIndex {
        let entries = self
            .postings
            .into_iter()
            .map(|(tag, ids)| (tag, CandidateSet::from_sorted_ids(self.universe_len, ids)))
            .collect();
        MetadataIndex {
            universe_len: self.universe_len,
            entries,
        }
    }
}

/// Immutable equality index over explicit integer metadata tags.
#[derive(Debug, Clone)]
pub(crate) struct MetadataIndex {
    universe_len: usize,
    entries: BTreeMap<MetadataTag, CandidateSet>,
}

impl MetadataIndex {
    /// Return candidates tagged with exactly `(key, value)`.
    pub(crate) fn candidates_eq(&self, key: MetadataKey, value: MetadataValue) -> CandidateSet {
        self.entries
            .get(&MetadataTag::new(key, value))
            .cloned()
            .unwrap_or_else(|| CandidateSet::empty(self.universe_len))
    }

    /// Persist this index as one deterministic, little-endian file.
    pub(crate) fn persist(&self, path: &Path) -> Result<()> {
        let universe_len = u64::try_from(self.universe_len)
            .map_err(|_| corrupt("metadata universe length does not fit in u64"))?;
        let entry_count = u64::try_from(self.entries.len())
            .map_err(|_| corrupt("metadata entry count does not fit in u64"))?;

        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(&METADATA_MAGIC)?;
        writer.write_all(&METADATA_VERSION.to_le_bytes())?;
        writer.write_all(&universe_len.to_le_bytes())?;
        writer.write_all(&entry_count.to_le_bytes())?;
        for (&tag, candidates) in &self.entries {
            write_entry(&mut writer, tag, candidates)?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Load and fully validate a persisted metadata index.
    pub(crate) fn load(path: &Path, expected_universe_len: usize) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let mut reader = MetadataReader::new(&bytes);
        let entry_count = read_header(&mut reader, expected_universe_len)?;
        if entry_count > reader.remaining() / ENTRY_HEADER_LEN {
            return Err(corrupt("metadata entry count exceeds file length"));
        }

        let mut entries = BTreeMap::new();
        let mut previous = None;
        for _ in 0..entry_count {
            let (tag, candidates) = read_entry(&mut reader, expected_universe_len)?;
            if previous.is_some_and(|last| last >= tag) {
                return Err(corrupt("metadata tags are duplicated or out of order"));
            }
            previous = Some(tag);
            entries.insert(tag, candidates);
        }
        if reader.remaining() != 0 {
            return Err(corrupt("metadata file has trailing bytes"));
        }

        Ok(Self {
            universe_len: expected_universe_len,
            entries,
        })
    }
}

fn write_entry(writer: &mut impl Write, tag: MetadataTag, set: &CandidateSet) -> Result<()> {
    writer.write_all(&tag.key.get().to_le_bytes())?;
    let (encoding, cardinality, payload_count) = entry_description(set)?;
    writer.write_all(&[encoding, 0, 0, 0])?;
    writer.write_all(&tag.value.get().to_le_bytes())?;
    writer.write_all(&cardinality.to_le_bytes())?;
    writer.write_all(&payload_count.to_le_bytes())?;

    match &set.repr {
        CandidateRepr::All | CandidateRepr::Empty => {}
        CandidateRepr::Sparse(ids) => {
            for &id in ids.iter() {
                writer.write_all(&id.to_le_bytes())?;
            }
        }
        CandidateRepr::Dense { words, .. } => {
            for &word in words.iter() {
                writer.write_all(&word.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

fn entry_description(set: &CandidateSet) -> Result<(u8, u64, u64)> {
    let cardinality = u64::try_from(set.len())
        .map_err(|_| corrupt("metadata candidate count does not fit in u64"))?;
    let description = match &set.repr {
        CandidateRepr::All => (ENCODING_ALL, cardinality, 0),
        CandidateRepr::Empty => {
            return Err(corrupt("metadata index contains an empty posting"));
        }
        CandidateRepr::Sparse(_) => (ENCODING_SPARSE, cardinality, cardinality),
        CandidateRepr::Dense { words, .. } => (
            ENCODING_DENSE,
            cardinality,
            u64::try_from(words.len())
                .map_err(|_| corrupt("metadata bitmap length does not fit in u64"))?,
        ),
    };
    Ok(description)
}

fn read_header(reader: &mut MetadataReader<'_>, expected_universe_len: usize) -> Result<usize> {
    if reader.total_len() < HEADER_LEN {
        return Err(corrupt("metadata file is too short"));
    }
    if reader.read_array::<8>()? != METADATA_MAGIC {
        return Err(corrupt("metadata file has invalid magic bytes"));
    }
    let version = reader.read_u32()?;
    if version != METADATA_VERSION {
        return Err(corrupt(format!(
            "unsupported metadata format version {version}"
        )));
    }
    let universe_len = usize_from_u64(reader.read_u64()?, "metadata universe length")?;
    if universe_len != expected_universe_len {
        return Err(corrupt(format!(
            "metadata contains {universe_len} vectors, expected {expected_universe_len}"
        )));
    }
    usize_from_u64(reader.read_u64()?, "metadata entry count")
}

fn read_entry(
    reader: &mut MetadataReader<'_>,
    universe_len: usize,
) -> Result<(MetadataTag, CandidateSet)> {
    let key = MetadataKey::new(reader.read_u32()?);
    let encoding_and_reserved = reader.read_array::<4>()?;
    if encoding_and_reserved[1..] != [0, 0, 0] {
        return Err(corrupt("metadata entry has non-zero reserved bytes"));
    }
    let value = MetadataValue::new(reader.read_u64()?);
    let cardinality = usize_from_u64(reader.read_u64()?, "metadata candidate count")?;
    let payload_count = usize_from_u64(reader.read_u64()?, "metadata payload count")?;
    let candidates = read_candidates(
        reader,
        encoding_and_reserved[0],
        cardinality,
        payload_count,
        universe_len,
    )?;
    Ok((MetadataTag::new(key, value), candidates))
}

fn read_candidates(
    reader: &mut MetadataReader<'_>,
    encoding: u8,
    cardinality: usize,
    payload_count: usize,
    universe_len: usize,
) -> Result<CandidateSet> {
    match encoding {
        ENCODING_ALL => read_all_candidates(cardinality, payload_count, universe_len),
        ENCODING_SPARSE => read_sparse_candidates(reader, cardinality, payload_count, universe_len),
        ENCODING_DENSE => read_dense_candidates(reader, cardinality, payload_count, universe_len),
        _ => Err(corrupt(format!(
            "metadata entry has unknown encoding {encoding}"
        ))),
    }
}

fn read_all_candidates(
    cardinality: usize,
    payload_count: usize,
    universe_len: usize,
) -> Result<CandidateSet> {
    if universe_len == 0 || cardinality != universe_len || payload_count != 0 {
        return Err(corrupt("invalid all-candidates metadata entry"));
    }
    Ok(CandidateSet::all(universe_len))
}

fn read_sparse_candidates(
    reader: &mut MetadataReader<'_>,
    cardinality: usize,
    payload_count: usize,
    universe_len: usize,
) -> Result<CandidateSet> {
    if cardinality == 0 || cardinality >= universe_len || payload_count != cardinality {
        return Err(corrupt("invalid sparse metadata entry counts"));
    }
    reader.ensure_values_fit(payload_count, 4, "sparse metadata IDs")?;

    let mut ids = Vec::with_capacity(payload_count);
    let mut previous = None;
    for _ in 0..payload_count {
        let id = reader.read_u32()?;
        if id as usize >= universe_len {
            return Err(corrupt(format!(
                "metadata vector ID {id} is outside a universe of {universe_len}"
            )));
        }
        if previous.is_some_and(|last| last >= id) {
            return Err(corrupt(
                "sparse metadata IDs are duplicated or out of order",
            ));
        }
        previous = Some(id);
        ids.push(id);
    }
    Ok(CandidateSet::from_sorted_ids(universe_len, ids))
}

fn read_dense_candidates(
    reader: &mut MetadataReader<'_>,
    cardinality: usize,
    payload_count: usize,
    universe_len: usize,
) -> Result<CandidateSet> {
    let expected_words = dense_word_count(universe_len);
    if cardinality == 0 || cardinality >= universe_len || payload_count != expected_words {
        return Err(corrupt("invalid dense metadata entry counts"));
    }
    reader.ensure_values_fit(payload_count, 8, "dense metadata words")?;

    let mut words = Vec::with_capacity(payload_count);
    for _ in 0..payload_count {
        words.push(reader.read_u64()?);
    }
    if dense_has_unused_bits(&words, universe_len) {
        return Err(corrupt(
            "dense metadata entry sets IDs outside its universe",
        ));
    }
    let actual_cardinality: usize = words.iter().map(|word| word.count_ones() as usize).sum();
    if actual_cardinality != cardinality {
        return Err(corrupt(
            "dense metadata entry has an invalid candidate count",
        ));
    }
    Ok(CandidateSet::from_dense_words(universe_len, words))
}

fn dense_has_unused_bits(words: &[u64], universe_len: usize) -> bool {
    let used_bits = universe_len % 64;
    if used_bits == 0 {
        return false;
    }
    words.last().is_some_and(|last| last >> used_bits != 0)
}

fn usize_from_u64(value: u64, label: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| corrupt(format!("{label} does not fit in usize")))
}

fn corrupt(message: impl Into<String>) -> RingDbError {
    RingDbError::Corrupt(message.into())
}

struct MetadataReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> MetadataReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn total_len(&self) -> usize {
        self.bytes.len()
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .position
            .checked_add(N)
            .ok_or_else(|| corrupt("metadata read position overflow"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| corrupt("metadata file is truncated"))?;
        self.position = end;
        Ok(bytes.try_into().expect("slice length was checked"))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn ensure_values_fit(&self, count: usize, width: usize, label: &str) -> Result<()> {
        let byte_count = count
            .checked_mul(width)
            .ok_or_else(|| corrupt(format!("{label} byte length overflows usize")))?;
        if byte_count > self.remaining() {
            return Err(corrupt(format!("{label} exceed the metadata file length")));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

    fn ids(set: &CandidateSet) -> Vec<u32> {
        (0..set.universe_len() as u32)
            .filter(|&id| set.contains(id).unwrap())
            .collect()
    }

    const fn tag(key: u32, value: u64) -> MetadataTag {
        MetadataTag::new(MetadataKey::new(key), MetadataValue::new(value))
    }

    fn temp_file() -> std::path::PathBuf {
        let sequence = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "ring-db-metadata-{}-{sequence}.bin",
            std::process::id()
        ))
    }

    #[test]
    fn candidate_sets_validate_and_apply_boolean_operations() {
        let left = CandidateSet::from_ids(10, [7, 1, 3, 3]).unwrap();
        let right = CandidateSet::from_ids(10, [2, 3, 7, 8]).unwrap();

        assert_eq!(ids(&left), vec![1, 3, 7]);
        assert_eq!(ids(&left.intersection(&right).unwrap()), vec![3, 7]);
        assert_eq!(ids(&left.union(&right).unwrap()), vec![1, 2, 3, 7, 8]);
        assert_eq!(ids(&left.complement()), vec![0, 2, 4, 5, 6, 8, 9]);
        assert!(left.contains(10).is_err());
        assert!(left.intersection(&CandidateSet::all(11)).is_err());
    }

    #[test]
    fn candidates_adapt_between_special_sparse_and_dense_forms() {
        assert!(matches!(
            CandidateSet::all(100).view(),
            CandidateSetView::All
        ));
        assert!(matches!(
            CandidateSet::empty(100).view(),
            CandidateSetView::Empty
        ));
        assert!(matches!(
            CandidateSet::from_ids(100, [1]).unwrap().view(),
            CandidateSetView::Sparse(_)
        ));
        assert!(matches!(
            CandidateSet::from_ids(100, 0..50).unwrap().view(),
            CandidateSetView::Dense(_)
        ));
    }

    #[test]
    fn metadata_builder_deduplicates_tags_and_advances_empty_vectors() {
        let bakery = tag(1, 7);
        let mut builder = MetadataIndexBuilder::new();
        builder.push_vector(&[bakery, bakery]);
        builder.push_vector(&[]);
        builder.push_vector(&[bakery]);
        let index = builder.finish();

        let bakeries = index.candidates_eq(bakery.key, bakery.value);
        assert_eq!(bakeries.universe_len(), 3);
        assert_eq!(ids(&bakeries), vec![0, 2]);
        assert!(
            index
                .candidates_eq(MetadataKey::new(99), MetadataValue::new(99))
                .is_empty()
        );
    }

    #[test]
    fn metadata_persistence_round_trips_deterministically() {
        let common = tag(1, 1);
        let occasional = tag(2, 9);
        let mut builder = MetadataIndexBuilder::new();
        for id in 0..100 {
            let tags: &[MetadataTag] = match id {
                3 | 70 => &[common, occasional],
                _ => &[common],
            };
            builder.push_vector(tags);
        }
        let index = builder.finish();
        let first = temp_file();
        let second = temp_file();

        index.persist(&first).unwrap();
        let loaded = MetadataIndex::load(&first, 100).unwrap();
        loaded.persist(&second).unwrap();

        assert_eq!(
            std::fs::read(&first).unwrap(),
            std::fs::read(&second).unwrap()
        );
        assert_eq!(
            ids(&loaded.candidates_eq(occasional.key, occasional.value)),
            vec![3, 70]
        );
        std::fs::remove_file(first).unwrap();
        std::fs::remove_file(second).unwrap();
    }

    #[test]
    fn metadata_load_rejects_truncation_and_wrong_universe() {
        let mut builder = MetadataIndexBuilder::new();
        builder.push_vector(&[tag(1, 2)]);
        let index = builder.finish();
        let path = temp_file();
        index.persist(&path).unwrap();

        assert!(MetadataIndex::load(&path, 2).is_err());
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.pop();
        std::fs::write(&path, bytes).unwrap();
        assert!(MetadataIndex::load(&path, 1).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
