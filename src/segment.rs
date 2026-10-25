//! Immutable blocks of records, the unit of search, storage, and upload.
use std::fs::File;
use std::sync::Arc;

use memmap2::Mmap;

use crate::backend::{DiskSpec, RowHit, VulkanDevice, VulkanSegment, cpu};
use crate::buf::Buf;
use crate::codec::{Decoder, Encoder, corrupt};
use crate::column::{Column, ColumnBuilder};
use crate::error::Result;
use crate::query::Key;
use crate::rows::RowSet;
use crate::schema::Schema;
use crate::value::Row;

const MAGIC: &[u8; 8] = b"RDBSEG\0\0";
/// Version 2 is aligned for reading in place, and has compact keyword columns,
/// `u32` offsets, and stored orders of numeric columns. Version 1 files, written
/// by ring-db 0.10, are still read.
const VERSION: u32 = 2;

/// Records sorted by key, with row-major vectors and one column per field.
pub(crate) struct Segment {
    /// Unique within a database; names the segment file.
    pub(crate) id: u64,
    pub(crate) dims: usize,
    pub(crate) keys: Buf<Key>,
    pub(crate) vectors: Buf<f32>,
    pub(crate) columns: Vec<Column>,
    /// The mapped file that the arrays borrow from, if any.
    file: Option<Arc<Mmap>>,
    gpu: Option<Resident>,
}

/// The copy of a segment's vectors on a GPU.
struct Resident {
    device: Arc<VulkanDevice>,
    vectors: VulkanSegment,
    /// Searches of fewer rows run on the CPU.
    min_rows: usize,
}

impl Segment {
    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    /// The segment file contents, ending with a CRC-32.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut e = Encoder::aligned();
        e.header(MAGIC, VERSION);
        e.u64(self.id);
        e.u32(self.dims as u32);
        e.u32(self.columns.len() as u32);
        e.slice(&self.keys);
        e.slice(&self.vectors);
        for column in &self.columns {
            column.encode(&mut e);
        }
        e.finish_with_crc()
    }

    /// Map a segment file written by [`Segment::encode`] for `schema`, and
    /// read its arrays in place.
    pub(crate) fn map(file: &File, schema: &Schema, id: u64) -> Result<Self> {
        // SAFETY: ring-db never changes a segment file once it is written, and
        // the database directory is for ring-db only.
        let map = Arc::new(unsafe { Mmap::map(file)? });
        let what = format!("segment {id}");
        let mut segment = Self::decode(Decoder::mapped(&map, &what)?, schema, id, &what)?;
        segment.file = Some(map);
        Ok(segment)
    }

    fn decode(mut d: Decoder<'_>, schema: &Schema, id: u64, what: &str) -> Result<Self> {
        let version = d.header_versions(MAGIC, 1..=VERSION, what)?;
        if version >= 2 {
            d.set_aligned();
        }
        let (stored_id, dims, columns) = (d.u64()?, d.u32()? as usize, d.u32()? as usize);
        if stored_id != id || dims != schema.dims() || columns != schema.len() {
            return Err(corrupt(format!("{what} does not match the manifest")));
        }
        let keys: Buf<Key> = d.buf()?;
        let vectors: Buf<f32> = d.buf()?;
        let rows = keys.len();
        if keys.is_empty() || keys.windows(2).any(|w| w[0] >= w[1]) {
            return Err(corrupt(format!("{what} keys are empty or out of order")));
        }
        if rows.checked_mul(dims) != Some(vectors.len()) || vectors.iter().any(|v| !v.is_finite()) {
            return Err(corrupt(format!("{what} has invalid vectors")));
        }
        let columns = (0..columns)
            .map(|field| Column::decode(&mut d, schema.ty(field), rows, version))
            .collect::<Result<_>>()?;
        if !d.is_empty() {
            return Err(corrupt(format!("{what} has trailing data")));
        }
        Ok(Self {
            id,
            dims,
            keys,
            vectors,
            columns,
            file: None,
            gpu: None,
        })
    }

    /// Whether the arrays are read from a mapped file.
    pub(crate) fn is_mapped(&self) -> bool {
        self.file.is_some()
    }

    pub(crate) fn is_uploaded(&self) -> bool {
        self.gpu.is_some()
    }

    /// Keep a copy of the vectors on `device` for searches of at least
    /// `min_rows` rows.
    pub(crate) fn upload(&mut self, device: &Arc<VulkanDevice>, min_rows: usize) -> Result<()> {
        self.gpu = Some(Resident {
            device: device.clone(),
            vectors: device.upload(self.dims, &self.vectors)?,
            min_rows,
        });
        self.release_vectors();
        Ok(())
    }

    /// Remove the pages of mapped vectors from this process. A CPU search
    /// reads them again from the page cache or the file.
    fn release_vectors(&self) {
        #[cfg(unix)]
        if let Some(map) = &self.file {
            let len = size_of_val(&*self.vectors);
            let offset = (self.vectors.as_ptr() as usize).checked_sub(map.as_ptr() as usize);
            if let Some(offset) = offset.filter(|&offset| offset + len <= map.len()) {
                // SAFETY: the mapping is private and never written, so the
                // file gives the same bytes again.
                let advice = memmap2::UncheckedAdvice::DontNeed;
                let _ = unsafe { map.unchecked_advise_range(advice, offset, len) };
            }
        }
    }

    /// The GPU copy, if a search of `rows` uses it.
    fn gpu(&self, rows: &RowSet) -> Option<&Resident> {
        self.gpu.as_ref().filter(|gpu| rows.len() >= gpu.min_rows)
    }

    /// The row holding `key`.
    pub(crate) fn row(&self, key: Key) -> Option<u32> {
        self.keys.binary_search(&key).ok().map(|row| row as u32)
    }

    pub(crate) fn read(&self, schema: &Schema, row: usize) -> Row {
        Row {
            vector: self.vectors[row * self.dims..(row + 1) * self.dims].to_vec(),
            values: self
                .columns
                .iter()
                .enumerate()
                .map(|(field, column)| column.value(row, schema.ty(field)))
                .collect(),
        }
    }

    /// The `k` best of `rows` by `(dist_sq, row)`, sorted.
    pub(crate) fn knn(
        &self,
        query: &[f32],
        weights: Option<&[f32]>,
        k: usize,
        rows: &RowSet,
    ) -> Result<Vec<RowHit>> {
        let Some(mask) = rows.mask() else {
            return Ok(Vec::new());
        };
        match self.gpu(rows) {
            Some(gpu) => gpu.device.knn(&gpu.vectors, query, weights, k, mask),
            None => Ok(cpu::knn(&self.vectors, self.dims, query, weights, k, mask)),
        }
    }

    /// Append the region records of `rows`, with rows offset by `base`.
    pub(crate) fn region(
        &self,
        base: u32,
        disks: &[DiskSpec<'_>],
        rows: &RowSet,
        records: &mut Vec<u32>,
    ) -> Result<()> {
        let Some(mask) = rows.mask() else {
            return Ok(());
        };
        match self.gpu(rows) {
            Some(gpu) => gpu.device.region(&gpu.vectors, base, disks, mask, records),
            None => {
                cpu::region(&self.vectors, self.dims, base, disks, mask, records);
                Ok(())
            }
        }
    }

    /// Number of `rows` inside all `disks`.
    pub(crate) fn region_count(&self, disks: &[DiskSpec<'_>], rows: &RowSet) -> Result<usize> {
        let Some(mask) = rows.mask() else {
            return Ok(0);
        };
        match self.gpu(rows) {
            Some(gpu) => gpu.device.region_count(&gpu.vectors, disks, mask),
            None => Ok(cpu::region_count(&self.vectors, self.dims, disks, mask)),
        }
    }
}

/// Collects records with increasing keys into a [`Segment`].
pub(crate) struct SegmentBuilder {
    dims: usize,
    keys: Vec<Key>,
    vectors: Vec<f32>,
    columns: Vec<ColumnBuilder>,
}

impl SegmentBuilder {
    pub(crate) fn new(schema: &Schema) -> Self {
        Self {
            dims: schema.dims(),
            keys: Vec::new(),
            vectors: Vec::new(),
            columns: schema
                .fields()
                .map(|(_, ty)| ColumnBuilder::new(ty))
                .collect(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The row holding `key`.
    pub(crate) fn row(&self, key: Key) -> Option<u32> {
        self.keys.binary_search(&key).ok().map(|row| row as u32)
    }

    /// Append a validated row. Keys must increase.
    pub(crate) fn push(&mut self, key: Key, row: Row) {
        debug_assert!(self.keys.last().is_none_or(|&last| last < key));
        self.keys.push(key);
        self.vectors.extend_from_slice(&row.vector);
        for (column, value) in self.columns.iter_mut().zip(row.values) {
            column.push(value);
        }
    }

    /// Freeze the records as segment `id`. Upload them separately.
    pub(crate) fn finish(self, id: u64) -> Segment {
        Segment {
            id,
            dims: self.dims,
            keys: self.keys.into(),
            vectors: self.vectors.into(),
            columns: self
                .columns
                .into_iter()
                .map(ColumnBuilder::finish)
                .collect(),
            file: None,
            gpu: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::schema::FieldType;
    use crate::value::Value;

    #[test]
    fn mapped_segments_read_their_arrays_in_place() {
        let types = [
            FieldType::U32,
            FieldType::U64,
            FieldType::I64,
            FieldType::F32,
            FieldType::Bool,
            FieldType::Keyword,
            FieldType::Keywords,
            FieldType::Text,
            FieldType::TextList,
            FieldType::Bytes,
        ];
        let schema = types
            .iter()
            .enumerate()
            .fold(Schema::new(3), |s, (i, &ty)| s.field(&format!("f{i}"), ty));
        let mut builder = SegmentBuilder::new(&schema);
        for i in 0..300u32 {
            let text = format!("v{}", i % 7);
            let values = [
                Value::U32(i),
                Value::U64(u64::from(i) * 3),
                Value::I64(-i64::from(i)),
                Value::F32(i as f32 / 2.0),
                Value::Bool(i % 3 == 0),
                Value::Text(text.clone()),
                Value::TextList(vec![text.clone(), format!("w{}", i % 50)]),
                Value::Text(text.clone()),
                Value::TextList(vec![text]),
                Value::Bytes(vec![i as u8; 3]),
            ];
            let row = Row {
                vector: vec![i as f32; 3],
                values: values
                    .into_iter()
                    .map(|v| (i % 5 != 0).then_some(v))
                    .collect(),
            };
            builder.push(Key(u64::from(i)), row);
        }
        let built = builder.finish(7);
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(&built.encode()).unwrap();
        let segment = Segment::map(&file, &schema, 7).unwrap();
        let map = segment.file.as_ref().unwrap();
        let mapped = map.as_ptr_range();
        let arrays: Vec<&[u8]> = [
            bytemuck::cast_slice::<Key, u8>(&segment.keys),
            bytemuck::cast_slice(&segment.vectors),
        ]
        .into_iter()
        .chain(segment.columns.iter().flat_map(Column::arrays))
        .collect();
        // An empty array, such as the order of a bool column, needs no memory.
        let arrays: Vec<&[u8]> = arrays.into_iter().filter(|a| !a.is_empty()).collect();
        assert!(arrays.len() > 30);
        for array in arrays {
            let range = array.as_ptr_range();
            assert!(mapped.start <= range.start && range.end <= mapped.end);
        }
        for row in 0..300 {
            assert_eq!(segment.read(&schema, row), built.read(&schema, row));
        }
    }
}
