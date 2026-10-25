//! Immutable blocks of records, the unit of search, storage, and upload.
use std::sync::Arc;

use crate::backend::{DiskSpec, RowHit, VulkanDevice, VulkanSegment, cpu};
use crate::codec::{Decoder, Encoder, corrupt};
use crate::column::{Column, ColumnBuilder};
use crate::error::Result;
use crate::query::Key;
use crate::rows::RowSet;
use crate::schema::Schema;
use crate::value::Row;

const MAGIC: &[u8; 8] = b"RDBSEG\0\0";
const VERSION: u32 = 1;

/// Records sorted by key, with row-major vectors and one column per field.
pub(crate) struct Segment {
    /// Unique within a database; names the segment file.
    pub(crate) id: u64,
    pub(crate) dims: usize,
    pub(crate) keys: Arc<[Key]>,
    pub(crate) vectors: Vec<f32>,
    pub(crate) columns: Vec<Column>,
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
        let mut e = Encoder::default();
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

    /// Read a segment file written by [`Segment::encode`] for `schema`.
    pub(crate) fn decode(bytes: &[u8], schema: &Schema, id: u64) -> Result<Self> {
        let what = format!("segment {id}");
        let mut d = Decoder::with_crc(bytes, &what)?;
        d.header(MAGIC, VERSION, &what)?;
        let (stored_id, dims, columns) = (d.u64()?, d.u32()? as usize, d.u32()? as usize);
        if stored_id != id || dims != schema.dims() || columns != schema.len() {
            return Err(corrupt(format!("{what} does not match the manifest")));
        }
        let keys: Vec<Key> = d.slice()?;
        let vectors: Vec<f32> = d.slice()?;
        let rows = keys.len();
        if keys.is_empty() || keys.windows(2).any(|w| w[0] >= w[1]) {
            return Err(corrupt(format!("{what} keys are empty or out of order")));
        }
        if rows.checked_mul(dims) != Some(vectors.len()) || vectors.iter().any(|v| !v.is_finite()) {
            return Err(corrupt(format!("{what} has invalid vectors")));
        }
        let columns = (0..columns)
            .map(|field| Column::decode(&mut d, schema.ty(field), rows))
            .collect::<Result<_>>()?;
        if !d.is_empty() {
            return Err(corrupt(format!("{what} has trailing data")));
        }
        Ok(Self {
            id,
            dims,
            keys: keys.into(),
            vectors,
            columns,
            gpu: None,
        })
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
        Ok(())
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
            vectors: self.vectors,
            columns: self
                .columns
                .into_iter()
                .map(ColumnBuilder::finish)
                .collect(),
            gpu: None,
        }
    }
}
