//! Immutable blocks of records, the unit of search, storage, and upload.
use std::sync::Arc;

use crate::backend::{DiskSpec, Mask, RowHit, VulkanDevice, VulkanSegment, cpu};
use crate::column::{Column, ColumnBuilder};
use crate::error::Result;
use crate::query::Key;
use crate::schema::Schema;
use crate::value::Row;

/// Records sorted by key, with row-major vectors and one column per field.
pub(crate) struct Segment {
    pub(crate) dims: usize,
    pub(crate) keys: Arc<[Key]>,
    pub(crate) vectors: Vec<f32>,
    pub(crate) columns: Vec<Column>,
    gpu: Option<(Arc<VulkanDevice>, VulkanSegment)>,
}

impl Segment {
    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    pub(crate) fn is_uploaded(&self) -> bool {
        self.gpu.is_some()
    }

    /// Keep a resident copy of the vectors on `device`.
    pub(crate) fn upload(&mut self, device: &Arc<VulkanDevice>) -> Result<()> {
        self.gpu = Some((device.clone(), device.upload(self.dims, &self.vectors)?));
        Ok(())
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

    pub(crate) fn knn(
        &self,
        query: &[f32],
        weights: Option<&[f32]>,
        k: usize,
        mask: Mask<'_>,
    ) -> Result<Vec<RowHit>> {
        match &self.gpu {
            Some((device, gpu)) => device.knn(gpu, query, weights, k, mask),
            None => Ok(cpu::knn(&self.vectors, self.dims, query, weights, k, mask)),
        }
    }

    /// Append region records with rows offset by `base`.
    pub(crate) fn region(
        &self,
        base: u32,
        disks: &[DiskSpec<'_>],
        mask: Mask<'_>,
        records: &mut Vec<u32>,
    ) -> Result<()> {
        match &self.gpu {
            Some((device, gpu)) => device.region(gpu, base, disks, mask, records),
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

    /// Freeze the records. Upload them separately.
    pub(crate) fn finish(self) -> Segment {
        Segment {
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
