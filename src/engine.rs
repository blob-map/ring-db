use std::mem::{align_of, size_of};
use std::path::Path;
use std::sync::Arc;

use bytemuck::Pod;
use smallvec::SmallVec;

use crate::BackendPreference;
use crate::backend::{DiskSpec, Mask, VulkanDevice, VulkanSegment, cpu};
use crate::config::RingDbConfig;
use crate::error::{Result, RingDbError};
use crate::metadata::{
    CandidateSet, CandidateSetView, MetadataIndex, MetadataIndexBuilder, MetadataKey, MetadataTag,
    MetadataValue,
};
use crate::payload::{read_pod_file, write_pod_file};
use crate::persist::{DatabaseMeta, read_f32_file, read_meta, write_f32_file, write_meta};
use crate::query::{
    Disk, Hit, Key, KeyLookup, Knn, MAX_DISKS, Metric, Query, Region, RegionResult,
};

fn make_device(preference: BackendPreference) -> Result<Option<Arc<VulkanDevice>>> {
    match preference {
        BackendPreference::Cpu => Ok(None),
        BackendPreference::Vulkan => Ok(Some(Arc::new(VulkanDevice::new()?))),
    }
}

fn validate_vector(dims: usize, vector: &[f32], label: &str) -> Result<()> {
    if vector.len() != dims {
        return Err(RingDbError::DimensionMismatch {
            expected: dims,
            got: vector.len(),
        });
    }
    if let Some((index, value)) = vector
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(RingDbError::InvalidQuery(format!(
            "{label} coordinate at index {index} must be finite, got {value}"
        )));
    }
    Ok(())
}

fn validate_metric(dims: usize, metric: Metric<'_>) -> Result<()> {
    let Some(weights) = metric.weights() else {
        return Ok(());
    };
    if weights.len() != dims {
        return Err(RingDbError::DimensionMismatch {
            expected: dims,
            got: weights.len(),
        });
    }
    if let Some((index, weight)) = weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| !weight.is_finite() || *weight < 0.0)
    {
        return Err(RingDbError::InvalidQuery(format!(
            "weight at index {index} must be finite and non-negative, got {weight}"
        )));
    }
    Ok(())
}

fn validate_disk<'a>(dims: usize, disk: &Disk<'a>) -> Result<DiskSpec<'a>> {
    validate_vector(dims, disk.center, "disk center")?;
    validate_metric(dims, disk.metric)?;
    if !disk.radius.is_finite() || disk.radius < 0.0 {
        return Err(RingDbError::InvalidQuery(format!(
            "disk radius must be finite and non-negative, got {}",
            disk.radius
        )));
    }
    Ok(DiskSpec {
        center: disk.center,
        weights: disk.metric.weights(),
        radius_sq: disk.radius * disk.radius,
    })
}

/// `None` means that no row can match.
fn mask(filter: Option<&CandidateSet>, n_vectors: usize) -> Result<Option<Mask<'_>>> {
    let Some(filter) = filter else {
        return Ok(Some(Mask::All));
    };
    if filter.universe_len() != n_vectors {
        return Err(RingDbError::CandidateSetMismatch {
            expected: n_vectors,
            got: filter.universe_len(),
        });
    }
    Ok(match filter.view() {
        CandidateSetView::All => Some(Mask::All),
        CandidateSetView::Empty => None,
        CandidateSetView::Sparse(ids) => Some(Mask::Sparse(ids)),
        CandidateSetView::Dense(words) => Some(Mask::Dense(words)),
    })
}

/// Vectors and keys of one block, with an optional resident device copy.
struct Block {
    vectors: Vec<f32>,
    keys: Arc<[Key]>,
    gpu: Option<(Arc<VulkanDevice>, VulkanSegment)>,
}

impl Block {
    fn new(dims: usize, vectors: Vec<f32>, device: Option<&Arc<VulkanDevice>>) -> Result<Self> {
        let keys = (0..(vectors.len() / dims) as u64).map(Key).collect();
        let gpu = match device {
            Some(device) => Some((device.clone(), device.upload(dims, &vectors)?)),
            None => None,
        };
        Ok(Self { vectors, keys, gpu })
    }
}

/// Read-only view used to execute one search.
pub struct Searcher<'a> {
    dims: usize,
    block: &'a Block,
}

impl Searcher<'_> {
    pub(crate) fn knn(&self, query: &Knn<'_>) -> Result<Vec<Hit>> {
        validate_vector(self.dims, query.query, "query")?;
        validate_metric(self.dims, query.metric)?;
        let Some(mask) = mask(query.filter, self.block.keys.len())? else {
            return Ok(Vec::new());
        };
        if query.k == 0 {
            return Ok(Vec::new());
        }
        let weights = query.metric.weights();
        let hits = match &self.block.gpu {
            Some((device, segment)) => device.knn(segment, query.query, weights, query.k, mask)?,
            None => cpu::knn(
                &self.block.vectors,
                self.dims,
                query.query,
                weights,
                query.k,
                mask,
            ),
        };
        Ok(hits
            .into_iter()
            .map(|hit| Hit {
                key: self.block.keys[hit.row as usize],
                dist_sq: hit.dist_sq,
            })
            .collect())
    }

    pub(crate) fn region(&self, region: &Region<'_>) -> Result<RegionResult> {
        if region.disks.len() > MAX_DISKS {
            return Err(RingDbError::InvalidQuery(format!(
                "a region can contain at most {MAX_DISKS} disks, got {}",
                region.disks.len()
            )));
        }
        let disks: SmallVec<[DiskSpec<'_>; 1]> = region
            .disks
            .iter()
            .map(|disk| validate_disk(self.dims, disk))
            .collect::<Result<_>>()?;
        let mut out = RegionResult::new(disks.len(), KeyLookup(self.block.keys.clone()));
        let Some(mask) = mask(region.filter, self.block.keys.len())? else {
            return Ok(out);
        };
        let records = &mut out.records;
        match &self.block.gpu {
            Some((device, segment)) => device.region(segment, 0, &disks, mask, records)?,
            None => cpu::region(&self.block.vectors, self.dims, 0, &disks, mask, records),
        }
        Ok(out)
    }
}

fn prepare_persist_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    if std::fs::read_dir(dir)?.next().transpose()?.is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("persist directory '{}' must be empty", dir.display()),
        )
        .into());
    }
    Ok(())
}

/// Mutable builder for an immutable exact vector database.
///
/// `T` is a fixed-size Pod record stored verbatim next to each vector. Metadata
/// used for filtering is supplied explicitly as integer tags; kernels never
/// interpret the payload layout.
pub struct RingDb<T: Pod + Send + Sync = ()> {
    config: RingDbConfig,
    device: Option<Arc<VulkanDevice>>,
    n_vectors: usize,
    vectors: Vec<f32>,
    payloads: Vec<T>,
    metadata_builder: MetadataIndexBuilder,
}

impl<T: Pod + Send + Sync> RingDb<T> {
    /// Create an empty database using `config`.
    pub fn new(config: RingDbConfig) -> Result<Self> {
        if config.dims == 0 {
            return Err(RingDbError::InvalidQuery(
                "database dimensions must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            device: make_device(config.backend_preference)?,
            config,
            n_vectors: 0,
            vectors: Vec::new(),
            payloads: Vec::new(),
            metadata_builder: MetadataIndexBuilder::new(),
        })
    }

    /// Insert a vector and payload without metadata tags.
    pub fn add_vector(&mut self, vector: &[f32], payload: T) -> Result<()> {
        self.add_vector_with_metadata(vector, payload, &[])
    }

    /// Insert a vector, its Pod payload, and equality-filterable metadata tags.
    pub fn add_vector_with_metadata(
        &mut self,
        vector: &[f32],
        payload: T,
        tags: &[MetadataTag],
    ) -> Result<()> {
        validate_vector(self.config.dims, vector, "vector")?;
        if self.n_vectors >= u32::MAX as usize {
            return Err(RingDbError::InvalidQuery(
                "ring-db supports at most u32::MAX vectors".to_string(),
            ));
        }
        self.payloads.push(payload);
        self.metadata_builder.push_vector(tags);
        self.vectors.extend_from_slice(vector);
        self.n_vectors += 1;
        Ok(())
    }

    /// Seal the database and upload its vectors to the selected backend.
    pub fn build(self) -> Result<SealedRingDb<T>> {
        let Self {
            config,
            device,
            n_vectors,
            vectors,
            payloads,
            metadata_builder,
        } = self;
        let metadata_index = metadata_builder.finish();

        let persist_meta = if let Some(dir) = &config.persist_dir {
            prepare_persist_dir(dir)?;
            write_f32_file(&dir.join("vectors.bin"), &vectors)?;
            metadata_index.persist(&dir.join("metadata.bin"))?;
            write_pod_file(&dir.join("payloads.bin"), &payloads)?;
            Some((
                dir.join("meta.bin"),
                DatabaseMeta {
                    dims: config.dims,
                    n_vectors,
                    payload_size: size_of::<T>(),
                    payload_align: align_of::<T>(),
                },
            ))
        } else {
            None
        };

        let block = Block::new(config.dims, vectors, device.as_ref())?;
        if let Some((path, meta)) = persist_meta {
            write_meta(&path, meta)?;
        }
        Ok(SealedRingDb {
            config,
            block,
            payloads: payloads.into_boxed_slice(),
            metadata_index,
        })
    }

    /// Load a database written by ring-db's current persistence format.
    pub fn load(dir: &Path, backend_preference: BackendPreference) -> Result<SealedRingDb<T>> {
        let meta = read_meta(&dir.join("meta.bin"))?;
        if meta.payload_size != size_of::<T>() || meta.payload_align != align_of::<T>() {
            return Err(RingDbError::Corrupt(format!(
                "payload layout mismatch: database uses size/alignment {}/{}, requested type uses {}/{}",
                meta.payload_size,
                meta.payload_align,
                size_of::<T>(),
                align_of::<T>(),
            )));
        }

        let expected_values = meta.n_vectors.checked_mul(meta.dims).ok_or_else(|| {
            RingDbError::Corrupt("vector matrix dimensions overflow usize".to_string())
        })?;
        let vectors = read_f32_file(&dir.join("vectors.bin"), expected_values)?;
        if let Some((index, value)) = vectors
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(RingDbError::Corrupt(format!(
                "vector coordinate at flat index {index} must be finite, got {value}"
            )));
        }
        let payloads = read_pod_file(&dir.join("payloads.bin"), meta.n_vectors)?;
        let metadata_index = MetadataIndex::load(&dir.join("metadata.bin"), meta.n_vectors)?;
        let device = make_device(backend_preference)?;
        let block = Block::new(meta.dims, vectors, device.as_ref())?;

        Ok(SealedRingDb {
            config: RingDbConfig::new(meta.dims)
                .with_persist_dir(dir)
                .with_backend_preference(backend_preference),
            block,
            payloads,
            metadata_index,
        })
    }

    pub fn len(&self) -> usize {
        self.n_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    pub fn dims(&self) -> usize {
        self.config.dims
    }

    pub fn backend_name(&self) -> &str {
        backend_name(self.config.backend_preference)
    }
}

fn backend_name(preference: BackendPreference) -> &'static str {
    match preference {
        BackendPreference::Cpu => "cpu",
        BackendPreference::Vulkan => "vulkan",
    }
}

/// Immutable database returned by [`RingDb::build`] or [`RingDb::load`].
pub struct SealedRingDb<T: Pod + Send + Sync = ()> {
    config: RingDbConfig,
    block: Block,
    payloads: Box<[T]>,
    metadata_index: MetadataIndex,
}

impl<T: Pod + Send + Sync> SealedRingDb<T> {
    /// Run a [`Knn`], [`Disk`], or [`Region`] search.
    pub fn search<Q: Query>(&self, query: Q) -> Result<Q::Output> {
        query.execute(&Searcher {
            dims: self.config.dims,
            block: &self.block,
        })
    }

    /// Return the vectors carrying exactly this metadata tag.
    pub fn metadata_eq(&self, key: MetadataKey, value: MetadataValue) -> CandidateSet {
        self.metadata_index.candidates_eq(key, value)
    }

    /// Create a validated candidate set from vector IDs.
    pub fn candidates_from_ids(&self, ids: impl IntoIterator<Item = u32>) -> Result<CandidateSet> {
        CandidateSet::from_ids(self.len(), ids)
    }

    /// Fetch one Pod payload by key without copying it.
    pub fn fetch_payload(&self, key: Key) -> Result<&T> {
        usize::try_from(key.0)
            .ok()
            .and_then(|row| self.payloads.get(row))
            .ok_or(RingDbError::InvalidVectorId {
                id: key.0 as u32,
                len: self.len(),
            })
    }

    /// Fetch Pod payload references in the same order as `keys`.
    pub fn fetch_payloads(&self, keys: &[Key]) -> Result<Vec<&T>> {
        keys.iter().map(|&key| self.fetch_payload(key)).collect()
    }

    pub fn len(&self) -> usize {
        self.block.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.block.keys.is_empty()
    }

    pub fn dims(&self) -> usize {
        self.config.dims
    }

    pub fn backend_name(&self) -> &str {
        backend_name(self.config.backend_preference)
    }
}
