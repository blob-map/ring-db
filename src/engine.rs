use std::mem::{align_of, size_of};
use std::path::Path;
use std::time::Instant;

use bytemuck::Pod;

use crate::BackendPreference;
#[cfg(feature = "cuda")]
use crate::backend::CudaBackend;
use crate::backend::{CpuBackend, DiskSpec, RingComputeBackend};
use crate::config::RingDbConfig;
use crate::error::{Result, RingDbError};
use crate::metadata::{
    CandidateSet, CandidateSetView, MetadataIndex, MetadataIndexBuilder, MetadataKey, MetadataTag,
    MetadataValue,
};
use crate::payload::{read_pod_file, write_pod_file};
use crate::persist::{DatabaseMeta, read_f32_file, read_meta, write_f32_file, write_meta};
use crate::query::{
    DiskIntersectionQuery, DiskQuery, KnnQuery, QueryResult, RangeQuery, RingQuery,
};

fn make_backend(preference: BackendPreference) -> Result<Box<dyn RingComputeBackend>> {
    match preference {
        BackendPreference::Cpu => Ok(Box::new(CpuBackend::new())),
        BackendPreference::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(Box::new(CudaBackend::new()?))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(RingDbError::BackendUnavailable(
                    "CUDA backend requires building ring-db with the `cuda` feature".to_string(),
                ))
            }
        }
    }
}

fn validate_query_input(dims: usize, query: &[f32], weights: Option<&[f32]>) -> Result<()> {
    if query.len() != dims {
        return Err(RingDbError::DimensionMismatch {
            expected: dims,
            got: query.len(),
        });
    }
    if let Some((index, value)) = query
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(RingDbError::InvalidQuery(format!(
            "query coordinate at index {index} must be finite, got {value}"
        )));
    }

    let Some(weights) = weights else {
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

fn validate_distance(value: f32, label: &str) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        return Ok(());
    }
    Err(RingDbError::InvalidQuery(format!(
        "{label} must be finite and non-negative, got {value}"
    )))
}

fn candidate_view(
    candidates: Option<&CandidateSet>,
    n_vectors: usize,
) -> Result<CandidateSetView<'_>> {
    match candidates {
        Some(candidates) => {
            if candidates.universe_len() != n_vectors {
                return Err(RingDbError::CandidateSetMismatch {
                    expected: n_vectors,
                    got: candidates.universe_len(),
                });
            }
            Ok(candidates.view())
        }
        None => Ok(CandidateSetView::All),
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
/// used for filtering is supplied explicitly as integer tags; query backends
/// never interpret the payload layout.
pub struct RingDb<T: Pod + Send + Sync = ()> {
    config: RingDbConfig,
    backend: Box<dyn RingComputeBackend>,
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
            backend: make_backend(config.backend_preference)?,
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
        if vector.len() != self.config.dims {
            return Err(RingDbError::DimensionMismatch {
                expected: self.config.dims,
                got: vector.len(),
            });
        }
        if self.n_vectors >= u32::MAX as usize {
            return Err(RingDbError::InvalidQuery(
                "ring-db supports at most u32::MAX vectors".to_string(),
            ));
        }
        if let Some((index, value)) = vector
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(RingDbError::InvalidQuery(format!(
                "vector coordinate at index {index} must be finite, got {value}"
            )));
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
            mut backend,
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

        backend.upload_f32_dataset(config.dims, vectors)?;
        if let Some((path, meta)) = persist_meta {
            write_meta(&path, meta)?;
        }
        Ok(SealedRingDb {
            config,
            backend,
            n_vectors,
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
        let mut backend = make_backend(backend_preference)?;
        backend.upload_f32_dataset(meta.dims, vectors)?;

        Ok(SealedRingDb {
            config: RingDbConfig::new(meta.dims)
                .with_persist_dir(dir)
                .with_backend_preference(backend_preference),
            backend,
            n_vectors: meta.n_vectors,
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
        self.backend.name()
    }
}

/// Immutable database returned by [`RingDb::build`] or [`RingDb::load`].
pub struct SealedRingDb<T: Pod + Send + Sync = ()> {
    config: RingDbConfig,
    backend: Box<dyn RingComputeBackend>,
    n_vectors: usize,
    payloads: Box<[T]>,
    metadata_index: MetadataIndex,
}

impl<T: Pod + Send + Sync> SealedRingDb<T> {
    pub fn query(&self, query: &RingQuery<'_>) -> Result<QueryResult> {
        let started = Instant::now();
        validate_query_input(self.dims(), query.query, query.weights)?;
        validate_distance(query.d, "ring target distance")?;
        validate_distance(query.lambda, "ring half-width")?;
        let candidates = candidate_view(query.candidates, self.n_vectors)?;
        let hits = self.backend.ring_query_f32(
            self.dims(),
            query.query,
            query.weights,
            (query.d - query.lambda).max(0.0),
            query.d + query.lambda,
            candidates,
        )?;
        Ok(self.result(hits, started))
    }

    pub fn query_range(&self, query: &RangeQuery<'_>) -> Result<QueryResult> {
        let started = Instant::now();
        validate_query_input(self.dims(), query.query, query.weights)?;
        validate_distance(query.d_min, "range lower bound")?;
        validate_distance(query.d_max, "range upper bound")?;
        if query.d_min > query.d_max {
            return Err(RingDbError::InvalidQuery(format!(
                "range lower bound {} exceeds upper bound {}",
                query.d_min, query.d_max
            )));
        }
        let candidates = candidate_view(query.candidates, self.n_vectors)?;
        let hits = self.backend.ring_query_f32(
            self.dims(),
            query.query,
            query.weights,
            query.d_min,
            query.d_max,
            candidates,
        )?;
        Ok(self.result(hits, started))
    }

    pub fn query_disk(&self, query: &DiskQuery<'_>) -> Result<QueryResult> {
        let started = Instant::now();
        validate_query_input(self.dims(), query.query, query.weights)?;
        validate_distance(query.d_max, "disk radius")?;
        let candidates = candidate_view(query.candidates, self.n_vectors)?;
        let hits = self.backend.disk_query_f32(
            self.dims(),
            query.query,
            query.weights,
            query.d_max,
            candidates,
        )?;
        Ok(self.result(hits, started))
    }

    pub fn query_disk_intersection(
        &self,
        query: &DiskIntersectionQuery<'_>,
    ) -> Result<QueryResult> {
        let started = Instant::now();
        if query.disks.is_empty() {
            return Err(RingDbError::InvalidQuery(
                "disk intersection requires at least one disk".to_string(),
            ));
        }

        let mut effective_candidates = query.candidates.cloned();
        for disk in query.disks {
            validate_query_input(self.dims(), disk.query, disk.weights)?;
            validate_distance(disk.d_max, "disk radius")?;
            if let Some(candidates) = disk.candidates {
                candidate_view(Some(candidates), self.n_vectors)?;
                effective_candidates = Some(match effective_candidates {
                    Some(current) => current.intersection(candidates)?,
                    None => candidates.clone(),
                });
            }
        }
        let candidates = candidate_view(effective_candidates.as_ref(), self.n_vectors)?;
        let disks: Vec<_> = query
            .disks
            .iter()
            .map(|disk| DiskSpec {
                query: disk.query,
                d_max: disk.d_max,
                weights: disk.weights,
            })
            .collect();
        let hits = self
            .backend
            .disk_intersection_query_f32(self.dims(), &disks, candidates)?;
        Ok(self.result(hits, started))
    }

    /// Execute an exact nearest-neighbor query.
    pub fn query_knn(&self, query: &KnnQuery<'_>) -> Result<QueryResult> {
        let started = Instant::now();
        validate_query_input(self.dims(), query.query, query.weights)?;
        let candidates = candidate_view(query.candidates, self.n_vectors)?;
        let hits = self.backend.knn_query_f32(
            self.dims(),
            query.query,
            query.weights,
            query.k,
            candidates,
        )?;
        Ok(self.result(hits, started))
    }

    /// Return the vectors carrying exactly this metadata tag.
    pub fn metadata_eq(&self, key: MetadataKey, value: MetadataValue) -> CandidateSet {
        self.metadata_index.candidates_eq(key, value)
    }

    /// Create a validated candidate set from vector IDs.
    pub fn candidates_from_ids(&self, ids: impl IntoIterator<Item = u32>) -> Result<CandidateSet> {
        CandidateSet::from_ids(self.n_vectors, ids)
    }

    /// Fetch one Pod payload by vector ID without copying it.
    pub fn fetch_payload(&self, id: u32) -> Result<&T> {
        self.payloads
            .get(id as usize)
            .ok_or(RingDbError::InvalidVectorId {
                id,
                len: self.n_vectors,
            })
    }

    /// Fetch Pod payload references in the same order as `ids`.
    pub fn fetch_payloads(&self, ids: &[u32]) -> Result<Vec<&T>> {
        ids.iter().map(|&id| self.fetch_payload(id)).collect()
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
        self.backend.name()
    }

    fn result(&self, hits: Vec<crate::query::Hit>, started: Instant) -> QueryResult {
        QueryResult {
            hits,
            backend_used: self.backend.name(),
            elapsed: started.elapsed(),
        }
    }
}
