use std::path::Path;
use std::time::Instant;

use crate::BackendPreference;
#[cfg(feature = "cuda")]
use crate::backend::CudaBackend;
use crate::backend::{CpuBackend, RingComputeBackend};
use crate::config::RingDbConfig;
use crate::error::{Result, RingDbError};
use crate::payload::{OwnedPayloadStore, Payload, PayloadBuilderOps, RefPayloadStore};
use crate::persist::{read_f32_file, read_meta, write_f32_file, write_meta};
use crate::query::Hit;
use crate::query::{DiskIntersectionQuery, DiskQuery, QueryResult, RangeQuery, RingQuery};

fn into_hits(responses: Vec<crate::backend::QueryResponse>) -> Vec<Hit> {
    responses
        .into_iter()
        .map(|r| Hit {
            id: r.id,
            dist_sq: r.dist_sq,
        })
        .collect()
}

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
                    "CUDA backend requires building ring-db with the `cuda` Cargo feature"
                        .to_string(),
                ))
            }
        }
    }
}

// ─── RingDb (builder) ────────────────────────────────────────────────────────

/// Builder for a ring-query vector database.
///
/// Insert vectors with their associated payloads via
/// [`add_vector()`](Self::add_vector), then call [`build()`](Self::build)
/// to obtain a [`SealedRingDb`].
///
/// `T` must implement [`Payload`], which is derived with `#[derive(Payload)]`.
/// Use `T = ()` when no payload is needed.
///
/// # Example — no payload
///
/// ```
/// use ringdb::{RingDb, RingDbConfig, RingQuery};
///
/// let mut db = RingDb::new(RingDbConfig::new(4)).unwrap();
/// db.add_vector(&[1.0, 0.0, 0.0, 0.0], ()).unwrap();
/// db.add_vector(&[0.0, 1.0, 0.0, 0.0], ()).unwrap();
///
/// let db = db.build().unwrap();
/// let result = db.query(&RingQuery { query: &[1.0f32, 0.0, 0.0, 0.0], d: 1.0, lambda: 0.1 }).unwrap();
/// println!("hits: {:?}", result.ids());
/// ```
pub struct RingDb<T: Payload = ()> {
    config: RingDbConfig,
    backend: Box<dyn RingComputeBackend>,
    n_vectors: usize,

    /// Staging buffer: f32 vectors, row-major, `n_vectors × dims`.
    vectors: Vec<f32>,

    /// Staging buffer: per-vector squared L2 norm.
    norms_sq: Vec<f32>,

    /// Concrete builder — `SerdeStoreBuilder<T>` or `PodStoreBuilder<T>`,
    /// determined at construction time by `T::make_builder()`.
    /// No heap indirection; lives directly in the struct.
    payload_builder: T::Builder,
}

impl<T: Payload> RingDb<T> {
    /// Create a new empty `RingDb`.
    ///
    /// The storage strategy (Serde or Pod) is determined entirely by `T`'s
    /// `#[derive(Payload)]` — no second constructor needed.
    ///
    /// # Example — with Serde payload
    ///
    /// ```
    /// use ringdb::{RingDb, RingDbConfig, RingQuery, Payload};
    /// use serde::{Serialize, Deserialize};
    ///
    /// #[derive(Serialize, Deserialize, Payload)]
    /// struct Meta { label: String }
    ///
    /// let mut db: RingDb<Meta> = RingDb::new(RingDbConfig::new(2)).unwrap();
    /// db.add_vector(&[1.0, 0.0], Meta { label: "dog".into() }).unwrap();
    /// db.add_vector(&[0.0, 1.0], Meta { label: "cat".into() }).unwrap();
    ///
    /// let db = db.build().unwrap();
    /// let result = db.query(&RingQuery { query: &[1.0f32, 0.0], d: 1.0, lambda: 0.1 }).unwrap();
    /// let payloads = db.fetch_payloads(&result.ids()).unwrap();
    /// ```
    pub fn new(config: RingDbConfig) -> Result<Self> {
        let backend = make_backend(config.backend_preference)?;
        Ok(Self {
            config,
            backend,
            n_vectors: 0,
            vectors: Vec::new(),
            norms_sq: Vec::new(),
            payload_builder: T::make_builder()?,
        })
    }

    /// Insert a single vector and its associated payload.
    ///
    /// Vectors are assigned sequential IDs starting from 0.
    /// The slice length must equal `dims`.
    pub fn add_vector(&mut self, vector: &[f32], payload: T) -> Result<()> {
        let dims = self.config.dims;
        if vector.len() != dims {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: vector.len(),
            });
        }
        let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
        self.norms_sq.push(norm_sq);
        self.vectors.extend_from_slice(vector);
        self.payload_builder.push(payload)?;
        self.n_vectors += 1;
        Ok(())
    }

    /// Seal the database.
    ///
    /// Transfers vectors to the compute backend and flushes the payload builder
    /// to its mmap. If [`RingDbConfig::persist_dir`] is set, all data is also
    /// written to disk (reload with [`RingDb::load`]).
    pub fn build(self) -> Result<SealedRingDb<T>> {
        let RingDb {
            config,
            mut backend,
            vectors,
            norms_sq,
            payload_builder,
            n_vectors,
        } = self;
        let dims = config.dims;

        if let Some(dir) = config.persist_dir.clone() {
            std::fs::create_dir_all(&dir)?;
            write_meta(&dir.join("meta.bin"), dims, n_vectors)?;
            write_f32_file(&dir.join("vectors.bin"), &vectors)?;
            write_f32_file(&dir.join("norms_sq.bin"), &norms_sq)?;
            let payload_store = payload_builder
                .finish_persisted(&dir.join("payloads.bin"), &dir.join("offsets.bin"))?;
            backend.upload_f32_dataset(dims, vectors, norms_sq)?;
            return Ok(SealedRingDb {
                config,
                backend,
                n_vectors,
                payload_store,
            });
        }

        backend.upload_f32_dataset(dims, vectors, norms_sq)?;
        let payload_store = payload_builder.finish()?;
        Ok(SealedRingDb {
            config,
            backend,
            n_vectors,
            payload_store,
        })
    }

    /// Reconstruct a [`SealedRingDb`] from a directory previously written by
    /// [`build()`](Self::build) with a persist dir configured.
    ///
    /// The correct store variant is selected automatically based on `T`'s
    /// `Payload` impl — no separate `load_pod` method needed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ringdb::{RingDb, RingDbConfig, BackendPreference};
    /// use std::path::Path;
    ///
    /// // --- save ---
    /// let mut db = RingDb::<()>::new(RingDbConfig::new(4).with_persist_dir("/tmp/mydb")).unwrap();
    /// db.add_vector(&[1.0, 0.0, 0.0, 0.0], ()).unwrap();
    /// let _sealed = db.build().unwrap();
    ///
    /// // --- load ---
    /// let loaded = RingDb::<()>::load(Path::new("/tmp/mydb"), BackendPreference::Cpu).unwrap();
    /// ```
    pub fn load(
        dir: &Path,
        backend_preference: crate::config::BackendPreference,
    ) -> Result<SealedRingDb<T>> {
        let (dims, n_vectors) = read_meta(&dir.join("meta.bin"))?;
        let vectors = read_f32_file(&dir.join("vectors.bin"))?;
        let norms_sq = read_f32_file(&dir.join("norms_sq.bin"))?;

        let expected = n_vectors * dims;
        if vectors.len() != expected {
            return Err(RingDbError::Corrupt(format!(
                "vectors.bin has {} f32 values, expected {}",
                vectors.len(),
                expected,
            )));
        }
        if norms_sq.len() != n_vectors {
            return Err(RingDbError::Corrupt(format!(
                "norms_sq.bin has {} f32 values, expected {}",
                norms_sq.len(),
                n_vectors,
            )));
        }

        let mut backend = make_backend(backend_preference)?;
        backend.upload_f32_dataset(dims, vectors, norms_sq)?;

        let payload_store = T::load_store(dir)?;

        Ok(SealedRingDb {
            config: RingDbConfig::new(dims)
                .with_persist_dir(dir)
                .with_backend_preference(backend_preference),
            backend,
            n_vectors,
            payload_store,
        })
    }

    /// Number of vectors currently staged.
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Returns `true` if no vectors have been inserted.
    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Number of dimensions per vector.
    pub fn dims(&self) -> usize {
        self.config.dims
    }

    /// Name of the backend currently in use.
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }
}

// ─── SealedRingDb ────────────────────────────────────────────────────────────

/// Sealed (immutable) ring-query database.
///
/// Obtained by calling [`RingDb::build()`] or [`RingDb::load()`].
///
/// The hot side (vectors + norms) is owned by the compute backend.
/// The cold side (payloads) lives in a file-backed mmap via `T::Store`.
pub struct SealedRingDb<T: Payload = ()> {
    config: RingDbConfig,
    backend: Box<dyn RingComputeBackend>,
    n_vectors: usize,
    payload_store: T::Store,
}

impl<T: Payload> SealedRingDb<T> {
    // ── Query methods ─────────────────────────────────────────────────────────

    /// Execute a ring query and return matching vector IDs.
    pub fn query(&self, q: &RingQuery<'_>) -> Result<QueryResult> {
        let dims = self.config.dims;
        if q.query.len() != dims {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: q.query.len(),
            });
        }
        let t = Instant::now();
        let hits = into_hits(self.backend.ring_query_f32(
            dims,
            q.query,
            (q.d - q.lambda).max(0.0),
            q.d + q.lambda,
        )?);
        Ok(QueryResult {
            hits,
            backend_used: self.backend.name(),
            elapsed: t.elapsed(),
        })
    }

    /// Execute a range query: all vectors with distance in `[d_min, d_max]`.
    pub fn query_range(&self, q: &RangeQuery<'_>) -> Result<QueryResult> {
        let dims = self.config.dims;
        if q.query.len() != dims {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: q.query.len(),
            });
        }
        let t = Instant::now();
        let hits = into_hits(
            self.backend
                .ring_query_f32(dims, q.query, q.d_min, q.d_max)?,
        );
        Ok(QueryResult {
            hits,
            backend_used: self.backend.name(),
            elapsed: t.elapsed(),
        })
    }

    /// Execute a disk query: all vectors within radius `d_max`.
    pub fn query_disk(&self, q: &DiskQuery<'_>) -> Result<QueryResult> {
        let dims = self.config.dims;
        if q.query.len() != dims {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: q.query.len(),
            });
        }
        let t = Instant::now();
        let hits = into_hits(self.backend.disk_query_f32(dims, q.query, q.d_max)?);
        Ok(QueryResult {
            hits,
            backend_used: self.backend.name(),
            elapsed: t.elapsed(),
        })
    }

    /// Execute a disk-intersection query: all vectors inside every disk.
    ///
    /// Returned hit distances are measured against the first disk.
    pub fn query_disk_intersection(&self, q: &DiskIntersectionQuery<'_>) -> Result<QueryResult> {
        let dims = self.config.dims;
        if q.disks.is_empty() {
            return Err(RingDbError::InvalidQuery(
                "disk intersection requires at least one disk".to_string(),
            ));
        }
        for disk in q.disks {
            if disk.query.len() != dims {
                return Err(RingDbError::DimensionMismatch {
                    expected: dims,
                    got: disk.query.len(),
                });
            }
        }

        let disks: Vec<(&[f32], f32)> = q
            .disks
            .iter()
            .map(|disk| (disk.query, disk.d_max))
            .collect();
        let t = Instant::now();
        let hits = into_hits(self.backend.disk_intersection_query_f32(dims, &disks)?);
        Ok(QueryResult {
            hits,
            backend_used: self.backend.name(),
            elapsed: t.elapsed(),
        })
    }

    // ── Serde payload fetch ───────────────────────────────────────────────────

    /// Fetch and deserialize the payload for a single vector ID.
    pub fn fetch_payload(&self, id: u32) -> Result<T> {
        self.payload_store.fetch_owned(id)
    }

    /// Fetch and deserialize payloads for a slice of IDs, in order.
    pub fn fetch_payloads(&self, ids: &[u32]) -> Result<Vec<T>> {
        self.payload_store.fetch_many_owned(ids)
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Number of vectors stored.
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Returns `true` if the database contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Number of dimensions per vector.
    pub fn dims(&self) -> usize {
        self.config.dims
    }

    /// Name of the backend currently in use.
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }
}

// ── Pod fetch — only when T::Store: RefPayloadStore<T> ───────────────────────
//
// This impl block is only available for types whose `#[derive(Payload)]`
// chose `storage = "pod"`. For Serde types, `T::Store = SerdeStore<T>` which
// does NOT implement `RefPayloadStore<T>`, so these methods simply don't exist.
// The compiler enforces this statically — no runtime error possible.

impl<T: Payload> SealedRingDb<T>
where
    T::Store: RefPayloadStore<T>,
{
    /// Fetch a zero-copy reference to the payload for a single vector ID.
    ///
    /// Returns a `&T` pointing directly into the mmap — O(1), no allocation,
    /// no deserialization. Only available for `#[payload(storage = "pod")]` types.
    pub fn fetch_pod(&self, id: u32) -> &T {
        self.payload_store.fetch_ref(id)
    }

    /// Fetch zero-copy references to payloads for a slice of IDs, in order.
    pub fn fetch_pods<'a>(&'a self, ids: &[u32]) -> Vec<&'a T> {
        self.payload_store.fetch_many_ref(ids)
    }
}
