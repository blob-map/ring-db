use std::time::Duration;

/// A single match returned by a query.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Insertion-order ID of the matching vector (first inserted = 0).
    pub id: u32,
    /// Squared Euclidean distance to the query vector. Use `.sqrt()` to get
    /// the actual distance.
    pub dist_sq: f32,
}

/// A ring query: find all vectors whose Euclidean distance to `query`
/// lies within `[d - lambda, d + lambda]`.
///
/// Internally, ringdb uses squared L2 distances to avoid computing square
/// roots. The ring bounds become:
///
/// ```text
/// lower_sq = max(0, d - lambda)²
/// upper_sq = (d + lambda)²
/// ```
pub struct RingQuery<'a> {
    /// The query vector. Must have length equal to `RingDb::dims()`.
    pub query: &'a [f32],
    /// Target distance (centre of the ring).
    pub d: f32,
    /// Half-width of the ring.
    pub lambda: f32,
}

/// A range query: find all vectors whose Euclidean distance to `query`
/// lies within `[d_min, d_max]`.
pub struct RangeQuery<'a> {
    /// The query vector. Must have length equal to `RingDb::dims()`.
    pub query: &'a [f32],
    /// Lower bound of the distance interval (inclusive). Must be ≥ 0.
    pub d_min: f32,
    /// Upper bound of the distance interval (inclusive). Must be ≥ `d_min`.
    pub d_max: f32,
}

/// A disk query: find all vectors whose Euclidean distance to `query`
/// is at most `d_max` (i.e. the full disk/ball of radius `d_max`).
///
/// This is equivalent to a [`RangeQuery`] with `d_min = 0`.
pub struct DiskQuery<'a> {
    /// The query vector. Must have length equal to `RingDb::dims()`.
    pub query: &'a [f32],
    /// Radius of the disk (inclusive upper bound on distance). Must be ≥ 0.
    pub d_max: f32,
}

/// A disk-intersection query: find all vectors inside every disk.
///
/// Each disk is a [`DiskQuery`] containing a center vector and an inclusive
/// radius. The query must contain at least one disk. Returned [`Hit::dist_sq`]
/// values are measured against the first disk's query vector.
pub struct DiskIntersectionQuery<'a> {
    /// Disks that all returned vectors must satisfy.
    pub disks: &'a [DiskQuery<'a>],
}

/// Result of a ring/range/disk query.
pub struct QueryResult {
    /// All matching vectors together with their squared distances.
    pub hits: Vec<Hit>,
    /// Name of the backend that executed the query (e.g. `"cpu"`, `"wgpu"`, `"cuda"`).
    pub backend_used: &'static str,
    /// Wall-clock time for the query (excluding dataset upload).
    pub elapsed: Duration,
}

impl QueryResult {
    /// Convenience: collect just the IDs from `hits` into a new `Vec<u32>`.
    ///
    /// Useful when calling `fetch_payloads` or `fetch_pods`, which take `&[u32]`.
    pub fn ids(&self) -> Vec<u32> {
        self.hits.iter().map(|h| h.id).collect()
    }
}
