//! # ringDB
//!
//! An exact vector database for geometric and nearest-neighbor queries in
//! high-dimensional spaces.
//!
//! ringDB supports exact ring/range/disk searches and exact K-nearest-neighbor
//! search. Queries use ordinary L2 by default and can optionally provide final
//! per-dimension coefficients for diagonal weighted L2.
//!
//! ## Quick start
//!
//! ```
//! use ringdb::{RingDb, RingDbConfig, RingQuery};
//!
//! let mut db = RingDb::new(RingDbConfig::new(4)).unwrap();
//! db.add_vector(&[1.0f32, 0.0, 0.0, 0.0], ()).unwrap();
//! db.add_vector(&[0.0, 5.0, 0.0, 0.0], ()).unwrap();
//!
//! let db = db.build().unwrap();
//! let result = db.query(&RingQuery::new(&[0.0f32; 4], 1.0, 0.1)).unwrap();
//! // result.hits contains all matching vectors with their squared distances
//! ```
mod backend;
mod config;
mod engine;
mod error;
mod metadata;
mod payload;
mod persist;
mod query;

pub use config::{BackendPreference, RingDbConfig};
pub use engine::{RingDb, SealedRingDb};
pub use error::RingDbError;
pub use metadata::{CandidateSet, MetadataKey, MetadataTag, MetadataValue};
pub use query::{
    DiskIntersectionQuery, DiskQuery, Hit, KnnQuery, QueryResult, RangeQuery, RingQuery,
};
