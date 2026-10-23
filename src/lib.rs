//! # ringDB
//!
//! An exact vector database for geometric and nearest-neighbor queries in
//! high-dimensional spaces.
//!
//! ringDB has two exact search primitives: k-nearest neighbors ([`Knn`]) and
//! regions ([`Region`]), the intersection of one or more [`Disk`]s. Each query
//! or disk uses ordinary L2 or diagonal weighted L2 ([`Metric`]).
//!
//! CPU and Vulkan backends are included without Cargo feature flags. CPU is the
//! default and needs no GPU runtime. Select [`BackendPreference::Vulkan`] to load
//! Vulkan at runtime; only a compatible loader, driver, and compute device are needed.
//! Shaders are embedded, so no Vulkan SDK or shader compiler is required.
//!
//! ## Quick start
//!
//! ```
//! use ringdb::{Disk, Knn, RingDb, RingDbConfig};
//!
//! let mut db = RingDb::new(RingDbConfig::new(4)).unwrap();
//! db.add_vector(&[1.0f32, 0.0, 0.0, 0.0], ()).unwrap();
//! db.add_vector(&[0.0, 5.0, 0.0, 0.0], ()).unwrap();
//!
//! let db = db.build().unwrap();
//! let origin = [0.0f32; 4];
//! let nearest = db.search(Knn::new(&origin, 1)).unwrap();
//! assert_eq!(nearest[0].dist_sq, 1.0);
//! let inside = db.search(Disk::new(&origin, 2.0)).unwrap();
//! assert_eq!(inside.len(), 1);
//! ```
mod backend;
mod config;
mod engine;
mod error;
mod metadata;
mod payload;
mod persist;
mod query;

// Compile README examples with the public API as part of `cargo test`.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme {}

pub use config::{BackendPreference, RingDbConfig};
pub use engine::{RingDb, SealedRingDb};
pub use error::RingDbError;
pub use metadata::{CandidateSet, MetadataKey, MetadataTag, MetadataValue};
pub use query::{Disk, Hit, Key, Knn, MAX_DISKS, Metric, Query, Region, RegionHit, RegionResult};
