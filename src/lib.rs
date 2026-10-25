//! # ringDB
//!
//! An exact vector database with typed, filterable fields.
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
//! use ringdb::{Db, Disk, FieldType, Filter, Knn, Options, Record, Schema};
//!
//! let schema = Schema::new(2)
//!     .field("city", FieldType::Keyword)
//!     .field("rooms", FieldType::U32);
//! let db = Db::in_memory(schema, Options::default()).unwrap();
//! let near = db.add(Record::new([1.0, 0.0]).set("city", "Lyon").set("rooms", 3)).unwrap();
//! db.add(Record::new([0.0, 5.0]).set("city", "Paris").set("rooms", 2)).unwrap();
//! db.flush().unwrap();
//!
//! let origin = [0.0, 0.0];
//! let nearest = db.search(Knn::new(&origin, 1)).unwrap();
//! assert_eq!((nearest[0].key, nearest[0].dist_sq), (near, 1.0));
//! let lyon = Filter::eq("city", "Lyon");
//! let inside = db.search(Disk::new(&origin, 10.0).filter(&lyon)).unwrap();
//! assert_eq!(inside.keys().collect::<Vec<_>>(), [near]);
//! assert_eq!(db.get(near).unwrap().get("rooms").and_then(|v| v.as_u32()), Some(3));
//! ```
mod backend;
mod codec;
mod column;
mod config;
mod db;
mod error;
mod filter;
mod query;
mod rows;
mod schema;
mod segment;
mod snapshot;
mod store;
mod value;

// Compile README examples with the public API as part of `cargo test`.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme {}

pub use config::{BackendPreference, Options};
pub use db::{Batch, Db};
pub use error::RingDbError;
pub use filter::Filter;
pub use query::{Disk, Hit, Key, Knn, MAX_DISKS, Metric, Query, Region, RegionHit, RegionResult};
pub use schema::{FieldType, Schema};
pub use snapshot::Snapshot;
pub use value::{Document, Record, Value};
