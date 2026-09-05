# AGENT.md — ring-db

Quick context for contributors and coding agents.

## Product definition

ring-db is a Rust 2024 exact vector database. It supports:

- ring, range, disk, and disk-intersection searches;
- exact K-nearest-neighbor search;
- ordinary L2 and per-query diagonal weighted L2;
- explicit integer metadata tags compiled into candidate sets;
- fixed-size Pod payload records with zero-copy reads;
- CPU and optional CUDA backends;
- versioned persistence.

The project intentionally avoids general document-database behavior. Payload records are
opaque to the search engine. Metadata filtering is equality-based and independent of the
payload layout.

## Engineering priorities

1. Prefer the simplest complete design.
2. Remove abstractions that no longer serve more than one real implementation.
3. Keep cyclomatic complexity low with small preparation and execution helpers.
4. Keep validation outside hot coordinate loops.
5. Preserve branch-free unfiltered paths.
6. Do not make CPU or CUDA kernels interpret user payload layouts.
7. Add functionality only for a demonstrated use case.

## Repository map

```text
src/
├── lib.rs             public exports
├── config.rs          RingDbConfig and BackendPreference
├── engine.rs          RingDb builder and immutable SealedRingDb
├── error.rs           RingDbError
├── metadata.rs        tags, adaptive CandidateSet, metadata persistence
├── payload.rs         fixed-size Pod persistence helpers
├── persist.rs         database header and vector-file I/O
├── query.rs           query types, Hit, and QueryResult
└── backend/
    ├── mod.rs         RingComputeBackend contract and DiskSpec
    ├── cpu.rs         Rayon-parallel exact reference backend
    └── cuda.rs        optional CUDA driver backend and embedded PTX

tests/
├── correctness.rs
├── metadata_knn.rs
├── random.rs
└── weighted.rs

examples/cli.rs        filtered weighted-KNN example
benches/query_backends.rs
```

## Public data model

`RingDb<T>` and `SealedRingDb<T>` require `T: bytemuck::Pod + Send + Sync` and
default to `T = ()`.

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Record {
    category_id: u32,
    score: f32,
}
```

Payload records are raw fixed-size result data. They cannot contain pointers or
heap-backed fields. Use integer interning or bounded byte arrays for strings.

Insertion has two forms:

```rust
db.add_vector(&vector, record)?;
db.add_vector_with_metadata(&vector, record, &tags)?;
```

IDs are sequential `u32` values in insertion order. `add_vector_with_metadata` rejects a
new row once the ID space is exhausted.

After sealing, payload access is always zero-copy and bounds checked:

```rust
let record: &Record = db.fetch_payload(id)?;
let records: Vec<&Record> = db.fetch_payloads(&ids)?;
```

## Metadata

The public primitives are:

```rust
MetadataKey::new(u32)
MetadataValue::new(u64)
MetadataTag::new(key, value)
```

`MetadataIndexBuilder` accumulates posting IDs while vectors are inserted.
`MetadataIndex::candidates_eq` produces a database-sized `CandidateSet`. Missing tags
produce an empty set, not an error. Duplicate tags on one vector do not duplicate IDs.

Candidate sets support:

- `intersection(&other) -> Result<CandidateSet>`;
- `union(&other) -> Result<CandidateSet>`;
- `complement() -> CandidateSet`;
- validated construction through `SealedRingDb::candidates_from_ids`;
- `len`, `is_empty`, `universe_len`, and checked membership.

Intersection and union require equal universe lengths. Queries also reject candidate sets
whose universe length differs from the database row count.

The representation is selected internally:

- `All` and `Empty` are allocation-free special cases;
- `Sparse` is a sorted, unique `u32` ID list;
- `Dense` is a bitset.

Keep this representation private. Backends consume the borrowed `CandidateSetView`.

## Query semantics

| Type | Meaning |
|---|---|
| `RingQuery` | inclusive `[max(0, d - lambda), d + lambda]` |
| `RangeQuery` | inclusive `[d_min, d_max]` |
| `DiskQuery` | inclusive `[0, d_max]` |
| `DiskIntersectionQuery` | vector must satisfy every disk |
| `KnnQuery` | exact best `k`, ordered by `(dist_sq, id)` |

All query builders accept `.with_candidates(&set)`. Ring, range, disk, and KNN queries
also accept `.with_weights(&weights)` directly. Each `DiskQuery` owns its optional weights
and candidate set. The engine intersects candidate sets attached to individual disks with
the outer intersection query's candidate set before backend dispatch.

`k = 0` returns no hits. If fewer than `k` candidates exist, KNN returns all of them.
Ties are deterministic by vector ID.

The metrics are:

```text
ordinary distance: dist_sq = sum((vector[i] - query[i])^2)
ordinary KNN:    dist_sq = sum((vector[i] - query[i])^2)
weighted:        dist_sq = sum(weights[i] * (vector[i] - query[i])^2)
```

Weights are query-only final coefficients. They must match `dims`, be finite, and be
non-negative. They are never persisted.

## Engine lifecycle

`RingDb<T>` stages row-major vectors, Pod records, and metadata postings.
`build()` finishes immutable stores and uploads vector data once. `SealedRingDb<T>` owns:

- `Box<dyn RingComputeBackend>` for hot vector data;
- one private, aligned `Box<[T]>` for payloads;
- `MetadataIndex` for candidate construction.

The sealed database is immutable and intended for repeated queries.

## CPU backend

The CPU backend is the correctness reference.

- Ordinary searches use direct coordinate deltas with four independent accumulators.
- Unweighted KNN accumulates coordinate deltas directly to avoid cancellation.
- Weighted distance has its own four-accumulator loop.
- Rayon partitions vector or candidate work.
- `All` retains the existing scan with no candidate branch per row.
- `Sparse` iterates IDs directly.
- `Dense` tests its bit before distance calculation.
- KNN keeps a bounded max-heap per Rayon fold and deterministically merges/sorts the
  finalists.

Do not replace the specialized unfiltered paths with a universal per-row predicate.

## CUDA backend

CUDA is behind the `cuda` feature. The implementation dynamically loads the NVIDIA
driver, creates a context on device 0, uploads vectors in transposed layout, and keeps
query execution behind a launch mutex. CPU-only builds have no link-time CUDA dependency.

CUDA implements the same range-query, candidate-filtering, diagonal-weight, and exact-KNN
semantics as CPU. KNN must return ascending `(dist_sq, id)` results.

Keep CUDA behavior equivalent to CPU within documented floating-point tolerances. Avoid
host transfers proportional to the full dataset for small KNN results. Do not describe a
specific CUDA top-K strategy in public documentation unless it is part of the stable
contract.

## Persistence format

The current database format version is 1. A persisted database contains:

```text
meta.bin       magic, version, endian marker, dims, rows, Pod size/alignment
vectors.bin    row-major little-endian f32 values
payloads.bin   consecutive native-layout Pod records
metadata.bin   versioned equality-tag posting sets
```

`metadata.bin` also has format version 1. There is no offset file because every payload
has `size_of::<T>()` bytes.

Build requires an absent or empty persistence directory. It writes data files first and
`meta.bin` last, so a build that returns an error is not loadable as a complete generation.
This is not a power-loss durability guarantee.

Load validates file lengths, finite vectors, geometry, row-count limits, endianness,
metadata universe length, and payload size/alignment. The format does not encode Rust
field names, so callers must load with the exact same Pod record layout. Older database
layouts must be rebuilt.

## Errors

`RingDbError` currently includes:

- `DimensionMismatch`;
- `Io`;
- `Corrupt`;
- `InvalidQuery`;
- `InvalidVectorId`;
- `CandidateSetMismatch`;
- `BackendUnavailable`;
- `Backend`.

Keep malformed caller input distinct from corrupt persisted data and backend failures.

## Commands

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features

cargo build --release
cargo build --release --features cuda
cargo test --features cuda

RUSTFLAGS="-C target-cpu=native" cargo bench --bench query_backends
RINGDB_BENCH_N=100000 cargo bench --bench query_backends cpu_knn_f32
RINGDB_BENCH_N=100000 cargo bench --bench query_backends metadata

cargo run --release --example cli -- --n 100000 --dims 64 --k 10 --category 3
```

CUDA tests that need a device may be gated by `RINGDB_CUDA_TEST=1`. CUDA benchmarks
should skip cleanly when the driver or device is unavailable.

## Dependency roles

| Dependency | Role |
|---|---|
| `rayon` | CPU data parallelism |
| `thiserror` | error definitions |
| `bytemuck` | Pod validation and byte casts |
| `criterion` | benchmarks |
| `rand` | deterministic benchmark/test data |
| `clap` | example CLI parsing |
| `tempfile` | isolated persistence tests |

The release publishes one crate. Keep release scripts and CI aligned with the package
version and the optional `cuda` feature.
