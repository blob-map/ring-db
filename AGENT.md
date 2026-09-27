# AGENT.md — ring-db

Quick context for contributors and coding agents.

## Product definition

ring-db is a Rust 2024 exact vector database. It supports:

- ring, range, disk, and disk-intersection searches;
- exact K-nearest-neighbor search;
- ordinary L2 and per-query diagonal weighted L2;
- explicit integer metadata tags compiled into candidate sets;
- fixed-size Pod payload records with zero-copy reads;
- CPU and Vulkan backends selected at runtime;
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
6. Do not make compute kernels interpret user payload layouts.
7. Add functionality only for a demonstrated use case.

## Documentation standard

Always write `README.md` in [ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/).
Use [Issue 9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf), including its dictionary.
Use approved meanings and parts of speech. Keep instructions active and imperative.
Limit instruction sentences to 20 words and descriptions to 25 words.
Give each paragraph one topic and no more than six sentences.
Use one instruction per sentence, consistent terms, and American English spelling.
Review meanings as well as sentence length; automated checks cannot establish compliance.

Project technical nouns include backend, candidate set, Cargo, database, dimension,
driver, filter, GPU, ID, KNN, loader, metadata, payload, Pod, query, shader, and vector.
Computer-operation verbs include allocate, build, compile, deserialize, execute,
filter, insert, load, query, return, serialize, sort, update, and upload.
Keep code identifiers and commands exact. Explain new technical terms at first use.
Keep the README focused on current usage. Do not add migration files unless requested.

README Rust examples are doctests. `cargo test` compiles all examples and executes
those without `no_run`. Use complete examples without hidden rustdoc-only lines.
Keep release versions and commands consistent with `Cargo.toml` and the CLI.

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
    └── vulkan/        Vulkan backend, runtime, and embedded SPIR-V shaders

tests/
├── correctness.rs
├── metadata_knn.rs
├── random.rs
├── weighted.rs
└── vulkan.rs

examples/cli.rs        filtered weighted-KNN example
benches/query_backends.rs
benches/backend_compare.rs sequential CPU/Vulkan comparison with raw timing output
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
also accept `.with_weights(&weights)` directly. Each `DiskQuery` borrows its optional weights
and candidate set. The engine intersects candidate sets attached to individual disks with
the outer intersection query's candidate set before backend dispatch.

`k = 0` returns no hits. If fewer than `k` candidates exist, KNN returns all of them.
Ties are deterministic by vector ID.

The metrics are:

```text
ordinary distance: dist_sq = sum((vector[i] - query[i])^2)
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

## Vulkan backend

CPU and Vulkan are compiled unconditionally. There are no crate features. Select
`BackendPreference::Cpu` (the default) or `BackendPreference::Vulkan` at runtime.
Vulkan uses `ash` with a dynamically loaded loader, opened only when Vulkan is
selected. CPU must work without a Vulkan loader or GPU driver. Explicit Vulkan
selection returns `BackendUnavailable` if no compatible runtime/device is found.
No Vulkan SDK or shader compiler is needed by consumers.

`src/backend/vulkan/mod.rs` implements the existing backend trait. `runtime.rs`
owns Vulkan resources and synchronous submissions; its execution mutex serializes
the queue, command pool, descriptor set, and reusable query buffers shared by
geometric and KNN queries.
Geometric queries copy the count in the scan submission and read hits directly
into an owned Pod `Vec<Hit>`. Buffer capacities grow on demand and remain allocated
until the backend is dropped; never expose references to this reusable storage.
Large host copies use up to four workers, including the caller, with scoped
helper threads. Do not use the caller's Rayon pool while holding the execution
mutex: work stealing could re-enter the backend. Join all helpers before unmapping
or reusing readback memory, including when spawning a helper fails.
Vectors are uploaded transposed once.
The three scan pipelines specialize All/Sparse/Dense candidate handling. Small-K
selection uses GPU reductions without 64-bit atomics or subgroup requirements.
Large-K queries may select on the host. Preserve deterministic `(dist_sq, id)` order
and zero-weight handling before coordinate subtraction.

`RINGDB_VULKAN_TIMINGS=1` enables optional GPU timestamps around each submission.
These measure the GPU command-buffer span, not full query latency. Keep profiling
disabled for ordinary benchmarks and check compute-queue timestamp support when
opting in. Do not assume a shader specialization improves performance: compare it
against the generic path across dimensions, weights, candidates, and selectivity.

Shader sources and precompiled SPIR-V live together in `shaders/`. Regenerate them
with `sh src/backend/vulkan/shaders/compile.sh` after every source change. This
development command requires `glslc` and `spirv-val`; Cargo embeds the binaries.
Buffers must fit `maxStorageBufferRange`; the backend does not chunk datasets.

Run device tests with `RINGDB_VULKAN_TEST=1 cargo test --test vulkan`.
An explicitly requested device test must fail if Vulkan is unavailable. CI uses
Mesa software Vulkan so shader execution is covered without a physical GPU.

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
cargo clippy --all-targets -- -D warnings
cargo test

cargo build --release

RUSTFLAGS="-C target-cpu=native" cargo bench --bench query_backends
RINGDB_BENCH_N=100000 cargo bench --bench query_backends cpu_knn_f32
RINGDB_BENCH_N=100000 cargo bench --bench query_backends metadata
cargo bench --bench backend_compare -- --n 10000000 --dims 64

cargo run --release --example cli -- --n 100000 --dims 64 --k 10 --category 3
```

Vulkan tests that need a device are gated by `RINGDB_VULKAN_TEST=1`; the missing-driver
regression always runs and also verifies CPU queries. Vulkan Criterion benchmarks
should skip cleanly when the runtime or device is unavailable. `RINGDB_BENCH_N` sets
both backends’ dataset size; `RINGDB_VULKAN_BENCH_N` overrides Vulkan only.

## Dependency roles

| Dependency | Role |
|---|---|
| `rayon` | CPU data parallelism |
| `thiserror` | error definitions |
| `bytemuck` | Pod validation and byte casts |
| `ash` | Vulkan bindings and runtime loader |
| `criterion` | benchmarks |
| `rand` | deterministic benchmark/test data |
| `clap` | example CLI parsing |
| `tempfile` | isolated persistence tests |

The release publishes one crate. Keep release scripts and CI aligned with the package
version. Query semantics and persistence format version 1 do not depend on the backend.
