# AGENT.md — ring-db

> Quick-start context for AI agents. Read this before touching any code.

## What is ring-db?

**ring-db** is a Rust library (published as `ringdb` on crates.io) for **ring queries** in high-dimensional vector spaces. Unlike nearest-neighbor databases, it retrieves all vectors whose Euclidean distance to a query falls within a specified distance interval `[d_min, d_max]`.

Use cases: anomaly detection, content deduplication, recommendation diversity, spatial shells (GPS bands), audio fingerprinting, security threat detection.

- Rust Edition: 2024
- License: MIT OR Apache-2.0
- Repo: https://github.com/mathias-vandaele/ring-db

---

## Directory Structure

```
ring-db/
├── src/
│   ├── lib.rs             # Public API root, re-exports, #[doc(hidden)] __private
│   ├── engine.rs          # RingDb (builder) and SealedRingDb (immutable DB)
│   ├── query.rs           # RingQuery, RangeQuery, DiskQuery, QueryResult
│   ├── config.rs          # RingDbConfig and BackendPreference
│   ├── error.rs           # RingDbError enum and Result alias
│   ├── persist.rs         # read/write helpers for binary files
│   └── backend/
│       ├── mod.rs         # RingComputeBackend trait
│       ├── cpu.rs         # CpuBackend: brute-force, Rayon-parallel, SIMD-friendly
│       └── cuda.rs        # CudaBackend: optional CUDA driver backend (`cuda` feature)
│   └── payload/
│       ├── mod.rs         # Payload trait, OwnedPayloadStore, RefPayloadStore
│       ├── traits.rs      # PayloadBuilderOps trait + open_mmap helper
│       ├── serde.rs       # SerdeStoreBuilder + SerdeStore (bincode, variable-size)
│       └── pod.rs         # PodStoreBuilder + PodStore (raw bytes, zero-copy &T)
├── derive/
│   ├── Cargo.toml         # ring-db-derive proc-macro crate
│   └── src/lib.rs         # #[derive(Payload)] — dispatches to serde or pod storage
├── examples/
│   └── cli.rs             # Standalone CLI: random dataset + ring query demo
├── benches/
│   └── query_backends.rs  # Criterion.rs benchmarks (30M vector scale)
├── tests/
│   ├── correctness.rs     # Hand-crafted vectors, analytically verified
│   └── random.rs          # Randomized sanity tests
├── .github/workflows/
│   ├── rust.yml           # CI: build + test on push/PR to master
│   └── publish.yml        # Publish ring-db-derive then ring-db on v*.*.* tags
├── release_tag.sh         # Release automation (bump both Cargo.tomls, fmt, clippy, tag, push)
└── Cargo.toml             # Workspace root + ring-db package
```

---

## Key Source Files

### [src/engine.rs](src/engine.rs)

Two main public types:

**`RingDb<T: Payload>`** — mutable builder (generic over payload type `T`, default `()`)
- Holds staging buffers: `vectors` (f32 flat row-major), `norms_sq` (precomputed)
- Holds `payload_builder: T::Builder` as a **concrete field** (no Box<dyn>)
- Methods: `new()`, `add_vector()`, `build()`, `load()`, `len()`, `is_empty()`, `dims()`, `backend_name()`

**`SealedRingDb<T: Payload>`** — immutable, `Send + Sync`
- Owns the backend's hot data (vectors + norms) via `Box<dyn RingComputeBackend>`
- Owns `payload_store: T::Store` as a **concrete field**
- Methods: `query()`, `query_range()`, `query_disk()`, `fetch_payload()`, `fetch_payloads()`
- Additional impl block (gated on `T::Store: RefPayloadStore<T>`): `fetch_pod()`, `fetch_pods()`

### [src/payload/mod.rs](src/payload/mod.rs)

**`Payload` trait** — the core abstraction tying a user type to its storage:

```rust
pub trait Payload: Sized {
    type Store: OwnedPayloadStore<Self>;
    type Builder: PayloadBuilderOps<Self, Store = Self::Store>;
    fn make_builder() -> Result<Self::Builder>;
    fn load_store(dir: &Path) -> Result<Self::Store>;
}
```

**`OwnedPayloadStore<T>`** — implemented by both `SerdeStore<T>` and `PodStore<T>`:
- `fetch_owned(id) -> T` — deserializes one payload (bincode for Serde; byte-copy for Pod)
- `fetch_many_owned(ids) -> Vec<T>`

**`RefPayloadStore<T>`** — implemented **only** by `PodStore<T>`:
- `fetch_ref(id) -> &T` — zero-copy reference into the mmap
- `fetch_many_ref(ids) -> Vec<&T>`

`impl Payload for ()` is provided so `RingDb::new()` works without a payload type.

### [src/payload/serde.rs](src/payload/serde.rs)

Variable-size bincode storage.
- **Build**: serializes each payload with bincode, streams to temp file, maintains `Vec<u64>` offset table
- **Read**: mmaps the payload file; `fetch_owned` slices by offset and runs `bincode::deserialize`
- Persist: writes `payloads.bin` + `offsets.bin`

### [src/payload/pod.rs](src/payload/pod.rs)

Zero-copy raw-bytes storage for `T: bytemuck::Pod`.
- **Build**: writes `bytemuck::bytes_of(&payload)` back-to-back, **no offset table**
- **Read**: `fetch_ref(id)` computes `offset = id * size_of::<T>()`, returns `bytemuck::from_bytes(&mmap[offset..offset+size])`
- No heap allocation, no deserialization — just a pointer into the mmap
- Persist: writes `payloads.bin` only (no `offsets.bin`)

### [derive/src/lib.rs](derive/src/lib.rs)

Proc-macro crate (`ring-db-derive`), re-exported as `ringdb::Payload`.

`#[derive(Payload)]` generates a `Payload` impl. Storage strategy is selected by:
```
#[payload(storage = "serde")]  // default — SerdeStore/SerdeStoreBuilder
#[payload(storage = "pod")]    // PodStore/PodStoreBuilder
```

Generated code references `::ringdb::__private::*`, which is a `#[doc(hidden)]`
module re-exporting the concrete store types.

### [src/query.rs](src/query.rs)

All query types take a borrowed `query: &[f32]`:

| Type | Semantics |
|------|-----------|
| `RingQuery` | Symmetric band `[d − λ, d + λ]` |
| `RangeQuery` | Explicit `[d_min, d_max]` interval |
| `DiskQuery` | Full ball `[0, d_max]` |

All three normalize to `(d_min, d_max)` and delegate to `backend.ring_query_f32()`.

`QueryResult` — `ids: Vec<u32>`, `backend_used: &'static str`, `elapsed: Duration`

### [src/backend/cpu.rs](src/backend/cpu.rs)

Always-available reference backend. Key optimizations:
1. **Four-accumulator dot product** — breaks dependency chains so LLVM emits `fmla.4s` / `vfmadd231ps`; ~4× FP throughput
2. **Precomputed `‖x‖²`** — inserted at index time, only dot product computed at query time
3. **Squared bounds** — `d_min²` and `d_max²` computed once before the loop (no `sqrt`)
4. **Rayon `par_chunks_exact(dims)`** — parallelizes the scan across all logical cores

### [src/backend/cuda.rs](src/backend/cuda.rs)

Optional backend behind the `cuda` Cargo feature. It dynamically loads the
NVIDIA CUDA driver at runtime, creates a context on device 0, uploads vectors
and norms to VRAM once, and launches one thread per vector for ring/range/disk
queries. It does not add link-time CUDA requirements to CPU-only builds.

`DiskIntersectionQuery` has a dedicated CUDA kernel: a single scan over the
resident dataset where each thread evaluates every disk for one vector and exits
early as soon as any disk fails.

Distance formula: `dist_sq = norm_sq_i + norm_sq_q − 2·dot(row, query)`

### [src/error.rs](src/error.rs)

```rust
pub enum RingDbError {
    DimensionMismatch { expected: usize, got: usize },
    Payload(String),        // bincode error
    Io(std::io::Error),     // temp file / persist I/O
    Corrupt(String),        // inconsistent persist files on load
}
```

---

## Architecture Decisions

| Decision | Reason |
|----------|--------|
| Two-phase builder (`RingDb` → `SealedRingDb`) | Upload once, query many; type-safe separation of concerns |
| `Payload` trait + `#[derive(Payload)]` | User types select their storage at compile time; zero runtime dispatch on the payload path |
| `T::Builder` / `T::Store` as concrete struct fields | Avoids `Box<dyn>` on every payload push/fetch; monomorphized by the compiler |
| `fetch_pod` gated on `T::Store: RefPayloadStore<T>` | Misuse is a compile error, not a runtime panic |
| Pod storage: no offset table | `size_of::<T>()` is a compile-time constant; position arithmetic beats an array lookup |
| Cold payload storage via mmap | Scales beyond available RAM; OS pages out under pressure |
| No `sqrt` in distance computation | Preserves ordering; avoids expensive operation; better SIMD scheduling |
| Backend trait abstraction | Add GPU/CUDA/AVX-512 backends without changing public API |
| Payload streamed to disk immediately | O(1) peak memory during build regardless of `T` size |

---

## Performance Summary

30 M vectors · 64 dimensions · ring `[6.529, 6.535]` · ~164 K hits.

| Operation | Time | Notes |
|-----------|-----:|-------|
| Ring query (CPU) | ~68 ms | ~440 M vectors/sec |
| Payload fetch — Serde | ~26.6 ms | bincode deserialize per hit |
| Payload fetch — Pod | ~92 µs | zero-copy `&T` from mmap |

Pod fetch is **~288× faster** than Serde fetch for fixed-size payloads.
Use `#[payload(storage = "pod")]` when retrieval latency is the bottleneck.

---

## Build and Test Commands

```bash
# Build
cargo build
cargo build --release
cargo build --release --features cuda
RUSTFLAGS="-C target-cpu=native" cargo build --release  # native SIMD

# Test
cargo test
cargo test --features cuda
RINGDB_CUDA_TEST=1 cargo test --features cuda test_cuda_backend_matches_cpu_when_requested
cargo test --test correctness
cargo test --test random
cargo test -- --nocapture

# Benchmark (recommended with native CPU features)
RUSTFLAGS="-C target-cpu=native" cargo bench
cargo bench --bench query_backends cpu_f32
cargo bench --features cuda --bench query_backends cuda_f32
RINGDB_CUDA_BENCH_N=5000000 cargo bench --features cuda --bench query_backends cuda_f32
cargo bench --bench query_backends payload_fetch

# Example CLI
cargo run --release --example cli -- --n 100000 --dims 64 --d 5.0 --lambda 0.2
cargo run --example cli -- --help

# Docs
cargo doc --open

# Release (automated — bumps both Cargo.tomls, tags, pushes)
./release_tag.sh <version>
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `rayon` | Data parallelism (work-stealing) |
| `thiserror` | Error derive macro |
| `memmap2` | Memory-mapped file I/O |
| `bincode` | Binary serialization for Serde payloads |
| `serde` | Serialization framework |
| `bytemuck` | Pod cast for zero-copy payload storage |
| `ring-db-derive` | `#[derive(Payload)]` proc-macro |
| CUDA driver | Runtime-loaded only when the `cuda` feature and `BackendPreference::Cuda` are used |
| `criterion` (dev) | Statistical benchmarking |
| `rand` (dev) | Deterministic random generation |
| `clap` (dev) | CLI argument parsing (example only) |

---

## Data Flow Cheatsheet

### Insert
```
add_vector(&[f32], T)
  → [dim check]
  → [compute norm_sq]
  → [extend vectors staging buffer]
  → T::Builder::push(payload)
      Serde: serialize T → bincode → append to temp file
      Pod:   bytemuck::bytes_of(&payload) → append to temp file
```

### Build
```
db.build()
  → backend.upload_f32_dataset(dims, vectors, norms_sq)
  → T::Builder::finish() → T::Store (mmap)
      Serde: flush + mmap payloads.bin + offsets.bin
      Pod:   flush + mmap payloads.bin (no offsets file)
  → SealedRingDb { backend, payload_store: T::Store }
```

### Query
```
db.query(&RingQuery { query, d, lambda })
  → normalize to (d_min, d_max)
  → backend.ring_query_f32(dims, query, d_min, d_max)
      → compute norm_sq_q
      → square bounds: lower_sq, upper_sq
      → par_chunks_exact(dims):
          dist_sq = norm_sq_i + norm_sq_q - 2·dot(row_i, query)
          if lower_sq ≤ dist_sq ≤ upper_sq → include id i
  → QueryResult { ids, backend_used, elapsed }
```

### Payload fetch — Serde
```
db.fetch_payloads(&[id])
  → for each id: offsets[id]..offsets[id+1] → mmap slice → bincode::deserialize
  → Vec<T>
```

### Payload fetch — Pod
```
db.fetch_pods(&[id])
  → for each id: offset = id * size_of::<T>()
                 bytemuck::from_bytes(&mmap[offset..offset+size]) → &T
  → Vec<&T>  (zero-copy, borrows mmap)
```

---

## Public API Quick Reference

```rust
use ringdb::{RingDb, RingDbConfig, RingQuery, RangeQuery, DiskQuery, Payload};

// --- Payload type (serde strategy, default) ---
#[derive(serde::Serialize, serde::Deserialize, Payload)]
struct MyDoc { title: String }

// --- Payload type (pod strategy — zero-copy fetch) ---
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Payload)]
#[repr(C)]
#[payload(storage = "pod")]
struct MyRecord { x: f32, y: f32 }

// Build
let mut db: RingDb<MyDoc> = RingDb::new(RingDbConfig::new(dims))?;
db.add_vector(&vec_f32, my_doc)?;
let db = db.build()?;  // → SealedRingDb<MyDoc>

// Query variants
let r = db.query(&RingQuery { query: &q, d: 5.0, lambda: 0.5 })?;
let r = db.query_range(&RangeQuery { query: &q, d_min: 4.5, d_max: 5.5 })?;
let r = db.query_disk(&DiskQuery { query: &q, d_max: 5.5 })?;

// Fetch payloads (serde or pod — owned T)
let payloads: Vec<MyDoc> = db.fetch_payloads(&r.ids)?;

// Fetch payloads (pod only — zero-copy &T)
let mut db2: RingDb<MyRecord> = RingDb::new(RingDbConfig::new(dims))?;
// ... add_vector calls ...
let db2 = db2.build()?;
let refs: Vec<&MyRecord> = db2.fetch_pods(&r.ids);  // no allocation
```

`T` defaults to `()` (no payload). Any `T: Payload` works; use `#[derive(Payload)]`.

---

## CI/CD

- **Push/PR to `master`** → `rust.yml`: `cargo build` + `cargo test`
- **Tag `v*.*.*`** → `publish.yml`:
  1. Full test suite
  2. Dry-run + publish `ring-db-derive` (proc-macro, no deps on ring-db)
  3. Sleep 30 s (crates.io indexing)
  4. Dry-run + publish `ring-db`
  5. Create GitHub release

---

