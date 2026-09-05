# ring-db

ring-db is an exact vector database for geometric range searches and K-nearest-neighbor
(KNN) searches. It keeps the data model deliberately small:

- vectors are `f32`;
- distance is ordinary L2 or per-query diagonal weighted L2;
- payloads are fixed-size `bytemuck::Pod` records;
- filterable metadata is supplied explicitly as integer tags;
- filters become reusable candidate sets before distance calculation.

There is no payload deserialization, payload reflection, filter language, or approximate
index hidden behind the API.

## Queries

| Query | Result |
|---|---|
| `RingQuery` | Every vector in `[d - lambda, d + lambda]` |
| `RangeQuery` | Every vector in `[d_min, d_max]` |
| `DiskQuery` | Every vector in `[0, d_max]` |
| `DiskIntersectionQuery` | Every vector inside all supplied disks |
| `KnnQuery` | Up to `k` closest vectors, exactly ranked |

Ring, range, disk, and KNN queries can borrow diagonal weights; intersection disks carry
their own weights. Every query shape can borrow a `CandidateSet`. KNN results are ordered
by ascending `(dist_sq, id)`, so equal-distance results are deterministic.

The weighted metric is:

```text
dist_sq = sum(weights[i] * (vector[i] - query[i])^2)
```

Weights are final coefficients: ring-db does not square them. They must be finite,
non-negative, and have the same length as the vectors. Zero is valid. Vector and query
coordinates must be finite, as must all non-negative distance bounds.

## Install

```toml
[dependencies]
ring-db = "0.8"
bytemuck = { version = "1", features = ["derive"] }
```

## Closest bakery example

This example stores a small Pod record, indexes application-defined metadata, composes a
filter, and runs exact weighted KNN only over matching IDs.

```rust
use bytemuck::{Pod, Zeroable};
use ringdb::{
    KnnQuery, MetadataKey, MetadataTag, MetadataValue, RingDb, RingDbConfig,
};

const KIND: MetadataKey = MetadataKey::new(1);
const STATUS: MetadataKey = MetadataKey::new(2);
const BAKERY: MetadataValue = MetadataValue::new(10);
const CAFE: MetadataValue = MetadataValue::new(11);
const OPEN: MetadataValue = MetadataValue::new(20);
const CLOSED: MetadataValue = MetadataValue::new(21);

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct Place {
    name_id: u32,
    rating_milli: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut db: RingDb<Place> = RingDb::new(RingDbConfig::new(2))?;

    db.add_vector_with_metadata(
        &[0.8, 1.5],
        Place { name_id: 100, rating_milli: 4700 },
        &[
            MetadataTag::new(KIND, BAKERY),
            MetadataTag::new(STATUS, OPEN),
        ],
    )?;
    db.add_vector_with_metadata(
        &[0.2, 0.3],
        Place { name_id: 101, rating_milli: 4500 },
        &[
            MetadataTag::new(KIND, CAFE),
            MetadataTag::new(STATUS, OPEN),
        ],
    )?;
    db.add_vector_with_metadata(
        &[1.0, 0.4],
        Place { name_id: 102, rating_milli: 4900 },
        &[
            MetadataTag::new(KIND, BAKERY),
            MetadataTag::new(STATUS, CLOSED),
        ],
    )?;

    let db = db.build()?;
    let bakeries = db.metadata_eq(KIND, BAKERY);
    let open = db.metadata_eq(STATUS, OPEN);
    let open_bakeries = bakeries.intersection(&open)?;

    let origin = [0.0, 0.0];
    let weights = [4.0, 1.0];
    let result = db.query_knn(
        &KnnQuery::new(&origin, 1)
            .with_weights(&weights)
            .with_candidates(&open_bakeries),
    )?;

    if let Some(hit) = result.hits.first() {
        let place = db.fetch_payload(hit.id)?;
        println!("closest bakery: {place:?}, distance={}", hit.dist_sq.sqrt());
    }
    Ok(())
}
```

Metadata tags and payload fields are independent on purpose. A tag is hot, indexable
query data; a payload is opaque result data. Store frequently filtered strings as stable
integer IDs, and keep the ID-to-string dictionary in the application. Bounded UTF-8 can
also be stored in a Pod field such as `[u8; N]`.

## Metadata and candidate sets

Metadata is equality-only in the current API:

```rust
let tag = MetadataTag::new(MetadataKey::new(1), MetadataValue::new(10));
db.add_vector_with_metadata(&vector, payload, &[tag])?;
```

After `build()`, a lookup returns a database-sized `CandidateSet`:

```rust
let bakeries = db.metadata_eq(KIND, BAKERY);
let cafes = db.metadata_eq(KIND, CAFE);
let open = db.metadata_eq(STATUS, OPEN);

let food = bakeries.union(&cafes)?;
let open_food = food.intersection(&open)?;
let not_open = open.complement();
let hand_picked = db.candidates_from_ids([4, 9, 12])?;
```

`union` and `intersection` reject sets with different universe lengths. Candidate sets
choose a sparse ID list or dense bitmap internally. On CPU, sparse sets iterate eligible
IDs directly; dense sets reject rows before calculating distance. An unfiltered query
keeps its branch-free full-scan path.

All query shapes accept `.with_candidates(...)`:

```rust
let ring = RingQuery::new(&point, 5.0, 0.5).with_candidates(&open_food);
let range = RangeQuery::new(&point, 2.0, 6.0).with_candidates(&open_food);
let disk = DiskQuery::new(&point, 6.0).with_candidates(&open_food);
let knn = KnnQuery::new(&point, 10).with_candidates(&open_food);
```

For a `DiskIntersectionQuery`, candidate sets on the outer query and on individual disks
are all intersected before execution.

## Pod payloads

`RingDb<T>` requires `T: Pod + Send + Sync`. Use `T = ()` when no payload is needed.

```rust
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct Record {
    category_id: u32,
    name_len: u32,
    name: [u8; 56],
}
```

A Pod record must have a stable fixed-size layout and cannot contain pointers or
heap-backed fields such as `String` and `Vec`. Avoid implicit padding; the Pod derive
rejects layouts that contain it.

Payloads are kept in one private, aligned Pod array. Persisted payloads are read once
into that array when loading. Fetches are zero-copy references with bounds checking:

```rust
let one: &Record = db.fetch_payload(id)?;
let many: Vec<&Record> = db.fetch_payloads(&ids)?;
```

## API overview

### Build and load

| Method | Purpose |
|---|---|
| `RingDb::new(config)` | Create a mutable database builder |
| `add_vector(vector, payload)` | Insert without metadata |
| `add_vector_with_metadata(vector, payload, tags)` | Insert with equality tags |
| `build()` | Seal the database and upload vectors to the backend |
| `RingDb::<T>::load(path, backend)` | Load a persisted database as the same Pod layout |

Vector IDs are sequential `u32` values in insertion order.

### Query and fetch

| Method | Purpose |
|---|---|
| `query(&RingQuery)` | Symmetric distance band |
| `query_range(&RangeQuery)` | Explicit distance interval |
| `query_disk(&DiskQuery)` | Inclusive radius |
| `query_disk_intersection(&DiskIntersectionQuery)` | Intersection of disks |
| `query_knn(&KnnQuery)` | Exact top-K |
| `metadata_eq(key, value)` | Reusable candidates for one tag |
| `candidates_from_ids(ids)` | Validated manual candidate set |
| `fetch_payload(id)` | One zero-copy Pod reference |
| `fetch_payloads(ids)` | Pod references in requested ID order |

`QueryResult` contains `hits: Vec<Hit>`, `backend_used`, and `elapsed`. Each `Hit`
contains an insertion-order `id` and `dist_sq` under the query metric.

## Backends

CPU is always available and is the reference exact implementation. It uses Rayon,
SIMD-friendly direct-distance loops, adaptive candidate traversal, and bounded local heaps.

CUDA is optional:

```bash
cargo build --release --features cuda
```

Select it with `BackendPreference::Cuda`. The NVIDIA driver is loaded dynamically, so
CPU-only builds have no CUDA link-time dependency. Construction returns
`BackendUnavailable` when the feature, driver, or device is unavailable. The CUDA backend
supports the same ordinary/weighted query shapes, candidate filtering, and exact KNN
contract as CPU; minor `f32` distance differences between backends are expected.

## Persistence

Set a directory before building:

```rust
let config = RingDbConfig::new(128).with_persist_dir("/data/places");
let mut db: RingDb<Record> = RingDb::new(config)?;
// insert rows
let db = db.build()?;

let db = RingDb::<Record>::load(
    std::path::Path::new("/data/places"),
    BackendPreference::Cpu,
)?;
```

The target must be absent or empty. ring-db does not overwrite a previous generation;
it writes the three data files first and `meta.bin` last. This prevents a build that
returns an error from looking complete; the format does not promise power-loss durability.

Persistence format version **1** uses four files:

| File | Contents |
|---|---|
| `meta.bin` | Magic, version, endian marker, dimensions, row count, payload size/alignment |
| `vectors.bin` | Row-major little-endian `f32` vectors |
| `payloads.bin` | Consecutive raw Pod records |
| `metadata.bin` | Versioned equality-tag posting sets |

Pod bytes use the record's native layout. Loading checks endianness, payload size, and
alignment, but the format does not encode Rust field names. Load with the exact same
record layout. No derived distance state is persisted.

## Migrating from 0.7

Version 0.8 deliberately replaces both previous payload paths with one Pod-only API:

- derive payload records with `bytemuck::Pod` and `bytemuck::Zeroable`;
- replace the old `Payload` derive and dynamic/Serde records with fixed Pod fields,
  integer IDs, or an application-owned external store;
- supply explicit integer `MetadataTag`s for fields that must be filterable;
- replace `fetch_pod`/owned payload decoding with the fallible, zero-copy
  `fetch_payload` and `fetch_payloads` methods;
- rebuild every database persisted by an earlier ring-db version.

## Design boundaries

- Search is exact brute force, not approximate ANN.
- Metadata supports explicit integer equality tags, not arbitrary payload inspection.
- Candidate-set boolean operations provide `AND`, `OR`, and `NOT` without a query AST.
- Filtered strings should normally be interned to integers.
- Weights are diagonal and supplied per query; a full matrix is not supported.
- The database is immutable after `build()`.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features

# Fast local benchmark run
RINGDB_BENCH_N=100000 cargo bench --bench query_backends

# Focused groups
cargo bench --bench query_backends cpu_knn_f32
cargo bench --bench query_backends metadata

# CLI: exact weighted KNN within one metadata category
cargo run --release --example cli -- --n 100000 --dims 64 --k 10 --category 3
```

The benchmark default is intentionally large. Use `RINGDB_BENCH_N` (and
`RINGDB_CUDA_BENCH_N` for CUDA groups) for smaller local runs.

## License

MIT OR Apache-2.0.
