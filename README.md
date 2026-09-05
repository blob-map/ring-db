# ring-db

ring-db is a Rust library for exact vector search on a CPU or an NVIDIA GPU.
It finds the closest vectors or all vectors within specified distance limits.
You can filter vectors by metadata and set a weight for each dimension.

Search is exact: ring-db calculates distances for all vectors that pass the filter.
It uses `f32` arithmetic. It does not use an approximate search index.

## Install

Add these dependencies to `Cargo.toml`:

```toml
[dependencies]
ring-db = "0.8"
bytemuck = { version = "1", features = ["derive"] }
```

The examples use `bytemuck` to define payload records. You can omit this dependency
if your application does not define its own payload type.

## Find the closest vector

This example creates a database with two dimensions and no payload data.
It finds the vector closest to `[0.0, 0.0]`.
The result field `dist_sq` contains the squared distance.

```rust
use ringdb::{KnnQuery, RingDb, RingDbConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut db = RingDb::new(RingDbConfig::new(2))?;
    db.add_vector(&[3.0, 4.0], ())?;
    db.add_vector(&[1.0, 0.0], ())?;
    let db = db.build()?;

    let origin = [0.0, 0.0];
    let result = db.query_knn(&KnnQuery::new(&origin, 1))?;

    assert_eq!(result.hits[0].id, 1);
    assert_eq!(result.hits[0].dist_sq, 1.0);
    Ok(())
}
```

Each vector receives a `u32` ID, starting at zero, in insertion order.
Call `build()` after you insert the data. This returns a `SealedRingDb` that you can
query repeatedly. You cannot insert, change, or remove records after `build()`.

## Vectors, payloads, and metadata

Each database record has these parts:

| Part | Purpose | Type |
|---|---|---|
| Vector | Calculate distance from a query vector | One `f32` value per dimension |
| Payload | Store data that the application retrieves with a result | A fixed-size `bytemuck::Pod` record, or `()` |
| Metadata tags | Select which vectors to search | Integer key and value pairs |

All vectors in a database must have the same number of dimensions.
The number of dimensions must be greater than zero.

ring-db uses metadata tags to select vectors before it calculates distances.
It does not read payload fields to evaluate a filter.
Your application supplies the tags separately when it inserts a vector.

## Find the closest open bakery

This example filters places by type and status, then searches the matching places.
The coordinates are positions on a flat map with the same unit on both axes.
They are not latitude and longitude.

The weights `[4.0, 1.0]` multiply the squared difference on the first axis by four.
The second axis has a weight of one.

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut db: RingDb<Place> = RingDb::new(RingDbConfig::new(2))?;

    // Position, name ID, type, and status for each place.
    let places = [
        ([0.8, 1.5], 100, BAKERY, OPEN),
        ([0.2, 0.3], 101, CAFE, OPEN),
        ([1.0, 0.4], 102, BAKERY, CLOSED),
        ([2.0, 0.0], 103, BAKERY, OPEN),
    ];

    for (position, name_id, kind, status) in places {
        db.add_vector_with_metadata(
            &position,
            Place { name_id },
            &[
                MetadataTag::new(KIND, kind),
                MetadataTag::new(STATUS, status),
            ],
        )?;
    }
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
        assert_eq!(place.name_id, 100);
        println!("name ID: {}, distance: {}", place.name_id, hit.dist_sq.sqrt());
    }
    Ok(())
}
```

The query compares the two open bakeries. It excludes the cafe and the closed bakery,
even though they are closer to the origin.
The result contains the bakery with name ID `100`.

## Metadata filters

A metadata tag contains a `MetadataKey` (`u32`) and a `MetadataValue` (`u64`).
Your application assigns these integer values and defines their meanings.
Use stable values if you save the database to disk.

`metadata_eq(key, value)` returns a `CandidateSet`: the vector IDs with that tag.
A tag with no matches returns an empty set.
You can reuse a candidate set in multiple queries.

Use these methods to combine filters:

| Method | Meaning | Selected IDs |
|---|---|---|
| `a.intersection(&b)?` | AND | IDs in both sets |
| `a.union(&b)?` | OR | IDs in either set |
| `a.complement()` | NOT | All database IDs outside `a` |
| `db.candidates_from_ids(ids)?` | Select by ID | The supplied IDs, after validation |

For example, `open.complement()` includes all records without the `STATUS = OPEN` tag.
This includes records with no status tag.

Use candidate sets from the database you will query.
ring-db checks the vector count associated with each set, but it does not check database identity.
Sets from different databases with the same vector count can pass this check.

Pass a filter to any query with `.with_candidates(&set)`.
Without this option, the query searches all vectors.
For `DiskIntersectionQuery`, a vector must pass the filters on the query and on every disk.

Metadata filters support integer equality. They do not support text matching or numeric
range comparisons. To filter strings, assign each distinct string an integer ID.
Keep the mapping between IDs and strings in your application.

## Query types and distances

K-nearest-neighbor (KNN) search returns up to `k` closest vectors that pass the filter.
Other query types return all vectors that satisfy their distance limits and filters.
All distance limits are inclusive.

| Query type | Database method | Distance condition |
|---|---|---|
| `KnnQuery` | `query_knn` | The `k` closest vectors |
| `RingQuery` | `query` | Between `max(0, d - lambda)` and `d + lambda` |
| `RangeQuery` | `query_range` | Between `d_min` and `d_max` |
| `DiskQuery` | `query_disk` | At most `d_max` from the query vector |
| `DiskIntersectionQuery` | `query_disk_intersection` | Within every disk |

For KNN, `k = 0` returns no results. If fewer than `k` vectors pass the filter,
the query returns all of them. Results are sorted by squared distance, then by vector ID.
For equal distances, the smaller ID comes first.

The default distance is Euclidean distance, also called L2 distance.
ring-db calculates its square:

```text
dist_sq = sum((vector[i] - query[i])^2)
```

To use diagonal weights, call `.with_weights(&weights)`:

```text
dist_sq = sum(weights[i] * (vector[i] - query[i])^2)
```

ring-db uses each weight as supplied. It does not square the weights.
A zero weight excludes that dimension from the distance calculation.
Each query can use different weights. For a disk intersection, set weights on each disk.
Full weight matrices are not supported.

Query inputs must meet these requirements:

- Vectors, query vectors, and weight arrays must have the database dimension count.
- Vector coordinates and query coordinates must be finite: no `NaN` or infinity.
- Weights and distance limits must be finite and non-negative.
- A range must have `d_min <= d_max`.
- A disk intersection must contain at least one disk.

Each query returns a `QueryResult` with `hits`, `backend_used`, and `elapsed`.
Each hit contains the vector ID and `dist_sq`. Use `dist_sq.sqrt()` to obtain the distance.
For a disk intersection, the reported distance uses the first disk's query vector and weights.

## Payload records

`RingDb<T>` requires `T: bytemuck::Pod + Send + Sync`.
`Pod` means plain old data. Use `()` when you do not need a payload.

Use a fixed-size record for payload data:

```rust
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct Record {
    category_id: u32,
    name_len: u32,
    name: [u8; 64],
}
```

A Pod record cannot contain pointers, `String`, or `Vec` fields.
The `Pod` derive checks the field types and rejects implicit padding between fields or
at the end of a record.

A byte array can hold text up to a fixed capacity.
In this example, `name_len` records the number of bytes used in `name`.
Your application must check the length and UTF-8 encoding when it reads the text.

`db.fetch_payload(id)?` returns a reference to one payload.
`db.fetch_payloads(&ids)?` returns a vector of references in the requested ID order.
Both methods check IDs and return an error for an invalid ID.
They do not copy or deserialize payload records.

## CPU and CUDA

The CPU backend is the default. It uses Rayon to distribute work across CPU threads.

To enable CUDA in your application, use this dependency:

```toml
ring-db = { version = "0.8", features = ["cuda"] }
```

Select CUDA with `RingDbConfig::new(dims).with_backend_preference(BackendPreference::Cuda)`.
You also need a compatible NVIDIA driver and GPU.
ring-db loads the driver at runtime. CPU use does not require CUDA.

Database creation returns `BackendUnavailable` if the CUDA feature, driver, or device is unavailable.
Both backends support all query types, weights, and metadata filters.
Floating-point rounding can cause small distance differences between the backends.

## Save and load a database

Set a directory with `with_persist_dir()` before you call `build()`.
The directory must be empty or must not exist.

This example saves one vector without a payload, then loads the database:

```rust,no_run
use std::path::Path;
use ringdb::{BackendPreference, RingDb, RingDbConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("places-db");
    let config = RingDbConfig::new(2).with_persist_dir(path);
    let mut db = RingDb::new(config)?;
    db.add_vector(&[0.8, 1.5], ())?;
    let _db = db.build()?;

    let loaded = RingDb::<()>::load(path, BackendPreference::Cpu)?;
    assert_eq!(loaded.len(), 1);
    Ok(())
}
```

Format version 1 uses four files:

| File | Contents |
|---|---|
| `meta.bin` | Format identifier, version, byte order, dimension count, vector count, payload size and alignment |
| `vectors.bin` | `f32` coordinates in vector order, with little-endian byte order |
| `payloads.bin` | Payload records in their native memory layout |
| `metadata.bin` | Metadata tags and their matching vector IDs |

ring-db writes the three data files first and `meta.bin` last.
It does not overwrite an existing database. It does not guarantee that writes survive a power failure.

Load with the same payload record layout that you used to save the database.
ring-db checks byte order, payload size, and alignment. It does not store field names
or verify that fields have the same meaning.
Loading reads payloads into memory once. Queries and weights are not saved.

## Migrate from version 0.7

Version 0.8 uses Pod payloads only. To update an application:

1. Replace the old `Payload` derive with `bytemuck::Pod` and `bytemuck::Zeroable`.
2. Replace dynamic payload fields with fixed-size fields or integer IDs.
   Store variable-size data in your application if needed.
3. Supply explicit `MetadataTag` values for data you want to filter.
4. Replace `fetch_pod` and payload decoding with `fetch_payload` or `fetch_payloads`.
   These methods return references and can return errors.
5. Rebuild databases saved with earlier ring-db versions.

## Development

Run the standard checks:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

To execute the CUDA comparison test, run this command on a machine with a compatible GPU and driver:

```bash
RINGDB_CUDA_TEST=1 cargo test --all-features test_cuda_backend_matches_cpu_when_requested
```

Without `RINGDB_CUDA_TEST=1`, that test skips GPU execution.

Run benchmarks with a smaller dataset:

```bash
RINGDB_BENCH_N=100000 cargo bench --bench query_backends
RINGDB_BENCH_N=100000 cargo bench --bench query_backends cpu_knn_f32
RINGDB_BENCH_N=100000 cargo bench --bench query_backends metadata
```

The default benchmark dataset is large. Use `RINGDB_BENCH_N` for CPU benchmarks
and `RINGDB_CUDA_BENCH_N` for CUDA benchmarks to change the vector count.

Run the command-line example to search one metadata category:

```bash
cargo run --release --example cli -- --n 100000 --dims 64 --k 10 --category 3
```

For the full API reference, run `cargo doc --open`.

## License

MIT OR Apache-2.0.
