# ring-db

ring-db is a Rust library for exact vector search on a CPU or a GPU with Vulkan.
It finds the closest vectors or all vectors between given distance limits.
You can filter vectors with metadata tags and set a weight for each dimension.

Exact search calculates distances for all vectors that pass the filter.
ring-db uses `f32` arithmetic.
It does not use an approximate search index.

## Install

Add this dependency to `Cargo.toml`:

```toml
[dependencies]
ring-db = "0.9"
```

The crate includes CPU and Vulkan backends.
It has no Cargo feature flags.
The CPU backend is the default.
A GPU, Vulkan loader, and GPU driver are not necessary for CPU queries.

## Find the closest vector

This example creates a database with two dimensions and no payload.
It finds the vector closest to `[0.0, 0.0]`.
The `dist_sq` field contains the squared distance.

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

Each vector has a `u32` ID.
The IDs start at zero and follow the insertion order.

After you insert the data, call `build()`.
This method returns a `SealedRingDb` for subsequent queries.
You cannot insert, change, or remove records after this call.

## Select CPU or Vulkan

The two backends can execute all query types, weights, and metadata filters.
The backend selection occurs at runtime.

| Backend | Selection | Runtime requirements |
|---|---|---|
| CPU | `BackendPreference::Cpu`, or the default | No GPU runtime |
| Vulkan | `BackendPreference::Vulkan` | A Vulkan 1.2 loader, driver, and compatible compute device |

To use Vulkan, set the backend in `RingDbConfig`:

```rust,no_run
use ringdb::{BackendPreference, RingDb, RingDbConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = RingDbConfig::new(64)
        .with_backend_preference(BackendPreference::Vulkan);
    let mut db: RingDb = RingDb::new(config)?;
    db.add_vector(&[0.0; 64], ())?;
    let db = db.build()?;
    assert_eq!(db.backend_name(), "vulkan");
    Ok(())
}
```

ring-db loads Vulkan only when you select it.
The crate contains compiled shaders.
A Vulkan SDK and shader compiler are not necessary to build or use the crate.

Vulkan can use compatible NVIDIA and AMD GPUs.
ring-db first tries discrete GPUs, then integrated GPUs, then other compute devices.
It can also use a software Vulkan driver.
The backend requires Vulkan 1.2, Buffer Device Address, and 64-bit shader integers.
ring-db examines device support and enables these features automatically.
Subgroup extensions and 64-bit shader atomics are not necessary.

If Vulkan is unavailable, database creation returns `BackendUnavailable`.
ring-db does not automatically change the backend to CPU.

The Vulkan backend uploads vectors one time and keeps them in device memory.
It executes queries on one database in sequence.
It keeps query buffers for subsequent queries until you drop the database.
Each result owns its hit data and stays valid after subsequent queries.

Vulkan uses 64-bit device addresses to access vector data and query buffers.
A single vector buffer can be more than 4 GiB.
The storage-buffer binding limit does not apply to these device addresses.

If a device's allocation limit is too small, ring-db divides the dataset into groups of vectors called chunks.
Queries combine the results from all chunks and keep the original vector IDs.
KNN returns the closest vectors across the full dataset.
All vectors and query buffers must fit the available device memory.
An allocation that is too large causes a backend error.
Floating-point rounding can cause small differences between CPU and Vulkan results, especially near distance limits.

To run the example with Vulkan, use this command:

```bash
cargo run --release --example cli -- --backend vulkan --n 100000 --dims 64 --k 10
```

## Vectors, payloads, and metadata

Each database record has these parts:

| Part | Function | Type |
|---|---|---|
| Vector | Calculate the distance from a query vector | One `f32` value for each dimension |
| Payload | Keep application data for retrieval with a result | A fixed-size `bytemuck::Pod` record, or `()` |
| Metadata tags | Select the vectors to search | Integer keys and values |

All vectors in a database must have the same number of dimensions.
The number of dimensions must be more than zero.

ring-db uses metadata tags to select vectors before it calculates distances.
It does not read payload fields to apply a filter.
Your application supplies the tags as separate values when it inserts a vector.

## Find the closest open bakery

This example filters places by type and status.
It then searches the places that pass the two filters.
The coordinates give positions on a flat map with the same unit on both axes.
They are not latitude and longitude.

The weights `[4.0, 1.0]` multiply the squared difference on the first axis by four.
The second axis has a weight of one.

For the payload examples, add this dependency to the `[dependencies]` section:

```toml
bytemuck = { version = "1", features = ["derive"] }
```

The `derive` feature belongs to `bytemuck`.
It generates the necessary traits for your payload type.
ring-db itself has no feature flags.
This direct dependency is not necessary for applications with only the unit payload `()`.

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

    let hit = result.hits.first().expect("one open bakery");
    let place = db.fetch_payload(hit.id)?;
    assert_eq!(place.name_id, 100);
    println!("name ID: {}, distance: {}", place.name_id, hit.dist_sq.sqrt());
    Ok(())
}
```

The query compares the two open bakeries.
It does not include the cafe and the closed bakery, although they are closer to the origin.
The result contains the bakery with name ID `100`.

## Metadata filters

A metadata tag contains a `MetadataKey` (`u32`) and a `MetadataValue` (`u64`).
Your application gives values and meanings to these integers.
If you save the database, keep the same meanings for these integers when you load it.

`metadata_eq(key, value)` returns a `CandidateSet` with the vector IDs that have that tag.
A tag with no matches gives an empty set.
You can use a candidate set in more than one query.

For filter operations, use these methods:

| Method | Operation | Selected IDs |
|---|---|---|
| `a.intersection(&b)?` | AND | IDs in the two sets |
| `a.union(&b)?` | OR | IDs in one or two sets |
| `a.complement()` | NOT | All database IDs not in `a` |
| `db.candidates_from_ids(ids)?` | Select by ID | Valid IDs from the supplied list |

For example, `open.complement()` includes all records without the `STATUS = OPEN` tag.
This includes records with no status tag.

Use candidate sets from the database that you query.
ring-db validates the vector count for each set.
It does not validate database identity.
Sets from different databases with the same vector count can pass this validation.

To apply a filter, call `.with_candidates(&set)` on the query.
Without this call, the query searches all vectors.
For `DiskIntersectionQuery`, a vector must pass the filters on the query and on all disks.

Metadata filters compare integers for equality.
They cannot compare text or select a numeric range.
To filter strings, give an integer ID to each different string.
Keep the relation between IDs and strings in your application.

## Query types and distances

K-nearest-neighbor (KNN) search returns up to `k` closest vectors that pass the filter.
Other query types return all vectors that satisfy their distance limits and filters.
All distance limits include their boundary values.

| Query type | Database method | Distance condition |
|---|---|---|
| `KnnQuery` | `query_knn` | The `k` closest vectors |
| `RingQuery` | `query` | Between `max(0, d - lambda)` and `d + lambda` |
| `RangeQuery` | `query_range` | Between `d_min` and `d_max` |
| `DiskQuery` | `query_disk` | At most `d_max` from the query vector |
| `DiskIntersectionQuery` | `query_disk_intersection` | In all disks |

For KNN, `k = 0` gives no results.
If fewer than `k` vectors pass the filter, the query returns all of them.
KNN sorts results by squared distance, then by vector ID.
For equal distances, the smaller ID comes first.
The result order for other query types can change between queries.

The default distance is Euclidean distance, also called L2 distance.
ring-db calculates its square:

```text
dist_sq = sum((vector[i] - query[i])^2)
```

To use diagonal weights, call `.with_weights(&weights)`:

```text
dist_sq = sum(weights[i] * (vector[i] - query[i])^2)
```

ring-db uses each weight as supplied.
It does not square the weights.
A zero weight removes that dimension from the distance calculation.
Each query can use different weights.

For a disk intersection, set weights on each disk.
Full weight matrices are not available.

Query inputs have these requirements:

- Vectors, query vectors, and weight arrays must have the database dimension count.
- Vector coordinates and query coordinates must be finite, without `NaN` or infinity.
- Weights and distance limits must be finite and non-negative.
- A range must have `d_min <= d_max`.
- A disk intersection must contain one or more disks.

Each query returns a `QueryResult` with `hits`, `backend_used`, and `elapsed`.
Each hit contains the vector ID and `dist_sq`.
For a disk intersection, `dist_sq` uses the first disk's query vector and weights.

To get the distance from `dist_sq`, use `dist_sq.sqrt()`.

## Payload records

The payload type `T` must implement `bytemuck::Pod + Send + Sync`.
`Pod` means plain old data.

For a database without payload data, use `()`.
For payload data, use a fixed-size record:

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
The `Pod` derive rejects field types and implicit padding that do not satisfy the trait requirements.

A byte array can hold text up to a fixed capacity.
In this example, `name_len` contains the number of bytes in the name.

When you read the text, make sure that `name_len` is not more than the array length.
Make sure that the text has correct UTF-8 encoding.

`db.fetch_payload(id)?` returns a reference to one payload.
`db.fetch_payloads(&ids)?` returns a vector of references in the requested ID order.
The two methods validate IDs and return an error for an invalid ID.
They do not copy or deserialize payload records.

## Save and load a database

Before you call `build()`, set a directory with `with_persist_dir()`.
Use an empty directory or a path for a new directory.

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

Format version 1 uses these files:

| File | Contents |
|---|---|
| `meta.bin` | Format version, byte order, dimensions, vector count, and payload layout |
| `vectors.bin` | `f32` coordinates in vector order, with little-endian byte order |
| `payloads.bin` | Payload records in their native memory layout |
| `metadata.bin` | Metadata tags and their vector IDs |

ring-db writes the three data files first and `meta.bin` last.
It does not overwrite a database.
A power failure can cause data loss.

Load the database with the same payload record layout that you used to save it.
ring-db validates byte order, payload size, and alignment.
It does not save field names or validate their meanings.

A database load reads payloads into memory one time.
The saved database does not contain queries or weights.

## Development

The shell commands in this section use Bash syntax.

Run the standard checks:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

`cargo test` also compiles the Rust examples in this README.
It executes the examples that do not have the `no_run` attribute.
A GPU is not necessary for the standard tests.

To compare Vulkan queries with CPU results, run the device tests:

```bash
RINGDB_VULKAN_TEST=1 cargo test --test vulkan
```

A compatible hardware or software Vulkan device is necessary for these tests.
Without `RINGDB_VULKAN_TEST=1`, the device tests do not execute GPU queries.
The test for an unavailable driver always executes.
The continuous integration (CI) tests use Mesa software Vulkan.

To test vector addresses beyond 4 GiB, use this command:

```bash
RINGDB_VULKAN_TEST=1 RINGDB_VULKAN_LARGE_TEST=1 cargo test --test vulkan vulkan_reads_vectors_past_four_gib
```

This test uses more than 4 GiB of device memory and additional host memory.
Devices with a small allocation limit use multiple buffers during this test.

The `src/backend/vulkan/shaders` directory contains shader sources and compiled SPIR-V files.
For shader development, install `glslc` and `spirv-val`.
After a shader source change, regenerate and validate the compiled files:

```bash
sh src/backend/vulkan/shaders/compile.sh
```

These shader tools are not necessary for applications that use ring-db.

## Benchmarks

To run benchmarks with a smaller dataset, use one of these commands:

```bash
RINGDB_BENCH_N=100000 cargo bench --bench query_backends
RINGDB_BENCH_N=100000 cargo bench --bench query_backends cpu_knn_f32
RINGDB_BENCH_N=100000 cargo bench --bench query_backends metadata
```

The default dataset has 30 million vectors.
`RINGDB_BENCH_N` sets the vector count for the two backends.
`RINGDB_VULKAN_BENCH_N` sets a different count for Vulkan alone.
The Vulkan benchmark groups do not execute if a compatible runtime or device is unavailable.

To compare CPU and Vulkan on 10 million vectors with 64 dimensions, use this command:

```bash
cargo bench --bench backend_compare -- \
  --n 10000000 --dims 64 --k 10 --samples 20 --rounds 2
```

This benchmark generates the same data from a fixed random seed.
It executes one backend at a time and reverses their order in each round.
It uses four query vectors for six workloads:

- Ring
- Disk
- Three-disk intersection
- Exact KNN
- Weighted KNN
- KNN with a filter that selects 1% of the vectors.

The times include result transfers.
They do not include dataset generation, upload, warm-up queries, validation, and result destruction.
The benchmark gives median and 95th-percentile query times.
It writes each sample to `target/backend-compare.csv`.

The benchmark validates KNN IDs, result order, and distances.
For geometric queries, it examines membership differences with `f64` calculations near distance limits.
Differences more than the accepted tolerance cause an error.

To change a benchmark run, use these command-line options:

| Option | Function |
|---|---|
| `--probe` | Show Vulkan device limits without a dataset |
| `--n 100000 --samples 2 --rounds 1` | Use a small dataset for a short test |
| `--backend cpu` | Run CPU alone, without a Vulkan load |
| `--backend vulkan` | Run Vulkan alone |
| `--workload intersection3` | Run only the three-disk intersection |
| `--output path.csv` | Select a different CSV file |
| `--radius-scale` | Multiply geometric radii by this value; default `1` |
| `--intersection-shift` | Set the coordinate offset of the other two centers; default `0.01` |

A run with only one backend does not compare results with the other backend.
For 10 million vectors with 64 dimensions, vector data uses 2.56 GB.
Query buffers and results use more memory.

To test smaller chunks, set `RINGDB_VULKAN_CHUNK_BYTES` to a positive byte count.
This development option sets an upper limit for each vector buffer.
It cannot increase device limits.
Leave it unset for automatic chunk sizes.

This command divides 10 million vectors into chunks of at most 5 million vectors:

```bash
RINGDB_VULKAN_CHUNK_BYTES=1280000000 cargo bench --bench backend_compare -- \
  --n 10000000 --dims 64 --k 10 --samples 20 --rounds 2
```

For timing measurements, disable Vulkan validation layers.
If necessary, use `VK_DRIVER_FILES` to select a Vulkan driver manifest.

To log Vulkan GPU times, set `RINGDB_VULKAN_TIMINGS=1`.
The selected compute queue must have timestamp support for this option.
The logged times include GPU work, barriers, and transfers in each submission.
They do not include CPU work and do not give the full query time.

For comparisons of full query times, keep `RINGDB_VULKAN_TIMINGS` unset.

## Command-line example

To search one metadata category, run this command:

```bash
cargo run --release --example cli -- --n 100000 --dims 64 --k 10 --category 3
```

For the API reference, run `cargo doc --open`.

## License

MIT OR Apache-2.0.
