# ring-db

ring-db is a Rust library for exact vector search on a CPU or a GPU with Vulkan.
It finds the closest vectors, or all vectors in a region made of one or more disks.
Each record has a vector and typed fields.
You can filter records by their fields and set a weight for each dimension.

Exact search calculates distances for all records that pass the filter.
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

This example creates a database with two dimensions and no fields.
It finds the vector closest to `[0.0, 0.0]`.
The `dist_sq` field contains the squared distance.

```rust
use ringdb::{Db, Knn, Options, Record, Schema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Db::in_memory(Schema::new(2), Options::default())?;
    db.add(Record::new([3.0, 4.0]))?;
    let near = db.add(Record::new([1.0, 0.0]))?;
    db.flush()?;

    let origin = [0.0, 0.0];
    let hits = db.search(Knn::new(&origin, 1))?;

    assert_eq!(hits[0].key, near);
    assert_eq!(hits[0].dist_sq, 1.0);
    Ok(())
}
```

`add` returns a new `Key` for each record.
ring-db does not use a key again, also after you delete its record.
`key.get()` returns the key as a `u64` value, and `Key::new` makes a key from this value.

Searches and `get` find a record after the next commit.
`flush` commits all changes immediately.

`Db::in_memory` keeps the database in memory only.
To save the database to disk, use `Db::open`.
All methods take `&self`.
To use a database from more than one thread, put it in an `Arc`.

## Save the database to disk

`Db::open(path, schema, options)` opens the database in the directory `path`.
If the directory has no database, ring-db creates a new database in it.

```rust
use ringdb::{Db, FieldType, Options, Record, Schema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join("ring-db-readme-open");
    let _ = std::fs::remove_dir_all(&dir);
    let schema = Schema::new(2).field("rooms", FieldType::U32);

    let db = Db::open(&dir, schema.clone(), Options::default())?;
    let key = db.add(Record::new([1.0, 2.0]).set("rooms", 3u32))?;
    drop(db);

    let db = Db::open(&dir, schema, Options::default())?;
    assert_eq!(db.get(key)?.vector(), [1.0, 2.0]);
    drop(db);
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
```

ring-db writes each add and delete to a log file on disk before the method returns.
Thus, a change stays in the database after a crash or a power failure.
Changes that were not in a commit before a crash become visible when you open the database again.

The schema of an existing database cannot change.
If the `schema` argument is different, `Db::open` returns `InvalidSchema`.
Only one `Db` can open a directory at a time.
A second `Db::open` of the same directory returns an `Io` error.

Each write to the log waits until the disk keeps the data.
Thus, one `add` can take some milliseconds.
To add many records, use `db.add_all(records)`.
It writes all records with one log write and returns their keys in order.
If one record is not valid, `add_all` adds no records.

`db.backup(dest)` makes a commit, then copies the database files into the directory `dest`.
The directory `dest` must not exist, or it must be empty.
To use the copy, open it with `Db::open`.
Changes wait until the copy is complete.

The database directory contains these files:

| File | Contents |
|---|---|
| `LOCK` | A lock that prevents a second `Db::open` |
| `MANIFEST` | The schema, the segment files, and the deleted records |
| `seg-*.seg` | The records of one segment |
| `wal-*.log` | The changes after the last commit |

Each file has a checksum.
If a file is damaged, `Db::open` returns `Corrupt`.
If a crash stops a log write, ring-db ignores the incomplete change.
Do not change the files in the directory.

## Delete and change records

`db.delete(key)` deletes one record.
It returns `KeyNotFound` if the record does not exist or is already deleted.
You cannot change a record.
To change a record, delete it and add the new version, which gets a new key.

ring-db collects changes and applies them in a commit.
A commit makes all changes since the previous commit visible at the same time.
A background thread makes a commit at each `Options::commit_interval`, one second by default.
`db.flush()` makes a commit immediately.
To make commits only with `flush`, set `commit_interval(None)`.

A delete and an add are visible at the same time if no commit occurs between them.
Thus, to change a record without a commit between the two operations, use `commit_interval(None)` and `flush`.

Each commit adds a segment that contains the new records.
After a commit, ring-db merges small segments into larger segments in the background.
A merge also removes deleted records from storage.
Searches do not stop during a merge.
To stop automatic merges, set `auto_merge(false)`.

`db.compact()` merges all segments into one segment.
A search on one segment is the fastest.
Use `compact` after you add many records, for example after an import.

## Select CPU or Vulkan

The two backends can execute all search types, weights, and filters.
The backend selection occurs at runtime.

| Backend | Selection | Runtime requirements |
|---|---|---|
| CPU | `BackendPreference::Cpu`, or the default | No GPU runtime |
| Vulkan | `BackendPreference::Vulkan` | A Vulkan 1.2 loader, driver, and compatible compute device |

To use Vulkan, set the backend in `Options`:

```rust,no_run
use ringdb::{BackendPreference, Db, Options, Record, Schema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options::default().backend(BackendPreference::Vulkan);
    let db = Db::in_memory(Schema::new(64), options)?;
    db.add(Record::new([0.0; 64]))?;
    db.flush()?;
    assert_eq!(db.backend(), BackendPreference::Vulkan);
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

The Vulkan backend uploads the vectors of each segment one time and keeps them in device memory.
ring-db searches small segments on the CPU, because a CPU search of a small segment is faster than a GPU dispatch.
By default, a segment is small if it has fewer than 16384 coordinates, for example 256 vectors with 64 dimensions.
To change this limit, set `Options::vulkan_min_rows(rows)`.
To search all segments on the GPU, set `vulkan_min_rows(0)`.
It executes queries on one database in sequence.
It keeps query buffers for subsequent queries until you drop the database.
Each result owns its data and stays valid after subsequent queries.

Vulkan uses 64-bit device addresses to access vector data and query buffers.
A single vector buffer can be more than 4 GiB.
The storage-buffer binding limit does not apply to these device addresses.

If a device's allocation limit is too small, ring-db divides a segment into groups of vectors called chunks.
Queries combine the results from all segments and chunks and keep the original keys.
KNN returns the closest vectors across the full dataset.
All vectors and query buffers must fit the available device memory.
An allocation that is too large causes a backend error.
Floating-point rounding can cause small differences between CPU and Vulkan results, especially near distance limits.

To run the example with Vulkan, use this command:

```bash
cargo run --release --example cli -- --backend vulkan --n 100000 --dims 64 --k 10
```

## Schemas and fields

A `Schema` gives the number of dimensions and the fields of each record.
You cannot change the schema after you create the database.
All vectors must have this number of dimensions, which must be more than zero.

Each field has a name and a `FieldType`:

| Type | Values | Filter |
|---|---|---|
| `U32`, `U64`, `I64` | Integers | Equality and ranges |
| `F32` | Finite `f32` numbers | Equality and ranges |
| `Bool` | `true` or `false` | Equality |
| `Keyword` | One string, such as a city | Exact equality |
| `Keywords` | Zero or more strings, such as amenities | Exact equality with one string |
| `Text` | A UTF-8 string | Stored only |
| `TextList` | A list of UTF-8 strings | Stored only |
| `Bytes` | Bytes | Stored only |

All fields are optional.
Use `Record::set(field, value)` to give a value to a field.
A record has no value for the fields that you do not set.

An integer value can go into all integer types that can hold it.
For example, `set("rooms", 3)` is correct for a `U32` field.
For a `Keywords` or `TextList` field, you can set one string or a list of strings.

`add` examines each record against the schema.
It returns an error for an unknown field, an incorrect value type, or a field that you set two times.
A record must not be larger than `Options::max_record_bytes`.
The default limit is 64 KiB for the vector and all field values together.

## Find the closest open bakery

This example filters places by type and status.
It then searches the places that pass the two filters.
The coordinates give positions on a flat map with the same unit on both axes.
They are not latitude and longitude.

The weights `[4.0, 1.0]` multiply the squared difference on the first axis by four.
The second axis has a weight of one.

```rust
use ringdb::{Db, FieldType, Filter, Knn, Metric, Options, Record, Schema, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Schema::new(2)
        .field("name", FieldType::Text)
        .field("kind", FieldType::Keyword)
        .field("open", FieldType::Bool);
    let db = Db::in_memory(schema, Options::default())?;

    // Position, name, type, and status for each place.
    let places = [
        ([0.8, 1.5], "Fournil", "bakery", true),
        ([0.2, 0.3], "Kawa", "cafe", true),
        ([1.0, 0.4], "Mie Dorée", "bakery", false),
        ([2.0, 0.0], "Levain", "bakery", true),
    ];
    for (position, name, kind, open) in places {
        db.add(
            Record::new(position)
                .set("name", name)
                .set("kind", kind)
                .set("open", open),
        )?;
    }
    db.flush()?;

    let open_bakeries = Filter::eq("kind", "bakery") & Filter::eq("open", true);
    let origin = [0.0, 0.0];
    let weights = [4.0, 1.0];
    let hits = db.search(
        Knn::new(&origin, 1)
            .metric(Metric::WeightedL2(&weights))
            .filter(&open_bakeries),
    )?;

    let hit = hits.first().expect("one open bakery");
    let place = db.get(hit.key)?;
    let name = place.get("name").and_then(Value::as_str);
    assert_eq!(name, Some("Fournil"));
    println!("{name:?} is at distance {}", hit.dist_sq.sqrt());
    Ok(())
}
```

The search compares the two open bakeries.
It does not include the cafe and the closed bakery, although they are closer to the origin.
The result contains the bakery with the name `Fournil`.

## Filters

A `Filter` selects records by their field values.
Each search can have one filter.
To apply it, call `.filter(&filter)` on the search.
Without this call, the search examines all records.

| Filter | Selected records |
|---|---|
| `Filter::eq(field, value)` | The field is equal to `value`. For `Keywords`, the field contains `value` |
| `Filter::any_of(field, values)` | The field is equal to, or contains, one or more of `values` |
| `Filter::range(field, low..high)` | The numeric field is in the range |
| `Filter::keys(keys)` | The record has one of `keys` |
| `a & b` | The records in `a` and in `b` |
| `a \| b` | The records in `a`, in `b`, or in the two filters |
| `!a` | All records not in `a` |

Ranges can use all Rust range forms, for example `100..=400`, `..3`, or `50.0..`.
Keyword filters compare full strings, with the same case.

A record with no value for a field does not pass a filter on that field.
Thus `!Filter::eq("garden", true)` includes the records with no `garden` value.

ring-db examines a filter against the schema before the search.
It returns `UnknownField` for an unknown field.
It returns `InvalidQuery` for a stored-only field or an incorrect value type.

## Searches and distances

ring-db has two search types: K-nearest-neighbor (KNN) search and region search.
Give each search to `db.search(...)`.
All distance limits include their boundary values.

| Search | Result | Condition |
|---|---|---|
| `Knn::new(&query, k)` | `Vec<Hit>` | The `k` closest vectors that pass the filter |
| `Disk::new(&center, radius)` | `RegionResult` | At most `radius` from `center` |
| `disk_a.and(disk_b)` | `RegionResult` | In all disks |

A region is the intersection of one or more disks.
A disk is a region with one disk.
To add a disk to a region, call `.and(disk)`.
A region can contain a maximum of `MAX_DISKS` (16) disks.

This example finds the vectors in two disks:

```rust
use ringdb::{Db, Disk, Options, Record, Schema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Db::in_memory(Schema::new(2), Options::default())?;
    db.add(Record::new([0.0, 0.0]))?;
    let middle = db.add(Record::new([1.0, 0.0]))?;
    db.add(Record::new([3.0, 0.0]))?;
    db.flush()?;

    let left = [0.0, 0.0];
    let right = [2.0, 0.0];
    let region = Disk::new(&left, 1.5).and(Disk::new(&right, 1.5));
    let result = db.search(region)?;

    assert_eq!(result.len(), 1);
    let hit = result.get(0).expect("one vector in the two disks");
    assert_eq!(hit.key, middle);
    assert_eq!(hit.dists_sq(), &[1.0, 1.0]);
    Ok(())
}
```

For KNN, `k = 0` gives no results.
If fewer than `k` vectors pass the filter, the search returns all of them.
KNN sorts hits by squared distance, then by key.
For equal distances, the smaller key comes first.
Each `Hit` contains the `key` and the squared distance `dist_sq`.

A `RegionResult` contains one `RegionHit` for each vector in the region.
Each region hit contains the `key`.
`dists_sq()` returns one squared distance for each disk, in the disk order.
The order of region hits can change between searches.

The default distance is Euclidean distance, also called L2 distance.
ring-db calculates its square:

```text
dist_sq = sum((vector[i] - query[i])^2)
```

To use diagonal weights, call `.metric(Metric::WeightedL2(&weights))`:

```text
dist_sq = sum(weights[i] * (vector[i] - query[i])^2)
```

ring-db uses each weight as supplied.
It does not square the weights.
A zero weight removes that dimension from the distance calculation.
Each KNN search and each disk can use different weights.
Full weight matrices are not available.

Search inputs have these requirements:

- Vectors, query vectors, centers, and weight arrays must have the database dimension count.
- Vector coordinates, query coordinates, and centers must be finite, without `NaN` or infinity.
- Weights and radii must be finite and non-negative.
- A region must contain 1 to 16 disks.

To get the distance from a squared distance, use `dist_sq.sqrt()`.

## Read a record

`db.get(key)` returns a `Document` with the key, the vector, and the field values.
It returns `KeyNotFound` if no visible record has this key.

`document.get(field)` returns the `Value` of one field, or `None` if the record has no value.
Use the `Value` methods, such as `as_u32()` or `as_str()`, to read the value.
`document.fields()` returns all fields that have a value, in schema order.

A `Keywords` value contains its strings in sorted order, without duplicates.
An empty list has the same result as no value.

`db.snapshot()` returns a `Snapshot`.
A snapshot is a view of the database at one time.
Searches and `get` on a snapshot do not see the records that you add after this time.

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
RINGDB_BENCH_N=100000 cargo bench --bench query_backends cpu_knn
RINGDB_BENCH_N=100000 cargo bench --bench query_backends cpu_region
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
It uses four query vectors for five workloads:

- Disk
- Three-disk intersection
- Exact KNN
- Weighted KNN
- KNN with a filter that selects 1% of the vectors.

The times include result transfers.
They do not include dataset generation, upload, warm-up queries, validation, and result destruction.
The benchmark gives median and 95th-percentile query times.
It writes each sample to `target/backend-compare.csv`.

The benchmark validates KNN keys, result order, and distances.
For region searches, it examines membership differences with `f64` calculations near distance limits.
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
| `--radius-scale` | Multiply disk radii by this value; default `1` |
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

To search one category, run this command:

```bash
cargo run --release --example cli -- --n 100000 --dims 64 --k 10 --category 3
```

For the API reference, run `cargo doc --open`.

## License

MIT OR Apache-2.0.
