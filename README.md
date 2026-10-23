# ring-db

**Exact vector search with filters, on a CPU or a GPU.**

ring-db is a Rust library that keeps records in a database.
Each record has a vector and some fields, for example a position, a price, and a city.
ring-db finds the records that are closest to a point, or all the records in an area.
You can filter the records by their fields.

ring-db always gives exact results.
It examines each record that passes the filter.
It does not use an approximate index.

- **Two searches**: the closest records (KNN), and the records in an area made of disks.
- **Typed fields and filters**: numbers, booleans, keywords, and ranges.
- **Changes**: add and delete records, alone or together in an atomic batch.
- **Durable storage**: a log that stays correct after a crash, checksums, and backups.
- **CPU or GPU**: the CPU works everywhere. Vulkan GPUs are optional.

## Is ring-db correct for your project?

| Use ring-db if you… | Use a different tool if you… |
|---|---|
| need exact results | can accept approximate results for more speed |
| have up to some tens of millions of records | have billions of records |
| filter records by their fields | need SQL or a database server |
| add or delete some records each second | write thousands of records each second |

ring-db is a library, not a server.
It runs in the process of your application.

## Install

Add ring-db to `Cargo.toml`:

```toml
[dependencies]
ring-db = "0.9"
```

The crate name is `ring-db`, and the library name is `ringdb`.
You do not need a GPU, a Vulkan SDK, or Cargo features to use the CPU.

## Quick start

This example keeps apartments for sale in a database.
It finds the two apartments with two rooms or more that are closest to a point.

```rust
use ringdb::{Db, FieldType, Filter, Knn, Options, Record, Schema, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Describe the records: a vector with 2 dimensions, and 3 fields.
    let schema = Schema::new(2)
        .field("title", FieldType::Text)
        .field("rooms", FieldType::U32)
        .field("price", FieldType::U64);
    let db = Db::in_memory(schema, Options::default())?;

    // 2. Add records. Here, the vector is a position on a map, in kilometers.
    let studio = Record::new([0.5, 0.2])
        .set("title", "Studio near the station")
        .set("rooms", 1)
        .set("price", 150_000);
    let apartment = Record::new([1.0, 1.0])
        .set("title", "Apartment with a balcony")
        .set("rooms", 3)
        .set("price", 320_000);
    let house = Record::new([4.0, 3.0])
        .set("title", "House with a garden")
        .set("rooms", 5)
        .set("price", 450_000);
    db.add_all([studio, apartment, house])?;

    // 3. Make the new records visible to searches now.
    db.flush()?;

    // 4. Find the 2 closest records with 2 rooms or more.
    let two_rooms_or_more = Filter::range("rooms", 2..);
    let point = [0.0, 0.0];
    let hits = db.search(Knn::new(&point, 2).filter(&two_rooms_or_more))?;

    // 5. Read the results.
    for hit in &hits {
        let record = db.get(hit.key)?;
        let title = record.get("title").and_then(Value::as_str).unwrap_or("");
        println!("{title}: {:.1} km", hit.dist_sq.sqrt());
    }
    assert_eq!(hits.len(), 2);
    Ok(())
}
```

The example prints this text:

```text
Apartment with a balcony: 1.4 km
House with a garden: 5.0 km
```

The studio is the closest record, but it has only one room.
Thus, the filter removes it.

## Main concepts

| Term | Meaning |
|---|---|
| Record | One vector and its field values. You add records with `add`. |
| Key | The identifier that `add` gives to a record. ring-db does not use a key two times. |
| Schema | The number of dimensions and the fields of each record. You cannot change it. |
| Commit | The step that makes changes visible to searches. It occurs each second, or when you call `flush`. |
| Segment | A block of records. Each commit adds one. ring-db merges small segments in the background. |
| Snapshot | A view of the database at one time. Each search uses a snapshot, so changes do not affect it. |

## Guide

### 1. Define the schema

A `Schema` gives the number of dimensions and the fields of each record.
All vectors must have this number of dimensions.

| Field type | Values | Filters |
|---|---|---|
| `U32`, `U64`, `I64` | Integers | Equality and ranges |
| `F32` | Finite numbers | Equality and ranges |
| `Bool` | `true` or `false` | Equality |
| `Keyword` | One string, for example a city | Equality |
| `Keywords` | Zero or more strings, for example amenities | Equality with one of the strings |
| `Text` | One string, for example a title | No filter. ring-db only keeps the value. |
| `TextList` | A list of strings, for example photo URLs | No filter. ring-db only keeps the value. |
| `Bytes` | Bytes | No filter. ring-db only keeps the value. |

All fields are optional.
A record has no value for a field that you do not set.

### 2. Add records

`Record::new(vector)` makes a record.
`.set(field, value)` gives a value to one field.

- `db.add(record)` adds one record and returns its `Key`.
- `db.add_all(records)` adds many records and returns their keys in the same order.

ring-db examines each record before it adds it.
It returns an error for an unknown field, a value of an incorrect type, or a field that you set two times.
If one record is not correct, `add_all` adds no records.

An integer goes into all integer fields that can hold it.
For example, `.set("rooms", 3)` is correct for a `U32` field.
For a `Keywords` or `TextList` field, set one string or a list, for example `["balcony", "lift"]`.

A record must not be larger than 64 KiB, with its vector and its values.
To change this limit, use `Options::max_record_bytes`.

A search does not find a new record before the next commit.
To make a commit immediately, call `db.flush()`.

### 3. Find the closest records

`Knn::new(&point, k)` finds the `k` records closest to `point`.
KNN means "k nearest neighbors".

```rust
use ringdb::{Db, Knn, Options, Record, Schema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Db::in_memory(Schema::new(2), Options::default())?;
    db.add(Record::new([3.0, 4.0]))?;
    let near = db.add(Record::new([1.0, 0.0]))?;
    db.flush()?;

    let hits = db.search(Knn::new(&[0.0, 0.0], 1))?;
    assert_eq!(hits[0].key, near);
    assert_eq!(hits[0].dist_sq, 1.0);
    Ok(())
}
```

Each `Hit` has the `key` of a record and its squared distance `dist_sq`.
To get the distance, use `dist_sq.sqrt()`.

The hits are in order of distance, then in order of key.
If fewer than `k` records pass the filter, you get all of them.

### 4. Find the records in an area

`Disk::new(&center, radius)` finds all records at a distance of `radius` or less from `center`.
To find the records in two or more disks at the same time, join the disks with `.and`.
This area is a region.
A region can have a maximum of 16 disks.

```rust
use ringdb::{Db, Disk, Options, Record, Schema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Db::in_memory(Schema::new(2), Options::default())?;
    db.add(Record::new([0.0, 0.0]))?;
    let middle = db.add(Record::new([1.0, 0.0]))?;
    db.add(Record::new([3.0, 0.0]))?;
    db.flush()?;

    // The records in the two disks.
    let left = [0.0, 0.0];
    let right = [2.0, 0.0];
    let region = Disk::new(&left, 1.5).and(Disk::new(&right, 1.5));
    let result = db.search(region)?;

    assert_eq!(result.len(), 1);
    let hit = result.get(0).expect("one record");
    assert_eq!(hit.key, middle);
    assert_eq!(hit.dists_sq(), &[1.0, 1.0]); // One distance for each disk.
    Ok(())
}
```

A search for a region returns a `RegionResult`.
Use `result.iter()` to read all hits, or `result.keys()` to read only the keys.
Each hit has one squared distance for each disk, in the order of the disks.
The order of the hits can change between searches.

### 5. Give a weight to each dimension

ring-db uses the Euclidean distance by default:

```text
dist_sq = sum((vector[i] - point[i])^2)
```

To make some dimensions more important, give a weight to each dimension:

```text
dist_sq = sum(weights[i] * (vector[i] - point[i])^2)
```

Call `.metric(Metric::WeightedL2(&weights))` on a `Knn` or a `Disk`.
Each search and each disk can have different weights.
A weight of zero removes that dimension from the distance.
Weights must be zero or more.

### 6. Filter the records

A `Filter` selects records by their field values.
Each search can have one filter.
Add it with `.filter(&filter)`.

| Filter | Selected records |
|---|---|
| `Filter::eq(field, value)` | The field has this value. For `Keywords`, the list contains this value. |
| `Filter::any_of(field, values)` | The field has one of these values. |
| `Filter::range(field, range)` | The number is in the range, for example `100..=400`, `..3`, or `2.5..`. |
| `Filter::keys(keys)` | The record has one of these keys. |
| `a & b` | The records in `a` and in `b`. |
| `a \| b` | The records in `a`, in `b`, or in the two. |
| `!a` | The records that are not in `a`. |

```rust
use ringdb::{Db, Disk, FieldType, Filter, Options, Record, Schema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Schema::new(2)
        .field("city", FieldType::Keyword)
        .field("amenities", FieldType::Keywords)
        .field("price", FieldType::U64);
    let db = Db::in_memory(schema, Options::default())?;
    let lyon = db.add(
        Record::new([0.0, 1.0])
            .set("city", "Lyon")
            .set("amenities", ["balcony", "lift"])
            .set("price", 280_000),
    )?;
    db.add(Record::new([1.0, 0.0]).set("city", "Paris").set("price", 510_000))?;
    db.flush()?;

    // In Lyon or Nantes, with a balcony, for 300 000 or less.
    let filter = Filter::any_of("city", ["Lyon", "Nantes"])
        & Filter::eq("amenities", "balcony")
        & Filter::range("price", ..=300_000);
    let center = [0.0, 0.0];
    let result = db.search(Disk::new(&center, 5.0).filter(&filter))?;
    assert_eq!(result.keys().collect::<Vec<_>>(), [lyon]);
    Ok(())
}
```

Keyword filters compare the full string, with the same uppercase and lowercase letters.

A record with no value for a field does not pass a filter on that field.
Thus, `!Filter::eq("garden", true)` includes the records with no value for `garden`.

### 7. Read a record

`db.get(key)` returns a `Document` with the key, the vector, and the field values.

- `document.vector()` returns the vector.
- `document.get(field)` returns the `Value` of one field, or `None`.
- `document.fields()` returns all fields that have a value.

To read a `Value`, use a method, for example `as_u32()`, `as_str()`, or `as_strs()`.
A `Keywords` value has its strings in alphabetical order, with no duplicates.

`db.snapshot()` returns a `Snapshot`, which has the same `search` and `get` methods.
Use a snapshot when two or more searches must see the same records.

### 8. Delete and replace records

`db.delete(key)` deletes one record.
You cannot change a record.
To change a record, delete it and add the new version.
The new version gets a new key.

Put the two changes in one `Batch`.
Then, all searches find the old version or the new version, never the two and never none.

```rust
use ringdb::{Batch, Db, FieldType, Options, Record, Schema, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Schema::new(2).field("price", FieldType::U64);
    let db = Db::in_memory(schema, Options::default())?;
    let old = db.add(Record::new([1.0, 2.0]).set("price", 300_000))?;

    // The price decreases: replace the record.
    let new = Record::new([1.0, 2.0]).set("price", 280_000);
    let keys = db.apply(Batch::new().delete(old).add(new))?;
    db.flush()?;

    assert!(db.get(old).is_err());
    let price = db.get(keys[0])?.get("price").and_then(Value::as_u64);
    assert_eq!(price, Some(280_000));
    Ok(())
}
```

`apply` returns the keys of the added records, in the order of the batch.
If a key is not found or a record is not correct, `apply` makes no change and returns an error.

Changes become visible at the next commit.
A background thread makes a commit each second.
`db.flush()` makes a commit immediately.
To make commits only when you call `flush`, use `Options::commit_interval(None)`.

### 9. Save the database to disk

`Db::in_memory` keeps the database in memory only.
To keep it on disk, use `Db::open(directory, schema, options)`.
If the directory has no database, ring-db makes a new database in it.

```rust
use ringdb::{Db, FieldType, Options, Record, Schema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join("ring-db-readme-open");
    let _ = std::fs::remove_dir_all(&dir);
    let schema = Schema::new(2).field("rooms", FieldType::U32);

    let db = Db::open(&dir, schema.clone(), Options::default())?;
    let key = db.add(Record::new([1.0, 2.0]).set("rooms", 3))?;
    drop(db);

    // Open the database again: the record is there.
    let db = Db::open(&dir, schema, Options::default())?;
    assert_eq!(db.get(key)?.vector(), [1.0, 2.0]);
    drop(db);
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
```

Each change is on the disk before `add`, `add_all`, `delete`, or `apply` returns.
Thus, your changes stay after a crash or a power failure.
After a crash, a batch is fully in the database, or it is not in the database.

Each write waits for the disk, which can take some milliseconds.
To add many records, use one `add_all`, not many `add` calls.

- `Db::open` returns `InvalidSchema` if the schema is different from the schema on the disk.
- Only one `Db` can open a directory at a time. A second `Db::open` returns an `Io` error.
- Each file has a checksum. If a file is damaged, `Db::open` returns `Corrupt`.
- `db.backup(directory)` copies the database to a new, empty directory. Open the copy with `Db::open`.

Do not change the files in the database directory.

### 10. Use a GPU

ring-db can search on a GPU with Vulkan.
Select the backend in `Options`:

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

The two backends give the same results, with all searches, weights, and filters.
Small rounding differences can occur near the edge of a disk.

Vulkan needs a Vulkan 1.2 loader, a GPU driver, and a compatible device.
NVIDIA, AMD, and integrated GPUs can be compatible.
If no compatible device is available, `Db::in_memory` and `Db::open` return `BackendUnavailable`.
ring-db does not change to the CPU automatically.

ring-db copies the vectors to the GPU one time and keeps them there.
All vectors must fit in the GPU memory.
ring-db searches small segments on the CPU, because the CPU is faster for them.
To change this limit, use `Options::vulkan_min_rows`.

## Reference

### Options

| Method | Default | Function |
|---|---|---|
| `backend(preference)` | `BackendPreference::Cpu` | Selects the CPU or Vulkan. |
| `commit_interval(interval)` | One second | Sets the time between background commits. `None` stops them. |
| `auto_merge(enabled)` | `true` | Merges small segments after each commit. |
| `max_record_bytes(bytes)` | 64 KiB | Sets the largest size of one record. |
| `vulkan_min_rows(rows)` | 16384 values divided by the dimensions | Sets the smallest segment that Vulkan searches. `0` sends all segments to the GPU. |

### Errors

| Error | Cause |
|---|---|
| `DimensionMismatch` | A vector, a point, or a weight array has an incorrect number of dimensions. |
| `InvalidRecord` | A record has an incorrect value, or it is too large. |
| `UnknownField` | A record or a filter uses a field that is not in the schema. |
| `InvalidQuery` | A search, a filter, or a vector is not correct, for example a coordinate that is not finite, or a filter on a `Text` field. |
| `KeyNotFound` | No record has this key, or the record is deleted. |
| `InvalidSchema` | The schema is not correct, or it is different from the schema on the disk. |
| `Io` | A file operation failed, or a different `Db` has the directory open. |
| `Corrupt` | A database file is damaged. |
| `BackendUnavailable` | Vulkan, its driver, or a compatible device is not available. |
| `Backend` | A GPU operation failed. |

### Limits

- Vector coordinates, points, and centers must be finite numbers.
- Weights and radii must be finite and zero or more.
- A region has 1 to 16 disks (`MAX_DISKS`).
- A database has fewer than 4 294 967 295 rows. Deleted rows count until a merge removes them.

### Tips for speed

- After a large import, call `db.compact()`. A search on one segment is the fastest.
- To add many records, use `add_all`, which writes to the disk one time.
- Keyword filters and narrow ranges are fast. A filter that selects few records makes the search faster.
- All methods take `&self`. To use one database from many threads, put it in an `Arc`.

## Command-line example

This command adds 100 000 random vectors and searches one category:

```bash
cargo run --release --example cli -- --n 100000 --dims 64 --k 10 --category 3
```

To use Vulkan, add `--backend vulkan`.
For the API reference, run `cargo doc --open`.

## Development

Run the standard checks:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

`cargo test` also tests the Rust examples in this README.
It does not need a GPU.

To compare Vulkan results with CPU results, use a hardware or software Vulkan device:

```bash
RINGDB_VULKAN_TEST=1 cargo test --test vulkan
```

To test vector addresses above 4 GiB, use this command.
It needs more than 4 GiB of GPU memory.

```bash
RINGDB_VULKAN_TEST=1 RINGDB_VULKAN_LARGE_TEST=1 cargo test --test vulkan vulkan_reads_vectors_past_four_gib
```

The crate contains compiled shaders.
To change a shader in `src/backend/vulkan/shaders`, install `glslc` and `spirv-val`.
Then compile and validate the shaders:

```bash
sh src/backend/vulkan/shaders/compile.sh
```

## Benchmarks

The benchmarks use 30 million vectors by default.
To use fewer vectors, set `RINGDB_BENCH_N`:

```bash
RINGDB_BENCH_N=100000 cargo bench --bench query_backends
RINGDB_BENCH_N=100000 cargo bench --bench query_backends cpu_knn
RINGDB_BENCH_N=100000 cargo bench --bench query_backends cpu_region
```

`RINGDB_VULKAN_BENCH_N` sets a different number of vectors for Vulkan.
If Vulkan is not available, the Vulkan benchmarks do not run.

To compare the CPU and Vulkan on the same data, use this command:

```bash
cargo bench --bench backend_compare -- \
  --n 10000000 --dims 64 --k 10 --samples 20 --rounds 2
```

This benchmark runs five searches: a disk, an intersection of three disks, KNN, weighted KNN, and KNN with a filter.
It runs one backend at a time and compares their results.
It gives the median and 95th-percentile times, and it writes all samples to `target/backend-compare.csv`.
The times include the transfer of results, but not the upload of the data.

| Option | Function |
|---|---|
| `--probe` | Shows the limits of the Vulkan device. It makes no data. |
| `--n 100000 --samples 2 --rounds 1` | Makes a short test with little data. |
| `--backend cpu` or `--backend vulkan` | Runs one backend only. It does not compare results. |
| `--workload intersection3` | Runs only the intersection of three disks. |
| `--output path.csv` | Writes the samples to a different file. |
| `--radius-scale` | Multiplies the disk radii. The default is `1`. |
| `--intersection-shift` | Moves the centers of the second and third disks. The default is `0.01`. |

These environment variables help you examine Vulkan:

| Variable | Function |
|---|---|
| `RINGDB_VULKAN_CHUNK_BYTES` | Sets the largest GPU buffer for vectors, in bytes, to test chunks. |
| `RINGDB_VULKAN_TIMINGS=1` | Logs the GPU time of each submission. Do not use it to measure full search times. |
| `VK_DRIVER_FILES` | Selects a Vulkan driver. |

For correct times, do not use the Vulkan validation layers.

## License

MIT OR Apache-2.0.
