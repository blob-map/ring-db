# AGENT.md — ring-db

Quick context for contributors and coding agents.

## Product definition

ring-db is a Rust 2024 exact vector database. It supports:

- exact K-nearest-neighbor search (`Knn`);
- region search: the intersection of 1 to `MAX_DISKS` disks (`Disk`, `Region`);
- ordinary L2 and diagonal weighted L2, per KNN query or per disk (`Metric`);
- a fixed `Schema` of typed, optional fields per record;
- typed `Filter`s over numeric, bool, and keyword fields;
- stored-only text, text-list, and byte fields read back with `get`;
- generated, never-reused `Key`s, `delete`, atomic `Batch`es, and commits;
- background commits and size-tiered merges of immutable segments;
- durable storage: a synced WAL, segment files, an atomic manifest, and `backup`;
- CPU and Vulkan backends selected at runtime.

ring-db is meant to be the source of truth for its records. Records are added and
deleted, never changed in place. `Db::open` is durable; `Db::in_memory` is not.

## Engineering priorities

1. Prefer the simplest complete design.
2. Remove abstractions that no longer serve more than one real implementation.
3. Keep cyclomatic complexity low with small preparation and execution helpers.
4. Keep validation outside hot coordinate loops.
5. Preserve branch-free unfiltered paths.
6. Do not make compute kernels interpret field values; filters become row masks first.
7. Add functionality only for a demonstrated use case.

## Documentation standard

Always write `README.md` in [ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/).
Use [Issue 9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf), including its dictionary.
Use approved meanings and parts of speech. Keep instructions active and imperative.
Limit instruction sentences to 20 words and descriptions to 25 words.
Give each paragraph one topic and no more than six sentences.
Use one instruction per sentence, consistent terms, and American English spelling.
Review meanings as well as sentence length; automated checks cannot establish compliance.

Project technical nouns include backend, Cargo, database, dimension, disk, driver,
field, filter, GPU, key, keyword, KNN, loader, record, region, schema, segment,
shader, snapshot, and vector.
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
├── config.rs          Options and BackendPreference
├── db.rs              Db handle, Batch, write path, commits, merges
├── snapshot.rs        Snapshot, its Parts, and multi-segment search
├── schema.rs          Schema and FieldType
├── value.rs           Value, Record, validated Row, Document
├── filter.rs          public Filter tree and its schema-checked Compiled form
├── segment.rs         immutable Segment and SegmentBuilder, mapped from files
├── column.rs          typed columns, keyword dictionaries, inverted indexes
├── buf.rs             Buf: an immutable array, owned or borrowed from a mapping
├── rows.rs            adaptive RowSet (all, empty, sparse, dense)
├── store.rs           directory lock, WAL, MANIFEST, segment files, recovery
├── codec.rs           checked little-endian Encoder/Decoder with CRC-32
├── error.rs           RingDbError
├── query.rs           Knn, Disk, Region, Count, Metric, Key, Hit, RegionResult
└── backend/
    ├── mod.rs         kernel inputs: Mask, DiskSpec, RowHit
    ├── cpu.rs         Rayon-parallel exact reference backend
    └── vulkan/        Vulkan backend, runtime, and embedded SPIR-V shaders

tests/
├── common/mod.rs      shared builders and reference distances
├── search.rs          analytical search semantics
├── randomized.rs      comparison with an f64 reference
├── fields.rs          schemas, records, typed filters, segments, get
├── mutation.rs        deletes, commits, merges, concurrency against a model
├── persistence.rs     reopen, crash and torn-log recovery, corruption, lock, backup
└── vulkan.rs          CPU/Vulkan parity

examples/cli.rs        filtered weighted-KNN example
benches/query_backends.rs
benches/backend_compare.rs sequential CPU/Vulkan comparison with raw timing output
```

## Public data model

```rust
let schema = Schema::new(64)
    .field("price", FieldType::U64)
    .field("city", FieldType::Keyword)
    .field("tags", FieldType::Keywords)
    .field("title", FieldType::Text);
let db = Db::in_memory(schema, Options::default())?;
let key = db.add(Record::new(vector).set("price", 350_000).set("city", "Lyon"))?;
db.flush()?;
let doc = db.get(key)?;
```

Every field is optional. `Record::validate` checks names and coerces each `Value` to
its field's canonical variant: any integer variant fits any numeric type that holds
it exactly, and `Text` becomes a one-item list for list fields. It rejects non-finite
vectors and `F32` values (as `InvalidRecord`), duplicate fields, and records over `Options::max_record_bytes` (64 KiB by
default). Keys come from a monotonic `u64` counter, only after validation succeeds.

`Db` methods take `&self`. Shared state is behind an `Arc` so the commit thread can
hold it; `Drop` stops and joins that thread.

## Commits, deletes, and merges

The writer mutex holds the next key, the pending `SegmentBuilder`, deleted pending
rows, and visible keys to delete. `Db::apply` is the only write path: it validates the
whole `Batch`, then logs and applies it under one writer lock, so no commit splits a
batch. A commit freezes the builder into a `Segment`,
uploads it, and publishes adds and deletes in one new `Snapshot` (an `Arc` replaced
under an `RwLock`). Lock `writer` before `current`, always. Searches and `get` run
on a snapshot clone without locks. Uncommitted changes are invisible.

A segment that fails to upload stays in `Writer::unpublished` and is retried at the
next commit, so no record is lost. The commit thread keeps its last error in
`Shared::background_error` for `Db::take_background_error`; `flush` repeats the
commit and reports errors directly. A commit holds the writer lock while it builds, writes,
and uploads its segment: writes wait, which bounds the backlog; searches do not.

A snapshot `Part` is an `Arc<Segment>` plus its sorted deleted rows and a cached live
`RowSet` (`None` when nothing is deleted, which keeps the unfiltered `Mask::All` path).
Deletes are copy-on-write per part.

Merges run after commits (`Options::auto_merge`) and on `compact`, one at a time
under `Shared::merging`. `merge_range` selects adjacent parts only: first a part with
at least 25% deleted rows, else the newest run whose older neighbor has less than
twice its live rows. `build_merge` copies live rows without the writer lock;
`publish_merge` then takes it, remaps rows deleted meanwhile into the new segment,
and replaces the inputs. Only merges remove parts, so the inputs are still contiguous.

## Segments, columns, and filters

A segment holds rows sorted by key, row-major vectors, and one `Column` per field.
Segments in a snapshot are ordered by key with disjoint key ranges, so `get` finds a
segment by binary search on first keys, then a row by binary search on keys. Merges
must combine adjacent segments only, to keep this property.

Every array of a segment and its columns is a `Buf<T>`: a `Vec<T>` for segments
built in memory, or a view into the segment file's `Mmap`, which the `Buf` keeps
alive. Builders hold `Vec`s and freeze into `Buf`s.

- Scalars: `Buf<T>` plus a presence bitmap, omitted when every row has a value.
  Numeric scalars also keep `order`, the present rows sorted by value, built
  with the segment, stored in format 2, and rebuilt when format 1 is decoded.
  `select_range` binary-searches it: under 1/64 of rows it sorts the matches,
  under 1/32 it sets bitmap bits, else it scans the column. Bool columns (0 or 1
  in a `u8`) scan.
- Keyword and Keywords: a sorted per-segment dictionary, term ids in 1, 2, or 4
  bytes (the narrowest that holds the dictionary), and an inverted index. When no
  row has two terms, `ids` has one entry per row (0 = no term, else id + 1) and
  no offsets; otherwise per-row offsets give each row's sorted ids. Postings are
  sorted row lists, except terms in more than 1/32 of rows, which are bitmaps
  (`rows::prefer_dense`). There is no global dictionary, so each segment is
  self-contained.
- Offsets (strings, lists, keyword rows, postings) are `u32` until a value needs
  `u64`. Segment format 2 stores these; format 1 (0.10) is still decoded, with
  its postings rebuilt, and `compact` rewrites it.
- Text and Bytes: packed bytes with offsets plus presence; TextList: nested offsets.

`Filter::compile` checks a filter against the schema once per search. The compiled
filter evaluates per segment into a `RowSet` (All, Empty, Sparse, or Dense, chosen
by density); `RowSet::mask` hands it to kernels without conversion. Missing values
never match a predicate, so `Not` includes them. Integer bounds compile to inclusive
`i128` ranges, so out-of-type bounds select everything or nothing.

## Query semantics

Every search goes through `Db::search` or `Snapshot::search`, which take a sealed `Query`:

| Type | Output | Meaning |
|---|---|---|
| `Knn` | `Vec<Hit>` | exact best `k`, ordered by `(dist_sq, key)` |
| `Disk` | `RegionResult` | inclusive `dist_sq <= radius^2` |
| `Region` | `RegionResult` | vector must satisfy every disk (1..=`MAX_DISKS`) |
| `Count` | `usize` | number of records in a `Region` (`Region::count`, `Disk::count`) |

`Snapshot::count(&Filter)` (and `Db::count`) sums the eligible `RowSet` lengths
without reading vectors. On Vulkan, `Count` runs the scan in mode 2, which only
increments the output count word.

A disk is a one-disk region: `Disk::and` and `Disk::filter` return a `Region`.
`Knn::metric` and `Disk::metric` take `Metric::L2` or `Metric::WeightedL2(&w)`.
Each search has at most one `&Filter`; there are no per-disk filters.

`RegionResult` stores flat `u32` records `[row, d0..dN-1]` (distances as `f32`
bits). Rows are numbered across the snapshot's segments; keys resolve lazily
through per-segment `(base row, Arc<[Key]>)` parts. A part whose keys have no gaps
computes `keys[0] + (row - base)` without reading `keys`: GPU hits come back in
random row order, so reading the key array cost a cache miss per hit. `RegionHit::dists_sq`
returns every disk's squared distance in disk order; there is no combined score.
Do not split records into separate key and distance vectors: large fresh result
allocations page-fault, and the extra bytes cost several percent on region scans.

`k = 0` returns no hits. If fewer than `k` candidates exist, KNN returns all of them.
KNN drops hits whose `dist_sq` overflowed to infinity, as regions do.
Ties are deterministic by key: each segment returns its best `k` by `(dist_sq, row)`,
which is key order within a segment, and the snapshot merges them by `(dist_sq, key)`.

The metrics are:

```text
ordinary distance: dist_sq = sum((vector[i] - query[i])^2)
weighted:        dist_sq = sum(weights[i] * (vector[i] - query[i])^2)
```

Weights are search-only final coefficients. They must match `dims`, be finite, and be
non-negative. They are never persisted.

## CPU backend

The CPU backend is the correctness reference.

- Ordinary searches use direct coordinate deltas with four independent accumulators.
- Unweighted KNN accumulates coordinate deltas directly to avoid cancellation.
- Weighted distance has its own four-accumulator loop.
- `Distance` selects the kernels' distance functions once per call: on x86-64
  builds without the `fma` target feature, `Fma` wraps them in
  `#[target_feature(enable = "fma")]` when the CPU has FMA. Otherwise each
  `mul_add` is a libm call, 2–10× slower. All paths fuse every `mul_add`, so their
  bits are identical (`fma_distances_have_the_portable_bits`).
- Rayon partitions vector or candidate work, at least `MIN_JOB_VALUES` coordinates per
  job, so a small segment scans on one thread instead of waking the pool.
- `All` retains the existing scan with no candidate branch per row.
- `Sparse` iterates rows directly.
- `Dense` visits set bits word by word (`set_bits`), so clear bits cost nothing.
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
The GPU backend requires Vulkan 1.2, `bufferDeviceAddress`, and `shaderInt64`.
Check loader/device versions and both features before device creation; reject
unsupported devices with `BackendUnavailable`. CPU has no Vulkan requirement.

`src/backend/vulkan/mod.rs` exposes `VulkanDevice` and a `VulkanSegment` per segment. `runtime.rs`
owns Vulkan resources and synchronous submissions; its execution mutex serializes
the queue, command pool, and reusable query buffers shared by
region and KNN queries.
Each segment uploads its vectors when it is built, unless it has fewer rows than
`Options::vulkan_min_rows_for(dims)` (default 6 Mi coordinates). A search of a
segment also runs on the CPU when fewer rows than that pass its filter
(`Segment::gpu`): a GPU search costs about 115 µs before any work (measured on
an RTX 4080 at 64 dims), as long as a CPU scan of about 100K rows. Tests set
`vulkan_min_rows(0)` through `common::manual`. Region scans write a count word
followed by `[row, d0..dN-1]` records; readback appends them directly to the owned
`RegionResult` records. Buffer capacities grow on demand and remain allocated
until the backend is dropped; never expose references to this reusable storage.
Large host copies use up to four workers, including the caller, with scoped
helper threads. Do not use the caller's Rayon pool while holding the execution
mutex: work stealing could re-enter the backend. Join all helpers before unmapping
or reusing readback memory, including when spawning a helper fails.
Vectors and query buffers use Buffer Device Address, passed through push constants.
There are no descriptor sets. Shaders calculate explicit 64-bit byte addresses,
then dereference a scalar buffer reference. Do not replace these with large-array
indexing: Vulkan 1.2 does not guarantee 64-bit offset arithmetic in access chains.
No `VK_EXT_shader_64bit_indexing` extension is required by this approach.
Vectors are uploaded once, transposed independently within each chunk. Chunk row
counts fit the device's allocation limit for vectors, output, and KNN scratch.
`row_offset` is the chunk start inside its segment and is used to decode candidates;
`row_base` (the segment's first snapshot row) is added only to region output rows.
Sparse candidates are sliced by segment row;
dense candidates retain their bit offset when a chunk starts inside a word.
KNN merges each chunk's best results and preserves exact `(dist_sq, row)` ordering.
The three scan pipelines specialize All/Sparse/Dense candidate handling. GPU KNN
selection is a tournament tree without 64-bit atomics or subgroup requirements:
`reduce` builds every level once (each entry is the best slot of 128 below it),
then each `take` removes the root's slot, and `reduce` with `reduce_shift` rebuilds
only the one entry per level above it. A query costs one scan plus about
`levels × k` single-workgroup dispatches. When `k × 64` reaches the candidate
count, the host selects from all distances instead. Preserve deterministic `(dist_sq, row)` order
and zero-weight handling before coordinate subtraction.

`RINGDB_VULKAN_TIMINGS=1` enables optional GPU timestamps around each submission.
These measure the GPU command-buffer span, not full query latency. Keep profiling
disabled for ordinary benchmarks and check compute-queue timestamp support when
opting in. Do not assume a shader specialization improves performance: compare it
against the generic path across dimensions, weights, candidates, and selectivity.

Shader sources and precompiled SPIR-V live together in `shaders/`. Regenerate them
with `sh src/backend/vulkan/shaders/compile.sh` after every source change. This
development command requires `glslc` and `spirv-val`; Cargo embeds the binaries.
Compile and validate shaders for Vulkan 1.2. A single buffer can exceed 4 GiB;
`maxStorageBufferRange` does not apply to device-address accesses. Use the queried
`maxMemoryAllocationSize` to split datasets only when allocation limits require it.
All chunks stay resident, so sufficient device memory is still required.
`RINGDB_VULKAN_CHUNK_BYTES` optionally caps vector-buffer bytes for development
tests and benchmarks. It can only reduce chunk sizes, never raise device limits.
Leave it unset for automatic sizing. Tests set it in a child process so environment
changes cannot race other tests. Keep unaligned chunk boundaries covered.

Run device tests with `RINGDB_VULKAN_TEST=1 cargo test --test vulkan`.
An explicitly requested device test must fail if Vulkan is unavailable. CI uses
Mesa software Vulkan so shader execution is covered without a physical GPU.
Set `RINGDB_VULKAN_LARGE_TEST=1` with `RINGDB_VULKAN_TEST=1` to run the opt-in
4-GiB boundary regression. It puts known nearest neighbors on both sides of the
address boundary and exercises unfiltered, sparse, and dense queries. Run on a
device whose allocation limit exceeds the dataset size to test one large buffer.
Device tests also cover multi-segment databases.

## Persistence

`Store` (only for `Db::open`) lives in `Writer`, so every log write and manifest
write is serialized by the writer mutex. Directory files:

- `LOCK`: held with `File::try_lock` while a writing `Db` is open, and with
  `try_lock_shared` by read-only ones;
- `MANIFEST`: schema, `next_key`, `next_segment`, and each part's segment id and
  deleted rows, with a CRC; replaced by tmp file + rename + directory fsync;
- `wal-{seq:016x}.log`: `[len u32][crc u32][payload]` frames. A payload is the
  `LogFrame` of one `Db::apply`: its deletes, then its adds (key + encoded `Row`);
- `seg-{id:016x}.seg`: an encoded `Segment` with a trailing CRC. Format 2 uses
  `Encoder::aligned`, which puts every slice at a multiple of 8 bytes, so
  `Decoder::buf` borrows each array from the mapping (on little-endian targets;
  otherwise it copies). Format 1 (0.10) is unaligned and mostly copied.

Segment files are never changed after they are written, which makes mapping them
sound. `Shared::prepare` writes a new segment's file, then replaces the built
segment with the mapped file (`store::read_segment`), so durable databases hold
no anonymous copy; `is_mapped` stops a retry from rewriting a mapped file. After
`Segment::upload`, `release_vectors` drops the mapped vector pages from the
process (`MADV_DONTNEED`); CPU searches fault them back. The Vulkan upload stages
`UPLOAD_STAGING_BYTES` of rows at a time and copies each dimension's run with
one `copy_regions` submission, so host memory stays bounded. Removing a mapped file
can fail on Windows; `remove_segment` ignores the error, and the next open
removes the file.

Each `apply` (and so `add`/`add_all`/`delete`) appends one frame and calls
`sync_data` before returning, so recovery keeps a whole batch or none of it. An
I/O error on the log leaves its end unknown, so later writes fail until the next
commit starts a new log (`Store::needs_new_log`; a commit runs for this even with
nothing pending). A commit writes and syncs the new
segment files, then the manifest naming them and a new WAL sequence, then deletes the
old WAL. If the manifest write fails, the segments stay unpublished for the retry.
Merges write their segment file before `publish_merge` writes the manifest, and
remove replaced files after.

`Store::open` locks, reads or creates the manifest (a new database writes only
the manifest, so a crash during creation leaves at most `MANIFEST.tmp`; a missing
log is created like an empty one), rejects a different schema,
loads and checks every named segment, replays the WAL up to the first torn or
corrupt frame (truncating it there), and removes unnamed files: only
`MANIFEST.tmp` and `seg-`/`wal-` names with 16 hex digits, never other files.
`Db::open` then re-applies the recovered changes and commits. `Drop` commits
durable databases.

`store::open_read_only` shares the lock (a missing `LOCK`, as in a backup, means no
lock), requires an existing manifest, and runs the same loading without truncating
the log or removing files. The `Db` has no `Store` and no `dir`: replayed changes
commit into memory only, no commit thread starts, and every write, `flush`,
`compact`, and `backup` returns `ReadOnly` (`Shared::writable`).
`backup` commits, then copies the manifest, the named segments, and the current
WAL under the writer lock.
All decoding goes through `codec::Decoder`, which returns `Corrupt`, never panics.

## Errors

`RingDbError` currently includes:

- `DimensionMismatch`;
- `Io`;
- `Corrupt`;
- `InvalidQuery` (searches and filters);
- `InvalidSchema`;
- `InvalidRecord`;
- `UnknownField`;
- `KeyNotFound`;
- `ReadOnly`;
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
RINGDB_BENCH_N=100000 cargo bench --bench query_backends cpu_knn
RINGDB_BENCH_N=100000 cargo bench --bench query_backends cpu_region
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
| `smallvec` | inline single-disk regions |
| `crc32fast` | checksums of WAL frames, segments, and the manifest |
| `tempfile` | isolated test directories |

The release publishes one crate. Keep release scripts and CI aligned with the package
version. Query semantics do not depend on the backend.
