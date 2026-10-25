# Update ring-db

## From 0.10 to 0.11

The API of 0.10 does not change.

Version 0.11 writes segment files in a new format, with smaller keyword fields.
It reads the files of version 0.10.
Each new segment file has the new format, and `db.compact()` writes all records again in the new format.
ring-db 0.10 cannot read the new files.
To go back to 0.10 later, keep a copy of the database before you open it for writing with 0.11.
`Db::open_read_only` writes no file, so 0.10 can still open the database after it.

## From 0.9 to 0.10

Version 0.10 has a new API and a new file format.
ring-db 0.10 cannot read the files of version 0.9.
To keep your data, add your records again to a new database.

### Main changes

| 0.9 | 0.10 |
|---|---|
| `RingDb::new(RingDbConfig::new(dims))`, then `build()` | `Db::in_memory(Schema::new(dims), Options::default())` |
| `with_persist_dir(dir)`, then `RingDb::load(dir, backend)` | `Db::open(dir, schema, options)` |
| `with_backend_preference(preference)` | `Options::backend(preference)` |
| `add_vector`, `add_vector_with_metadata` | `db.add(record)` or `db.add_all(records)`, then `db.flush()` |
| A built database cannot change | `db.delete(key)` and `db.apply(batch)` at all times |
| A `bytemuck::Pod` payload, read with `fetch_payload(id)` | Typed fields in the `Schema`, read with `db.get(key)?.get(field)` |
| `MetadataTag::new(key, value)` | A `Keyword` or integer field |
| `db.metadata_eq(key, value)` | `Filter::eq(field, value)` |
| `CandidateSet`: `intersection`, `union`, `complement` | `Filter`: `a & b`, `a \| b`, `!a` |
| `db.candidates_from_ids(ids)` | `Filter::keys(keys)` |
| `.with_candidates(&set)` | `.filter(&filter)` |
| `.with_weights(&weights)` | `.metric(Metric::WeightedL2(&weights))` |
| `KnnQuery::new(&point, k)` with `query_knn` | `db.search(Knn::new(&point, k))` |
| `DiskQuery` with `query_disk` | `db.search(Disk::new(&center, radius))` |
| `DiskIntersectionQuery` with `query_disk_intersection` | `db.search(disk_a.and(disk_b))` |
| `RingQuery`, `RangeQuery` | Removed. Search the outer `Disk`, and remove the hits that are too close. |
| `QueryResult` with `hits`, `backend_used`, and `elapsed` | `Vec<Hit>` for KNN, `RegionResult` for a region |
| `hit.id`, a `u32` index | `hit.key`, a `Key` (`u64`) |

### Behavior changes

- **Keys.** `add` gives each record a new key. Keys increase in the order of the adds, and ring-db does not use a key two times. Deletes do not change the other keys.
- **Distances of an intersection.** In 0.9, a hit had only the distance to the first disk. In 0.10, `hit.dists_sq()` has one distance for each disk, in the order of the disks.
- **Commits.** A search does not find a new record before the next commit. A commit occurs each second, or when you call `db.flush()`.
- **Strings.** Filters on `Keyword` and `Keywords` fields compare strings. You do not need to give an integer to each string.
- **Ranges.** `Filter::range` selects numbers in a range, on integer and `F32` fields.
- **Durability.** `Db::open` writes each change to a log before the call returns. Your changes stay after a crash.
