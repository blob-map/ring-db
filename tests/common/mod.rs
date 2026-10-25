//! Shared builders, scalar references, and result comparisons.
#![allow(dead_code)]

use ringdb::{
    BackendPreference, Db, FieldType, Filter, Hit, Key, Options, Record, RegionResult, Schema,
};

/// Row `i` has key `i`, `group = i % 3`, and `id = i`.
pub fn schema(dims: usize) -> Schema {
    Schema::new(dims)
        .field("group", FieldType::U32)
        .field("id", FieldType::U64)
}

pub fn build(vectors: &[Vec<f32>], backend: BackendPreference) -> Db {
    build_dims(vectors[0].len(), vectors, backend)
}

/// Options with explicit commits and merges, so tests control segments, and
/// every segment on the selected backend.
pub fn manual(backend: BackendPreference) -> Options {
    Options::default()
        .backend(backend)
        .commit_interval(None)
        .auto_merge(false)
        .vulkan_min_rows(0)
}

pub fn build_dims(dims: usize, vectors: &[Vec<f32>], backend: BackendPreference) -> Db {
    let db = Db::in_memory(schema(dims), manual(backend)).unwrap();
    for (id, vector) in vectors.iter().enumerate() {
        let record = Record::new(vector.as_slice())
            .set("group", (id % 3) as u32)
            .set("id", id as u64);
        assert_eq!(db.add(record).unwrap().get(), id as u64);
    }
    db.flush().unwrap();
    db
}

pub fn cpu(vectors: &[Vec<f32>]) -> Db {
    build(vectors, BackendPreference::Cpu)
}

pub fn group(value: u32) -> Filter {
    Filter::eq("group", value)
}

pub fn keys(ids: impl IntoIterator<Item = u64>) -> Filter {
    Filter::keys(ids.into_iter().map(Key::new))
}

pub fn knn_keys(hits: &[Hit]) -> Vec<u64> {
    hits.iter().map(|hit| hit.key.get()).collect()
}

/// Region keys in ascending order.
pub fn region_keys(result: &RegionResult) -> Vec<u64> {
    let mut keys: Vec<u64> = result.keys().map(|key| key.get()).collect();
    keys.sort_unstable();
    keys
}

/// `(key, distances)` pairs sorted by key.
pub fn region_rows(result: &RegionResult) -> Vec<(u64, Vec<f32>)> {
    let mut rows: Vec<_> = result
        .iter()
        .map(|hit| (hit.key.get(), hit.dists_sq().to_vec()))
        .collect();
    rows.sort_by_key(|row| row.0);
    rows
}

pub fn close(actual: f32, expected: f64) -> bool {
    let tolerance = 2e-5 * expected.abs().max(1.0);
    (f64::from(actual) - expected).abs() <= tolerance
}

pub fn assert_close(actual: f32, expected: f64) {
    assert!(
        close(actual, expected),
        "actual={actual}, expected={expected}"
    );
}

pub fn same_knn(expected: &[Hit], actual: &[Hit]) {
    assert_eq!(knn_keys(expected), knn_keys(actual));
    for (a, b) in expected.iter().zip(actual) {
        assert!(
            a.dist_sq == b.dist_sq || close(b.dist_sq, f64::from(a.dist_sq)),
            "key {:?}: {} vs {}",
            a.key,
            a.dist_sq,
            b.dist_sq
        );
    }
}

pub fn same_region(expected: &RegionResult, actual: &RegionResult) {
    assert_eq!(expected.n_disks(), actual.n_disks());
    let (expected, actual) = (region_rows(expected), region_rows(actual));
    assert_eq!(
        expected.iter().map(|r| r.0).collect::<Vec<_>>(),
        actual.iter().map(|r| r.0).collect::<Vec<_>>()
    );
    for ((key, a), (_, b)) in expected.iter().zip(&actual) {
        for (a, b) in a.iter().zip(b) {
            assert!(a == b || close(*b, f64::from(*a)), "key {key}: {a} vs {b}");
        }
    }
}

/// Scalar f64 reference distance.
pub fn reference_dist(vector: &[f32], query: &[f32], weights: Option<&[f32]>) -> f64 {
    vector
        .iter()
        .zip(query)
        .enumerate()
        .map(|(i, (&x, &q))| {
            let weight = weights.map_or(1.0, |w| f64::from(w[i]));
            if weight == 0.0 {
                return 0.0;
            }
            let delta = f64::from(x) - f64::from(q);
            weight * delta * delta
        })
        .sum()
}

/// 900 records in 3 segments, 40 of them deleted in the first two, with a
/// `city` keyword (or none), up to three `tags`, and `rooms`. Record `i` is at
/// `[i / 30, i % 30]`.
pub fn keyword_db(options: Options) -> Db {
    let schema = Schema::new(2)
        .field("city", FieldType::Keyword)
        .field("tags", FieldType::Keywords)
        .field("rooms", FieldType::U32);
    let db = Db::in_memory(schema, options).unwrap();
    let mut x = 5u64;
    let mut next = move |n: u64| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) % n
    };
    for part in 0..3 {
        let records: Vec<Record> = (part * 300..part * 300 + 300)
            .map(|i| {
                let mut record = Record::new([(i / 30) as f32, (i % 30) as f32])
                    .set("rooms", next(6) as u32)
                    .set(
                        "tags",
                        (0..next(4))
                            .map(|_| format!("tag{}", next(6)))
                            .collect::<Vec<_>>(),
                    );
                // Lille is in about half of the records, so it is a bitmap.
                match next(10) {
                    0 => {}
                    1..5 => record = record.set("city", "Lille"),
                    n => record = record.set("city", format!("city{}", n * 7 + next(7))),
                }
                record
            })
            .collect();
        db.add_all(records).unwrap();
        db.flush().unwrap();
    }
    for key in (0..600).step_by(15) {
        db.delete(Key::new(key)).unwrap();
    }
    db.flush().unwrap();
    db
}
