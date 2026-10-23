//! Deletes, commits, merges, and concurrent changes compared with a model.
mod common;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use common::{group, knn_keys, manual, region_keys};
use rand::{RngExt, SeedableRng};
use ringdb::{BackendPreference, Batch, Db, Disk, Key, Knn, Options, Record, RingDbError, Value};

fn db(options: Options) -> Db {
    Db::in_memory(common::schema(1), options).unwrap()
}

fn add(db: &Db, x: f32) -> Key {
    let group = (x.abs() as u32) % 3;
    db.add(
        Record::new([x])
            .set("group", group)
            .set("id", x.to_bits() as u64),
    )
    .unwrap()
}

/// Every visible key, from both search primitives.
fn visible(db: &Db) -> Vec<u64> {
    let snapshot = db.snapshot();
    let mut knn = knn_keys(&snapshot.search(Knn::new(&[0.0], 1 << 20)).unwrap());
    knn.sort_unstable();
    let region = region_keys(&snapshot.search(Disk::new(&[0.0], 1e9)).unwrap());
    assert_eq!(knn, region);
    assert_eq!(knn.len(), snapshot.len());
    knn
}

fn not_found<T: std::fmt::Debug>(result: Result<T, RingDbError>) {
    assert!(
        matches!(result, Err(RingDbError::KeyNotFound(_))),
        "{result:?}"
    );
}

#[test]
fn deletes_hide_records_at_the_next_commit() {
    let db = db(manual(BackendPreference::Cpu));
    let keys: Vec<Key> = (0..6).map(|i| add(&db, i as f32)).collect();
    // A record deleted before its first commit never becomes visible.
    db.delete(keys[5]).unwrap();
    not_found(db.delete(keys[5]));
    db.flush().unwrap();
    assert_eq!(visible(&db), [0, 1, 2, 3, 4]);

    db.delete(keys[1]).unwrap();
    db.delete(keys[3]).unwrap();
    not_found(db.delete(keys[1]));
    not_found(db.delete(Key::new(99)));
    assert_eq!(visible(&db), [0, 1, 2, 3, 4], "not committed yet");
    let before = db.snapshot();
    db.flush().unwrap();
    assert_eq!(visible(&db), [0, 2, 4]);
    assert_eq!(before.len(), 5, "snapshots do not change");
    assert!(before.get(keys[1]).is_ok());
    not_found(db.get(keys[1]));
    not_found(db.delete(keys[3]));

    // KNN still returns k live records, and filters skip deleted rows.
    let hits = db.search(Knn::new(&[1.0], 2)).unwrap();
    assert_eq!(knn_keys(&hits), [0, 2]);
    // Groups are x % 3: deleted keys 1 and 3 were in groups 1 and 0.
    let ones = group(1);
    assert_eq!(
        knn_keys(&db.search(Knn::new(&[0.0], 9).filter(&ones)).unwrap()),
        [4]
    );
    let twos = group(2);
    let inside = db.search(Disk::new(&[0.0], 10.0).filter(&twos)).unwrap();
    assert_eq!(region_keys(&inside), [2]);

    // Keys are never reused.
    assert_eq!(add(&db, 7.0), Key::new(6));
}

#[test]
fn an_update_between_explicit_commits_is_atomic() {
    let db = db(manual(BackendPreference::Cpu));
    let mut key = add(&db, 1.0);
    db.flush().unwrap();
    for version in 2..6 {
        db.delete(key).unwrap();
        let next = add(&db, version as f32);
        assert_eq!(visible(&db), [key.get()]);
        db.flush().unwrap();
        assert_eq!(visible(&db), [next.get()]);
        let doc = db.get(next).unwrap();
        assert_eq!(doc.vector(), [version as f32]);
        key = next;
    }
}

#[test]
fn a_failed_batch_makes_no_change() {
    let db = db(manual(BackendPreference::Cpu));
    let keys: Vec<Key> = (0..3).map(|i| add(&db, i as f32)).collect();
    db.flush().unwrap();
    let record = |x: f32| Record::new([x]).set("group", 0u32);
    not_found(
        db.apply(
            Batch::new()
                .delete(keys[0])
                .delete(Key::new(9))
                .add(record(5.0)),
        ),
    );
    not_found(db.apply(Batch::new().delete(keys[1]).delete(keys[1])));
    let invalid = Batch::new()
        .delete(keys[2])
        .add(record(6.0).set("nope", 1u32));
    assert!(matches!(
        db.apply(invalid),
        Err(RingDbError::UnknownField(_))
    ));
    assert!(db.apply(Batch::new()).unwrap().is_empty());
    db.flush().unwrap();
    assert_eq!(visible(&db), [0, 1, 2]);
    // No key was used by the failed batches.
    let added = db
        .apply(
            Batch::new()
                .delete(keys[0])
                .add(record(7.0))
                .add(record(8.0)),
        )
        .unwrap();
    assert_eq!(added, [Key::new(3), Key::new(4)]);
    db.flush().unwrap();
    assert_eq!(visible(&db), [1, 2, 3, 4]);
}

#[test]
fn batches_replace_records_atomically_during_background_commits() {
    let db = db_with_interval(Duration::from_millis(1));
    let first = add(&db, 0.0);
    db.flush().unwrap();
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut key = first;
            for version in 1..=500 {
                let record = Record::new([version as f32]).set("group", 0u32);
                key = db.apply(Batch::new().delete(key).add(record)).unwrap()[0];
            }
            done.store(true, Ordering::Relaxed);
        });
        while !done.load(Ordering::Relaxed) {
            // Exactly one version is visible in every snapshot.
            assert_eq!(visible(&db).len(), 1);
        }
    });
    db.flush().unwrap();
    let keys = visible(&db);
    assert_eq!(db.get(Key::new(keys[0])).unwrap().vector(), [500.0]);
}

fn db_with_interval(interval: Duration) -> Db {
    db(Options::default().commit_interval(Some(interval)))
}

#[test]
fn background_commits_make_changes_visible() {
    let db = db_with_interval(Duration::from_millis(10));
    let key = add(&db, 3.0);
    let started = Instant::now();
    while db.get(key).is_err() {
        assert!(started.elapsed() < Duration::from_secs(5), "no commit");
        std::thread::sleep(Duration::from_millis(5));
    }
    db.delete(key).unwrap();
    while db.get(key).is_ok() {
        assert!(started.elapsed() < Duration::from_secs(5), "no commit");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(db.is_empty());
}

#[test]
fn merges_keep_few_segments_and_compact_makes_one() {
    let db = db(Options::default().commit_interval(None));
    let mut expected = Vec::new();
    for batch in 0..200u32 {
        for i in 0..1 + batch % 5 {
            expected.push(add(&db, (batch * 10 + i) as f32).get());
        }
        if batch % 7 == 3 {
            let key = expected.remove(expected.len() / 2);
            db.delete(Key::new(key)).unwrap();
        }
        db.flush().unwrap();
        // Live sizes at least double from newest to oldest.
        assert!(
            db.snapshot().segments() <= 12,
            "{} segments",
            db.snapshot().segments()
        );
    }
    assert_eq!(visible(&db), expected);

    for key in expected.drain(..expected.len() / 2) {
        db.delete(Key::new(key)).unwrap();
    }
    db.compact().unwrap();
    assert_eq!(db.snapshot().segments(), 1);
    assert_eq!(visible(&db), expected);
    for &key in &expected {
        let doc = db.get(Key::new(key)).unwrap();
        let x = doc.vector()[0];
        assert_eq!(
            doc.get("id").and_then(Value::as_u64),
            Some(x.to_bits() as u64)
        );
    }
    for key in expected.drain(..) {
        db.delete(Key::new(key)).unwrap();
    }
    db.compact().unwrap();
    assert_eq!(db.snapshot().segments(), 0);
    assert!(visible(&db).is_empty());
}

#[test]
fn random_changes_match_a_model() {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0x5EED_0003);
    for auto_merge in [false, true] {
        let db = db(Options::default()
            .commit_interval(None)
            .auto_merge(auto_merge));
        let mut committed: BTreeMap<u64, f32> = BTreeMap::new();
        let mut pending = committed.clone();
        let mut next = 0;
        for step in 0..3000 {
            match rng.random_range(0..10) {
                0..5 => {
                    let x = rng.random_range(-50.0f32..50.0);
                    let key = add(&db, x);
                    assert_eq!(key.get(), next);
                    next += 1;
                    pending.insert(key.get(), x);
                }
                5..8 if !pending.is_empty() => {
                    let index = rng.random_range(0..pending.len());
                    let key = *pending.keys().nth(index).unwrap();
                    db.delete(Key::new(key)).unwrap();
                    pending.remove(&key);
                }
                8 => not_found(db.delete(Key::new(rng.random_range(next..next + 5)))),
                _ => {
                    if step % 3 == 0 {
                        db.compact().unwrap();
                    } else {
                        db.flush().unwrap();
                    }
                    committed = pending.clone();
                }
            }
            if step % 97 == 0 {
                assert_eq!(visible(&db), committed.keys().copied().collect::<Vec<_>>());
                let query = rng.random_range(-50.0f32..50.0);
                let hits = db.search(Knn::new(&[query], 5)).unwrap();
                let mut expected: Vec<(f32, u64)> = committed
                    .iter()
                    .map(|(&key, &x)| ((x - query) * (x - query), key))
                    .collect();
                expected.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let expected: Vec<u64> = expected.iter().take(5).map(|e| e.1).collect();
                assert_eq!(knn_keys(&hits), expected);
            }
        }
    }
}

#[test]
fn concurrent_writers_readers_and_merges_agree() {
    let db = db_with_interval(Duration::from_millis(2));
    let done = AtomicBool::new(false);
    let survivors: Vec<Vec<u64>> = std::thread::scope(|scope| {
        let writers: Vec<_> = (0..4)
            .map(|thread| {
                let db = &db;
                scope.spawn(move || {
                    let mut rng = rand::rngs::SmallRng::seed_from_u64(thread);
                    let mut mine = Vec::new();
                    for i in 0..400 {
                        mine.push(add(db, (thread * 1000 + i) as f32).get());
                        if rng.random_range(0..3) == 0 {
                            let index = rng.random_range(0..mine.len());
                            db.delete(Key::new(mine.swap_remove(index))).unwrap();
                        }
                    }
                    mine
                })
            })
            .collect();
        scope.spawn(|| {
            while !done.load(Ordering::Relaxed) {
                db.flush().unwrap();
            }
        });
        scope.spawn(|| {
            while !done.load(Ordering::Relaxed) {
                let keys = visible(&db);
                assert!(keys.windows(2).all(|w| w[0] < w[1]), "duplicate keys");
            }
        });
        let survivors = writers.into_iter().map(|w| w.join().unwrap()).collect();
        done.store(true, Ordering::Relaxed);
        survivors
    });
    db.flush().unwrap();
    let mut expected: Vec<u64> = survivors.into_iter().flatten().collect();
    expected.sort_unstable();
    assert_eq!(visible(&db), expected);
    for key in expected {
        let doc = db.get(Key::new(key)).unwrap();
        assert_eq!(
            doc.get("id").and_then(Value::as_u64),
            Some(doc.vector()[0].to_bits() as u64)
        );
    }
}
