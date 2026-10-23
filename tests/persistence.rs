//! Durable databases: reopening, crash recovery, corruption, locking, backups.
mod common;

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use common::knn_keys;
use ringdb::{
    Batch, Db, Disk, FieldType, Filter, Key, Knn, Options, Record, RingDbError, Schema, Value,
};

fn schema() -> Schema {
    Schema::new(2)
        .field("rooms", FieldType::U32)
        .field("price", FieldType::U64)
        .field("floor", FieldType::I64)
        .field("surface", FieldType::F32)
        .field("garden", FieldType::Bool)
        .field("city", FieldType::Keyword)
        .field("tags", FieldType::Keywords)
        .field("title", FieldType::Text)
        .field("photos", FieldType::TextList)
        .field("blob", FieldType::Bytes)
}

fn manual() -> Options {
    Options::default().commit_interval(None)
}

fn open(dir: &Path) -> Db {
    open_with(dir, schema(), manual()).unwrap()
}

/// Open, waiting while another test's child process briefly holds the lock: a
/// process forked by any thread keeps a copy of every open file until it execs.
fn open_with(dir: &Path, schema: Schema, options: Options) -> Result<Db, RingDbError> {
    let started = std::time::Instant::now();
    loop {
        match Db::open(dir, schema.clone(), options.clone()) {
            Err(RingDbError::Io(error))
                if error.kind() == std::io::ErrorKind::WouldBlock
                    && started.elapsed() < std::time::Duration::from_secs(5) =>
            {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            result => return result,
        }
    }
}

/// Record `i`, with every field type and some missing values.
fn record(i: u32) -> Record {
    let mut record = Record::new([i as f32, (i % 7) as f32])
        .set("rooms", i % 5)
        .set("price", u64::from(i) * 1000)
        .set("floor", i as i64 - 3)
        .set("city", ["Lyon", "Paris", "Nantes"][i as usize % 3])
        .set("tags", vec![format!("t{}", i % 4), format!("t{}", i % 6)]);
    if i.is_multiple_of(2) {
        record = record
            .set("surface", i as f32 * 1.5)
            .set("garden", i.is_multiple_of(4))
            .set("title", format!("Listing {i} é"))
            .set("photos", vec![format!("{i}.jpg"), "plan.png".to_string()])
            .set("blob", vec![i as u8, 0, 255]);
    }
    record
}

/// A key, its vector, and its fields.
type Contents = (u64, Vec<f32>, Vec<(String, Value)>);

/// Everything visible, as comparable documents in key order.
fn contents(db: &Db) -> Vec<Contents> {
    let mut keys = knn_keys(&db.search(Knn::new(&[0.0, 0.0], 1 << 20)).unwrap());
    keys.sort_unstable();
    keys.into_iter()
        .map(|key| {
            let doc = db.get(Key::new(key)).unwrap();
            let fields = doc
                .fields()
                .map(|(n, v)| (n.to_string(), v.clone()))
                .collect();
            (key, doc.vector().to_vec(), fields)
        })
        .collect()
}

fn files(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn file_with_prefix(dir: &Path, prefix: &str) -> PathBuf {
    let name = files(dir)
        .into_iter()
        .find(|name| name.starts_with(prefix))
        .unwrap_or_else(|| panic!("no {prefix} file"));
    dir.join(name)
}

#[test]
fn committed_records_survive_reopening() {
    let dir = tempfile::tempdir().unwrap();
    let expected = {
        let db = open(dir.path());
        let keys = db.add_all((0..40).map(record)).unwrap();
        db.flush().unwrap();
        for key in &keys[..10] {
            db.delete(*key).unwrap();
        }
        db.flush().unwrap();
        db.add(record(40)).unwrap();
        db.flush().unwrap();
        contents(&db)
    };
    assert_eq!(expected.len(), 31);
    let db = open(dir.path());
    assert_eq!(contents(&db), expected);
    // Keys continue after the largest key ever assigned.
    assert_eq!(db.add(record(1)).unwrap(), Key::new(41));
    let lyon = Filter::eq("city", "Lyon") & Filter::range("rooms", 2..);
    let hits = db.search(Knn::new(&[0.0, 0.0], 100).filter(&lyon)).unwrap();
    assert!(!hits.is_empty());
    for hit in &hits {
        let doc = db.get(hit.key).unwrap();
        assert_eq!(doc.get("city").and_then(Value::as_str), Some("Lyon"));
        assert!(doc.get("rooms").and_then(Value::as_u32).unwrap() >= 2);
    }
    assert!(!db.search(Disk::new(&[10.0, 3.0], 5.0)).unwrap().is_empty());
}

#[test]
fn merges_and_compaction_replace_files() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with(
        dir.path(),
        schema(),
        Options::default().commit_interval(None),
    )
    .unwrap();
    for batch in 0..20 {
        db.add_all((batch * 5..batch * 5 + 5).map(record)).unwrap();
        db.delete(Key::new(u64::from(batch) * 5)).unwrap();
        db.flush().unwrap();
    }
    let segments = db.snapshot().segments();
    let names = files(dir.path());
    assert_eq!(
        names.iter().filter(|n| n.starts_with("seg-")).count(),
        segments
    );
    assert_eq!(names.iter().filter(|n| n.starts_with("wal-")).count(), 1);
    db.compact().unwrap();
    let expected = contents(&db);
    assert_eq!(expected.len(), 80);
    drop(db);
    let names = files(dir.path());
    assert_eq!(names.iter().filter(|n| n.starts_with("seg-")).count(), 1);
    assert_eq!(
        names
            .iter()
            .filter(|n| !n.starts_with("seg-") && !n.starts_with("wal-"))
            .collect::<Vec<_>>(),
        ["LOCK", "MANIFEST"]
    );
    let db = open(dir.path());
    assert_eq!(db.snapshot().segments(), 1);
    assert_eq!(contents(&db), expected);
}

const CHILD: &str = "RINGDB_CRASH_CHILD";

/// Run `test` in a child process that writes into `dir` and aborts.
fn crash(test: &str, dir: &Path) {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(CHILD, dir)
        .output()
        .unwrap();
    assert!(!output.status.success(), "the child must abort");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
fn logged_changes_survive_a_crash_and_a_torn_tail() {
    if let Some(dir) = std::env::var_os(CHILD) {
        let db = open(Path::new(&dir));
        let keys = db.add_all((0..10).map(record)).unwrap();
        db.flush().unwrap();
        // Logged but never committed.
        db.apply(Batch::new().delete(keys[3]).add(record(10)))
            .unwrap();
        let doomed = db.add(record(11)).unwrap();
        db.delete(doomed).unwrap();
        db.add(record(12)).unwrap();
        // The parent tears this batch, so that none of it is recovered.
        db.apply(Batch::new().delete(keys[0]).add(record(13)))
            .unwrap();
        std::process::abort();
    }
    let dir = tempfile::tempdir().unwrap();
    crash("logged_changes_survive_a_crash_and_a_torn_tail", dir.path());
    // A crash in the middle of a write leaves a partial frame.
    let wal = file_with_prefix(dir.path(), "wal-");
    let mut file = OpenOptions::new().append(true).open(&wal).unwrap();
    file.set_len(file.metadata().unwrap().len() - 1).unwrap();
    file.write_all(&[40, 0, 0, 0, 1, 2, 3, 4, 1, 9]).unwrap();
    drop(file);

    let db = open(dir.path());
    let keys: Vec<u64> = contents(&db).iter().map(|c| c.0).collect();
    assert_eq!(keys, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12]);
    assert_eq!(db.get(Key::new(12)).unwrap().vector(), [12.0, 5.0]);
    assert_eq!(db.add(record(0)).unwrap(), Key::new(13));
    drop(db);
    let db = open(dir.path());
    assert_eq!(contents(&db).len(), 12);
}

#[test]
fn corruption_is_detected() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        db.add_all((0..20).map(record)).unwrap();
        db.flush().unwrap();
    }
    let segment = file_with_prefix(dir.path(), "seg-");
    let mut bytes = fs::read(&segment).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0x40;
    fs::write(&segment, &bytes).unwrap();
    assert!(matches!(
        open_with(dir.path(), schema(), manual()),
        Err(RingDbError::Corrupt(_))
    ));
    bytes[middle] ^= 0x40;
    fs::write(&segment, &bytes).unwrap();
    fs::remove_file(&segment).unwrap();
    assert!(matches!(
        open_with(dir.path(), schema(), manual()),
        Err(RingDbError::Io(_))
    ));

    let manifest = dir.path().join("MANIFEST");
    let mut bytes = fs::read(&manifest).unwrap();
    bytes[20] ^= 1;
    fs::write(&manifest, &bytes).unwrap();
    assert!(matches!(
        open_with(dir.path(), schema(), manual()),
        Err(RingDbError::Corrupt(_))
    ));
    fs::remove_file(&manifest).unwrap();
    // Without a manifest, other files mean a damaged database, not a new one.
    assert!(matches!(
        open_with(dir.path(), schema(), manual()),
        Err(RingDbError::Corrupt(_))
    ));
}

#[test]
fn the_schema_and_the_lock_are_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    match Db::open(dir.path(), schema(), manual()) {
        Err(RingDbError::Io(error)) => assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock),
        Err(other) => panic!("expected a lock error, got {other}"),
        Ok(_) => panic!("a second open must fail"),
    }
    drop(db);
    for other in [
        Schema::new(3),
        schema().field("extra", FieldType::U32),
        Schema::new(2).field("rooms", FieldType::U64),
    ] {
        let result = open_with(dir.path(), other, manual());
        assert!(
            matches!(result, Err(RingDbError::InvalidSchema(_))),
            "{:?}",
            result.err()
        );
    }
    let db = open(dir.path());
    assert!(db.is_empty());
}

#[test]
fn leftover_files_are_removed_on_open() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        db.add_all((0..5).map(record)).unwrap();
        db.flush().unwrap();
    }
    let segment = file_with_prefix(dir.path(), "seg-");
    fs::copy(&segment, dir.path().join("seg-00000000000000ff.seg")).unwrap();
    fs::write(dir.path().join("wal-00000000000000ff.log"), b"stale").unwrap();
    fs::write(dir.path().join("MANIFEST.tmp"), b"partial").unwrap();
    fs::write(dir.path().join("notes.txt"), b"the user's file").unwrap();
    let db = open(dir.path());
    assert_eq!(db.len(), 5);
    let names = files(dir.path());
    assert!(names.contains(&"notes.txt".to_string()));
    assert!(
        !names
            .iter()
            .any(|n| n.contains("ff") || n.ends_with(".tmp"))
    );
}

#[test]
fn backups_open_as_independent_databases() {
    let dir = tempfile::tempdir().unwrap();
    let copy = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let keys = db.add_all((0..30).map(record)).unwrap();
    db.flush().unwrap();
    db.delete(keys[0]).unwrap();
    db.add(record(30)).unwrap();
    // Uncommitted changes are committed first.
    db.backup(copy.path()).unwrap();
    assert!(db.backup(copy.path()).is_err(), "the target must be empty");
    let expected = contents(&db);
    db.delete(keys[1]).unwrap();
    db.flush().unwrap();

    let restored = open(copy.path());
    assert_eq!(contents(&restored), expected);
    assert_eq!(restored.add(record(0)).unwrap(), Key::new(31));
    assert_eq!(db.len(), expected.len() - 1);

    let memory = Db::in_memory(schema(), manual()).unwrap();
    assert!(memory.backup(tempfile::tempdir().unwrap().path()).is_err());
}

#[test]
fn background_commits_are_durable() {
    let dir = tempfile::tempdir().unwrap();
    let options = Options::default().commit_interval(Some(std::time::Duration::from_millis(5)));
    let db = open_with(dir.path(), schema(), options.clone()).unwrap();
    for i in 0..50 {
        db.add(record(i)).unwrap();
        if i % 3 == 0 {
            db.delete(Key::new(u64::from(i))).unwrap();
        }
    }
    db.flush().unwrap();
    let expected = contents(&db);
    drop(db);
    let db = open_with(dir.path(), schema(), options).unwrap();
    assert_eq!(contents(&db), expected);
}
