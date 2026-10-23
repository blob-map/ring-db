//! Schemas, records, typed filters, and reads across segments.
mod common;

use common::{knn_keys, region_keys};
use ringdb::{Db, Disk, FieldType, Filter, Key, Knn, Options, Record, RingDbError, Schema, Value};

fn listings() -> Schema {
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
        .field("thumbnail", FieldType::Bytes)
}

/// Five listings on a line; listing `i` is at `[i, 0]` and has key `i`.
fn db() -> Db {
    let db = Db::in_memory(listings(), Options::default()).unwrap();
    let records = [
        Record::new([0.0, 0.0])
            .set("rooms", 2)
            .set("price", 150_000)
            .set("floor", -1)
            .set("surface", 45.5f32)
            .set("garden", false)
            .set("city", "Lyon")
            .set("tags", ["cellar"]),
        Record::new([1.0, 0.0])
            .set("rooms", 3)
            .set("price", 320_000)
            .set("floor", 2)
            .set("surface", 70.0f32)
            .set("garden", true)
            .set("city", "Paris")
            .set("tags", ["balcony", "elevator"]),
        Record::new([2.0, 0.0])
            .set("rooms", 4)
            .set("price", 410_000)
            .set("surface", 92.0f32)
            .set("city", "Lyon")
            .set("tags", ["garden", "elevator", "garden"]),
        Record::new([3.0, 0.0]).set("city", "Nantes"),
        Record::new([4.0, 0.0])
            .set("rooms", 5)
            .set("price", 900_000)
            .set("floor", 0)
            .set("garden", true)
            .set("title", "Maison de ville")
            .set("photos", ["a.jpg", "b.jpg"])
            .set("thumbnail", vec![0u8, 159, 146, 150]),
    ];
    for record in records {
        db.add(record).unwrap();
    }
    db.flush().unwrap();
    db
}

fn matching(db: &Db, filter: Filter) -> Vec<u64> {
    let origin = [0.0, 0.0];
    let hits = db.search(Knn::new(&origin, 100).filter(&filter)).unwrap();
    let region = db
        .search(Disk::new(&origin, 100.0).filter(&filter))
        .unwrap();
    let mut keys = knn_keys(&hits);
    keys.sort_unstable();
    assert_eq!(keys, region_keys(&region), "KNN and regions disagree");
    keys
}

#[test]
fn documents_return_every_stored_value() {
    let db = db();
    let doc = db.get(Key::new(4)).unwrap();
    assert_eq!(doc.key(), Key::new(4));
    assert_eq!(doc.vector(), [4.0, 0.0]);
    assert_eq!(doc.get("rooms"), Some(&Value::U32(5)));
    assert_eq!(doc.get("price").and_then(Value::as_u64), Some(900_000));
    assert_eq!(doc.get("floor").and_then(Value::as_i64), Some(0));
    assert_eq!(doc.get("surface"), None);
    assert_eq!(doc.get("garden").and_then(Value::as_bool), Some(true));
    assert_eq!(
        doc.get("title").and_then(Value::as_str),
        Some("Maison de ville")
    );
    assert_eq!(
        doc.get("photos").and_then(Value::as_strs),
        Some(["a.jpg".to_string(), "b.jpg".to_string()].as_slice())
    );
    assert_eq!(
        doc.get("thumbnail").and_then(Value::as_bytes),
        Some([0u8, 159, 146, 150].as_slice())
    );
    assert_eq!(doc.get("unknown"), None);

    // Keywords read back sorted and without duplicates.
    let doc = db.get(Key::new(2)).unwrap();
    assert_eq!(doc.get("city").and_then(Value::as_str), Some("Lyon"));
    assert_eq!(
        doc.get("tags"),
        Some(&Value::TextList(vec!["elevator".into(), "garden".into()]))
    );
    let doc = db.get(Key::new(3)).unwrap();
    let fields: Vec<(&str, &Value)> = doc.fields().collect();
    assert_eq!(fields, [("city", &Value::Text("Nantes".into()))]);
    assert!(matches!(
        db.get(Key::new(5)),
        Err(RingDbError::KeyNotFound(5))
    ));
}

#[test]
fn numeric_filters_compare_present_values_only() {
    let db = db();
    assert_eq!(matching(&db, Filter::eq("rooms", 3)), [1]);
    assert_eq!(matching(&db, Filter::range("rooms", 3..)), [1, 2, 4]);
    assert_eq!(matching(&db, Filter::range("rooms", ..3)), [0]);
    assert_eq!(
        matching(&db, Filter::range("price", 150_000..=410_000)),
        [0, 1, 2]
    );
    assert_eq!(
        matching(&db, Filter::range("price", 150_000..410_000)),
        [0, 1]
    );
    assert_eq!(matching(&db, Filter::range("floor", ..=0)), [0, 4]);
    assert_eq!(matching(&db, Filter::eq("floor", -1)), [0]);
    assert_eq!(matching(&db, Filter::range("surface", 50.0f32..)), [1, 2]);
    assert_eq!(matching(&db, Filter::range("surface", 45..=70)), [0, 1]);
    assert_eq!(matching(&db, Filter::eq("garden", true)), [1, 4]);
    assert_eq!(matching(&db, Filter::eq("garden", false)), [0]);
    assert_eq!(matching(&db, Filter::any_of("rooms", [2, 5, 7])), [0, 4]);
    // Bounds outside the field type select everything or nothing.
    assert_eq!(matching(&db, Filter::range("rooms", -10..)), [0, 1, 2, 4]);
    assert_eq!(
        matching(&db, Filter::range("rooms", 5_000_000_000u64..)),
        []
    );
    let (high, low) = (10, 1);
    assert_eq!(matching(&db, Filter::range("price", high..=low)), []);
}

#[test]
fn keyword_filters_use_exact_terms() {
    let db = db();
    assert_eq!(matching(&db, Filter::eq("city", "Lyon")), [0, 2]);
    assert_eq!(matching(&db, Filter::eq("city", "lyon")), []);
    assert_eq!(
        matching(&db, Filter::any_of("city", ["Paris", "Nantes", "Nice"])),
        [1, 3]
    );
    assert_eq!(matching(&db, Filter::eq("tags", "elevator")), [1, 2]);
    assert_eq!(
        matching(&db, Filter::any_of("tags", ["cellar", "garden"])),
        [0, 2]
    );
    assert_eq!(
        matching(&db, Filter::any_of("tags", Vec::<&str>::new())),
        []
    );
}

#[test]
fn filters_combine_and_negate() {
    let db = db();
    let lyon = || Filter::eq("city", "Lyon");
    assert_eq!(matching(&db, lyon() & Filter::range("rooms", 3..)), [2]);
    assert_eq!(
        matching(&db, lyon() | Filter::eq("garden", true)),
        [0, 1, 2, 4]
    );
    // A missing value never matches, so negation includes it.
    assert_eq!(matching(&db, !Filter::eq("garden", true)), [0, 2, 3]);
    assert_eq!(matching(&db, !!lyon()), [0, 2]);
    let complex = (lyon() | Filter::eq("city", "Paris")) & !Filter::eq("tags", "garden");
    assert_eq!(matching(&db, complex), [0, 1]);
    let keys = Filter::keys([Key::new(4), Key::new(1), Key::new(99)]);
    assert_eq!(matching(&db, keys & Filter::range("rooms", 4..)), [4]);
}

#[test]
fn invalid_filters_are_rejected_before_searching() {
    let db = db();
    let origin = [0.0, 0.0];
    let check = |filter: Filter| db.search(Knn::new(&origin, 1).filter(&filter)).unwrap_err();
    assert!(matches!(check(Filter::eq("nope", 1)), RingDbError::UnknownField(f) if f == "nope"));
    for filter in [
        Filter::eq("title", "x"),
        Filter::eq("rooms", "three"),
        Filter::eq("city", 3),
        Filter::eq("garden", 1),
        Filter::range("city", "a".."b"),
        Filter::range("garden", false..=true),
        Filter::eq("rooms", 1) & Filter::eq("photos", "a.jpg"),
    ] {
        assert!(matches!(check(filter), RingDbError::InvalidQuery(_)));
    }
    let empty = Db::in_memory(listings(), Options::default()).unwrap();
    let filter = Filter::eq("nope", 1);
    assert!(
        empty
            .search(Disk::new(&origin, 1.0).filter(&filter))
            .is_err()
    );
}

#[test]
fn records_are_validated_against_the_schema() {
    let db = Db::in_memory(listings(), Options::default().max_record_bytes(64)).unwrap();
    let base = || Record::new([0.0, 0.0]);
    let invalid = |record: Record| {
        let error = db.add(record).unwrap_err();
        assert!(
            matches!(
                error,
                RingDbError::InvalidRecord(_) | RingDbError::UnknownField(_)
            ),
            "{error}"
        );
    };
    invalid(base().set("nope", 1));
    invalid(base().set("rooms", -1));
    invalid(base().set("rooms", u64::MAX));
    invalid(base().set("rooms", 1.5f32));
    invalid(base().set("surface", f32::NAN));
    invalid(base().set("surface", 3));
    invalid(base().set("garden", "yes"));
    invalid(base().set("city", ["a", "b"]));
    invalid(base().set("thumbnail", "text"));
    invalid(base().set("rooms", 1).set("rooms", 2));
    invalid(base().set("title", "x".repeat(57)));
    db.add(base().set("title", "x".repeat(56))).unwrap();
    // A single string is a one-item list.
    let key = db
        .add(base().set("tags", "pool").set("photos", "p.jpg"))
        .unwrap();
    db.flush().unwrap();
    let doc = db.get(key).unwrap();
    assert_eq!(doc.get("tags"), Some(&Value::TextList(vec!["pool".into()])));
    assert_eq!(
        doc.get("photos"),
        Some(&Value::TextList(vec!["p.jpg".into()]))
    );
    // Failed adds do not use keys.
    assert_eq!(key, Key::new(1));

    for schema in [
        Schema::new(0),
        Schema::new(1).field("", FieldType::U32),
        Schema::new(1)
            .field("a", FieldType::U32)
            .field("a", FieldType::Text),
    ] {
        assert!(matches!(
            Db::in_memory(schema, Options::default()),
            Err(RingDbError::InvalidSchema(_))
        ));
    }
}

#[test]
fn records_become_visible_after_flush_in_new_segments() {
    let db = Db::in_memory(common::schema(1), common::manual(Default::default())).unwrap();
    let origin = [0.0];
    let add = |x: f32, group: u32| db.add(Record::new([x]).set("group", group)).unwrap();
    // Equal distances in different segments still sort by key.
    let first = [add(3.0, 0), add(-1.0, 1), add(5.0, 2)];
    let snapshot = db.snapshot();
    assert!(db.is_empty());
    assert!(db.search(Knn::new(&origin, 10)).unwrap().is_empty());
    assert!(db.get(first[0]).is_err());
    db.flush().unwrap();
    let second = [add(1.0, 1), add(-3.0, 0)];
    db.flush().unwrap();
    db.flush().unwrap();
    let third = [add(0.5, 1)];
    db.flush().unwrap();
    assert_eq!(db.len(), 6);
    assert_eq!(db.snapshot().segments(), 3);
    assert!(snapshot.is_empty(), "snapshots do not change");

    let hits = db.search(Knn::new(&origin, 4)).unwrap();
    let keys: Vec<Key> = hits.iter().map(|hit| hit.key).collect();
    assert_eq!(keys, [third[0], first[1], second[0], first[0]]);
    assert_eq!(
        hits.iter().map(|hit| hit.dist_sq).collect::<Vec<_>>(),
        [0.25, 1.0, 1.0, 9.0]
    );

    let group = common::group(1);
    let inside = db.search(Disk::new(&origin, 1.0).filter(&group)).unwrap();
    let mut found: Vec<(Key, f32)> = inside
        .iter()
        .map(|hit| (hit.key, hit.dists_sq()[0]))
        .collect();
    found.sort_by_key(|hit| hit.0);
    assert_eq!(found, [(first[1], 1.0), (second[0], 1.0), (third[0], 0.25)]);

    let keys = Filter::keys([first[2], second[1]]);
    assert_eq!(
        knn_keys(&db.search(Knn::new(&origin, 9).filter(&keys)).unwrap()),
        [4, 2]
    );
    for key in first.iter().chain(&second).chain(&third) {
        assert_eq!(db.get(*key).unwrap().key(), *key);
    }
    assert_eq!(db.get(second[1]).unwrap().vector(), [-3.0]);
}
