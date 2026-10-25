//! Facet counts compared with counts built from `get`.
mod common;

use std::collections::BTreeMap;

use common::keyword_db;
use ringdb::{Db, Disk, Facets, Filter, Key, Options, RingDbError, Value};

const FIELDS: [&str; 2] = ["city", "tags"];

/// The expected facets of `keys`, sorted like a facet search.
fn reference(db: &Db, keys: impl Iterator<Item = Key>) -> Vec<Vec<(String, usize)>> {
    let mut counts = vec![BTreeMap::<String, usize>::new(); FIELDS.len()];
    for key in keys {
        let doc = db.get(key).unwrap();
        for (field, counts) in FIELDS.iter().zip(&mut counts) {
            let values = match doc.get(field) {
                Some(Value::Text(value)) => vec![value.clone()],
                Some(Value::TextList(values)) => values.clone(),
                _ => Vec::new(),
            };
            for value in values {
                *counts.entry(value).or_default() += 1;
            }
        }
    }
    counts
        .into_iter()
        .map(|counts| {
            let mut values: Vec<(String, usize)> = counts.into_iter().collect();
            values.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            values
        })
        .collect()
}

fn top(facets: &[Vec<(String, usize)>], n: usize) -> Vec<Vec<(String, usize)>> {
    facets
        .iter()
        .map(|values| values.iter().take(n).cloned().collect())
        .collect()
}

#[test]
fn facets_match_counts_of_the_records() {
    let db = keyword_db(Options::default().commit_interval(None).auto_merge(false));
    assert_eq!(db.snapshot().segments(), 3);
    let filters = [
        None,
        Some(Filter::range("rooms", 2..=4)),
        Some(Filter::eq("tags", "tag1") | Filter::eq("city", "city44")),
        Some(!Filter::eq("city", "Lille")),
    ];
    let all: Vec<Key> = db
        .search(Disk::new(&[0.0, 0.0], 1e6))
        .unwrap()
        .keys()
        .collect();
    assert_eq!(all.len(), 860);
    for filter in &filters {
        let passes = |key: &Key| {
            filter
                .as_ref()
                .is_none_or(|f| db.count(&(f.clone() & Filter::keys([*key]))).unwrap() == 1)
        };
        let mut query = Facets::new(&FIELDS);
        if let Some(filter) = filter {
            query = query.filter(filter);
        }
        let expected = reference(&db, all.iter().copied().filter(passes));
        assert_eq!(db.search(query.clone()).unwrap(), expected);
        assert_eq!(db.search(query.clone().top(3)).unwrap(), top(&expected, 3));
        assert_eq!(db.search(query.top(0)).unwrap(), top(&expected, 0));

        for (center, radius) in [([10.0, 10.0], 8.0), ([0.0, 0.0], 4.5), ([50.0, 0.0], 1.0)] {
            let mut region = Disk::new(&center, radius).and(Disk::new(&[12.0, 12.0], 9.0));
            if let Some(filter) = filter {
                region = region.filter(filter);
            }
            let inside = db.search(region.clone()).unwrap();
            let expected = reference(&db, inside.keys());
            assert_eq!(db.search(region.clone().facets(&FIELDS)).unwrap(), expected);
            assert_eq!(
                db.search(region.facets(&FIELDS).top(2)).unwrap(),
                top(&expected, 2)
            );
        }
    }
    // Lille is in about half of the live records.
    let lille = &db.search(Facets::new(&["city"])).unwrap()[0][0];
    assert_eq!(lille.0, "Lille");
    assert!((300..500).contains(&lille.1));
}

#[test]
fn facets_need_keyword_fields() {
    let db = keyword_db(Options::default().commit_interval(None));
    assert!(matches!(
        db.search(Facets::new(&["rooms"])),
        Err(RingDbError::InvalidQuery(_))
    ));
    assert!(matches!(
        db.search(Disk::new(&[0.0, 0.0], 1.0).facets(&["street"])),
        Err(RingDbError::UnknownField(_))
    ));
    assert_eq!(db.search(Facets::new(&[])).unwrap(), Vec::<Vec<_>>::new());
}
