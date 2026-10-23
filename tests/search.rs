//! Analytical semantics of the two search primitives on the CPU backend.
mod common;

use common::{assert_close, cpu, group, keys, knn_keys, region_keys, region_rows};
use ringdb::{Db, Disk, Knn, MAX_DISKS, Metric, Options, Record, RingDbError, Schema};

fn line(xs: &[f32]) -> Db {
    cpu(&xs.iter().map(|&x| vec![x, 0.0]).collect::<Vec<_>>())
}

fn invalid<T>(result: Result<T, RingDbError>) {
    assert!(matches!(result, Err(RingDbError::InvalidQuery(_))));
}

fn mismatch<T>(result: Result<T, RingDbError>, expected: usize, got: usize) {
    match result {
        Err(RingDbError::DimensionMismatch {
            expected: e,
            got: g,
        }) => assert_eq!((e, g), (expected, got)),
        Err(other) => panic!("expected a dimension mismatch, got {other}"),
        Ok(_) => panic!("expected a dimension mismatch"),
    }
}

#[test]
fn disk_bounds_are_inclusive() {
    let db = line(&[0.0, 1.0, 3.0, 5.0, 7.0]);
    let origin = [0.0, 0.0];
    assert_eq!(
        region_keys(&db.search(Disk::new(&origin, 5.0)).unwrap()),
        [0, 1, 2, 3]
    );
    assert_eq!(
        region_keys(&db.search(Disk::new(&origin, 0.0)).unwrap()),
        [0]
    );
    assert!(db.search(Disk::new(&[100.0, 0.0], 1.0)).unwrap().is_empty());
}

#[test]
fn disk_reports_squared_distance() {
    let db = cpu(&[vec![3.0, 4.0], vec![1.0, 0.0]]);
    let result = db.search(Disk::new(&[0.0, 0.0], 10.0)).unwrap();
    assert_eq!(result.n_disks(), 1);
    assert_eq!(region_rows(&result), [(0, vec![25.0]), (1, vec![1.0])]);
}

#[test]
fn intersections_keep_rows_inside_every_disk_and_report_each_distance() {
    // Inside the first disk only, both, both (first boundary), second only, first only.
    let db = cpu(&[
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![2.0, 0.0],
        vec![3.0, 0.0],
        vec![1.0, 1.0],
    ]);
    let (origin, right) = ([0.0, 0.0], [2.0, 0.0]);
    let result = db
        .search(Disk::new(&origin, 2.0).and(Disk::new(&right, 1.0)))
        .unwrap();
    assert_eq!(result.n_disks(), 2);
    assert_eq!(
        region_rows(&result),
        [(1, vec![1.0, 1.0]), (2, vec![4.0, 0.0])]
    );
    let none = db
        .search(Disk::new(&origin, 0.5).and(Disk::new(&right, 0.5)))
        .unwrap();
    assert!(none.is_empty());
}

#[test]
fn every_disk_uses_its_own_metric() {
    let db = cpu(&[
        vec![0.0, 1.0],
        vec![0.5, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 2.0],
    ]);
    let (origin, upper_right, lower_right) = ([0.0, 0.0], [1.0, 1.0], [1.0, 0.0]);
    let (first, second) = ([4.0, 1.0], [1.0, 4.0]);
    let result = db
        .search(
            Disk::new(&origin, 2.0)
                .metric(Metric::WeightedL2(&first))
                .and(Disk::new(&upper_right, 1.0).metric(Metric::WeightedL2(&second)))
                .and(Disk::new(&lower_right, 1.5)),
        )
        .unwrap();
    let rows = region_rows(&result);
    assert_eq!(rows.iter().map(|r| r.0).collect::<Vec<_>>(), [0, 1]);
    // Ordinary L2 from the origin would be 1.25 for key 1.
    assert_eq!(rows[0].1, [1.0, 1.0, 2.0]);
    assert_eq!(rows[1].1, [2.0, 0.25, 1.25]);
}

#[test]
fn weights_are_final_coefficients() {
    let db = cpu(&[vec![1.0, 1.0], vec![4.0, 6.0]]);
    let weights = [2.0, 3.0];
    let rows = region_rows(
        &db.search(Disk::new(&[1.0, 2.0], 20.0).metric(Metric::WeightedL2(&weights)))
            .unwrap(),
    );
    assert_eq!(rows[0], (0, vec![3.0]));
    assert_eq!(rows[1], (1, vec![66.0]));
}

#[test]
fn changing_weights_reverses_knn_order() {
    let db = cpu(&[vec![3.0, 1.0], vec![1.0, 3.0]]);
    let origin = [0.0, 0.0];
    for (weights, order) in [([4.0, 1.0], [1, 0]), ([1.0, 4.0], [0, 1])] {
        let hits = db
            .search(Knn::new(&origin, 2).metric(Metric::WeightedL2(&weights)))
            .unwrap();
        assert_eq!(knn_keys(&hits), order);
        assert_eq!(hits[0].dist_sq, 13.0);
        assert_eq!(hits[1].dist_sq, 37.0);
    }
}

#[test]
fn zero_weights_ignore_even_overflowing_differences() {
    let db = cpu(&[vec![-f32::MAX, 0.0], vec![0.0, 1.0], vec![f32::MAX, 0.0]]);
    let query = [f32::MAX, 0.0];
    let weights = [0.0, 1.0];
    let metric = Metric::WeightedL2(&weights);
    let hits = db.search(Knn::new(&query, 3).metric(metric)).unwrap();
    assert_eq!(knn_keys(&hits), [0, 2, 1]);
    assert_eq!(
        hits.iter().map(|h| h.dist_sq).collect::<Vec<_>>(),
        [0.0, 0.0, 1.0]
    );
    let inside = db.search(Disk::new(&query, 0.0).metric(metric)).unwrap();
    assert_eq!(region_keys(&inside), [0, 2]);
}

#[test]
fn unweighted_distances_use_direct_differences() {
    let db = cpu(&[vec![10_000_003.0], vec![10_000_001.0], vec![10_000_002.0]]);
    let query = [10_000_000.0];
    let hits = db.search(Knn::new(&query, 3)).unwrap();
    assert_eq!(knn_keys(&hits), [1, 2, 0]);
    assert_eq!(
        hits.iter().map(|h| h.dist_sq).collect::<Vec<_>>(),
        [1.0, 4.0, 9.0]
    );
    let db = cpu(&[vec![f32::MAX], vec![-f32::MAX]]);
    let inside = db.search(Disk::new(&[f32::MAX], 0.0)).unwrap();
    assert_eq!(region_keys(&inside), [0]);
}

#[test]
fn knn_is_ordered_by_distance_then_key() {
    let db = line(&[2.0, -1.0, 1.0, 0.0, -2.0, 3.0]);
    let origin = [0.0, 0.0];
    let hits = db.search(Knn::new(&origin, 4)).unwrap();
    assert_eq!(knn_keys(&hits), [3, 1, 2, 0]);
    assert!(db.search(Knn::new(&origin, 0)).unwrap().is_empty());
    assert_eq!(db.search(Knn::new(&origin, 100)).unwrap().len(), 6);
}

#[test]
fn filters_restrict_both_primitives() {
    let db = line(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let origin = [0.0, 0.0];
    // Group 1 holds keys 1 and 4.
    let group = group(1);
    let hits = db.search(Knn::new(&origin, 10).filter(&group)).unwrap();
    assert_eq!(knn_keys(&hits), [1, 4]);
    let inside = db.search(Disk::new(&origin, 3.0).filter(&group)).unwrap();
    assert_eq!(region_keys(&inside), [1]);
    let others = !group;
    let both = db
        .search(
            Disk::new(&origin, 10.0)
                .and(Disk::new(&[5.0, 0.0], 4.5))
                .filter(&others),
        )
        .unwrap();
    assert_eq!(region_keys(&both), [2, 3, 5]);

    let missing = common::group(99);
    assert!(
        db.search(Knn::new(&origin, 10).filter(&missing))
            .unwrap()
            .is_empty()
    );
    assert!(
        db.search(Disk::new(&origin, 9.0).filter(&missing))
            .unwrap()
            .is_empty()
    );

    let sparse = keys([0, 5, 77]);
    assert_eq!(
        knn_keys(&db.search(Knn::new(&origin, 1).filter(&sparse)).unwrap()),
        [0]
    );
}

#[test]
fn empty_databases_return_no_results() {
    let db = Db::in_memory(Schema::new(2), Options::default()).unwrap();
    let origin = [0.0, 0.0];
    assert!(db.search(Knn::new(&origin, 3)).unwrap().is_empty());
    let result = db.search(Disk::new(&origin, 3.0)).unwrap();
    assert!(result.is_empty());
    assert_eq!(result.n_disks(), 1);
    // Malformed queries are still rejected.
    mismatch(db.search(Knn::new(&[0.0], 1)), 2, 1);
}

#[test]
fn malformed_queries_are_rejected() {
    let db = line(&[1.0]);
    let origin = [0.0, 0.0];
    let short = [1.0];
    let schemaless = Db::in_memory(Schema::new(2), Options::default()).unwrap();
    invalid(schemaless.add(Record::new([f32::NAN, 0.0])));
    mismatch(schemaless.add(Record::new([0.0])), 2, 1);
    mismatch(db.search(Knn::new(&[0.0; 3], 1)), 2, 3);
    mismatch(
        db.search(Knn::new(&origin, 1).metric(Metric::WeightedL2(&short))),
        2,
        1,
    );
    invalid(db.search(Knn::new(&[f32::INFINITY, 0.0], 1)));
    invalid(db.search(Knn::new(&origin, 1).metric(Metric::WeightedL2(&[-1.0, 1.0]))));
    invalid(db.search(Knn::new(&origin, 1).metric(Metric::WeightedL2(&[f32::NAN, 1.0]))));
    invalid(db.search(Disk::new(&origin, -1.0)));
    invalid(db.search(Disk::new(&origin, f32::INFINITY)));
    invalid(db.search(Disk::new(&origin, f32::NAN)));
    // Later disks are validated too.
    let later = Disk::new(&origin, 1.0);
    mismatch(db.search(later.and(Disk::new(&[0.0; 3], 1.0))), 2, 3);
    invalid(db.search(later.and(Disk::new(&origin, 1.0).metric(Metric::WeightedL2(&[1.0, -1.0])))));
    let mut region = later.and(later);
    for _ in 2..MAX_DISKS {
        region = region.and(later);
    }
    assert_eq!(db.search(region.clone()).unwrap().n_disks(), MAX_DISKS);
    invalid(db.search(region.and(later)));
}

#[test]
fn all_one_weights_match_ordinary_l2() {
    let vectors: Vec<Vec<f32>> = (0..97)
        .map(|i| {
            (0..17)
                .map(|d| ((i * 31 + d * 7) % 23) as f32 / 5.0 - 2.0)
                .collect()
        })
        .collect();
    let db = cpu(&vectors);
    let query: Vec<f32> = (0..17).map(|d| (d % 5) as f32 / 3.0 - 1.0).collect();
    let ones = vec![1.0; 17];
    let ordinary = db.search(Knn::new(&query, 20)).unwrap();
    let weighted = db
        .search(Knn::new(&query, 20).metric(Metric::WeightedL2(&ones)))
        .unwrap();
    common::same_knn(&ordinary, &weighted);
    let ordinary = db
        .search(Disk::new(&query, 6.0).and(Disk::new(&ones, 7.0)))
        .unwrap();
    let weighted = db
        .search(
            Disk::new(&query, 6.0)
                .metric(Metric::WeightedL2(&ones))
                .and(Disk::new(&ones, 7.0).metric(Metric::WeightedL2(&ones))),
        )
        .unwrap();
    assert!(!ordinary.is_empty() && ordinary.len() < vectors.len());
    common::same_region(&ordinary, &weighted);
    for hit in ordinary.iter() {
        let vector = &vectors[hit.key.get() as usize];
        assert_close(
            hit.dists_sq()[0],
            common::reference_dist(vector, &query, None),
        );
        assert_close(
            hit.dists_sq()[1],
            common::reference_dist(vector, &ones, None),
        );
    }
}
