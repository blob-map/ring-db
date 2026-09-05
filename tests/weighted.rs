use std::collections::BTreeSet;

use rand::{RngExt, SeedableRng};
use ringdb::{
    BackendPreference, DiskIntersectionQuery, DiskQuery, Hit, QueryResult, RangeQuery, RingDb,
    RingDbConfig, RingDbError, RingQuery, SealedRingDb,
};

fn build_db(vectors: &[Vec<f32>]) -> SealedRingDb {
    let dims = vectors
        .first()
        .expect("test dataset must not be empty")
        .len();
    let mut db: RingDb = RingDb::new(RingDbConfig::new(dims)).unwrap();
    for vector in vectors {
        assert_eq!(vector.len(), dims);
        db.add_vector(vector, ()).unwrap();
    }
    db.build().unwrap()
}

fn sorted_hits(mut result: QueryResult) -> Vec<Hit> {
    result.hits.sort_by_key(|hit| hit.id);
    result.hits
}

fn assert_close(actual: f32, expected: f64) {
    let actual = f64::from(actual);
    let tolerance = 2.0e-4 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "distance mismatch: actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn assert_results_close(actual: QueryResult, expected: QueryResult) {
    let actual = sorted_hits(actual);
    let expected = sorted_hits(expected);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.id, expected.id);
        assert_close(actual.dist_sq, f64::from(expected.dist_sq));
    }
}

fn reference_weighted_l2_sq(vector: &[f32], query: &[f32], weights: &[f32]) -> f64 {
    assert_eq!(vector.len(), query.len());
    assert_eq!(vector.len(), weights.len());
    vector
        .iter()
        .zip(query)
        .zip(weights)
        .map(|((&value, &query_value), &weight)| {
            let delta = f64::from(value) - f64::from(query_value);
            f64::from(weight) * delta * delta
        })
        .sum()
}

fn expected_hits(distances: &[f64], predicate: impl Fn(f64) -> bool) -> Vec<(u32, f64)> {
    distances
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(id, distance)| predicate(distance).then_some((id as u32, distance)))
        .collect()
}

fn assert_matches_reference(result: QueryResult, expected: &[(u32, f64)]) {
    let hits = sorted_hits(result);
    assert_eq!(
        hits.len(),
        expected.len(),
        "unexpected hit IDs: {:?}",
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>()
    );
    for (hit, &(expected_id, expected_distance)) in hits.iter().zip(expected) {
        assert_eq!(hit.id, expected_id);
        assert_close(hit.dist_sq, expected_distance);
    }
}

fn query_error<T>(result: Result<T, RingDbError>) -> RingDbError {
    match result {
        Ok(_) => panic!("query unexpectedly succeeded"),
        Err(error) => error,
    }
}

fn assert_dimension_mismatch(error: RingDbError, expected: usize, got: usize) {
    assert!(matches!(
        error,
        RingDbError::DimensionMismatch {
            expected: actual_expected,
            got: actual_got,
        } if actual_expected == expected && actual_got == got
    ));
}

fn assert_invalid_query(error: RingDbError) {
    assert!(matches!(error, RingDbError::InvalidQuery(_)));
}

#[test]
fn analytical_weighted_distances_use_weights_as_final_coefficients() {
    let vectors = vec![vec![1.0, 1.0], vec![4.0, 6.0]];
    let db = build_db(&vectors);

    let origin = [0.0, 0.0];
    let weights = [4.0, 1.0];
    let hits = sorted_hits(
        db.query_disk(&DiskQuery::new(&origin, 20.0).with_weights(&weights))
            .unwrap(),
    );
    assert_close(hits[0].dist_sq, 5.0);

    let query = [1.0, 2.0];
    let weights = [2.0, 3.0];
    let hits = sorted_hits(
        db.query_disk(&DiskQuery::new(&query, 20.0).with_weights(&weights))
            .unwrap(),
    );
    let hit = hits.iter().find(|hit| hit.id == 1).unwrap();
    assert_close(hit.dist_sq, 66.0);
}

#[test]
fn ring_range_and_disk_apply_weighted_l2() {
    let vectors = vec![
        vec![1.0, 1.0],
        vec![0.0, 2.0],
        vec![2.0, 0.0],
        vec![0.0, 3.0],
    ];
    let db = build_db(&vectors);
    let query = [0.0, 0.0];
    let weights = [4.0, 1.0];

    let ring = db
        .query(&RingQuery::new(&query, 5.0f32.sqrt(), 0.05).with_weights(&weights))
        .unwrap();
    let ring_hits = sorted_hits(ring);
    assert_eq!(ring_hits.iter().map(|hit| hit.id).collect::<Vec<_>>(), [0]);
    assert_close(ring_hits[0].dist_sq, 5.0);

    let range = db
        .query_range(&RangeQuery::new(&query, 2.1, 2.5).with_weights(&weights))
        .unwrap();
    let range_hits = sorted_hits(range);
    assert_eq!(range_hits.iter().map(|hit| hit.id).collect::<Vec<_>>(), [0]);
    assert_close(range_hits[0].dist_sq, 5.0);

    let disk = db
        .query_disk(&DiskQuery::new(&query, 2.5).with_weights(&weights))
        .unwrap();
    let disk_hits = sorted_hits(disk);
    assert_eq!(
        disk_hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        [0, 1]
    );
    assert_close(disk_hits[0].dist_sq, 5.0);
    assert_close(disk_hits[1].dist_sq, 4.0);
}

#[test]
fn zero_weight_dimensions_contribute_exactly_zero() {
    let vectors = vec![vec![0.0, 100.0], vec![1.0, 1_000.0], vec![2.0, 0.0]];
    let db = build_db(&vectors);
    let query = [0.0, 0.0];
    let weights = [1.0, 0.0];

    let disk = db
        .query_disk(&DiskQuery::new(&query, 1.0).with_weights(&weights))
        .unwrap();
    let hits = sorted_hits(disk);
    assert_eq!(hits.iter().map(|hit| hit.id).collect::<Vec<_>>(), [0, 1]);
    assert_eq!(hits[0].dist_sq, 0.0);
    assert_eq!(hits[1].dist_sq, 1.0);

    let ring = db
        .query(&RingQuery::new(&query, 0.0, 0.0).with_weights(&weights))
        .unwrap();
    let hits = sorted_hits(ring);
    assert_eq!(hits.iter().map(|hit| hit.id).collect::<Vec<_>>(), [0]);
    assert_eq!(hits[0].dist_sq, 0.0);
}

#[test]
fn changing_per_dimension_weights_reverses_candidate_order() {
    let vectors = vec![vec![3.0, 1.0], vec![1.0, 3.0]];
    let db = build_db(&vectors);
    let query = [0.0, 0.0];
    let emphasize_first = [4.0, 1.0];
    let emphasize_second = [1.0, 4.0];

    let mut first = db
        .query_disk(&DiskQuery::new(&query, 10.0).with_weights(&emphasize_first))
        .unwrap()
        .hits;
    first.sort_by(|a, b| a.dist_sq.total_cmp(&b.dist_sq));
    assert_eq!(first.iter().map(|hit| hit.id).collect::<Vec<_>>(), [1, 0]);
    assert_close(first[0].dist_sq, 13.0);
    assert_close(first[1].dist_sq, 37.0);

    let mut second = db
        .query_disk(&DiskQuery::new(&query, 10.0).with_weights(&emphasize_second))
        .unwrap()
        .hits;
    second.sort_by(|a, b| a.dist_sq.total_cmp(&b.dist_sq));
    assert_eq!(second.iter().map(|hit| hit.id).collect::<Vec<_>>(), [0, 1]);
    assert_close(second[0].dist_sq, 13.0);
    assert_close(second[1].dist_sq, 37.0);
}

#[test]
fn mixed_intersection_uses_each_disks_metric_and_reports_the_first() {
    let vectors = vec![
        vec![0.0, 1.0],
        vec![0.5, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 2.0],
    ];
    let db = build_db(&vectors);
    let origin = [0.0, 0.0];
    let upper_right = [1.0, 1.0];
    let lower_right = [1.0, 0.0];
    let first_weights = [4.0, 1.0];
    let second_weights = [1.0, 4.0];
    let disks = [
        DiskQuery::new(&origin, 2.0).with_weights(&first_weights),
        DiskQuery::new(&upper_right, 1.0).with_weights(&second_weights),
        DiskQuery::new(&lower_right, 1.5),
    ];

    let hits = sorted_hits(
        db.query_disk_intersection(&DiskIntersectionQuery::new(&disks))
            .unwrap(),
    );
    assert_eq!(hits.iter().map(|hit| hit.id).collect::<Vec<_>>(), [0, 1]);
    assert_close(hits[0].dist_sq, 1.0);
    // Ordinary L2 would be 1.25. The first disk's weighted metric is 2.0.
    assert_close(hits[1].dist_sq, 2.0);
}

#[test]
fn all_one_weights_match_ordinary_l2_for_every_query_type() {
    let dims = 17;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0xA11_0AE5);
    let vectors: Vec<Vec<f32>> = (0..97)
        .map(|_| (0..dims).map(|_| rng.random_range(-2.0f32..2.0)).collect())
        .collect();
    let query: Vec<f32> = (0..dims).map(|_| rng.random_range(-2.0f32..2.0)).collect();
    let second_query: Vec<f32> = (0..dims).map(|_| rng.random_range(-2.0f32..2.0)).collect();
    let ones = vec![1.0; dims];
    let db = build_db(&vectors);

    assert_results_close(
        db.query(&RingQuery::new(&query, 50.0, 50.0)).unwrap(),
        db.query(&RingQuery::new(&query, 50.0, 50.0).with_weights(&ones))
            .unwrap(),
    );
    assert_results_close(
        db.query_range(&RangeQuery::new(&query, 0.0, 100.0))
            .unwrap(),
        db.query_range(&RangeQuery::new(&query, 0.0, 100.0).with_weights(&ones))
            .unwrap(),
    );
    assert_results_close(
        db.query_disk(&DiskQuery::new(&query, 100.0)).unwrap(),
        db.query_disk(&DiskQuery::new(&query, 100.0).with_weights(&ones))
            .unwrap(),
    );

    let ordinary_disks = [
        DiskQuery::new(&query, 100.0),
        DiskQuery::new(&second_query, 100.0),
    ];
    let weighted_disks = [
        DiskQuery::new(&query, 100.0).with_weights(&ones),
        DiskQuery::new(&second_query, 100.0).with_weights(&ones),
    ];
    assert_results_close(
        db.query_disk_intersection(&DiskIntersectionQuery::new(&ordinary_disks))
            .unwrap(),
        db.query_disk_intersection(&DiskIntersectionQuery::new(&weighted_disks))
            .unwrap(),
    );
}

#[test]
fn wrong_weight_dimensions_are_rejected_for_every_query_type() {
    let db = build_db(&[vec![0.0, 0.0]]);
    let query = [0.0, 0.0];
    let short = [1.0];

    assert_dimension_mismatch(
        query_error(db.query(&RingQuery::new(&query, 1.0, 0.5).with_weights(&short))),
        2,
        1,
    );
    assert_dimension_mismatch(
        query_error(db.query_range(&RangeQuery::new(&query, 0.0, 1.0).with_weights(&short))),
        2,
        1,
    );
    assert_dimension_mismatch(
        query_error(db.query_disk(&DiskQuery::new(&query, 1.0).with_weights(&short))),
        2,
        1,
    );

    let disks = [DiskQuery::new(&query, 1.0).with_weights(&short)];
    assert_dimension_mismatch(
        query_error(db.query_disk_intersection(&DiskIntersectionQuery::new(&disks))),
        2,
        1,
    );
}

#[test]
fn negative_and_non_finite_weights_are_rejected_for_every_query_type() {
    let db = build_db(&[vec![0.0, 0.0]]);
    let query = [0.0, 0.0];

    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.25] {
        let weights = [1.0, invalid];
        assert_invalid_query(query_error(
            db.query(&RingQuery::new(&query, 1.0, 0.5).with_weights(&weights)),
        ));
        assert_invalid_query(query_error(
            db.query_range(&RangeQuery::new(&query, 0.0, 1.0).with_weights(&weights)),
        ));
        assert_invalid_query(query_error(
            db.query_disk(&DiskQuery::new(&query, 1.0).with_weights(&weights)),
        ));

        let disks = [DiskQuery::new(&query, 1.0).with_weights(&weights)];
        assert_invalid_query(query_error(
            db.query_disk_intersection(&DiskIntersectionQuery::new(&disks)),
        ));
    }
}

#[test]
fn malformed_weights_are_rejected_on_empty_databases_and_nonfirst_disks() {
    let empty: SealedRingDb = RingDb::new(RingDbConfig::new(2)).unwrap().build().unwrap();
    let query = [0.0, 0.0];
    let short = [1.0];
    let negative = [1.0, -1.0];

    assert_dimension_mismatch(
        query_error(empty.query_disk(&DiskQuery::new(&query, 1.0).with_weights(&short))),
        2,
        1,
    );
    assert_invalid_query(query_error(
        empty.query(&RingQuery::new(&query, 1.0, 0.5).with_weights(&negative)),
    ));

    let db = build_db(&[vec![0.0, 0.0]]);
    let invalid_second = [
        DiskQuery::new(&query, 1.0),
        DiskQuery::new(&query, 1.0).with_weights(&negative),
    ];
    assert_invalid_query(query_error(
        db.query_disk_intersection(&DiskIntersectionQuery::new(&invalid_second)),
    ));

    let wrong_length_second = [
        DiskQuery::new(&query, 1.0),
        DiskQuery::new(&query, 1.0).with_weights(&short),
    ];
    assert_dimension_mismatch(
        query_error(db.query_disk_intersection(&DiskIntersectionQuery::new(&wrong_length_second))),
        2,
        1,
    );
}

fn midpoint_distance(sorted_distances_sq: &[f64], split: usize) -> f32 {
    assert!(split > 0 && split < sorted_distances_sq.len());
    let left = sorted_distances_sq[split - 1].sqrt();
    let right = sorted_distances_sq[split].sqrt();
    ((left + right) * 0.5) as f32
}

#[test]
fn randomized_weighted_queries_match_a_scalar_reference() {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0x005C_A1A2);

    for dims in [
        1usize, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128,
    ] {
        let vectors: Vec<Vec<f32>> = (0..73)
            .map(|_| (0..dims).map(|_| rng.random_range(-3.0f32..3.0)).collect())
            .collect();
        let query: Vec<f32> = (0..dims).map(|_| rng.random_range(-3.0f32..3.0)).collect();
        let second_query: Vec<f32> = (0..dims).map(|_| rng.random_range(-3.0f32..3.0)).collect();
        let weights: Vec<f32> = (0..dims)
            .map(|index| {
                if index % 11 == 7 {
                    0.0
                } else {
                    rng.random_range(0.05f32..3.0)
                }
            })
            .collect();
        let second_weights: Vec<f32> = (0..dims)
            .map(|index| {
                if index % 13 == 5 {
                    0.0
                } else {
                    rng.random_range(0.05f32..3.0)
                }
            })
            .collect();
        let distances: Vec<f64> = vectors
            .iter()
            .map(|vector| reference_weighted_l2_sq(vector, &query, &weights))
            .collect();
        let second_distances: Vec<f64> = vectors
            .iter()
            .map(|vector| reference_weighted_l2_sq(vector, &second_query, &second_weights))
            .collect();
        let mut sorted_distances = distances.clone();
        sorted_distances.sort_by(f64::total_cmp);
        let mut sorted_second_distances = second_distances.clone();
        sorted_second_distances.sort_by(f64::total_cmp);

        let lower = midpoint_distance(&sorted_distances, vectors.len() / 4);
        let disk_radius = midpoint_distance(&sorted_distances, vectors.len() / 2);
        let upper = midpoint_distance(&sorted_distances, vectors.len() * 3 / 4);
        let second_radius = midpoint_distance(&sorted_second_distances, vectors.len() / 2);
        let db = build_db(&vectors);

        // Backends square public distance bounds in f32, so mirror that rounding
        // before comparing against the higher-precision scalar distances.
        let disk_upper_sq = f64::from(disk_radius * disk_radius);
        let expected = expected_hits(&distances, |distance| distance <= disk_upper_sq);
        assert_matches_reference(
            db.query_disk(&DiskQuery::new(&query, disk_radius).with_weights(&weights))
                .unwrap(),
            &expected,
        );

        let lower_sq = f64::from(lower * lower);
        let upper_sq = f64::from(upper * upper);
        let expected = expected_hits(&distances, |distance| {
            distance >= lower_sq && distance <= upper_sq
        });
        assert_matches_reference(
            db.query_range(&RangeQuery::new(&query, lower, upper).with_weights(&weights))
                .unwrap(),
            &expected,
        );

        let d = (lower + upper) * 0.5;
        let lambda = (upper - lower) * 0.5;
        let ring_lower = (d - lambda).max(0.0);
        let ring_upper = d + lambda;
        let ring_lower_sq = f64::from(ring_lower * ring_lower);
        let ring_upper_sq = f64::from(ring_upper * ring_upper);
        let expected = expected_hits(&distances, |distance| {
            distance >= ring_lower_sq && distance <= ring_upper_sq
        });
        assert_matches_reference(
            db.query(&RingQuery::new(&query, d, lambda).with_weights(&weights))
                .unwrap(),
            &expected,
        );

        let second_upper_sq = f64::from(second_radius * second_radius);
        let expected: Vec<(u32, f64)> = distances
            .iter()
            .copied()
            .zip(&second_distances)
            .enumerate()
            .filter_map(|(id, (first, &second))| {
                (first <= disk_upper_sq && second <= second_upper_sq).then_some((id as u32, first))
            })
            .collect();
        let disks = [
            DiskQuery::new(&query, disk_radius).with_weights(&weights),
            DiskQuery::new(&second_query, second_radius).with_weights(&second_weights),
        ];
        assert_matches_reference(
            db.query_disk_intersection(&DiskIntersectionQuery::new(&disks))
                .unwrap(),
            &expected,
        );
    }
}

#[test]
fn persisted_database_accepts_arbitrary_query_weights_without_storing_them() {
    let dir = tempfile::tempdir().unwrap();
    let mut db: RingDb = RingDb::new(RingDbConfig::new(2).with_persist_dir(dir.path())).unwrap();
    for vector in [[1.0, 1.0], [2.0, 0.0], [0.0, 3.0]] {
        db.add_vector(&vector, ()).unwrap();
    }
    let db = db.build().unwrap();
    let query = [0.0, 0.0];
    let first_weights = [4.0, 1.0];
    let before_reload = db
        .query_disk(&DiskQuery::new(&query, 3.0).with_weights(&first_weights))
        .unwrap();
    assert_eq!(
        sorted_hits(before_reload)
            .iter()
            .map(|hit| hit.id)
            .collect::<Vec<_>>(),
        [0, 2]
    );
    drop(db);

    let files_before: BTreeSet<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert!(files_before.contains(std::ffi::OsStr::new("vectors.bin")));
    assert!(!files_before.contains(std::ffi::OsStr::new("norms_sq.bin")));
    assert!(
        files_before
            .iter()
            .all(|name| !name.to_string_lossy().contains("weight"))
    );

    let loaded = RingDb::<()>::load(dir.path(), BackendPreference::Cpu).unwrap();
    let after_reload = loaded
        .query_disk(&DiskQuery::new(&query, 3.0).with_weights(&first_weights))
        .unwrap();
    assert_eq!(
        sorted_hits(after_reload)
            .iter()
            .map(|hit| hit.id)
            .collect::<Vec<_>>(),
        [0, 2]
    );

    let second_weights = [1.0, 4.0];
    let different_metric = loaded
        .query_disk(&DiskQuery::new(&query, 3.0).with_weights(&second_weights))
        .unwrap();
    assert_eq!(
        sorted_hits(different_metric)
            .iter()
            .map(|hit| hit.id)
            .collect::<Vec<_>>(),
        [0, 1]
    );

    let files_after: BTreeSet<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(files_before, files_after);
}
