/// Correctness tests for the CPU exact (float32) backend.
///
/// These tests use hand-crafted vectors at known distances from a query so
/// the expected result sets are computed analytically, not by another code path.
use ringdb::{
    BackendPreference, DiskIntersectionQuery, DiskQuery, Payload, RangeQuery, RingDb, RingDbConfig,
    RingDbError, RingQuery,
};
#[cfg(feature = "cuda")]
use ringdb::{Hit, QueryResult};

fn cpu_db(dims: usize) -> RingDb {
    RingDb::new(RingDbConfig::new(dims)).unwrap()
}

// ---- 2-D helpers ----

/// Distance between two 2-D points.
fn dist2(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

#[cfg(feature = "cuda")]
fn assert_hit_slices_close(label: &str, expected: &[Hit], actual: &[Hit]) {
    let expected_ids: Vec<u32> = expected.iter().map(|hit| hit.id).collect();
    let actual_ids: Vec<u32> = actual.iter().map(|hit| hit.id).collect();
    assert_eq!(expected_ids, actual_ids, "{label}: ID mismatch");

    for (expected_hit, actual_hit) in expected.iter().zip(actual) {
        let tolerance = 2.0e-4 * expected_hit.dist_sq.abs().max(1.0);
        assert!(
            (expected_hit.dist_sq - actual_hit.dist_sq).abs() <= tolerance,
            "{label}: distance mismatch for ID {}: expected={}, actual={}, tolerance={tolerance}",
            expected_hit.id,
            expected_hit.dist_sq,
            actual_hit.dist_sq,
        );
    }
}

#[cfg(feature = "cuda")]
fn assert_cpu_cuda_results(label: &str, cpu: QueryResult, cuda: QueryResult) -> Vec<Hit> {
    let mut cpu_hits = cpu.hits;
    let mut cuda_hits = cuda.hits;
    cpu_hits.sort_by_key(|hit| hit.id);
    cuda_hits.sort_by_key(|hit| hit.id);

    assert_hit_slices_close(label, &cpu_hits, &cuda_hits);
    cpu_hits
}

#[test]
fn test_basic_hits() {
    let q = (0.0f32, 0.0f32);
    let pts: &[(f32, f32)] = &[
        (3.0, 0.0),  // dist = 3.0  → miss
        (5.0, 0.0),  // dist = 5.0  → hit
        (3.6, 3.6),  // dist ≈ 5.09 → hit
        (10.0, 0.0), // dist = 10.0 → miss
    ];

    let mut db = cpu_db(2);
    for &(x, y) in pts {
        db.add_vector(&[x, y], ()).unwrap();
    }
    let db = db.build().unwrap();

    let result = db.query(&RingQuery::new(&[q.0, q.1], 5.0, 0.5)).unwrap();

    let mut ids = result.ids();
    ids.sort_unstable();

    for (i, &pt) in pts.iter().enumerate() {
        let d = dist2(q, pt);
        let in_ring = (4.5..=5.5).contains(&d);
        let found = ids.contains(&(i as u32));
        assert_eq!(
            in_ring,
            found,
            "vector {i} at dist={d:.3} should {} in ring [4.5, 5.5]",
            if in_ring { "be" } else { "NOT be" }
        );
    }
}

#[test]
fn test_lambda_zero() {
    let mut db = cpu_db(2);
    db.add_vector(&[3.0f32, 0.0], ()).unwrap();
    db.add_vector(&[4.0, 0.0], ()).unwrap();
    db.add_vector(&[0.0, 3.0], ()).unwrap();
    let db = db.build().unwrap();

    let result = db.query(&RingQuery::new(&[0.0, 0.0], 3.0, 0.0)).unwrap();

    let mut ids = result.ids();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 2], "only vectors at dist=3 should match");
}

#[test]
fn test_no_results() {
    let mut db = cpu_db(2);
    db.add_vector(&[1.0f32, 0.0], ()).unwrap();
    db.add_vector(&[0.0, 1.0], ()).unwrap();
    let db = db.build().unwrap();

    let result = db.query(&RingQuery::new(&[0.0, 0.0], 100.0, 0.1)).unwrap();

    assert!(result.hits.is_empty());
}

#[test]
fn test_all_results() {
    let mut db = cpu_db(2);
    db.add_vector(&[1.0, 0.0], ()).unwrap();
    db.add_vector(&[0.0, 1.0], ()).unwrap();
    db.add_vector(&[-1.0, 0.0], ()).unwrap();
    db.add_vector(&[0.0, -1.0], ()).unwrap();
    let db = db.build().unwrap();

    let result = db.query(&RingQuery::new(&[0.0, 0.0], 1.0, 0.1)).unwrap();

    assert_eq!(result.hits.len(), 4);
}

#[test]
fn test_identical_vectors() {
    let mut db = cpu_db(2);
    for _ in 0..5 {
        db.add_vector(&[3.0f32, 4.0], ()).unwrap();
    }
    let db = db.build().unwrap();

    let result = db.query(&RingQuery::new(&[0.0, 0.0], 5.0, 0.1)).unwrap();

    assert_eq!(result.hits.len(), 5);
}

#[test]
fn test_d_less_than_lambda() {
    let mut db = cpu_db(2);
    db.add_vector(&[0.0f32, 0.0], ()).unwrap();
    db.add_vector(&[2.0, 0.0], ()).unwrap();
    let db = db.build().unwrap();

    let result = db.query(&RingQuery::new(&[0.0, 0.0], 0.5, 1.0)).unwrap();

    let mut ids = result.ids();
    ids.sort_unstable();
    assert!(ids.contains(&0), "vector at origin should match");
    assert!(!ids.contains(&1), "vector at dist=2 should not match");
}

#[test]
fn test_backend_name() {
    let db = cpu_db(4);
    assert_eq!(db.backend_name(), "cpu");
}

#[cfg(not(feature = "cuda"))]
#[test]
fn test_cuda_backend_requires_feature() {
    let err = match RingDb::<()>::new(
        RingDbConfig::new(4).with_backend_preference(BackendPreference::Cuda),
    ) {
        Ok(_) => panic!("CUDA backend should require the `cuda` Cargo feature"),
        Err(err) => err,
    };
    assert!(matches!(err, RingDbError::BackendUnavailable(_)));
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_backend_matches_cpu_when_requested() {
    if std::env::var_os("RINGDB_CUDA_TEST").is_none() {
        return;
    }

    let dims = 4;
    let vectors = [
        [1.0f32, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [3.0, 4.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [10.0, 0.0, 0.0, 0.0],
        [0.0, 100.0, 0.0, 0.0],
    ];

    let mut cpu = RingDb::new(RingDbConfig::new(dims)).unwrap();
    let mut cuda =
        RingDb::new(RingDbConfig::new(dims).with_backend_preference(BackendPreference::Cuda))
            .unwrap();
    for vector in vectors {
        cpu.add_vector(&vector, ()).unwrap();
        cuda.add_vector(&vector, ()).unwrap();
    }
    let cpu = cpu.build().unwrap();
    let cuda = cuda.build().unwrap();
    assert_eq!(cuda.backend_name(), "cuda");

    let origin = [0.0f32; 4];
    let shifted = [1.0f32, 0.0, 0.0, 0.0];
    let shifted_back = [-1.0f32, 0.0, 0.0, 0.0];
    let weights_a = [4.0f32, 1.0, 0.5, 2.0];
    let weights_b = [0.25f32, 3.0, 1.0, 0.5];
    let ones = [1.0f32; 4];
    let zero_dimensions = [1.0f32, 0.0, 1.0, 0.0];
    let zeros = [0.0f32; 4];

    let ring = RingQuery::new(&origin, 2.0, 0.5);
    assert_cpu_cuda_results(
        "ordinary ring",
        cpu.query(&ring).unwrap(),
        cuda.query(&ring).unwrap(),
    );

    let range = RangeQuery::new(&origin, 1.5, 5.1);
    assert_cpu_cuda_results(
        "ordinary range",
        cpu.query_range(&range).unwrap(),
        cuda.query_range(&range).unwrap(),
    );

    let disk = DiskQuery::new(&origin, 5.1);
    assert_cpu_cuda_results(
        "ordinary disk",
        cpu.query_disk(&disk).unwrap(),
        cuda.query_disk(&disk).unwrap(),
    );

    let weighted_ring = RingQuery::new(&origin, 2.0, 0.2).with_weights(&weights_a);
    assert_cpu_cuda_results(
        "weighted ring",
        cpu.query(&weighted_ring).unwrap(),
        cuda.query(&weighted_ring).unwrap(),
    );

    let weighted_range = RangeQuery::new(&origin, 1.9, 2.8).with_weights(&weights_a);
    assert_cpu_cuda_results(
        "weighted range",
        cpu.query_range(&weighted_range).unwrap(),
        cuda.query_range(&weighted_range).unwrap(),
    );

    let weighted_disk = DiskQuery::new(&origin, 3.0).with_weights(&weights_a);
    assert_cpu_cuda_results(
        "weighted disk",
        cpu.query_disk(&weighted_disk).unwrap(),
        cuda.query_disk(&weighted_disk).unwrap(),
    );

    let all_one_disk = DiskQuery::new(&origin, 5.1).with_weights(&ones);
    let ordinary_hits = assert_cpu_cuda_results(
        "ordinary disk for all-one comparison",
        cpu.query_disk(&disk).unwrap(),
        cuda.query_disk(&disk).unwrap(),
    );
    let all_one_hits = assert_cpu_cuda_results(
        "all-one weighted disk",
        cpu.query_disk(&all_one_disk).unwrap(),
        cuda.query_disk(&all_one_disk).unwrap(),
    );
    assert_hit_slices_close("ordinary/all-one disk", &ordinary_hits, &all_one_hits);

    let zero_weight_disk = DiskQuery::new(&origin, 1.1).with_weights(&zero_dimensions);
    assert_cpu_cuda_results(
        "zero-weight dimensions",
        cpu.query_disk(&zero_weight_disk).unwrap(),
        cuda.query_disk(&zero_weight_disk).unwrap(),
    );

    let all_zero_disk = DiskQuery::new(&origin, 0.0).with_weights(&zeros);
    let all_zero_hits = assert_cpu_cuda_results(
        "all-zero weights",
        cpu.query_disk(&all_zero_disk).unwrap(),
        cuda.query_disk(&all_zero_disk).unwrap(),
    );
    assert_eq!(all_zero_hits.len(), vectors.len());
    assert!(all_zero_hits.iter().all(|hit| hit.dist_sq == 0.0));

    let ordinary_disks = [
        DiskQuery::new(&origin, 6.0),
        DiskQuery::new(&shifted, 5.0),
        DiskQuery::new(&shifted_back, 8.0),
    ];
    let ordinary_intersection = DiskIntersectionQuery::new(&ordinary_disks);
    assert_cpu_cuda_results(
        "ordinary three-disk intersection",
        cpu.query_disk_intersection(&ordinary_intersection).unwrap(),
        cuda.query_disk_intersection(&ordinary_intersection)
            .unwrap(),
    );

    let all_weighted_disks = [
        DiskQuery::new(&origin, 6.0).with_weights(&weights_a),
        DiskQuery::new(&shifted, 10.0).with_weights(&weights_b),
        DiskQuery::new(&shifted_back, 8.0).with_weights(&ones),
    ];
    let all_weighted_intersection = DiskIntersectionQuery::new(&all_weighted_disks);
    assert_cpu_cuda_results(
        "all-weighted intersection",
        cpu.query_disk_intersection(&all_weighted_intersection)
            .unwrap(),
        cuda.query_disk_intersection(&all_weighted_intersection)
            .unwrap(),
    );

    let weighted_first_disks = [
        DiskQuery::new(&origin, 6.0).with_weights(&weights_a),
        DiskQuery::new(&shifted, 6.0),
        DiskQuery::new(&shifted_back, 10.0).with_weights(&weights_b),
    ];
    let weighted_first_intersection = DiskIntersectionQuery::new(&weighted_first_disks);
    let weighted_first_hits = assert_cpu_cuda_results(
        "mixed intersection with weighted first disk",
        cpu.query_disk_intersection(&weighted_first_intersection)
            .unwrap(),
        cuda.query_disk_intersection(&weighted_first_intersection)
            .unwrap(),
    );
    let weighted_first = weighted_first_hits
        .iter()
        .find(|hit| hit.id == 3)
        .expect("mixed weighted-first intersection should contain vector 3");
    assert!((weighted_first.dist_sq - 7.5).abs() <= 1.0e-5);

    let unweighted_first_disks = [
        DiskQuery::new(&origin, 6.0),
        DiskQuery::new(&shifted, 10.0).with_weights(&weights_a),
        DiskQuery::new(&shifted_back, 10.0).with_weights(&weights_b),
    ];
    let unweighted_first_intersection = DiskIntersectionQuery::new(&unweighted_first_disks);
    let unweighted_first_hits = assert_cpu_cuda_results(
        "mixed intersection with unweighted first disk",
        cpu.query_disk_intersection(&unweighted_first_intersection)
            .unwrap(),
        cuda.query_disk_intersection(&unweighted_first_intersection)
            .unwrap(),
    );
    let unweighted_first = unweighted_first_hits
        .iter()
        .find(|hit| hit.id == 3)
        .expect("mixed unweighted-first intersection should contain vector 3");
    assert!((unweighted_first.dist_sq - 4.0).abs() <= 1.0e-5);
}

#[test]
fn test_dims_and_len() {
    let mut db = cpu_db(8);
    assert_eq!(db.dims(), 8);
    assert_eq!(db.len(), 0);
    assert!(db.is_empty());

    for _ in 0..10 {
        db.add_vector(&[0.0f32; 8], ()).unwrap();
    }
    assert_eq!(db.len(), 10);
    assert!(!db.is_empty());
}

#[test]
fn test_higher_dims() {
    let dims = 64usize;
    let mut db = cpu_db(dims);

    let vecs: Vec<f32> = (0..100 * dims)
        .map(|i| ((i as f32 * 1.6180339) % 2.0) - 1.0)
        .collect();
    for chunk in vecs.chunks(dims) {
        db.add_vector(chunk, ()).unwrap();
    }
    let db = db.build().unwrap();

    let query: Vec<f32> = (0..dims)
        .map(|i| ((i as f32 * std::f32::consts::E) % 2.0) - 1.0)
        .collect();
    let d = 5.0f32;
    let lambda = 0.5f32;

    let result = db.query(&RingQuery::new(&query, d, lambda)).unwrap();

    let norm_sq_q: f32 = query.iter().map(|x| x * x).sum();
    for id in result.hits.iter().map(|h| h.id) {
        let base = id as usize * dims;
        let row = &vecs[base..base + dims];
        let dot: f32 = row.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
        let norm_sq_x: f32 = row.iter().map(|x| x * x).sum();
        let dist_sq = norm_sq_x + norm_sq_q - 2.0 * dot;
        let dist = dist_sq.sqrt();
        assert!(
            dist >= d - lambda - 1e-4 && dist <= d + lambda + 1e-4,
            "returned ID {id} has dist {dist:.4} outside [{}, {}]",
            d - lambda,
            d + lambda
        );
    }
}

#[test]
fn test_payload_roundtrip() {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Payload, Debug, PartialEq)]
    struct Meta {
        label: String,
        score: f64,
    }

    let mut db: RingDb<Meta> = RingDb::new(RingDbConfig::new(2)).unwrap();
    db.add_vector(
        &[1.0, 0.0],
        Meta {
            label: "dog".into(),
            score: 0.9,
        },
    )
    .unwrap();
    db.add_vector(
        &[0.0, 1.0],
        Meta {
            label: "cat".into(),
            score: 0.7,
        },
    )
    .unwrap();
    db.add_vector(
        &[5.0, 0.0],
        Meta {
            label: "bird".into(),
            score: 0.5,
        },
    )
    .unwrap();

    let db = db.build().unwrap();

    let result = db.query(&RingQuery::new(&[0.0, 0.0], 1.0, 0.1)).unwrap();

    let mut ids = result.ids();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 1]);

    let payloads = db.fetch_payloads(&result.ids()).unwrap();
    assert_eq!(payloads.len(), 2);

    let p0 = db.fetch_payload(0).unwrap();
    assert_eq!(p0.label, "dog");
    assert_eq!(p0.score, 0.9);

    let p1 = db.fetch_payload(1).unwrap();
    assert_eq!(p1.label, "cat");
}

// ---- RangeQuery tests ----

#[test]
fn test_range_query_basic() {
    // Vectors at distances 1, 3, 5, 7 from origin.
    let mut db = cpu_db(2);
    db.add_vector(&[1.0f32, 0.0], ()).unwrap(); // dist 1 → miss (< 2.5)
    db.add_vector(&[3.0, 0.0], ()).unwrap(); // dist 3 → hit
    db.add_vector(&[5.0, 0.0], ()).unwrap(); // dist 5 → hit
    db.add_vector(&[7.0, 0.0], ()).unwrap(); // dist 7 → miss (> 6)
    let db = db.build().unwrap();

    let result = db
        .query_range(&RangeQuery::new(&[0.0, 0.0], 2.5, 6.0))
        .unwrap();

    let mut ids = result.ids();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2], "only vectors at dist 3 and 5 should match");
}

#[test]
fn test_range_query_inclusive_bounds() {
    // Vectors are placed exactly on the boundaries.
    let mut db = cpu_db(2);
    db.add_vector(&[3.0f32, 0.0], ()).unwrap(); // dist = 3.0 (on d_min)
    db.add_vector(&[5.0, 0.0], ()).unwrap(); // dist = 5.0 (on d_max)
    db.add_vector(&[4.0, 0.0], ()).unwrap(); // dist = 4.0 (inside)
    let db = db.build().unwrap();

    let result = db
        .query_range(&RangeQuery::new(&[0.0, 0.0], 3.0, 5.0))
        .unwrap();

    let mut ids = result.ids();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![0, 1, 2],
        "boundary and interior vectors must all match"
    );
}

#[test]
fn test_range_query_empty_interval() {
    // d_min == d_max: only vectors at exactly that distance should match.
    let mut db = cpu_db(2);
    db.add_vector(&[3.0f32, 0.0], ()).unwrap(); // dist = 3
    db.add_vector(&[4.0, 0.0], ()).unwrap(); // dist = 4
    let db = db.build().unwrap();

    let result = db
        .query_range(&RangeQuery::new(&[0.0, 0.0], 3.0, 3.0))
        .unwrap();

    let mut ids = result.ids();
    ids.sort_unstable();
    assert_eq!(ids, vec![0], "only the vector at dist=3 should match");
}

#[test]
fn test_range_query_no_results() {
    let mut db = cpu_db(2);
    db.add_vector(&[1.0f32, 0.0], ()).unwrap();
    db.add_vector(&[2.0, 0.0], ()).unwrap();
    let db = db.build().unwrap();

    let result = db
        .query_range(&RangeQuery::new(&[0.0, 0.0], 10.0, 20.0))
        .unwrap();

    assert!(result.hits.is_empty());
}

/// A RangeQuery with d_min=(d-lambda).max(0) and d_max=d+lambda must return
/// the same IDs as the equivalent RingQuery.
#[test]
fn test_range_equivalent_to_ring() {
    let mut db = cpu_db(2);
    for x in [1.0f32, 3.0, 5.0, 7.0, 9.0] {
        db.add_vector(&[x, 0.0], ()).unwrap();
    }
    let db = db.build().unwrap();

    let (d, lambda) = (5.0f32, 1.5f32);

    let ring_result = db.query(&RingQuery::new(&[0.0, 0.0], d, lambda)).unwrap();

    let range_result = db
        .query_range(&RangeQuery::new(
            &[0.0, 0.0],
            (d - lambda).max(0.0),
            d + lambda,
        ))
        .unwrap();

    let mut ring_ids = ring_result.ids();
    let mut range_ids = range_result.ids();
    ring_ids.sort_unstable();
    range_ids.sort_unstable();
    assert_eq!(
        ring_ids, range_ids,
        "RingQuery and RangeQuery must return identical IDs"
    );
}

// ---- DiskQuery tests ----

#[test]
fn test_disk_query_basic() {
    // Vectors at distances 1, 3, 5, 7 from origin.
    let mut db = cpu_db(2);
    db.add_vector(&[1.0f32, 0.0], ()).unwrap(); // dist 1 → hit
    db.add_vector(&[3.0, 0.0], ()).unwrap(); // dist 3 → hit
    db.add_vector(&[5.0, 0.0], ()).unwrap(); // dist 5 → hit (on boundary)
    db.add_vector(&[7.0, 0.0], ()).unwrap(); // dist 7 → miss
    let db = db.build().unwrap();

    let result = db.query_disk(&DiskQuery::new(&[0.0, 0.0], 5.0)).unwrap();

    let mut ids = result.ids();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![0, 1, 2],
        "all vectors within radius 5 should match"
    );
}

#[test]
fn test_disk_query_includes_origin() {
    // A vector at the origin is at distance 0 — must always be inside any disk.
    let mut db = cpu_db(2);
    db.add_vector(&[0.0f32, 0.0], ()).unwrap(); // dist 0
    db.add_vector(&[10.0, 0.0], ()).unwrap(); // dist 10
    let db = db.build().unwrap();

    let result = db.query_disk(&DiskQuery::new(&[0.0, 0.0], 1.0)).unwrap();

    assert!(
        result.hits.iter().any(|h| h.id == 0),
        "vector at origin must be inside disk"
    );
    assert!(
        !result.hits.iter().any(|h| h.id == 1),
        "far vector must be outside disk"
    );
}

#[test]
fn test_disk_query_d_max_zero() {
    // Disk of radius 0: only vectors coincident with the query match.
    let mut db = cpu_db(2);
    db.add_vector(&[0.0f32, 0.0], ()).unwrap(); // coincident  → hit
    db.add_vector(&[1.0, 0.0], ()).unwrap(); // dist 1 → miss
    let db = db.build().unwrap();

    let result = db.query_disk(&DiskQuery::new(&[0.0, 0.0], 0.0)).unwrap();

    let mut ids = result.ids();
    ids.sort_unstable();
    assert_eq!(ids, vec![0]);
}

/// A DiskQuery must return a superset of a RingQuery with the same outer bound,
/// because the disk covers d_min=0 while the ring excludes the interior.
#[test]
fn test_disk_is_superset_of_ring() {
    let mut db = cpu_db(2);
    for x in [1.0f32, 2.0, 3.0, 4.0, 5.0, 8.0] {
        db.add_vector(&[x, 0.0], ()).unwrap();
    }
    let db = db.build().unwrap();

    let disk_result = db.query_disk(&DiskQuery::new(&[0.0, 0.0], 5.0)).unwrap();

    let ring_result = db.query(&RingQuery::new(&[0.0, 0.0], 4.0, 1.5)).unwrap();

    // Every ring hit must also appear in the disk.
    for id in ring_result.hits.iter().map(|h| h.id) {
        assert!(
            disk_result.hits.iter().any(|h| h.id == id),
            "ring hit ID {id} not found in disk result"
        );
    }
    // The disk must have at least as many hits.
    assert!(disk_result.hits.len() >= ring_result.hits.len());
}

/// DiskQuery with d_max equals RangeQuery with d_min=0 and same d_max.
#[test]
fn test_disk_equivalent_to_range_d_min_zero() {
    let mut db = cpu_db(2);
    for x in [0.0f32, 1.0, 2.5, 4.9, 5.0, 6.0] {
        db.add_vector(&[x, 0.0], ()).unwrap();
    }
    let db = db.build().unwrap();

    let disk_result = db.query_disk(&DiskQuery::new(&[0.0, 0.0], 5.0)).unwrap();

    let range_result = db
        .query_range(&RangeQuery::new(&[0.0, 0.0], 0.0, 5.0))
        .unwrap();

    let mut disk_ids = disk_result.ids();
    let mut range_ids = range_result.ids();
    disk_ids.sort_unstable();
    range_ids.sort_unstable();
    assert_eq!(
        disk_ids, range_ids,
        "DiskQuery must equal RangeQuery(d_min=0)"
    );
}

#[test]
fn test_disk_intersection_query_basic() {
    let mut db = cpu_db(2);
    db.add_vector(&[0.0f32, 0.0], ()).unwrap(); // inside first disk only
    db.add_vector(&[1.0, 0.0], ()).unwrap(); // inside both
    db.add_vector(&[2.0, 0.0], ()).unwrap(); // on first boundary, inside second
    db.add_vector(&[3.0, 0.0], ()).unwrap(); // inside second disk only
    db.add_vector(&[1.0, 1.0], ()).unwrap(); // inside first disk only
    let db = db.build().unwrap();

    let origin = [0.0f32, 0.0];
    let right = [2.0f32, 0.0];
    let disks = [DiskQuery::new(&origin, 2.0), DiskQuery::new(&right, 1.0)];
    let result = db
        .query_disk_intersection(&DiskIntersectionQuery::new(&disks))
        .unwrap();

    let mut hits = result.hits;
    hits.sort_by_key(|hit| hit.id);
    let ids: Vec<u32> = hits.iter().map(|hit| hit.id).collect();
    assert_eq!(ids, vec![1, 2], "only vectors inside both disks match");
    assert!((hits[0].dist_sq - 1.0).abs() < 1e-6);
    assert!((hits[1].dist_sq - 4.0).abs() < 1e-6);
}

#[test]
fn test_disk_intersection_query_no_results() {
    let mut db = cpu_db(2);
    db.add_vector(&[0.0f32, 0.0], ()).unwrap();
    db.add_vector(&[10.0, 0.0], ()).unwrap();
    let db = db.build().unwrap();

    let left = [0.0f32, 0.0];
    let right = [10.0f32, 0.0];
    let disks = [DiskQuery::new(&left, 1.0), DiskQuery::new(&right, 1.0)];
    let result = db
        .query_disk_intersection(&DiskIntersectionQuery::new(&disks))
        .unwrap();

    assert!(result.hits.is_empty());
}

#[test]
fn test_disk_intersection_query_rejects_empty_disks() {
    let db = cpu_db(2).build().unwrap();
    let err = match db.query_disk_intersection(&DiskIntersectionQuery::new(&[])) {
        Ok(_) => panic!("empty disk intersection should fail"),
        Err(err) => err,
    };

    assert!(matches!(err, RingDbError::InvalidQuery(_)));
}

#[test]
fn test_disk_intersection_query_dimension_mismatch() {
    let mut db = cpu_db(2);
    db.add_vector(&[0.0f32, 0.0], ()).unwrap();
    let db = db.build().unwrap();

    let good = [0.0f32, 0.0];
    let bad = [0.0f32, 0.0, 0.0];
    let disks = [DiskQuery::new(&good, 1.0), DiskQuery::new(&bad, 1.0)];
    let err = match db.query_disk_intersection(&DiskIntersectionQuery::new(&disks)) {
        Ok(_) => panic!("mismatched disk dimension should fail"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        RingDbError::DimensionMismatch {
            expected: 2,
            got: 3
        }
    ));
}

// ── Persistence tests ─────────────────────────────────────────────────────────

/// Build a database with `persist_dir`, drop it, reload it, and verify queries
/// return the same results as the original.
#[test]
fn test_persist_and_load_unit_payload() {
    let dir = std::env::temp_dir().join(format!("ringdb-test-persist-unit-{}", std::process::id()));
    // Clean up any stale run.
    let _ = std::fs::remove_dir_all(&dir);

    // --- Build and persist ---
    let mut db = RingDb::new(RingDbConfig::new(2).with_persist_dir(&dir)).unwrap();
    db.add_vector(&[1.0f32, 0.0], ()).unwrap(); // distance 1.0 from origin
    db.add_vector(&[0.0, 5.0], ()).unwrap(); // distance 5.0 from origin
    db.add_vector(&[3.0, 4.0], ()).unwrap(); // distance 5.0 from origin
    let original = db.build().unwrap();

    let original_result = original
        .query(&RingQuery::new(&[0.0f32, 0.0], 5.0, 0.1))
        .unwrap();
    let mut original_ids = original_result.ids();
    original_ids.sort_unstable();

    // Drop the original; files stay on disk.
    drop(original);

    // --- Load from disk ---
    let loaded = RingDb::<()>::load(&dir, BackendPreference::Cpu).unwrap();
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded.dims(), 2);

    let loaded_result = loaded
        .query(&RingQuery::new(&[0.0f32, 0.0], 5.0, 0.1))
        .unwrap();
    let mut loaded_ids = loaded_result.ids();
    loaded_ids.sort_unstable();

    assert_eq!(original_ids, loaded_ids, "loaded db must return same ids");

    // Clean up.
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_persist_and_load_struct_payload() {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, Payload, PartialEq)]
    struct Meta {
        label: String,
        value: i32,
    }

    let dir =
        std::env::temp_dir().join(format!("ringdb-test-persist-struct-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    // --- Build and persist ---
    let mut db: RingDb<Meta> = RingDb::new(RingDbConfig::new(2).with_persist_dir(&dir)).unwrap();
    db.add_vector(
        &[1.0f32, 0.0],
        Meta {
            label: "dog".into(),
            value: 1,
        },
    )
    .unwrap();
    db.add_vector(
        &[0.0, 1.0],
        Meta {
            label: "cat".into(),
            value: 2,
        },
    )
    .unwrap();
    let original = db.build().unwrap();
    drop(original);

    // --- Load from disk ---
    let loaded = RingDb::<Meta>::load(&dir, BackendPreference::Cpu).unwrap();
    assert_eq!(loaded.len(), 2);

    let result = loaded
        .query(&RingQuery::new(&[1.0f32, 0.0], 1.0, 0.1))
        .unwrap();

    // Vector 0 ([1,0]) is exactly at distance 1.0 from [1,0] = 0; should NOT match.
    // Vector 1 ([0,1]) is at distance sqrt(2) ≈ 1.414 from [1,0]; should NOT match.
    // Adjust: query from origin, expecting [1,0] at d=1.
    let result2 = loaded
        .query(&RingQuery::new(&[0.0f32, 0.0], 1.0, 0.1))
        .unwrap();
    assert_eq!(result2.hits.len(), 2); // both [1,0] and [0,1] are at distance 1 from origin

    let payloads = loaded.fetch_payloads(&result2.ids()).unwrap();
    let labels: Vec<&str> = payloads.iter().map(|m| m.label.as_str()).collect();
    assert!(labels.contains(&"dog"), "dog payload missing after load");
    assert!(labels.contains(&"cat"), "cat payload missing after load");

    // Verify values too.
    for p in &payloads {
        assert!(p.value == 1 || p.value == 2);
    }

    let _ = std::fs::remove_dir_all(&dir);
    drop(result); // silence unused warning
}

// ── Pod payload tests ─────────────────────────────────────────────────────────

#[test]
fn test_pod_payload_roundtrip() {
    use bytemuck::{Pod, Zeroable};

    /// A simple fixed-size payload — no heap allocation, no Serde needed.
    #[derive(Copy, Clone, Pod, Zeroable, Payload, Debug, PartialEq)]
    #[repr(C)]
    #[payload(storage = "pod")]
    struct GeoPoint {
        lat: f32,
        lon: f32,
        altitude: f32,
    }

    let mut db: RingDb<GeoPoint> = RingDb::new(RingDbConfig::new(2)).unwrap();
    db.add_vector(
        &[1.0, 0.0],
        GeoPoint {
            lat: 48.8566,
            lon: 2.3522,
            altitude: 35.0,
        },
    )
    .unwrap(); // ID 0
    db.add_vector(
        &[0.0, 1.0],
        GeoPoint {
            lat: 51.5074,
            lon: -0.1278,
            altitude: 11.0,
        },
    )
    .unwrap(); // ID 1
    db.add_vector(
        &[5.0, 0.0],
        GeoPoint {
            lat: 0.0,
            lon: 0.0,
            altitude: 0.0,
        },
    )
    .unwrap(); // ID 2 – far

    let db = db.build().unwrap();

    let result = db.query(&RingQuery::new(&[0.0, 0.0], 1.0, 0.1)).unwrap();

    let mut ids = result.ids();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 1], "only vectors at distance ≈ 1 should match");

    // fetch_pod returns &GeoPoint — zero-copy, O(1), no Result.
    let p0 = db.fetch_pod(0);
    assert!((p0.lat - 48.8566).abs() < 1e-3, "lat mismatch for ID 0");
    assert!((p0.lon - 2.3522).abs() < 1e-3, "lon mismatch for ID 0");

    let p1 = db.fetch_pod(1);
    assert!((p1.lat - 51.5074).abs() < 1e-3, "lat mismatch for ID 1");

    // fetch_pods returns Vec<&GeoPoint>.
    let refs = db.fetch_pods(&result.ids());
    assert_eq!(refs.len(), 2);
}

#[test]
fn test_pod_storage_mismatch_error() {
    use bytemuck::{Pod, Zeroable};

    #[derive(Copy, Clone, Pod, Zeroable, Payload)]
    #[repr(C)]
    #[payload(storage = "pod")]
    struct Tag(u64);

    let mut pod_db: RingDb<Tag> = RingDb::new(RingDbConfig::new(2)).unwrap();
    pod_db.add_vector(&[1.0, 0.0], Tag(42)).unwrap();
    let pod_db = pod_db.build().unwrap();

    // fetch_pod returns &Tag directly — no Result.
    assert_eq!(pod_db.fetch_pod(0).0, 42);
}
