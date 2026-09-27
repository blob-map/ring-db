/// Correctness tests for the CPU exact (float32) backend.
///
/// These tests use hand-crafted vectors at known distances from a query so
/// the expected result sets are computed analytically, not by another code path.
use bytemuck::{Pod, Zeroable};
use ringdb::{
    BackendPreference, DiskIntersectionQuery, DiskQuery, KnnQuery, RangeQuery, RingDb,
    RingDbConfig, RingDbError, RingQuery,
};

fn cpu_db(dims: usize) -> RingDb {
    RingDb::new(RingDbConfig::new(dims)).unwrap()
}

fn assert_invalid_query<T>(result: Result<T, RingDbError>) {
    assert!(matches!(result, Err(RingDbError::InvalidQuery(_))));
}

// ---- 2-D helpers ----

/// Distance between two 2-D points.
fn dist2(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
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
    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable, Debug, PartialEq)]
    struct Meta {
        label: [u8; 8],
        score: f64,
    }

    let mut db: RingDb<Meta> = RingDb::new(RingDbConfig::new(2)).unwrap();
    db.add_vector(
        &[1.0, 0.0],
        Meta {
            label: *b"dog\0\0\0\0\0",
            score: 0.9,
        },
    )
    .unwrap();
    db.add_vector(
        &[0.0, 1.0],
        Meta {
            label: *b"cat\0\0\0\0\0",
            score: 0.7,
        },
    )
    .unwrap();
    db.add_vector(
        &[5.0, 0.0],
        Meta {
            label: *b"bird\0\0\0\0",
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
    assert_eq!(&p0.label[..3], b"dog");
    assert_eq!(p0.score, 0.9);

    let p1 = db.fetch_payload(1).unwrap();
    assert_eq!(&p1.label[..3], b"cat");
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

#[test]
fn test_non_finite_coordinates_and_invalid_bounds_are_rejected() {
    assert_invalid_query(RingDb::<()>::new(RingDbConfig::new(0)));

    let mut builder = cpu_db(2);
    assert_invalid_query(builder.add_vector(&[f32::NAN, 0.0], ()));
    builder.add_vector(&[1.0, 0.0], ()).unwrap();
    let db = builder.build().unwrap();
    let origin = [0.0, 0.0];

    assert_invalid_query(db.query(&RingQuery::new(&origin, -1.0, 0.5)));
    assert_invalid_query(db.query(&RingQuery::new(&origin, 1.0, f32::NAN)));
    assert_invalid_query(db.query_range(&RangeQuery::new(&origin, 2.0, 1.0)));
    assert_invalid_query(db.query_range(&RangeQuery::new(&origin, 0.0, f32::INFINITY)));
    assert_invalid_query(db.query_disk(&DiskQuery::new(&origin, -1.0)));
    assert_invalid_query(db.query_knn(&KnnQuery::new(&[f32::INFINITY, 0.0], 1)));

    let disks = [DiskQuery::new(&origin, -1.0)];
    assert_invalid_query(db.query_disk_intersection(&DiskIntersectionQuery::new(&disks)));
}

#[test]
fn test_unweighted_queries_use_stable_direct_differences() {
    let mut builder = cpu_db(1);
    builder.add_vector(&[f32::MAX], ()).unwrap();
    builder.add_vector(&[-f32::MAX], ()).unwrap();
    let db = builder.build().unwrap();
    let query = [f32::MAX];
    let first = db.candidates_from_ids([0]).unwrap();

    assert_eq!(
        db.query(&RingQuery::new(&query, 0.0, 0.0)).unwrap().ids(),
        [0]
    );
    assert_eq!(
        db.query_range(&RangeQuery::new(&query, 0.0, 0.0).with_candidates(&first))
            .unwrap()
            .ids(),
        [0]
    );
    assert_eq!(
        db.query_disk(&DiskQuery::new(&query, 0.0)).unwrap().ids(),
        [0]
    );

    let disks = [DiskQuery::new(&query, 0.0)];
    assert_eq!(
        db.query_disk_intersection(&DiskIntersectionQuery::new(&disks).with_candidates(&first))
            .unwrap()
            .ids(),
        [0]
    );
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
    assert_eq!(*loaded.fetch_payload(0).unwrap(), ());
    assert!(matches!(
        loaded.fetch_payload(3),
        Err(RingDbError::InvalidVectorId { id: 3, len: 3 })
    ));

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
fn test_persistence_refuses_to_overwrite_a_nonempty_directory() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("keep-me");
    std::fs::write(&marker, b"user data").unwrap();

    let mut db = RingDb::new(RingDbConfig::new(1).with_persist_dir(dir.path())).unwrap();
    db.add_vector(&[0.0], ()).unwrap();
    let error = match db.build() {
        Ok(_) => panic!("a nonempty persistence directory should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        RingDbError::Io(ref error) if error.kind() == std::io::ErrorKind::AlreadyExists
    ));
    assert_eq!(std::fs::read(marker).unwrap(), b"user data");
}

#[test]
fn test_load_rejects_non_finite_persisted_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = RingDb::new(RingDbConfig::new(1).with_persist_dir(dir.path())).unwrap();
    db.add_vector(&[1.0], ()).unwrap();
    db.build().unwrap();

    std::fs::write(dir.path().join("vectors.bin"), f32::NAN.to_le_bytes()).unwrap();
    assert!(matches!(
        RingDb::<()>::load(dir.path(), BackendPreference::Cpu),
        Err(RingDbError::Corrupt(_))
    ));
}

#[test]
fn test_load_rejects_row_counts_above_the_u32_id_limit() {
    let dir = tempfile::tempdir().unwrap();
    RingDb::<()>::new(RingDbConfig::new(1).with_persist_dir(dir.path()))
        .unwrap()
        .build()
        .unwrap();

    let meta_path = dir.path().join("meta.bin");
    let mut meta = std::fs::read(&meta_path).unwrap();
    meta[24..32].copy_from_slice(&(u64::from(u32::MAX) + 1).to_le_bytes());
    std::fs::write(meta_path, meta).unwrap();

    assert!(matches!(
        RingDb::<()>::load(dir.path(), BackendPreference::Cpu),
        Err(RingDbError::Corrupt(_))
    ));
}

#[test]
fn test_persist_and_load_struct_payload() {
    #[repr(C)]
    #[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq)]
    struct Meta {
        label: [u8; 8],
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
            label: *b"dog\0\0\0\0\0",
            value: 1,
        },
    )
    .unwrap();
    db.add_vector(
        &[0.0, 1.0],
        Meta {
            label: *b"cat\0\0\0\0\0",
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

    // The distances are 0 and sqrt(2), outside the ring [0.9, 1.1].
    assert!(result.hits.is_empty());

    // Both vectors are at distance 1 from the origin.
    let result2 = loaded
        .query(&RingQuery::new(&[0.0f32, 0.0], 1.0, 0.1))
        .unwrap();
    assert_eq!(result2.hits.len(), 2); // both [1,0] and [0,1] are at distance 1 from origin

    let payloads = loaded.fetch_payloads(&result2.ids()).unwrap();
    let labels: Vec<&[u8]> = payloads.iter().map(|m| &m.label[..3]).collect();
    assert!(
        labels.contains(&b"dog".as_slice()),
        "dog payload missing after load"
    );
    assert!(
        labels.contains(&b"cat".as_slice()),
        "cat payload missing after load"
    );

    // Verify values too.
    for p in &payloads {
        assert!(p.value == 1 || p.value == 2);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// ── Pod payload tests ─────────────────────────────────────────────────────────

#[test]
fn test_pod_payload_roundtrip() {
    /// A fixed-size payload with no heap allocation.
    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable, Debug, PartialEq)]
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

    // Payload fetches return zero-copy references into the Pod store.
    let p0 = db.fetch_payload(0).unwrap();
    assert!((p0.lat - 48.8566).abs() < 1e-3, "lat mismatch for ID 0");
    assert!((p0.lon - 2.3522).abs() < 1e-3, "lon mismatch for ID 0");

    let p1 = db.fetch_payload(1).unwrap();
    assert!((p1.lat - 51.5074).abs() < 1e-3, "lat mismatch for ID 1");

    let refs = db.fetch_payloads(&result.ids()).unwrap();
    assert_eq!(refs.len(), 2);
}

#[test]
fn test_pod_newtype_payload() {
    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct Tag(u64);

    let mut pod_db: RingDb<Tag> = RingDb::new(RingDbConfig::new(2)).unwrap();
    pod_db.add_vector(&[1.0, 0.0], Tag(42)).unwrap();
    let pod_db = pod_db.build().unwrap();

    assert_eq!(pod_db.fetch_payload(0).unwrap().0, 42);
}
