use rand::{RngExt, SeedableRng};
use ringdb::{
    BackendPreference, CandidateSet, DiskIntersectionQuery, DiskQuery, KnnQuery, MetadataKey,
    MetadataTag, MetadataValue, QueryResult, RangeQuery, RingDb, RingDbConfig, RingDbError,
    RingQuery, SealedRingDb,
};

fn requested() -> bool {
    std::env::var("RINGDB_VULKAN_TEST").as_deref() == Ok("1")
}

#[test]
fn cpu_works_and_vulkan_reports_unavailable_without_a_driver() {
    // Use a child process so loader environment overrides cannot race the
    // device tests or change a loader that is already initialized.
    const CHILD: &str = "RINGDB_VULKAN_MISSING_DRIVER_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let result = RingDb::<()>::new(
            RingDbConfig::new(1).with_backend_preference(BackendPreference::Vulkan),
        );
        assert!(matches!(result, Err(RingDbError::BackendUnavailable(_))));
        let mut cpu: RingDb = RingDb::new(RingDbConfig::new(1)).unwrap();
        cpu.add_vector(&[1.0], ()).unwrap();
        cpu.add_vector(&[3.0], ()).unwrap();
        let cpu = cpu.build().unwrap();
        assert_eq!(cpu.backend_name(), "cpu");
        let hits = cpu.query_knn(&KnnQuery::new(&[0.0], 1)).unwrap();
        assert_eq!(hits.ids(), vec![0]);
        assert_eq!(hits.hits[0].dist_sq, 1.0);
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing-driver.json");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "cpu_works_and_vulkan_reports_unavailable_without_a_driver",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("VK_DRIVER_FILES", &missing)
        .env("VK_ICD_FILENAMES", &missing)
        .env_remove("VK_ADD_DRIVER_FILES")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn build(vectors: &[Vec<f32>], backend: BackendPreference) -> SealedRingDb<u32> {
    let mut db =
        RingDb::new(RingDbConfig::new(vectors[0].len()).with_backend_preference(backend)).unwrap();
    for (id, vector) in vectors.iter().enumerate() {
        db.add_vector_with_metadata(
            vector,
            id as u32,
            &[MetadataTag::new(
                MetadataKey::new(1),
                MetadataValue::new((id % 3) as u64),
            )],
        )
        .unwrap();
    }
    db.build().unwrap()
}

fn same_hits(mut cpu: QueryResult, mut gpu: QueryResult, ordered: bool) {
    assert_eq!(gpu.backend_used, "vulkan");
    if !ordered {
        cpu.hits.sort_unstable_by_key(|h| h.id);
        gpu.hits.sort_unstable_by_key(|h| h.id);
    }
    assert_eq!(cpu.ids(), gpu.ids());
    for (a, b) in cpu.hits.iter().zip(&gpu.hits) {
        if a.dist_sq == b.dist_sq {
            continue;
        }
        assert!(
            (a.dist_sq - b.dist_sq).abs() <= 2e-5 * a.dist_sq.abs().max(1.0),
            "id {}: CPU {}, Vulkan {}",
            a.id,
            a.dist_sq,
            b.dist_sq
        );
    }
}

fn candidates(db: &SealedRingDb<u32>) -> Vec<CandidateSet> {
    vec![
        db.candidates_from_ids(0..db.len() as u32).unwrap(),
        db.candidates_from_ids([]).unwrap(),
        db.candidates_from_ids([0, 31, 64, db.len() as u32 - 1])
            .unwrap(),
        db.metadata_eq(MetadataKey::new(1), MetadataValue::new(1)),
        db.candidates_from_ids([0, 63]).unwrap().complement(),
    ]
}

#[test]
fn vulkan_matches_cpu_for_queries_weights_and_candidate_forms() {
    if !requested() {
        return;
    }
    let mut rng = rand::rngs::SmallRng::seed_from_u64(90419);
    for dims in [1, 3, 4, 7, 16, 64, 129] {
        // Partial workgroup and partial u64/u32 bitmap words.
        let vectors: Vec<Vec<f32>> = (0..269)
            .map(|_| (0..dims).map(|_| rng.random_range(-2.0f32..2.0)).collect())
            .collect();
        let cpu = build(&vectors, BackendPreference::Cpu);
        let gpu = build(&vectors, BackendPreference::Vulkan);
        assert_eq!(gpu.backend_name(), "vulkan");
        let origin = vec![0.0; dims];
        let shifted = vec![0.5; dims];
        let weights: Vec<_> = (0..dims).map(|i| [0.0, 0.25, 2.0, 1.0][i % 4]).collect();
        let ones = vec![1.0; dims];
        let radius = (dims as f32).sqrt() * 1.3;
        let sets = candidates(&gpu);
        for filter in std::iter::once(None).chain(sets.iter().map(Some)) {
            for metric in [None, Some(weights.as_slice()), Some(ones.as_slice())] {
                let mut ring = RingQuery::new(&origin, radius, radius * 0.4);
                let mut range = RangeQuery::new(&origin, radius * 0.6, radius * 1.4);
                let mut disk = DiskQuery::new(&origin, radius);
                if let Some(filter) = filter {
                    ring = ring.with_candidates(filter);
                    range = range.with_candidates(filter);
                    disk = disk.with_candidates(filter);
                }
                if let Some(metric) = metric {
                    ring = ring.with_weights(metric);
                    range = range.with_weights(metric);
                    disk = disk.with_weights(metric);
                }
                same_hits(cpu.query(&ring).unwrap(), gpu.query(&ring).unwrap(), false);
                same_hits(
                    cpu.query_range(&range).unwrap(),
                    gpu.query_range(&range).unwrap(),
                    false,
                );
                same_hits(
                    cpu.query_disk(&disk).unwrap(),
                    gpu.query_disk(&disk).unwrap(),
                    false,
                );
                for k in [0, 1, 3, 19, 400] {
                    let mut knn = KnnQuery::new(&origin, k);
                    if let Some(filter) = filter {
                        knn = knn.with_candidates(filter);
                    }
                    if let Some(metric) = metric {
                        knn = knn.with_weights(metric);
                    }
                    same_hits(
                        cpu.query_knn(&knn).unwrap(),
                        gpu.query_knn(&knn).unwrap(),
                        true,
                    );
                }
                let disks = [
                    disk,
                    DiskQuery::new(&shifted, radius * 1.2)
                        .with_weights(&ones)
                        .with_candidates(&sets[4]),
                ];
                let mut intersection = DiskIntersectionQuery::new(&disks);
                if let Some(filter) = filter {
                    intersection = intersection.with_candidates(filter);
                }
                same_hits(
                    cpu.query_disk_intersection(&intersection).unwrap(),
                    gpu.query_disk_intersection(&intersection).unwrap(),
                    false,
                );
            }
        }
    }
}

#[test]
fn vulkan_matches_cpu_for_three_disks_and_candidate_tails() {
    if !requested() {
        return;
    }
    for dims in [3, 64, 65] {
        let vectors: Vec<Vec<f32>> = (0..269)
            .map(|id| {
                (0..dims)
                    .map(|d| ((id * 7 + d * 3) % 17) as f32 / 4.0)
                    .collect()
            })
            .collect();
        let cpu = build(&vectors, BackendPreference::Cpu);
        let gpu = build(&vectors, BackendPreference::Vulkan);
        let a = vec![0.0; dims];
        let b = vec![0.5; dims];
        let c = vec![-0.5; dims];
        let weights = vec![0.25; dims];
        let sets = candidates(&gpu);
        for filter in std::iter::once(None).chain(sets.iter().map(Some)) {
            for radius in [0.0, 2.0, 3.0 * (dims as f32).sqrt(), 100.0] {
                for weighted in [false, true] {
                    let mut second = DiskQuery::new(&b, radius * 1.1);
                    if weighted {
                        second = second.with_weights(&weights);
                    }
                    let disks = [
                        DiskQuery::new(&a, radius),
                        second,
                        DiskQuery::new(&c, radius),
                    ];
                    let mut query = DiskIntersectionQuery::new(&disks);
                    if let Some(filter) = filter {
                        query = query.with_candidates(filter);
                    }
                    same_hits(
                        cpu.query_disk_intersection(&query).unwrap(),
                        gpu.query_disk_intersection(&query).unwrap(),
                        false,
                    );
                }
            }
        }
    }
}

#[test]
fn vulkan_knn_reduction_preserves_ties_infinity_and_zero_weights() {
    if !requested() {
        return;
    }
    // More than 128² rows exercises three reduction levels. Equal distances
    // across workgroups exercise ID tie-breaking and winner invalidation.
    let vectors: Vec<Vec<f32>> = (0..16_513).map(|id| vec![(id % 7) as f32]).collect();
    let cpu = build(&vectors, BackendPreference::Cpu);
    let gpu = build(&vectors, BackendPreference::Vulkan);
    let sparse = gpu
        .candidates_from_ids((0..gpu.len() as u32).step_by(33))
        .unwrap();
    let dense = gpu
        .candidates_from_ids((0..gpu.len() as u32).step_by(2))
        .unwrap();
    for filter in [None, Some(&sparse), Some(&dense)] {
        for k in [1, 5, 17, 300, 20_000] {
            let mut query = KnnQuery::new(&[0.0], k);
            if let Some(filter) = filter {
                query = query.with_candidates(filter);
            }
            same_hits(
                cpu.query_knn(&query).unwrap(),
                gpu.query_knn(&query).unwrap(),
                true,
            );
        }
    }
    let vectors = vec![vec![f32::MAX, 1.0]; 269];
    let cpu = build(&vectors, BackendPreference::Cpu);
    let gpu = build(&vectors, BackendPreference::Vulkan);
    let point = [-f32::MAX, 0.0];
    for weights in [None, Some([0.0, 1.0]), Some([1.0, 1.0])] {
        for k in [3, 269] {
            let mut query = KnnQuery::new(&point, k);
            if let Some(weights) = &weights {
                query = query.with_weights(weights);
            }
            same_hits(
                cpu.query_knn(&query).unwrap(),
                gpu.query_knn(&query).unwrap(),
                true,
            );
        }
        let mut disk = DiskQuery::new(&point, 1.0);
        if let Some(weights) = &weights {
            disk = disk.with_weights(weights);
        }
        same_hits(
            cpu.query_disk(&disk).unwrap(),
            gpu.query_disk(&disk).unwrap(),
            false,
        );
    }
}

#[test]
fn vulkan_handles_empty_databases_bounds_and_validation() {
    if !requested() {
        return;
    }
    let empty: SealedRingDb =
        RingDb::new(RingDbConfig::new(2).with_backend_preference(BackendPreference::Vulkan))
            .unwrap()
            .build()
            .unwrap();
    assert!(
        empty
            .query_knn(&KnnQuery::new(&[0.0, 0.0], 10))
            .unwrap()
            .hits
            .is_empty()
    );
    assert!(
        empty
            .query_disk(&DiskQuery::new(&[0.0, 0.0], 10.0))
            .unwrap()
            .hits
            .is_empty()
    );
    assert!(matches!(
        empty.query_knn(&KnnQuery::new(&[0.0], 0)),
        Err(RingDbError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        empty.query_knn(&KnnQuery::new(&[0.0, 0.0], 0).with_weights(&[-1.0, 1.0])),
        Err(RingDbError::InvalidQuery(_))
    ));
    assert!(matches!(
        empty.query_disk_intersection(&DiskIntersectionQuery::new(&[])),
        Err(RingDbError::InvalidQuery(_))
    ));
    let vectors = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![6.0, 8.0]];
    let gpu = build(&vectors, BackendPreference::Vulkan);
    let cpu = build(&vectors, BackendPreference::Cpu);
    assert_eq!(
        gpu.query_range(&RangeQuery::new(&[0.0, 0.0], 5.0, 10.0))
            .unwrap()
            .hits
            .len(),
        2
    );
    same_hits(
        cpu.query(&RingQuery::new(&[0.0, 0.0], 5.0, 0.0)).unwrap(),
        gpu.query(&RingQuery::new(&[0.0, 0.0], 5.0, 0.0)).unwrap(),
        false,
    );
    assert_eq!(
        gpu.query_disk(&DiskQuery::new(&[0.0, 0.0], 0.0))
            .unwrap()
            .ids(),
        [0]
    );
    let wrong = empty.candidates_from_ids([]).unwrap();
    assert!(matches!(
        gpu.query_knn(&KnnQuery::new(&[0.0, 0.0], 1).with_candidates(&wrong)),
        Err(RingDbError::CandidateSetMismatch { .. })
    ));

    // Nearby values at a large offset must use direct deltas, not norms.
    let vectors = vec![vec![1e10], vec![1e10 + 2048.0], vec![1e10 + 1024.0]];
    let cpu = build(&vectors, BackendPreference::Cpu);
    let gpu = build(&vectors, BackendPreference::Vulkan);
    same_hits(
        cpu.query_knn(&KnnQuery::new(&[1e10], 3)).unwrap(),
        gpu.query_knn(&KnnQuery::new(&[1e10], 3)).unwrap(),
        true,
    );
}

#[test]
fn vulkan_reuses_query_buffers_without_stale_or_aliased_results() {
    if !requested() {
        return;
    }
    let vectors: Vec<Vec<f32>> = (0..2049)
        .map(|id| vec![1.0 + id as f32 / 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .collect();
    let cpu = build(&vectors, BackendPreference::Cpu);
    let gpu = build(&vectors, BackendPreference::Vulkan);
    let origin = [0.0; 7];
    let retained = gpu.query_disk(&DiskQuery::new(&origin, 4.0)).unwrap();
    let snapshot = retained.hits.clone();
    let retained_knn = gpu.query_knn(&KnnQuery::new(&origin, 300)).unwrap();
    let knn_snapshot = retained_knn.hits.clone();
    let sparse = gpu.candidates_from_ids([0, 50, 2048]).unwrap();
    let dense = gpu.candidates_from_ids((0..2049).step_by(2)).unwrap();
    for filter in [None, Some(&sparse), Some(&dense), None] {
        for radius in [1.0, 256.0, 0.0, 8.0, 128.0, 0.0] {
            let mut disk = DiskQuery::new(&origin, radius);
            if let Some(filter) = filter {
                disk = disk.with_candidates(filter);
            }
            same_hits(
                cpu.query_disk(&disk).unwrap(),
                gpu.query_disk(&disk).unwrap(),
                false,
            );
            // Reuse the same buffers for geometry, GPU KNN selection, and
            // host KNN selection, including shrinking candidate counts.
            for k in [3, 300] {
                let mut query = KnnQuery::new(&origin, k);
                if let Some(filter) = filter {
                    query = query.with_candidates(filter);
                }
                same_hits(
                    cpu.query_knn(&query).unwrap(),
                    gpu.query_knn(&query).unwrap(),
                    true,
                );
            }
            let disks = [
                disk,
                DiskQuery::new(&origin, radius).with_weights(&[1.0; 7]),
            ];
            let intersection = DiskIntersectionQuery::new(&disks);
            same_hits(
                cpu.query_disk_intersection(&intersection).unwrap(),
                gpu.query_disk_intersection(&intersection).unwrap(),
                false,
            );
        }
    }
    assert_eq!(
        retained.hits, snapshot,
        "returned hits must own their storage"
    );
    assert_eq!(retained_knn.hits, knn_snapshot);
}

#[test]
fn vulkan_reads_large_results_from_concurrent_rayon_callers() {
    if !requested() {
        return;
    }
    // Exceed the parallel readback threshold, with hit records that can
    // straddle byte-copy chunks. Keep dimensions small for software CI.
    let rows = 2_100_013;
    let mut builder =
        RingDb::new(RingDbConfig::new(1).with_backend_preference(BackendPreference::Vulkan))
            .unwrap();
    for id in 0..rows {
        builder.add_vector(&[(id % 1024) as f32], ()).unwrap();
    }
    let db = builder.build().unwrap();
    // A query must not run unrelated caller jobs while holding its mutex.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        rayon::join(
            || {
                let result = db.query_disk(&DiskQuery::new(&[0.0], 1024.0)).unwrap();
                assert_eq!(result.hits.len(), rows);
                let mut seen = vec![false; rows];
                for hit in result.hits {
                    let id = hit.id as usize;
                    assert!(id < rows);
                    assert!(!seen[id], "duplicate ID {id}");
                    seen[id] = true;
                    let value = (id % 1024) as f32;
                    assert_eq!(hit.dist_sq, value * value);
                }
            },
            || {
                for _ in 0..4 {
                    assert!(
                        db.query_disk(&DiskQuery::new(&[-1.0], 0.0))
                            .unwrap()
                            .hits
                            .is_empty()
                    );
                }
            },
        );
    });
}

#[test]
fn vulkan_loads_persisted_payloads_metadata_and_supports_concurrent_queries() {
    if !requested() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let mut db = RingDb::new(RingDbConfig::new(3).with_persist_dir(dir.path())).unwrap();
    for id in 0..269u32 {
        db.add_vector_with_metadata(
            &[id as f32, 1.0, 2.0],
            id,
            &[MetadataTag::new(
                MetadataKey::new(1),
                MetadataValue::new((id % 3) as u64),
            )],
        )
        .unwrap();
    }
    let cpu = db.build().unwrap();
    let gpu = RingDb::<u32>::load(dir.path(), BackendPreference::Vulkan).unwrap();
    let filter = gpu.metadata_eq(MetadataKey::new(1), MetadataValue::new(1));
    std::thread::scope(|scope| {
        for shift in 0..4 {
            let (cpu, gpu, filter) = (&cpu, &gpu, &filter);
            scope.spawn(move || {
                let point = [shift as f32, 0.0, 0.0];
                for _ in 0..4 {
                    let query = KnnQuery::new(&point, 3).with_candidates(filter);
                    let result = gpu.query_knn(&query).unwrap();
                    for hit in &result.hits {
                        assert_eq!(*gpu.fetch_payload(hit.id).unwrap(), hit.id);
                    }
                    same_hits(cpu.query_knn(&query).unwrap(), result, true);
                    let disk = DiskQuery::new(&point, 100.0).with_candidates(filter);
                    same_hits(
                        cpu.query_disk(&disk).unwrap(),
                        gpu.query_disk(&disk).unwrap(),
                        false,
                    );
                }
            });
        }
    });

    // Persisting on Vulkan also reloads on CPU without a format change.
    let dir = tempfile::tempdir().unwrap();
    let mut db = RingDb::new(
        RingDbConfig::new(1)
            .with_persist_dir(dir.path())
            .with_backend_preference(BackendPreference::Vulkan),
    )
    .unwrap();
    db.add_vector(&[3.0], 17u32).unwrap();
    let _gpu = db.build().unwrap();
    let loaded = RingDb::<u32>::load(dir.path(), BackendPreference::Cpu).unwrap();
    assert_eq!(*loaded.fetch_payload(0).unwrap(), 17);
}
