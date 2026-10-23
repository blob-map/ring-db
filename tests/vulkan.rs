mod common;

use common::{build, same_knn, same_region};
use rand::{RngExt, SeedableRng};
use ringdb::{
    BackendPreference, CandidateSet, Disk, Knn, MetadataValue, Metric, Region, RingDb,
    RingDbConfig, RingDbError, SealedRingDb,
};

fn requested() -> bool {
    std::env::var("RINGDB_VULKAN_TEST").as_deref() == Ok("1")
}

fn metric(weights: Option<&[f32]>) -> Metric<'_> {
    weights.map_or(Metric::L2, Metric::WeightedL2)
}

fn knn<'a>(
    query: &'a [f32],
    k: usize,
    weights: Option<&'a [f32]>,
    filter: Option<&'a CandidateSet>,
) -> Knn<'a> {
    let query = Knn::new(query, k).metric(metric(weights));
    match filter {
        Some(filter) => query.filter(filter),
        None => query,
    }
}

fn region<'a>(disks: &[Disk<'a>], filter: Option<&'a CandidateSet>) -> Region<'a> {
    let mut region = Region::from(disks[0]);
    for disk in &disks[1..] {
        region = region.and(*disk);
    }
    match filter {
        Some(filter) => region.filter(filter),
        None => region,
    }
}

fn parity_knn(cpu: &SealedRingDb<u32>, gpu: &SealedRingDb<u32>, query: Knn<'_>) {
    assert_eq!(gpu.backend_name(), "vulkan");
    same_knn(&cpu.search(query).unwrap(), &gpu.search(query).unwrap());
}

fn parity_region(cpu: &SealedRingDb<u32>, gpu: &SealedRingDb<u32>, query: Region<'_>) {
    same_region(
        &cpu.search(query.clone()).unwrap(),
        &gpu.search(query).unwrap(),
    );
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
        let hits = cpu.search(Knn::new(&[0.0], 1)).unwrap();
        assert_eq!(hits[0].key.get(), 0);
        assert_eq!(hits[0].dist_sq, 1.0);
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

fn candidates(db: &SealedRingDb<u32>) -> Vec<CandidateSet> {
    vec![
        db.candidates_from_ids(0..db.len() as u32).unwrap(),
        db.candidates_from_ids([]).unwrap(),
        db.candidates_from_ids([0, 31, 64, db.len() as u32 - 1])
            .unwrap(),
        db.metadata_eq(common::GROUP, MetadataValue::new(1)),
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
            for weights in [None, Some(weights.as_slice()), Some(ones.as_slice())] {
                let disk = Disk::new(&origin, radius).metric(metric(weights));
                parity_region(&cpu, &gpu, region(&[disk], filter));
                for k in [0, 1, 3, 19, 400] {
                    parity_knn(&cpu, &gpu, knn(&origin, k, weights, filter));
                }
                let second = Disk::new(&shifted, radius * 1.2).metric(Metric::WeightedL2(&ones));
                parity_region(&cpu, &gpu, region(&[disk, second], filter));
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
                    let mut second = Disk::new(&b, radius * 1.1);
                    if weighted {
                        second = second.metric(Metric::WeightedL2(&weights));
                    }
                    let disks = [Disk::new(&a, radius), second, Disk::new(&c, radius)];
                    parity_region(&cpu, &gpu, region(&disks, filter));
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
            parity_knn(&cpu, &gpu, knn(&[0.0], k, None, filter));
        }
    }
    let vectors = vec![vec![f32::MAX, 1.0]; 269];
    let cpu = build(&vectors, BackendPreference::Cpu);
    let gpu = build(&vectors, BackendPreference::Vulkan);
    let point = [-f32::MAX, 0.0];
    for weights in [None, Some([0.0, 1.0]), Some([1.0, 1.0])] {
        let weights = weights.as_ref().map(|w| w.as_slice());
        for k in [3, 269] {
            parity_knn(&cpu, &gpu, knn(&point, k, weights, None));
        }
        parity_region(
            &cpu,
            &gpu,
            region(&[Disk::new(&point, 1.0).metric(metric(weights))], None),
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
    let origin = [0.0, 0.0];
    assert!(empty.search(Knn::new(&origin, 10)).unwrap().is_empty());
    assert!(empty.search(Disk::new(&origin, 10.0)).unwrap().is_empty());
    assert!(matches!(
        empty.search(Knn::new(&[0.0], 0)),
        Err(RingDbError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        empty.search(Knn::new(&origin, 0).metric(Metric::WeightedL2(&[-1.0, 1.0]))),
        Err(RingDbError::InvalidQuery(_))
    ));
    let vectors = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![6.0, 8.0]];
    let gpu = build(&vectors, BackendPreference::Vulkan);
    let cpu = build(&vectors, BackendPreference::Cpu);
    assert_eq!(gpu.search(Disk::new(&origin, 10.0)).unwrap().len(), 3);
    parity_region(&cpu, &gpu, region(&[Disk::new(&origin, 5.0)], None));
    assert_eq!(
        common::region_keys(&gpu.search(Disk::new(&origin, 0.0)).unwrap()),
        [0]
    );
    let wrong = empty.candidates_from_ids([]).unwrap();
    assert!(matches!(
        gpu.search(Knn::new(&origin, 1).filter(&wrong)),
        Err(RingDbError::CandidateSetMismatch { .. })
    ));

    // Nearby values at a large offset must use direct deltas, not norms.
    let vectors = vec![vec![1e10], vec![1e10 + 2048.0], vec![1e10 + 1024.0]];
    let cpu = build(&vectors, BackendPreference::Cpu);
    let gpu = build(&vectors, BackendPreference::Vulkan);
    parity_knn(&cpu, &gpu, Knn::new(&[1e10], 3));
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
    let retained = gpu.search(Disk::new(&origin, 4.0)).unwrap();
    let snapshot = retained.clone();
    let retained_knn = gpu.search(Knn::new(&origin, 300)).unwrap();
    let knn_snapshot = retained_knn.clone();
    let sparse = gpu.candidates_from_ids([0, 50, 2048]).unwrap();
    let dense = gpu.candidates_from_ids((0..2049).step_by(2)).unwrap();
    let ones = [1.0; 7];
    for filter in [None, Some(&sparse), Some(&dense), None] {
        for radius in [1.0, 256.0, 0.0, 8.0, 128.0, 0.0] {
            let disk = Disk::new(&origin, radius);
            parity_region(&cpu, &gpu, region(&[disk], filter));
            // Reuse the same buffers for regions, GPU KNN selection, and host
            // KNN selection, including shrinking candidate counts.
            for k in [3, 300] {
                parity_knn(&cpu, &gpu, knn(&origin, k, None, filter));
            }
            let weighted = Disk::new(&origin, radius).metric(Metric::WeightedL2(&ones));
            parity_region(&cpu, &gpu, region(&[disk, weighted], filter));
        }
    }
    assert_eq!(retained, snapshot, "returned hits must own their storage");
    assert_eq!(retained_knn, knn_snapshot);
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
                let result = db.search(Disk::new(&[0.0], 1024.0)).unwrap();
                assert_eq!(result.len(), rows);
                let mut seen = vec![false; rows];
                for hit in result.iter() {
                    let id = hit.key.get() as usize;
                    assert!(id < rows);
                    assert!(!seen[id], "duplicate key {id}");
                    seen[id] = true;
                    let value = (id % 1024) as f32;
                    assert_eq!(hit.dists_sq(), [value * value]);
                }
            },
            || {
                for _ in 0..4 {
                    assert!(db.search(Disk::new(&[-1.0], 0.0)).unwrap().is_empty());
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
            &[ringdb::MetadataTag::new(
                common::GROUP,
                MetadataValue::new((id % 3) as u64),
            )],
        )
        .unwrap();
    }
    let cpu = db.build().unwrap();
    let gpu = RingDb::<u32>::load(dir.path(), BackendPreference::Vulkan).unwrap();
    let filter = gpu.metadata_eq(common::GROUP, MetadataValue::new(1));
    std::thread::scope(|scope| {
        for shift in 0..4 {
            let (cpu, gpu, filter) = (&cpu, &gpu, &filter);
            scope.spawn(move || {
                let point = [shift as f32, 0.0, 0.0];
                for _ in 0..4 {
                    let query = Knn::new(&point, 3).filter(filter);
                    let result = gpu.search(query).unwrap();
                    for hit in &result {
                        assert_eq!(
                            u64::from(*gpu.fetch_payload(hit.key).unwrap()),
                            hit.key.get()
                        );
                    }
                    same_knn(&cpu.search(query).unwrap(), &result);
                    parity_region(cpu, gpu, Disk::new(&point, 100.0).filter(filter));
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
    let hit = loaded.search(Knn::new(&[0.0], 1)).unwrap()[0];
    assert_eq!(*loaded.fetch_payload(hit.key).unwrap(), 17);
}

#[test]
fn vulkan_chunks_preserve_global_ids_filters_and_exact_knn() {
    if !requested() {
        return;
    }
    const CHILD: &str = "RINGDB_VULKAN_CHUNKS_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // Run in isolation: environment changes must not race other GPU tests.
        // Cut both u32 and u64 bitmap words. 1,023 rows also reproduces the
        // last-bit boundary of 64D chunks at a 4-GiB-minus-one device limit.
        for rows in [1023, 1027] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "vulkan_chunks_preserve_global_ids_filters_and_exact_knn",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("RINGDB_VULKAN_CHUNK_BYTES", (rows * 3 * 4).to_string())
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            print!("{stdout}");
            eprint!("{stderr}");
            assert!(output.status.success(), "{stdout}\n{stderr}");
        }
        return;
    }
    let vectors: Vec<Vec<f32>> = (0..2309)
        .map(|id| vec![(id % 7) as f32, (id % 3) as f32, (id % 13) as f32])
        .collect();
    let cpu = build(&vectors, BackendPreference::Cpu);
    let gpu = build(&vectors, BackendPreference::Vulkan);
    let sets = [
        gpu.candidates_from_ids([]).unwrap(),
        gpu.candidates_from_ids(0..2309).unwrap(),
        gpu.candidates_from_ids([
            0, 31, 32, 63, 64, 1022, 1023, 1026, 1027, 1028, 2045, 2046, 2053, 2054, 2308,
        ])
        .unwrap(),
        gpu.candidates_from_ids((0..2309).step_by(2)).unwrap(),
        gpu.candidates_from_ids([2054, 2308]).unwrap(),
        gpu.candidates_from_ids([0, 1026, 1027, 2053, 2054, 2308])
            .unwrap()
            .complement(),
    ];
    let retained = gpu.search(Disk::new(&[0.0; 3], 100.0)).unwrap();
    let snapshot = retained.clone();
    for filter in std::iter::once(None).chain(sets.iter().map(Some)) {
        for weights in [None, Some([0.0, 0.25, 2.0]), Some([0.0; 3])] {
            let weights = weights.as_ref().map(|w| w.as_slice());
            for k in [0, 1, 10, 1300, 3000] {
                parity_knn(&cpu, &gpu, knn(&[0.0; 3], k, weights, filter));
            }
            for radius in [0.0, 5.0, 100.0] {
                let disk = Disk::new(&[0.0; 3], radius).metric(metric(weights));
                parity_region(&cpu, &gpu, region(&[disk], filter));
                let disks = [
                    disk,
                    Disk::new(&[0.5; 3], radius),
                    Disk::new(&[-0.5; 3], radius).metric(Metric::WeightedL2(&[0.25; 3])),
                ];
                parity_region(&cpu, &gpu, region(&disks, filter));
            }
        }
    }
    assert_eq!(retained, snapshot);
    for hit in retained.iter() {
        assert_eq!(
            u64::from(*gpu.fetch_payload(hit.key).unwrap()),
            hit.key.get()
        );
    }

    // Overflowed distances remain valid KNN candidates. Zero weights must
    // bypass overflowing subtraction, including in later chunks.
    let extremes = vec![vec![f32::MAX, 1.0, 0.0]; 2309];
    let extreme_cpu = build(&extremes, BackendPreference::Cpu);
    let extreme_gpu = build(&extremes, BackendPreference::Vulkan);
    for weights in [[1.0; 3], [0.0, 1.0, 0.0]] {
        for k in [10, 1300] {
            let query = Knn::new(&[-f32::MAX, 0.0, 0.0], k).metric(Metric::WeightedL2(&weights));
            parity_knn(&extreme_cpu, &extreme_gpu, query);
        }
    }

    // Loading a persisted CPU database follows the same chunked upload path.
    let dir = tempfile::tempdir().unwrap();
    let mut persisted = RingDb::new(RingDbConfig::new(3).with_persist_dir(dir.path())).unwrap();
    for (id, vector) in vectors.iter().enumerate() {
        persisted.add_vector(vector, id as u32).unwrap();
    }
    let cpu = persisted.build().unwrap();
    let gpu = RingDb::<u32>::load(dir.path(), BackendPreference::Vulkan).unwrap();
    std::thread::scope(|scope| {
        for center in [[0.0; 3], [2.0; 3], [6.0; 3]] {
            let (cpu, gpu) = (&cpu, &gpu);
            scope.spawn(move || {
                same_knn(
                    &cpu.search(Knn::new(&center, 10)).unwrap(),
                    &gpu.search(Knn::new(&center, 10)).unwrap(),
                );
                same_region(
                    &cpu.search(Disk::new(&center, 100.0)).unwrap(),
                    &gpu.search(Disk::new(&center, 100.0)).unwrap(),
                );
            });
        }
    });
}

#[test]
fn vulkan_reads_vectors_past_four_gib() {
    // Opt in separately: this regression allocates >4 GiB of vector data.
    if !requested() || std::env::var("RINGDB_VULKAN_LARGE_TEST").as_deref() != Ok("1") {
        return;
    }
    let rows = 16_777_219u32;
    let special = [0, rows - 300, rows - 2, rows - 1];
    let mut db: RingDb =
        RingDb::new(RingDbConfig::new(64).with_backend_preference(BackendPreference::Vulkan))
            .unwrap();
    for id in 0..rows {
        let mut row = [100.0; 64];
        if let Some(rank) = special.iter().position(|&candidate| candidate == id) {
            row.fill(0.0);
            row[63] = (3 - rank) as f32;
        }
        db.add_vector(&row, ()).unwrap();
    }
    let db = db.build().unwrap();
    let point = [0.0; 64];
    // The last coordinate of the last two vectors is beyond byte offset 2^32
    // when this device can hold the vectors in one allocation.
    let query = Knn::new(&point, 4);
    let hits = db.search(query).unwrap();
    assert_eq!(
        common::knn_keys(&hits),
        [rows - 1, rows - 2, rows - 300, 0].map(u64::from)
    );
    assert_eq!(
        hits.iter().map(|h| h.dist_sq).collect::<Vec<_>>(),
        [0.0, 1.0, 4.0, 9.0]
    );
    let sparse = db.candidates_from_ids([0, rows - 1]).unwrap();
    assert_eq!(
        common::knn_keys(&db.search(query.filter(&sparse)).unwrap()),
        [rows - 1, 0].map(u64::from)
    );
    let dense = db.candidates_from_ids([0]).unwrap().complement();
    let inside = db.search(Disk::new(&point, 2.0).filter(&dense)).unwrap();
    assert_eq!(
        common::region_keys(&inside),
        [rows - 300, rows - 2, rows - 1].map(u64::from)
    );
}
