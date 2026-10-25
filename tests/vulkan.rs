mod common;

use common::{build, keys, same_knn, same_region};
use rand::{RngExt, SeedableRng};
use ringdb::{
    BackendPreference, Db, Disk, Filter, Key, Knn, Metric, Options, Record, Region, RingDbError,
    Schema, Value,
};

fn vulkan() -> Options {
    Options::default()
        .backend(BackendPreference::Vulkan)
        .vulkan_min_rows(0)
}

/// The `id` field of `key`, which the common builders set to the key.
fn id(db: &Db, key: Key) -> u64 {
    db.get(key)
        .unwrap()
        .get("id")
        .and_then(Value::as_u64)
        .unwrap()
}

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
    filter: Option<&'a Filter>,
) -> Knn<'a> {
    let query = Knn::new(query, k).metric(metric(weights));
    match filter {
        Some(filter) => query.filter(filter),
        None => query,
    }
}

fn region<'a>(disks: &[Disk<'a>], filter: Option<&'a Filter>) -> Region<'a> {
    let mut region = Region::from(disks[0]);
    for disk in &disks[1..] {
        region = region.and(*disk);
    }
    match filter {
        Some(filter) => region.filter(filter),
        None => region,
    }
}

fn parity_knn(cpu: &Db, gpu: &Db, query: Knn<'_>) {
    assert_eq!(gpu.backend(), BackendPreference::Vulkan);
    same_knn(&cpu.search(query).unwrap(), &gpu.search(query).unwrap());
}

fn parity_region(cpu: &Db, gpu: &Db, query: Region<'_>) {
    let expected = cpu.search(query.clone()).unwrap();
    same_region(&expected, &gpu.search(query.clone()).unwrap());
    assert_eq!(gpu.search(query.count()).unwrap(), expected.len());
}

#[test]
fn cpu_works_and_vulkan_reports_unavailable_without_a_driver() {
    // Use a child process so loader environment overrides cannot race the
    // device tests or change a loader that is already initialized.
    const CHILD: &str = "RINGDB_VULKAN_MISSING_DRIVER_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let result = Db::in_memory(Schema::new(1), vulkan());
        assert!(matches!(result, Err(RingDbError::BackendUnavailable(_))));
        let cpu = Db::in_memory(Schema::new(1), Options::default()).unwrap();
        cpu.add(Record::new([1.0])).unwrap();
        cpu.add(Record::new([3.0])).unwrap();
        cpu.flush().unwrap();
        assert_eq!(cpu.backend(), BackendPreference::Cpu);
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

fn candidates(db: &Db) -> Vec<Filter> {
    let n = db.len() as u64;
    vec![
        keys(0..n),
        keys([]),
        keys([0, 31, 64, n - 1]),
        common::group(1),
        !keys([0, 63]),
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
        assert_eq!(gpu.backend(), BackendPreference::Vulkan);
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
    // across workgroups exercise ID tie-breaking, winner invalidation, and
    // the repair of the tree above each winner. Up to k = 200 is selected on
    // the GPU; larger k are selected on the host.
    let vectors: Vec<Vec<f32>> = (0..16_513).map(|id| vec![(id % 7) as f32]).collect();
    let cpu = build(&vectors, BackendPreference::Cpu);
    let gpu = build(&vectors, BackendPreference::Vulkan);
    let sparse = keys((0..gpu.len() as u64).step_by(33));
    let dense = keys((0..gpu.len() as u64).step_by(2));
    for filter in [None, Some(&sparse), Some(&dense)] {
        for k in [1, 5, 17, 200, 300, 20_000] {
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
    let empty = Db::in_memory(Schema::new(2), vulkan()).unwrap();
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
    let unknown = Filter::eq("missing", 1);
    assert!(matches!(
        gpu.search(Knn::new(&origin, 1).filter(&unknown)),
        Err(RingDbError::UnknownField(_))
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
    let sparse = keys([0, 50, 2048]);
    let dense = keys((0..2049).step_by(2));
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
    let db = Db::in_memory(Schema::new(1), vulkan()).unwrap();
    for id in 0..rows {
        db.add(Record::new([(id % 1024) as f32])).unwrap();
    }
    db.flush().unwrap();
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

/// The same records on `backend`, split into segments after each of `cuts`.
fn segmented(vectors: &[Vec<f32>], cuts: &[usize], options: Options) -> Db {
    let db = Db::in_memory(common::schema(vectors[0].len()), options).unwrap();
    for (id, vector) in vectors.iter().enumerate() {
        let record = Record::new(vector.as_slice())
            .set("group", (id % 3) as u32)
            .set("id", id as u64);
        db.add(record).unwrap();
        if cuts.contains(&id) {
            db.flush().unwrap();
        }
    }
    db.flush().unwrap();
    assert_eq!(db.snapshot().segments(), cuts.len() + 1);
    db
}

#[test]
fn vulkan_searches_segments_and_supports_concurrent_queries() {
    if !requested() {
        return;
    }
    let vectors: Vec<Vec<f32>> = (0..269).map(|id| vec![id as f32, 1.0, 2.0]).collect();
    // Segments of 1, 63, 64, and 141 rows, cut inside and at bitmap words.
    let cuts = [0, 63, 127];
    let cpu = build(&vectors, BackendPreference::Cpu);
    let filter = common::group(1);
    // All segments on the GPU, then the two small ones on the CPU.
    for min_rows in [0, 64] {
        let options = common::manual(BackendPreference::Vulkan).vulkan_min_rows(min_rows);
        let gpu = segmented(&vectors, &cuts, options);
        concurrent_parity(&cpu, &gpu, &filter);
    }
}

fn concurrent_parity(cpu: &Db, gpu: &Db, filter: &Filter) {
    std::thread::scope(|scope| {
        for shift in 0..4 {
            scope.spawn(move || {
                let point = [shift as f32 * 40.0, 0.0, 0.0];
                for k in [3, 100] {
                    let query = Knn::new(&point, k).filter(filter);
                    let result = gpu.search(query).unwrap();
                    for hit in &result {
                        assert_eq!(id(gpu, hit.key), hit.key.get());
                    }
                    same_knn(&cpu.search(query).unwrap(), &result);
                    parity_knn(cpu, gpu, Knn::new(&point, k));
                }
                parity_region(cpu, gpu, Disk::new(&point, 100.0).filter(filter));
                let region = Disk::new(&point, 120.0).and(Disk::new(&[60.0, 1.0, 2.0], 90.0));
                parity_region(cpu, gpu, region);
                let edges = keys([0, 1, 63, 64, 127, 128, 268]);
                parity_region(cpu, gpu, Disk::new(&point, 1e4).filter(&edges));
            });
        }
    });
}

#[test]
fn vulkan_matches_cpu_after_deletes_and_merges() {
    if !requested() {
        return;
    }
    let schema = || common::schema(4);
    let cpu = Db::in_memory(schema(), common::manual(BackendPreference::Cpu)).unwrap();
    let gpu = Db::in_memory(schema(), common::manual(BackendPreference::Vulkan)).unwrap();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
    let mut live = Vec::new();
    for batch in 0..12 {
        for _ in 0..97 {
            let vector: Vec<f32> = (0..4).map(|_| rng.random_range(-2.0f32..2.0)).collect();
            let record = || Record::new(vector.as_slice()).set("group", live.len() as u32 % 3);
            let key = cpu.add(record()).unwrap();
            assert_eq!(gpu.add(record()).unwrap(), key);
            live.push(key);
        }
        for _ in 0..40 {
            let key = live.swap_remove(rng.random_range(0..live.len()));
            cpu.delete(key).unwrap();
            gpu.delete(key).unwrap();
        }
        for db in [&cpu, &gpu] {
            if batch % 4 == 3 {
                db.compact().unwrap();
            } else {
                db.flush().unwrap();
            }
        }
        let filter = common::group(1);
        let origin = [0.0; 4];
        for k in [1, 10, 500] {
            parity_knn(&cpu, &gpu, Knn::new(&origin, k));
            parity_knn(&cpu, &gpu, Knn::new(&origin, k).filter(&filter));
        }
        parity_region(&cpu, &gpu, Disk::new(&origin, 2.5).into());
        parity_region(
            &cpu,
            &gpu,
            Disk::new(&origin, 3.0).and(Disk::new(&[1.0; 4], 2.0)),
        );
        parity_region(&cpu, &gpu, Disk::new(&origin, 9.0).filter(&filter));
        assert_eq!(gpu.len(), live.len());
    }
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
        keys([]),
        keys(0..2309),
        keys([
            0, 31, 32, 63, 64, 1022, 1023, 1026, 1027, 1028, 2045, 2046, 2053, 2054, 2308,
        ]),
        keys((0..2309).step_by(2)),
        keys([2054, 2308]),
        !keys([0, 1026, 1027, 2053, 2054, 2308]),
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
        assert_eq!(id(&gpu, hit.key), hit.key.get());
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

    // Chunked segments of different sizes keep their row offsets.
    let gpu = segmented(
        &vectors,
        &[700, 1500],
        common::manual(BackendPreference::Vulkan),
    );
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
    let db = Db::in_memory(Schema::new(64), vulkan()).unwrap();
    for id in 0..rows {
        let mut row = [100.0; 64];
        if let Some(rank) = special.iter().position(|&candidate| candidate == id) {
            row.fill(0.0);
            row[63] = (3 - rank) as f32;
        }
        db.add(Record::new(row)).unwrap();
    }
    db.flush().unwrap();
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
    let sparse = keys([0, u64::from(rows - 1)]);
    assert_eq!(
        common::knn_keys(&db.search(query.filter(&sparse)).unwrap()),
        [rows - 1, 0].map(u64::from)
    );
    let dense = !keys([0]);
    let inside = db.search(Disk::new(&point, 2.0).filter(&dense)).unwrap();
    assert_eq!(
        common::region_keys(&inside),
        [rows - 300, rows - 2, rows - 1].map(u64::from)
    );
}

#[test]
fn vulkan_facets_match_cpu() {
    if !requested() {
        return;
    }
    let cpu = common::keyword_db(common::manual(BackendPreference::Cpu));
    let gpu = common::keyword_db(common::manual(BackendPreference::Vulkan));
    assert_eq!(gpu.backend(), BackendPreference::Vulkan);
    let lille = Filter::eq("city", "Lille");
    for filter in [None, Some(&lille)] {
        let mut region = Disk::new(&[10.0, 10.0], 8.0).and(Disk::new(&[12.0, 12.0], 9.0));
        if let Some(filter) = filter {
            region = region.filter(filter);
        }
        let facets = region.facets(&["city", "tags"]);
        let expected = cpu.search(facets.clone()).unwrap();
        assert!(!expected[0].is_empty());
        assert_eq!(gpu.search(facets).unwrap(), expected);
    }
}
