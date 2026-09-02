use bytemuck::{Pod, Zeroable};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng};
use ringdb::{
    BackendPreference, DiskIntersectionQuery, DiskQuery, Payload, RingDb, RingDbConfig,
    RingDbError, RingQuery, SealedRingDb,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const N: usize = 30_000_000;
const SEED: u64 = 42;
const QUERY_DIMS: &[usize] = &[16, 32, 64, 128];

#[derive(Serialize, Deserialize, Payload)]
struct DynamicPayload {
    label: String,
}

#[derive(Clone, Copy, Pod, Zeroable, Payload)]
#[repr(C)]
#[payload(storage = "pod")]
struct StaticPayload {
    score: f32,
}

fn bench_filter_allows(group_name: &str) -> bool {
    let filters: Vec<String> = std::env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with('-'))
        .filter(|arg| {
            let lower = arg.to_ascii_lowercase();
            [
                "cpu",
                "cuda",
                "ring",
                "disk",
                "intersection",
                "weighted",
                "mixed",
                "scaling",
                "payload",
                "f32",
            ]
            .iter()
            .any(|token| lower.contains(token))
        })
        .collect();

    if filters.is_empty() {
        return true;
    }

    filters
        .iter()
        .any(|filter| group_name.contains(filter) || filter.contains(group_name))
}

fn env_bench_n(name: &str) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| match raw.parse::<usize>() {
            Ok(n) => Some(n),
            Err(err) => {
                eprintln!("ignoring invalid {name} value: {err}; using default {N}");
                None
            }
        })
        .unwrap_or(N)
}

fn cpu_bench_n() -> usize {
    env_bench_n("RINGDB_BENCH_N")
}

#[cfg(feature = "cuda")]
fn cuda_bench_n() -> usize {
    env_bench_n("RINGDB_CUDA_BENCH_N")
}

#[cfg(feature = "cuda")]
fn cuda_is_available(bench_name: &str) -> bool {
    match RingDb::<()>::new(RingDbConfig::new(1).with_backend_preference(BackendPreference::Cuda)) {
        Ok(_) => true,
        Err(err) => {
            eprintln!("skipping {bench_name}: CUDA backend unavailable: {err}");
            false
        }
    }
}

fn build_unit_db(
    dims: usize,
    n_vectors: usize,
    backend: BackendPreference,
) -> std::result::Result<SealedRingDb<()>, RingDbError> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
    let mut buf = vec![0.0f32; dims];
    let mut db = RingDb::new(RingDbConfig::new(dims).with_backend_preference(backend))?;

    for _ in 0..n_vectors {
        for x in buf.iter_mut() {
            *x = rng.random_range(-1.0f32..1.0);
        }
        db.add_vector(&buf, ())?;
    }

    db.build()
}

fn random_query(dims: usize) -> Vec<f32> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
    (0..dims).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

fn mean_one_weights(dims: usize, phase: usize) -> Vec<f32> {
    const PATTERN: [f32; 4] = [0.25, 0.75, 1.25, 1.75];
    (0..dims)
        .map(|i| PATTERN[(i + phase) % PATTERN.len()])
        .collect()
}

fn expected_distance(dims: usize) -> f32 {
    ((2.0 * dims as f32) / 3.0).sqrt()
}

fn intersection_centers(dims: usize) -> [Vec<f32>; 3] {
    let query_a = random_query(dims);
    let mut query_b = query_a.clone();
    let mut query_c = query_a.clone();
    for x in &mut query_b {
        *x += 0.01;
    }
    for x in &mut query_c {
        *x -= 0.01;
    }
    [query_a, query_b, query_c]
}

fn bench_cpu_f32(c: &mut Criterion) {
    let bench_name = "cpu_f32";
    if !bench_filter_allows(bench_name) {
        return;
    }

    let n_vectors = cpu_bench_n();
    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);
    group.throughput(Throughput::Elements(n_vectors as u64));

    for &dims in QUERY_DIMS {
        let db = build_unit_db(dims, n_vectors, BackendPreference::Cpu).unwrap();
        let query = random_query(dims);
        let weights = mean_one_weights(dims, 0);
        let d = expected_distance(dims);
        let lambda = d * 0.0005;
        let ordinary = RingQuery::new(&query, d, lambda);
        let weighted = RingQuery::new(&query, d, lambda).with_weights(&weights);

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| db.query(&ordinary).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("weighted", dims), &dims, |b, _| {
            b.iter(|| db.query(&weighted).unwrap());
        });
    }
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_cuda_f32(c: &mut Criterion) {
    let bench_name = "cuda_f32";
    if !bench_filter_allows(bench_name) {
        return;
    }
    if !cuda_is_available(bench_name) {
        return;
    }

    let n_vectors = cuda_bench_n();
    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    group.throughput(Throughput::Elements(n_vectors as u64));

    for &dims in QUERY_DIMS {
        let db = match build_unit_db(dims, n_vectors, BackendPreference::Cuda) {
            Ok(db) => db,
            Err(err) => {
                eprintln!("skipping {bench_name}/{dims}: failed to build CUDA db: {err}");
                continue;
            }
        };
        let query = random_query(dims);
        let weights = mean_one_weights(dims, 0);
        let d = expected_distance(dims);
        let lambda = d * 0.0005;
        let ordinary = RingQuery::new(&query, d, lambda);
        let weighted = RingQuery::new(&query, d, lambda).with_weights(&weights);

        if let Err(err) = db.query(&weighted) {
            eprintln!("skipping {bench_name}/{dims}: warmup query failed: {err}");
            continue;
        }

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| db.query(&ordinary).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("weighted", dims), &dims, |b, _| {
            b.iter(|| db.query(&weighted).unwrap());
        });
    }
    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn bench_cuda_f32(_: &mut Criterion) {}

fn bench_cpu_disk_f32(c: &mut Criterion) {
    let bench_name = "cpu_disk_f32";
    if !bench_filter_allows(bench_name) {
        return;
    }

    let n_vectors = cpu_bench_n();
    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);
    group.throughput(Throughput::Elements(n_vectors as u64));

    for &dims in QUERY_DIMS {
        let db = build_unit_db(dims, n_vectors, BackendPreference::Cpu).unwrap();
        let query = random_query(dims);
        let weights = mean_one_weights(dims, 0);
        let d_max = expected_distance(dims);
        let ordinary = DiskQuery::new(&query, d_max);
        let weighted = DiskQuery::new(&query, d_max).with_weights(&weights);

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| db.query_disk(&ordinary).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("weighted", dims), &dims, |b, _| {
            b.iter(|| db.query_disk(&weighted).unwrap());
        });
    }
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_cuda_disk_f32(c: &mut Criterion) {
    let bench_name = "cuda_disk_f32";
    if !bench_filter_allows(bench_name) {
        return;
    }
    if !cuda_is_available(bench_name) {
        return;
    }

    let n_vectors = cuda_bench_n();
    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);
    group.throughput(Throughput::Elements(n_vectors as u64));

    for &dims in QUERY_DIMS {
        let db = match build_unit_db(dims, n_vectors, BackendPreference::Cuda) {
            Ok(db) => db,
            Err(err) => {
                eprintln!("skipping {bench_name}/{dims}: failed to build CUDA db: {err}");
                continue;
            }
        };
        let query = random_query(dims);
        let weights = mean_one_weights(dims, 0);
        let d_max = expected_distance(dims);
        let ordinary = DiskQuery::new(&query, d_max);
        let weighted = DiskQuery::new(&query, d_max).with_weights(&weights);

        if let Err(err) = db.query_disk(&weighted) {
            eprintln!("skipping {bench_name}/{dims}: warmup query failed: {err}");
            continue;
        }

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| db.query_disk(&ordinary).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("weighted", dims), &dims, |b, _| {
            b.iter(|| db.query_disk(&weighted).unwrap());
        });
    }
    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn bench_cuda_disk_f32(_: &mut Criterion) {}

fn bench_cpu_disk_intersection_f32(c: &mut Criterion) {
    let bench_name = "cpu_disk_intersection_f32";
    if !bench_filter_allows(bench_name) {
        return;
    }

    let n_vectors = cpu_bench_n();
    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);
    group.throughput(Throughput::Elements(n_vectors as u64));

    for &dims in QUERY_DIMS {
        let db = build_unit_db(dims, n_vectors, BackendPreference::Cpu).unwrap();
        let centers = intersection_centers(dims);
        let weights = [
            mean_one_weights(dims, 0),
            mean_one_weights(dims, 1),
            mean_one_weights(dims, 2),
        ];
        let d_max = expected_distance(dims);
        let ordinary_disks = [
            DiskQuery::new(&centers[0], d_max),
            DiskQuery::new(&centers[1], d_max),
            DiskQuery::new(&centers[2], d_max),
        ];
        let weighted_disks = [
            DiskQuery::new(&centers[0], d_max).with_weights(&weights[0]),
            DiskQuery::new(&centers[1], d_max).with_weights(&weights[1]),
            DiskQuery::new(&centers[2], d_max).with_weights(&weights[2]),
        ];
        let mixed_disks = [
            DiskQuery::new(&centers[0], d_max).with_weights(&weights[0]),
            DiskQuery::new(&centers[1], d_max),
            DiskQuery::new(&centers[2], d_max).with_weights(&weights[2]),
        ];
        let ordinary = DiskIntersectionQuery::new(&ordinary_disks);
        let all_weighted = DiskIntersectionQuery::new(&weighted_disks);
        let mixed = DiskIntersectionQuery::new(&mixed_disks);

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| db.query_disk_intersection(&ordinary).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("all_weighted", dims), &dims, |b, _| {
            b.iter(|| db.query_disk_intersection(&all_weighted).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("mixed", dims), &dims, |b, _| {
            b.iter(|| db.query_disk_intersection(&mixed).unwrap());
        });
    }
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_cuda_disk_intersection_f32(c: &mut Criterion) {
    let bench_name = "cuda_disk_intersection_f32";
    if !bench_filter_allows(bench_name) {
        return;
    }
    if !cuda_is_available(bench_name) {
        return;
    }

    let n_vectors = cuda_bench_n();
    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    group.throughput(Throughput::Elements(n_vectors as u64));

    for &dims in QUERY_DIMS {
        let db = match build_unit_db(dims, n_vectors, BackendPreference::Cuda) {
            Ok(db) => db,
            Err(err) => {
                eprintln!("skipping {bench_name}/{dims}: failed to build CUDA db: {err}");
                continue;
            }
        };

        let centers = intersection_centers(dims);
        let weights = [
            mean_one_weights(dims, 0),
            mean_one_weights(dims, 1),
            mean_one_weights(dims, 2),
        ];
        let d_max = expected_distance(dims);
        let ordinary_disks = [
            DiskQuery::new(&centers[0], d_max),
            DiskQuery::new(&centers[1], d_max),
            DiskQuery::new(&centers[2], d_max),
        ];
        let weighted_disks = [
            DiskQuery::new(&centers[0], d_max).with_weights(&weights[0]),
            DiskQuery::new(&centers[1], d_max).with_weights(&weights[1]),
            DiskQuery::new(&centers[2], d_max).with_weights(&weights[2]),
        ];
        let mixed_disks = [
            DiskQuery::new(&centers[0], d_max).with_weights(&weights[0]),
            DiskQuery::new(&centers[1], d_max),
            DiskQuery::new(&centers[2], d_max).with_weights(&weights[2]),
        ];
        let ordinary = DiskIntersectionQuery::new(&ordinary_disks);
        let all_weighted = DiskIntersectionQuery::new(&weighted_disks);
        let mixed = DiskIntersectionQuery::new(&mixed_disks);

        if let Err(err) = db.query_disk_intersection(&mixed) {
            eprintln!("skipping {bench_name}/{dims}: warmup query failed: {err}");
            continue;
        }

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| db.query_disk_intersection(&ordinary).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("all_weighted", dims), &dims, |b, _| {
            b.iter(|| db.query_disk_intersection(&all_weighted).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("mixed", dims), &dims, |b, _| {
            b.iter(|| db.query_disk_intersection(&mixed).unwrap());
        });
    }
    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn bench_cuda_disk_intersection_f32(_: &mut Criterion) {}

fn bench_cpu_weighted_intersection_scaling(c: &mut Criterion) {
    let bench_name = "cpu_weighted_intersection_scaling_f32";
    if !bench_filter_allows(bench_name) {
        return;
    }

    const DIMS: usize = 64;
    const MAX_DISKS: usize = 8;
    let n_vectors = cpu_bench_n();
    let db = build_unit_db(DIMS, n_vectors, BackendPreference::Cpu).unwrap();
    let base = random_query(DIMS);
    let centers: Vec<Vec<f32>> = (0..MAX_DISKS)
        .map(|disk_i| {
            let shift = (disk_i as f32 - 3.5) * 0.005;
            base.iter()
                .enumerate()
                .map(|(dim, &x)| if dim % 2 == 0 { x + shift } else { x - shift })
                .collect()
        })
        .collect();
    let weights: Vec<Vec<f32>> = (0..MAX_DISKS)
        .map(|disk_i| mean_one_weights(DIMS, disk_i))
        .collect();
    let expected = expected_distance(DIMS);

    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);
    group.throughput(Throughput::Elements(n_vectors as u64));

    for count in [1usize, 2, 4, 8] {
        let make_disk = |disk_i: usize| {
            let radius = if disk_i == 0 {
                expected * 0.75
            } else {
                expected * 1.35
            };
            DiskQuery::new(&centers[disk_i], radius).with_weights(&weights[disk_i])
        };
        let early_disks: Vec<_> = (0..count).map(&make_disk).collect();
        let late_disks: Vec<_> = (1..count)
            .chain(std::iter::once(0))
            .map(make_disk)
            .collect();
        let early = DiskIntersectionQuery::new(&early_disks);
        let late = DiskIntersectionQuery::new(&late_disks);

        group.bench_with_input(BenchmarkId::new("early_reject", count), &count, |b, _| {
            b.iter(|| db.query_disk_intersection(&early).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("late_reject", count), &count, |b, _| {
            b.iter(|| db.query_disk_intersection(&late).unwrap());
        });
    }

    group.finish();
}

fn bench_payload_fetch_dynamic(c: &mut Criterion) {
    let bench_name = "payload_fetch_dynamic";
    if !bench_filter_allows(bench_name) {
        return;
    }

    let n_vectors = cpu_bench_n();
    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    let label: String = "x".repeat(92);

    for &dims in &[64usize, 128usize] {
        let expected_dist = expected_distance(dims);
        let d = expected_dist;
        let lambda = expected_dist * 0.0005;

        let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
        let mut buf = vec![0.0f32; dims];

        let mut db: RingDb<DynamicPayload> = RingDb::new(RingDbConfig::new(dims)).unwrap();
        for _ in 0..n_vectors {
            for x in buf.iter_mut() {
                *x = rng.random_range(-1.0f32..1.0);
            }
            db.add_vector(
                &buf,
                DynamicPayload {
                    label: label.clone(),
                },
            )
            .unwrap();
        }
        let db = db.build().unwrap();

        let query = random_query(dims);

        let result = db.query(&RingQuery::new(&query, d, lambda)).unwrap();
        let ids = result.ids();
        let n_hits = ids.len();

        println!(
            "dims={dims} ring=[{:.3}, {:.3}] hits={n_hits} ({:.2}%)",
            d - lambda,
            d + lambda,
            100.0 * n_hits as f32 / n_vectors.max(1) as f32
        );

        group.throughput(Throughput::Elements(n_hits as u64));
        group.bench_with_input(
            BenchmarkId::new("100B_string_payload", dims),
            &n_hits,
            |b, _| b.iter(|| db.fetch_payloads(&ids).unwrap()),
        );
    }

    group.finish();
}

fn bench_payload_fetch_static(c: &mut Criterion) {
    let bench_name = "payload_fetch_static";
    if !bench_filter_allows(bench_name) {
        return;
    }

    let n_vectors = cpu_bench_n();
    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    let score = 42f32;
    for &dims in &[64usize, 128usize] {
        let expected_dist = expected_distance(dims);
        let d = expected_dist;
        let lambda = expected_dist * 0.0005;

        let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
        let mut buf = vec![0.0f32; dims];

        let mut db: RingDb<StaticPayload> = RingDb::new(RingDbConfig::new(dims)).unwrap();
        for _ in 0..n_vectors {
            for x in buf.iter_mut() {
                *x = rng.random_range(-1.0f32..1.0);
            }
            db.add_vector(&buf, StaticPayload { score }).unwrap();
        }
        let db = db.build().unwrap();

        let query = random_query(dims);

        let result = db.query(&RingQuery::new(&query, d, lambda)).unwrap();
        let ids = result.ids();
        let n_hits = ids.len();

        println!(
            "dims={dims} ring=[{:.3}, {:.3}] hits={n_hits} ({:.2}%)",
            d - lambda,
            d + lambda,
            100.0 * n_hits as f32 / n_vectors.max(1) as f32
        );

        group.throughput(Throughput::Elements(n_hits as u64));
        group.bench_with_input(
            BenchmarkId::new("100B_string_payload", dims),
            &n_hits,
            |b, _| b.iter(|| db.fetch_pods(&ids)),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cpu_f32,
    bench_cpu_disk_f32,
    bench_cpu_disk_intersection_f32,
    bench_cpu_weighted_intersection_scaling,
    bench_cuda_f32,
    bench_cuda_disk_f32,
    bench_cuda_disk_intersection_f32,
    bench_payload_fetch_dynamic,
    bench_payload_fetch_static
);
criterion_main!(benches);
