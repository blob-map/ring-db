use bytemuck::{Pod, Zeroable};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng};
#[cfg(feature = "cuda")]
use ringdb::{BackendPreference, RingDbError, SealedRingDb};
use ringdb::{DiskIntersectionQuery, DiskQuery, Payload, RingDb, RingDbConfig, RingQuery};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const N: usize = 30_000_000;
const SEED: u64 = 42;

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
            ["cpu", "cuda", "disk", "payload", "f32"]
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

#[cfg(feature = "cuda")]
fn cuda_bench_n() -> usize {
    std::env::var("RINGDB_CUDA_BENCH_N")
        .ok()
        .and_then(|raw| match raw.parse::<usize>() {
            Ok(n) => Some(n),
            Err(err) => {
                eprintln!("ignoring invalid RINGDB_CUDA_BENCH_N value: {err}; using default {N}");
                None
            }
        })
        .unwrap_or(N)
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

#[cfg(feature = "cuda")]
fn build_cuda_unit_db(
    dims: usize,
    n_vectors: usize,
) -> std::result::Result<SealedRingDb<()>, RingDbError> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
    let mut buf = vec![0.0f32; dims];
    let mut db =
        RingDb::new(RingDbConfig::new(dims).with_backend_preference(BackendPreference::Cuda))?;

    for _ in 0..n_vectors {
        for x in buf.iter_mut() {
            *x = rng.random_range(-1.0f32..1.0);
        }
        db.add_vector(&buf, ())?;
    }

    db.build()
}

#[cfg(feature = "cuda")]
fn random_query(dims: usize) -> Vec<f32> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
    (0..dims).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

fn bench_cpu_f32(c: &mut Criterion) {
    let bench_name = "cpu_f32";
    if !bench_filter_allows(bench_name) {
        return;
    }

    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &dims in &[64usize, 128usize] {
        // Expected L2 distance between random vectors in [-1,1]^d is sqrt(2d/3).
        // 64 dims → ~6.5, 128 dims → ~9.2
        let expected_dist = ((2.0 * dims as f32) / 3.0).sqrt();
        let d = expected_dist;
        let lambda = expected_dist * 0.0005; // .05% ring width → guaranteed hits

        let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
        let mut buf = vec![0.0f32; dims];

        let mut db = RingDb::new(RingDbConfig::new(dims)).unwrap();
        for _ in 0..N {
            for x in buf.iter_mut() {
                *x = rng.random_range(-1.0f32..1.0);
            }
            db.add_vector(&buf, ()).unwrap();
        }
        let db = db.build().unwrap();

        let query: Vec<f32> = {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
            (0..dims).map(|_| rng.random_range(-1.0f32..1.0)).collect()
        };

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| {
                db.query(&RingQuery {
                    query: &query,
                    d,
                    lambda,
                })
                .unwrap()
            });
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

    for &dims in &[64usize, 128usize] {
        let expected_dist = ((2.0 * dims as f32) / 3.0).sqrt();
        let d = expected_dist;
        let lambda = expected_dist * 0.0005;

        let db = match build_cuda_unit_db(dims, n_vectors) {
            Ok(db) => db,
            Err(err) => {
                eprintln!("skipping {bench_name}/{dims}: failed to build CUDA db: {err}");
                continue;
            }
        };
        let query = random_query(dims);

        match db.query(&RingQuery {
            query: &query,
            d,
            lambda,
        }) {
            Ok(result) => println!(
                "{bench_name}/{dims}: n={n_vectors} hits={} backend={}",
                result.hits.len(),
                result.backend_used
            ),
            Err(err) => {
                eprintln!("skipping {bench_name}/{dims}: warmup query failed: {err}");
                continue;
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| {
                db.query(&RingQuery {
                    query: &query,
                    d,
                    lambda,
                })
                .unwrap()
            });
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

    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &dims in &[64usize, 128usize] {
        // Use the same expected distance as the ring bench so hit counts are comparable.
        let d_max = ((2.0 * dims as f32) / 3.0).sqrt();

        let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
        let mut buf = vec![0.0f32; dims];

        let mut db = RingDb::new(RingDbConfig::new(dims)).unwrap();
        for _ in 0..N {
            for x in buf.iter_mut() {
                *x = rng.random_range(-1.0f32..1.0);
            }
            db.add_vector(&buf, ()).unwrap();
        }
        let db = db.build().unwrap();

        let query: Vec<f32> = {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
            (0..dims).map(|_| rng.random_range(-1.0f32..1.0)).collect()
        };

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| {
                db.query_disk(&DiskQuery {
                    query: &query,
                    d_max,
                })
                .unwrap()
            });
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

    for &dims in &[64usize, 128usize] {
        let d_max = ((2.0 * dims as f32) / 3.0).sqrt();
        let db = match build_cuda_unit_db(dims, n_vectors) {
            Ok(db) => db,
            Err(err) => {
                eprintln!("skipping {bench_name}/{dims}: failed to build CUDA db: {err}");
                continue;
            }
        };
        let query = random_query(dims);

        match db.query_disk(&DiskQuery {
            query: &query,
            d_max,
        }) {
            Ok(result) => println!(
                "{bench_name}/{dims}: n={n_vectors} hits={} backend={}",
                result.hits.len(),
                result.backend_used
            ),
            Err(err) => {
                eprintln!("skipping {bench_name}/{dims}: warmup query failed: {err}");
                continue;
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| {
                db.query_disk(&DiskQuery {
                    query: &query,
                    d_max,
                })
                .unwrap()
            });
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

    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &dims in &[64usize, 128usize] {
        // Same scale as the disk bench. The shifted centers keep the disks
        // meaningfully overlapping while still forcing every disk check.
        let d_max = ((2.0 * dims as f32) / 3.0).sqrt();

        let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
        let mut buf = vec![0.0f32; dims];

        let mut db = RingDb::new(RingDbConfig::new(dims)).unwrap();
        for _ in 0..N {
            for x in buf.iter_mut() {
                *x = rng.random_range(-1.0f32..1.0);
            }
            db.add_vector(&buf, ()).unwrap();
        }
        let db = db.build().unwrap();

        let query_a: Vec<f32> = {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
            (0..dims).map(|_| rng.random_range(-1.0f32..1.0)).collect()
        };
        let mut query_b = query_a.clone();
        for x in &mut query_b {
            *x += 0.01;
        }
        let mut query_c = query_a.clone();
        for x in &mut query_c {
            *x -= 0.01;
        }
        let disks = [
            DiskQuery {
                query: &query_a,
                d_max,
            },
            DiskQuery {
                query: &query_b,
                d_max,
            },
            DiskQuery {
                query: &query_c,
                d_max,
            },
        ];

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| {
                db.query_disk_intersection(&DiskIntersectionQuery { disks: &disks })
                    .unwrap()
            });
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

    for &dims in &[64usize, 128usize] {
        let d_max = ((2.0 * dims as f32) / 3.0).sqrt();
        let db = match build_cuda_unit_db(dims, n_vectors) {
            Ok(db) => db,
            Err(err) => {
                eprintln!("skipping {bench_name}/{dims}: failed to build CUDA db: {err}");
                continue;
            }
        };

        let query_a = random_query(dims);
        let mut query_b = query_a.clone();
        for x in &mut query_b {
            *x += 0.01;
        }
        let mut query_c = query_a.clone();
        for x in &mut query_c {
            *x -= 0.01;
        }
        let disks = [
            DiskQuery {
                query: &query_a,
                d_max,
            },
            DiskQuery {
                query: &query_b,
                d_max,
            },
            DiskQuery {
                query: &query_c,
                d_max,
            },
        ];

        match db.query_disk_intersection(&DiskIntersectionQuery { disks: &disks }) {
            Ok(result) => println!(
                "{bench_name}/{dims}: n={n_vectors} hits={} backend={}",
                result.hits.len(),
                result.backend_used
            ),
            Err(err) => {
                eprintln!("skipping {bench_name}/{dims}: warmup query failed: {err}");
                continue;
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| {
                db.query_disk_intersection(&DiskIntersectionQuery { disks: &disks })
                    .unwrap()
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn bench_cuda_disk_intersection_f32(_: &mut Criterion) {}

fn bench_payload_fetch_dynamic(c: &mut Criterion) {
    let bench_name = "payload_fetch_dynamic";
    if !bench_filter_allows(bench_name) {
        return;
    }

    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    let label: String = "x".repeat(92);

    for &dims in &[64usize, 128usize] {
        let expected_dist = ((2.0 * dims as f32) / 3.0).sqrt();
        let d = expected_dist;
        let lambda = expected_dist * 0.0005;

        let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
        let mut buf = vec![0.0f32; dims];

        let mut db: RingDb<DynamicPayload> = RingDb::new(RingDbConfig::new(dims)).unwrap();
        for _ in 0..N {
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

        let query: Vec<f32> = {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
            (0..dims).map(|_| rng.random_range(-1.0f32..1.0)).collect()
        };

        let result = db
            .query(&RingQuery {
                query: &query,
                d,
                lambda,
            })
            .unwrap();
        let ids = result.ids();
        let n_hits = ids.len();

        println!(
            "dims={dims} ring=[{:.3}, {:.3}] hits={n_hits} ({:.2}%)",
            d - lambda,
            d + lambda,
            100.0 * n_hits as f32 / N as f32
        );

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

    let mut group = c.benchmark_group(bench_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    let score = 42f32;
    for &dims in &[64usize, 128usize] {
        let expected_dist = ((2.0 * dims as f32) / 3.0).sqrt();
        let d = expected_dist;
        let lambda = expected_dist * 0.0005;

        let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
        let mut buf = vec![0.0f32; dims];

        let mut db: RingDb<StaticPayload> = RingDb::new(RingDbConfig::new(dims)).unwrap();
        for _ in 0..N {
            for x in buf.iter_mut() {
                *x = rng.random_range(-1.0f32..1.0);
            }
            db.add_vector(&buf, StaticPayload { score }).unwrap();
        }
        let db = db.build().unwrap();

        let query: Vec<f32> = {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
            (0..dims).map(|_| rng.random_range(-1.0f32..1.0)).collect()
        };

        let result = db
            .query(&RingQuery {
                query: &query,
                d,
                lambda,
            })
            .unwrap();
        let ids = result.ids();
        let n_hits = ids.len();

        println!(
            "dims={dims} ring=[{:.3}, {:.3}] hits={n_hits} ({:.2}%)",
            d - lambda,
            d + lambda,
            100.0 * n_hits as f32 / N as f32
        );

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
    bench_cuda_f32,
    bench_cuda_disk_f32,
    bench_cuda_disk_intersection_f32,
    bench_payload_fetch_dynamic,
    bench_payload_fetch_static
);
criterion_main!(benches);
