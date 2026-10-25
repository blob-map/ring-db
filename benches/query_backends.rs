//! Criterion benchmarks of every search shape on each available backend.
//!
//! `RINGDB_BENCH_N` sets the row count (default 30M); `RINGDB_VULKAN_BENCH_N`
//! overrides it for Vulkan. Positional arguments filter benchmark groups.
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng};
use ringdb::{
    BackendPreference, Db, Disk, FieldType, Filter, Knn, Metric, Options, Record, Region, Schema,
};
use std::time::Duration;

const N: usize = 30_000_000;
const SEED: u64 = 42;
const DIMS: &[usize] = &[16, 32, 64, 128];

const SELECTIVITIES: [(&str, u64); 3] = [
    ("10_percent", 10),
    ("1_percent", 100),
    ("0.1_percent", 1000),
];

fn rows(name: &str) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(N)
}

fn backends() -> Vec<(&'static str, BackendPreference, usize)> {
    let mut backends = vec![("cpu", BackendPreference::Cpu, rows("RINGDB_BENCH_N"))];
    match Db::in_memory(
        Schema::new(1),
        Options::default().backend(BackendPreference::Vulkan),
    ) {
        Ok(_) => {
            let n = std::env::var_os("RINGDB_VULKAN_BENCH_N")
                .map_or(backends[0].2, |_| rows("RINGDB_VULKAN_BENCH_N"));
            backends.push(("vulkan", BackendPreference::Vulkan, n));
        }
        Err(error) => eprintln!("skipping Vulkan benchmarks: {error}"),
    }
    backends
}

fn allowed(group: &str) -> bool {
    let filters: Vec<String> = std::env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with('-'))
        .collect();
    filters.is_empty()
        || filters
            .iter()
            .any(|f| group.contains(f.as_str()) || f.contains(group))
}

/// Uniform rows in [-1, 1). Row `i` has `id = i`, and the keyword `"{every}"`
/// in `tags` for each selectivity that divides `i`.
fn build(dims: usize, n: usize, backend: BackendPreference) -> Db {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
    let mut row = vec![0.0f32; dims];
    let schema = Schema::new(dims)
        .field("id", FieldType::U64)
        .field("tags", FieldType::Keywords);
    let options = Options::default().backend(backend).commit_interval(None);
    let db = Db::in_memory(schema, options).unwrap();
    for id in 0..n as u64 {
        for x in &mut row {
            *x = rng.random_range(-1.0f32..1.0);
        }
        let tags: Vec<String> = SELECTIVITIES
            .iter()
            .filter(|(_, every)| id % every == 0)
            .map(|(_, every)| every.to_string())
            .collect();
        db.add(Record::new(row.as_slice()).set("id", id).set("tags", tags))
            .unwrap();
    }
    db.flush().unwrap();
    db
}

fn query(dims: usize, shift: f32) -> Vec<f32> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
    (0..dims)
        .map(|_| rng.random_range(-1.0f32..1.0) + shift)
        .collect()
}

/// Weights with mean one, so radii stay comparable with ordinary L2.
fn weights(dims: usize, phase: usize) -> Vec<f32> {
    const PATTERN: [f32; 4] = [0.25, 0.75, 1.25, 1.75];
    (0..dims).map(|i| PATTERN[(i + phase) % 4]).collect()
}

/// Expected distance between two uniform rows: a radius selecting about half.
fn radius(dims: usize) -> f32 {
    (2.0 * dims as f32 / 3.0).sqrt()
}

fn group<'a>(
    c: &'a mut Criterion,
    name: &str,
    n: usize,
) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);
    group.throughput(Throughput::Elements(n as u64));
    group
}

fn bench_regions(c: &mut Criterion) {
    for (backend, preference, n) in backends() {
        let name = format!("{backend}_region");
        if !allowed(&name) {
            continue;
        }
        let mut group = group(c, &name, n);
        for &dims in DIMS {
            let db = build(dims, n, preference);
            let centers = [query(dims, 0.0), query(dims, 0.01), query(dims, -0.01)];
            let weights = [weights(dims, 0), weights(dims, 1), weights(dims, 2)];
            let r = radius(dims);
            let disk = |i: usize, weighted: bool| {
                let disk = Disk::new(&centers[i], r);
                if weighted {
                    disk.metric(Metric::WeightedL2(&weights[i]))
                } else {
                    disk
                }
            };
            let cases: [(&str, Region<'_>); 5] = [
                ("disk", disk(0, false).into()),
                ("weighted_disk", disk(0, true).into()),
                (
                    "intersection3",
                    disk(0, false).and(disk(1, false)).and(disk(2, false)),
                ),
                (
                    "weighted_intersection3",
                    disk(0, true).and(disk(1, true)).and(disk(2, true)),
                ),
                (
                    "mixed_intersection3",
                    disk(0, true).and(disk(1, false)).and(disk(2, true)),
                ),
            ];
            for (label, region) in cases {
                group.bench_with_input(BenchmarkId::new(label, dims), &dims, |b, _| {
                    b.iter(|| db.search(region.clone()).unwrap());
                });
            }
        }
        group.finish();
    }
}

fn bench_intersection_scaling(c: &mut Criterion) {
    const DIMS: usize = 64;
    let n = rows("RINGDB_BENCH_N");
    if !allowed("cpu_intersection_scaling") {
        return;
    }
    let db = build(DIMS, n, BackendPreference::Cpu);
    let base = query(DIMS, 0.0);
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|i| {
            let shift = (i as f32 - 3.5) * 0.005;
            base.iter()
                .enumerate()
                .map(|(d, &x)| if d % 2 == 0 { x + shift } else { x - shift })
                .collect()
        })
        .collect();
    let weights: Vec<Vec<f32>> = (0..8).map(|i| weights(DIMS, i)).collect();
    let r = radius(DIMS);
    // The first disk is narrow; putting it last makes rejection happen late.
    let disk = |i: usize| {
        let scale = if i == 0 { 0.75 } else { 1.35 };
        Disk::new(&centers[i], r * scale).metric(Metric::WeightedL2(&weights[i]))
    };
    let mut group = group(c, "cpu_intersection_scaling", n);
    for count in [1usize, 2, 4, 8] {
        let region = |order: Vec<usize>| {
            order[1..]
                .iter()
                .fold(Region::from(disk(order[0])), |region, &i| {
                    region.and(disk(i))
                })
        };
        let early = region((0..count).collect());
        let late = region((1..count).chain([0]).collect());
        group.bench_with_input(BenchmarkId::new("early_reject", count), &count, |b, _| {
            b.iter(|| db.search(early.clone()).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("late_reject", count), &count, |b, _| {
            b.iter(|| db.search(late.clone()).unwrap());
        });
    }
    group.finish();
}

fn bench_knn(c: &mut Criterion) {
    const DIMS: usize = 64;
    for (backend, preference, n) in backends() {
        let name = format!("{backend}_knn");
        if !allowed(&name) {
            continue;
        }
        let db = build(DIMS, n, preference);
        let point = query(DIMS, 0.0);
        let weights = weights(DIMS, 0);
        let mut group = group(c, &name, n);
        for k in [1usize, 10, 100] {
            group.bench_with_input(BenchmarkId::new("ordinary", k), &k, |b, &k| {
                b.iter(|| db.search(Knn::new(&point, k)).unwrap());
            });
            group.bench_with_input(BenchmarkId::new("weighted", k), &k, |b, &k| {
                b.iter(|| {
                    db.search(Knn::new(&point, k).metric(Metric::WeightedL2(&weights)))
                        .unwrap()
                });
            });
        }
        for (label, every) in SELECTIVITIES {
            let keyword = Filter::eq("tags", every.to_string());
            group.bench_function(BenchmarkId::new("keyword_filter", label), |b| {
                b.iter(|| db.search(Knn::new(&point, 10).filter(&keyword)).unwrap());
            });
            let range = Filter::range("id", ..n as u64 / every);
            group.bench_function(BenchmarkId::new("range_filter", label), |b| {
                b.iter(|| db.search(Knn::new(&point, 10).filter(&range)).unwrap());
            });
        }
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_regions,
    bench_intersection_scaling,
    bench_knn
);
criterion_main!(benches);
