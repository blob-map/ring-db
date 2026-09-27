//! Compare synchronous CPU/Vulkan query latency on identical seeded data.
//! Run backends sequentially with CPU-reference checks outside the timed calls.
use std::error::Error;
use std::fs::{File, create_dir_all};
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ash::{Entry, vk};
use clap::{Parser, ValueEnum};
use rand::{RngExt, SeedableRng};
use ringdb::{
    BackendPreference, CandidateSet, DiskIntersectionQuery, DiskQuery, Hit, KnnQuery, QueryResult,
    RingDb, RingDbConfig, RingQuery, SealedRingDb,
};

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum Backend {
    #[default]
    Both,
    Cpu,
    Vulkan,
}

#[derive(Debug, Parser)]
struct Args {
    /// Run both backends with parity checks, or one backend for isolated profiling.
    #[arg(long, value_enum, default_value_t = Backend::Both)]
    backend: Backend,
    #[arg(long, default_value_t = 10_000_000)]
    n: usize,
    #[arg(long, default_value_t = 64)]
    dims: usize,
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// Timed calls per workload per backend per round.
    #[arg(long, default_value_t = 20)]
    samples: usize,
    /// Untimed warm-up calls per workload before sampling.
    #[arg(long, default_value_t = 4)]
    warmup: usize,
    /// Each round runs both backends; their order reverses each round.
    #[arg(long, default_value_t = 2)]
    rounds: usize,
    /// Number of distinct query vectors, rotated during sampling.
    #[arg(long, default_value_t = 4)]
    queries: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Scale geometric-query radii to vary the fraction of returned vectors.
    #[arg(long, default_value_t = 1.0)]
    radius_scale: f32,
    /// Per-coordinate displacement of the other two intersection centers.
    #[arg(long, default_value_t = 0.01)]
    intersection_shift: f32,
    /// Optional single workload (ring, disk, intersection3, knn, weighted_knn, filtered_knn).
    #[arg(long)]
    workload: Option<String>,
    #[arg(long, default_value = "target/backend-compare.csv")]
    output: PathBuf,
    /// Print Vulkan device capabilities without allocating a dataset.
    #[arg(long)]
    probe: bool,
    // Cargo passes --bench to custom benchmark executables.
    #[arg(long, hide = true)]
    bench: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Workload {
    Ring,
    Disk,
    Intersection,
    Knn,
    WeightedKnn,
    FilteredKnn,
}

impl Workload {
    const ALL: [Self; 6] = [
        Self::Ring,
        Self::Disk,
        Self::Intersection,
        Self::Knn,
        Self::WeightedKnn,
        Self::FilteredKnn,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Ring => "ring",
            Self::Disk => "disk",
            Self::Intersection => "intersection3",
            Self::Knn => "knn",
            Self::WeightedKnn => "weighted_knn",
            Self::FilteredKnn => "filtered_knn",
        }
    }

    fn ordered(self) -> bool {
        matches!(self, Self::Knn | Self::WeightedKnn | Self::FilteredKnn)
    }
}

struct Query {
    point: Vec<f32>,
    positive: Vec<f32>,
    negative: Vec<f32>,
}

struct Inputs {
    queries: Vec<Query>,
    weights: Vec<f32>,
    radius: f32,
    candidates: CandidateSet,
    k: usize,
}

impl Inputs {
    fn new(db: &SealedRingDb, args: &Args) -> BenchResult<Self> {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(args.seed.wrapping_add(1));
        let queries = (0..args.queries)
            .map(|_| {
                let point: Vec<f32> = (0..args.dims)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect();
                Query {
                    positive: point.iter().map(|x| x + args.intersection_shift).collect(),
                    negative: point.iter().map(|x| x - args.intersection_shift).collect(),
                    point,
                }
            })
            .collect();
        Ok(Self {
            queries,
            weights: (0..args.dims)
                .map(|i| [0.25, 0.75, 1.25, 1.75][i % 4])
                .collect(),
            radius: (2.0 * args.dims as f32 / 3.0).sqrt() * args.radius_scale,
            // A reusable 1% candidate set, built outside query timing.
            candidates: db.candidates_from_ids((0..args.n as u32).step_by(100))?,
            k: args.k,
        })
    }

    fn run(&self, db: &SealedRingDb, workload: Workload, index: usize) -> BenchResult<QueryResult> {
        let q = &self.queries[index % self.queries.len()];
        Ok(match workload {
            Workload::Ring => {
                db.query(&RingQuery::new(&q.point, self.radius, self.radius * 0.0005))?
            }
            Workload::Disk => db.query_disk(&DiskQuery::new(&q.point, self.radius))?,
            Workload::Intersection => {
                let disks = [
                    DiskQuery::new(&q.point, self.radius),
                    DiskQuery::new(&q.positive, self.radius),
                    DiskQuery::new(&q.negative, self.radius),
                ];
                db.query_disk_intersection(&DiskIntersectionQuery::new(&disks))?
            }
            Workload::Knn => db.query_knn(&KnnQuery::new(&q.point, self.k))?,
            Workload::WeightedKnn => {
                db.query_knn(&KnnQuery::new(&q.point, self.k).with_weights(&self.weights))?
            }
            Workload::FilteredKnn => {
                db.query_knn(&KnnQuery::new(&q.point, self.k).with_candidates(&self.candidates))?
            }
        })
    }
}

fn build(
    args: &Args,
    backend: BackendPreference,
) -> BenchResult<(SealedRingDb, Duration, Duration)> {
    let mut db = RingDb::new(RingDbConfig::new(args.dims).with_backend_preference(backend))?;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(args.seed);
    let mut row = vec![0.0; args.dims];
    let started = Instant::now();
    for _ in 0..args.n {
        for x in &mut row {
            *x = rng.random_range(-1.0..1.0);
        }
        db.add_vector(&row, ())?;
    }
    let generate = started.elapsed();
    let started = Instant::now();
    let sealed = db.build()?;
    Ok((sealed, generate, started.elapsed()))
}

struct BoundaryDifference {
    id: u32,
    workload: Workload,
    query: usize,
}

fn verify(
    reference: &[Hit],
    actual: &[Hit],
    workload: Workload,
    query: usize,
    boundaries: &mut Vec<BoundaryDifference>,
) -> BenchResult<()> {
    if workload.ordered() && reference.len() != actual.len() {
        return Err(format!(
            "{} hit-count mismatch: {} vs {}",
            workload.name(),
            reference.len(),
            actual.len()
        )
        .into());
    }
    let (mut i, mut j) = (0, 0);
    while i < reference.len() || j < actual.len() {
        let a = reference.get(i);
        let b = actual.get(j);
        if !workload.ordered() && a.map(|h| h.id) != b.map(|h| h.id) {
            let id = match (a, b) {
                (Some(a), Some(b)) if a.id < b.id => {
                    i += 1;
                    a.id
                }
                (_, Some(b)) => {
                    j += 1;
                    b.id
                }
                (Some(a), None) => {
                    i += 1;
                    a.id
                }
                _ => unreachable!(),
            };
            boundaries.push(BoundaryDifference {
                id,
                workload,
                query,
            });
            continue;
        }
        let (a, b) = (a.unwrap(), b.unwrap());
        if a.id != b.id
            || !a.dist_sq.is_finite()
            || !b.dist_sq.is_finite()
            || (a.dist_sq - b.dist_sq).abs() > 2e-5 * a.dist_sq.abs().max(1.0)
        {
            return Err(format!("{} result mismatch: {a:?} vs {b:?}", workload.name()).into());
        }
        i += 1;
        j += 1;
    }
    Ok(())
}

fn verify_boundaries(
    args: &Args,
    inputs: &Inputs,
    differences: &mut [BoundaryDifference],
) -> BenchResult<()> {
    if differences.is_empty() {
        return Ok(());
    }
    differences.sort_by_key(|d| d.id);
    // Regenerate rows once in ID order instead of retaining a second 2.56 GB
    // dataset. This untimed scalar f64 check only allows membership differences
    // immediately at a distance cutoff, never arbitrary missing/extra hits.
    let mut rng = rand::rngs::SmallRng::seed_from_u64(args.seed);
    let mut row = vec![0.0f32; args.dims];
    let mut index = 0;
    let last_id = differences.last().unwrap().id;
    let mut largest_gap = 0.0f64;
    for id in 0..=last_id {
        for value in &mut row {
            *value = rng.random_range(-1.0..1.0);
        }
        while index < differences.len() && differences[index].id == id {
            let difference = &differences[index];
            let query = &inputs.queries[difference.query];
            let distance = |point: &[f32]| -> f64 {
                row.iter()
                    .zip(point)
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                    .sum()
            };
            let d = distance(&query.point);
            let radius = inputs.radius;
            let upper = f64::from(radius * radius);
            let gap = match difference.workload {
                Workload::Ring => {
                    let lower = (radius - radius * 0.0005).max(0.0);
                    let upper = radius + radius * 0.0005;
                    (d - f64::from(lower * lower))
                        .abs()
                        .min((d - f64::from(upper * upper)).abs())
                }
                Workload::Disk => (d - upper).abs(),
                Workload::Intersection => {
                    let distances = [d, distance(&query.positive), distance(&query.negative)];
                    if distances.iter().any(|d| *d > upper + 2e-5 * upper.max(1.0)) {
                        return Err(
                            format!("intersection mismatch for id {id} is outside a disk").into(),
                        );
                    }
                    distances
                        .into_iter()
                        .map(|d| (d - upper).abs())
                        .fold(f64::INFINITY, f64::min)
                }
                _ => unreachable!(),
            };
            if gap > 2e-5 * upper.max(1.0) {
                return Err(format!(
                    "{} id {id} differs away from a boundary (gap={gap})",
                    difference.workload.name()
                )
                .into());
            }
            largest_gap = largest_gap.max(gap);
            index += 1;
        }
    }
    println!(
        "Validated {} geometric boundary membership differences with scalar f64; largest squared-distance gap={largest_gap:.9}",
        differences.len()
    );
    Ok(())
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let pos = (sorted.len() - 1) as f64 * p;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

fn report(samples: &[f64]) -> String {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    format!(
        "median={:.3} ms, p95={:.3} ms, min={:.3} ms ({} calls)",
        percentile(&sorted, 0.5),
        percentile(&sorted, 0.95),
        sorted[0],
        sorted.len()
    )
}

fn vulkan_devices() -> BenchResult<()> {
    // This read-only probe creates no device or buffers.
    let entry = unsafe { Entry::load()? };
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )?
    };
    let result = (|| -> BenchResult<()> {
        for physical in unsafe { instance.enumerate_physical_devices()? } {
            let props = unsafe { instance.get_physical_device_properties(physical) };
            let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) };
            println!(
                "Vulkan device: {} (vendor={:#x}, device={:#x}, API={}.{}.{})",
                name.to_string_lossy(),
                props.vendor_id,
                props.device_id,
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version),
            );
            if props.api_version < vk::API_VERSION_1_2 {
                println!("Unsupported: Vulkan 1.2 or newer is required.");
                continue;
            }
            let mut address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut address);
            unsafe { instance.get_physical_device_features2(physical, &mut features) };
            let int64 = features.features.shader_int64;
            let mut allocation = vk::PhysicalDeviceMaintenance3Properties::default();
            let mut properties =
                vk::PhysicalDeviceProperties2::default().push_next(&mut allocation);
            unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
            println!(
                "bufferDeviceAddress={}, shaderInt64={}, maxMemoryAllocationSize={} bytes",
                address.buffer_device_address == vk::TRUE,
                int64 == vk::TRUE,
                allocation.max_memory_allocation_size
            );
        }
        Ok(())
    })();
    unsafe {
        instance.destroy_instance(None);
    }
    result
}

fn main() -> BenchResult<()> {
    let args = Args::parse();
    if args.probe || args.backend != Backend::Cpu {
        vulkan_devices()?;
    }
    if args.probe {
        return Ok(());
    }
    if args.n == 0
        || args.n > u32::MAX as usize
        || args.dims == 0
        || args.samples == 0
        || args.queries == 0
        || args.rounds == 0
        || args.k == 0
    {
        return Err("n, dims, k, samples, queries, rounds must be positive; n must fit u32".into());
    }
    if !args.radius_scale.is_finite()
        || args.radius_scale < 0.0
        || !args.intersection_shift.is_finite()
        || args.intersection_shift < 0.0
        || !((2.0 * args.dims as f32 / 3.0).sqrt() * args.radius_scale).is_finite()
    {
        return Err("radius scale and intersection shift must be finite and non-negative, with a finite radius".into());
    }
    let workloads: Vec<_> = Workload::ALL
        .into_iter()
        .filter(|w| args.workload.as_ref().is_none_or(|name| name == w.name()))
        .collect();
    if workloads.is_empty() {
        return Err("unknown --workload".into());
    }
    println!(
        "Dataset: {} × {} f32, {:.3} GiB; seed={}; k={}; queries={}; warmup={}; samples={}; rounds={}",
        args.n,
        args.dims,
        args.n as f64 * args.dims as f64 * 4.0 / 1073741824.0,
        args.seed,
        args.k,
        args.queries,
        args.warmup,
        args.samples,
        args.rounds
    );
    println!(
        "Times include synchronous public query calls and result transfers; exclude generation, upload, validation, and result destruction."
    );
    println!(
        "Geometric radius scale={}; intersection shift={}",
        args.radius_scale, args.intersection_shift
    );
    if let Ok(bytes) = std::env::var("RINGDB_VULKAN_CHUNK_BYTES") {
        println!("Vulkan vector-buffer byte cap: {bytes}");
    }
    if args.backend != Backend::Both {
        println!("Single-backend run: cross-backend correctness checks are disabled.");
    }
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        create_dir_all(parent)?;
    }
    let mut output = BufWriter::new(File::create(&args.output)?);
    writeln!(
        output,
        "backend,round,workload,sample,query,n,dims,k,hits,elapsed_ms"
    )?;
    // Keep the first backend's answers on the host for untimed parity checks.
    let mut references: Vec<Vec<Vec<Hit>>> = (0..workloads.len()).map(|_| Vec::new()).collect();
    let mut totals = vec![vec![Vec::new(); workloads.len()]; 2];
    for round in 0..args.rounds {
        let backends = if round % 2 == 0 {
            [BackendPreference::Cpu, BackendPreference::Vulkan]
        } else {
            [BackendPreference::Vulkan, BackendPreference::Cpu]
        };
        for backend in backends {
            if (args.backend == Backend::Cpu && backend != BackendPreference::Cpu)
                || (args.backend == Backend::Vulkan && backend != BackendPreference::Vulkan)
            {
                continue;
            }
            let name = if backend == BackendPreference::Cpu {
                "cpu"
            } else {
                "vulkan"
            };
            let backend_index = usize::from(backend == BackendPreference::Vulkan);
            println!("\nRound {}: building {name} dataset...", round + 1);
            std::io::stdout().flush()?;
            let (db, generation, upload) = build(&args, backend)?;
            println!(
                "{name}: generation/insertion={:.3}s, seal/upload={:.3}s",
                generation.as_secs_f64(),
                upload.as_secs_f64()
            );
            let inputs = Inputs::new(&db, &args)?;
            let mut boundaries = Vec::new();
            for (workload_index, &workload) in workloads.iter().enumerate() {
                let reference = &mut references[workload_index];
                let first = reference.is_empty();
                let mut query0_hits = 0;
                for query in 0..args.queries {
                    let mut result = inputs.run(&db, workload, query)?;
                    if query == 0 {
                        query0_hits = result.hits.len();
                    }
                    if !workload.ordered() {
                        result.hits.sort_unstable_by_key(|h| h.id);
                    }
                    if first {
                        reference.push(result.hits);
                    } else {
                        verify(
                            &reference[query],
                            &result.hits,
                            workload,
                            query,
                            &mut boundaries,
                        )?;
                    }
                }
                for i in 0..args.warmup {
                    black_box(inputs.run(&db, workload, i)?);
                }
                let mut samples = Vec::with_capacity(args.samples);
                for sample in 0..args.samples {
                    let query = sample % args.queries;
                    let start = Instant::now();
                    let result = black_box(inputs.run(&db, workload, query)?);
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    writeln!(
                        output,
                        "{name},{},{},{sample},{query},{},{},{},{},{elapsed:.6}",
                        round + 1,
                        workload.name(),
                        args.n,
                        args.dims,
                        args.k,
                        result.hits.len()
                    )?;
                    samples.push(elapsed);
                }
                output.flush()?;
                println!(
                    "{name}/{}, hits(query0)={}: {}",
                    workload.name(),
                    query0_hits,
                    report(&samples)
                );
                std::io::stdout().flush()?;
                totals[backend_index][workload_index].extend(samples);
            }
            verify_boundaries(&args, &inputs, &mut boundaries)?;
            // db and inputs are dropped before the next backend is constructed.
        }
    }
    println!("\nCombined results:");
    for (i, workload) in workloads.iter().enumerate() {
        let cpu = &mut totals[0][i];
        if cpu.is_empty() {
            println!("vulkan/{}: {}", workload.name(), report(&totals[1][i]));
            continue;
        }
        cpu.sort_by(f64::total_cmp);
        let cpu_median = percentile(cpu, 0.5);
        println!("cpu/{}: {}", workload.name(), report(cpu));
        let vulkan = &mut totals[1][i];
        if vulkan.is_empty() {
            continue;
        }
        vulkan.sort_by(f64::total_cmp);
        let vulkan_median = percentile(vulkan, 0.5);
        println!(
            "vulkan/{}: {}; CPU median / Vulkan median={:.3}x (>1 means Vulkan is faster)",
            workload.name(),
            report(vulkan),
            cpu_median / vulkan_median
        );
    }
    println!("Raw samples: {}", args.output.display());
    Ok(())
}
