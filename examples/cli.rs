use clap::{Parser, ValueEnum};
use rand::{RngExt, SeedableRng};
use ringdb::{BackendPreference, QueryResult, RingDb, RingDbConfig, RingQuery};

/// ringdb CLI — generate a random dataset and run a ring query.
///
/// Example:
///
///   cargo run --example cli -- --n 100000 --dims 64 --d 5.0 --lambda 0.2
#[derive(Parser)]
#[command(name = "ringdb-cli", about = "ringdb demo: random dataset ring query")]
struct Cli {
    /// Number of random vectors to generate.
    #[arg(short, long, default_value_t = 10_000)]
    n: usize,

    /// Number of dimensions per vector.
    #[arg(long, default_value_t = 128)]
    dims: usize,

    /// Target distance (centre of ring).
    #[arg(short = 'd', long, default_value_t = 5.0)]
    d: f32,

    /// Half-width of the ring (λ).
    #[arg(short = 'l', long, default_value_t = 0.5)]
    lambda: f32,

    /// Random seed for reproducibility.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Compute backend to use.
    #[arg(long, value_enum, default_value_t = CliBackend::Cpu)]
    backend: CliBackend,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliBackend {
    Cpu,
    Cuda,
}

impl From<CliBackend> for BackendPreference {
    fn from(value: CliBackend) -> Self {
        match value {
            CliBackend::Cpu => BackendPreference::Cpu,
            CliBackend::Cuda => BackendPreference::Cuda,
        }
    }
}

fn main() {
    let cli = Cli::parse();

    let config = RingDbConfig::new(cli.dims).with_backend_preference(cli.backend.into());

    println!("ringdb demo");
    println!("  dataset : {} vectors × {} dims", cli.n, cli.dims);
    println!("  ring    : d={}, λ={}", cli.d, cli.lambda);
    println!();

    print!("Building database … ");
    let build_start = std::time::Instant::now();
    let mut db = RingDb::new(config).expect("failed to create RingDb");

    let mut rng = rand::rngs::SmallRng::seed_from_u64(cli.seed);
    let mut buf = vec![0.0f32; cli.dims];

    for x in buf.iter_mut() {
        *x = rng.random_range(-1.0f32..1.0);
    }
    let query_vec = buf.clone();
    db.add_vector(&buf, ()).expect("failed to add vector");

    for _ in 1..cli.n {
        for x in buf.iter_mut() {
            *x = rng.random_range(-1.0f32..1.0);
        }
        db.add_vector(&buf, ()).expect("failed to add vector");
    }

    let db = db.build().expect("failed to build");

    let build_elapsed = build_start.elapsed();
    println!("done in {:.2?}", build_elapsed);
    println!("  backend : {}", db.backend_name());
    println!();

    let rq = RingQuery::new(&query_vec, cli.d, cli.lambda);

    print!("Running ring query … ");
    let QueryResult {
        hits,
        backend_used,
        elapsed,
    } = db.query(&rq).expect("query failed");

    println!("done");
    println!();
    println!("Results:");
    println!("  backend      : {backend_used}");
    println!("  hits         : {}", hits.len());
    println!("  query time   : {elapsed:.2?}");

    if hits.len() <= 20 {
        println!(
            "  hit IDs      : {:?}",
            hits.iter().map(|h| h.id).collect::<Vec<_>>()
        );
    } else {
        println!(
            "  hit IDs      : [{}, {}, … {} total]",
            hits[0].id,
            hits[1].id,
            hits.len()
        );
    }
}
