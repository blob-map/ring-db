use clap::{Parser, ValueEnum};
use rand::{RngExt, SeedableRng};
use ringdb::{
    BackendPreference, Db, FieldType, Filter, Knn, Metric, Options, Record, Schema, Value,
};

/// Build a random database, then find exact nearest neighbors in one category.
///
/// Example:
///
///   cargo run --release --example cli -- --n 100000 --dims 64 --k 10 --category 3
#[derive(Parser)]
#[command(
    name = "ringdb-cli",
    about = "ring-db demo: filtered exact KNN with diagonal weights"
)]
struct Cli {
    /// Number of random vectors to generate.
    #[arg(short, long, default_value_t = 10_000)]
    n: usize,

    /// Number of dimensions per vector.
    #[arg(long, default_value_t = 128)]
    dims: usize,

    /// Maximum number of nearest neighbors to return.
    #[arg(short, long, default_value_t = 10)]
    k: usize,

    /// Category ID to search.
    #[arg(long, default_value_t = 0)]
    category: u32,

    /// Number of categories assigned round-robin to the generated vectors.
    #[arg(long, default_value_t = 10)]
    categories: u32,

    /// Diagonal weight for dimension zero; all other dimensions use 1.
    #[arg(long, default_value_t = 4.0)]
    first_dim_weight: f32,

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
    Vulkan,
}

impl From<CliBackend> for BackendPreference {
    fn from(value: CliBackend) -> Self {
        match value {
            CliBackend::Cpu => Self::Cpu,
            CliBackend::Vulkan => Self::Vulkan,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if cli.categories == 0 {
        return Err("--categories must be greater than zero".into());
    }
    if cli.category >= cli.categories {
        return Err("--category must be smaller than --categories".into());
    }
    if !cli.first_dim_weight.is_finite() || cli.first_dim_weight < 0.0 {
        return Err("--first-dim-weight must be finite and non-negative".into());
    }

    println!("ring-db filtered KNN demo");
    println!("  dataset    : {} vectors × {} dims", cli.n, cli.dims);
    println!("  filter     : category = {}", cli.category);
    println!("  neighbors  : k = {}", cli.k);
    println!("  dim-0 weight: {}", cli.first_dim_weight);

    let schema = Schema::new(cli.dims)
        .field("source", FieldType::U64)
        .field("category", FieldType::U32);
    let db = Db::in_memory(schema, Options::default().backend(cli.backend.into()))?;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(cli.seed);
    let mut vector = vec![0.0f32; cli.dims];
    let query: Vec<f32> = (0..cli.dims)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();

    let build_started = std::time::Instant::now();
    for source_id in 0..cli.n {
        for value in &mut vector {
            *value = rng.random_range(-1.0f32..1.0);
        }
        let category = source_id as u32 % cli.categories;
        db.add(
            Record::new(vector.as_slice())
                .set("source", source_id)
                .set("category", category),
        )?;
    }
    db.flush()?;
    println!("  build       : {:.2?}", build_started.elapsed());
    println!("  backend     : {:?}", db.backend());

    let filter = Filter::eq("category", cli.category);
    let mut weights = vec![1.0f32; cli.dims];
    weights[0] = cli.first_dim_weight;
    let started = std::time::Instant::now();
    let hits = db.search(
        Knn::new(&query, cli.k)
            .metric(Metric::WeightedL2(&weights))
            .filter(&filter),
    )?;

    println!("  query time  : {:.2?}", started.elapsed());
    println!("  hits        : {}", hits.len());
    for hit in hits.iter().take(20) {
        let doc = db.get(hit.key)?;
        let field = |name| doc.get(name).and_then(Value::as_u64).unwrap_or_default();
        println!(
            "    key={} source={} category={} distance={:.6}",
            hit.key.get(),
            field("source"),
            field("category"),
            hit.dist_sq.sqrt()
        );
    }
    Ok(())
}
