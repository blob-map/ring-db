use bytemuck::{Pod, Zeroable};
use clap::{Parser, ValueEnum};
use rand::{RngExt, SeedableRng};
use ringdb::{
    BackendPreference, KnnQuery, MetadataKey, MetadataTag, MetadataValue, RingDb, RingDbConfig,
};

const CATEGORY: MetadataKey = MetadataKey::new(1);

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Record {
    source_id: u32,
    category_id: u32,
}

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
    Cuda,
}

impl From<CliBackend> for BackendPreference {
    fn from(value: CliBackend) -> Self {
        match value {
            CliBackend::Cpu => Self::Cpu,
            CliBackend::Cuda => Self::Cuda,
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
    if cli.n > u32::MAX as usize {
        return Err("--n exceeds ring-db's u32 vector-ID limit".into());
    }
    if !cli.first_dim_weight.is_finite() || cli.first_dim_weight < 0.0 {
        return Err("--first-dim-weight must be finite and non-negative".into());
    }

    println!("ring-db filtered KNN demo");
    println!("  dataset    : {} vectors × {} dims", cli.n, cli.dims);
    println!("  filter     : category = {}", cli.category);
    println!("  neighbors  : k = {}", cli.k);
    println!("  dim-0 weight: {}", cli.first_dim_weight);

    let config = RingDbConfig::new(cli.dims).with_backend_preference(cli.backend.into());
    let mut db: RingDb<Record> = RingDb::new(config)?;
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
        let source_id = source_id as u32;
        let category_id = source_id % cli.categories;
        let tag = MetadataTag::new(CATEGORY, MetadataValue::new(u64::from(category_id)));
        db.add_vector_with_metadata(
            &vector,
            Record {
                source_id,
                category_id,
            },
            &[tag],
        )?;
    }
    let db = db.build()?;
    println!("  build       : {:.2?}", build_started.elapsed());
    println!("  backend     : {}", db.backend_name());

    let candidates = db.metadata_eq(CATEGORY, MetadataValue::new(u64::from(cli.category)));
    let mut weights = vec![1.0f32; cli.dims];
    weights[0] = cli.first_dim_weight;
    let result = db.query_knn(
        &KnnQuery::new(&query, cli.k)
            .with_weights(&weights)
            .with_candidates(&candidates),
    )?;

    println!("  candidates  : {}", candidates.len());
    println!("  query time  : {:.2?}", result.elapsed);
    println!("  hits        : {}", result.hits.len());
    for hit in result.hits.iter().take(20) {
        let record = db.fetch_payload(hit.id)?;
        println!(
            "    id={} source={} category={} distance={:.6}",
            hit.id,
            record.source_id,
            record.category_id,
            hit.dist_sq.sqrt()
        );
    }
    Ok(())
}
