/// All errors that ringdb can produce.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum RingDbError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("corrupt database: {0}")]
    Corrupt(String),

    #[error("invalid query: {0}")]
    InvalidQuery(String),

    #[error("vector ID {id} is out of bounds for a database with {len} vectors")]
    InvalidVectorId { id: u32, len: usize },

    #[error("candidate set has universe length {got}, expected {expected}")]
    CandidateSetMismatch { expected: usize, got: usize },

    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("backend error: {0}")]
    Backend(String),
}

pub type Result<T> = std::result::Result<T, RingDbError>;
