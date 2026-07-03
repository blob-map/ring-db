/// All errors that ringdb can produce.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum RingDbError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("payload serialization error: {0}")]
    Payload(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("corrupt database: {0}")]
    Corrupt(String),

    #[error("invalid query: {0}")]
    InvalidQuery(String),

    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error(
        "storage mode mismatch: use fetch_pod() for Pod storage and fetch_payload() for Serde storage"
    )]
    StorageMismatch,
}

pub type Result<T> = std::result::Result<T, RingDbError>;
