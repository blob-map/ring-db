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

    #[error("invalid schema: {0}")]
    InvalidSchema(String),

    #[error("invalid record: {0}")]
    InvalidRecord(String),

    #[error("unknown field '{0}'")]
    UnknownField(String),

    #[error("no record has key {0}")]
    KeyNotFound(u64),

    /// A write to a database opened with [`Db::open_read_only`](crate::Db::open_read_only).
    #[error("the database is open read-only")]
    ReadOnly,

    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("backend error: {0}")]
    Backend(String),
}

pub type Result<T> = std::result::Result<T, RingDbError>;
