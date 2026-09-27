use std::path::PathBuf;

/// Which compute backend to use for vector queries.
///
/// Both backends are included in every build; selection happens at runtime.
/// CPU needs no GPU runtime. Vulkan requires a compatible loader, driver, and device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BackendPreference {
    /// CPU brute-force (always available).
    #[default]
    Cpu,

    /// Vulkan compute, preferring a discrete GPU, then an integrated GPU.
    ///
    /// Loads Vulkan only when selected. Requires a Vulkan loader and a compatible
    /// driver and compute device. Otherwise construction returns
    /// [`RingDbError::BackendUnavailable`](crate::RingDbError::BackendUnavailable).
    /// No Cargo feature, Vulkan SDK, or shader compiler is needed.
    Vulkan,
}

/// Configuration for a `RingDb` instance.
#[derive(Debug, Clone)]
pub struct RingDbConfig {
    /// Number of dimensions per vector. Must be > 0.
    pub dims: usize,
    /// Backend selection strategy.
    pub backend_preference: BackendPreference,
    /// Optional directory for persisting the database.
    ///
    /// When set, [`RingDb::build()`](crate::engine::RingDb::build) writes the
    /// full database (vectors, Pod payloads, and metadata indexes) to this
    /// directory. The sealed database can later be reloaded with
    /// [`RingDb::load()`](crate::engine::RingDb::load).
    ///
    /// Leave `None` (the default) for a purely in-memory database.
    pub persist_dir: Option<PathBuf>,
}

impl RingDbConfig {
    /// Create a config with default settings (CPU backend, no persistence).
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            backend_preference: BackendPreference::Cpu,
            persist_dir: None,
        }
    }

    /// Set the directory to which the database will be persisted on
    /// [`build()`](crate::engine::RingDb::build).
    ///
    /// The directory is created automatically if it does not exist and must be
    /// empty when it already exists.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ringdb::RingDbConfig;
    ///
    /// let config = RingDbConfig::new(128).with_persist_dir("/var/data/mydb");
    /// ```
    #[must_use]
    pub fn with_persist_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.persist_dir = Some(dir.into());
        self
    }

    /// Set the backend preference.
    ///
    /// Defaults to [`BackendPreference::Cpu`]. Use this when loading a
    /// persisted database onto a specific backend via [`RingDb::load()`](crate::engine::RingDb::load).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ringdb::RingDbConfig;
    /// use ringdb::BackendPreference;
    ///
    /// let config = RingDbConfig::new(128).with_backend_preference(BackendPreference::Cpu);
    /// ```
    #[must_use]
    pub fn with_backend_preference(mut self, preference: BackendPreference) -> Self {
        self.backend_preference = preference;
        self
    }
}
