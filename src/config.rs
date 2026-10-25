use std::time::Duration;

/// Which compute backend to use for vector searches.
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
    /// driver and compute device. Otherwise opening returns
    /// [`RingDbError::BackendUnavailable`](crate::RingDbError::BackendUnavailable).
    /// No Cargo feature, Vulkan SDK, or shader compiler is needed.
    Vulkan,
}

/// Runtime settings of a [`Db`](crate::Db).
#[derive(Debug, Clone)]
pub struct Options {
    pub(crate) backend: BackendPreference,
    pub(crate) max_record_bytes: usize,
    pub(crate) commit_interval: Option<Duration>,
    pub(crate) auto_merge: bool,
    pub(crate) vulkan_min_rows: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            backend: BackendPreference::Cpu,
            max_record_bytes: 64 * 1024,
            commit_interval: Some(Duration::from_secs(1)),
            auto_merge: true,
            vulkan_min_rows: None,
        }
    }
}

impl Options {
    /// Select the compute backend. The default is [`BackendPreference::Cpu`].
    #[must_use]
    pub fn backend(mut self, backend: BackendPreference) -> Self {
        self.backend = backend;
        self
    }

    /// Limit the size of one record: its vector and all field values. The
    /// default is 64 KiB.
    #[must_use]
    pub fn max_record_bytes(mut self, bytes: usize) -> Self {
        self.max_record_bytes = bytes;
        self
    }

    /// Commit changes in the background at this interval, or only on
    /// [`Db::flush`](crate::Db::flush) with `None`. The default is one second.
    #[must_use]
    pub fn commit_interval(mut self, interval: Option<Duration>) -> Self {
        self.commit_interval = interval;
        self
    }

    /// Merge segments after each commit (the default). Without it, segments
    /// merge only on [`Db::compact`](crate::Db::compact).
    #[must_use]
    pub fn auto_merge(mut self, enabled: bool) -> Self {
        self.auto_merge = enabled;
        self
    }

    /// With [`BackendPreference::Vulkan`], search a segment on the CPU when
    /// fewer of its rows than this pass the filter, and keep segments with
    /// fewer rows on the CPU only. A GPU search has a fixed cost of about
    /// 100 µs, the time that a CPU takes to scan about six million
    /// coordinates. The default is 6 Mi divided by the dimension count, for
    /// example 98304 rows of 64 dimensions; 0 puts every search on the GPU.
    #[must_use]
    pub fn vulkan_min_rows(mut self, rows: usize) -> Self {
        self.vulkan_min_rows = Some(rows);
        self
    }

    pub(crate) fn vulkan_min_rows_for(&self, dims: usize) -> usize {
        self.vulkan_min_rows
            .unwrap_or(6 * 1024 * 1024 / dims.max(1))
    }
}
