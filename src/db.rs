//! The database handle and its consistent read views.
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use smallvec::SmallVec;

use crate::backend::{DiskSpec, VulkanDevice};
use crate::config::{BackendPreference, Options};
use crate::error::{Result, RingDbError};
use crate::filter::Filter;
use crate::query::{
    Hit, Key, KeyLookup, Knn, MAX_DISKS, Query, Region, RegionResult, validate_disk,
    validate_metric, validate_vector,
};
use crate::rows::RowSet;
use crate::schema::Schema;
use crate::segment::{Segment, SegmentBuilder};
use crate::value::{Document, Record};

/// An exact vector database with typed fields.
///
/// Every method takes `&self`; share the handle between threads with an `Arc`.
/// [`add`](Self::add) assigns a new key. Added records become visible to
/// searches and [`get`](Self::get) after [`flush`](Self::flush).
pub struct Db {
    schema: Arc<Schema>,
    options: Options,
    device: Option<Arc<VulkanDevice>>,
    writer: Mutex<Writer>,
    current: RwLock<Snapshot>,
}

struct Writer {
    next_key: u64,
    pending: SegmentBuilder,
}

impl Db {
    /// Create an empty database that is held in memory only.
    pub fn in_memory(schema: Schema, options: Options) -> Result<Self> {
        schema.validate()?;
        let device = match options.backend {
            BackendPreference::Cpu => None,
            BackendPreference::Vulkan => Some(Arc::new(VulkanDevice::new()?)),
        };
        let schema = Arc::new(schema);
        Ok(Self {
            writer: Mutex::new(Writer {
                next_key: 0,
                pending: SegmentBuilder::new(&schema),
            }),
            current: RwLock::new(Snapshot::new(schema.clone(), Vec::new())),
            schema,
            options,
            device,
        })
    }

    /// Validate `record` against the schema and assign it a new key.
    pub fn add(&self, record: Record) -> Result<Key> {
        let row = record.validate(&self.schema, self.options.max_record_bytes)?;
        let mut writer = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        if writer.pending.len() + self.len() >= u32::MAX as usize {
            return Err(RingDbError::InvalidRecord(
                "a database holds at most u32::MAX - 1 records".to_string(),
            ));
        }
        let key = Key(writer.next_key);
        writer.next_key += 1;
        writer.pending.push(key, row);
        Ok(key)
    }

    /// Make every added record visible.
    pub fn flush(&self) -> Result<()> {
        let mut writer = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        if writer.pending.is_empty() {
            return Ok(());
        }
        let builder = std::mem::replace(&mut writer.pending, SegmentBuilder::new(&self.schema));
        let segment = Arc::new(builder.finish(self.device.as_ref())?);
        let mut current = self.current.write().unwrap_or_else(PoisonError::into_inner);
        let mut segments = current.0.segments.clone();
        segments.push(segment);
        *current = Snapshot::new(self.schema.clone(), segments);
        Ok(())
    }

    /// A consistent view of the visible records. Later changes do not affect it.
    pub fn snapshot(&self) -> Snapshot {
        self.current
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Run a [`Knn`], [`Disk`](crate::Disk), or [`Region`] search.
    pub fn search<Q: Query>(&self, query: Q) -> Result<Q::Output> {
        self.snapshot().search(query)
    }

    pub fn get(&self, key: Key) -> Result<Document> {
        self.snapshot().get(key)
    }

    /// Number of visible records.
    pub fn len(&self) -> usize {
        self.snapshot().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn backend(&self) -> BackendPreference {
        self.options.backend
    }
}

/// A read-only view of a database at one point in time.
///
/// Cloning is cheap. Searches on one snapshot always see the same records.
#[derive(Clone)]
pub struct Snapshot(Arc<View>);

struct View {
    schema: Arc<Schema>,
    /// Ordered by key; their key ranges do not overlap.
    segments: Vec<Arc<Segment>>,
    keys: KeyLookup,
    len: usize,
}

impl Snapshot {
    fn new(schema: Arc<Schema>, segments: Vec<Arc<Segment>>) -> Self {
        let mut base = 0u32;
        let keys = segments
            .iter()
            .map(|segment| {
                let part = (base, segment.keys.clone());
                base += segment.len() as u32;
                part
            })
            .collect();
        Self(Arc::new(View {
            schema,
            segments,
            keys: KeyLookup(keys),
            len: base as usize,
        }))
    }

    /// Run a [`Knn`], [`Disk`](crate::Disk), or [`Region`] search.
    pub fn search<Q: Query>(&self, query: Q) -> Result<Q::Output> {
        query.execute(self)
    }

    pub fn get(&self, key: Key) -> Result<Document> {
        let view = &self.0;
        let index = view
            .segments
            .partition_point(|segment| segment.keys.first().is_some_and(|&first| first <= key));
        let segment = index
            .checked_sub(1)
            .map(|index| &view.segments[index])
            .ok_or(RingDbError::KeyNotFound(key.0))?;
        let row = segment.row(key).ok_or(RingDbError::KeyNotFound(key.0))?;
        Ok(Document {
            key,
            schema: view.schema.clone(),
            row: segment.read(&view.schema, row as usize),
        })
    }

    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// The rows of each segment that pass `filter`, with each segment's base row.
    fn eligible(&self, filter: Option<&Filter>) -> Result<Vec<(u32, &Segment, RowSet)>> {
        let filter = filter.map(|f| f.compile(&self.0.schema)).transpose()?;
        let mut base = 0;
        Ok(self
            .0
            .segments
            .iter()
            .map(|segment| {
                let rows = filter
                    .as_ref()
                    .map_or_else(|| RowSet::all(segment.len()), |f| f.rows(segment));
                let part = (base, &**segment, rows);
                base += segment.len() as u32;
                part
            })
            .collect())
    }

    pub(crate) fn knn(&self, query: &Knn<'_>) -> Result<Vec<Hit>> {
        let dims = self.0.schema.dims();
        validate_vector(dims, query.query, "query")?;
        validate_metric(dims, query.metric)?;
        let eligible = self.eligible(query.filter)?;
        if query.k == 0 {
            return Ok(Vec::new());
        }
        let weights = query.metric.weights();
        let mut hits = Vec::new();
        for (_, segment, rows) in &eligible {
            let Some(mask) = rows.mask() else { continue };
            let found = segment.knn(query.query, weights, query.k, mask)?;
            hits.extend(found.into_iter().map(|hit| Hit {
                key: segment.keys[hit.row as usize],
                dist_sq: hit.dist_sq,
            }));
        }
        if eligible.len() > 1 {
            // Within a segment, row order is key order; restore it across them.
            hits.sort_unstable_by(|a, b| {
                a.dist_sq
                    .total_cmp(&b.dist_sq)
                    .then_with(|| a.key.cmp(&b.key))
            });
            hits.truncate(query.k);
        }
        Ok(hits)
    }

    pub(crate) fn region(&self, region: &Region<'_>) -> Result<RegionResult> {
        if region.disks.len() > MAX_DISKS {
            return Err(RingDbError::InvalidQuery(format!(
                "a region can contain at most {MAX_DISKS} disks, got {}",
                region.disks.len()
            )));
        }
        let dims = self.0.schema.dims();
        let disks: SmallVec<[DiskSpec<'_>; 1]> = region
            .disks
            .iter()
            .map(|disk| validate_disk(dims, disk))
            .collect::<Result<_>>()?;
        let eligible = self.eligible(region.filter)?;
        let mut out = RegionResult::new(disks.len(), self.0.keys.clone());
        for (base, segment, rows) in &eligible {
            if let Some(mask) = rows.mask() {
                segment.region(*base, &disks, mask, &mut out.records)?;
            }
        }
        Ok(out)
    }
}
