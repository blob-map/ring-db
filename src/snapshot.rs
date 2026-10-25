//! Consistent read views of a database, and the searches that run on them.
use std::borrow::Cow;
use std::sync::Arc;

use smallvec::SmallVec;

use crate::backend::DiskSpec;
use crate::error::{Result, RingDbError};
use crate::filter::Filter;
use crate::query::{
    Hit, Key, KeyLookup, Knn, MAX_DISKS, Query, Region, RegionResult, validate_disk,
    validate_metric, validate_vector,
};
use crate::rows::RowSet;
use crate::schema::Schema;
use crate::segment::Segment;
use crate::value::Document;

/// One segment as seen by a snapshot: its deleted rows and the live rest.
#[derive(Clone)]
pub(crate) struct Part {
    pub(crate) segment: Arc<Segment>,
    /// Sorted rows.
    pub(crate) deleted: Arc<[u32]>,
    /// The complement of `deleted`, or `None` when nothing is deleted.
    live: Option<Arc<RowSet>>,
}

impl Part {
    pub(crate) fn new(segment: Arc<Segment>, deleted: Vec<u32>) -> Self {
        let n = segment.len();
        let live = (!deleted.is_empty())
            .then(|| Arc::new(RowSet::from_sorted(n, deleted.clone()).complement()));
        Self {
            segment,
            deleted: deleted.into(),
            live,
        }
    }

    /// Number of rows that are not deleted.
    pub(crate) fn live(&self) -> usize {
        self.segment.len() - self.deleted.len()
    }

    pub(crate) fn is_deleted(&self, row: u32) -> bool {
        self.deleted.binary_search(&row).is_ok()
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
    parts: Vec<Part>,
    keys: KeyLookup,
    len: usize,
    rows: usize,
}

impl Snapshot {
    pub(crate) fn new(schema: Arc<Schema>, parts: Vec<Part>) -> Self {
        let mut base = 0u32;
        let keys = parts
            .iter()
            .map(|part| {
                let entry = (base, part.segment.keys.clone());
                base += part.segment.len() as u32;
                entry
            })
            .collect();
        let len = parts.iter().map(Part::live).sum();
        Self(Arc::new(View {
            schema,
            parts,
            keys: KeyLookup(keys),
            len,
            rows: base as usize,
        }))
    }

    /// Run a [`Knn`], [`Disk`](crate::Disk), or [`Region`] search.
    pub fn search<Q: Query>(&self, query: Q) -> Result<Q::Output> {
        query.execute(self)
    }

    pub fn get(&self, key: Key) -> Result<Document> {
        let (part, row) = self.find(key).ok_or(RingDbError::KeyNotFound(key.0))?;
        Ok(Document {
            key,
            schema: self.0.schema.clone(),
            row: part.segment.read(&self.0.schema, row as usize),
        })
    }

    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// Number of segments. Commits add segments and merges combine them.
    pub fn segments(&self) -> usize {
        self.0.parts.len()
    }

    pub(crate) fn parts(&self) -> &[Part] {
        &self.0.parts
    }

    /// Rows of all segments, deleted ones included.
    pub(crate) fn rows(&self) -> usize {
        self.0.rows
    }

    pub(crate) fn contains(&self, key: Key) -> bool {
        self.find(key).is_some()
    }

    /// The part and row of a live record.
    fn find(&self, key: Key) -> Option<(&Part, u32)> {
        let parts = &self.0.parts;
        let index = parts.partition_point(|part| part.segment.keys[0] <= key);
        let part = &parts[index.checked_sub(1)?];
        let row = part.segment.row(key)?;
        (!part.is_deleted(row)).then_some((part, row))
    }

    /// The live rows of each segment that pass `filter`, with each segment's
    /// base row.
    fn eligible(&self, filter: Option<&Filter>) -> Result<Vec<(u32, &Segment, Cow<'_, RowSet>)>> {
        let filter = filter.map(|f| f.compile(&self.0.schema)).transpose()?;
        let mut base = 0;
        Ok(self
            .0
            .parts
            .iter()
            .map(|part| {
                let segment = &*part.segment;
                let rows = match (&filter, &part.live) {
                    (None, None) => Cow::Owned(RowSet::all(segment.len())),
                    (None, Some(live)) => Cow::Borrowed(&**live),
                    (Some(filter), None) => Cow::Owned(filter.rows(segment)),
                    (Some(filter), Some(live)) => {
                        Cow::Owned(filter.rows(segment).intersection(live))
                    }
                };
                let entry = (base, segment, rows);
                base += segment.len() as u32;
                entry
            })
            .collect())
    }

    pub(crate) fn knn(&self, query: &Knn<'_>) -> Result<Vec<Hit>> {
        let dims = self.0.schema.dims();
        validate_vector(dims, query.query, "query", RingDbError::InvalidQuery)?;
        validate_metric(dims, query.metric)?;
        let eligible = self.eligible(query.filter)?;
        if query.k == 0 {
            return Ok(Vec::new());
        }
        let weights = query.metric.weights();
        let mut hits = Vec::new();
        for (_, segment, rows) in &eligible {
            let found = segment.knn(query.query, weights, query.k, rows)?;
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
            segment.region(*base, &disks, rows, &mut out.records)?;
        }
        Ok(out)
    }
}
