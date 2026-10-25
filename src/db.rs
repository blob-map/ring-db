//! The database handle, its writes, commits, and merges.
use std::collections::BTreeSet;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, RwLock};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::backend::VulkanDevice;
use crate::codec::corrupt;
use crate::config::{BackendPreference, Options};
use crate::error::{Result, RingDbError};
use crate::filter::Filter;
use crate::query::{Key, Query};
use crate::schema::Schema;
use crate::segment::{Segment, SegmentBuilder};
use crate::snapshot::{Part, Snapshot};
use crate::store::{self, Change, LogFrame, Manifest, Store};
use crate::value::{Document, Record, Row};

/// A segment is rewritten when at least this fraction of its rows is deleted.
const PURGE_FRACTION: f64 = 0.25;
/// A run of newer segments merges into an older one smaller than this many
/// times the run, so live sizes at least double from newest to oldest.
const MERGE_FACTOR: usize = 2;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// An exact vector database with typed fields.
///
/// Every method takes `&self`; share the handle between threads with an `Arc`.
///
/// [`add`](Self::add) and [`delete`](Self::delete) take effect at the next
/// commit: all changes since the previous commit become visible together.
/// A background thread commits every [`Options::commit_interval`];
/// [`flush`](Self::flush) commits at once. To change a record, delete it and
/// add the new version, which gets a new key.
///
/// [`apply`](Self::apply) makes several changes at once: they become visible
/// in the same commit, so a delete and an add can replace a record atomically.
///
/// A database opened with [`Db::open`] is durable: each change is synced to a
/// write-ahead log before the call returns, and survives a crash.
pub struct Db {
    shared: Arc<Shared>,
    committer: Option<Committer>,
}

struct Shared {
    schema: Arc<Schema>,
    options: Options,
    device: Option<Arc<VulkanDevice>>,
    /// The database directory, or `None` in memory or read-only.
    dir: Option<PathBuf>,
    /// Set when the database is open read-only.
    read_only: Option<store::ReadLock>,
    next_segment: AtomicU64,
    writer: Mutex<Writer>,
    current: RwLock<Snapshot>,
    /// Held while a merge builds, so that only one merge runs at a time.
    merging: Mutex<()>,
    /// The last error of the commit thread, not yet taken.
    background_error: Mutex<Option<RingDbError>>,
}

/// Changes since the last commit. Lock `writer` before `current`.
struct Writer {
    store: Option<Store>,
    next_key: u64,
    /// Built segments not yet published, because an upload or a write failed.
    unpublished: Vec<Segment>,
    pending: SegmentBuilder,
    /// Rows of `pending` that are deleted.
    pending_deleted: BTreeSet<u32>,
    /// Visible keys to delete.
    deletes: BTreeSet<Key>,
}

impl Writer {
    fn is_empty(&self) -> bool {
        self.unpublished.is_empty() && self.pending.is_empty() && self.deletes.is_empty()
    }

    /// Rows not yet visible, including deleted ones.
    fn rows(&self) -> usize {
        self.unpublished.iter().map(Segment::len).sum::<usize>() + self.pending.len()
    }

    /// Whether `key` is added and not deleted, committed or not.
    fn is_live(&self, key: Key, visible: &Snapshot) -> bool {
        match self.pending.row(key) {
            Some(row) => !self.pending_deleted.contains(&row),
            None => {
                !self.deletes.contains(&key)
                    && (self.unpublished.iter().any(|s| s.row(key).is_some())
                        || visible.contains(key))
            }
        }
    }

    /// Make the changes recovered from the log again, as pending changes.
    fn replay(&mut self, changes: Vec<Change>, visible: &Snapshot) -> Result<()> {
        for change in changes {
            match change {
                Change::Add(key, row) => {
                    if key.0 < self.next_key {
                        return Err(corrupt(format!("log adds key {} twice", key.0)));
                    }
                    self.next_key = key.0 + 1;
                    self.pending.push(key, row);
                }
                // Replaying a delete of a missing record has no effect.
                Change::Delete(key) => {
                    self.delete(key, visible);
                }
            }
        }
        Ok(())
    }

    /// Record a delete of `key` and report whether it was live.
    fn delete(&mut self, key: Key, visible: &Snapshot) -> bool {
        if !self.is_live(key, visible) {
            return false;
        }
        match self.pending.row(key) {
            Some(row) => self.pending_deleted.insert(row),
            None => self.deletes.insert(key),
        }
    }
}

/// Deletes and adds that [`Db::apply`] makes together.
///
/// ```
/// use ringdb::{Batch, Db, Options, Record, Schema};
///
/// let db = Db::in_memory(Schema::new(1), Options::default()).unwrap();
/// let old = db.add(Record::new([1.0])).unwrap();
/// let keys = db.apply(Batch::new().delete(old).add(Record::new([2.0]))).unwrap();
/// db.flush().unwrap();
/// assert_eq!(db.get(keys[0]).unwrap().vector(), [2.0]);
/// assert!(db.get(old).is_err());
/// ```
#[derive(Debug, Clone, Default)]
pub struct Batch {
    deletes: Vec<Key>,
    adds: Vec<Record>,
}

impl Batch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Delete the record with `key`.
    #[must_use]
    pub fn delete(mut self, key: Key) -> Self {
        self.deletes.push(key);
        self
    }

    /// Add `record`. [`Db::apply`] returns its key.
    #[must_use]
    #[allow(clippy::should_implement_trait)] // the same name as `Db::add`
    pub fn add(mut self, record: Record) -> Self {
        self.adds.push(record);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.deletes.is_empty() && self.adds.is_empty()
    }
}

impl Db {
    /// Create an empty database that is held in memory only.
    pub fn in_memory(schema: Schema, options: Options) -> Result<Self> {
        schema.validate()?;
        Self::start(schema, options, None)
    }

    /// Open the durable database in directory `path`, or create it there.
    ///
    /// An existing database must have exactly `schema`; otherwise this returns
    /// [`RingDbError::InvalidSchema`]. Only one `Db` can have a directory open
    /// for writing: another open returns an [`RingDbError::Io`] error. Opening
    /// recovers the changes that were synced but not yet committed and commits
    /// them.
    pub fn open(path: impl AsRef<Path>, schema: Schema, options: Options) -> Result<Self> {
        schema.validate()?;
        let (store, recovered) = Store::open(path.as_ref(), &schema)?;
        Self::start(schema, options, Some((Opened::Write(store), recovered)))
    }

    /// Open the durable database in directory `path` without changing any of
    /// its files, for example on a read-only mount.
    ///
    /// The database must exist and have exactly `schema`. Changes that are
    /// logged but not committed are visible, and kept in memory only. Every
    /// write, including [`flush`](Self::flush), [`compact`](Self::compact), and
    /// [`backup`](Self::backup), returns [`RingDbError::ReadOnly`]. Several
    /// read-only `Db`s can share a directory, but not with a `Db` that writes.
    pub fn open_read_only(
        path: impl AsRef<Path>,
        schema: Schema,
        options: Options,
    ) -> Result<Self> {
        schema.validate()?;
        let (lock, recovered) = store::open_read_only(path.as_ref(), &schema)?;
        Self::start(schema, options, Some((Opened::Read(lock), recovered)))
    }

    fn start(
        schema: Schema,
        options: Options,
        storage: Option<(Opened, store::Recovered)>,
    ) -> Result<Self> {
        let device = match options.backend {
            BackendPreference::Cpu => None,
            BackendPreference::Vulkan => Some(Arc::new(VulkanDevice::new()?)),
        };
        let schema = Arc::new(schema);
        let mut writer = Writer {
            store: None,
            next_key: 0,
            unpublished: Vec::new(),
            pending: SegmentBuilder::new(&schema),
            pending_deleted: BTreeSet::new(),
            deletes: BTreeSet::new(),
        };
        let mut parts = Vec::new();
        let mut next_segment = 0;
        let mut changes = Vec::new();
        let mut dir = None;
        let mut read_only = None;
        if let Some((opened, recovered)) = storage {
            let manifest = recovered.manifest;
            for (mut segment, (_, deleted)) in recovered.segments.into_iter().zip(manifest.parts) {
                if segment.id >= manifest.next_segment {
                    return Err(corrupt("MANIFEST counters are behind its segments"));
                }
                upload(device.as_ref(), &options, &mut segment)?;
                parts.push(Part::new(Arc::new(segment), deleted));
            }
            writer.next_key = manifest.next_key;
            next_segment = manifest.next_segment;
            changes = recovered.changes;
            match opened {
                Opened::Write(store) => {
                    dir = Some(store.dir().to_path_buf());
                    writer.store = Some(store);
                }
                Opened::Read(lock) => read_only = Some(lock),
            }
        }
        let current = Snapshot::new(schema.clone(), parts);
        writer.replay(changes, &current)?;
        let shared = Arc::new(Shared {
            writer: Mutex::new(writer),
            current: RwLock::new(current),
            merging: Mutex::new(()),
            background_error: Mutex::new(None),
            next_segment: AtomicU64::new(next_segment),
            dir,
            read_only,
            schema,
            options,
            device,
        });
        shared.commit()?;
        let committer = shared
            .options
            .commit_interval
            .filter(|_| shared.read_only.is_none())
            .map(|interval| Committer::start(shared.clone(), interval));
        Ok(Self { shared, committer })
    }

    /// Validate `record` against the schema and assign it a new key.
    pub fn add(&self, record: Record) -> Result<Key> {
        Ok(self.add_all([record])?[0])
    }

    /// Add `records` with one log sync, returning their keys in order.
    ///
    /// If one record is invalid, none is added. Use this to import many
    /// records into a durable database.
    pub fn add_all(&self, records: impl IntoIterator<Item = Record>) -> Result<Vec<Key>> {
        let adds = records.into_iter().collect();
        self.apply(Batch {
            deletes: Vec::new(),
            adds,
        })
    }

    /// Delete the record with `key`, visible or not yet committed.
    ///
    /// Returns [`RingDbError::KeyNotFound`] if no such record exists or it is
    /// already deleted.
    pub fn delete(&self, key: Key) -> Result<()> {
        self.apply(Batch::new().delete(key)).map(drop)
    }

    /// Make the deletes and adds of `batch`, and return the keys of its added
    /// records in order.
    ///
    /// All changes become visible in the same commit. A durable database logs
    /// them with one sync, and recovery after a crash keeps all of them or none.
    /// If a record is invalid, or a key is not found or is deleted twice, this
    /// returns an error and makes no change.
    pub fn apply(&self, batch: Batch) -> Result<Vec<Key>> {
        let shared = &self.shared;
        shared.writable()?;
        let rows = batch
            .adds
            .into_iter()
            .map(|record| record.validate(&shared.schema, shared.options.max_record_bytes))
            .collect::<Result<Vec<Row>>>()?;
        let mut writer = lock(&shared.writer);
        let visible = shared.snapshot();
        let mut deleted = BTreeSet::new();
        for &key in &batch.deletes {
            if !deleted.insert(key) || !writer.is_live(key, &visible) {
                return Err(RingDbError::KeyNotFound(key.0));
            }
        }
        if !rows.is_empty() && writer.rows() + visible.rows() + rows.len() >= u32::MAX as usize {
            return Err(RingDbError::InvalidRecord(
                "a database holds at most u32::MAX - 1 rows, deleted rows included \
                 until they are merged away"
                    .to_string(),
            ));
        }
        let first = writer.next_key;
        let keys: Vec<Key> = (first..first + rows.len() as u64).map(Key).collect();
        if let Some(store) = &mut writer.store
            && !(keys.is_empty() && deleted.is_empty())
        {
            let mut frame = LogFrame::default();
            deleted.iter().for_each(|&key| frame.delete(key));
            keys.iter()
                .zip(&rows)
                .for_each(|(&key, row)| frame.add(key, row));
            store.log(&frame)?;
        }
        for key in deleted {
            writer.delete(key, &visible);
        }
        writer.next_key += rows.len() as u64;
        for (&key, row) in keys.iter().zip(rows) {
            writer.pending.push(key, row);
        }
        Ok(keys)
    }

    /// Commit every change now. With [`Options::auto_merge`], then merge
    /// segments as the merge policy asks.
    pub fn flush(&self) -> Result<()> {
        self.shared.writable()?;
        self.shared.flush()
    }

    /// Commit, then merge all segments into one without deleted rows. Searches
    /// are fastest on a single segment, for example after a bulk import.
    pub fn compact(&self) -> Result<()> {
        self.shared.writable()?;
        self.shared.commit()?;
        self.shared.merge(|parts| {
            (parts.len() > 1 || parts.iter().any(|p| !p.deleted.is_empty()))
                .then_some(0..parts.len())
        })
    }

    /// Commit, then copy the database files into `dest`, which must not exist
    /// or be empty. Open the copy with [`Db::open`]. Changes wait while the
    /// files are copied. An in-memory database returns an [`RingDbError::Io`]
    /// error of kind `Unsupported`.
    pub fn backup(&self, dest: impl AsRef<Path>) -> Result<()> {
        self.shared.writable()?;
        self.shared.commit()?;
        let writer = lock(&self.shared.writer);
        let Some(store) = &writer.store else {
            return Err(RingDbError::Io(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "an in-memory database has no files to back up",
            )));
        };
        let ids: Vec<u64> = self
            .snapshot()
            .parts()
            .iter()
            .map(|p| p.segment.id)
            .collect();
        store.backup(dest.as_ref(), &ids)
    }

    /// A consistent view of the visible records. Later changes do not affect it.
    pub fn snapshot(&self) -> Snapshot {
        self.shared.snapshot()
    }

    /// Run a [`Knn`](crate::Knn), [`Disk`](crate::Disk), [`Region`](crate::Region),
    /// or [`Count`](crate::Count) search.
    pub fn search<Q: Query>(&self, query: Q) -> Result<Q::Output> {
        self.snapshot().search(query)
    }

    pub fn get(&self, key: Key) -> Result<Document> {
        self.snapshot().get(key)
    }

    /// Number of visible records that pass `filter`. No vector is read.
    pub fn count(&self, filter: &Filter) -> Result<usize> {
        self.snapshot().count(filter)
    }

    /// Number of visible records.
    pub fn len(&self) -> usize {
        self.snapshot().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn schema(&self) -> &Schema {
        &self.shared.schema
    }

    pub fn backend(&self) -> BackendPreference {
        self.shared.options.backend
    }

    /// Take the last error of a background commit or merge, if one failed
    /// since the previous call. The changes of a failed commit stay pending,
    /// and the next commit retries them.
    pub fn take_background_error(&self) -> Option<RingDbError> {
        lock(&self.shared.background_error).take()
    }
}

impl Drop for Db {
    fn drop(&mut self) {
        if let Some(committer) = self.committer.take() {
            committer.stop();
        }
        if self.shared.dir.is_some() {
            // Logged changes are durable already; committing now only makes
            // the next open faster.
            let _ = self.shared.commit();
        }
    }
}

impl Shared {
    fn snapshot(&self) -> Snapshot {
        self.current
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn writable(&self) -> Result<()> {
        match self.read_only {
            Some(_) => Err(RingDbError::ReadOnly),
            None => Ok(()),
        }
    }

    fn segment_id(&self) -> u64 {
        self.next_segment.fetch_add(1, Ordering::Relaxed)
    }

    /// Upload `segment` and write its file, as configured.
    fn prepare(&self, segment: &mut Segment) -> Result<()> {
        upload(self.device.as_ref(), &self.options, segment)?;
        if let Some(dir) = &self.dir {
            store::write_segment(dir, segment)?;
        }
        Ok(())
    }

    fn manifest(&self, writer: &Writer, parts: &[Part]) -> Manifest {
        Manifest {
            next_key: writer.next_key,
            next_segment: self.next_segment.load(Ordering::Relaxed),
            parts: parts
                .iter()
                .map(|part| (part.segment.id, part.deleted.to_vec()))
                .collect(),
        }
    }

    /// Commit, then merge as the merge policy asks if `auto_merge` is set.
    fn flush(&self) -> Result<()> {
        self.commit()?;
        if self.options.auto_merge {
            self.merge(merge_range)?;
        }
        Ok(())
    }

    /// Publish pending records and deletes in one new snapshot. On failure the
    /// changes stay pending, and the next commit retries them. A commit also
    /// starts a new log after a failed log write.
    fn commit(&self) -> Result<()> {
        let mut guard = lock(&self.writer);
        let writer = &mut *guard;
        if writer.is_empty() && !writer.store.as_ref().is_some_and(Store::needs_new_log) {
            return Ok(());
        }
        if !writer.pending.is_empty() {
            let builder = std::mem::replace(&mut writer.pending, SegmentBuilder::new(&self.schema));
            let segment = builder.finish(self.segment_id());
            for row in std::mem::take(&mut writer.pending_deleted) {
                writer.deletes.insert(segment.keys[row as usize]);
            }
            writer.unpublished.push(segment);
        }
        for segment in &mut writer.unpublished {
            self.prepare(segment)?;
        }
        let mut parts = self.snapshot().parts().to_vec();
        let added = writer.unpublished.len();
        let segments = std::mem::take(&mut writer.unpublished);
        parts.extend(
            segments
                .into_iter()
                .map(|segment| Part::new(Arc::new(segment), Vec::new())),
        );
        apply_deletes(&mut parts, &writer.deletes);
        let manifest = writer
            .store
            .is_some()
            .then(|| self.manifest(writer, &parts));
        if let (Some(store), Some(manifest)) = (&mut writer.store, manifest)
            && let Err(error) = store.write_manifest(&self.schema, &manifest, true)
        {
            // Keep the new segments for the retry.
            writer.unpublished = parts
                .drain(parts.len() - added..)
                .map(|part| Arc::into_inner(part.segment).expect("not shared yet"))
                .collect();
            return Err(error);
        }
        writer.deletes.clear();
        let mut current = self.current.write().unwrap_or_else(PoisonError::into_inner);
        *current = Snapshot::new(self.schema.clone(), parts);
        Ok(())
    }

    /// Merge the parts that `policy` selects until it selects none. Writers
    /// are not blocked while a merged segment builds.
    fn merge(&self, policy: impl Fn(&[Part]) -> Option<Range<usize>>) -> Result<()> {
        let _merging = lock(&self.merging);
        loop {
            let snapshot = self.snapshot();
            let Some(range) = policy(snapshot.parts()) else {
                return Ok(());
            };
            let inputs = &snapshot.parts()[range];
            let merged = self.build_merge(inputs)?;
            self.publish_merge(inputs, merged)?;
        }
    }

    /// The live rows of adjacent `inputs` in one prepared segment, or `None`
    /// if none is live.
    fn build_merge(&self, inputs: &[Part]) -> Result<Option<Segment>> {
        let mut builder = SegmentBuilder::new(&self.schema);
        for part in inputs {
            for row in 0..part.segment.len() as u32 {
                if !part.is_deleted(row) {
                    let key = part.segment.keys[row as usize];
                    builder.push(key, part.segment.read(&self.schema, row as usize));
                }
            }
        }
        if builder.is_empty() {
            return Ok(None);
        }
        let mut segment = builder.finish(self.segment_id());
        self.prepare(&mut segment)?;
        Ok(Some(segment))
    }

    /// Replace `inputs` with `merged`, keeping rows deleted since `inputs`
    /// were read deleted.
    fn publish_merge(&self, inputs: &[Part], merged: Option<Segment>) -> Result<()> {
        let mut writer = lock(&self.writer);
        let mut parts = self.snapshot().parts().to_vec();
        let start = parts
            .iter()
            .position(|part| Arc::ptr_eq(&part.segment, &inputs[0].segment))
            .expect("only merges remove segments, one at a time");
        let end = start + inputs.len();
        let late: Vec<Key> = parts[start..end]
            .iter()
            .zip(inputs)
            .flat_map(|(now, before)| {
                now.deleted
                    .iter()
                    .filter(|&&row| !before.is_deleted(row))
                    .map(|&row| now.segment.keys[row as usize])
            })
            .collect();
        let merged_id = merged.as_ref().map(|segment| segment.id);
        let replacement = merged.map(|segment| {
            let deleted = late.iter().filter_map(|&key| segment.row(key)).collect();
            Part::new(Arc::new(segment), deleted)
        });
        parts.splice(start..end, replacement);
        if writer.store.is_some() {
            let manifest = self.manifest(&writer, &parts);
            let store = writer.store.as_mut().expect("checked above");
            if let Err(error) = store.write_manifest(&self.schema, &manifest, false) {
                if let Some(id) = merged_id {
                    store.remove_segment(id);
                }
                return Err(error);
            }
            for input in inputs {
                store.remove_segment(input.segment.id);
            }
        }
        let mut current = self.current.write().unwrap_or_else(PoisonError::into_inner);
        *current = Snapshot::new(self.schema.clone(), parts);
        Ok(())
    }
}

/// Keep the vectors of `segment` on the device, unless it is small enough that
/// every search of it runs on the CPU.
fn upload(
    device: Option<&Arc<VulkanDevice>>,
    options: &Options,
    segment: &mut Segment,
) -> Result<()> {
    let min_rows = options.vulkan_min_rows_for(segment.dims);
    match device {
        Some(device) if segment.len() >= min_rows && !segment.is_uploaded() => {
            segment.upload(device, min_rows)
        }
        _ => Ok(()),
    }
}

/// Mark `keys` deleted in the parts that hold them. Every key must be live.
fn apply_deletes(parts: &mut [Part], keys: &BTreeSet<Key>) {
    let mut keys = keys.iter().copied().peekable();
    for part in parts.iter_mut() {
        let last = part.segment.keys[part.segment.len() - 1];
        let mut rows = Vec::new();
        while let Some(key) = keys.next_if(|&key| key <= last) {
            rows.extend(part.segment.row(key));
        }
        if !rows.is_empty() {
            let mut deleted = part.deleted.to_vec();
            deleted.extend(rows);
            deleted.sort_unstable();
            *part = Part::new(part.segment.clone(), deleted);
        }
    }
}

/// The adjacent parts to merge next, if any.
fn merge_range(parts: &[Part]) -> Option<Range<usize>> {
    if let Some(index) = parts
        .iter()
        .position(|part| part.deleted.len() as f64 >= PURGE_FRACTION * part.segment.len() as f64)
    {
        return Some(index..index + 1);
    }
    let last = parts.len().checked_sub(1)?;
    let mut run = parts[last].live();
    let mut start = last;
    while start > 0 && parts[start - 1].live() < MERGE_FACTOR * run {
        start -= 1;
        run += parts[start].live();
    }
    (start < last).then_some(start..last + 1)
}

/// How [`Db::start`] reaches the files of a durable database.
enum Opened {
    Write(Store),
    Read(store::ReadLock),
}

/// Commits in the background every `interval` until stopped.
struct Committer {
    stop: Arc<(Mutex<bool>, Condvar)>,
    thread: JoinHandle<()>,
}

impl Committer {
    fn start(shared: Arc<Shared>, interval: Duration) -> Self {
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let signal = stop.clone();
        let thread = std::thread::Builder::new()
            .name("ringdb-commit".to_string())
            .spawn(move || {
                let (stopped, wake) = &*signal;
                let mut stopped = lock(stopped);
                while !*stopped {
                    stopped = wake
                        .wait_timeout(stopped, interval)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                    if !*stopped {
                        drop(stopped);
                        // A failure leaves the changes pending, and the
                        // next commit retries them.
                        if let Err(error) = shared.flush() {
                            *lock(&shared.background_error) = Some(error);
                        }
                        stopped = lock(&signal.0);
                    }
                }
            })
            .expect("spawn the commit thread");
        Self { stop, thread }
    }

    fn stop(self) {
        *lock(&self.stop.0) = true;
        self.stop.1.notify_all();
        let _ = self.thread.join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::Knn;
    use crate::schema::FieldType;

    fn db() -> Db {
        let schema = Schema::new(1).field("n", FieldType::U32);
        let options = Options::default().commit_interval(None).auto_merge(false);
        Db::in_memory(schema, options).unwrap()
    }

    fn keys(db: &Db) -> Vec<u64> {
        let mut keys: Vec<u64> = db
            .search(Knn::new(&[0.0], 1000))
            .unwrap()
            .iter()
            .map(|hit| hit.key.get())
            .collect();
        keys.sort_unstable();
        keys
    }

    #[test]
    fn deletes_during_a_merge_stay_deleted() {
        let db = db();
        for batch in 0..3 {
            for i in 0..4 {
                db.add(Record::new([(batch * 4 + i) as f32]).set("n", i))
                    .unwrap();
            }
            db.flush().unwrap();
        }
        db.delete(Key(1)).unwrap();
        db.flush().unwrap();
        let shared = &db.shared;
        let before = shared.snapshot();
        let inputs = &before.parts()[0..2];
        let merged = shared.build_merge(inputs).unwrap();
        // While the merge builds: delete from both inputs and the third part.
        for key in [2, 5, 9] {
            db.delete(Key(key)).unwrap();
        }
        db.flush().unwrap();
        shared.publish_merge(inputs, merged).unwrap();

        let after = db.snapshot();
        assert_eq!(after.segments(), 2);
        assert_eq!(after.parts()[0].segment.len(), 7, "key 1 was purged");
        assert_eq!(&*after.parts()[0].deleted, [1, 4], "keys 2 and 5, remapped");
        assert_eq!(keys(&db), [0, 3, 4, 6, 7, 8, 10, 11]);
        assert!(matches!(
            db.delete(Key(5)),
            Err(RingDbError::KeyNotFound(5))
        ));
        assert!(db.get(Key(2)).is_err());
        assert_eq!(
            db.get(Key(6)).unwrap().get("n").and_then(|v| v.as_u32()),
            Some(2)
        );
    }

    #[test]
    fn a_commit_starts_a_new_log_after_a_failed_log_write() {
        let dir = tempfile::tempdir().unwrap();
        let options = Options::default().commit_interval(None);
        let db = Db::open(dir.path(), Schema::new(1), options.clone()).unwrap();
        let kept = db.add(Record::new([1.0])).unwrap();
        db.flush().unwrap();
        lock(&db.shared.writer).store.as_mut().unwrap().fail();
        assert!(matches!(
            db.add(Record::new([2.0])),
            Err(RingDbError::Io(_))
        ));
        // Nothing is pending, but the commit must still replace the log.
        db.flush().unwrap();
        let added = db.add(Record::new([3.0])).unwrap();
        drop(db);
        let db = Db::open(dir.path(), Schema::new(1), options).unwrap();
        assert_eq!(db.len(), 2);
        assert!(db.get(kept).is_ok() && db.get(added).is_ok());
    }

    #[test]
    fn merge_policy_keeps_sizes_geometric_and_purges_deletes() {
        let part = |rows: usize, deleted: usize| {
            let mut builder = SegmentBuilder::new(&Schema::new(1));
            for key in 0..rows as u64 {
                builder.push(
                    Key(key),
                    crate::value::Row {
                        vector: vec![0.0],
                        values: Vec::new(),
                    },
                );
            }
            Part::new(Arc::new(builder.finish(0)), (0..deleted as u32).collect())
        };
        let range = |sizes: &[(usize, usize)]| {
            let parts: Vec<Part> = sizes.iter().map(|&(n, d)| part(n, d)).collect();
            merge_range(&parts)
        };
        assert_eq!(range(&[]), None);
        assert_eq!(range(&[(10, 0)]), None);
        assert_eq!(range(&[(100, 0), (10, 0)]), None);
        assert_eq!(range(&[(100, 0), (40, 0), (10, 0)]), None);
        assert_eq!(range(&[(100, 0), (15, 0), (10, 0)]), Some(1..3));
        assert_eq!(range(&[(30, 0), (15, 0), (10, 0)]), Some(0..3));
        // Live sizes count: a mostly deleted older part is no barrier.
        assert_eq!(range(&[(100, 0), (40, 20), (12, 0)]), Some(1..2));
        assert_eq!(range(&[(100, 24), (30, 0)]), None);
        assert_eq!(range(&[(100, 25), (30, 0)]), Some(0..1));
        assert_eq!(range(&[(100, 0), (50, 0), (8, 2)]), Some(2..3));
    }
}
