//! Durable storage: a write-ahead log, immutable segment files, and a manifest.
//!
//! A database directory contains:
//!
//! - `LOCK`, locked exclusively while a [`Db`](crate::Db) has the directory open
//!   for writing, and shared while it is open read-only;
//! - `MANIFEST`, the schema, counters, live segment files, and their deleted rows;
//! - `seg-<id>.seg`, one immutable file per segment;
//! - `wal-<seq>.log`, the changes since the manifest was written.
//!
//! The changes of each write call ([`Db::apply`](crate::Db::apply) and the
//! methods built on it) are appended to the log as one frame and synced before
//! the call returns. A commit writes new segment files, starts a new log, and
//! then replaces the manifest with a rename. Recovery loads the manifest's
//! segments and replays its log up to the first torn frame. Files that the manifest does not name
//! are left over from an interrupted commit or merge and are removed.
//!
//! A read-only open changes no file: it skips the cleanup and keeps the
//! changes of the log in memory.
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::codec::{Decoder, Encoder, corrupt};
use crate::error::{Result, RingDbError};
use crate::query::Key;
use crate::schema::{FieldType, Schema};
use crate::segment::Segment;
use crate::value::Row;

const MANIFEST_MAGIC: &[u8; 8] = b"RDBMANI\0";
const WAL_MAGIC: &[u8; 8] = b"RDBWAL\0\0";
const VERSION: u32 = 1;
const WAL_HEADER_LEN: u64 = 12;

const FIELD_TYPES: [FieldType; 10] = [
    FieldType::U32,
    FieldType::U64,
    FieldType::I64,
    FieldType::F32,
    FieldType::Bool,
    FieldType::Keyword,
    FieldType::Keywords,
    FieldType::Text,
    FieldType::TextList,
    FieldType::Bytes,
];

/// The durable state named by `MANIFEST`.
pub(crate) struct Manifest {
    pub(crate) next_key: u64,
    pub(crate) next_segment: u64,
    /// Segment ids in key order, with their sorted deleted rows.
    pub(crate) parts: Vec<(u64, Vec<u32>)>,
}

/// One logged change.
pub(crate) enum Change {
    Add(Key, Row),
    Delete(Key),
}

impl Change {
    fn decode(d: &mut Decoder<'_>, schema: &Schema) -> Result<Self> {
        Ok(match (d.u8()?, Key(d.u64()?)) {
            (1, key) => Self::Add(key, Row::decode(d, schema)?),
            (2, key) => Self::Delete(key),
            _ => return Err(corrupt("unknown log record")),
        })
    }
}

/// The changes of one log frame, which recovery replays all together or not
/// at all.
#[derive(Default)]
pub(crate) struct LogFrame(Encoder);

impl LogFrame {
    pub(crate) fn add(&mut self, key: Key, row: &Row) {
        self.0.u8(1);
        self.0.u64(key.0);
        row.encode(&mut self.0);
    }

    pub(crate) fn delete(&mut self, key: Key) {
        self.0.u8(2);
        self.0.u64(key.0);
    }
}

/// What [`Store::open`] found on disk.
pub(crate) struct Recovered {
    pub(crate) manifest: Manifest,
    pub(crate) segments: Vec<Segment>,
    pub(crate) changes: Vec<Change>,
}

pub(crate) struct Store {
    dir: PathBuf,
    _lock: File,
    wal: File,
    wal_seq: u64,
    /// Set when a log write fails: the log end is then unknown, so writes
    /// wait for a new log.
    failed: bool,
}

fn manifest_path(dir: &Path) -> PathBuf {
    dir.join("MANIFEST")
}

fn wal_path(dir: &Path, seq: u64) -> PathBuf {
    dir.join(format!("wal-{seq:016x}.log"))
}

fn segment_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("seg-{id:016x}.seg"))
}

/// Whether `name` is `prefix`, 16 hexadecimal digits, and `suffix`, as in the
/// names of segment files and logs.
fn is_numbered(name: &str, prefix: &str, suffix: &str) -> bool {
    name.strip_prefix(prefix)
        .and_then(|rest| rest.strip_suffix(suffix))
        .is_some_and(|id| id.len() == 16 && id.bytes().all(|b| b.is_ascii_hexdigit()))
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

/// Write `bytes` to `path` through a synced temporary file and a rename.
fn write_atomic(dir: &Path, path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    let mut file = File::create(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&tmp, path)?;
    sync_dir(dir)
}

fn encode_manifest(schema: &Schema, manifest: &Manifest, wal_seq: u64) -> Vec<u8> {
    let mut e = Encoder::default();
    e.header(MANIFEST_MAGIC, VERSION);
    e.u32(schema.dims() as u32);
    e.u32(schema.len() as u32);
    for (name, ty) in schema.fields() {
        e.bytes(name.as_bytes());
        e.u8(FIELD_TYPES
            .iter()
            .position(|t| *t == ty)
            .expect("every type is listed") as u8);
    }
    e.u64(manifest.next_key);
    e.u64(manifest.next_segment);
    e.u64(wal_seq);
    e.u64(manifest.parts.len() as u64);
    for (id, deleted) in &manifest.parts {
        e.u64(*id);
        e.slice(deleted);
    }
    e.finish_with_crc()
}

fn decode_manifest(bytes: &[u8]) -> Result<(Schema, Manifest, u64)> {
    let mut d = Decoder::with_crc(bytes, "MANIFEST")?;
    d.header(MANIFEST_MAGIC, VERSION, "MANIFEST")?;
    let mut schema = Schema::new(d.u32()? as usize);
    for _ in 0..d.u32()? {
        let name = d.string()?;
        let ty = *FIELD_TYPES
            .get(d.u8()? as usize)
            .ok_or_else(|| corrupt("MANIFEST names an unknown field type"))?;
        schema = schema.field(&name, ty);
    }
    let (next_key, next_segment, wal_seq) = (d.u64()?, d.u64()?, d.u64()?);
    let parts = (0..d.u64()?)
        .map(|_| Ok((d.u64()?, d.slice::<u32>()?)))
        .collect::<Result<_>>()?;
    if !d.is_empty() {
        return Err(corrupt("MANIFEST has trailing data"));
    }
    let manifest = Manifest {
        next_key,
        next_segment,
        parts,
    };
    Ok((schema, manifest, wal_seq))
}

/// Write a new segment file into `dir`. It is unused until a manifest names it.
pub(crate) fn write_segment(dir: &Path, segment: &Segment) -> Result<()> {
    let mut file = File::create(segment_path(dir, segment.id))?;
    file.write_all(&segment.encode())?;
    file.sync_all()?;
    Ok(())
}

/// Read log frames until the end or the first torn frame. With `repair`, cut
/// the file there so that new frames follow the last complete one.
fn replay(path: &Path, schema: &Schema, repair: bool) -> Result<Vec<Change>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let Some(header) = bytes.get(..WAL_HEADER_LEN as usize) else {
        // A crash while the log was created leaves a short header.
        return Ok(Vec::new());
    };
    Decoder::new(header).header(WAL_MAGIC, VERSION, &path.display().to_string())?;
    let mut changes = Vec::new();
    let mut at = WAL_HEADER_LEN as usize;
    while let Some(header) = bytes.get(at..at + 8) {
        let len = u32::from_le_bytes(header[..4].try_into().expect("four bytes")) as usize;
        let crc = u32::from_le_bytes(header[4..].try_into().expect("four bytes"));
        let Some(payload) = bytes.get(at + 8..at + 8 + len) else {
            break;
        };
        if crc32fast::hash(payload) != crc {
            break;
        }
        let mut d = Decoder::new(payload);
        while !d.is_empty() {
            changes.push(Change::decode(&mut d, schema)?);
        }
        at += 8 + len;
    }
    if repair && at < bytes.len() {
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(at as u64)?;
        file.sync_all()?;
    }
    Ok(changes)
}

/// Create log `seq` with only its header, synced.
fn create_wal(dir: &Path, seq: u64) -> io::Result<File> {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(wal_path(dir, seq))?;
    let mut header = Encoder::default();
    header.header(WAL_MAGIC, VERSION);
    file.write_all(header.as_bytes())?;
    file.sync_all()?;
    sync_dir(dir)?;
    Ok(file)
}

/// Turn the result of locking the directory into an error if the lock is held.
fn check_lock(dir: &Path, locked: std::result::Result<(), TryLockError>) -> Result<()> {
    match locked {
        Ok(()) => Ok(()),
        Err(TryLockError::WouldBlock) => Err(RingDbError::Io(io::Error::new(
            io::ErrorKind::WouldBlock,
            format!("{} is open in another Db", dir.display()),
        ))),
        Err(TryLockError::Error(error)) => Err(error.into()),
    }
}

/// Decode the manifest `bytes` of `dir` and check that it holds `schema`.
fn read_manifest(dir: &Path, bytes: &[u8], schema: &Schema) -> Result<(Manifest, u64)> {
    let (stored, manifest, wal_seq) = decode_manifest(bytes)?;
    if stored != *schema {
        return Err(RingDbError::InvalidSchema(format!(
            "{} holds a database with a different schema: {stored:?}",
            dir.display()
        )));
    }
    Ok((manifest, wal_seq))
}

/// Read the segments that `manifest` names and the changes of log `wal_seq`.
fn load(
    dir: &Path,
    schema: &Schema,
    manifest: &Manifest,
    wal_seq: u64,
    repair: bool,
) -> Result<(Vec<Segment>, Vec<Change>)> {
    let segments = manifest
        .parts
        .iter()
        .map(|(id, deleted)| {
            let segment = Segment::decode(&fs::read(segment_path(dir, *id))?, schema, *id)?;
            let valid = deleted.windows(2).all(|w| w[0] < w[1])
                && deleted
                    .last()
                    .is_none_or(|&row| (row as usize) < segment.len());
            if !valid {
                return Err(corrupt(format!("deleted rows of segment {id} are invalid")));
            }
            Ok(segment)
        })
        .collect::<Result<Vec<_>>>()?;
    let ordered = segments
        .windows(2)
        .all(|w| w[0].keys[w[0].len() - 1] < w[1].keys[0])
        && segments
            .last()
            .is_none_or(|s| s.keys[s.len() - 1].0 < manifest.next_key);
    if !ordered {
        return Err(corrupt("segment key ranges overlap"));
    }
    let changes = replay(&wal_path(dir, wal_seq), schema, repair)?;
    Ok((segments, changes))
}

/// The shared lock of a database opened read-only, if its directory has a
/// `LOCK` file.
pub(crate) struct ReadLock {
    _lock: Option<File>,
}

/// Open the database in `dir` without changing any file.
pub(crate) fn open_read_only(dir: &Path, schema: &Schema) -> Result<(ReadLock, Recovered)> {
    // A backup has no `LOCK`, and no writer can have it open without one.
    let lock = match File::open(dir.join("LOCK")) {
        Ok(lock) => {
            check_lock(dir, lock.try_lock_shared())?;
            Some(lock)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let bytes = fs::read(manifest_path(dir)).map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            RingDbError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("{} holds no database", dir.display()),
            ))
        } else {
            error.into()
        }
    })?;
    let (manifest, wal_seq) = read_manifest(dir, &bytes, schema)?;
    let (segments, changes) = load(dir, schema, &manifest, wal_seq, false)?;
    Ok((
        ReadLock { _lock: lock },
        Recovered {
            manifest,
            segments,
            changes,
        },
    ))
}

impl Store {
    /// Open or create the database in `dir`, which must hold `schema`.
    pub(crate) fn open(dir: &Path, schema: &Schema) -> Result<(Self, Recovered)> {
        fs::create_dir_all(dir)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join("LOCK"))?;
        check_lock(dir, lock.try_lock())?;

        let (manifest, wal_seq) = match fs::read(manifest_path(dir)) {
            Ok(bytes) => read_manifest(dir, &bytes, schema)?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => Self::initialize(dir, schema)?,
            Err(error) => return Err(error.into()),
        };
        let (segments, changes) = load(dir, schema, &manifest, wal_seq, true)?;

        let wal = wal_path(dir, wal_seq);
        let wal = if wal.exists() && fs::metadata(&wal)?.len() >= WAL_HEADER_LEN {
            OpenOptions::new().append(true).open(&wal)?
        } else {
            create_wal(dir, wal_seq)?
        };
        let store = Self {
            dir: dir.to_path_buf(),
            _lock: lock,
            wal,
            wal_seq,
            failed: false,
        };
        store.remove_unreferenced(&manifest)?;
        Ok((
            store,
            Recovered {
                manifest,
                segments,
                changes,
            },
        ))
    }

    /// Write the manifest of an empty database into `dir`. Its log is created
    /// like a missing log, so a crash here leaves no file but `MANIFEST.tmp`.
    fn initialize(dir: &Path, schema: &Schema) -> Result<(Manifest, u64)> {
        let unexpected = fs::read_dir(dir)?
            .map(|entry| entry.map(|e| e.file_name()))
            .collect::<io::Result<Vec<_>>>()?
            .into_iter()
            .find(|name| name != "LOCK" && name != "MANIFEST.tmp");
        if let Some(name) = unexpected {
            return Err(corrupt(format!(
                "{} has no MANIFEST but contains {}",
                dir.display(),
                name.to_string_lossy()
            )));
        }
        let manifest = Manifest {
            next_key: 0,
            next_segment: 0,
            parts: Vec::new(),
        };
        write_atomic(
            dir,
            &manifest_path(dir),
            &encode_manifest(schema, &manifest, 0),
        )?;
        Ok((manifest, 0))
    }

    /// Remove the segment files and logs that the manifest does not name, and a
    /// left-over `MANIFEST.tmp`. Other files are not touched.
    fn remove_unreferenced(&self, manifest: &Manifest) -> Result<()> {
        let keep: Vec<String> = manifest
            .parts
            .iter()
            .map(|(id, _)| format!("seg-{id:016x}.seg"))
            .chain([format!("wal-{:016x}.log", self.wal_seq)])
            .collect();
        for entry in fs::read_dir(&self.dir)? {
            let name = entry?.file_name();
            let name = name.to_string_lossy();
            let ours = name == "MANIFEST.tmp"
                || is_numbered(&name, "seg-", ".seg")
                || is_numbered(&name, "wal-", ".log");
            if ours && !keep.iter().any(|k| *k == name) {
                fs::remove_file(self.dir.join(&*name))?;
            }
        }
        Ok(())
    }

    /// Append encoded changes to the log and sync them.
    pub(crate) fn log(&mut self, frame: &LogFrame) -> Result<()> {
        if self.failed {
            return Err(RingDbError::Io(io::Error::other(
                "an earlier log write failed; the next commit starts a new log",
            )));
        }
        let payload = frame.0.as_bytes();
        let len = u32::try_from(payload.len())
            .map_err(|_| RingDbError::InvalidRecord("a batch must be smaller than 4 GiB".into()))?;
        let mut bytes = Vec::with_capacity(8 + payload.len());
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
        bytes.extend_from_slice(payload);
        let result = self
            .wal
            .write_all(&bytes)
            .and_then(|()| self.wal.sync_data());
        if result.is_err() {
            self.failed = true;
        }
        Ok(result?)
    }

    /// Whether writes wait for a commit to start a new log.
    pub(crate) fn needs_new_log(&self) -> bool {
        self.failed
    }

    #[cfg(test)]
    pub(crate) fn fail(&mut self) {
        self.failed = true;
    }

    /// Make `manifest` current. With `new_log`, also start a new log, so the
    /// changes logged so far are dropped: `manifest` must contain them.
    pub(crate) fn write_manifest(
        &mut self,
        schema: &Schema,
        manifest: &Manifest,
        new_log: bool,
    ) -> Result<()> {
        let old = self.wal_seq;
        let seq = if new_log { old + 1 } else { old };
        let wal = if new_log {
            Some(create_wal(&self.dir, seq)?)
        } else {
            sync_dir(&self.dir)?;
            None
        };
        let bytes = encode_manifest(schema, manifest, seq);
        write_atomic(&self.dir, &manifest_path(&self.dir), &bytes)?;
        if let Some(wal) = wal {
            self.wal = wal;
            self.wal_seq = seq;
            self.failed = false;
            // The manifest no longer names the old log; a failed removal
            // leaves it for the next open to remove.
            let _ = fs::remove_file(wal_path(&self.dir, old));
        }
        Ok(())
    }

    /// Remove the file of a segment that no manifest names any more.
    pub(crate) fn remove_segment(&self, id: u64) {
        // A file left behind is removed at the next open.
        let _ = fs::remove_file(segment_path(&self.dir, id));
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    /// Copy the current files into `dest`, which must not exist or be empty.
    pub(crate) fn backup(&self, dest: &Path, segments: &[u64]) -> Result<()> {
        fs::create_dir_all(dest)?;
        if fs::read_dir(dest)?.next().is_some() {
            return Err(RingDbError::Io(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("backup directory {} is not empty", dest.display()),
            )));
        }
        let names = segments
            .iter()
            .map(|&id| format!("seg-{id:016x}.seg"))
            .chain([
                format!("wal-{:016x}.log", self.wal_seq),
                "MANIFEST".to_string(),
            ]);
        for name in names {
            let target = dest.join(&name);
            fs::copy(self.dir.join(&name), &target)?;
            File::open(&target)?.sync_all()?;
        }
        sync_dir(dest)?;
        Ok(())
    }
}
