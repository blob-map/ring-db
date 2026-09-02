/// Binary file-format utilities for ring-db persistence.
///
/// All numeric values are stored in **little-endian** byte order so the format
/// is portable across architectures.
///
/// ### Layout of a persisted database folder
///
/// | File | Content |
/// |------|---------|
/// | `meta.bin` | `dims` (u64 LE) + `n_vectors` (u64 LE) = 16 bytes |
/// | `vectors.bin` | `n_vectors × dims` f32 values in row-major order |
/// | `norms_sq.bin` | `n_vectors` precomputed squared L2 norms (f32) |
/// | `payloads.bin` | concatenated bincode bytes for every payload |
/// | `offsets.bin` | `n_vectors + 1` byte offsets (u64) into `payloads.bin` |
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{Result, RingDbError};

// ── Private generic helpers ───────────────────────────────────────────────────

/// Write a slice of `Copy` values as raw little-endian bytes.
///
/// `to_le` converts one value to its `N`-byte little-endian representation.
fn write_le_file<T: Copy, const N: usize>(
    path: &Path,
    data: &[T],
    to_le: fn(T) -> [u8; N],
) -> Result<()> {
    let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
    for &x in data {
        w.write_all(&to_le(x))?;
    }
    Ok(())
}

/// Read a file of raw little-endian values into a `Vec<T>`.
///
/// Returns `RingDbError::Corrupt` if the file length is not a multiple of `N`.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn read_le_file<T, const N: usize>(path: &Path, from_le: fn([u8; N]) -> T) -> Result<Vec<T>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % N != 0 {
        return Err(RingDbError::Corrupt(format!(
            "file '{}' has {} bytes (not a multiple of {})",
            path.display(),
            bytes.len(),
            N,
        )));
    }
    // chunks_exact(N) guarantees each chunk is exactly N bytes, so
    // try_into().unwrap() is always safe here.
    Ok(bytes
        .chunks_exact(N)
        .map(|b| from_le(b.try_into().unwrap()))
        .collect())
}

// ── Meta ─────────────────────────────────────────────────────────────────────

/// Write `(dims, n_vectors)` as two little-endian u64 values.
pub fn write_meta(path: &Path, dims: usize, n_vectors: usize) -> Result<()> {
    let mut f = File::create(path)?;
    f.write_all(&(dims as u64).to_le_bytes())?;
    f.write_all(&(n_vectors as u64).to_le_bytes())?;
    Ok(())
}

/// Read `(dims, n_vectors)` from the meta file.
pub fn read_meta(path: &Path) -> Result<(usize, usize)> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 16 {
        return Err(RingDbError::Corrupt(format!(
            "meta file '{}' is too short ({} bytes, expected ≥ 16)",
            path.display(),
            bytes.len()
        )));
    }
    // Both slices are exactly 8 bytes after the length check above.
    let dims = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let n_vectors = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    Ok((dims, n_vectors))
}

// ── f32 vectors ──────────────────────────────────────────────────────────────

/// Write a `&[f32]` slice as raw little-endian bytes.
pub fn write_f32_file(path: &Path, data: &[f32]) -> Result<()> {
    write_le_file(path, data, f32::to_le_bytes)
}

/// Read a file of raw little-endian f32 values into a `Vec<f32>`.
pub fn read_f32_file(path: &Path) -> Result<Vec<f32>> {
    read_le_file(path, f32::from_le_bytes)
}

// ── u64 offsets ──────────────────────────────────────────────────────────────

/// Write a `&[u64]` slice as raw little-endian bytes.
pub fn write_u64_file(path: &Path, data: &[u64]) -> Result<()> {
    write_le_file(path, data, u64::to_le_bytes)
}

/// Read a file of raw little-endian u64 values into a `Vec<u64>`.
pub fn read_u64_file(path: &Path) -> Result<Vec<u64>> {
    read_le_file(path, u64::from_le_bytes)
}

// ── File move ────────────────────────────────────────────────────────────────

/// Move `src` to `dst`.
///
/// Tries an atomic `rename` first (O(1) on the same filesystem). Falls back to
/// a copy-then-delete if the two paths span different mount points.
pub fn move_file(src: &Path, dst: &Path) -> Result<()> {
    match std::fs::rename(src, dst) {
        Ok(()) => Ok(()),
        Err(_) => {
            std::fs::copy(src, dst)?;
            std::fs::remove_file(src)?;
            Ok(())
        }
    }
}
