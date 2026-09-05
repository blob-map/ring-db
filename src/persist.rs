/// Binary file-format utilities for ring-db persistence.
///
/// Database geometry and vector values are stored in **little-endian** byte
/// order. Pod payload bytes retain their native layout, so the header records
/// and validates the originating byte order.
///
/// ### Layout of a persisted database folder
///
/// | File | Content |
/// |------|---------|
/// | `meta.bin` | versioned database geometry and Pod layout |
/// | `vectors.bin` | `n_vectors × dims` f32 values in row-major order |
/// | `payloads.bin` | fixed-size Pod records |
/// | `metadata.bin` | equality-tag posting sets |
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::mem::size_of;
use std::path::Path;

use crate::error::{Result, RingDbError};

// ── Meta ─────────────────────────────────────────────────────────────────────

const META_MAGIC: &[u8; 8] = b"RINGDB\0\0";
const META_VERSION: u32 = 1;
const ENDIAN_MARKER: u32 = 0x0102_0304;
const META_LEN: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DatabaseMeta {
    pub(crate) dims: usize,
    pub(crate) n_vectors: usize,
    pub(crate) payload_size: usize,
    pub(crate) payload_align: usize,
}

pub(crate) fn write_meta(path: &Path, meta: DatabaseMeta) -> Result<()> {
    let mut file = File::create(path)?;
    file.write_all(META_MAGIC)?;
    file.write_all(&META_VERSION.to_le_bytes())?;
    file.write_all(&ENDIAN_MARKER.to_ne_bytes())?;
    for value in [
        meta.dims,
        meta.n_vectors,
        meta.payload_size,
        meta.payload_align,
    ] {
        let value = u64::try_from(value)
            .map_err(|_| RingDbError::Corrupt("database metadata exceeds u64".to_string()))?;
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

pub(crate) fn read_meta(path: &Path) -> Result<DatabaseMeta> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != META_LEN {
        return Err(RingDbError::Corrupt(format!(
            "meta file '{}' has {} bytes, expected {META_LEN}",
            path.display(),
            bytes.len(),
        )));
    }
    if &bytes[..8] != META_MAGIC {
        return Err(RingDbError::Corrupt("invalid database magic".to_string()));
    }

    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != META_VERSION {
        return Err(RingDbError::Corrupt(format!(
            "unsupported database format version {version}; expected {META_VERSION}"
        )));
    }
    if bytes[12..16] != ENDIAN_MARKER.to_ne_bytes() {
        return Err(RingDbError::Corrupt(
            "Pod payload endianness differs from this host".to_string(),
        ));
    }

    let read_usize = |offset: usize| -> Result<usize> {
        let value = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        usize::try_from(value)
            .map_err(|_| RingDbError::Corrupt(format!("metadata value {value} does not fit usize")))
    };
    let meta = DatabaseMeta {
        dims: read_usize(16)?,
        n_vectors: read_usize(24)?,
        payload_size: read_usize(32)?,
        payload_align: read_usize(40)?,
    };
    if meta.dims == 0 {
        return Err(RingDbError::Corrupt(
            "database dimensions must be greater than zero".to_string(),
        ));
    }
    if meta.n_vectors > u32::MAX as usize {
        return Err(RingDbError::Corrupt(format!(
            "database contains {} vectors, exceeding the u32 ID limit",
            meta.n_vectors
        )));
    }
    if meta.payload_align == 0 || !meta.payload_align.is_power_of_two() {
        return Err(RingDbError::Corrupt(format!(
            "invalid payload alignment {}",
            meta.payload_align
        )));
    }
    Ok(meta)
}

// ── f32 vectors ──────────────────────────────────────────────────────────────

/// Write a `&[f32]` slice as raw little-endian bytes.
pub(crate) fn write_f32_file(path: &Path, data: &[f32]) -> Result<()> {
    let mut writer = BufWriter::with_capacity(1 << 20, File::create(path)?);
    if cfg!(target_endian = "little") {
        writer.write_all(bytemuck::cast_slice(data))?;
    } else {
        for &value in data {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Read exactly `expected_values` little-endian f32 values.
pub(crate) fn read_f32_file(path: &Path, expected_values: usize) -> Result<Vec<f32>> {
    let expected_bytes = expected_values
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| RingDbError::Corrupt("vector byte length overflows usize".to_string()))?;
    let expected_bytes_u64 = u64::try_from(expected_bytes)
        .map_err(|_| RingDbError::Corrupt("vector byte length does not fit u64".to_string()))?;

    let mut file = File::open(path)?;
    let actual_bytes = file.metadata()?.len();
    if actual_bytes != expected_bytes_u64 {
        return Err(RingDbError::Corrupt(format!(
            "file '{}' has {actual_bytes} bytes, expected {expected_bytes}",
            path.display()
        )));
    }

    let mut values = vec![0.0f32; expected_values];
    file.read_exact(bytemuck::cast_slice_mut(&mut values))?;
    if cfg!(target_endian = "big") {
        for value in &mut values {
            *value = f32::from_bits(u32::from_le(value.to_bits()));
        }
    }
    Ok(values)
}
