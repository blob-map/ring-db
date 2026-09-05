use std::fs::File;
use std::io::{Read, Write};
use std::mem::size_of;
use std::path::Path;

use bytemuck::Pod;

use crate::error::{Result, RingDbError};

pub(crate) fn write_pod_file<T: Pod>(path: &Path, records: &[T]) -> Result<()> {
    let mut file = File::create(path)?;
    if size_of::<T>() != 0 {
        file.write_all(bytemuck::cast_slice(records))?;
    }
    file.flush()?;
    Ok(())
}

pub(crate) fn read_pod_file<T: Pod>(path: &Path, n_records: usize) -> Result<Box<[T]>> {
    let expected_bytes = n_records.checked_mul(size_of::<T>()).ok_or_else(|| {
        RingDbError::Corrupt(format!(
            "payload byte length overflows for {n_records} records"
        ))
    })?;
    let expected_bytes_u64 = u64::try_from(expected_bytes).map_err(|_| {
        RingDbError::Corrupt(format!(
            "payload byte length does not fit in u64 for {n_records} records"
        ))
    })?;

    let mut file = File::open(path)?;
    let actual_bytes = file.metadata()?.len();
    if actual_bytes != expected_bytes_u64 {
        return Err(RingDbError::Corrupt(format!(
            "payload file '{}' has {actual_bytes} bytes, expected {expected_bytes}",
            path.display()
        )));
    }

    let mut records = vec![T::zeroed(); n_records];
    if expected_bytes != 0 {
        file.read_exact(bytemuck::cast_slice_mut(&mut records))?;
    }
    Ok(records.into_boxed_slice())
}
