//! Little-endian binary encoding shared by storage files.
//!
//! Integers are little-endian. A slice is its element count as `u64`, then its
//! elements, then zero padding to a multiple of 8 bytes. Decoding checks every bound and
//! returns [`RingDbError::Corrupt`] instead of panicking.
use bytemuck::Pod;

use crate::error::{Result, RingDbError};

pub(crate) fn corrupt(message: impl Into<String>) -> RingDbError {
    RingDbError::Corrupt(message.into())
}

#[derive(Default)]
pub(crate) struct Encoder {
    pub(crate) bytes: Vec<u8>,
}

impl Encoder {
    pub(crate) fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Length-prefixed bytes without padding.
    pub(crate) fn bytes(&mut self, bytes: &[u8]) {
        self.u32(u32::try_from(bytes.len()).expect("values are smaller than 4 GiB"));
        self.bytes.extend_from_slice(bytes);
    }

    /// A count, then the little-endian values, padded to 8 bytes.
    pub(crate) fn slice<T: Pod + ToLe>(&mut self, values: &[T]) {
        self.u64(values.len() as u64);
        if cfg!(target_endian = "little") {
            self.bytes.extend_from_slice(bytemuck::cast_slice(values));
        } else {
            values.iter().for_each(|v| v.put_le(&mut self.bytes));
        }
        let len = size_of_val(values);
        self.bytes
            .resize(self.bytes.len() + len.next_multiple_of(8) - len, 0);
    }

    /// Append the CRC-32 of everything written so far.
    pub(crate) fn finish_with_crc(mut self) -> Vec<u8> {
        let crc = crc32fast::hash(&self.bytes);
        self.u32(crc);
        self.bytes
    }
}

pub(crate) struct Decoder<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Decoder<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    /// A decoder over `bytes` without its CRC-32 trailer, after checking it.
    pub(crate) fn with_crc(bytes: &'a [u8], what: &str) -> Result<Self> {
        let body = bytes
            .len()
            .checked_sub(4)
            .ok_or_else(|| corrupt(format!("{what} is truncated")))?;
        let stored = u32::from_le_bytes(bytes[body..].try_into().expect("four bytes"));
        if crc32fast::hash(&bytes[..body]) != stored {
            return Err(corrupt(format!("{what} has a wrong checksum")));
        }
        Ok(Self::new(&bytes[..body]))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.at == self.bytes.len()
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| corrupt("unexpected end of data"))?;
        let bytes = &self.bytes[self.at..end];
        self.at = end;
        Ok(bytes)
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    pub(crate) fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.u32()? as usize;
        self.take(len)
    }

    pub(crate) fn string(&mut self) -> Result<String> {
        let bytes = self.bytes()?;
        std::str::from_utf8(bytes)
            .map(str::to_string)
            .map_err(|_| corrupt("text is not UTF-8"))
    }

    pub(crate) fn slice<T: Pod + ToLe>(&mut self) -> Result<Vec<T>> {
        let count = usize::try_from(self.u64()?).map_err(|_| corrupt("slice is too long"))?;
        let len = count
            .checked_mul(size_of::<T>())
            .ok_or_else(|| corrupt("slice is too long"))?;
        let bytes = self.take(len)?;
        self.take(len.next_multiple_of(8) - len)?;
        let mut values: Vec<T> = bytemuck::pod_collect_to_vec(bytes);
        if cfg!(target_endian = "big") {
            values.iter_mut().for_each(ToLe::swap);
        }
        Ok(values)
    }
}

/// Pod values stored little-endian.
pub(crate) trait ToLe: Copy {
    fn put_le(self, out: &mut Vec<u8>);
    fn swap(&mut self);
}

macro_rules! to_le {
    ($($t:ty),*) => {$(
        impl ToLe for $t {
            fn put_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }

            fn swap(&mut self) {
                *self = <$t>::from_le_bytes(self.to_ne_bytes());
            }
        }
    )*};
}

to_le!(u8, u32, u64, i64, f32);

impl ToLe for crate::query::Key {
    fn put_le(self, out: &mut Vec<u8>) {
        self.0.put_le(out);
    }

    fn swap(&mut self) {
        self.0.swap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip_and_bounds_are_checked() {
        let mut e = Encoder::default();
        e.u8(7);
        e.u32(0xDEAD_BEEF);
        e.slice(&[1.5f32, -2.0, 3.25]);
        e.bytes(b"hello");
        e.slice::<u64>(&[]);
        e.u64(u64::MAX);
        let bytes = e.finish_with_crc();

        let mut d = Decoder::with_crc(&bytes, "test").unwrap();
        assert_eq!(d.u8().unwrap(), 7);
        assert_eq!(d.u32().unwrap(), 0xDEAD_BEEF);
        assert_eq!(d.slice::<f32>().unwrap(), [1.5, -2.0, 3.25]);
        assert_eq!(d.string().unwrap(), "hello");
        assert!(d.slice::<u64>().unwrap().is_empty());
        assert_eq!(d.u64().unwrap(), u64::MAX);
        assert!(d.is_empty());
        assert!(d.u8().is_err());

        let mut flipped = bytes.clone();
        flipped[3] ^= 1;
        assert!(Decoder::with_crc(&flipped, "test").is_err());
        assert!(Decoder::with_crc(&bytes[..2], "test").is_err());
        // A huge count must not allocate or panic.
        let mut e = Encoder::default();
        e.u64(u64::MAX / 2);
        assert!(Decoder::new(&e.bytes).slice::<u32>().is_err());
    }
}
