//! Little-endian binary encoding shared by storage files.
//!
//! Integers are little-endian. A slice is its element count as `u64`, then its
//! elements, then zero padding to a multiple of 8 bytes. In an aligned encoding,
//! zero padding also puts each count at a multiple of 8 bytes, so that the
//! elements of a mapped file can be read in place. Decoding checks every bound
//! and returns [`RingDbError::Corrupt`] instead of panicking.
use std::io::{self, Write};
use std::sync::Arc;

use bytemuck::Pod;
use memmap2::Mmap;

use crate::buf::Buf;
use crate::error::{Result, RingDbError};

pub(crate) fn corrupt(message: impl Into<String>) -> RingDbError {
    RingDbError::Corrupt(message.into())
}

#[derive(Default)]
pub(crate) struct Encoder<'w> {
    bytes: Vec<u8>,
    aligned: bool,
    sink: Option<Sink<'w>>,
}

/// Where a streaming encoder writes full buffers and large slices.
struct Sink<'w> {
    writer: &'w mut dyn Write,
    crc: crc32fast::Hasher,
    /// Bytes written to `writer` so far.
    written: usize,
    /// The first write error; later writes are skipped.
    error: Option<io::Error>,
}

/// Bytes that a streaming encoder buffers before it writes them.
const STREAM_BUFFER: usize = 1 << 20;

impl Encoder<'_> {
    /// An encoder of slices that start at multiples of 8 bytes.
    pub(crate) fn aligned() -> Self {
        Self {
            aligned: true,
            ..Self::default()
        }
    }

    /// Position of the next byte in the encoding.
    fn position(&self) -> usize {
        self.sink.as_ref().map_or(0, |sink| sink.written) + self.bytes.len()
    }

    fn pad(&mut self) {
        let padding = self.position().next_multiple_of(8) - self.position();
        self.bytes.resize(self.bytes.len() + padding, 0);
    }

    /// Write the buffer, then `bytes`, to the sink.
    fn flush(&mut self, bytes: &[u8]) {
        let Some(sink) = &mut self.sink else {
            return;
        };
        for chunk in [&self.bytes[..], bytes] {
            if sink.error.is_none() {
                sink.crc.update(chunk);
                sink.written += chunk.len();
                sink.error = sink.writer.write_all(chunk).err();
            }
        }
        self.bytes.clear();
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The magic bytes and format version that start a file.
    pub(crate) fn header(&mut self, magic: &[u8; 8], version: u32) {
        self.bytes.extend_from_slice(magic);
        self.u32(version);
    }

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
        if self.aligned {
            self.pad();
        }
        self.u64(values.len() as u64);
        let len = size_of_val(values);
        if cfg!(target_endian = "little") && self.sink.is_some() && len >= STREAM_BUFFER {
            // Write a large slice in place instead of copying it.
            self.flush(bytemuck::cast_slice(values));
        } else if cfg!(target_endian = "little") {
            self.bytes.extend_from_slice(bytemuck::cast_slice(values));
        } else {
            values.iter().for_each(|v| v.put_le(&mut self.bytes));
        }
        self.bytes
            .resize(self.bytes.len() + len.next_multiple_of(8) - len, 0);
        if self.bytes.len() >= STREAM_BUFFER {
            self.flush(&[]);
        }
    }

    /// Append the CRC-32 of everything written so far.
    pub(crate) fn finish_with_crc(mut self) -> Vec<u8> {
        debug_assert!(self.sink.is_none());
        let crc = crc32fast::hash(&self.bytes);
        self.u32(crc);
        self.bytes
    }
}

impl<'w> Encoder<'w> {
    /// An aligned encoder that writes to `writer` as it goes.
    pub(crate) fn aligned_to(writer: &'w mut dyn Write) -> Self {
        Self {
            sink: Some(Sink {
                writer,
                crc: crc32fast::Hasher::new(),
                written: 0,
                error: None,
            }),
            ..Self::aligned()
        }
    }

    /// Write the rest and the CRC-32 of everything to the writer.
    pub(crate) fn finish_to_writer(mut self) -> io::Result<()> {
        self.flush(&[]);
        let sink = self.sink.take().expect("a streaming encoder");
        if let Some(error) = sink.error {
            return Err(error);
        }
        sink.writer.write_all(&sink.crc.finalize().to_le_bytes())
    }
}

pub(crate) struct Decoder<'a> {
    bytes: &'a [u8],
    at: usize,
    aligned: bool,
    /// The mapping that `bytes` start, from which [`Decoder::buf`] borrows.
    map: Option<&'a Arc<Mmap>>,
}

impl<'a> Decoder<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            at: 0,
            aligned: false,
            map: None,
        }
    }

    /// A decoder over a mapped file without its CRC-32 trailer, after checking
    /// it. Its slices borrow from `map` where they can.
    pub(crate) fn mapped(map: &'a Arc<Mmap>, what: &str) -> Result<Self> {
        Ok(Self {
            map: Some(map),
            ..Self::with_crc(map, what)?
        })
    }

    /// Read the rest as an aligned encoding.
    pub(crate) fn set_aligned(&mut self) {
        self.aligned = true;
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

    /// Check a header written by [`Encoder::header`].
    pub(crate) fn header(&mut self, magic: &[u8; 8], version: u32, what: &str) -> Result<()> {
        self.header_versions(magic, version..=version, what)
            .map(drop)
    }

    /// Check a header written by [`Encoder::header`] with one of `versions`,
    /// and return its version.
    pub(crate) fn header_versions(
        &mut self,
        magic: &[u8; 8],
        versions: std::ops::RangeInclusive<u32>,
        what: &str,
    ) -> Result<u32> {
        let (stored, version) = (self.take(8)?, self.u32()?);
        if stored != magic || !versions.contains(&version) {
            return Err(corrupt(format!("{what} has an unknown format")));
        }
        Ok(version)
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
        let bytes = self.slice_bytes::<T>()?;
        let mut values: Vec<T> = bytemuck::pod_collect_to_vec(bytes);
        if cfg!(target_endian = "big") {
            values.iter_mut().for_each(ToLe::swap);
        }
        Ok(values)
    }

    /// A slice, borrowed from the mapped file when its elements are aligned
    /// and in native byte order, and copied otherwise.
    pub(crate) fn buf<T: Pod + ToLe + Send + Sync>(&mut self) -> Result<Buf<T>> {
        let start = self.bytes.as_ptr() as usize;
        let bytes = self.slice_bytes::<T>()?;
        let offset = bytes.as_ptr() as usize - start;
        let borrowed = self
            .map
            .filter(|_| cfg!(target_endian = "little"))
            .and_then(|map| Buf::mapped(map, offset..offset + bytes.len()));
        match borrowed {
            Some(buf) => Ok(buf),
            None => {
                let mut values: Vec<T> = bytemuck::pod_collect_to_vec(bytes);
                if cfg!(target_endian = "big") {
                    values.iter_mut().for_each(ToLe::swap);
                }
                Ok(values.into())
            }
        }
    }

    pub(crate) fn skip_slice<T>(&mut self) -> Result<()> {
        self.slice_bytes::<T>().map(drop)
    }

    /// The bytes of the next slice of `T`, after its count.
    fn slice_bytes<T>(&mut self) -> Result<&'a [u8]> {
        if self.aligned {
            let padding = self.at.next_multiple_of(8) - self.at;
            self.take(padding)?;
        }
        let count = usize::try_from(self.u64()?).map_err(|_| corrupt("slice is too long"))?;
        let len = count
            .checked_mul(size_of::<T>())
            .ok_or_else(|| corrupt("slice is too long"))?;
        let bytes = self.take(len)?;
        self.take(len.next_multiple_of(8) - len)?;
        Ok(bytes)
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

to_le!(u8, u16, u32, u64, i64, f32);

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
        e.header(b"TESTFILE", 3);
        e.u8(7);
        e.u32(0xDEAD_BEEF);
        e.slice(&[1.5f32, -2.0, 3.25]);
        e.bytes(b"hello");
        e.slice::<u64>(&[]);
        e.u64(u64::MAX);
        let bytes = e.finish_with_crc();

        let mut d = Decoder::with_crc(&bytes, "test").unwrap();
        assert!(Decoder::new(&bytes).header(b"TESTFILE", 4, "test").is_err());
        d.header(b"TESTFILE", 3, "test").unwrap();
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
        assert!(Decoder::new(e.as_bytes()).slice::<u32>().is_err());
    }

    #[test]
    fn streaming_encodings_match_buffered_ones() {
        let large: Vec<u64> = (0..(STREAM_BUFFER as u64 / 8) * 3 + 5).collect();
        let encode = |e: &mut Encoder<'_>| {
            e.header(b"TESTFILE", 1);
            for round in 0..3u8 {
                e.u8(round);
                e.slice(&[1u16, 2, 3]);
                e.slice(&large[..large.len() >> round]);
                e.slice(&vec![round; STREAM_BUFFER / 3]);
            }
        };
        let mut buffered = Encoder::aligned();
        encode(&mut buffered);
        let buffered = buffered.finish_with_crc();
        let mut streamed = Vec::new();
        let mut e = Encoder::aligned_to(&mut streamed);
        encode(&mut e);
        e.finish_to_writer().unwrap();
        assert!(streamed == buffered);
        let mut d = Decoder::with_crc(&streamed, "test").unwrap();
        d.header(b"TESTFILE", 1, "test").unwrap();
        d.set_aligned();
        assert_eq!(d.u8().unwrap(), 0);
        assert_eq!(d.slice::<u16>().unwrap(), [1, 2, 3]);
        assert!(d.slice::<u64>().unwrap() == large);
    }
}
