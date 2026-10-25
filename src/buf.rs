//! Immutable arrays that own their memory or borrow it from a mapped file.
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use bytemuck::Pod;
use memmap2::Mmap;

/// An immutable array, kept alive by its owner: a vector, or a file mapping.
/// Clones share the elements.
pub(crate) struct Buf<T> {
    ptr: *const T,
    len: usize,
    _owner: Arc<dyn Send + Sync>,
}

// SAFETY: a `Buf` only reads its elements, which its owner keeps alive and
// unchanged, so it can move and be shared like a `&[T]`.
unsafe impl<T: Sync> Send for Buf<T> {}
unsafe impl<T: Sync> Sync for Buf<T> {}

impl<T: Pod + Send + Sync> From<Vec<T>> for Buf<T> {
    fn from(values: Vec<T>) -> Self {
        // Moving the vector into the `Arc` leaves its elements in place.
        Self {
            ptr: values.as_ptr(),
            len: values.len(),
            _owner: Arc::new(values),
        }
    }
}

impl<T: Pod + Send + Sync> Default for Buf<T> {
    fn default() -> Self {
        Vec::new().into()
    }
}

impl<T: Pod> Buf<T> {
    /// The elements in `range` of `map`, if they are aligned for `T`.
    pub(crate) fn mapped(map: &Arc<Mmap>, range: std::ops::Range<usize>) -> Option<Self> {
        let values: &[T] = bytemuck::try_cast_slice(map.get(range)?).ok()?;
        Some(Self {
            ptr: values.as_ptr(),
            len: values.len(),
            _owner: map.clone(),
        })
    }
}

impl<T> Deref for Buf<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` and `len` describe initialized, aligned elements that
        // `_owner` keeps alive and never changes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> Clone for Buf<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            _owner: self._owner.clone(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Buf<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}
