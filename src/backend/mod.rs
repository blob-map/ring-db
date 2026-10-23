//! Exact distance kernels over one block of row-major vectors.
//!
//! The CPU kernels read host vectors directly. The Vulkan device keeps its own
//! resident copy per block. Callers validate inputs and handle empty masks.
pub mod cpu;
pub mod vulkan;

pub use vulkan::{VulkanDevice, VulkanSegment};

/// Rows of one block eligible for a search. `Empty` never reaches a kernel.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Mask<'a> {
    All,
    /// Sorted, unique row indexes.
    Sparse(&'a [u32]),
    /// One bit per row, least-significant bit first.
    Dense(&'a [u64]),
}

/// One disk prepared for a kernel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DiskSpec<'a> {
    pub(crate) center: &'a [f32],
    pub(crate) weights: Option<&'a [f32]>,
    pub(crate) radius_sq: f32,
}

/// A nearest-neighbor candidate identified by its row in the block.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RowHit {
    pub(crate) row: u32,
    pub(crate) dist_sq: f32,
}

#[inline]
pub(crate) fn compare_row_hits(a: &RowHit, b: &RowHit) -> std::cmp::Ordering {
    a.dist_sq
        .total_cmp(&b.dist_sq)
        .then_with(|| a.row.cmp(&b.row))
}
