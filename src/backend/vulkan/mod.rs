//! Exact Vulkan compute backend. Shaders and their sources are distributed with
//! the crate; neither a shader compiler nor a Vulkan SDK is needed by consumers.
//!
//! A [`VulkanDevice`] owns the queue, pipelines, and reusable query buffers.
//! Each [`VulkanSegment`] holds one block of resident vectors.
mod runtime;

use std::sync::{Arc, Mutex, MutexGuard};

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::backend::{DiskSpec, Mask, RowHit, compare_row_hits};
use crate::error::{Result, RingDbError};
use crate::query::MAX_DISKS;
use runtime::{Buffer, Context, Execution, Pipelines, WORKGROUP_SIZE, barrier, compute_barrier};

const INVALID: u32 = u32::MAX;
const REGION: u32 = 0;
const KNN: u32 = 1;

// Must match shaders/common.glsl. Addresses come first for 8-byte alignment;
// reserved makes the final padding explicit for Pod.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct Params {
    vectors: u64,
    inputs: u64,
    scratch: u64,
    output_address: u64,
    reduce_src: u64,
    reduce_dst: u64,
    dims: u32,
    rows: u32,
    work_count: u32,
    candidate_offset: u32,
    disk_count: u32,
    mode: u32,
    reduce_count: u32,
    output_index: u32,
    row_offset: u32,
    reserved: u32,
}

struct Chunk {
    vectors: Buffer,
    offset: usize,
    rows: usize,
}

/// One block of vectors resident in device memory, split into chunks that
/// fit the device allocation limit.
pub struct VulkanSegment {
    chunks: Vec<Chunk>,
    dims: usize,
}

struct PreparedQuery {
    params: Params,
    inputs: Vec<u32>,
    candidate_mode: usize,
    eligible: usize,
}

struct QueryBuffers {
    inputs: Buffer,
    scratch: Buffer,
    output: Buffer,
    count: Buffer,
    readback: Buffer,
}

impl QueryBuffers {
    fn new(context: &Arc<Context>) -> Result<Self> {
        Ok(Self {
            inputs: Buffer::new(context, 4, true)?,
            scratch: Buffer::new(context, 4, false)?,
            output: Buffer::new(context, 4, false)?,
            count: Buffer::new(context, 4, true)?,
            readback: Buffer::new(context, 4, true)?,
        })
    }
}

struct ExecutionState {
    execution: Execution,
    // Reused under the same mutex as submissions. Capacities grow on demand;
    // no result data escapes as a reference into these buffers.
    buffers: Option<QueryBuffers>,
}

impl ExecutionState {
    fn buffers(&mut self, context: &Arc<Context>) -> Result<(&mut Execution, &mut QueryBuffers)> {
        if self.buffers.is_none() {
            self.buffers = Some(QueryBuffers::new(context)?);
        }
        Ok((
            &mut self.execution,
            self.buffers.as_mut().expect("initialized above"),
        ))
    }
}

pub struct VulkanDevice {
    execution: Mutex<ExecutionState>,
    pipelines: Pipelines,
    context: Arc<Context>,
}

impl VulkanDevice {
    pub fn new() -> Result<Self> {
        let context = Context::new()?;
        let pipelines = Pipelines::new(&context, size_of::<Params>() as u32)?;
        let execution = Mutex::new(ExecutionState {
            execution: Execution::new(&context)?,
            buffers: None,
        });
        Ok(Self {
            execution,
            pipelines,
            context,
        })
    }

    fn execution(&self) -> Result<MutexGuard<'_, ExecutionState>> {
        self.execution
            .lock()
            .map_err(|_| RingDbError::Backend("Vulkan execution mutex is poisoned".to_string()))
    }

    /// Upload row-major `vectors`. Each chunk is transposed independently so
    /// coordinate reads are coalesced across rows.
    pub(crate) fn upload(&self, dims: usize, vectors: &[f32]) -> Result<VulkanSegment> {
        if dims == 0 || dims > u32::MAX as usize || !vectors.len().is_multiple_of(dims) {
            return Err(RingDbError::Backend(
                "invalid Vulkan dataset dimensions".to_string(),
            ));
        }
        let rows = vectors.len() / dims;
        if rows > u32::MAX as usize {
            return Err(size_error());
        }
        let limit = usize::try_from(self.context.allocation_limit).unwrap_or(usize::MAX);
        let chunk_rows = chunk_row_limit(dims, vector_byte_limit(limit)?, limit)?;
        let mut chunks = Vec::new();
        for offset in (0..rows).step_by(chunk_rows) {
            let count = (rows - offset).min(chunk_rows);
            let source = &vectors[offset * dims..(offset + count) * dims];
            let mut transposed = vec![0.0f32; source.len()];
            for (id, row) in source.chunks_exact(dims).enumerate() {
                for (dim, value) in row.iter().enumerate() {
                    transposed[dim * count + id] = *value;
                }
            }
            let vectors = self.resident(bytemuck::cast_slice(&transposed))?;
            chunks.push(Chunk {
                vectors,
                offset,
                rows: count,
            });
        }
        Ok(VulkanSegment { chunks, dims })
    }

    fn resident(&self, bytes: &[u8]) -> Result<Buffer> {
        let mut staging = Buffer::new(&self.context, bytes.len(), true)?;
        staging.write(bytes)?;
        let buffer = Buffer::new(&self.context, bytes.len(), false)?;
        self.execution()?
            .execution
            .copy(&staging, 0, &buffer, bytes.len())?;
        Ok(buffer)
    }

    /// The `k` best rows of `segment` by `(dist_sq, row)`, sorted.
    pub(crate) fn knn(
        &self,
        segment: &VulkanSegment,
        query: &[f32],
        weights: Option<&[f32]>,
        k: usize,
        mask: Mask<'_>,
    ) -> Result<Vec<RowHit>> {
        let disk = DiskSpec {
            center: query,
            weights,
            radius_sq: 0.0,
        };
        let mut hits = Vec::new();
        for chunk in &segment.chunks {
            let mut next = self.knn_chunk(self.prepare(segment, chunk, &[disk], mask)?, k)?;
            if hits.is_empty() {
                hits = next;
            } else {
                hits.append(&mut next);
                if hits.len() > k {
                    hits.select_nth_unstable_by(k, compare_row_hits);
                    hits.truncate(k);
                }
            }
        }
        if segment.chunks.len() > 1 {
            hits.sort_unstable_by(compare_row_hits);
        }
        Ok(hits)
    }

    /// Append one record per row of `segment` inside all `disks` to
    /// `records`: the row plus `base`, then one distance per disk.
    pub(crate) fn region(
        &self,
        segment: &VulkanSegment,
        base: u32,
        disks: &[DiskSpec<'_>],
        mask: Mask<'_>,
        records: &mut Vec<u32>,
    ) -> Result<()> {
        if disks.is_empty() || disks.len() > MAX_DISKS {
            return Err(RingDbError::InvalidQuery(format!(
                "a region needs between 1 and {MAX_DISKS} disks"
            )));
        }
        for chunk in &segment.chunks {
            let mut query = self.prepare(segment, chunk, disks, mask)?;
            query.params.row_offset = base + chunk.offset as u32;
            self.region_chunk(query, records)?;
        }
        Ok(())
    }

    fn prepare(
        &self,
        segment: &VulkanSegment,
        chunk: &Chunk,
        disks: &[DiskSpec<'_>],
        mask: Mask<'_>,
    ) -> Result<PreparedQuery> {
        let dims = segment.dims;
        let disk_words = dims
            .checked_mul(2)
            .and_then(|n| n.checked_add(2))
            .and_then(|n| n.checked_mul(disks.len()))
            .ok_or_else(size_error)?;
        let end = chunk.offset + chunk.rows;
        let mask = match mask {
            Mask::Sparse(ids) => {
                let start = ids.partition_point(|&id| (id as usize) < chunk.offset);
                let stop = ids.partition_point(|&id| (id as usize) < end);
                Mask::Sparse(&ids[start..stop])
            }
            other => other,
        };
        let first_word = chunk.offset / 32;
        let candidate_end = end.div_ceil(32);
        let (candidate_mode, work_count, eligible, candidate_words) = match mask {
            Mask::All => (0, chunk.rows, chunk.rows, 0),
            Mask::Sparse(ids) => (1, ids.len(), ids.len(), ids.len()),
            Mask::Dense(words) => {
                let eligible = (first_word..candidate_end)
                    .map(|i| {
                        let low = chunk.offset.saturating_sub(i * 32);
                        let high = (end - i * 32).min(32);
                        let mask = (u32::MAX >> (32 - high)) & (u32::MAX << low);
                        (dense_word(words, i) & mask).count_ones() as usize
                    })
                    .sum();
                (2, chunk.rows, eligible, candidate_end - first_word)
            }
        };
        let total_words = disk_words
            .checked_add(candidate_words)
            .ok_or_else(size_error)?;
        // Shader query/candidate indices are u32; byte addresses are u64.
        u32::try_from(total_words).map_err(|_| size_error())?;
        self.context.check_buffer_size(word_bytes(total_words)?)?;
        let mut data = Vec::with_capacity(total_words);
        for disk in disks {
            if disk.center.len() != dims || disk.weights.is_some_and(|w| w.len() != dims) {
                return Err(RingDbError::Backend(
                    "Vulkan query dimensions do not match dataset".to_string(),
                ));
            }
            data.push(u32::from(disk.weights.is_some()));
            data.push(disk.radius_sq.to_bits());
            data.extend(disk.center.iter().map(|v| v.to_bits()));
            match disk.weights {
                Some(weights) => data.extend(weights.iter().map(|w| w.to_bits())),
                None => data.resize(data.len() + dims, 0),
            }
        }
        match mask {
            Mask::Sparse(ids) => data.extend_from_slice(ids),
            // Explicitly split into low/high u32 words: behavior does not
            // depend on host endianness.
            Mask::Dense(words) => {
                data.extend((first_word..candidate_end).map(|i| dense_word(words, i)))
            }
            Mask::All => {}
        }
        Ok(PreparedQuery {
            params: Params {
                dims: dims as u32,
                rows: chunk.rows as u32,
                row_offset: chunk.offset as u32,
                vectors: chunk.vectors.address,
                work_count: work_count as u32,
                candidate_offset: disk_words as u32,
                disk_count: disks.len() as u32,
                ..Params::default()
            },
            inputs: data,
            candidate_mode,
            eligible,
        })
    }

    fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        params: &Params,
        count: u32,
    ) {
        // SAFETY: the mutex protects the command buffer. All buffers referenced
        // by the pushed addresses outlive the synchronous submission.
        unsafe {
            self.context
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.context.device.cmd_push_constants(
                cmd,
                self.pipelines.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(params),
            );
        }
        self.context.dispatch(cmd, count);
    }

    fn region_chunk(&self, mut query: PreparedQuery, records: &mut Vec<u32>) -> Result<()> {
        if query.eligible == 0 {
            return Ok(());
        }
        let stride = 1 + query.params.disk_count as usize;
        let output_words = query
            .eligible
            .checked_mul(stride)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(size_error)?;
        query.params.mode = REGION;
        let mut state = self.execution()?;
        let (execution, buffers) = state.buffers(&self.context)?;
        buffers.inputs.grow(word_bytes(query.inputs.len())?)?;
        buffers.inputs.write(bytemuck::cast_slice(&query.inputs))?;
        buffers.output.grow(word_bytes(output_words)?)?;
        query.params.inputs = buffers.inputs.address;
        query.params.output_address = buffers.output.address;
        execution.run("region", |cmd| {
            unsafe {
                self.context
                    .device
                    .cmd_fill_buffer(cmd, buffers.output.raw, 0, 4, 0);
            }
            barrier(
                &self.context.device,
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            );
            self.dispatch(
                cmd,
                self.pipelines.scan[query.candidate_mode],
                &query.params,
                query.params.work_count,
            );
            barrier(
                &self.context.device,
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
            );
            // Return the count in the scan submission, avoiding a separate
            // command-buffer submission and fence wait for four bytes.
            unsafe {
                self.context.device.cmd_copy_buffer(
                    cmd,
                    buffers.output.raw,
                    buffers.count.raw,
                    &[vk::BufferCopy::default().size(4)],
                );
            }
        })?;
        let count = buffers.count.read::<u32>(1)?[0] as usize;
        if count > query.eligible {
            return Err(RingDbError::Backend(
                "Vulkan returned an invalid hit count".to_string(),
            ));
        }
        if count == 0 {
            return Ok(());
        }
        // Records arrive in their final layout: one plain copy.
        let words = count * stride;
        buffers.readback.grow(word_bytes(words)?)?;
        execution.copy(&buffers.output, 4, &buffers.readback, word_bytes(words)?)?;
        buffers.readback.read_into(records, 0, words)
    }

    fn knn_chunk(&self, mut query: PreparedQuery, k: usize) -> Result<Vec<RowHit>> {
        let k = k.min(query.eligible);
        if k == 0 {
            return Ok(Vec::new());
        }
        query.params.mode = KNN;
        let work = query.params.work_count;
        let host_selection = k.saturating_mul(64) >= work as usize;
        let groups = work.div_ceil(WORKGROUP_SIZE);
        let distance_words = work as usize * 2;
        let scratch_words = distance_words
            .checked_add(groups as usize * 2)
            .ok_or_else(size_error)?;
        let mut state = self.execution()?;
        let (execution, buffers) = state.buffers(&self.context)?;
        buffers.scratch.grow(word_bytes(scratch_words)?)?;
        buffers.output.grow(if host_selection {
            4
        } else {
            word_bytes(k * 2)?
        })?;
        buffers.inputs.grow(word_bytes(query.inputs.len())?)?;
        buffers.inputs.write(bytemuck::cast_slice(&query.inputs))?;
        query.params.inputs = buffers.inputs.address;
        query.params.scratch = buffers.scratch.address;
        query.params.output_address = buffers.output.address;
        execution.run("knn", |cmd| {
            self.dispatch(
                cmd,
                self.pipelines.scan[query.candidate_mode],
                &query.params,
                work,
            );
            if host_selection {
                return;
            }
            compute_barrier(&self.context, cmd);
            for index in 0..k {
                let mut params = query.params;
                params.reduce_count = work;
                params.reduce_src = u64::MAX;
                params.reduce_dst = distance_words as u64;
                params.output_index = index as u32;
                loop {
                    self.dispatch(cmd, self.pipelines.reduce, &params, params.reduce_count);
                    compute_barrier(&self.context, cmd);
                    let remaining = params.reduce_count.div_ceil(WORKGROUP_SIZE);
                    if remaining == 1 {
                        break;
                    }
                    params.reduce_src = params.reduce_dst;
                    params.reduce_dst = if params.reduce_dst == distance_words as u64 {
                        distance_words as u64 + u64::from(groups)
                    } else {
                        distance_words as u64
                    };
                    params.reduce_count = remaining;
                }
                self.dispatch(cmd, self.pipelines.take, &params, 1);
                compute_barrier(&self.context, cmd);
            }
        })?;
        let (source, count) = if host_selection {
            (&buffers.scratch, work as usize)
        } else {
            (&buffers.output, k)
        };
        let bytes = count * size_of::<RowHit>();
        buffers.readback.grow(bytes)?;
        execution.copy(source, 0, &buffers.readback, bytes)?;
        let mut hits: Vec<RowHit> = buffers.readback.read(count)?;
        hits.retain(|hit| hit.row != INVALID);
        if hits.len() < k {
            return Err(RingDbError::Backend(
                "Vulkan KNN returned fewer candidates than expected".to_string(),
            ));
        }
        hits.sort_unstable_by(compare_row_hits);
        hits.truncate(k);
        Ok(hits)
    }
}

fn size_error() -> RingDbError {
    RingDbError::Backend("Vulkan buffer dimensions exceed addressable size".to_string())
}

fn word_bytes(words: usize) -> Result<usize> {
    words.checked_mul(4).ok_or_else(size_error)
}

fn dense_word(words: &[u64], index: usize) -> u32 {
    (words[index / 2] >> ((index % 2) * 32)) as u32
}

/// `RINGDB_VULKAN_CHUNK_BYTES` lowers the vector-buffer size to exercise chunked
/// paths in tests and benchmarks. It never raises a device limit.
fn vector_byte_limit(limit: usize) -> Result<usize> {
    match std::env::var("RINGDB_VULKAN_CHUNK_BYTES") {
        Ok(value) => Ok(value
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                RingDbError::Backend(
                    "RINGDB_VULKAN_CHUNK_BYTES must be a positive byte count".into(),
                )
            })?
            .min(limit)),
        Err(std::env::VarError::NotPresent) => Ok(limit),
        Err(error) => Err(RingDbError::Backend(error.to_string())),
    }
}

fn chunk_row_limit(dims: usize, vector_bytes: usize, buffer_bytes: usize) -> Result<usize> {
    let row_bytes = word_bytes(dims)?;
    // Region output holds a count plus a row and up to MAX_DISKS distances
    // per row. KNN scratch holds two words per row plus two
    // workgroup-reduction regions.
    let words = buffer_bytes / 4;
    let mut rows = (vector_bytes / row_bytes)
        .min(words.saturating_sub(1) / (1 + MAX_DISKS))
        .min(u32::MAX as usize);
    let groups = rows.div_ceil(WORKGROUP_SIZE as usize);
    rows = rows.min(words.saturating_sub(2 * groups) / 2);
    if rows == 0 {
        return Err(RingDbError::Backend(
            "Vulkan buffer limit cannot fit one vector and its query workspace".into(),
        ));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_limits_cover_vectors_outputs_and_reductions() {
        for limit in [16, 1024, 128 * 1024 * 1024, u32::MAX as usize] {
            for dims in [1, 2, 3, 64, 129, 4096] {
                if let Ok(rows) = chunk_row_limit(dims, limit, limit) {
                    assert!(rows * dims * 4 <= limit);
                    assert!((1 + rows * (1 + MAX_DISKS)) * 4 <= limit);
                    assert!((2 * rows + 2 * rows.div_ceil(128)) * 4 <= limit);
                } else {
                    assert!(dims * 4 > limit || limit < (2 + MAX_DISKS) * 4);
                }
            }
        }
        let rows = chunk_row_limit(64, u32::MAX as usize, u32::MAX as usize).unwrap();
        assert_eq!(rows, 16_777_215);
        assert!(chunk_row_limit(64, 255, u32::MAX as usize).is_err());
        #[cfg(target_pointer_width = "64")]
        {
            // Device addresses remove the storage-binding range limit.
            let rows = chunk_row_limit(64, usize::MAX, usize::MAX).unwrap();
            assert_eq!(rows, u32::MAX as usize);
            assert_eq!(chunk_row_limit(1, usize::MAX, usize::MAX).unwrap(), rows);
        }
    }
}
