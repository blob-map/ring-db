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
const COUNT: u32 = 2;

// Must match shaders/common.glsl. Addresses come first for 8-byte alignment,
// and the u32 fields fill the struct, so Pod sees no padding.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct Params {
    vectors: u64,
    inputs: u64,
    scratch: u64,
    output_address: u64,
    reduce_src: u64,
    reduce_dst: u64,
    taken: u64,
    dims: u32,
    rows: u32,
    work_count: u32,
    candidate_offset: u32,
    disk_count: u32,
    mode: u32,
    reduce_count: u32,
    output_index: u32,
    /// First row of the chunk within its segment.
    row_offset: u32,
    /// Added to region output rows: the segment's first row in a snapshot.
    row_base: u32,
    reduce_shift: u32,
    /// First slot of a region batch, whose end is `work_count`.
    slot_base: u32,
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
        self.upload_staged(dims, vectors, UPLOAD_STAGING_BYTES)
    }

    /// [`VulkanDevice::upload`], staging about `staging_bytes` of vectors at
    /// a time.
    fn upload_staged(
        &self,
        dims: usize,
        vectors: &[f32],
        staging_bytes: usize,
    ) -> Result<VulkanSegment> {
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
            chunks.push(Chunk {
                vectors: self.resident_transposed(source, dims, staging_bytes)?,
                offset,
                rows: count,
            });
        }
        Ok(VulkanSegment { chunks, dims })
    }

    /// A device buffer of row-major `source`, with the coordinates of each
    /// dimension stored together. Blocks of rows go through a staging buffer
    /// of about `staging_bytes`, so host memory stays bounded.
    fn resident_transposed(
        &self,
        source: &[f32],
        dims: usize,
        staging_bytes: usize,
    ) -> Result<Buffer> {
        let rows = source.len() / dims;
        let buffer = Buffer::new(&self.context, size_of_val(source), false)?;
        let block = (staging_bytes / (dims * 4)).clamp(1, rows);
        let mut staging = Buffer::new(&self.context, block * dims * 4, true)?;
        let mut transposed = vec![0.0f32; block * dims];
        let mut regions = Vec::with_capacity(dims);
        for start in (0..rows).step_by(block) {
            let count = (rows - start).min(block);
            let block_rows = &source[start * dims..(start + count) * dims];
            for (id, row) in block_rows.chunks_exact(dims).enumerate() {
                for (dim, value) in row.iter().enumerate() {
                    transposed[dim * count + id] = *value;
                }
            }
            staging.write(bytemuck::cast_slice(&transposed[..count * dims]))?;
            regions.clear();
            regions.extend(
                (0..dims).map(|dim| (dim * count * 4, (dim * rows + start) * 4, count * 4)),
            );
            self.execution()?
                .execution
                .copy_regions(&staging, &buffer, &regions)?;
        }
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
        self.region_batched(segment, base, disks, mask, records, REGION_OUTPUT_BYTES)
    }

    /// [`VulkanDevice::region`], with at most about `output_bytes` of records
    /// in the output buffer at a time.
    fn region_batched(
        &self,
        segment: &VulkanSegment,
        base: u32,
        disks: &[DiskSpec<'_>],
        mask: Mask<'_>,
        records: &mut Vec<u32>,
        output_bytes: usize,
    ) -> Result<()> {
        if disks.is_empty() || disks.len() > MAX_DISKS {
            return Err(RingDbError::InvalidQuery(format!(
                "a region needs between 1 and {MAX_DISKS} disks"
            )));
        }
        for chunk in &segment.chunks {
            let mut query = self.prepare(segment, chunk, disks, mask)?;
            query.params.row_base = base;
            self.region_chunk(query, Some(records), output_bytes)?;
        }
        Ok(())
    }

    /// Number of rows of `segment` inside all `disks`.
    pub(crate) fn region_count(
        &self,
        segment: &VulkanSegment,
        disks: &[DiskSpec<'_>],
        mask: Mask<'_>,
    ) -> Result<usize> {
        if disks.is_empty() || disks.len() > MAX_DISKS {
            return Err(RingDbError::InvalidQuery(format!(
                "a region needs between 1 and {MAX_DISKS} disks"
            )));
        }
        let mut count = 0;
        for chunk in &segment.chunks {
            count += self.region_chunk(self.prepare(segment, chunk, disks, mask)?, None, 0)?;
        }
        Ok(count)
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

    /// Count the rows of a chunk inside the region, and append their records
    /// to `records` if given. Records are read in batches of slots whose
    /// output holds at most about `output_bytes`.
    fn region_chunk(
        &self,
        mut query: PreparedQuery,
        mut records: Option<&mut Vec<u32>>,
        output_bytes: usize,
    ) -> Result<usize> {
        if query.eligible == 0 {
            return Ok(0);
        }
        let stride = 1 + query.params.disk_count as usize;
        let work_count = query.params.work_count as usize;
        // A slot holds at most one record. A count needs only the count word.
        let batch = match records {
            Some(_) => (output_bytes / 4 / stride).max(1),
            None => work_count,
        };
        query.params.mode = if records.is_some() { REGION } else { COUNT };
        let mut state = self.execution()?;
        let (execution, buffers) = state.buffers(&self.context)?;
        buffers.inputs.grow(word_bytes(query.inputs.len())?)?;
        buffers.inputs.write(bytemuck::cast_slice(&query.inputs))?;
        query.params.inputs = buffers.inputs.address;
        let mut total = 0;
        for start in (0..work_count).step_by(batch) {
            let end = (start + batch).min(work_count);
            let output_words = match records {
                Some(_) => (end - start)
                    .min(query.eligible)
                    .checked_mul(stride)
                    .and_then(|n| n.checked_add(1))
                    .ok_or_else(size_error)?,
                None => 1,
            };
            buffers.output.grow(word_bytes(output_words)?)?;
            query.params.output_address = buffers.output.address;
            query.params.slot_base = start as u32;
            query.params.work_count = end as u32;
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
                    (end - start) as u32,
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
            if count > (end - start).min(query.eligible) {
                return Err(RingDbError::Backend(
                    "Vulkan returned an invalid hit count".to_string(),
                ));
            }
            total += count;
            if let Some(records) = records.as_deref_mut().filter(|_| count > 0) {
                // Records arrive in their final layout: one plain copy.
                let words = count * stride;
                buffers.readback.grow(word_bytes(words)?)?;
                execution.copy(&buffers.output, 4, &buffers.readback, word_bytes(words)?)?;
                buffers.readback.read_into(records, 0, words)?;
            }
        }
        Ok(total)
    }

    fn knn_chunk(&self, mut query: PreparedQuery, k: usize) -> Result<Vec<RowHit>> {
        let k = k.min(query.eligible);
        if k == 0 {
            return Ok(Vec::new());
        }
        query.params.mode = KNN;
        let work = query.params.work_count;
        let host_selection = k.saturating_mul(64) >= work as usize;
        let distance_words = work as usize * 2;
        // A tournament tree over the distance slots: each level holds the best
        // slot of every 128 entries of the level below, up to one root entry.
        let mut levels = Vec::new();
        let (mut entries, mut offset) = (work, distance_words);
        loop {
            entries = entries.div_ceil(WORKGROUP_SIZE);
            levels.push((offset as u64, entries));
            offset += entries as usize;
            if entries == 1 {
                break;
            }
        }
        // One more word holds the slot of the last hit.
        let taken = offset;
        let scratch_words = taken + 1;
        debug_assert!(scratch_words <= distance_words + 2 * work.div_ceil(WORKGROUP_SIZE) as usize);
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
        query.params.taken = taken as u64;
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
            // Reduce level `level` of the tree from the level below. With
            // `repair`, rebuild only the entry above the slot last taken.
            let reduce = |level: usize, repair: bool| {
                let mut params = query.params;
                (params.reduce_src, params.reduce_count) = match level {
                    0 => (u64::MAX, work),
                    _ => levels[level - 1],
                };
                params.reduce_dst = levels[level].0;
                let threads = if repair {
                    params.reduce_shift = WORKGROUP_SIZE.trailing_zeros() * (level as u32 + 1);
                    WORKGROUP_SIZE
                } else {
                    params.reduce_count
                };
                self.dispatch(cmd, self.pipelines.reduce, &params, threads);
                compute_barrier(&self.context, cmd);
            };
            (0..levels.len()).for_each(|level| reduce(level, false));
            for index in 0..k {
                let mut params = query.params;
                params.reduce_dst = levels[levels.len() - 1].0;
                params.output_index = index as u32;
                self.dispatch(cmd, self.pipelines.take, &params, 1);
                compute_barrier(&self.context, cmd);
                if index + 1 < k {
                    (0..levels.len()).for_each(|level| reduce(level, true));
                }
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

/// Bytes of region records that the device holds at a time.
const REGION_OUTPUT_BYTES: usize = 256 << 20;

/// Bytes of vectors that an upload stages at a time.
const UPLOAD_STAGING_BYTES: usize = 64 << 20;

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
    fn region_batches_return_every_record() {
        if std::env::var("RINGDB_VULKAN_TEST").as_deref() != Ok("1") {
            return;
        }
        let device = VulkanDevice::new().unwrap();
        let (dims, rows) = (2, 5000);
        let vectors: Vec<f32> = (0..rows * dims).map(|i| (i % 97) as f32).collect();
        let segment = device.upload(dims, &vectors).unwrap();
        let disks = [DiskSpec {
            center: &[40.0, 40.0],
            weights: None,
            radius_sq: 900.0,
        }];
        let sparse: Vec<u32> = (0..rows as u32).step_by(3).collect();
        let dense: Vec<u64> = vec![0x00ff_00ff_00ff_00ff; rows.div_ceil(64)];
        let sorted = |records: Vec<u32>| {
            let mut pairs = records.as_chunks::<2>().0.to_vec();
            pairs.sort_unstable();
            pairs
        };
        for mask in [Mask::All, Mask::Sparse(&sparse), Mask::Dense(&dense)] {
            let mut expected = Vec::new();
            device
                .region(&segment, 10, &disks, mask, &mut expected)
                .unwrap();
            assert!(expected.len() > 1000);
            // 7 records per batch, so most batches have hits and some do not.
            let mut batched = Vec::new();
            device
                .region_batched(&segment, 10, &disks, mask, &mut batched, 7 * 2 * 4)
                .unwrap();
            assert_eq!(sorted(batched), sorted(expected));
        }
    }

    #[test]
    fn staged_uploads_transpose_every_block() {
        if std::env::var("RINGDB_VULKAN_TEST").as_deref() != Ok("1") {
            return;
        }
        let device = VulkanDevice::new().unwrap();
        let (dims, rows) = (3, 1000);
        let vectors: Vec<f32> = (0..rows * dims).map(|i| i as f32).collect();
        // Blocks of 7 rows, the last one shorter.
        let segment = device.upload_staged(dims, &vectors, 7 * dims * 4).unwrap();
        let chunk = &segment.chunks[0];
        let readback = Buffer::new(&device.context, rows * dims * 4, true).unwrap();
        let mut state = device.execution().unwrap();
        state
            .execution
            .copy(&chunk.vectors, 0, &readback, rows * dims * 4)
            .unwrap();
        let actual: Vec<f32> = readback.read(rows * dims).unwrap();
        let expected: Vec<f32> = (0..dims)
            .flat_map(|dim| (0..rows).map(move |row| (row * dims + dim) as f32))
            .collect();
        assert_eq!(segment.chunks.len(), 1);
        assert_eq!(actual, expected);
    }

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
