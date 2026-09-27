//! Exact Vulkan compute backend. Shaders and their sources are distributed with
//! the crate; neither a shader compiler nor a Vulkan SDK is needed by consumers.
mod runtime;

use std::sync::{Arc, Mutex, MutexGuard};

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::backend::{DiskSpec, RingComputeBackend};
use crate::error::{Result, RingDbError};
use crate::metadata::CandidateSetView;
use crate::query::Hit;
use runtime::{Buffer, Context, Execution, Pipelines, WORKGROUP_SIZE, barrier, compute_barrier};

const INVALID: u32 = u32::MAX;
const RANGE: u32 = 0;
const INTERSECTION: u32 = 1;
const KNN: u32 = 2;

// Must match shaders/common.glsl. Addresses and reduction offsets come first
// for 8-byte alignment; reserved makes the final padding explicit for Pod.
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
    lower_sq: f32,
    upper_sq: f32,
    reduce_count: u32,
    output_index: u32,
    row_offset: u32,
    reserved: u32,
}

struct VectorChunk {
    buffer: Buffer,
    offset: usize,
    rows: usize,
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

pub struct VulkanBackend {
    execution: Mutex<ExecutionState>,
    pipelines: Pipelines,
    vectors: Vec<VectorChunk>,
    context: Arc<Context>,
    dims: usize,
}

impl VulkanBackend {
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
            vectors: Vec::new(),
            context,
            dims: 0,
        })
    }

    fn execution(&self) -> Result<MutexGuard<'_, ExecutionState>> {
        self.execution
            .lock()
            .map_err(|_| RingDbError::Backend("Vulkan execution mutex is poisoned".to_string()))
    }

    fn prepare(
        &self,
        chunk: &VectorChunk,
        dims: usize,
        disks: &[DiskSpec<'_>],
        candidates: CandidateSetView<'_>,
    ) -> Result<PreparedQuery> {
        if dims != self.dims {
            return Err(RingDbError::DimensionMismatch {
                expected: self.dims,
                got: dims,
            });
        }
        let disk_words = dims
            .checked_mul(2)
            .and_then(|n| n.checked_add(2))
            .and_then(|n| n.checked_mul(disks.len()))
            .ok_or_else(size_error)?;
        let end = chunk.offset + chunk.rows;
        let candidates = match candidates {
            CandidateSetView::Sparse(ids) => {
                let start = ids.partition_point(|&id| (id as usize) < chunk.offset);
                let stop = ids.partition_point(|&id| (id as usize) < end);
                CandidateSetView::Sparse(&ids[start..stop])
            }
            other => other,
        };
        let first_word = chunk.offset / 32;
        let candidate_end = end.div_ceil(32);
        let (candidate_mode, work_count, eligible, candidate_words) = match candidates {
            CandidateSetView::All => (0, chunk.rows, chunk.rows, 0),
            CandidateSetView::Empty => (0, 0, 0, 0),
            CandidateSetView::Sparse(ids) => (1, ids.len(), ids.len(), ids.len()),
            CandidateSetView::Dense(words) => {
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
        let input_bytes = word_bytes(total_words)?;
        self.context.check_buffer_size(input_bytes)?;
        let mut data = Vec::with_capacity(total_words);
        for disk in disks {
            if disk.query.len() != dims || disk.weights.is_some_and(|w| w.len() != dims) {
                return Err(RingDbError::Backend(
                    "Vulkan query dimensions do not match dataset".to_string(),
                ));
            }
            data.push(u32::from(disk.weights.is_some()));
            data.push((disk.d_max * disk.d_max).to_bits());
            data.extend(disk.query.iter().map(|v| v.to_bits()));
            if let Some(weights) = disk.weights {
                data.extend(weights.iter().map(|w| w.to_bits()));
            } else {
                data.resize(data.len() + dims, 0);
            }
        }
        match candidates {
            CandidateSetView::Sparse(ids) => data.extend_from_slice(ids),
            CandidateSetView::Dense(words) => {
                // Explicitly split into low/high u32 words: shaderInt64 is not
                // required and behavior does not depend on host endianness.
                data.extend((first_word..candidate_end).map(|i| dense_word(words, i)));
            }
            _ => {}
        }
        Ok(PreparedQuery {
            params: Params {
                dims: dims as u32,
                rows: chunk.rows as u32,
                row_offset: chunk.offset as u32,
                vectors: chunk.buffer.address,
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

    fn readback<T: Pod>(
        &self,
        execution: &mut Execution,
        source: &Buffer,
        host: &mut Buffer,
        offset: usize,
        count: usize,
    ) -> Result<Vec<T>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let bytes = count.checked_mul(size_of::<T>()).ok_or_else(size_error)?;
        host.grow(bytes)?;
        execution.copy(source, offset, host, bytes)?;
        host.read(count)
    }

    fn geometric(
        &self,
        mut query: PreparedQuery,
        mode: u32,
        lower: f32,
        upper: f32,
        hits: &mut Vec<Hit>,
    ) -> Result<()> {
        if query.eligible == 0 {
            return Ok(());
        }
        query.params.mode = mode;
        query.params.lower_sq = lower * lower;
        query.params.upper_sq = upper * upper;
        let output_words = query
            .eligible
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(size_error)?;
        let mut state = self.execution()?;
        let (execution, buffers) = state.buffers(&self.context)?;
        buffers.inputs.grow(word_bytes(query.inputs.len())?)?;
        buffers.inputs.write(bytemuck::cast_slice(&query.inputs))?;
        buffers.output.grow(word_bytes(output_words)?)?;
        query.params.inputs = buffers.inputs.address;
        query.params.scratch = buffers.scratch.address;
        query.params.output_address = buffers.output.address;
        execution.run(
            if mode == INTERSECTION {
                "intersection"
            } else {
                "range"
            },
            |cmd| {
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
            },
        )?;
        let count = buffers.count.read::<u32>(1)?[0] as usize;
        if count > query.eligible {
            return Err(RingDbError::Backend(
                "Vulkan returned an invalid hit count".to_string(),
            ));
        }
        if count != 0 {
            let bytes = count.checked_mul(size_of::<Hit>()).ok_or_else(size_error)?;
            buffers.readback.grow(bytes)?;
            execution.copy(&buffers.output, 4, &buffers.readback, bytes)?;
            buffers.readback.read_into(hits, count)?;
        }
        Ok(())
    }

    fn knn(&self, mut query: PreparedQuery, k: usize) -> Result<Vec<Hit>> {
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
        let mut hits: Vec<Hit> = if host_selection {
            self.readback(
                execution,
                &buffers.scratch,
                &mut buffers.readback,
                0,
                work as usize,
            )?
        } else {
            self.readback(execution, &buffers.output, &mut buffers.readback, 0, k)?
        };
        hits.retain(|hit| hit.id != INVALID);
        if hits.len() < k {
            return Err(RingDbError::Backend(
                "Vulkan KNN returned fewer candidates than expected".to_string(),
            ));
        }
        hits.sort_unstable_by(hit_order);
        hits.truncate(k);
        Ok(hits)
    }
}

impl RingComputeBackend for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn upload_f32_dataset(&mut self, dims: usize, vectors: Vec<f32>) -> Result<()> {
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
        // Optional smaller chunks exercise the same path in tests and benchmarks
        // without requiring multi-GiB datasets. Never override device limits.
        let vector_limit = match std::env::var("RINGDB_VULKAN_CHUNK_BYTES") {
            Ok(value) => value
                .parse::<usize>()
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| {
                    RingDbError::Backend(
                        "RINGDB_VULKAN_CHUNK_BYTES must be a positive byte count".into(),
                    )
                })?
                .min(limit),
            Err(std::env::VarError::NotPresent) => limit,
            Err(error) => return Err(RingDbError::Backend(error.to_string())),
        };
        let chunk_rows = chunk_row_limit(dims, vector_limit, limit)?;
        let mut chunks = Vec::new();
        for offset in (0..rows.max(1)).step_by(chunk_rows) {
            let count = (rows - offset).min(chunk_rows);
            let source = &vectors[offset * dims..(offset + count) * dims];
            let bytes = word_bytes(source.len())?;
            // Transpose each chunk independently; coordinate reads stay
            // contiguous and temporary host storage is bounded by one chunk.
            let mut transposed = vec![0.0f32; source.len()];
            for (id, row) in source.chunks_exact(dims).enumerate() {
                for (dim, value) in row.iter().enumerate() {
                    transposed[dim * count + id] = *value;
                }
            }
            let mut staging = Buffer::new(&self.context, bytes, true)?;
            staging.write(bytemuck::cast_slice(&transposed))?;
            drop(transposed);
            let buffer = Buffer::new(&self.context, bytes, false)?;
            if bytes != 0 {
                self.execution()?
                    .execution
                    .copy(&staging, 0, &buffer, bytes)?;
            }
            chunks.push(VectorChunk {
                buffer,
                offset,
                rows: count,
            });
        }
        self.vectors = chunks;
        self.dims = dims;
        Ok(())
    }

    fn ring_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        weights: Option<&[f32]>,
        d_min: f32,
        d_max: f32,
        candidates: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        let disk = DiskSpec {
            query,
            weights,
            d_max,
        };
        let mut hits = Vec::new();
        for chunk in &self.vectors {
            self.geometric(
                self.prepare(chunk, dims, &[disk], candidates)?,
                RANGE,
                d_min,
                d_max,
                &mut hits,
            )?;
        }
        Ok(hits)
    }

    fn disk_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        weights: Option<&[f32]>,
        d_max: f32,
        candidates: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        self.ring_query_f32(dims, query, weights, 0.0, d_max, candidates)
    }

    fn disk_intersection_query_f32(
        &self,
        dims: usize,
        disks: &[DiskSpec<'_>],
        candidates: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        if disks.is_empty() {
            return Err(RingDbError::InvalidQuery(
                "disk intersection requires at least one disk".to_string(),
            ));
        }
        let mut hits = Vec::new();
        for chunk in &self.vectors {
            self.geometric(
                self.prepare(chunk, dims, disks, candidates)?,
                INTERSECTION,
                0.0,
                0.0,
                &mut hits,
            )?;
        }
        Ok(hits)
    }

    fn knn_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        weights: Option<&[f32]>,
        k: usize,
        candidates: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        let disk = DiskSpec {
            query,
            weights,
            d_max: 0.0,
        };
        let mut hits = Vec::new();
        for chunk in &self.vectors {
            let mut next = self.knn(self.prepare(chunk, dims, &[disk], candidates)?, k)?;
            if hits.is_empty() {
                hits = next;
            } else {
                hits.append(&mut next);
                if hits.len() > k {
                    hits.select_nth_unstable_by(k, hit_order);
                    hits.truncate(k);
                }
            }
        }
        if self.vectors.len() > 1 {
            hits.sort_unstable_by(hit_order);
        }
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

fn hit_order(a: &Hit, b: &Hit) -> std::cmp::Ordering {
    a.dist_sq
        .total_cmp(&b.dist_sq)
        .then_with(|| a.id.cmp(&b.id))
}

fn chunk_row_limit(dims: usize, vector_bytes: usize, buffer_bytes: usize) -> Result<usize> {
    let row_bytes = word_bytes(dims)?;
    let words = buffer_bytes / 4;
    // Geometry needs a count plus two words per hit. KNN needs two words
    // per distance plus two workgroup-reduction regions. Bound both, even
    // for low dimensions where vector storage would permit more rows.
    let mut rows = (vector_bytes / row_bytes)
        .min(words.saturating_sub(1) / 2)
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
    fn chunk_limits_cover_vectors_results_and_reductions() {
        for limit in [16, 1024, 128 * 1024 * 1024, u32::MAX as usize] {
            for dims in [1, 2, 3, 64, 129, 4096] {
                if let Ok(rows) = chunk_row_limit(dims, limit, limit) {
                    assert!(rows * dims * 4 <= limit);
                    assert!((1 + 2 * rows) * 4 <= limit);
                    assert!((2 * rows + 2 * rows.div_ceil(128)) * 4 <= limit);
                } else {
                    assert!(dims * 4 > limit || limit < 16);
                }
            }
        }
        let rows = chunk_row_limit(64, u32::MAX as usize, u32::MAX as usize).unwrap();
        assert_eq!(rows, 16_777_215);
        assert_eq!(32_124_552usize.div_ceil(rows), 2);
        assert!(chunk_row_limit(64, 255, u32::MAX as usize).is_err());
        #[cfg(target_pointer_width = "64")]
        {
            // Device addresses remove the storage-binding range limit. A
            // sufficiently large allocation fits France in one chunk.
            let rows = chunk_row_limit(64, usize::MAX, usize::MAX).unwrap();
            assert_eq!(rows, u32::MAX as usize);
            assert_eq!(32_124_552usize.div_ceil(rows), 1);
            // Low-dimensional scratch/output also uses 64-bit addresses.
            assert_eq!(chunk_row_limit(1, usize::MAX, usize::MAX).unwrap(), rows);
        }
    }
}
