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

// Must match shaders/common.glsl. All fields have scalar 32-bit alignment.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct Params {
    dims: u32,
    rows: u32,
    work_count: u32,
    candidate_offset: u32,
    disk_count: u32,
    mode: u32,
    lower_sq: f32,
    upper_sq: f32,
    reduce_count: u32,
    reduce_src: u32,
    reduce_dst: u32,
    output_index: u32,
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
    vectors: Option<Buffer>,
    context: Arc<Context>,
    dims: usize,
    rows: usize,
}

impl VulkanBackend {
    pub fn new() -> Result<Self> {
        let context = Context::new()?;
        let pipelines = Pipelines::new(&context, size_of::<Params>() as u32)?;
        let execution = Mutex::new(ExecutionState {
            execution: Execution::new(&context, pipelines.descriptors)?,
            buffers: None,
        });
        Ok(Self {
            execution,
            pipelines,
            vectors: None,
            context,
            dims: 0,
            rows: 0,
        })
    }

    fn execution(&self) -> Result<MutexGuard<'_, ExecutionState>> {
        self.execution
            .lock()
            .map_err(|_| RingDbError::Backend("Vulkan execution mutex is poisoned".to_string()))
    }

    fn prepare(
        &self,
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
        let (candidate_mode, work_count, eligible, candidate_words) = match candidates {
            CandidateSetView::All => (0, self.rows, self.rows, 0),
            CandidateSetView::Empty => (0, 0, 0, 0),
            CandidateSetView::Sparse(ids) => (1, ids.len(), ids.len(), ids.len()),
            CandidateSetView::Dense(words) => {
                let eligible = words
                    .iter()
                    .enumerate()
                    .map(|(i, word)| {
                        let remaining = self.rows.saturating_sub(i * 64);
                        let mask = if remaining >= 64 {
                            u64::MAX
                        } else {
                            (1u64 << remaining) - 1
                        };
                        (word & mask).count_ones() as usize
                    })
                    .sum();
                (2, self.rows, eligible, self.rows.div_ceil(32))
            }
        };
        let total_words = disk_words
            .checked_add(candidate_words)
            .ok_or_else(size_error)?;
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
                data.extend(
                    words
                        .iter()
                        .flat_map(|word| [*word as u32, (word >> 32) as u32])
                        .take(candidate_words),
                );
            }
            _ => {}
        }
        Ok(PreparedQuery {
            params: Params {
                dims: dims as u32,
                rows: self.rows as u32,
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

    fn bind(
        &self,
        execution: &mut Execution,
        inputs: &Buffer,
        scratch: &Buffer,
        output: &Buffer,
    ) -> Result<()> {
        let vectors = self.vectors.as_ref().ok_or_else(|| {
            RingDbError::Backend("Vulkan dataset has not been uploaded".to_string())
        })?;
        execution.bind_buffers([vectors, inputs, scratch, output]);
        Ok(())
    }

    fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        set: vk::DescriptorSet,
        pipeline: vk::Pipeline,
        params: &Params,
        count: u32,
    ) {
        // SAFETY: the execution mutex protects the live set and command buffer;
        // all bound resources outlive the synchronous submission.
        unsafe {
            self.context
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.context.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.layout,
                0,
                &[set],
                &[],
            );
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
    ) -> Result<Vec<Hit>> {
        if query.eligible == 0 {
            return Ok(Vec::new());
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
        self.bind(
            execution,
            &buffers.inputs,
            &buffers.scratch,
            &buffers.output,
        )?;
        let set = execution.set;
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
                    set,
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
        self.readback(execution, &buffers.output, &mut buffers.readback, 4, count)
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
        self.bind(
            execution,
            &buffers.inputs,
            &buffers.scratch,
            &buffers.output,
        )?;
        let set = execution.set;
        execution.run("knn", |cmd| {
            self.dispatch(
                cmd,
                set,
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
                params.reduce_src = INVALID;
                params.reduce_dst = distance_words as u32;
                params.output_index = index as u32;
                loop {
                    self.dispatch(
                        cmd,
                        set,
                        self.pipelines.reduce,
                        &params,
                        params.reduce_count,
                    );
                    compute_barrier(&self.context, cmd);
                    let remaining = params.reduce_count.div_ceil(WORKGROUP_SIZE);
                    if remaining == 1 {
                        break;
                    }
                    params.reduce_src = params.reduce_dst;
                    params.reduce_dst = if params.reduce_dst == distance_words as u32 {
                        distance_words as u32 + groups
                    } else {
                        distance_words as u32
                    };
                    params.reduce_count = remaining;
                }
                self.dispatch(cmd, set, self.pipelines.take, &params, 1);
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
        hits.sort_unstable_by(|a, b| {
            a.dist_sq
                .total_cmp(&b.dist_sq)
                .then_with(|| a.id.cmp(&b.id))
        });
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
        let bytes = word_bytes(vectors.len())?;
        self.context.check_buffer_size(bytes)?;
        // Adjacent invocations process adjacent rows. Transposing once makes
        // their coordinate reads contiguous for all subsequent queries.
        let mut transposed = vec![0.0f32; vectors.len()];
        for (id, row) in vectors.chunks_exact(dims).enumerate() {
            for (dim, value) in row.iter().enumerate() {
                transposed[dim * rows + id] = *value;
            }
        }
        let mut staging = Buffer::new(&self.context, bytes, true)?;
        staging.write(bytemuck::cast_slice(&transposed))?;
        let device = Buffer::new(&self.context, bytes, false)?;
        self.execution()?
            .execution
            .copy(&staging, 0, &device, bytes)?;
        self.vectors = Some(device);
        self.dims = dims;
        self.rows = rows;
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
        self.geometric(
            self.prepare(dims, &[disk], candidates)?,
            RANGE,
            d_min,
            d_max,
        )
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
        self.geometric(
            self.prepare(dims, disks, candidates)?,
            INTERSECTION,
            0.0,
            0.0,
        )
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
        self.knn(self.prepare(dims, &[disk], candidates)?, k)
    }
}

fn size_error() -> RingDbError {
    RingDbError::Backend("Vulkan buffer dimensions exceed addressable size".to_string())
}

fn word_bytes(words: usize) -> Result<usize> {
    words.checked_mul(4).ok_or_else(size_error)
}
