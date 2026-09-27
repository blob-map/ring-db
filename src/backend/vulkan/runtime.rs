//! Vulkan ownership and synchronous submissions. A backend's execution mutex
//! serializes queue, descriptor-set, and command-pool access. Every submission
//! completes before its buffers can be freed or mapped by the host.
use std::io::Cursor;
use std::mem::MaybeUninit;
use std::sync::{Arc, OnceLock};

use ash::{Entry, vk};

use crate::error::{Result, RingDbError};

pub(super) const WORKGROUP_SIZE: u32 = 128;

fn backend_error(error: impl std::fmt::Display) -> RingDbError {
    RingDbError::Backend(format!("Vulkan: {error}"))
}

fn unavailable(error: impl std::fmt::Display) -> RingDbError {
    RingDbError::BackendUnavailable(format!("Vulkan: {error}"))
}

fn copy_readback(source: &[u8], destination: &mut [MaybeUninit<u8>]) -> Result<()> {
    fn copy(source: &[u8], destination: &mut [MaybeUninit<u8>]) {
        debug_assert_eq!(source.len(), destination.len());
        // SAFETY: the mapped source and the owned result do not overlap. Each
        // worker receives a disjoint destination slice of the same length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.as_ptr(),
                destination.as_mut_ptr().cast::<u8>(),
                source.len(),
            );
        }
    }

    const PARALLEL_BYTES: usize = 16 * 1024 * 1024;
    static WORKERS: OnceLock<usize> = OnceLock::new();
    let workers = if source.len() >= PARALLEL_BYTES {
        *WORKERS.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get().min(4)))
    } else {
        1
    };
    if workers == 1 {
        copy(source, destination);
        return Ok(());
    }
    // Use scoped helpers, not the caller's Rayon pool: this runs under the GPU
    // execution mutex, where stealing other query jobs could re-enter the lock.
    // Scope joins every worker, including on a spawn error, before unmapping.
    std::thread::scope(|scope| {
        let chunk = source.len().div_ceil(workers);
        let mut chunks = source.chunks(chunk).zip(destination.chunks_mut(chunk));
        let first = chunks.next().expect("nonempty large readback");
        for (source, destination) in chunks {
            std::thread::Builder::new()
                .spawn_scoped(scope, move || copy(source, destination))
                .map_err(backend_error)?;
        }
        copy(first.0, first.1);
        Ok(())
    })
}

struct Instance {
    raw: ash::Instance,
    // Keep the dynamically loaded library alive until the instance is destroyed.
    _entry: Entry,
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: all logical devices retain this instance through Context.
        unsafe { self.raw.destroy_instance(None) };
    }
}

pub(super) struct Context {
    pub device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    pub limits: vk::PhysicalDeviceLimits,
    memory: vk::PhysicalDeviceMemoryProperties,
    timestamp_bits: u32,
    _instance: Instance,
}

impl Context {
    pub fn new() -> Result<Arc<Self>> {
        // SAFETY: Entry owns the loader and outlives all derived handles.
        let entry = unsafe { Entry::load() }.map_err(unavailable)?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"ring-db")
            .api_version(vk::API_VERSION_1_0);
        let info = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: all pointers in info refer to live values for this call.
        let raw = unsafe { entry.create_instance(&info, None) }.map_err(unavailable)?;
        let instance = Instance { raw, _entry: entry };
        // SAFETY: instance is live and enumeration has no external-sync requirement.
        let devices = unsafe { instance.raw.enumerate_physical_devices() }.map_err(unavailable)?;
        let mut choices = Vec::new();
        for physical in devices {
            // SAFETY: physical was enumerated from this instance.
            let props = unsafe { instance.raw.get_physical_device_properties(physical) };
            let families = unsafe {
                instance
                    .raw
                    .get_physical_device_queue_family_properties(physical)
            };
            if props.limits.max_compute_work_group_invocations < WORKGROUP_SIZE
                || props.limits.max_compute_work_group_size[0] < WORKGROUP_SIZE
                || props.limits.max_per_stage_descriptor_storage_buffers < 4
                || props.limits.max_descriptor_set_storage_buffers < 4
            {
                continue;
            }
            if let Some(family) = families.iter().position(|family| {
                family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            }) {
                let rank = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    vk::PhysicalDeviceType::CPU => 3,
                    _ => 4,
                };
                choices.push((
                    rank,
                    physical,
                    family as u32,
                    props.limits,
                    families[family].timestamp_valid_bits,
                ));
            }
        }
        choices.sort_by_key(|choice| choice.0);
        let mut failure = "no compatible compute device".to_string();
        for (_, physical, queue_family, limits, timestamp_bits) in choices {
            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let info = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
            // SAFETY: queue family and counts were checked; no optional features
            // or extensions are used by the embedded Vulkan 1.0 shaders.
            let device = match unsafe { instance.raw.create_device(physical, &info, None) } {
                Ok(device) => device,
                Err(error) => {
                    failure = error.to_string();
                    continue;
                }
            };
            let queue = unsafe { device.get_device_queue(queue_family, 0) };
            let memory = unsafe { instance.raw.get_physical_device_memory_properties(physical) };
            return Ok(Arc::new(Self {
                device,
                queue,
                queue_family,
                limits,
                memory,
                timestamp_bits,
                _instance: instance,
            }));
        }
        Err(unavailable(failure))
    }

    pub fn check_buffer_size(&self, size: usize) -> Result<()> {
        if size as u64 > u64::from(self.limits.max_storage_buffer_range) {
            return Err(backend_error(format!(
                "buffer requires {size} bytes, device storage-buffer limit is {} bytes",
                self.limits.max_storage_buffer_range,
            )));
        }
        Ok(())
    }

    fn memory_type(&self, bits: u32, host: bool) -> Result<(u32, bool)> {
        let required = if host {
            vk::MemoryPropertyFlags::HOST_VISIBLE
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let mut choices: Vec<_> = self.memory.memory_types
            [..self.memory.memory_type_count as usize]
            .iter()
            .enumerate()
            .filter(|(index, ty)| bits & (1 << index) != 0 && ty.property_flags.contains(required))
            .collect();
        // Coherent memory avoids flush/invalidate calls. Prefer cached memory
        // for readback when both are available; still support non-coherent heaps.
        choices.sort_by_key(|(_, ty)| {
            let flags = ty.property_flags;
            if host {
                (
                    !flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
                    !flags.contains(vk::MemoryPropertyFlags::HOST_CACHED),
                )
            } else {
                // Prefer ordinary VRAM over a potentially small host-visible
                // BAR heap on discrete GPUs.
                (flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE), false)
            }
        });
        let (index, ty) = choices
            .first()
            .ok_or_else(|| backend_error("no suitable memory type"))?;
        Ok((
            *index as u32,
            ty.property_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT),
        ))
    }

    pub fn dispatch(&self, cmd: vk::CommandBuffer, count: u32) {
        let groups = count.div_ceil(WORKGROUP_SIZE);
        let x = groups.min(self.limits.max_compute_work_group_count[0]);
        let y = groups.div_ceil(x);
        // Buffer limits bound count to far below the Vulkan minimum 2D dispatch
        // capacity. Shaders flatten group coordinates and discard padded groups.
        debug_assert!(y <= self.limits.max_compute_work_group_count[1]);
        unsafe { self.device.cmd_dispatch(cmd, x, y, 1) };
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: every resource and execution object retains an Arc<Context>;
        // all submitted work has completed before the final Arc is released.
        unsafe { self.device.destroy_device(None) };
    }
}

pub(super) struct Buffer {
    context: Arc<Context>,
    pub raw: vk::Buffer,
    memory: vk::DeviceMemory,
    pub size: usize,
    host: bool,
    coherent: bool,
}

impl Buffer {
    pub fn new(context: &Arc<Context>, size: usize, host: bool) -> Result<Self> {
        let size = size.max(4);
        context.check_buffer_size(size)?;
        let mut buffer = Self {
            context: context.clone(),
            raw: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size,
            host,
            coherent: false,
        };
        let info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: all handles belong to context; allocation follows the queried
        // requirements. The partially initialized buffer cleans up on any error.
        unsafe {
            buffer.raw = context
                .device
                .create_buffer(&info, None)
                .map_err(backend_error)?;
            let requirements = context.device.get_buffer_memory_requirements(buffer.raw);
            let (memory_type, coherent) =
                context.memory_type(requirements.memory_type_bits, host)?;
            buffer.coherent = coherent;
            let allocation = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            buffer.memory = context
                .device
                .allocate_memory(&allocation, None)
                .map_err(backend_error)?;
            context
                .device
                .bind_buffer_memory(buffer.raw, buffer.memory, 0)
                .map_err(backend_error)?;
        }
        Ok(buffer)
    }

    pub fn write(&mut self, bytes: &[u8]) -> Result<()> {
        if !self.host || bytes.len() > self.size {
            return Err(backend_error("invalid host buffer write"));
        }
        // SAFETY: callers only map buffers outside submissions. The entire
        // allocation is mapped so WHOLE_SIZE flushes satisfy atom alignment.
        unsafe {
            let ptr = self
                .context
                .device
                .map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(backend_error)?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
            let result = if self.coherent {
                Ok(())
            } else {
                self.context
                    .device
                    .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(self.memory)
                        .size(vk::WHOLE_SIZE)])
            };
            self.context.device.unmap_memory(self.memory);
            result.map_err(backend_error)
        }
    }

    /// Grow only between completed submissions. Keep the previous allocation
    /// intact on failure so a later, smaller query can still use it.
    pub fn grow(&mut self, size: usize) -> Result<()> {
        if size > self.size {
            *self = Self::new(&self.context, size, self.host)?;
        }
        Ok(())
    }

    pub fn read<T: bytemuck::Pod>(&self, count: usize) -> Result<Vec<T>> {
        if !self.host || size_of::<T>() == 0 || count > self.size / size_of::<T>() {
            return Err(backend_error("invalid host buffer read"));
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut values: Vec<T> = Vec::with_capacity(count);
        // SAFETY: readbacks are fence-complete before mapping. Pod permits every
        // copied bit pattern. Copy directly into the final Vec's spare capacity
        // and initialize its length only after all bytes have been written.
        // Byte copying does not require T alignment on the mapped source.
        unsafe {
            let ptr = self
                .context
                .device
                .map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(backend_error)?;
            let result = (|| -> Result<()> {
                if !self.coherent {
                    self.context
                        .device
                        .invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                            .memory(self.memory)
                            .size(vk::WHOLE_SIZE)])
                        .map_err(backend_error)?;
                }
                let bytes = count * size_of::<T>();
                let source = std::slice::from_raw_parts(ptr.cast::<u8>(), bytes);
                let destination = std::slice::from_raw_parts_mut(
                    values.as_mut_ptr().cast::<MaybeUninit<u8>>(),
                    bytes,
                );
                copy_readback(source, destination)?;
                values.set_len(count);
                Ok(())
            })();
            self.context.device.unmap_memory(self.memory);
            result?;
        }
        Ok(values)
    }

    fn descriptor(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.raw)
            .range(self.size as u64)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: synchronous execution guarantees no in-flight use. Null
        // handles are allowed, including during failed construction.
        unsafe {
            self.context.device.destroy_buffer(self.raw, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

pub(super) struct Pipelines {
    context: Arc<Context>,
    pub descriptors: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    pub scan: [vk::Pipeline; 3],
    pub reduce: vk::Pipeline,
    pub take: vk::Pipeline,
}

impl Pipelines {
    pub fn new(context: &Arc<Context>, push_size: u32) -> Result<Self> {
        let mut pipelines = Self {
            context: context.clone(),
            descriptors: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            scan: [vk::Pipeline::null(); 3],
            reduce: vk::Pipeline::null(),
            take: vk::Pipeline::null(),
        };
        let bindings: Vec<_> = (0..4)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: layouts describe the four storage buffers and the shared
        // Params ABI used by all three checked-in shaders.
        unsafe {
            pipelines.descriptors = context
                .device
                .create_descriptor_set_layout(&info, None)
                .map_err(backend_error)?;
            let layouts = [pipelines.descriptors];
            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(push_size)];
            let info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push);
            pipelines.layout = context
                .device
                .create_pipeline_layout(&info, None)
                .map_err(backend_error)?;
        }
        for mode in 0..3 {
            pipelines.scan[mode] =
                pipelines.create(include_bytes!("shaders/scan.spv"), Some(mode as u32))?;
        }
        pipelines.reduce = pipelines.create(include_bytes!("shaders/reduce.spv"), None)?;
        pipelines.take = pipelines.create(include_bytes!("shaders/take.spv"), None)?;
        Ok(pipelines)
    }

    fn create(&self, bytes: &[u8], candidate_mode: Option<u32>) -> Result<vk::Pipeline> {
        let code = ash::util::read_spv(&mut Cursor::new(bytes)).map_err(backend_error)?;
        let info = vk::ShaderModuleCreateInfo::default().code(&code);
        // SAFETY: SPIR-V is compiled and validated for Vulkan 1.0. The module is
        // destroyed after pipeline creation on both success and failure paths.
        unsafe {
            let shader = self
                .context
                .device
                .create_shader_module(&info, None)
                .map_err(backend_error)?;
            let entries = [vk::SpecializationMapEntry::default()
                .constant_id(0)
                .offset(0)
                .size(4)];
            let data = candidate_mode.unwrap_or(0).to_ne_bytes();
            let specialization = vk::SpecializationInfo::default()
                .map_entries(&entries)
                .data(&data);
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(c"main");
            if candidate_mode.is_some() {
                stage = stage.specialization_info(&specialization);
            }
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(self.layout);
            let result = self.context.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[info],
                None,
            );
            self.context.device.destroy_shader_module(shader, None);
            match result {
                Ok(pipelines) => Ok(pipelines[0]),
                Err((pipelines, error)) => {
                    for pipeline in pipelines {
                        self.context.device.destroy_pipeline(pipeline, None);
                    }
                    Err(backend_error(error))
                }
            }
        }
    }
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        unsafe {
            for pipeline in self.scan.into_iter().chain([self.reduce, self.take]) {
                self.context.device.destroy_pipeline(pipeline, None);
            }
            self.context
                .device
                .destroy_pipeline_layout(self.layout, None);
            self.context
                .device
                .destroy_descriptor_set_layout(self.descriptors, None);
        }
    }
}

pub(super) struct Execution {
    context: Arc<Context>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    descriptors: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
    timestamps: vk::QueryPool,
}

impl Execution {
    pub fn new(context: &Arc<Context>, layout: vk::DescriptorSetLayout) -> Result<Self> {
        let mut execution = Self {
            context: context.clone(),
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            descriptors: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            timestamps: vk::QueryPool::null(),
        };
        unsafe {
            let info =
                vk::CommandPoolCreateInfo::default().queue_family_index(context.queue_family);
            execution.pool = context
                .device
                .create_command_pool(&info, None)
                .map_err(backend_error)?;
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(execution.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            execution.command = context
                .device
                .allocate_command_buffers(&info)
                .map_err(backend_error)?[0];
            execution.fence = context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(backend_error)?;
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4)];
            let info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&sizes);
            execution.descriptors = context
                .device
                .create_descriptor_pool(&info, None)
                .map_err(backend_error)?;
            let layouts = [layout];
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(execution.descriptors)
                .set_layouts(&layouts);
            execution.set = context
                .device
                .allocate_descriptor_sets(&info)
                .map_err(backend_error)?[0];
            if std::env::var_os("RINGDB_VULKAN_TIMINGS").is_some() {
                if context.timestamp_bits == 0 || context.limits.timestamp_period <= 0.0 {
                    return Err(backend_error(
                        "selected queue does not support GPU timestamps",
                    ));
                }
                execution.timestamps = context
                    .device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                    .map_err(backend_error)?;
            }
        }
        Ok(execution)
    }

    pub fn bind_buffers(&mut self, buffers: [&Buffer; 4]) {
        let infos = buffers.map(|buffer| [buffer.descriptor()]);
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        // SAFETY: the execution mutex is held and previous submissions finished.
        unsafe { self.context.device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn run(&mut self, label: &str, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        // SAFETY: exclusively accessed through the backend mutex; every command
        // buffer is reset only after its previous submission has finished.
        unsafe {
            let device = &self.context.device;
            device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(backend_error)?;
            device.reset_fences(&[self.fence]).map_err(backend_error)?;
            let info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.command, &info)
                .map_err(backend_error)?;
            if self.timestamps != vk::QueryPool::null() {
                device.cmd_reset_query_pool(self.command, self.timestamps, 0, 2);
                device.cmd_write_timestamp(
                    self.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.timestamps,
                    0,
                );
            }
            record(self.command);
            // Publish transfer/shader writes to later submissions and host
            // readbacks. Host writes are published by queue submission.
            barrier(
                device,
                self.command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_WRITE,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            );
            if self.timestamps != vk::QueryPool::null() {
                device.cmd_write_timestamp(
                    self.command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.timestamps,
                    1,
                );
            }
            device
                .end_command_buffer(self.command)
                .map_err(backend_error)?;
            let commands = [self.command];
            let info = vk::SubmitInfo::default().command_buffers(&commands);
            device
                .queue_submit(self.context.queue, &[info], self.fence)
                .map_err(backend_error)?;
            if let Err(error) = device.wait_for_fences(&[self.fence], true, u64::MAX) {
                // Try to drain work before local resources unwind on an error.
                let _ = device.device_wait_idle();
                return Err(backend_error(error));
            }
            if self.timestamps != vk::QueryPool::null() {
                let mut ticks = [0u64; 2];
                device
                    .get_query_pool_results(
                        self.timestamps,
                        0,
                        &mut ticks,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .map_err(backend_error)?;
                let mask = u64::MAX >> (64 - self.context.timestamp_bits);
                let elapsed = ticks[1].wrapping_sub(ticks[0]) & mask;
                eprintln!(
                    "VULKAN_GPU {label} ms={:.6}",
                    elapsed as f64 * f64::from(self.context.limits.timestamp_period) / 1e6
                );
            }
        }
        Ok(())
    }

    pub fn copy(&mut self, src: &Buffer, offset: usize, dst: &Buffer, size: usize) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        if size > dst.size || offset > src.size || size > src.size - offset {
            return Err(backend_error("buffer copy exceeds allocation"));
        }
        let context = self.context.clone();
        self.run("copy", |cmd| unsafe {
            let region = vk::BufferCopy::default()
                .src_offset(offset as u64)
                .size(size as u64);
            context
                .device
                .cmd_copy_buffer(cmd, src.raw, dst.raw, &[region]);
        })
    }
}

impl Drop for Execution {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_query_pool(self.timestamps, None);
            self.context
                .device
                .destroy_descriptor_pool(self.descriptors, None);
            self.context.device.destroy_fence(self.fence, None);
            self.context.device.destroy_command_pool(self.pool, None);
        }
    }
}

pub(super) fn compute_barrier(context: &Context, cmd: vk::CommandBuffer) {
    barrier(
        &context.device,
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
    );
}

pub(super) fn barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_stage: vk::PipelineStageFlags,
    dst_access: vk::AccessFlags,
) {
    let memory = vk::MemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    // SAFETY: called only while recording a primary command buffer; stage and
    // access masks are compatible with the compute/transfer queue.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[memory],
            &[],
            &[],
        )
    };
}
