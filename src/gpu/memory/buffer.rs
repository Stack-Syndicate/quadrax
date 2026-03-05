use bytemuck::{Pod, cast_vec};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::PrimaryCommandBufferAbstract;
use vulkano::sync::GpuFuture;
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo},
    memory::allocator::MemoryTypeFilter,
};

use crate::gpu::device::DeviceContext;

#[derive(Clone)]
pub enum UpdateMode {
    Static,
    Dynamic,
    Readback,
}

#[derive(Clone)]
pub enum Location {
    Device(UpdateMode),
    Host,
}

pub enum BufferRole {
    Uniform,
    Storage,
    Vertex,
    Generic,
}

pub struct BufferFuture {
    pub inner: Option<Box<dyn GpuFuture>>,
}
impl BufferFuture {
    pub fn wait(self) {
        if let Some(f) = self.inner {
            f.then_signal_fence_and_flush().unwrap().wait(None).unwrap();
        }
    }
    pub fn join(self, other: BufferFuture) -> BufferFuture {
        let inner = match (self.inner, other.inner) {
            (Some(f1), Some(f2)) => Some(f1.join(f2).boxed()),
            (Some(f1), None) => Some(f1),
            (None, Some(f2)) => Some(f2),
            (None, None) => None,
        };
        BufferFuture { inner }
    }
}

#[derive(Clone)]
pub struct Buffer {
    ctx: DeviceContext,
    inner: Subbuffer<[u8]>,
    stage: Option<Box<Buffer>>,
    location: Location,
    pub length: u32,
}
impl Buffer {
    pub fn new_with_role<T: Pod + Send + Sync>(
        ctx: DeviceContext,
        data: Vec<T>,
        location: Location,
        buffer_role: BufferRole,
    ) -> Self {
        let role_usage = match buffer_role {
            BufferRole::Uniform => BufferUsage::UNIFORM_BUFFER,
            BufferRole::Storage => BufferUsage::STORAGE_BUFFER,
            BufferRole::Vertex => BufferUsage::VERTEX_BUFFER,
            BufferRole::Generic => BufferUsage::empty(),
        };
        match location {
            Location::Host => {
                let subbuffer_allocator = SubbufferAllocator::new(
                    ctx.memory_allocator.clone(),
                    SubbufferAllocatorCreateInfo {
                        buffer_usage: BufferUsage::TRANSFER_SRC
                            | BufferUsage::TRANSFER_DST
                            | role_usage,
                        memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS
                            | MemoryTypeFilter::PREFER_HOST,
                        ..Default::default()
                    },
                );
                let inner = subbuffer_allocator
                    .allocate_slice(data.len() as u64)
                    .unwrap();
                {
                    let mut write_lock = inner.write().unwrap();
                    write_lock.copy_from_slice(&data);
                }
                Self {
                    ctx: ctx.clone(),
                    inner: inner.into_bytes(),
                    stage: None,
                    location,
                    length: data.len() as u32,
                }
            }
            Location::Device(ref update_mode) => match update_mode {
                UpdateMode::Static => {
                    let subbuffer_allocator = SubbufferAllocator::new(
                        ctx.memory_allocator.clone(),
                        SubbufferAllocatorCreateInfo {
                            buffer_usage: BufferUsage::TRANSFER_DST | role_usage,
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                    );
                    let inner = subbuffer_allocator
                        .allocate_slice(data.len() as u64)
                        .unwrap();
                    {
                        let mut write_lock = inner.write().unwrap();
                        write_lock.copy_from_slice(&data);
                    }
                    Self {
                        ctx: ctx.clone(),
                        inner: inner.into_bytes(),
                        stage: None,
                        location,
                        length: data.len() as u32,
                    }
                }
                UpdateMode::Dynamic => {
                    let subbuffer_allocator = SubbufferAllocator::new(
                        ctx.memory_allocator.clone(),
                        SubbufferAllocatorCreateInfo {
                            buffer_usage: BufferUsage::TRANSFER_SRC
                                | BufferUsage::TRANSFER_DST
                                | role_usage,
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                    );
                    let inner = subbuffer_allocator
                        .allocate_slice(data.len() as u64)
                        .unwrap();
                    {
                        let mut write_lock = inner.write().unwrap();
                        write_lock.copy_from_slice(&data);
                    }
                    Self {
                        ctx: ctx.clone(),
                        inner: inner.into_bytes(),
                        stage: Some(Box::new(Buffer::new_with_role(
                            ctx.clone(),
                            data.clone(),
                            Location::Host,
                            buffer_role,
                        ))),
                        location,
                        length: data.len() as u32,
                    }
                }
                UpdateMode::Readback => {
                    let subbuffer_allocator = SubbufferAllocator::new(
                        ctx.memory_allocator.clone(),
                        SubbufferAllocatorCreateInfo {
                            buffer_usage: BufferUsage::TRANSFER_SRC | role_usage,
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                    );
                    let inner = subbuffer_allocator
                        .allocate_slice(data.len() as u64)
                        .unwrap();
                    {
                        let mut write_lock = inner.write().unwrap();
                        write_lock.copy_from_slice(&data);
                    }
                    Self {
                        ctx: ctx.clone(),
                        inner: inner.into_bytes(),
                        stage: None,
                        location,
                        length: data.len() as u32,
                    }
                }
            },
        }
    }
    pub fn new<T: Pod + Send + Sync>(ctx: DeviceContext, data: Vec<T>, location: Location) -> Self {
        Self::new_with_role(ctx, data, location, BufferRole::Generic)
    }
    pub fn read<T: Pod + Send + Sync>(&self) -> Vec<T> {
        match &self.location {
            Location::Host => {
                let subbuffer = self.inner.clone().reinterpret::<[T]>();
                let read_lock = subbuffer.read().expect("Could not read host buffer.");
                read_lock.to_vec()
            }
            Location::Device(update_mode) => match update_mode {
                UpdateMode::Static => panic!("Cannot read back from static buffer."),
                UpdateMode::Dynamic => {
                    let slice = self
                        .stage
                        .as_ref()
                        .expect("Dynamic buffer missing stage host buffer.")
                        .read::<T>();
                    slice
                }
                UpdateMode::Readback => {
                    let slice = self
                        .stage
                        .as_ref()
                        .expect("Readback buffer missing stage host buffer.")
                        .read::<T>();
                    slice
                }
            },
        }
    }
    pub fn stage<T: Pod + Send + Sync>(&self) -> BufferFuture {
        if let Some(stage) = &self.stage {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.ctx.command_allocator.clone(),
                self.ctx.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    self.inner.clone(),
                    stage.inner.clone(),
                ))
                .unwrap();
            let cb = builder.build().unwrap();
            let future = cb.execute(self.ctx.queue.clone()).unwrap();
            BufferFuture {
                inner: Some(future.boxed()),
            }
        } else {
            panic!("This buffer type does not have a staging buffer.")
        }
    }
    pub fn write<T: Pod + Send + Sync>(&mut self, data: Vec<T>) -> BufferFuture {
        let data_bytes = cast_vec(data);
        match &self.location {
            Location::Host => {
                let mut slice = self.inner.write().unwrap();
                slice.copy_from_slice(&data_bytes);
                return BufferFuture { inner: None };
            }
            Location::Device(update_mode) => match update_mode {
                UpdateMode::Static => {
                    panic!("Cannot write to a static buffer directly.")
                }
                UpdateMode::Dynamic => {
                    if let Some(stage) = &mut self.stage {
                        stage.write(data_bytes);
                        let mut builder = AutoCommandBufferBuilder::primary(
                            self.ctx.command_allocator.clone(),
                            self.ctx.queue.queue_family_index(),
                            CommandBufferUsage::OneTimeSubmit,
                        )
                        .unwrap();

                        builder
                            .copy_buffer(CopyBufferInfo::buffers(
                                stage.inner.clone(),
                                self.inner.clone(),
                            ))
                            .unwrap();

                        let command_buffer = builder.build().unwrap();

                        let future = command_buffer
                            .execute(self.ctx.queue.clone())
                            .unwrap()
                            .then_signal_fence_and_flush()
                            .unwrap();
                        return BufferFuture {
                            inner: Some(future.boxed()),
                        };
                    } else {
                        panic!("Dynamic device buffer staging is missing!");
                    }
                }
                UpdateMode::Readback => {
                    panic!("Cannot write to a readback buffer from host");
                }
            },
        }
    }
    pub fn inner<T: Pod + Send + Sync>(&self) -> Subbuffer<[T]> {
        let inner = self.inner_bytes();
        inner.cast_aligned().clone()
    }
    pub fn inner_bytes(&self) -> Subbuffer<[u8]> {
        self.inner.clone()
    }
}
