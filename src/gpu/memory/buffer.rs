use core::slice;
use std::sync::Arc;

use bytemuck::{Pod, cast_vec, checked::cast_slice};
use vulkano::command_buffer::PrimaryCommandBufferAbstract;
use vulkano::sync::GpuFuture;
use vulkano::{
    buffer::{BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::gpu::backend::BackendContext;

type VulkanoBuffer = vulkano::buffer::Buffer;

pub enum UpdateMode {
    Static,
    Dynamic,
    Readback,
}

pub enum Location {
    Device(UpdateMode),
    Host,
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

pub struct Buffer {
    ctx: BackendContext,
    inner: Subbuffer<[u8]>,
    stage: Option<Box<Buffer>>,
    location: Location,
}
impl Buffer {
    pub fn new<T: Pod + Send + Sync>(
        ctx: BackendContext,
        data: Vec<T>,
        location: Location,
    ) -> Self {
        let data_bytes = cast_vec(data);
        match location {
            Location::Host => Self {
                ctx: ctx.clone(),
                inner: VulkanoBuffer::from_iter(
                    ctx.memory_allocator,
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS
                            | MemoryTypeFilter::PREFER_HOST,
                        ..Default::default()
                    },
                    data_bytes,
                )
                .expect("Could not create host buffer."),
                stage: None,
                location,
            },
            Location::Device(ref update_mode) => match update_mode {
                UpdateMode::Static => Self {
                    ctx: ctx.clone(),
                    inner: VulkanoBuffer::from_iter(
                        ctx.memory_allocator,
                        BufferCreateInfo {
                            usage: BufferUsage::TRANSFER_DST,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                        data_bytes,
                    )
                    .expect("Could not create static device buffer."),
                    stage: None,
                    location,
                },
                UpdateMode::Dynamic => Self {
                    ctx: ctx.clone(),
                    inner: VulkanoBuffer::from_iter(
                        ctx.memory_allocator.clone(),
                        BufferCreateInfo {
                            usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                        data_bytes.clone(),
                    )
                    .expect("Could not create dynamic device buffer."),
                    stage: Some(Box::new(Buffer::new(
                        ctx.clone(),
                        data_bytes,
                        Location::Host,
                    ))),
                    location,
                },
                UpdateMode::Readback => Self {
                    ctx: ctx.clone(),
                    inner: VulkanoBuffer::from_iter(
                        ctx.memory_allocator,
                        BufferCreateInfo {
                            usage: BufferUsage::TRANSFER_SRC,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                        data_bytes,
                    )
                    .expect("Could not create host buffer."),
                    stage: None,
                    location,
                },
            },
        }
    }
    pub fn read<T: Pod + Send + Sync>(&self) -> Vec<T> {
        match &self.location {
            Location::Host => {
                let slice = self
                    .inner
                    .read()
                    .expect("Could not read host buffer.")
                    .to_vec();
                cast_vec(slice)
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
        match &self.location {
            Location::Host => panic!("Cannot stage host buffer."),
            Location::Device(update_mode) => match update_mode {
                UpdateMode::Static => panic!("Cannot stage a static buffer."),
                UpdateMode::Dynamic => {
                    if let Some(stage) = &self.stage {
                        return stage.stage::<T>();
                    } else {
                        panic!("Dynamic buffer missing stage host buffer.")
                    }
                }
                UpdateMode::Readback => {
                    if let Some(stage) = &self.stage {
                        return stage.stage::<T>();
                    } else {
                        panic!("Readback buffer missing stage host buffer.")
                    }
                }
            },
        }
    }
    pub fn write<T: Pod + Send + Sync>(&mut self, data: Vec<T>) -> BufferFuture {
        let data_bytes = cast_vec(data);
        match &self.location {
            Location::Host => {
                let mut slice = self.inner.write().unwrap();
                slice.copy_from_slice(&data_bytes);
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

                        let finished = command_buffer
                            .execute(self.ctx.queue.clone())
                            .unwrap()
                            .then_signal_fence_and_flush()
                            .unwrap();

                        finished.wait(None).unwrap();
                    }
                }
                UpdateMode::Readback => {
                    panic!("Cannot write to a readback buffer from host");
                }
            },
        }
        todo!()
    }
    pub fn inner<T: Pod + Send + Sync>(&self) -> Subbuffer<[T]> {
        let inner = self.inner_bytes();
        inner.cast_aligned().clone()
    }
    pub fn inner_bytes(&self) -> Subbuffer<[u8]> {
        self.inner.clone()
    }
}
