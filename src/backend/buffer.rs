use std::marker::ConstParamTy;
use std::sync::Arc;

use vulkano::buffer::BufferCreateInfo;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::sync::{GpuFuture, now};
use vulkano::{
    buffer::{BufferContents, BufferUsage, Subbuffer},
    memory::allocator::MemoryTypeFilter,
};

use crate::backend::Context;

/// Distinguishes between different types of GPU buffer during initialization.
#[derive(ConstParamTy, PartialEq, Eq)]
pub enum Intent {
    /// The buffer never changes and is not accessed from the CPU again.
    Static,
    /// The buffer may change and is accessible from the CPU.
    Dynamic,
}
/// Chooses a [MemoryTypeFilter] and [BufferUsage] for each [Intent] variant.
pub trait BufferStrategy {
    fn memory_filter(&self) -> MemoryTypeFilter;
    fn buffer_usage(&self) -> BufferUsage;
}
impl BufferStrategy for Intent {
    fn memory_filter(&self) -> MemoryTypeFilter {
        match self {
            Intent::Static => return MemoryTypeFilter::PREFER_DEVICE,
            Intent::Dynamic => {
                return MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE;
            }
        }
    }
    fn buffer_usage(&self) -> BufferUsage {
        match self {
            Intent::Static => {
                return BufferUsage::VERTEX_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST;
            }
            Intent::Dynamic => return BufferUsage::UNIFORM_BUFFER | BufferUsage::VERTEX_BUFFER,
        }
    }
}
/// High-level buffer object with behavior defined by [Intent].
pub struct Buffer<T: BufferContents + Copy> {
    ctx: Arc<Context>,
    inner: Subbuffer<[T]>,
    pub(crate) intent: Intent,
}
impl<T: BufferContents + Copy> Buffer<T> {
    /// Creates a new [Buffer] object filled with data implementing [BufferContents], and returns a
    /// boxed [GpuFuture] tracking the buffer updating with new data.
    pub fn from_data_async(
        ctx: Arc<Context>,
        data: &[T],
        intent: Intent,
    ) -> (Self, Box<dyn GpuFuture>) {
        let usage = intent.buffer_usage();
        let filter = intent.memory_filter();
        let inner = vulkano::buffer::Buffer::new_slice(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: filter,
                ..Default::default()
            },
            data.len() as u64,
        )
        .expect("GPU buffer allocation failed");
        let mut buffer = Self { ctx, inner, intent };
        let future = buffer.update_async(data);
        (buffer, future)
    }
    pub fn from_data(ctx: Arc<Context>, data: &[T], intent: Intent) -> Self {
        let (buffer, future) = Buffer::from_data_async(ctx, data, intent);
        future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        buffer
    }
    /// Update the [Buffer] with new data, depending on the [Buffer] [Intent], and return a
    /// boxed [GpuFuture].
    pub fn update_async(&mut self, data: &[T]) -> Box<dyn GpuFuture> {
        if data.len() as u64 > self.inner.len() {
            todo!()
        }
        match self.intent {
            Intent::Dynamic => {
                let mut mapping = self.inner.write().unwrap();
                mapping[..data.len()].copy_from_slice(data);
                Box::new(now(self.ctx.device.clone()))
            }
            Intent::Static => {
                let staging = vulkano::buffer::Buffer::new_slice(
                    self.ctx.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    data.len() as u64,
                )
                .unwrap();
                staging.write().unwrap().copy_from_slice(data);
                let mut builder = AutoCommandBufferBuilder::primary(
                    self.ctx.command_allocator.clone(),
                    self.ctx.queue.queue_family_index(),
                    vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();
                builder
                    .copy_buffer(CopyBufferInfo::buffers(staging, self.inner.clone()))
                    .unwrap();
                let command_buffer = builder.build().unwrap();
                let future = now(self.ctx.device.clone())
                    .then_execute(self.ctx.queue.clone(), command_buffer)
                    .unwrap();
                let boxed = future.boxed();
                boxed
            }
        }
    }
    /// Update the [Buffer] with new data, depending on the [Buffer] [Intent].
    pub fn update(&mut self, data: &[T]) {
        let future = self.update_async(data);
        future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        println!("Buffer update successful");
    }
    pub fn read(&self) -> Vec<T> {
        match self.intent {
            Intent::Static => {
                let staging = vulkano::buffer::Buffer::new_slice(
                    self.ctx.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    self.inner.len(),
                )
                .expect("Failed to create staging buffer for read");
                let mut builder = AutoCommandBufferBuilder::primary(
                    self.ctx.command_allocator.clone(),
                    self.ctx.queue.queue_family_index(),
                    vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();
                builder
                    .copy_buffer(CopyBufferInfo::buffers(self.inner.clone(), staging.clone()))
                    .unwrap();
                let command_buffer = builder.build().unwrap();
                now(self.ctx.device.clone())
                    .then_execute(self.ctx.queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap()
                    .wait(None)
                    .unwrap();
                let mapping = staging.read().expect("Failed to map staging buffer");
                mapping.to_vec()
            }
            Intent::Dynamic => {
                let mapping = self.inner.read().expect("Dynamic buffer read failed.");
                mapping.to_vec()
            }
        }
    }
}
