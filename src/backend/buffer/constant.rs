use vulkano::{
    buffer::{BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::{self, GpuFuture},
};

use crate::backend::{Context, buffer::Buffer};

pub struct ConstantBuffer<T: BufferContents + Copy> {
    ctx: Context,
    inner: Subbuffer<[T]>,
}
impl<T: BufferContents + Copy> Buffer<T> for ConstantBuffer<T> {
    fn from_data(ctx: Context, data: &[T]) -> Self {
        let buffer = vulkano::buffer::Buffer::new_slice(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            data.len() as u64,
        )
        .expect("Failed to create device local buffer.");
        Self { ctx, inner: buffer }
    }

    fn read(&self) -> Vec<T> {
        let staging = vulkano::buffer::Buffer::new_slice::<T>(
            self.ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            self.inner.len(),
        )
        .expect("Failed to create staging buffer");
        let mut builder = AutoCommandBufferBuilder::primary(
            self.ctx.command_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create command buffer builder");
        builder
            .copy_buffer(CopyBufferInfo::buffers(self.inner.clone(), staging.clone()))
            .expect("Failed to record copy command");
        let command_buffer = builder.build().expect("Failed to build command buffer");
        sync::now(self.ctx.device.clone())
            .then_execute(self.ctx.queue.clone(), command_buffer)
            .expect("Failed to submit copy command")
            .then_signal_fence_and_flush()
            .expect("Failed to flush fence")
            .wait(None)
            .expect("Failed to wait for fence");
        let mapping = staging.read().expect("Failed to map staging buffer");
        mapping.to_vec()
    }

    fn update(&mut self, data: &[T]) {
        use vulkano::{
            buffer::{Buffer, BufferCreateInfo, BufferUsage},
            command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo},
            memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
            sync::{self, GpuFuture},
        };

        let staging = Buffer::from_iter(
            self.ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.iter().cloned(),
        )
        .expect("Failed to create staging buffer");

        let mut builder = AutoCommandBufferBuilder::primary(
            self.ctx.command_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create command buffer");

        builder
            .copy_buffer(CopyBufferInfo::buffers(staging, self.inner.clone()))
            .expect("Failed to record copy");

        let command_buffer = builder.build().expect("Failed to build command buffer");

        // 3. Submit and wait
        sync::now(self.ctx.device.clone())
            .then_execute(self.ctx.queue.clone(), command_buffer)
            .expect("Failed to submit")
            .then_signal_fence_and_flush()
            .expect("Failed to flush")
            .wait(None)
            .expect("Failed to wait");
    }
}
impl<T: BufferContents + Copy> ConstantBuffer<T> {}
