use vulkano::{
    buffer::{BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::{GpuFuture, future::NowFuture, now},
};

use crate::backend::{
    Context,
    buffer::{Buffer, BufferReadFuture, BufferWriteFuture},
};

pub struct StagedBuffer<T: BufferContents + Copy> {
    ctx: Context,
    inner: Subbuffer<[T]>,
    staging: Subbuffer<[T]>,
}
impl<T: BufferContents + Copy> Buffer<T> for StagedBuffer<T> {
    fn from_data(ctx: Context, data: &[T]) -> Self {
        let staging = vulkano::buffer::Buffer::from_iter(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            data.iter().copied(),
        )
        .expect("Failed to create staged buffer.");
        let buffer = vulkano::buffer::Buffer::from_iter(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            data.iter().copied(),
        )
        .expect("Failed to create device-local buffer.");
        Self {
            ctx,
            inner: buffer,
            staging,
        }
    }

    fn read(&self) -> BufferReadFuture<T> {
        let mut cmd = AutoCommandBufferBuilder::primary(
            self.ctx.command_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        cmd.copy_buffer(CopyBufferInfo::buffers(
            self.inner.clone(),
            self.staging.clone(),
        ))
        .unwrap();
        let cmd_buf = cmd.build().unwrap();
        let future = now(self.ctx.device.clone())
            .then_execute(self.ctx.queue.clone(), cmd_buf)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        let staging = self.staging.clone();
        let data = Box::new(move || {
            staging
                .read()
                .expect("Failed to read staging buffer.")
                .to_vec()
        });
        BufferReadFuture {
            inner: Some(future.boxed()),
            data,
        }
    }

    fn write(&mut self, data: &[T]) -> BufferWriteFuture {
        {
            let mut mapping = self.staging.write().unwrap();
            mapping[..data.len()].copy_from_slice(data);
        }

        let mut cmd = AutoCommandBufferBuilder::primary(
            self.ctx.command_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cmd.copy_buffer(CopyBufferInfo::buffers(
            self.staging.clone(),
            self.inner.clone(),
        ))
        .unwrap();
        let cmd_buf = cmd.build().unwrap();
        let future = now(self.ctx.device.clone()) // start a dummy future
            .then_execute(self.ctx.queue.clone(), cmd_buf) // submit command buffer
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        BufferWriteFuture {
            inner: Some(future.boxed()),
        }
    }
}

#[cfg(test)]
mod staged_buffer_tests {
    use crate::backend::Buffer;
    use crate::backend::Context;
    #[test]
    fn staged_buffer_round_trip() {
        let ctx = Context::new();
        let initial = vec![1.0f32, 2.0, 3.0, 4.0];

        let mut buffer = ctx.create_coherent_buffer(&initial);

        assert_eq!(buffer.read().wait(), initial);

        let updated = vec![5.0, 2.3, 17.6, 32.0];
        buffer.write(&updated).wait();

        assert_eq!(buffer.read().wait(), updated);
    }
    #[test]
    fn staged_buffer_partial_update() {
        let ctx = Context::new();
        let initial = vec![1u32, 2, 3, 4, 5];

        let mut buffer = ctx.create_coherent_buffer(&initial);

        buffer.write(&[10, 20]);

        let result = buffer.read().wait();
        assert_eq!(result, vec![10, 20, 3, 4, 5]);
    }
    #[test]
    fn staged_buffer_multiple_updates() {
        let ctx = Context::new();
        let mut buffer = ctx.create_coherent_buffer(&[0i32; 4]);

        for i in 0..10 {
            let data = vec![i, i + 1, i + 2, i + 3];
            buffer.write(&data).wait();
            assert_eq!(buffer.read().wait(), data);
        }
    }
}
