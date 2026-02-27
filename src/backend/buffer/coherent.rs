use vulkano::{
    buffer::{BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::backend::{
    Context,
    buffer::{Buffer, BufferReadFuture, BufferWriteFuture},
};

pub struct CoherentBuffer<T: BufferContents + Copy> {
    ctx: Context,
    inner: Subbuffer<[T]>,
}
impl<T: BufferContents + Copy> Buffer<T> for CoherentBuffer<T> {
    fn from_data(ctx: Context, data: &[T]) -> Self {
        let buffer = vulkano::buffer::Buffer::from_iter(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            data.iter().copied(),
        )
        .expect("Failed to create variable buffer.");
        Self { ctx, inner: buffer }
    }

    fn read(&self) -> BufferReadFuture<T> {
        let mapping = self.inner.read().expect("Dynamic buffer read failed.");
        let snapshot = mapping.to_vec().clone();
        let data = Box::new(move || snapshot);
        BufferReadFuture { inner: None, data }
    }

    fn write(&mut self, data: &[T]) -> BufferWriteFuture {
        let mut mapping = self.inner.write().unwrap();
        mapping[..data.len()].copy_from_slice(data);
        BufferWriteFuture { inner: None }
    }
}

#[cfg(test)]
mod variable_buffer_tests {
    use crate::backend::Buffer;
    use crate::backend::Context;
    #[test]
    fn variable_buffer_round_trip() {
        let ctx = Context::new();
        let initial = vec![1.0f32, 2.0, 3.0, 4.0];

        let mut buffer = ctx.create_coherent_buffer(&initial);

        assert_eq!(buffer.read().wait(), initial);

        let updated = vec![5.0, 2.3, 17.6, 32.0];
        buffer.write(&updated).wait();

        assert_eq!(buffer.read().wait(), updated);
    }
    #[test]
    fn variable_buffer_partial_update() {
        let ctx = Context::new();
        let initial = vec![1u32, 2, 3, 4, 5];

        let mut buffer = ctx.create_coherent_buffer(&initial);

        buffer.write(&[10, 20]);

        let result = buffer.read().wait();
        assert_eq!(result, vec![10, 20, 3, 4, 5]);
    }
    #[test]
    fn variable_buffer_multiple_updates() {
        let ctx = Context::new();
        let mut buffer = ctx.create_coherent_buffer(&[0i32; 4]);

        for i in 0..10 {
            let data = vec![i, i + 1, i + 2, i + 3];
            buffer.write(&data).wait();
            assert_eq!(buffer.read().wait(), data);
        }
    }
}
