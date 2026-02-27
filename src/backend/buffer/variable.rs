use vulkano::{
    buffer::{BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::backend::{Context, buffer::Buffer};

pub struct VariableBuffer<T: BufferContents + Copy> {
    ctx: Context,
    inner: Subbuffer<[T]>,
}
impl<T: BufferContents + Copy> Buffer<T> for VariableBuffer<T> {
    fn from_data(ctx: Context, data: &[T]) -> Self {
        let buffer = vulkano::buffer::Buffer::from_iter(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
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

    fn read(&self) -> Vec<T> {
        let mapping = self.inner.read().expect("Dynamic buffer read failed.");
        mapping.to_vec()
    }

    fn update(&mut self, data: &[T]) {
        let mut mapping = self.inner.write().unwrap();
        mapping[..data.len()].copy_from_slice(data);
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

        let mut buffer = ctx.create_variable_buffer(&initial);

        assert_eq!(buffer.read(), initial);

        let updated = vec![5.0, 2.3, 17.6, 32.0];
        buffer.update(&updated);

        assert_eq!(buffer.read(), updated);
    }
    #[test]
    fn variable_buffer_partial_update() {
        let ctx = Context::new();
        let initial = vec![1u32, 2, 3, 4, 5];

        let mut buffer = ctx.create_variable_buffer(&initial);

        buffer.update(&[10, 20]);

        let result = buffer.read();
        assert_eq!(result, vec![10, 20, 3, 4, 5]);
    }
    #[test]
    fn variable_buffer_multiple_updates() {
        let ctx = Context::new();
        let mut buffer = ctx.create_variable_buffer(&[0i32; 4]);

        for i in 0..10 {
            let data = vec![i, i + 1, i + 2, i + 3];
            buffer.update(&data);
            assert_eq!(buffer.read(), data);
        }
    }
}
