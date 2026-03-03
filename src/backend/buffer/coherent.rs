use bytemuck::{Pod, cast_vec, checked::cast_slice};
use vulkano::{
    buffer::{BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::backend::{
    BackendContext,
    buffer::{Buffer, BufferReadFuture, BufferWriteFuture},
};

#[derive(Clone, Debug)]
pub struct CoherentBuffer {
    ptr: Subbuffer<[u8]>,
}
impl CoherentBuffer {
    pub fn from_data<T: Pod + Send + Sync>(ctx: BackendContext, data: &[T]) -> Self {
        let data_bytes = cast_slice(data);
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
            data_bytes.iter().copied(),
        )
        .expect("Failed to create variable buffer.");
        Self { ptr: buffer }
    }
}
impl Buffer for CoherentBuffer {
    fn read_bytes(&self) -> BufferReadFuture<u8> {
        let mapping = self
            .ptr
            .read()
            .expect("Dynamic buffer read failed.")
            .to_vec();
        let snapshot = cast_vec(mapping);
        let data = Box::new(move || snapshot);
        BufferReadFuture {
            inner: None,
            data: data.clone(),
        }
    }

    fn write_bytes(&mut self, data: &[u8]) -> BufferWriteFuture {
        let mut mapping = self.ptr.write().unwrap();
        mapping[..data.len()].copy_from_slice(data);
        BufferWriteFuture { inner: None }
    }
    fn ptr_bytes(&self) -> Subbuffer<[u8]> {
        self.ptr.clone()
    }
    fn len(&self) -> usize {
        self.ptr.len() as usize
    }
}

#[cfg(test)]
mod variable_buffer_tests {
    use crate::backend::{BackendContext, buffer::BufferTyped};
    #[test]
    fn variable_buffer_round_trip() {
        let ctx = BackendContext::new();
        let initial = vec![1.0f32, 2.0, 3.0, 4.0];

        let mut buffer = ctx.create_coherent_buffer(&initial);

        assert_eq!(buffer.read::<f32>().wait(), initial);

        let updated = vec![5.0, 2.3, 17.6, 32.0];
        buffer.write(&updated).wait();

        assert_eq!(buffer.read::<f32>().wait(), updated);
    }
    #[test]
    fn variable_buffer_partial_update() {
        let ctx = BackendContext::new();
        let initial = vec![1u32, 2, 3, 4, 5];

        let mut buffer = ctx.create_coherent_buffer(&initial);

        buffer.write(&[10, 20]);

        let result = buffer.read::<u32>().wait();
        assert_eq!(result, vec![10, 20, 3, 4, 5]);
    }
    #[test]
    fn variable_buffer_multiple_updates() {
        let ctx = BackendContext::new();
        let mut buffer = ctx.create_coherent_buffer(&[0i32; 4]);

        for i in 0..10 {
            let data = vec![i, i + 1, i + 2, i + 3];
            buffer.write(&data).wait();
            assert_eq!(buffer.read::<i32>().wait(), data);
        }
    }
}
