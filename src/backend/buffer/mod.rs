pub mod coherent;
pub mod staged;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use bytemuck::{Pod, checked::cast_slice};
use vulkano::sync::GpuFuture;

pub trait Buffer {
    fn read_bytes(&self) -> BufferReadFuture<u8>;
    fn write_bytes(&mut self, bytes: &[u8]) -> BufferWriteFuture;
    fn len(&self) -> usize;
}
pub trait BufferTyped: Buffer {
    fn read<T: Pod>(&self) -> BufferReadFuture<T> {
        assert!(self.len() % std::mem::size_of::<T>() == 0);
        self.read_bytes().cast::<T>()
    }

    fn write<T: Pod>(&mut self, data: &[T]) -> BufferWriteFuture {
        let bytes = cast_slice(data);
        assert!(bytes.len() <= self.len());
        self.write_bytes(bytes)
    }
}
impl<B: Buffer + ?Sized> BufferTyped for B {}

pub struct BufferWriteFuture {
    inner: Option<Box<dyn GpuFuture>>,
}
impl BufferWriteFuture {
    pub fn wait(self) {
        if let Some(fut) = self.inner {
            fut.then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
        }
    }
    pub fn is_trivial(&self) -> bool {
        self.inner.is_none()
    }
}

pub struct BufferReadFuture<T> {
    inner: Option<Box<dyn GpuFuture>>,
    data: Box<dyn FnOnce() -> Vec<T>>,
}
impl<T> BufferReadFuture<T> {
    pub fn wait(self) -> Vec<T> {
        if let Some(fut) = self.inner {
            fut.then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
        }
        (self.data)()
    }
    pub fn cast<D: Pod>(&self) -> BufferReadFuture<D> {
        todo!()
    }
}

#[derive(Clone)]
pub struct BufferRegistry {
    inner: Arc<Mutex<HashMap<u32, Arc<dyn Buffer>>>>,
}
impl BufferRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    pub fn insert(&mut self, key: u32, value: Arc<dyn Buffer>) {
        let mut inner_lock = self.inner.lock().unwrap();
        inner_lock.insert(key, value);
    }
    pub fn get(&self) -> Arc<dyn Buffer> {
        todo!()
    }
}
