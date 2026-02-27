pub mod coherent;
pub mod staged;

use vulkano::{buffer::BufferContents, sync::GpuFuture};

use crate::backend::Context;

pub trait Buffer<T: BufferContents + Copy> {
    fn from_data(ctx: Context, data: &[T]) -> Self;
    fn read(&self) -> BufferReadFuture<T>;
    fn write(&mut self, data: &[T]) -> BufferWriteFuture;
}

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
}
