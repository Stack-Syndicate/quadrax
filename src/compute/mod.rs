pub mod mathematics;

use crate::backend::{
    BackendContext,
    buffer::{Buffer, BufferRegistry},
};
use bytemuck::{Pod, Zeroable};
use std::{collections::VecDeque, sync::Arc};
use vulkano::{descriptor_set::allocator::StandardDescriptorSetAllocator, sync::GpuFuture};

pub mod vector_ops {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/vector_ops.comp",
    }
}

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy, Debug, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}
impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
}

pub trait ComputePass {
    fn buffers(&self) -> Vec<Arc<dyn Buffer>>;
    fn dispatch(&self) -> ComputeFuture;
    fn backend(&self) -> BackendContext;
    fn parameters(&self) -> Vec<Arc<dyn ComputeParameter>>;
}
impl ComputePass for Arc<dyn ComputePass> {
    fn dispatch(&self) -> ComputeFuture {
        (**self).dispatch()
    }
    fn backend(&self) -> BackendContext {
        (**self).backend()
    }
    fn buffers(&self) -> Vec<Arc<dyn Buffer>> {
        (**self).buffers()
    }
    fn parameters(&self) -> Vec<Arc<dyn ComputeParameter>> {
        (**self).parameters()
    }
}

pub struct ComputeContext {
    backend: BackendContext,
    queue: VecDeque<Arc<dyn ComputePass>>,
    registry: BufferRegistry,
    descriptor_allocator: Arc<StandardDescriptorSetAllocator>,
}
impl ComputeContext {
    pub fn new(backend: BackendContext) -> Self {
        Self {
            backend: backend.clone(),
            queue: VecDeque::new(),
            registry: BufferRegistry::new(),
            descriptor_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                backend.device.clone(),
                Default::default(),
            )),
        }
    }

    pub fn add_pass(&mut self, pass: Arc<dyn ComputePass>) {
        self.queue.push_back(pass);
    }

    pub fn dispatch(&mut self) {
        while let Some(p) = self.queue.pop_front() {
            p.dispatch().wait();
        }
    }
}

pub trait ComputeParameter {
    fn to_u32(&self) -> u32;
}

pub struct ComputeFuture {
    inner: Option<Box<dyn GpuFuture>>,
}
impl ComputeFuture {
    pub fn wait(self) {
        if let Some(fut) = self.inner {
            fut.then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
        }
    }
}
