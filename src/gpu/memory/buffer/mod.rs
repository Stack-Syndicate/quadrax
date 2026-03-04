use bytemuck::{Pod, cast_vec};
use vulkano::{
    buffer::{BufferCreateInfo, BufferUsage, Subbuffer},
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
            Location::Host => {}
            Location::Device(update_mode) => match update_mode {
                UpdateMode::Static => {}
                UpdateMode::Dynamic => {}
                UpdateMode::Readback => {}
            },
        }
        todo!()
    }
    pub fn write(&mut self) {
        match &self.location {
            Location::Host => {}
            Location::Device(update_mode) => match update_mode {
                UpdateMode::Static => {}
                UpdateMode::Dynamic => {}
                UpdateMode::Readback => {}
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
