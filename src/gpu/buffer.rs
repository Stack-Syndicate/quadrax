use std::sync::Arc;
use wgpu::util::DeviceExt;

use bytemuck::{Pod, cast_slice};
use wgpu::{BufferDescriptor, util::BufferInitDescriptor};

use crate::gpu::backend::Backend;

pub enum BufferRole {
    Uniform,
    Storage,
    Vertex,
    Generic,
    StagingWrite,
    StagingRead,
}
impl BufferRole {
    pub fn to_usage(&self) -> wgpu::BufferUsages {
        return match self {
            BufferRole::Storage => {
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
            }
            BufferRole::Uniform => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            BufferRole::Vertex => wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            BufferRole::Generic => wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            BufferRole::StagingRead => wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            BufferRole::StagingWrite => {
                wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC
            }
        };
    }
}

#[derive(Clone)]
pub struct Buffer {
    pub inner: Arc<wgpu::Buffer>,
    pub usage: Arc<wgpu::BufferUsages>,
    pub backend: Arc<Backend>,
    pub length: u64,
    pub size: u64,
}
impl Buffer {
    pub fn new_empty<T: Pod>(backend: Arc<Backend>, size: u64, role: BufferRole) -> Self {
        let inner = backend.device.create_buffer(&BufferDescriptor {
            label: None,
            usage: role.to_usage(),
            size: size,
            mapped_at_creation: false,
        });
        let length = size / std::mem::size_of::<T>() as u64;
        Self {
            inner: inner.into(),
            usage: role.to_usage().into(),
            backend,
            length,
            size,
        }
    }
    pub fn new<T: Pod>(backend: Arc<Backend>, data: Vec<T>, role: BufferRole) -> Self {
        let inner = backend.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            usage: role.to_usage(),
            contents: cast_slice(&data),
        });
        let length = data.len() as u64;
        let size = std::mem::size_of::<T>() as u64 * length as wgpu::BufferAddress;
        Self {
            inner: inner.into(),
            usage: role.to_usage().into(),
            backend,
            length,
            size,
        }
    }
    pub async fn read<T: Pod>(&self) -> Vec<T> {
        let staging =
            Buffer::new_empty::<T>(self.backend.clone(), self.size, BufferRole::StagingRead);
        let mut encoder = self
            .backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.inner, 0, &staging.inner, 0, self.size);
        self.backend.queue.submit(Some(encoder.finish()));
        let buffer_slice = staging.inner.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| sender.send(res).unwrap());
        self.backend
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        receiver.await.unwrap().unwrap();
        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.inner.unmap();
        result
    }
    pub async fn write<T: Pod>(&mut self, data: Vec<T>) {
        let staging = Buffer::new(self.backend.clone(), data, BufferRole::StagingWrite);
        let mut encoder = self
            .backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&staging.inner, 0, &self.inner, 0, self.size);
        self.backend.queue.submit(Some(encoder.finish()));
        self.backend
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
    }
}

#[pollster::test]
async fn buffer_crud() {
    let backend = Backend::new().await;
    let mut buffer = Buffer::new(backend.into(), vec![0u32; 16], BufferRole::Generic);
    assert_eq!(buffer.read::<u32>().await, vec![0u32; 16]);
    buffer.write(vec![1u32; 16]).await;
    assert_eq!(buffer.read::<u32>().await, vec![1u32; 16]);
}
