use crate::gpu::{
    backend::Backend,
    buffer::{Buffer, BufferRole},
};
use bytemuck::Pod;
use std::sync::Arc;
use wgpu::{COPY_BYTES_PER_ROW_ALIGNMENT, Extent3d, TexelCopyBufferInfo};

pub use wgpu::TextureFormat;

pub enum TextureRole {
    Sampled,
    Storage,
    RenderTarget,
    Generic,
}
impl TextureRole {
    pub fn to_usage(&self) -> wgpu::TextureUsages {
        match self {
            TextureRole::Sampled => {
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
            }
            TextureRole::Storage => {
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST
            }
            TextureRole::RenderTarget => {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
            }
            TextureRole::Generic => wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        }
    }
}

pub struct Texture {
    pub inner: Arc<wgpu::Texture>,
    pub view: Arc<wgpu::TextureView>,
    pub usage: Arc<wgpu::TextureUsages>,
    pub backend: Arc<Backend>,
    pub size: wgpu::Extent3d,
    pub format: TextureFormat,
}
impl Texture {
    pub fn new_empty(
        backend: Arc<Backend>,
        size: &[u32; 3],
        role: TextureRole,
        format: TextureFormat,
    ) -> Texture {
        let size = Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: size[2],
            ..Default::default()
        };
        let inner = backend.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: role.to_usage(),
            view_formats: &[format],
        });
        let view = inner.create_view(&wgpu::TextureViewDescriptor::default());
        Texture {
            inner: inner.into(),
            view: view.into(),
            usage: role.to_usage().into(),
            backend,
            size,
            format,
        }
    }
    pub async fn read<T: Pod>(&self) -> Vec<T> {
        let bytes_per_pixel = std::mem::size_of::<T>() as u32;
        let bytes_per_row = (self.size.width * bytes_per_pixel + COPY_BYTES_PER_ROW_ALIGNMENT - 1)
            / COPY_BYTES_PER_ROW_ALIGNMENT
            * COPY_BYTES_PER_ROW_ALIGNMENT;
        let staging = Buffer::new_empty::<T>(
            self.backend.clone(),
            (bytes_per_row * self.size.height * self.size.depth_or_array_layers).into(),
            BufferRole::StagingRead,
        );
        let mut encoder = self
            .backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            self.inner.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &staging.inner,
                layout: wgpu::TexelCopyBufferLayout {
                    bytes_per_row: Some(bytes_per_row),
                    ..Default::default()
                },
            },
            self.size,
        );
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
        let mapped = buffer_slice.get_mapped_range();
        let mut result = Vec::with_capacity((self.size.width * self.size.height) as usize);

        for row in 0..self.size.height as usize {
            let row_start = row * bytes_per_row as usize;
            let row_end = row_start + (self.size.width as usize * bytes_per_pixel as usize);
            let row_bytes = &mapped[row_start..row_end];
            let row_t = row_bytes
                .chunks_exact(bytes_per_pixel as usize)
                .map(|chunk| -> T { bytemuck::from_bytes::<T>(chunk).clone() })
                .collect::<Vec<T>>();
            result.extend_from_slice(&row_t);
        }
        drop(mapped);
        staging.inner.unmap();
        result
    }
    pub async fn write<T: Pod>(&self, data: &[T]) {
        let bytes_per_pixel = std::mem::size_of::<T>() as u32;
        let bytes_per_row = (self.size.width * bytes_per_pixel + COPY_BYTES_PER_ROW_ALIGNMENT - 1)
            / COPY_BYTES_PER_ROW_ALIGNMENT
            * COPY_BYTES_PER_ROW_ALIGNMENT;
        let mut staging_bytes = vec![0u8; (bytes_per_row * self.size.height) as usize];

        for row in 0..self.size.height as usize {
            let src_start = row * self.size.width as usize;
            let src_end = src_start + self.size.width as usize;
            let dst_start = row * bytes_per_row as usize;
            let row_slice: &[u8] = bytemuck::cast_slice(&data[src_start..src_end]);
            staging_bytes[dst_start..dst_start + row_slice.len()].copy_from_slice(row_slice);
        }
        let staging = Buffer::new(
            self.backend.clone(),
            staging_bytes,
            BufferRole::StagingWrite,
        );

        let mut encoder = self
            .backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_texture(
            TexelCopyBufferInfo {
                buffer: &staging.inner,
                layout: wgpu::TexelCopyBufferLayout {
                    bytes_per_row: Some(bytes_per_row),
                    ..Default::default()
                },
            },
            self.inner.as_image_copy(),
            self.size,
        );
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
async fn texture_crud() {
    let backend = Backend::new().await;
    let texture = Texture::new_empty(
        backend.into(),
        &[64, 1, 1],
        TextureRole::Generic,
        TextureFormat::Rgba8Uint,
    );
    println!("{:?}", texture.read::<[u8; 4]>().await);
    texture.write(&vec![[255u8; 4]; 64 * 1]).await;
    println!("{:?}", texture.read::<[u8; 4]>().await);
}
