use bytemuck::Pod;
use std::sync::Arc;
use vulkano::command_buffer::{CommandBufferUsage, CopyBufferToImageInfo, CopyImageToBufferInfo};
use vulkano::sync::now;
use vulkano::{
    command_buffer::AutoCommandBufferBuilder,
    format::Format,
    image::{ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::GpuFuture,
};

use crate::gpu::{
    backend::BackendContext,
    memory::buffer::{Buffer, Location},
};

type VulkanoImage = vulkano::image::Image;
type VulkanoBuffer = vulkano::buffer::Buffer;
pub enum ImageIntent {
    Texture,
    RenderTarget,
    Storage,
    Readback,
}
pub enum TexelSize {
    R8,
    RG8,
    RGB8,
    RGBA8,
}
impl TexelSize {
    pub fn format(&self) -> Format {
        match self {
            TexelSize::R8 => Format::R8_UNORM,
            TexelSize::RG8 => Format::R8G8_UNORM,
            TexelSize::RGB8 => Format::R8G8B8_UNORM,
            TexelSize::RGBA8 => Format::R8G8B8A8_UNORM,
        }
    }
}

pub struct Image {
    ctx: BackendContext,
    inner: Arc<VulkanoImage>,
}
impl Image {
    pub fn new<T: Pod + Send + Sync>(
        ctx: BackendContext,
        texel_size: TexelSize,
        intent: ImageIntent,
        extent: [u32; 3],
    ) -> Self {
        let mut dim = 0;
        for e in extent {
            if e > 1 {
                dim += 1;
            }
        }
        let image_type = match dim {
            1 => ImageType::Dim1d,
            2 => ImageType::Dim2d,
            3 => ImageType::Dim3d,
            _ => panic!("Invalid image extent."),
        };
        let format = texel_size.format();
        let (usage, memory_type_filter) = Image::usage_and_memory(intent);
        let image = VulkanoImage::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                image_type,
                format,
                extent,
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter,
                ..Default::default()
            },
        )
        .unwrap();
        Self { ctx, inner: image }
    }
    fn usage_and_memory(intent: ImageIntent) -> (ImageUsage, MemoryTypeFilter) {
        match intent {
            ImageIntent::Texture => (
                ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
                MemoryTypeFilter::PREFER_DEVICE,
            ),
            ImageIntent::RenderTarget => (
                ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC,
                MemoryTypeFilter::PREFER_DEVICE,
            ),
            ImageIntent::Storage => (
                ImageUsage::STORAGE | ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
                MemoryTypeFilter::PREFER_DEVICE,
            ),
            ImageIntent::Readback => (
                ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST,
                MemoryTypeFilter::PREFER_HOST,
            ),
        }
    }
    pub fn write<T: Pod + Send + Sync>(&self, data: &[T]) -> Arc<dyn GpuFuture> {
        let staging = Buffer::new(self.ctx.clone(), data.to_vec(), Location::Host);

        let mut cmd_buf = AutoCommandBufferBuilder::primary(
            self.ctx.command_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cmd_buf
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                staging.inner::<T>(),
                self.inner.clone(),
            ))
            .unwrap();

        let cb = cmd_buf.build().unwrap();

        Arc::new(
            now(self.ctx.device.clone())
                .then_execute(self.ctx.queue.clone(), cb)
                .unwrap()
                .boxed(),
        )
    }

    pub fn read<T: Pod + Send + Sync>(&self) -> Vec<T> {
        let bytes_per_pixel = std::mem::size_of::<T>();
        let width = self.inner.extent()[0];
        let height = self.inner.extent()[1];
        let depth = self.inner.extent()[2];
        let size = (width * height * depth) as usize * bytes_per_pixel;

        let staging = Buffer::new(self.ctx.clone(), vec![0u8; size], Location::Host);

        let mut cmd_buf = AutoCommandBufferBuilder::primary(
            self.ctx.command_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cmd_buf
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                self.inner.clone(),
                staging.inner::<T>(),
            ))
            .unwrap();

        let cb = cmd_buf.build().unwrap();

        now(self.ctx.device.clone())
            .then_execute(self.ctx.queue.clone(), cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let slice: Vec<T> = staging.read();
        slice
    }
}
