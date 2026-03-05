use bytemuck::Pod;
use image::{ImageBuffer, Rgba};
use std::sync::Arc;
use vulkano::command_buffer::{
    ClearColorImageInfo, CommandBufferUsage, CopyBufferToImageInfo, CopyImageToBufferInfo,
};
use vulkano::format::ClearColorValue;
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{ImageAspects, ImageSubresourceRange};
use vulkano::sync::{self, now};
use vulkano::{
    command_buffer::AutoCommandBufferBuilder,
    format::Format,
    image::{ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::GpuFuture,
};

use crate::gpu::memory::buffer::BufferFuture;
use crate::gpu::{
    backend::BackendContext,
    memory::buffer::{Buffer, Location},
};
type VulkanoImage = vulkano::image::Image;
pub enum ImageIntent {
    Texture,
    RenderTarget,
    Storage,
    Readback,
}
#[derive(Clone, Debug)]
pub enum TexelSize {
    R8 = 1,
    RG8 = 2,
    RGB8 = 3,
    RGBA8 = 4,
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

pub type ImageFuture = BufferFuture;

#[derive(Clone)]
pub struct Image {
    ctx: BackendContext,
    pub inner: Arc<ImageView>,
    extent: [u32; 3],
    texel_size: TexelSize,
    staging: Buffer,
}
impl Image {
    pub fn new(
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
        let image_view = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                subresource_range: ImageSubresourceRange {
                    aspects: ImageAspects::COLOR,
                    ..image.subresource_range()
                },
                format: image.format(),
                usage: image.usage(),
                ..Default::default()
            },
        )
        .unwrap();
        let staging = Buffer::new(
            ctx.clone(),
            vec![0u8; extent.iter().product::<u32>() as usize * texel_size.format() as usize],
            Location::Host,
        );
        Self {
            ctx,
            inner: image_view,
            extent,
            staging,
            texel_size,
        }
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
                ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::COLOR_ATTACHMENT,
                MemoryTypeFilter::PREFER_HOST,
            ),
        }
    }
    pub fn write<T: Pod + Send + Sync>(&mut self, data: Vec<T>) -> ImageFuture {
        self.staging.write(data);
        let mut cmd_buf = AutoCommandBufferBuilder::primary(
            self.ctx.command_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cmd_buf
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                self.staging.inner::<T>(),
                self.inner.image().clone(),
            ))
            .unwrap();

        let cb = cmd_buf.build().unwrap();
        ImageFuture {
            inner: Some(
                now(self.ctx.device.clone())
                    .then_execute(self.ctx.queue.clone(), cb)
                    .unwrap()
                    .boxed(),
            ),
        }
    }

    pub fn read<T: Pod + Send + Sync>(&self) -> Vec<T> {
        let mut cmd_buf = AutoCommandBufferBuilder::primary(
            self.ctx.command_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cmd_buf
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                self.inner.image().clone(),
                self.staging.inner::<T>(),
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

        let slice: Vec<T> = self.staging.read();
        slice
    }
    pub fn clear(&self, rgb: [f32; 4]) -> BufferFuture {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.ctx.command_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .clear_color_image(ClearColorImageInfo {
                clear_value: ClearColorValue::Float(rgb),
                ..ClearColorImageInfo::image(self.inner.image().clone())
            })
            .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                self.inner.image().clone(),
                self.staging.inner::<u8>(),
            ))
            .unwrap();
        let command_buffer = builder.build().unwrap();
        let future = sync::now(self.ctx.device.clone())
            .then_execute(self.ctx.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        BufferFuture {
            inner: Some(future.boxed()),
        }
    }
    pub fn save(&self, path: &str) {
        let content = self.read();
        let image =
            ImageBuffer::<Rgba<u8>, _>::from_raw(self.extent[0], self.extent[1], &content[..])
                .unwrap();
        image.save(path).unwrap();
    }
    pub fn texel(&self) -> TexelSize {
        self.texel_size.clone()
    }
}

#[test]
fn clear_test() {
    let ctx = BackendContext::new();
    let t = Image::new(ctx, TexelSize::RGBA8, ImageIntent::Readback, [800, 600, 1]);
    t.clear([0.0, 0.3, 0.8, 1.0]).wait();
    t.save("image.png");
}
