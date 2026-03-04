use bytemuck::Pod;
use image::{ImageBuffer, Rgba};
use std::sync::Arc;
use vulkano::command_buffer::{
    self, ClearColorImageInfo, CommandBufferUsage, CopyBufferToImageInfo, CopyImageToBufferInfo,
};
use vulkano::format::ClearColorValue;
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

pub struct Image {
    ctx: BackendContext,
    inner: Arc<VulkanoImage>,
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
        let staging = Buffer::new(
            ctx.clone(),
            vec![0u8; extent.iter().product::<u32>() as usize * texel_size.format() as usize],
            Location::Host,
        );
        Self {
            ctx,
            texel_size,
            inner: image,
            extent,
            staging,
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
                ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST,
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
                self.inner.clone(),
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
        let bytes_per_pixel = std::mem::size_of::<T>();
        let width = self.inner.extent()[0];
        let height = self.inner.extent()[1];
        let depth = self.inner.extent()[2];
        let size = (width * height * depth) as usize * bytes_per_pixel;

        let mut cmd_buf = AutoCommandBufferBuilder::primary(
            self.ctx.command_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cmd_buf
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                self.inner.clone(),
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
    pub fn clear(&self, rgb: [f32; 4]) {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.ctx.command_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .clear_color_image(ClearColorImageInfo {
                clear_value: ClearColorValue::Float(rgb),
                ..ClearColorImageInfo::image(self.inner.clone())
            })
            .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                self.inner.clone(),
                self.staging.inner::<u8>(),
            ))
            .unwrap();
        let command_buffer = builder.build().unwrap();
        let future = sync::now(self.ctx.device.clone())
            .then_execute(self.ctx.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
        let buffer_content = self.staging.read();
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
            self.extent[0],
            self.extent[1],
            &buffer_content[..],
        )
        .unwrap();
        image.save("image.png").unwrap();
    }
}

#[cfg(test)]
mod image_tests {
    use crate::gpu::{
        backend::BackendContext,
        memory::image::{Image, ImageIntent, TexelSize},
    };

    #[test]
    fn test() {
        let ctx = BackendContext::new();
        let t = Image::new(ctx, TexelSize::RGBA8, ImageIntent::Readback, [800, 600, 1]);
        t.clear([1.0, 0.3, 0.8, 1.0]);
    }
}
