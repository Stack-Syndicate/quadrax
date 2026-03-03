use crate::backend::BackendContext;
use image::{ImageBuffer, Rgba};
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents, SubpassEndInfo,
    },
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::{self, GpuFuture},
};

pub mod screen_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/screen.frag",
    }
}
pub mod screen_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/screen.vert"
    }
}
#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub struct ScreenVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

pub struct Screen {
    pub pipeline: Arc<GraphicsPipeline>,
    pub vertex_buffer: Subbuffer<[ScreenVertex]>,
    pub render_pass: Arc<RenderPass>,
    pub ctx: BackendContext,
    pub size: (f32, f32),
}

impl Screen {
    pub fn new(ctx: BackendContext, size: (f32, f32)) -> Self {
        let vs = screen_vert::load(ctx.device.clone()).expect("failed to create shader module");
        let fs = screen_frag::load(ctx.device.clone()).expect("failed to create shader module");

        let vertices = [
            ScreenVertex {
                position: [-1.0, -1.0],
            },
            ScreenVertex {
                position: [-1.0, 1.0],
            },
            ScreenVertex {
                position: [1.0, -1.0],
            },
            ScreenVertex {
                position: [1.0, -1.0],
            },
            ScreenVertex {
                position: [-1.0, 1.0],
            },
            ScreenVertex {
                position: [1.0, 1.0],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();
        let render_pass = vulkano::single_pass_renderpass!(
            ctx.device.clone(),
            attachments: {
                color: {
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let pipeline = {
            let vs_entry = vs.entry_point("main").unwrap();
            let fs_entry = fs.entry_point("main").unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(vs_entry.clone()),
                PipelineShaderStageCreateInfo::new(fs_entry.clone()),
            ];
            let layout = PipelineLayout::new(
                ctx.device.clone(),
                vulkano::pipeline::layout::PipelineLayoutCreateInfo::default(),
            )
            .unwrap();
            let vertex_input_state = ScreenVertex::per_vertex().definition(&vs_entry).unwrap();
            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
            GraphicsPipeline::new(
                ctx.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(Default::default()),
                    viewport_state: Some(ViewportState::default()),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: vulkano::pipeline::graphics::rasterization::CullMode::None,
                        ..Default::default()
                    }),
                    multisample_state: Some(Default::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };
        Self {
            pipeline,
            vertex_buffer,
            render_pass,
            ctx,
            size,
        }
    }
    pub fn draw(&self) {
        let image = Image::new(
            self.ctx.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [self.size.0 as u32, self.size.1 as u32, 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();
        let destination_buffer = Buffer::from_iter(
            self.ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (0..self.size.0 as u32 * self.size.1 as u32 * 4).map(|_| 0u8),
        )
        .expect("failed to create buffer");
        let view = ImageView::new_default(image.clone()).unwrap();
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![view],
                ..Default::default()
            },
        )
        .unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            self.ctx.command_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        unsafe {
            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                .bind_pipeline_graphics(self.pipeline.clone())
                .unwrap()
                .set_viewport(
                    0,
                    [Viewport {
                        offset: [0.0, 0.0],
                        extent: [self.size.0, self.size.1],
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                )
                .unwrap()
                .bind_vertex_buffers(0, self.vertex_buffer.clone())
                .unwrap()
                .draw(
                    6, 1, 0, 0, // 3 is the number of vertices, 1 is the number of instances
                )
                .unwrap()
                .end_render_pass(SubpassEndInfo::default())
                .unwrap()
                .copy_image_to_buffer(
                    vulkano::command_buffer::CopyImageToBufferInfo::image_buffer(
                        image,
                        destination_buffer.clone(),
                    ),
                )
                .unwrap();
        }
        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.ctx.device.clone())
            .then_execute(self.ctx.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
        let buffer_content = destination_buffer.read().unwrap();
        let image_out = ImageBuffer::<Rgba<u8>, _>::from_raw(
            self.size.0 as u32,
            self.size.1 as u32,
            &buffer_content[..],
        )
        .unwrap();
        image_out.save("image.png").unwrap();
    }
}
