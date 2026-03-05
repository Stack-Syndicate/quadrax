use crate::gpu::{
    backend::BackendContext,
    memory::{
        BufferID, BufferRegistry, ImageRegistry,
        buffer::{Buffer, BufferRole, Location},
        image::{Image, ImageIntent, TexelSize},
    },
};
use bytemuck::{Pod, Zeroable};
use std::{
    collections::{HashSet, VecDeque},
    sync::Arc,
};
use vulkano::{
    buffer::view,
    command_buffer::{
        AutoCommandBufferBuilder, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    image::SampleCount,
    pipeline::{
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            input_assembly::InputAssemblyState,
            subpass::PipelineSubpassType,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, Framebuffer,
        FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass,
    },
    shader::ShaderModule,
    sync::GpuFuture,
};
use vulkano::{
    image::ImageLayout,
    pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState},
    render_pass::AttachmentReference,
};

type VulkanoImage = vulkano::image::Image;
#[derive(Pod, Zeroable, Vertex, Clone, Copy, Debug)]
#[repr(C)]
struct DefaultVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}
pub enum TaskType {
    Compute,
    Graphics,
}
pub struct Task {
    ctx: BackendContext,
    compute_pipeline: Option<Arc<ComputePipeline>>,
    graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    task_type: TaskType,
    read_buffers: Vec<Arc<Buffer>>,
    write_buffers: Vec<Arc<Buffer>>,
    read_images: Vec<Arc<Image>>,
    write_images: Vec<Arc<Image>>,
}
impl Task {
    pub fn new_graphics<V>(
        ctx: BackendContext,
        vertex_shader: Arc<ShaderModule>,
        fragment_shader: Arc<ShaderModule>,
        read_buffers: Vec<Arc<Buffer>>,
        write_buffers: Vec<Arc<Buffer>>,
        read_images: Vec<Arc<Image>>,
        write_images: Vec<Arc<Image>>,
    ) -> Self
    where
        V: Vertex,
    {
        let vs = vertex_shader.entry_point("main").unwrap();
        let fs = fragment_shader.entry_point("main").unwrap();
        let output_extent = write_images[0].inner.image().extent();
        let viewport = Viewport {
            extent: [output_extent[0] as f32, output_extent[1] as f32],
            ..Default::default()
        };
        let vertex_input_state = V::per_vertex().definition(&vs).unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(
            ctx.device.clone(),
            vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap(),
        )
        .unwrap();
        let render_pass = {
            let attachments = write_images
                .iter()
                .map(|img| AttachmentDescription {
                    format: img.texel().format(),
                    samples: SampleCount::Sample1,
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    final_layout: ImageLayout::General,
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let attachment_references = attachments
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    Some(AttachmentReference {
                        attachment: i as u32,
                        layout: ImageLayout::General,
                        ..Default::default()
                    })
                })
                .collect();
            let subpasses = vec![vulkano::render_pass::SubpassDescription {
                color_attachments: attachment_references,
                depth_stencil_attachment: None,
                ..Default::default()
            }];
            RenderPass::new(
                ctx.device.clone(),
                RenderPassCreateInfo {
                    attachments,
                    subpasses,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let color_blend_state = Some(ColorBlendState {
            attachments: write_images
                .iter()
                .map(|_| ColorBlendAttachmentState::default())
                .collect::<Vec<_>>(),
            ..Default::default()
        });
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let graphics_pipeline = GraphicsPipeline::new(
            ctx.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into(),
                    ..Default::default()
                }),
                subpass: Some(PipelineSubpassType::BeginRenderPass(subpass)),
                rasterization_state: Some(Default::default()),
                multisample_state: Some(Default::default()),
                color_blend_state,
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap();

        Self {
            ctx,
            compute_pipeline: None,
            graphics_pipeline: Some(graphics_pipeline),
            task_type: TaskType::Graphics,
            read_buffers,
            write_buffers,
            read_images,
            write_images,
        }
    }

    pub fn new_compute(
        ctx: BackendContext,
        shader: Arc<ShaderModule>,
        read_buffers: Vec<Arc<Buffer>>,
        write_buffers: Vec<Arc<Buffer>>,
        read_images: Vec<Arc<Image>>,
        write_images: Vec<Arc<Image>>,
    ) -> Self {
        let cs = shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            ctx.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap(),
        )
        .unwrap();
        let compute_pipeline = Some(
            ComputePipeline::new(
                ctx.device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap(),
        );
        let graphics_pipeline = None;

        Self {
            ctx,
            compute_pipeline,
            graphics_pipeline,
            task_type: TaskType::Compute,
            read_buffers,
            write_buffers,
            read_images,
            write_images,
        }
    }
    pub fn execute(&self, ctx: BackendContext, group_counts: [u32; 3]) -> Box<dyn GpuFuture> {
        self.execute_typed::<DefaultVertex>(ctx, group_counts)
    }
    pub fn execute_typed<V: Vertex>(
        &self,
        ctx: BackendContext,
        group_counts: [u32; 3],
    ) -> Box<dyn GpuFuture> {
        let read_buffer_descriptor_sets = self
            .read_buffers
            .iter()
            .enumerate()
            .map(|(i, b)| WriteDescriptorSet::buffer(i as u32, b.inner_bytes()))
            .collect::<Vec<_>>();
        let write_buffer_descriptor_sets = self
            .write_buffers
            .iter()
            .enumerate()
            .map(|(i, b)| {
                WriteDescriptorSet::buffer(
                    read_buffer_descriptor_sets.len() as u32 + i as u32,
                    b.inner_bytes(),
                )
            })
            .collect::<Vec<_>>();
        let read_image_descriptor_sets = self
            .read_images
            .iter()
            .enumerate()
            .map(|(i, b)| {
                WriteDescriptorSet::image_view(
                    read_buffer_descriptor_sets.len() as u32
                        + write_buffer_descriptor_sets.len() as u32
                        + i as u32,
                    b.inner.clone(),
                )
            })
            .collect::<Vec<_>>();
        let write_image_descriptor_sets = self
            .write_images
            .iter()
            .enumerate()
            .map(|(i, b)| {
                WriteDescriptorSet::image_view(
                    read_buffer_descriptor_sets.len() as u32
                        + write_buffer_descriptor_sets.len() as u32
                        + read_image_descriptor_sets.len() as u32
                        + i as u32,
                    b.inner.clone(),
                )
            })
            .collect::<Vec<_>>();
        let mut descriptor_writes = Vec::new();
        descriptor_writes.extend(read_buffer_descriptor_sets);
        descriptor_writes.extend(write_buffer_descriptor_sets);
        descriptor_writes.extend(read_image_descriptor_sets);
        descriptor_writes.extend(write_image_descriptor_sets);
        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_allocator.clone(),
            ctx.queue.queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        match self.task_type {
            TaskType::Compute => unsafe {
                let descriptor_set = DescriptorSet::new(
                    ctx.descriptor_allocator,
                    self.compute_pipeline
                        .clone()
                        .unwrap()
                        .layout()
                        .set_layouts()[0]
                        .clone(),
                    descriptor_writes,
                    None,
                )
                .unwrap();
                builder
                    .bind_pipeline_compute(self.compute_pipeline.clone().unwrap())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        self.compute_pipeline.clone().unwrap().layout().clone(),
                        0 as u32,
                        descriptor_set,
                    )
                    .unwrap()
                    .dispatch(group_counts)
                    .unwrap();
            },
            TaskType::Graphics => {
                let pipeline = self.graphics_pipeline.clone().unwrap();
                let subpass = match pipeline.subpass() {
                    PipelineSubpassType::BeginRenderPass(subpass) => subpass.clone(),
                    _ => todo!(),
                };
                let render_pass = subpass.render_pass().clone();
                let framebuffer = Framebuffer::new(
                    render_pass,
                    FramebufferCreateInfo {
                        attachments: self.write_images.iter().map(|f| f.inner.clone()).collect(),
                        ..Default::default()
                    },
                )
                .unwrap();
                unsafe {
                    builder
                        .begin_render_pass(
                            RenderPassBeginInfo {
                                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                                ..RenderPassBeginInfo::framebuffer(framebuffer)
                            },
                            SubpassBeginInfo {
                                contents: SubpassContents::Inline,
                                ..Default::default()
                            },
                        )
                        .unwrap()
                        .bind_pipeline_graphics(self.graphics_pipeline.clone().unwrap())
                        .unwrap()
                        .bind_vertex_buffers(
                            0,
                            self.read_buffers[0].inner_bytes().reinterpret::<[V]>(),
                        )
                        .unwrap()
                        .draw(self.read_buffers[0].length, 1, 0, 0)
                        .unwrap()
                        .end_render_pass(SubpassEndInfo::default())
                        .unwrap();
                }
            }
        }
        let command_buffer = builder.build().unwrap();
        Box::new(
            vulkano::sync::now(ctx.device.clone())
                .then_execute(ctx.queue.clone(), command_buffer)
                .unwrap()
                .boxed(),
        )
    }
}

pub struct TaskManager {
    ctx: BackendContext,
    tasks: VecDeque<Task>,
    buffer_registry: BufferRegistry,
    image_registry: ImageRegistry,
    used_buffer_ids: HashSet<BufferID>,
    used_image_ids: HashSet<BufferID>,
}
impl TaskManager {
    pub fn new(ctx: BackendContext) -> Self {
        Self {
            ctx,
            tasks: VecDeque::new(),
            buffer_registry: BufferRegistry::new(),
            image_registry: ImageRegistry::new(),
            used_buffer_ids: HashSet::new(),
            used_image_ids: HashSet::new(),
        }
    }
    pub fn add_compute_task(
        &mut self,
        compute_shader: Arc<ShaderModule>,
        read_buffers: Vec<Arc<Buffer>>,
        read_images: Vec<Arc<Image>>,
        write_buffers: Vec<Arc<Buffer>>,
        write_images: Vec<Arc<Image>>,
    ) {
        self.tasks.push_front(Task::new_compute(
            self.ctx.clone(),
            compute_shader,
            read_buffers,
            write_buffers,
            read_images,
            write_images,
        ));
    }
    fn add_graphics_task<V: Vertex>(
        &mut self,
        vertex_shader: Arc<ShaderModule>,
        fragment_shader: Arc<ShaderModule>,
        read_buffers: Vec<Arc<Buffer>>,
        read_images: Vec<Arc<Image>>,
        write_buffers: Vec<Arc<Buffer>>,
        write_images: Vec<Arc<Image>>,
    ) {
        self.tasks.push_front(Task::new_graphics::<V>(
            self.ctx.clone(),
            vertex_shader,
            fragment_shader,
            read_buffers,
            write_buffers,
            read_images,
            write_images,
        ));
    }
}

#[test]
fn compute_task() {
    let ctx = BackendContext::new();
    let mut tm = TaskManager::new(ctx.clone());
    mod shader {
        vulkano_shaders::shader! {
            ty: "compute",
            src: r#"
                #version 450

                layout(local_size_x = 128) in;

                layout(set = 0, binding = 0) buffer Data {
                    uint data[];
                } buf;

                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    buf.data[idx] = idx;
                }
            "#
        }
    }

    let shader = shader::load(ctx.device.clone()).unwrap();
    let buffer = Buffer::new_with_role(
        ctx.clone(),
        vec![0u32; 128],
        Location::Host,
        BufferRole::Storage,
    );

    tm.add_compute_task(
        shader,
        vec![],
        vec![],
        vec![Arc::new(buffer.clone())],
        vec![],
    );

    let task = tm.tasks.pop_front().unwrap();
    let future = task.execute(ctx.clone(), [1, 1, 1]);
    future
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
    println!("{:?}", buffer.read::<u32>());
}

#[test]
fn graphics_task() {
    let ctx = BackendContext::new();
    let mut tm = TaskManager::new(ctx.clone());
    mod vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r#"
            #version 460

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        "#
        }
    }
    mod frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r#"
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 1.0, 0.0, 1.0);
            }
        "#
        }
    }

    let vertices = vec![
        DefaultVertex {
            position: [-0.66, -0.66],
        },
        DefaultVertex {
            position: [0.66, -0.66],
        },
        DefaultVertex {
            position: [0.0, 0.66],
        },
    ];
    let fs = frag::load(ctx.device.clone()).unwrap();
    let vs = vert::load(ctx.device.clone()).unwrap();
    let vertex_buffer =
        Buffer::new_with_role(ctx.clone(), vertices, Location::Host, BufferRole::Vertex);

    let frag_color_image = Image::new(
        ctx.clone(),
        TexelSize::RGBA8,
        ImageIntent::RenderTarget,
        [1024, 1024, 1],
    );
    tm.add_graphics_task::<DefaultVertex>(
        vs,
        fs,
        vec![Arc::new(vertex_buffer)],
        vec![],
        vec![],
        vec![Arc::new(frag_color_image.clone())],
    );
    let t = tm
        .tasks
        .iter()
        .map(|t| t.execute(ctx.clone(), [1024, 1024, 1]))
        .collect::<Vec<_>>();
    for t in t {
        t.then_signal_fence_and_flush().unwrap().wait(None).unwrap();
    }
    frag_color_image.save("image.png");
}
