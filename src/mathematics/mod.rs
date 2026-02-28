use std::sync::Arc;

use vulkano::{
    buffer::BufferContents,
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    sync::GpuFuture,
};

use crate::backend::{
    Context,
    buffer::{Buffer, staged::StagedBuffer},
};

pub mod matrix;
pub mod shaders;
pub mod vector;

pub enum OpCode {
    Add = 0,
    Sub = 1,
    Dot = 2,
    Mul = 3,
    Cross = 4,
    Distance = 5,
}

#[repr(C)]
#[derive(BufferContents, Clone, Copy, Debug, PartialEq)]
pub struct Vec4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}
impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
}

pub struct GPULA {
    pub pipeline: Arc<ComputePipeline>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}
impl GPULA {
    pub fn new(ctx: &Context) -> Self {
        let shader = shaders::vector_ops::load(ctx.device.clone())
            .expect("Could not load vector ops shader.");
        let entry_point = shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let pipeline = ComputePipeline::new(
            ctx.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                stage.clone(),
                PipelineLayout::new(
                    ctx.device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                        .into_pipeline_layout_create_info(ctx.device.clone())
                        .unwrap(),
                )
                .unwrap(),
            ),
        )
        .expect("Failed to create compute pipeline");
        Self {
            pipeline,
            descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                ctx.device.clone(),
                Default::default(),
            )),
        }
    }
    pub fn dispatch<T: vulkano::buffer::BufferContents + Copy>(
        &self,
        ctx: &Context,
        op: OpCode,
        a: &StagedBuffer<T>,
        b: &StagedBuffer<T>,
        c: &StagedBuffer<T>,
        count: u32,
    ) {
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, a.inner.clone()),
                WriteDescriptorSet::buffer(1, b.inner.clone()),
                WriteDescriptorSet::buffer(2, c.inner.clone()),
            ],
            [],
        )
        .expect("Failed to create descriptor set.");
        let push_constants = shaders::vector_ops::PushConstants {
            op_code: op as u32,
            count,
        };
        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_allocator.clone(),
            ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
            .unwrap();
        unsafe { builder.dispatch([128, 1, 1]) }.unwrap();
        let command_buffer = builder.build().unwrap();
        vulkano::sync::now(ctx.device.clone())
            .then_execute(ctx.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }
}

#[test]
fn test_gpu_add() {
    let ctx = Context::new();
    let gpula = GPULA::new(&ctx);

    let a_data = vec![Vec4::new(1.0, 1.0, 1.0, 1.0)];
    let b_data = vec![Vec4::new(1.0, 1.0, 1.0, 1.0)];
    let empty = vec![Vec4::new(0.0, 0.0, 0.0, 0.0)];

    let buf_a = ctx.create_staged_buffer(&a_data);
    let buf_b = ctx.create_staged_buffer(&b_data);
    let buf_c = ctx.create_staged_buffer(&empty);

    gpula.dispatch(&ctx, OpCode::Add, &buf_a, &buf_b, &buf_c, 1);

    let result = buf_c.read().wait();
    assert_eq!(result[0], Vec4::new(2.0, 2.0, 2.0, 2.0));
}

#[test]
fn test_gpu_add_bulk() {
    let ctx = Context::new();
    let gpula = GPULA::new(&ctx);
    let count = 1024;
    let a_data: Vec<Vec4> = (0..count)
        .map(|i| Vec4::new(i as f32, 1.0, 2.0, 3.0))
        .collect();
    let b_data: Vec<Vec4> = (0..count)
        .map(|i| Vec4::new(1.0, i as f32, 1.0, 1.0))
        .collect();
    let empty = vec![Vec4::new(0.0, 0.0, 0.0, 0.0); count];
    let buf_a = ctx.create_staged_buffer(&a_data);
    let buf_b = ctx.create_staged_buffer(&b_data);
    let buf_c = ctx.create_staged_buffer(&empty);
    gpula.dispatch(&ctx, OpCode::Add, &buf_a, &buf_b, &buf_c, count as u32);
    let result = buf_c.read().wait();
    assert_eq!(result.len(), count, "Result buffer length mismatch");
    for i in 0..count {
        let expected = Vec4::new(i as f32 + 1.0, 1.0 + i as f32, 3.0, 4.0);
        assert_eq!(result[i], expected, "Mismatch found at index {}", i);
    }
}
