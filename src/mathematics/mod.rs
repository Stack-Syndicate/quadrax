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

use crate::backend::{Context, buffer::staged::StagedBuffer};

pub mod matrix;
pub mod shaders;

#[repr(u32)]
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
    ) -> ComputeFuture {
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
            count: a.inner.len() as u32,
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
        ComputeFuture {
            inner: Some(Box::new(
                vulkano::sync::now(ctx.device.clone())
                    .then_execute(ctx.queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap(),
            )),
        }
    }
}

#[cfg(test)]
mod mathematics_tests {
    use super::*;
    use crate::backend::Context;

    #[test]
    fn gpu_vs_cpu_vector_add_benchmark() {
        use std::time::Instant;
    }
}
