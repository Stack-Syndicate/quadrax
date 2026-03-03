use std::sync::Arc;
use vulkano::pipeline::Pipeline;
use vulkano::sync::GpuFuture;

use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    pipeline::{
        ComputePipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
};

use crate::backend::buffer::Buffer;
use crate::backend::buffer::staged::StagedBuffer;
use crate::compute::{ComputeParameter, ComputePass};
use crate::{
    backend::BackendContext,
    compute::{ComputeFuture, vector_ops},
};
#[derive(Clone, Debug)]
#[repr(u32)]
pub enum OpCode {
    Add = 0,
    Sub = 1,
    Dot = 2,
    Mul = 3,
    Cross = 4,
    Dist = 5,
}
impl ComputeParameter for OpCode {
    fn to_u32(&self) -> u32 {
        self.clone() as u32
    }
}

pub struct LinearAlgebra {
    pub pipeline: Arc<ComputePipeline>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub in1: StagedBuffer,
    pub in2: StagedBuffer,
    pub out: StagedBuffer,
    pub ctx: BackendContext,
    pub op: OpCode,
}
impl LinearAlgebra {
    pub fn new(
        ctx: &BackendContext,
        in1: StagedBuffer,
        in2: StagedBuffer,
        out: StagedBuffer,
        op: OpCode,
    ) -> Self {
        let shader =
            vector_ops::load(ctx.device.clone()).expect("Could not load vector ops shader.");
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
            in1,
            in2,
            out,
            pipeline,
            descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                ctx.device.clone(),
                Default::default(),
            )),
            ctx: ctx.clone(),
            op: op,
        }
    }
}
impl ComputePass for LinearAlgebra {
    fn dispatch(&self) -> ComputeFuture {
        let ctx = self.backend();
        let buffers = self.buffers();
        let a = buffers[0].clone();
        let b = buffers[1].clone();
        let c = buffers[2].clone();
        let op = self.parameters()[0].clone();
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, a.ptr_bytes()),
                WriteDescriptorSet::buffer(1, b.ptr_bytes()),
                WriteDescriptorSet::buffer(2, c.ptr_bytes()),
            ],
            [],
        )
        .expect("Failed to create descriptor set.");
        let push_constants = vector_ops::PushConstants {
            op_code: op.to_u32(),
            count: a.ptr_bytes().len() as u32,
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
    fn buffers(&self) -> Vec<Arc<dyn Buffer>> {
        vec![
            Arc::new(self.in1.clone()),
            Arc::new(self.in2.clone()),
            Arc::new(self.out.clone()),
        ]
    }
    fn backend(&self) -> BackendContext {
        self.ctx.clone()
    }
    fn parameters(&self) -> Vec<Arc<dyn ComputeParameter>> {
        vec![Arc::new(self.op.clone())]
    }
}
