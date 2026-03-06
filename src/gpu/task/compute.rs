use std::{fs, sync::Arc};

use futures::{StreamExt, lock::Mutex, stream};
use wgpu::CommandEncoderDescriptor;

use crate::gpu::{
    backend::Backend,
    buffer::{Buffer, BufferRole},
};

pub struct ComputeTask {
    backend: Arc<Mutex<Backend>>,
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group: Arc<wgpu::BindGroup>,
    dispatches: (u32, u32, u32),
    input_buffers: Vec<Arc<Buffer>>,
    output_buffers: Vec<Arc<Buffer>>,
}
impl ComputeTask {
    pub async fn new(
        backend: Arc<Mutex<Backend>>,
        path: &str,
        input_buffers: Vec<Arc<Buffer>>,
        output_buffers: Vec<Arc<Buffer>>,
        dispatches: (u32, u32, u32),
    ) -> Self {
        let backend_lock = backend.lock().await;
        let shader_code = fs::read_to_string(path).expect("Could not find/read shader path.");
        let shader = backend_lock
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });
        let pipeline =
            backend_lock
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: Default::default(),
                });
        let mut buffers = Vec::new();
        buffers.extend(input_buffers.clone());
        buffers.extend(output_buffers.clone());
        let bind_group_entries = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.inner.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind_group = backend_lock
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &bind_group_entries,
            });
        Self {
            backend: backend.clone(),
            pipeline: pipeline.into(),
            bind_group: bind_group.into(),
            dispatches,
            input_buffers,
            output_buffers,
        }
    }
    pub async fn execute(&self) {
        let temp_buffers = stream::iter(&self.output_buffers)
            .enumerate()
            .filter_map(|(i, b)| async move {
                match b.role {
                    BufferRole::Uniform | BufferRole::Vertex => None,
                    _ => Some((
                        i,
                        Buffer::new_empty::<u8>(
                            self.backend.clone(),
                            b.size,
                            BufferRole::StagingRead,
                        )
                        .await,
                    )),
                }
            })
            .collect::<Vec<_>>()
            .await;
        let (mut encoder, queue) = {
            let backend_lock = self.backend.lock().await;
            (
                backend_lock
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor::default()),
                backend_lock.queue.clone(),
            )
        };
        for (index, temp_buffer) in temp_buffers {
            let output_buffer = self.output_buffers.get(index).unwrap();
            encoder.copy_buffer_to_buffer(
                &output_buffer.inner,
                0,
                &temp_buffer.inner,
                0,
                output_buffer.size,
            );
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                ..Default::default()
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &*self.bind_group.clone(), &[]);
            cpass.dispatch_workgroups(self.dispatches.0, self.dispatches.1, self.dispatches.2);
        };
        queue.submit([encoder.finish()]);
    }
}
