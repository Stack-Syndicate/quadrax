use std::{collections::VecDeque, sync::Arc};

use vulkano::{pipeline::graphics::vertex_input::Vertex, shader::ShaderModule};

use crate::gpu::{
    device::DeviceContext,
    memory::{buffer::Buffer, image::Image},
    task::Task,
};

pub struct TaskManager {
    ctx: DeviceContext,
    pub tasks: VecDeque<Task>,
}
impl TaskManager {
    pub fn new(ctx: DeviceContext) -> Self {
        Self {
            ctx,
            tasks: VecDeque::new(),
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
    pub fn add_graphics_task<V: Vertex>(
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
