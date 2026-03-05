use std::sync::{Arc, Mutex};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyImageInfo};
use vulkano::image::{ImageLayout, ImageUsage};
use vulkano::{swapchain::acquire_next_image, sync::GpuFuture};

use vulkano::swapchain::{
    FullScreenExclusive, PresentMode, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo,
    SwapchainPresentInfo,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{self, WindowAttributes},
};

use crate::gpu::{
    device::DeviceContext,
    memory::{
        buffer::{Buffer, BufferRole, Location},
        image::{Image, ImageIntent, TexelSize},
    },
    task::{DefaultVertex, manager::TaskManager},
};

type WinitWindow = winit::window::Window;
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
                f_color = vec4(0.0, 0.0, 0.0, 1.0);
            }
        "#
    }
}

pub struct App {
    backend: Option<DeviceContext>,
    window: Option<Arc<WinitWindow>>,
    surface: Option<Arc<Surface>>,
    gpu_task_manager: Option<Arc<Mutex<TaskManager>>>,
}
impl App {
    pub fn new() -> Self {
        let app = Self {
            backend: None,
            window: None,
            surface: None,
            gpu_task_manager: None,
        };
        app
    }
    pub fn run(&mut self) {
        let event_loop = EventLoop::new().unwrap();
        event_loop.run_app(self).unwrap();
    }
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(WindowAttributes::default())
                    .unwrap(),
            );
            self.window = Some(window.clone());
            let extensions = Surface::required_extensions(&event_loop).unwrap();
            let backend = DeviceContext::new(extensions);
            self.backend = Some(backend.clone());
            let surface = Surface::from_window(
                self.backend.as_ref().unwrap().clone().instance.clone(),
                window.clone(),
            )
            .unwrap();
            self.surface = Some(surface.clone());
            let capabilities = backend.device.physical_devices()[0]
                .surface_capabilities(
                    &surface.clone(),
                    SurfaceInfo {
                        ..Default::default()
                    },
                )
                .unwrap();
            let formats = backend.device.physical_devices()[0]
                .surface_formats(
                    &surface.clone(),
                    SurfaceInfo {
                        ..Default::default()
                    },
                )
                .unwrap();
            let (image_format, color_space) = formats[0];
            let image_extent = capabilities
                .current_extent
                .unwrap_or_else(|| window.inner_size().into());
            let min_images = capabilities.min_image_count;
            let mut tm = Arc::new(Mutex::new(TaskManager::new(self.backend.clone().unwrap())));
            self.gpu_task_manager = Some(tm.clone());
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
            let fs = frag::load(backend.device.clone()).unwrap();
            let vs = vert::load(backend.device.clone()).unwrap();
            let vertex_buffer = Buffer::new_with_role(
                backend.clone(),
                vertices,
                Location::Host,
                BufferRole::Vertex,
            );

            let frag_color_image = Image::new(
                backend.clone(),
                TexelSize::RGBA8,
                ImageIntent::RenderTarget,
                [1024, 1024, 1],
            );
            let mut tm_lock = tm.lock().unwrap();
            tm_lock.add_graphics_task::<DefaultVertex>(
                vs,
                fs,
                vec![Arc::new(vertex_buffer)],
                vec![],
                vec![],
                vec![Arc::new(frag_color_image.clone())],
            );
            for task in tm_lock.tasks.iter() {
                let future = task.execute(backend.clone(), [1024, 1024, 1]);
                future
                    .then_signal_fence_and_flush()
                    .unwrap()
                    .wait(None)
                    .unwrap();
            }
            let (swapchain, swapchain_images) = Swapchain::new(
                backend.device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: min_images,
                    image_format,
                    image_extent,
                    image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                    present_mode: PresentMode::Fifo,
                    full_screen_exclusive: FullScreenExclusive::Default,
                    ..Default::default()
                },
            )
            .unwrap();
            let (image_index, suboptimal, acquire_future) =
                acquire_next_image(swapchain.clone(), None).unwrap();
            let swapchain_image = swapchain_images[image_index as usize].clone();
            let mut cb = AutoCommandBufferBuilder::primary(
                backend.command_allocator,
                backend.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            cb.copy_image(CopyImageInfo::images(
                frag_color_image.inner.clone().image().clone(),
                swapchain_image,
            ))
            .unwrap();

            let cb = cb.build().unwrap();
            let future = acquire_future
                .then_execute(backend.queue.clone(), cb)
                .unwrap()
                .then_swapchain_present(
                    backend.queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain, image_index),
                )
                .then_signal_fence_and_flush()
                .unwrap();

            future.wait(None).unwrap();
        }
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        println!("{:?}", event);
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => {}
        }
    }
}
