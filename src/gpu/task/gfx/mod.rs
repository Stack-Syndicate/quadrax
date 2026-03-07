use crate::gpu::task::gfx::{rasteriser::RasteriseTask, raytracer::RaytraceTask};

pub mod immediate;
pub mod rasteriser;
pub mod raytracer;

pub enum GraphicsTask {
    Rasteriser(RasteriseTask),
    Raytracer(RaytraceTask),
    Immediate,
}
