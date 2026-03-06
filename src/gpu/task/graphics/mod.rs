use crate::gpu::task::graphics::{rasteriser::RasteriseTask, raytracer::RaytraceTask};

pub mod rasteriser;
pub mod raytracer;

pub enum GraphicsTask {
    Rasteriser(RasteriseTask),
    Raytracer(RaytraceTask),
}
