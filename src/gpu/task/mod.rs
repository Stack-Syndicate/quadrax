use petgraph::graph::DiGraph;

use crate::gpu::task::{compute::ComputeTask, graphics::GraphicsTask};

pub mod compute;
pub mod graphics;

pub struct TaskManager {
    compute_graph: DiGraph<ComputeTask, ()>,
    graphics_graph: DiGraph<GraphicsTask, ()>,
}
impl TaskManager {
    pub fn new() -> Self {
        Self {
            compute_graph: DiGraph::new(),
            graphics_graph: DiGraph::new(),
        }
    }
}
