use petgraph::graph::{DiGraph, NodeIndex};

use crate::gpu::task::{compute::ComputeTask, graphics::GraphicsTask};

pub mod compute;
pub mod graphics;

pub enum TaskID {
    Compute(NodeIndex),
    Graphics(NodeIndex),
}
impl Into<NodeIndex> for TaskID {
    fn into(self) -> NodeIndex {
        match self {
            TaskID::Compute(id) => id,
            TaskID::Graphics(id) => id,
        }
    }
}

pub enum Task {
    Compute(ComputeTask),
    Graphics(GraphicsTask),
}

pub struct TaskManager {
    task_graph: DiGraph<Task, ()>,
}
impl TaskManager {
    pub fn new() -> Self {
        Self {
            task_graph: DiGraph::new(),
        }
    }
    pub fn add_after(&mut self, parent_id: TaskID, new_child: Task) {
        let child_id = self.task_graph.add_node(new_child);
        self.task_graph.add_edge(parent_id.into(), child_id, ());
    }
}
