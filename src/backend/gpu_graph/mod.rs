use std::sync::{Arc, Mutex};

use petgraph::{
    algo::toposort,
    graph::{DiGraph, NodeIndex},
};

use crate::backend::buffer::BufferRegistry;

pub trait Pass {
    fn clone_box(&self) -> Box<dyn Pass>;
    fn io(&self) -> PassIO;
}
impl Clone for Box<dyn Pass> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Clone)]
pub struct ComputePass {}
impl Pass for ComputePass {
    fn clone_box(&self) -> Box<dyn Pass> {
        Box::new(self.clone())
    }
    fn io(&self) -> PassIO {
        todo!()
    }
}

#[derive(Clone)]
pub struct GraphicsPass {}
impl Pass for GraphicsPass {
    fn clone_box(&self) -> Box<dyn Pass> {
        todo!()
    }
    fn io(&self) -> PassIO {
        todo!()
    }
}

pub struct PassIO {
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
}
impl PassIO {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PassTransfer {}

pub struct PassGraph {
    inner: Arc<Mutex<DiGraph<Box<dyn Pass>, bool>>>,
    registry: BufferRegistry,
}
impl PassGraph {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(DiGraph::new())),
            registry: BufferRegistry::new(),
        }
    }
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) -> NodeIndex {
        let mut inner_lock = self.inner.lock().unwrap();
        inner_lock.add_node(pass)
    }
    pub fn add_pass_merge(&mut self, parents: Vec<NodeIndex>, pass: Box<dyn Pass>) -> NodeIndex {
        let mut inner_lock = self.inner.lock().unwrap();
        let new_node = inner_lock.add_node(pass.clone_box());
        for p in parents {
            inner_lock.add_edge(p, new_node, true);
        }
        new_node
    }
    pub fn toposort(&self) -> Vec<NodeIndex> {
        let inner_lock = self.inner.lock().unwrap();
        let sorted_graph = toposort(&inner_lock.clone(), None).unwrap();
        sorted_graph
    }
    pub fn execute(&self) {}
}
