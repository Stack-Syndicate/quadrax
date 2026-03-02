use std::sync::{Arc, Mutex};

use petgraph::graph::{DiGraph, NodeIndex};

pub trait Pass {
    fn clone_box(&self) -> Box<dyn Pass>;
    fn io(&self) -> PassIO;
}
impl Clone for Box<dyn Pass> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
pub struct PassIO {
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
}
impl PassIO {
    pub fn new() {}
}

#[derive(Clone, Debug)]
pub struct PassTransfer {}

pub struct PassGraphBuilder {
    inner: Arc<Mutex<DiGraph<Box<dyn Pass>, PassTransfer>>>,
}
impl PassGraphBuilder {
    pub fn new() -> Self {
        todo!()
    }
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) -> NodeIndex {
        let mut inner_lock = self.inner.lock().unwrap();
        inner_lock.add_node(pass)
    }
    pub fn add_pass_merge(&mut self, parents: Vec<NodeIndex>, pass: Box<dyn Pass>) -> NodeIndex {
        let mut inner_lock = self.inner.lock().unwrap();
        let new_node = inner_lock.add_node(pass.clone_box());
        let new_node_io = pass.io();

        todo!()
    }
    pub fn build() -> Graph {
        todo!()
    }
}

pub struct Graph {}
impl Graph {
    pub fn new() -> Self {
        todo!()
    }
    pub fn iter() {
        todo!()
    }
}
