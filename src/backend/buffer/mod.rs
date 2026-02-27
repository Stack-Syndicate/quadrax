pub mod constant;
pub mod variable;

use vulkano::buffer::BufferContents;

use crate::backend::Context;

pub trait Buffer<T: BufferContents + Copy> {
    fn from_data(ctx: Context, data: &[T]) -> Self;
    fn read(&self) -> Vec<T>;
    fn update(&mut self, data: &[T]);
}
