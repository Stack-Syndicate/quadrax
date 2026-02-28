use vulkano::buffer::BufferContents;

#[repr(C)]
#[derive(BufferContents, Clone, Copy, Debug, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}
