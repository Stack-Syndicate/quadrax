use std::collections::HashMap;

use crate::gpu::memory::{buffer::Buffer, image::Image};

pub mod buffer;
pub mod image;

pub type BufferID = u32;
pub type ImageID = u32;

pub type BufferRegistry = HashMap<BufferID, Buffer>;
pub type ImageRegistry = HashMap<ImageID, Image>;
