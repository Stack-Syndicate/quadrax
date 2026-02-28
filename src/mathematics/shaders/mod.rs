pub mod vector_ops {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/mathematics/shaders/vector_ops.comp",
    }
}
