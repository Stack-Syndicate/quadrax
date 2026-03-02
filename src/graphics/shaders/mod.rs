pub mod screen_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/graphics/shaders/screen.frag",
    }
}

pub mod screen_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/graphics/shaders/screen.vert"
    }
}
