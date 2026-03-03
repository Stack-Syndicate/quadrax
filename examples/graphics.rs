use quadrax::{backend::BackendContext, graphics::Screen};

fn main() {
    let gfx = Screen::new(BackendContext::new(), (800f32, 600f32));
    gfx.draw();
}
