use quadrax::{backend::Context, graphics::Screen};

fn main() {
    let gfx = Screen::new(Context::new(), (800f32, 600f32));
    gfx.draw();
}
