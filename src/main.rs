use quadrax::backend::{Context, buffer::Intent};

fn main() {
    let ctx = Context::new();
    let initial_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut buffer = ctx.create_buffer_from_data(&initial_data, Intent::Static);
    println!("Old data: {:?}", buffer.read());
    buffer.update(&vec![5.0, 2.3, 17.6]);
    println!("New data: {:?}", buffer.read());
}
