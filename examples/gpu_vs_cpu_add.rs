use quadrax::mathematics::OpCode;
use std::time::Instant;

use quadrax::{
    backend::Context,
    mathematics::{GPULA, Vec4},
};

fn main() {
    let ctx = Context::new();
    let la = GPULA::new(&ctx);

    let n = 10_000_000; // this must be large
    let iters = 50;

    let a_data: Vec<Vec4> = (0..n).map(|i| Vec4::new(i as f32, 1.0, 2.0, 3.0)).collect();

    let b_data: Vec<Vec4> = (0..n).map(|i| Vec4::new(1.0, i as f32, 3.0, 4.0)).collect();

    let mut cpu_out = vec![Vec4::new(0.0, 0.0, 0.0, 0.0); n];

    let a = ctx.create_staged_buffer(&a_data);
    let b = ctx.create_staged_buffer(&b_data);
    let c = ctx.create_staged_buffer(&a_data.clone());
    let cpu_start = Instant::now();
    for _ in 0..iters {
        for i in 0..n {
            let av = &a_data[i];
            let bv = &b_data[i];
            cpu_out[i] = Vec4::new(av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w);
        }
    }
    let cpu_time = cpu_start.elapsed();

    let gpu_start = Instant::now();
    for _ in 0..iters {
        let fut = la.dispatch(&ctx, OpCode::Add, &a, &b, &c);
        fut.wait();
    }
    let gpu_time = gpu_start.elapsed();

    println!("CPU time: {:?}", cpu_time);
    println!("GPU time: {:?}", gpu_time);

    assert!(gpu_time < cpu_time, "GPU was not faster");
}
