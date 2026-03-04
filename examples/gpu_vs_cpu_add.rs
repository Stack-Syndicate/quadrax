use quadrax::backend::buffer::{Buffer, BufferTyped};
use quadrax::compute::mathematics::OpCode;
use quadrax::compute::{ComputeContext, Vec4, mathematics::LinearAlgebra};
use std::sync::Arc;
use std::time::Instant;

use quadrax::backend::BackendContext;

fn main() {
    let ctx = BackendContext::new();
    let mut compute = ComputeContext::new(ctx.clone());

    let n = 50_000_000;
    let iters = 100;

    let a_data: Vec<Vec4> = (0..n).map(|i| Vec4::new(i as f32, 1.0, 2.0, 3.0)).collect();
    let b_data: Vec<Vec4> = (0..n).map(|i| Vec4::new(1.0, i as f32, 3.0, 4.0)).collect();

    let mut cpu_out = vec![Vec4::new(0.0, 0.0, 0.0, 0.0); n];

    let a = ctx.create_staged_buffer(&a_data);
    let b = ctx.create_staged_buffer(&b_data);
    let c = ctx.create_staged_buffer(&a_data.clone());
    let maths = LinearAlgebra::new(&ctx, a, b, c, OpCode::Add);
    compute.add_pass(Arc::new(maths.clone()));
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
        compute.dispatch();
    }
    let result = maths.out.read::<Vec4>().wait();
    let gpu_time = gpu_start.elapsed();

    println!("CPU time: {:?}", cpu_time);
    println!("GPU time: {:?}", gpu_time);

    assert!(gpu_time < cpu_time, "GPU was not faster");
}
