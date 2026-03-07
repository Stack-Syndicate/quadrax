use std::path::PathBuf;
use std::sync::Arc;

use quadrax::gpu::task::compute::ComputeTask;
use quadrax::gpu::{
    backend::Backend,
    buffer::{Buffer, BufferRole},
};
#[pollster::test]
async fn add_one() {
    let backend = Backend::new().await.arc_mutex();
    let input_data: Vec<u32> = vec![0, 1, 2, 3, 4];
    let input_buffer =
        Arc::new(Buffer::new(backend.clone(), input_data.clone(), BufferRole::Storage).await);
    let output_buffer = Arc::new(
        Buffer::new_empty::<u32>(
            backend.clone(),
            (input_data.len() * 4) as u64,
            BufferRole::Storage,
        )
        .await,
    );
    let shader_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/shaders/compute.wgsl");
    let task = ComputeTask::new(
        backend.clone(),
        shader_path.to_str().unwrap(),
        vec![input_buffer.clone()],
        vec![output_buffer.clone()],
        (input_data.len() as u32, 1, 1),
    )
    .await;
    task.execute().await;
    let result = output_buffer.read::<u32>().await;
    let expected: Vec<u32> = input_data.iter().map(|v| v + 1).collect();
    assert_eq!(result, expected);
}
