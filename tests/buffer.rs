use quadrax::gpu::{
    backend::Backend,
    buffer::{Buffer, BufferRole},
};

#[tokio::test]
async fn buffer_crud() {
    let backend = Backend::new().await;
    let buffer = Buffer::new(
        backend.arc_mutex().clone(),
        vec![0u32; 16],
        BufferRole::Generic,
    )
    .await;
    assert_eq!(buffer.read::<u32>().await, vec![0u32; 16]);
    buffer.write(vec![1u32; 16]).await;
    assert_eq!(buffer.read::<u32>().await, vec![1u32; 16]);
}
