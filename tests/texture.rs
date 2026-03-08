use quadrax::gpu::{
    backend::Backend,
    texture::{Texture, TextureRole},
};
use wgpu::TextureFormat;

#[tokio::test]
async fn texture_crud() {
    use image::{ImageBuffer, Rgba};
    let backend = Backend::new().await;
    let texture = Texture::new_empty(
        backend.arc_mutex(),
        &[64, 1, 1],
        TextureRole::Generic,
        TextureFormat::Rgba8Uint,
    )
    .await;
    assert_eq!(texture.read::<[u8; 4]>().await, [[0u8; 4]; 64 * 1]);
    texture
        .write(&vec![[255u8, 0u8, 100u8, 255u8]; 64 * 1])
        .await;
    let final_read = texture.read::<[u8; 4]>().await;
    assert_eq!(final_read, vec![[255u8, 0u8, 100u8, 255u8]; 64 * 1]);
    let raw_bytes = final_read.into_iter().flatten().collect();
    let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_vec(64, 1, raw_bytes).unwrap();
    img.save("image.png").unwrap();
}
