use quadrax::cpu::storage::Storage;
use tempfile::TempDir;

#[tokio::test]
async fn storage_worker() {
    let tmp_dir = TempDir::new().unwrap();
    let db_path = tmp_dir.path().join("test.redb");
    let db_path_str = db_path.to_str().unwrap();

    let storage: Storage<u32, u64> = Storage::new(db_path_str);

    storage.insert(1, 100).await;
    storage.insert(2, 200).await;

    let v1 = storage.get(1).await;
    let v2 = storage.get(2).await;

    assert_eq!(v1, Some(100));
    assert_eq!(v2, Some(200));

    storage.remove(1).await;
    let v1_after = storage.get(1).await;
    assert_eq!(v1_after, None);

    let v3 = storage.get(42).await;
    assert_eq!(v3, None);

    storage.close().await;
}
