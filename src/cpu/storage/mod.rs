use bytemuck::{Pod, bytes_of};
use redb::ReadableDatabase;
use std::sync::{mpsc::Sender, oneshot};

pub enum StorageCommand<K: Pod, V: Pod> {
    Insert {
        key: K,
        value: V,
    },
    Remove {
        key: K,
    },
    Get {
        key: K,
        response: oneshot::Sender<Option<V>>,
    },
    Close,
}

pub struct Storage<K: Pod, V: Pod> {
    sender: Sender<StorageCommand<K, V>>,
    thread_handle: std::thread::JoinHandle<()>,
}
impl<K: Pod + Send, V: Pod + Send> Storage<K, V> {
    pub fn new(path: &str) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let path = path.to_string();
        let thread_handle = std::thread::spawn(move || {
            let db = redb::Database::create(&path).expect("Storage database creation failed.");
            let table: redb::TableDefinition<'static, String, &[u8]> =
                redb::TableDefinition::new("data");
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    StorageCommand::Insert { key, value } => {
                        let txn = db.begin_write().unwrap();
                        {
                            let mut t = txn.open_table(table).unwrap();
                            let key_string = hex::encode(bytes_of(&key));
                            t.insert(&key_string, &bytes_of::<V>(&value)).unwrap();
                        }
                        txn.commit().unwrap();
                    }
                    StorageCommand::Get { key, response } => {
                        let txn = db.begin_read().unwrap();
                        let t = txn.open_table(table).unwrap();
                        let key_string = hex::encode(bytes_of(&key));
                        let result = t.get(&key_string).unwrap().map(|b| {
                            let bytes = b.value();
                            let mut aligned = vec![0u8; std::mem::size_of::<V>()];
                            aligned.copy_from_slice(bytes);
                            bytemuck::from_bytes::<V>(&aligned).clone()
                        });
                        let _ = response.send(result);
                    }
                    StorageCommand::Remove { key } => {
                        let txn = db.begin_write().unwrap();
                        {
                            let mut t = txn.open_table(table).unwrap();
                            let key_string = hex::encode(bytes_of(&key));
                            t.remove(&key_string).unwrap();
                        }
                        txn.commit().unwrap();
                    }
                    StorageCommand::Close => break,
                }
            }
        });
        Self {
            sender: tx,
            thread_handle,
        }
    }
    pub async fn insert(&self, key: K, value: V) {
        self.sender
            .send(StorageCommand::Insert { key, value })
            .unwrap();
    }
    pub async fn get(&self, key: K) -> Option<V> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.sender
            .send(StorageCommand::Get {
                key,
                response: resp_tx,
            })
            .expect("Unable to send get message.");
        resp_rx.recv().unwrap()
    }
    pub async fn remove(&self, key: K) {
        self.sender.send(StorageCommand::Remove { key }).unwrap();
    }
    pub async fn close(self) {
        self.sender
            .send(StorageCommand::Close)
            .expect("Close message send failed.");
        self.thread_handle
            .join()
            .expect("Could not join storage thread.");
    }
}
