use std::{
    sync::mpsc::{self, Sender},
    thread::JoinHandle,
};

use hecs::{QueryBorrow, QueryMut};

pub type EntityID = hecs::Entity;
pub trait System = FnMut(&mut World) + Send;
pub trait Query = hecs::Query;

pub struct World {
    hecs_world: hecs::World,
    systems_bundles: Vec<Vec<Box<dyn System>>>,
}
impl World {
    fn new() -> Self {
        Self {
            hecs_world: hecs::World::new(),
            systems_bundles: Vec::new(),
        }
    }
    pub fn spawn(&mut self, components: impl hecs::DynamicBundle) -> EntityID {
        self.hecs_world.spawn(components)
    }
    pub fn despawn(&mut self, entity: EntityID) -> anyhow::Result<()> {
        self.hecs_world.despawn(entity)?;
        Ok(())
    }
    fn add_systems(&mut self, systems: Vec<Box<dyn System>>) {
        self.systems_bundles.push(systems);
    }
    fn run_systems(&mut self) {
        let mut bundles = std::mem::take(&mut self.systems_bundles);
        for bundle in bundles.iter_mut() {
            for system in bundle {
                system(self)
            }
        }
        self.systems_bundles = bundles;
    }
    pub fn query<T: hecs::Query>(&self) -> QueryBorrow<'_, T> {
        self.hecs_world.query::<T>()
    }
    pub fn query_mut<T: hecs::Query>(&mut self) -> QueryMut<'_, T> {
        self.hecs_world.query_mut::<T>()
    }
}

enum Message {
    Tick,
    Create {
        spawn_fn: Box<dyn FnOnce(&mut World) -> EntityID + Send>,
        response: std::sync::mpsc::Sender<EntityID>,
    },
    Delete {
        entity: EntityID,
    },
    Systems {
        systems: Vec<Box<dyn System>>,
    },
}

pub struct TECS {
    thread: JoinHandle<()>,
    tx: Sender<Message>,
}
impl TECS {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            let mut ecs = World::new();
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    Message::Tick => {
                        ecs.run_systems();
                    }
                    Message::Create { spawn_fn, response } => {
                        let e = spawn_fn(&mut ecs);
                        response.send(e).unwrap();
                    }
                    Message::Delete { entity } => {
                        ecs.despawn(entity).expect("Could not despawn entity.")
                    }
                    Message::Systems { systems } => {
                        ecs.add_systems(systems);
                    }
                }
            }
        });
        Self { thread, tx }
    }
    fn send(&self, message: Message) {
        self.tx.send(message).expect("Could not send message.");
    }
    pub fn tick(&self) {
        self.send(Message::Tick);
    }
    pub fn create_entity(&self, components: impl hecs::DynamicBundle + Send + 'static) -> EntityID {
        let (tx, rx) = std::sync::mpsc::channel();
        let boxed_components = Box::new(components);
        self.send(Message::Create {
            spawn_fn: Box::new(|world| {
                return world.spawn(*boxed_components);
            }),
            response: tx,
        });
        rx.recv()
            .expect("Could not receive EntityID object after creation.")
    }
    pub fn remove_entity(&self, entity: EntityID) {
        self.send(Message::Delete { entity });
    }
    pub fn add_systems(&self, systems: Vec<impl System + Sync + 'static>) {
        let boxed_systems = systems
            .into_iter()
            .map(|s| Box::new(s) as Box<dyn System>)
            .collect();
        self.send(Message::Systems {
            systems: boxed_systems,
        });
    }
}

#[test]
fn test() {
    let tecs = TECS::new();
    tecs.tick();
    tecs.create_entity((43u32, "hello!"));
    tecs.add_systems(vec![test_system]);
    tecs.tick();
}

fn test_system(world: &mut World) {
    let mut entities = world.query::<&u32>();
    for e in entities.iter() {
        println!("{:?}", e);
    }
}
