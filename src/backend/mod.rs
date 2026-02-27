pub mod buffer;

use std::sync::Arc;

use vulkano::{
    VulkanLibrary,
    buffer::BufferContents,
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    device::{
        Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags, physical::PhysicalDevice,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
};

use crate::backend::buffer::{Buffer, constant::ConstantBuffer, variable::VariableBuffer};

#[derive(Clone, Debug)]
pub struct Context {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_allocator: Arc<StandardCommandBufferAllocator>,
}
impl Context {
    pub fn new() -> Self {
        let physical_device = Context::create_physical_device();
        let queue_family_index = Context::create_queue_family_index(physical_device.clone());
        let (device, queue) = Context::create_device_queue(physical_device, queue_family_index);
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        Self {
            device,
            queue,
            memory_allocator,
            command_allocator,
        }
    }
    pub fn create_variable_buffer<T: BufferContents + Copy>(
        &self,
        data: &[T],
    ) -> VariableBuffer<T> {
        VariableBuffer::from_data(self.clone(), data)
    }
    pub fn create_constant_buffer<T: BufferContents + Copy>(
        &self,
        data: &[T],
    ) -> ConstantBuffer<T> {
        ConstantBuffer::from_data(self.clone(), data)
    }
    fn create_physical_device() -> Arc<PhysicalDevice> {
        let library = VulkanLibrary::new().expect("No local Vulkan library found.");
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("Failed to create Vulkan instance.");
        let physical_device = instance
            .enumerate_physical_devices()
            .expect("Could not enumerate physical devices.")
            .next()
            .expect("No physical devices available.");
        println!(
            "Physical device name: {:?}",
            physical_device.properties().device_name
        );
        physical_device
    }
    fn create_queue_family_index(physical_device: Arc<PhysicalDevice>) -> u32 {
        for family in physical_device.queue_family_properties() {
            println!(
                "Found a queue family with {:?} queue(s)",
                family.queue_count
            );
        }
        physical_device
            .queue_family_properties()
            .iter()
            .position(|queue_family_properties| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS)
            })
            .expect("Couldn't find a graphical queue family.") as u32
    }
    fn create_device_queue(
        physical_device: Arc<PhysicalDevice>,
        queue_family_index: u32,
    ) -> (Arc<Device>, Arc<Queue>) {
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("Failed to create device.");
        let queue = queues.next().unwrap();
        return (device, queue);
    }
}
