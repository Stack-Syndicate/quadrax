use bytemuck::Pod;
use std::sync::Arc;
use vulkano::{
    VulkanLibrary,
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags, physical::PhysicalDevice,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
};

use crate::gpu::memory::buffer::{Buffer, Location};

#[derive(Clone, Debug)]
pub struct BackendContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_allocator: Arc<StandardDescriptorSetAllocator>,
}
impl BackendContext {
    pub fn new() -> Self {
        let physical_device = BackendContext::create_physical_device();
        let queue_family_index = BackendContext::create_queue_family_index(physical_device.clone());
        let (device, queue) =
            BackendContext::create_device_queue(physical_device, queue_family_index);
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        Self {
            device: device.clone(),
            queue,
            memory_allocator,
            command_allocator,
            descriptor_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                device.clone(),
                Default::default(),
            )),
        }
    }
    pub fn create_buffer<T: Pod + Send + Sync>(&self, data: Vec<T>, location: Location) -> Buffer {
        Buffer::new(self.clone(), data, location)
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
        let devices: Vec<_> = instance
            .enumerate_physical_devices()
            .expect("Could not enumerate physical devices.")
            .collect();
        if devices.is_empty() {
            panic!("No physical devices available.");
        }
        if let Some(dev) = devices.iter().find(|d| {
            d.properties().device_type == vulkano::device::physical::PhysicalDeviceType::DiscreteGpu
        }) {
            println!("Using discrete GPU: {}", dev.properties().device_name);
            return dev.clone();
        }

        if let Some(dev) = devices.iter().find(|d| {
            d.properties().device_type
                == vulkano::device::physical::PhysicalDeviceType::IntegratedGpu
        }) {
            println!("Using integrated GPU: {}", dev.properties().device_name);
            return dev.clone();
        }

        let dev = devices[0].clone();
        println!(
            "Using fallback GPU: {} ({:?})",
            dev.properties().device_name,
            dev.properties().device_type
        );
        dev
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
