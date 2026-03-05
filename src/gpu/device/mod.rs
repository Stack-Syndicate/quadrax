use bytemuck::Pod;
use std::sync::Arc;
use vulkano::{
    VulkanLibrary,
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDevice,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::StandardMemoryAllocator,
};

use crate::gpu::memory::buffer::{Buffer, Location};

#[derive(Clone, Debug)]
pub struct DeviceContext {
    pub library: Arc<VulkanLibrary>,
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_allocator: Arc<StandardDescriptorSetAllocator>,
}
impl DeviceContext {
    pub fn new_headless() -> Self {
        Self::new(InstanceExtensions::default())
    }
    pub fn new(extensions: InstanceExtensions) -> Self {
        let (library, instance, physical_device) =
            DeviceContext::create_physical_device(extensions);
        let queue_family_index = DeviceContext::create_queue_family_index(physical_device.clone());
        let (device, queue) =
            DeviceContext::create_device_queue(physical_device, queue_family_index);
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        Self {
            library,
            instance,
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
    fn create_physical_device(
        enabled_extensions: InstanceExtensions,
    ) -> (Arc<VulkanLibrary>, Arc<Instance>, Arc<PhysicalDevice>) {
        let library = VulkanLibrary::new().expect("No local Vulkan library found.");
        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions,
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
            return (library.clone(), instance.clone(), dev.clone());
        }

        if let Some(dev) = devices.iter().find(|d| {
            d.properties().device_type
                == vulkano::device::physical::PhysicalDeviceType::IntegratedGpu
        }) {
            println!("Using integrated GPU: {}", dev.properties().device_name);
            return (library.clone(), instance.clone(), dev.clone());
        }

        let dev = devices[0].clone();
        println!(
            "Using fallback GPU: {} ({:?})",
            dev.properties().device_name,
            dev.properties().device_type
        );
        (library, instance, dev)
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
                enabled_extensions: DeviceExtensions {
                    khr_swapchain: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("Failed to create device.");
        let queue = queues.next().unwrap();
        return (device, queue);
    }
}
