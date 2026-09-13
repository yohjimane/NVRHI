#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <nvrhi/vulkan.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

auto& api = VULKAN_HPP_DEFAULT_DISPATCHER;

void check(VkResult result) {
    if (result != VK_SUCCESS) throw std::runtime_error("Vulkan result " + std::to_string(result));
}

struct Messages : nvrhi::IMessageCallback {
    bool errors = false;
    void message(nvrhi::MessageSeverity severity, const char* text) override {
        if (severity == nvrhi::MessageSeverity::Error || severity == nvrhi::MessageSeverity::Fatal) errors = true;
        std::cerr << text << std::endl;
    }
};

struct Context {
    vk::detail::DynamicLoader library;
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    nvrhi::vulkan::DeviceHandle nvrhi;
    Messages messages;
    explicit Context(const std::string& libraryName) : library(libraryName) {
        auto getInstanceProcAddr = library.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        if (!getInstanceProcAddr) throw std::runtime_error("Vulkan loader entry point unavailable");
        api.init(getInstanceProcAddr);
        uint32_t count = 0;
        check(api.vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
        std::vector<VkExtensionProperties> extensions(count);
        check(api.vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()));
        bool portability = false;
        for (const auto& extension : extensions)
            portability |= std::strcmp(extension.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
        const char* instanceExtensions[] = {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo createInstance{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        createInstance.pApplicationInfo = &application;
        createInstance.flags = portability ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0;
        createInstance.enabledExtensionCount = portability ? 1 : 0;
        createInstance.ppEnabledExtensionNames = instanceExtensions;
        check(api.vkCreateInstance(&createInstance, nullptr, &instance));
        api.init(vk::Instance(instance));
        check(api.vkEnumeratePhysicalDevices(instance, &count, nullptr));
        if (!count) throw std::runtime_error("No Vulkan device");
        std::vector<VkPhysicalDevice> devices(count);
        check(api.vkEnumeratePhysicalDevices(instance, &count, devices.data()));
        VkPhysicalDevice physical = devices[0];
        api.vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        api.vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
        uint32_t family = 0;
        while ((families.at(family).queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) != (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ++family;
        check(api.vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr));
        extensions.resize(count);
        check(api.vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data()));
        bool subset = false;
        for (const auto& extension : extensions)
            subset |= std::strcmp(extension.extensionName, "VK_KHR_portability_subset") == 0;
        const char* deviceExtensions[] = {"VK_KHR_portability_subset"};
        float priority = 1;
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;
        VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        features13.synchronization2 = VK_TRUE;
        features13.dynamicRendering = VK_TRUE;
        VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        features12.pNext = &features13;
        features12.timelineSemaphore = VK_TRUE;
        VkDeviceCreateInfo createDevice{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        createDevice.pNext = &features12;
        createDevice.queueCreateInfoCount = 1;
        createDevice.pQueueCreateInfos = &queueInfo;
        createDevice.enabledExtensionCount = subset ? 1 : 0;
        createDevice.ppEnabledExtensionNames = deviceExtensions;
        check(api.vkCreateDevice(physical, &createDevice, nullptr, &device));
        api.init(vk::Device(device));
        VkQueue queue;
        api.vkGetDeviceQueue(device, family, 0, &queue);
        nvrhi::vulkan::DeviceDesc descriptor{};
        descriptor.errorCB = &messages;
        descriptor.instance = instance;
        descriptor.physicalDevice = physical;
        descriptor.device = device;
        descriptor.graphicsQueue = queue;
        descriptor.graphicsQueueIndex = int(family);
        descriptor.computeQueue = queue;
        descriptor.computeQueueIndex = int(family);
        descriptor.deviceExtensions = deviceExtensions;
        descriptor.numDeviceExtensions = subset ? 1 : 0;
        descriptor.vulkanLibraryName = libraryName;
        nvrhi = nvrhi::vulkan::createDevice(descriptor);
        if (!nvrhi) throw std::runtime_error("NVRHI device creation failed");
    }
    ~Context() {
        if (nvrhi) {
            nvrhi->waitForIdle();
            nvrhi->runGarbageCollection();
            nvrhi = nullptr;
        }
        if (device) api.vkDestroyDevice(device, nullptr);
        if (instance) api.vkDestroyInstance(instance, nullptr);
    }
};

int main(int argc, char** argv) {
    try {
        Context context(argc > 1 ? argv[1] : "");
        auto device = context.nvrhi;
        bool stale = false;
        for (auto queue : {nvrhi::CommandQueue::Graphics, nvrhi::CommandQueue::Compute}) {
            nvrhi::CommandListParameters parameters;
            parameters.queueType = queue;
            auto first = device->createCommandList(parameters);
            auto second = device->createCommandList(parameters);
            auto query = device->createTimerQuery();
            first->open();
            first->beginTimerQuery(query);
            first->endTimerQuery(query);
            first->close();
            device->executeCommandList(first, queue);
            device->waitForIdle();
            device->runGarbageCollection();
            if (!device->pollTimerQuery(query)) throw std::runtime_error("Completed initial query not ready");
            float previous = device->getTimerQueryTime(query);
            device->resetTimerQuery(query);
            second->open();
            second->beginTimerQuery(query);
            second->endTimerQuery(query);
            second->close();
            bool readyBeforeSubmit = device->pollTimerQuery(query);
            std::cout << "queue=" << unsigned(queue) << " previous_us=" << previous * 1e6f
                << " reused_ready_before_submit=" << readyBeforeSubmit << std::endl;
            if (readyBeforeSubmit) {
                std::cout << "stale_us=" << device->getTimerQueryTime(query) * 1e6f << std::endl;
                stale = true;
            }
            device->executeCommandList(second, queue);
            device->waitForIdle();
            device->runGarbageCollection();
            if (!readyBeforeSubmit) {
                if (!device->pollTimerQuery(query)) throw std::runtime_error("New completed query not ready");
                std::cout << "new_generation_us=" << device->getTimerQueryTime(query) * 1e6f << std::endl;
            }
        }
        if (context.messages.errors) throw std::runtime_error("NVRHI reported an error");
        if (stale) {
            std::cerr << "FAIL: reused timer returned an old result before its command list was submitted" << std::endl;
            return 42;
        }
        std::cout << "PASS: reused timers remain pending until their own submission completes" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
