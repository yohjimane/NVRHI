#include <nvrhi/metal3.h>
#include <nvrhi/common/resource.h>
#include <cstdio>
#include <stdexcept>

namespace
{
    void require(bool success, const char* message)
    {
        if (!success)
            throw std::runtime_error(message);
    }

    struct Messages final : nvrhi::IMessageCallback
    {
        unsigned errors = 0;
        void message(nvrhi::MessageSeverity severity, const char* text) override
        {
            std::fprintf(stderr, "%s\n", text);
            if (severity >= nvrhi::MessageSeverity::Error)
                ++errors;
        }
    };

    struct ExternalTracker final : nvrhi::RefCounter<nvrhi::ICommandListLifetimeTracker>
    {
        unsigned collections = 0;
        void runGarbageCollection() override { ++collections; }
        nvrhi::Object getNativeObject(nvrhi::ObjectType) override { return nullptr; }
    };

    struct Fixture
    {
        Messages messages;
        unsigned expectedErrors = 0;
        nvrhi::metal3::DeviceDesc desc;
        nvrhi::metal3::DeviceHandle device;
        nvrhi::BufferHandle buffer;
        uint32_t* words = nullptr;

        Fixture()
        {
            desc.pDevice = MTLCreateSystemDefaultDevice();
            require(desc.pDevice && [desc.pDevice supportsFamily:MTLGPUFamilyMetal3], "Metal 3 device required");
            desc.commonQueue = [desc.pDevice newCommandQueue];
            desc.errorCB = &messages;
            device = nvrhi::metal3::createDevice(desc);
            require(device != nullptr, "Device creation failed");
            nvrhi::BufferDesc bufferDesc;
            bufferDesc.byteSize = 4096;
            bufferDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
            buffer = device->createBuffer(bufferDesc);
            require(buffer != nullptr, "Buffer creation failed");
            id<MTLBuffer> native = (__bridge id<MTLBuffer>)buffer->getNativeObject(nvrhi::ObjectTypes::MTL3_Buffer).pointer;
            words = static_cast<uint32_t*>(native.contents);
            require(words != nullptr, "Shared buffer mapping failed");
            for (unsigned i = 0; i < 1024; ++i)
                words[i] = 0x55555555u;
        }

        ~Fixture()
        {
            if (device)
                device->waitForIdle();
        }

        nvrhi::CommandListHandle list(bool immediate = false, nvrhi::CommandQueue queue = nvrhi::CommandQueue::Graphics)
        {
            nvrhi::CommandListParameters params;
            params.enableImmediateExecution = immediate;
            params.queueType = queue;
            auto commandList = device->createCommandList(params);
            require(commandList != nullptr, "Command list creation failed");
            return commandList;
        }

        void record(nvrhi::ICommandList* commands, uint32_t value, unsigned index = 0)
        {
            commands->open();
            commands->writeBuffer(buffer, &value, sizeof(value), index * sizeof(value));
            commands->close();
        }

        uint64_t execute(nvrhi::ICommandList* commands, nvrhi::CommandQueue queue = nvrhi::CommandQueue::Graphics)
        {
            uint64_t serial = device->executeCommandList(commands, queue);
            require(serial != 0, "Valid submission rejected");
            return serial;
        }

        template<typename F>
        void expectError(F action)
        {
            action();
            ++expectedErrors;
            require(messages.errors == expectedErrors, "Invalid operation did not report exactly one error");
        }

        void checkErrors()
        {
            require(messages.errors == expectedErrors, "Unexpected backend error");
        }
    };

    void deferredOrder()
    {
        Fixture f;
        auto first = f.list();
        auto second = f.list();
        first->open();
        first->clearBufferUInt(f.buffer, 0x11111111u);
        first->close();
        f.record(second, 0x22222222u);
        f.device->waitForIdle();
        require(f.words[0] == 0x55555555u, "Closing a command list executed GPU work");
        nvrhi::ICommandList* reversed[] = {second, first};
        require(f.device->executeCommandLists(reversed, 2) != 0, "Batch submission rejected");
        f.device->waitForIdle();
        for (unsigned i = 0; i < 1024; ++i)
            require(f.words[i] == 0x11111111u, "Requested submission order was not preserved");
        f.checkErrors();
    }

    void invalidBatches()
    {
        Fixture f;
        auto commands = f.list();
        auto unopened = f.list();
        f.record(commands, 0x12345678u);
        nvrhi::ICommandList* duplicate[] = {commands, commands};
        f.expectError([&] { require(f.device->executeCommandLists(duplicate, 2) == 0, "Duplicate batch accepted"); });
        nvrhi::ICommandList* nullEntry[] = {commands, nullptr};
        f.expectError([&] { require(f.device->executeCommandLists(nullEntry, 2) == 0, "Null list accepted"); });
        nvrhi::ICommandList* unready[] = {commands, unopened};
        f.expectError([&] { require(f.device->executeCommandLists(unready, 2) == 0, "Unopened list accepted"); });
        f.expectError([&] { require(f.device->executeCommandLists(nullptr, 1) == 0, "Null array accepted"); });
        f.expectError([&] { require(f.device->executeCommandList(commands, nvrhi::CommandQueue::Count) == 0, "Invalid queue accepted"); });
        require(f.device->executeCommandLists(nullptr, 0) == 0, "Empty batch received an execution identifier");
        f.device->waitForIdle();
        require(f.words[0] == 0x55555555u, "Invalid batch partially submitted GPU work");
        uint64_t firstSerial = f.execute(commands);
        f.device->waitForIdle();
        require(f.words[0] == 0x12345678u, "Rejected batch damaged the valid recording");
        f.expectError([&] { require(f.device->executeCommandList(commands) == 0, "Recording was submitted twice"); });
        commands->open();
        f.expectError([&] { commands->open(); });
        f.expectError([&] { require(f.device->executeCommandList(commands) == 0, "Open recording was submitted"); });
        commands->close();
        f.expectError([&] { commands->close(); });
        require(f.execute(commands) > firstSerial, "Execution identifiers did not advance");
        f.checkErrors();
    }

    void queuesAndOwnership()
    {
        Fixture f;
        for (auto queue : {nvrhi::CommandQueue::Compute, nvrhi::CommandQueue::Copy})
        {
            auto commands = f.list(false, queue);
            f.record(commands, 0x76543210u);
            f.expectError([&] { require(f.device->executeCommandList(commands) == 0, "Mismatched logical queue accepted"); });
            f.execute(commands, queue);
            f.device->waitForIdle();
            require(f.words[0] == 0x76543210u, "Logical queue submission lost its write");
        }
        auto commands = f.list();
        f.record(commands, 0xabcdef01u);
        auto otherDesc = f.desc;
        otherDesc.commonQueue = [f.desc.pDevice newCommandQueue];
        auto other = nvrhi::metal3::createDevice(otherDesc);
        f.expectError([&] { require(other->executeCommandList(commands) == 0, "Foreign-device recording accepted"); });
        f.execute(commands);
        f.device->waitForIdle();
        require(f.words[0] == 0xabcdef01u, "Foreign submission attempt damaged original recording");
        nvrhi::CommandListParameters invalid;
        invalid.queueType = nvrhi::CommandQueue::Count;
        f.expectError([&] { require(f.device->createCommandList(invalid) == nullptr, "Invalid creation queue accepted"); });
        ExternalTracker tracker;
        invalid.queueType = nvrhi::CommandQueue::Graphics;
        invalid.lifetimeTracker = &tracker;
        f.expectError([&] { require(f.device->createCommandList(invalid) == nullptr, "Unsupported external tracker was silently accepted"); });
        f.checkErrors();
    }

    void immediateRules()
    {
        Fixture f;
        auto first = f.list(true);
        auto second = f.list(true);
        f.expectError([&] { first->close(); });
        first->open();
        f.expectError([&] { second->open(); });
        first->clearBufferUInt(f.buffer, 0x33333333u);
        first->close();
        f.expectError([&] { first->open(); });
        f.device->waitForIdle();
        require(f.words[0] == 0x55555555u, "Immediate flag bypassed deferred submission");
        f.execute(first);
        f.record(second, 0x44444444u);
        f.execute(second);
        f.device->waitForIdle();
        require(f.words[0] == 0x44444444u, "Immediate recording reuse failed");
        first->open();
        first = nullptr;
        second->open();
        second->close();
        f.execute(second);
        f.checkErrors();
    }

    void discardedRecording()
    {
        Fixture f;
        auto commands = f.list();
        for (unsigned i = 0; i < 32; ++i)
            f.record(commands, 0xdeadbeefu);
        f.record(commands, 0xcafebabeu, 1);
        f.execute(commands);
        f.device->waitForIdle();
        require(f.words[0] == 0x55555555u && f.words[1] == 0xcafebabeu, "Abandoned recording was executed or corrupted replacement data");
        f.checkErrors();
    }

    struct Gate
    {
        id<MTLSharedEvent> event;
        explicit Gate(id<MTLDevice> device) : event([device newSharedEvent])
        {
            require(event != nil, "Shared event creation failed");
        }
        ~Gate() { event.signaledValue = 1; }
    };

    void inFlightReuse()
    {
        Fixture f;
        Gate gate(f.desc.pDevice);
        id<MTLCommandBuffer> blocker = [f.desc.commonQueue commandBuffer];
        [blocker encodeWaitForEvent:gate.event value:1];
        [blocker commit];
        auto commands = f.list();
        for (unsigned i = 0; i < 16; ++i)
        {
            f.record(commands, 0x12340000u + i, i);
            f.execute(commands);
        }
        require(f.words[0] == 0x55555555u, "GPU gate did not hold submitted work");
        commands = nullptr;
        gate.event.signaledValue = 1;
        f.device->waitForIdle();
        for (unsigned i = 0; i < 16; ++i)
            require(f.words[i] == 0x12340000u + i, "In-flight recording reuse corrupted upload data");
        f.checkErrors();
    }
}

int main()
{
    @autoreleasepool
    {
        try
        {
            deferredOrder();
            invalidBatches();
            queuesAndOwnership();
            immediateRules();
            discardedRecording();
            inFlightReuse();
            std::puts("PASS: deferred ordering, atomic batch rejection, queue/device ownership, immediate rules, discard, and in-flight reuse");
            return 0;
        }
        catch (const std::exception& error)
        {
            std::fprintf(stderr, "FAIL: %s\n", error.what());
            return 1;
        }
    }
}
