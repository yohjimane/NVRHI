#include <nvrhi/metal3.h>
#include <nvrhi/common/resource.h>
#include <atomic>
#include <thread>
#include <cstdio>
#include <cmath>
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
        std::atomic<unsigned> errors{0};
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

    void eventRearm()
    {
        Fixture f;
        Gate firstGate(f.desc.pDevice);
        Gate secondGate(f.desc.pDevice);
        auto query = f.device->createEventQuery();
        id<MTLCommandBuffer> firstBlocker = [f.desc.commonQueue commandBuffer];
        [firstBlocker encodeWaitForEvent:firstGate.event value:1];
        [firstBlocker commit];
        f.device->setEventQuery(query, nvrhi::CommandQueue::Graphics);
        require(!f.device->pollEventQuery(query), "Event completed before blocked GPU work");
        id<MTLCommandBuffer> boundary = [f.desc.commonQueue commandBuffer];
        [boundary commit];
        id<MTLCommandBuffer> secondBlocker = [f.desc.commonQueue commandBuffer];
        [secondBlocker encodeWaitForEvent:secondGate.event value:1];
        [secondBlocker commit];
        f.device->resetEventQuery(query);
        f.device->setEventQuery(query, nvrhi::CommandQueue::Graphics);
        firstGate.event.signaledValue = 1;
        [boundary waitUntilCompleted];
        require(!f.device->pollEventQuery(query), "An old completion signaled a rearmed event query");
        secondGate.event.signaledValue = 1;
        f.device->waitEventQuery(query);
        require(f.device->pollEventQuery(query), "Completed event did not become observable");
        f.checkErrors();
    }

    void unsupportedCapabilities()
    {
        Fixture f;
        const auto compressed = f.device->queryFormatSupport(nvrhi::Format::BC1_UNORM);
        require((compressed & (nvrhi::FormatSupport::RenderTarget | nvrhi::FormatSupport::ShaderUavStore)) == nvrhi::FormatSupport::None,
            "Compressed format advertises unsupported render/storage usage");
        require(f.device->queryFeatureSupport(nvrhi::Feature::DeferredCommandLists), "Implemented deferred command lists are not advertised");
        const auto integer = f.device->queryFormatSupport(nvrhi::Format::R32_UINT);
        require((integer & (nvrhi::FormatSupport::Blendable | nvrhi::FormatSupport::ShaderSample)) == nvrhi::FormatSupport::None,
            "Integer format advertises blending or filtering");
        nvrhi::TextureDesc texture;
        texture.width = texture.height = 4;
        texture.format = nvrhi::Format::BC1_UNORM;
        texture.isUAV = true;
        f.expectError([&] { require(f.device->createTexture(texture) == nullptr, "Compressed storage texture accepted"); });
        texture.isUAV = false;
        texture.isRenderTarget = true;
        f.expectError([&] { require(f.device->createTexture(texture) == nullptr, "Compressed render target accepted"); });
        f.expectError([&] { require(f.device->createTimerQuery() == nullptr, "Unimplemented timer returned a usable handle"); });
        f.expectError([&] { require(std::isnan(f.device->getTimerQueryTime(nullptr)), "Unsupported timer returned a fabricated duration"); });
        f.expectError([&] { require(f.device->createRayTracingPipeline({}) == nullptr, "Unimplemented ray tracing returned a usable pipeline"); });
        f.expectError([&] { require(f.device->createAccelStruct({}) == nullptr, "Unimplemented acceleration structure returned a usable handle"); });
        f.expectError([&] { require(f.device->createStagingTexture(texture, nvrhi::CpuAccessMode::Read) == nullptr, "Unimplemented staging texture accepted"); });
        nvrhi::BufferDesc buffer;
        buffer.byteSize = f.desc.pDevice.maxBufferLength + 1;
        f.expectError([&] { require(f.device->createBuffer(buffer) == nullptr, "Buffer exceeding device limit accepted"); });
        nvrhi::metal3::DeviceDesc invalid = f.desc;
        invalid.commonQueue = nil;
        f.expectError([&] { require(nvrhi::metal3::createDevice(invalid) == nullptr, "Device without a queue accepted"); });
        f.checkErrors();
    }

    void unsupportedRecording()
    {
        Fixture f;
        auto invalid = f.list(true);
        auto valid = f.list();
        invalid->open();
        invalid->clearBufferUInt(f.buffer, 1);
        f.expectError([&] { invalid->dispatchIndirect(0); });
        invalid->close();
        f.record(valid, 7, 1);
        nvrhi::ICommandList* batch[] = {valid, invalid};
        f.expectError([&] { require(f.device->executeCommandLists(batch, 2) == 0, "Unsupported command recording was submitted"); });
        require(f.device->waitForIdle(), "Idle wait after rejected batch failed");
        require(f.words[0] == 0x55555555u && f.words[1] == 0x55555555u, "Rejected batch partially executed");
        f.execute(valid);
        require(f.device->waitForIdle(), "Valid list from rejected batch failed");
        require(f.words[0] == 0x55555555u && f.words[1] == 7, "Valid list was damaged by batch rejection");
        f.record(invalid, 9);
        f.execute(invalid);
        require(f.device->waitForIdle(), "Re-recorded immediate list failed");
        require(f.words[0] == 9 && f.words[1] == 7, "Failed immediate recording could not recover cleanly");
        f.checkErrors();
    }

    void queueDependencies()
    {
        Fixture f;
        Gate gate(f.desc.pDevice);
        id<MTLCommandBuffer> blocker = [f.desc.commonQueue commandBuffer];
        [blocker encodeWaitForEvent:gate.event value:1];
        [blocker commit];
        auto producer = f.list(false, nvrhi::CommandQueue::Copy);
        f.record(producer, 0x98765432u);
        const uint64_t copyID = f.execute(producer, nvrhi::CommandQueue::Copy);
        f.expectError([&] { f.device->queueWaitForCommandList(nvrhi::CommandQueue::Compute, nvrhi::CommandQueue::Graphics, copyID); });
        f.device->queueWaitForCommandList(nvrhi::CommandQueue::Compute, nvrhi::CommandQueue::Copy, copyID);
        auto compute = f.list(false, nvrhi::CommandQueue::Compute);
        compute->open();
        compute->copyBuffer(f.buffer, 4, f.buffer, 0, 4);
        compute->close();
        const uint64_t computeID = f.execute(compute, nvrhi::CommandQueue::Compute);
        f.device->queueWaitForCommandList(nvrhi::CommandQueue::Graphics, nvrhi::CommandQueue::Compute, computeID);
        auto graphics = f.list();
        graphics->open();
        graphics->copyBuffer(f.buffer, 8, f.buffer, 4, 4);
        graphics->close();
        f.execute(graphics);
        auto query = f.device->createEventQuery();
        f.device->setEventQuery(query, nvrhi::CommandQueue::Graphics);
        f.device->runGarbageCollection();
        require(!f.device->pollEventQuery(query), "Queue chain completed while its producer was blocked");
        gate.event.signaledValue = 1;
        f.device->waitEventQuery(query);
        require(f.device->pollEventQuery(query), "Queue chain completion was not observed");
        require(f.words[2] == 0x98765432u, "Cross-queue consumer missed its producer's write");
        f.device->runGarbageCollection();
        f.device->queueWaitForCommandList(nvrhi::CommandQueue::Graphics, nvrhi::CommandQueue::Copy, copyID);
        f.device->queueWaitForCommandList(nvrhi::CommandQueue::Copy, nvrhi::CommandQueue::Graphics, 0);
        f.expectError([&] { f.device->queueWaitForCommandList(nvrhi::CommandQueue::Compute, nvrhi::CommandQueue::Copy, copyID + 1); });
        f.expectError([&] { f.device->queueWaitForCommandList(nvrhi::CommandQueue::Count, nvrhi::CommandQueue::Copy, copyID); });
        require(f.device->waitForIdle(), "Successful queue chain reported an idle failure");
        f.checkErrors();
    }

    void simultaneousWaiters()
    {
        Fixture f;
        Gate gate(f.desc.pDevice);
        id<MTLCommandBuffer> blocker = [f.desc.commonQueue commandBuffer];
        [blocker encodeWaitForEvent:gate.event value:1];
        [blocker commit];
        auto query = f.device->createEventQuery();
        f.device->setEventQuery(query, nvrhi::CommandQueue::Graphics);
        std::atomic<unsigned> entered{0};
        std::atomic<unsigned> finished{0};
        std::atomic<bool> idleSucceeded{false};
        auto wait = [&] {
            @autoreleasepool
            {
                ++entered;
                f.device->waitEventQuery(query);
                ++finished;
            }
        };
        std::thread first(wait);
        std::thread second(wait);
        std::thread idle([&] {
            @autoreleasepool
            {
                ++entered;
                idleSucceeded = f.device->waitForIdle();
                ++finished;
            }
        });
        while (entered.load() != 3)
            std::this_thread::yield();
        f.device->runGarbageCollection();
        const bool stayedPending = finished.load() == 0 && !f.device->pollEventQuery(query);
        gate.event.signaledValue = 1;
        first.join();
        second.join();
        idle.join();
        require(stayedPending, "A wait returned before GPU completion");
        require(finished == 3 && idleSucceeded, "Concurrent waiters did not all observe completion");
        f.checkErrors();
    }

    void lifetimeTrackers()
    {
        Fixture f;
        Gate gate(f.desc.pDevice);
        id<MTLCommandBuffer> blocker = [f.desc.commonQueue commandBuffer];
        [blocker encodeWaitForEvent:gate.event value:1];
        [blocker commit];
        auto tracker = f.device->createCommandListLifetimeTracker(nvrhi::CommandQueue::Graphics);
        require(tracker != nullptr, "Native lifetime tracker unavailable");
        nvrhi::CommandListParameters params;
        params.enableImmediateExecution = false;
        params.lifetimeTracker = tracker;
        auto commands = f.device->createCommandList(params);
        require(commands != nullptr, "Native lifetime tracker was rejected");
        for (unsigned index = 0; index < 8; ++index)
        {
            f.record(commands, 0x23450000u + index, index);
            f.execute(commands);
            tracker->runGarbageCollection();
        }
        auto query = f.device->createEventQuery();
        f.device->setEventQuery(query, nvrhi::CommandQueue::Graphics);
        tracker = nullptr;
        params.lifetimeTracker = nullptr;
        commands = nullptr;
        f.device->runGarbageCollection();
        require(!f.device->pollEventQuery(query), "Tracker destruction manufactured GPU completion");
        gate.event.signaledValue = 1;
        f.device->waitEventQuery(query);
        for (unsigned index = 0; index < 8; ++index)
            require(f.words[index] == 0x23450000u + index, "Tracker disposal corrupted in-flight work");
        f.device->runGarbageCollection();

        tracker = f.device->createCommandListLifetimeTracker(nvrhi::CommandQueue::Copy);
        params.queueType = nvrhi::CommandQueue::Copy;
        params.lifetimeTracker = tracker;
        commands = f.device->createCommandList(params);
        require(commands != nullptr, "Copy-queue lifetime tracker rejected");
        f.record(commands, 0xabcdef12u);
        f.execute(commands, nvrhi::CommandQueue::Copy);
        f.device->setEventQuery(query, nvrhi::CommandQueue::Copy);
        f.device->waitEventQuery(query);
        commands = nullptr;
        tracker->runGarbageCollection();
        require(f.words[0] == 0xabcdef12u, "Custom tracker collection lost submitted work");
        params.queueType = nvrhi::CommandQueue::Graphics;
        f.expectError([&] { require(f.device->createCommandList(params) == nullptr, "Wrong-queue lifetime tracker accepted"); });
        auto otherDesc = f.desc;
        otherDesc.commonQueue = [f.desc.pDevice newCommandQueue];
        auto other = nvrhi::metal3::createDevice(otherDesc);
        params.queueType = nvrhi::CommandQueue::Copy;
        f.expectError([&] { require(other->createCommandList(params) == nullptr, "Foreign-device lifetime tracker accepted"); });
        f.expectError([&] { require(f.device->createCommandListLifetimeTracker(nvrhi::CommandQueue::Count) == nullptr, "Invalid tracker queue accepted"); });
        f.expectError([&] { require(!other->pollEventQuery(query), "Foreign-device event query accepted"); });
        f.expectError([&] { f.device->setEventQuery(query, nvrhi::CommandQueue::Count); });
        f.checkErrors();
    }

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
            eventRearm();
            queueDependencies();
            simultaneousWaiters();
            lifetimeTrackers();
            unsupportedCapabilities();
            unsupportedRecording();
            std::puts("PASS: submission, event rearm, queue dependencies, concurrent waits, and lifetime tracking");
            return 0;
        }
        catch (const std::exception& error)
        {
            std::fprintf(stderr, "FAIL: %s\n", error.what());
            return 1;
        }
    }
}
