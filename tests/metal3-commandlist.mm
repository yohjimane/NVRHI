#include <nvrhi/metal3.h>
#include <nvrhi/common/resource.h>
#include <atomic>
#include <array>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <cstring>
#pragma push_macro("BOOL")
#undef BOOL
#define BOOL MetalDxcBOOL
#include <dxcapi.h>
#pragma pop_macro("BOOL")

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

    void timerQueries()
    {
        Fixture f;
        std::array<nvrhi::TimerQueryHandle, 260> queries;
        for (auto& query : queries)
        {
            query = f.device->createTimerQuery();
            require(query != nullptr, "GPU timer allocation failed");
        }
        auto commands = f.list();
        for (unsigned cycle = 0; cycle < 3; ++cycle)
        {
            Gate gate(f.desc.pDevice);
            id<MTLCommandBuffer> blocker = [f.desc.commonQueue commandBuffer];
            [blocker encodeWaitForEvent:gate.event value:1];
            [blocker commit];
            commands->open();
            for (unsigned index = 0; index < queries.size(); ++index)
            {
                auto& query = queries[index];
                f.device->resetTimerQuery(query);
                commands->beginTimerQuery(query);
                const uint32_t value = cycle * 1000 + index;
                if (index % 2)
                    commands->writeBuffer(f.buffer, &value, sizeof(value), index * sizeof(value));
                commands->endTimerQuery(query);
            }
            commands->close();
            f.execute(commands);
            require(!f.device->pollTimerQuery(queries.front()), "Blocked GPU timer completed early");
            f.expectError([&] { f.device->resetTimerQuery(queries.front()); });
            gate.event.signaledValue = 1;
            f.device->waitForIdle();
            Gate laterGate(f.desc.pDevice);
            id<MTLCommandBuffer> laterBlocker = [f.desc.commonQueue commandBuffer];
            [laterBlocker encodeWaitForEvent:laterGate.event value:1];
            [laterBlocker commit];
            auto later = f.list();
            auto laterQuery = f.device->createTimerQuery();
            later->open();
            later->beginTimerQuery(laterQuery);
            const uint32_t laterValue = 0x12345678;
            later->writeBuffer(f.buffer, &laterValue, sizeof(laterValue), 1023 * sizeof(laterValue));
            later->endTimerQuery(laterQuery);
            later->close();
            f.execute(later);
            require(!f.device->pollTimerQuery(laterQuery), "Later GPU timer completed while blocked");
            float elapsed = 0.f;
            for (auto& query : queries)
            {
                require(f.device->pollTimerQuery(query), "Completed GPU timer remains unavailable");
                const float duration = f.device->getTimerQueryTime(query);
                require(std::isfinite(duration) && duration > 0.f, "GPU timer samples are missing or invalid");
                elapsed += duration;
            }
            require(elapsed > 0.f && elapsed < 5.f, "GPU timer durations have invalid units");
            laterGate.event.signaledValue = 1;
            f.device->waitForIdle();
            const float laterDuration = f.device->getTimerQueryTime(laterQuery);
            require(std::isfinite(laterDuration) && laterDuration > 0.f, "Later GPU timer samples are missing or invalid");
            require(f.words[259] == cycle * 1000 + 259 && f.words[1023] == laterValue,
                "Timer instrumentation corrupted GPU work");
            f.device->runGarbageCollection();
        }
        f.checkErrors();
    }

    void reusedTimerQueriesAcrossGpuWaits()
    {
        Fixture f;
        auto query = f.device->createTimerQuery();
        require(query != nullptr, "GPU timer allocation failed");
        auto commands = f.list();
        for (unsigned delayMs : {2u, 8u, 3u})
        {
            Gate started(f.desc.pDevice);
            Gate release(f.desc.pDevice);
            f.device->resetTimerQuery(query);
            commands->open();
            commands->beginTimerQuery(query);
            auto native = static_cast<nvrhi::metal3::ICommandList*>(commands.Get())->getNativeCommandBuffer();
            [native encodeSignalEvent:started.event value:1];
            [native encodeWaitForEvent:release.event value:1];
            commands->writeBuffer(f.buffer, &delayMs, sizeof(delayMs));
            commands->endTimerQuery(query);
            commands->close();
            f.execute(commands);
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (started.event.signaledValue != 1)
            {
                require(std::chrono::steady_clock::now() < deadline, "GPU timer begin was not reached");
                std::this_thread::yield();
            }
            require(!f.device->pollTimerQuery(query), "GPU timer reused a result before its end executed");
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
            release.event.signaledValue = 1;
            deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!f.device->pollTimerQuery(query))
            {
                require(std::chrono::steady_clock::now() < deadline, "GPU timer end did not complete");
                std::this_thread::yield();
            }
            const float elapsed = f.device->getTimerQueryTime(query);
            require(std::isfinite(elapsed) && elapsed >= float(delayMs) * 0.001f,
                "Reused GPU timer omitted its current GPU wait or returned stale samples");
            [native waitUntilCompleted];
            const double commandBufferDuration = native.GPUEndTime - native.GPUStartTime;
            require(std::isfinite(commandBufferDuration) && commandBufferDuration > 0.0,
                "Native command buffer GPU duration is unavailable");
            require(double(elapsed) <= commandBufferDuration + 0.001,
                "GPU timer exceeds its enclosing command buffer duration");
            require(f.words[0] == delayMs, "GPU timer completion preceded its recorded write");
            f.device->runGarbageCollection();
        }
        f.checkErrors();
    }

    void mixedTimerQueries()
    {
        Fixture f;
        NSError* error = nil;
        NSString* source = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void work(device uint* words [[buffer(0)]], uint tid [[thread_position_in_grid]])"
            "{ uint v = words[tid]; for (uint i = 0; i < 512; ++i) v = v * 1664525u + 1013904223u; words[tid] = v; }";
        id<MTLLibrary> library = [f.desc.pDevice newLibraryWithSource:source options:nil error:&error];
        require(library != nil, "Timer workload shader compilation failed");
        id<MTLComputePipelineState> pipeline = [f.desc.pDevice
            newComputePipelineStateWithFunction:[library newFunctionWithName:@"work"] error:&error];
        require(pipeline != nil, "Timer workload pipeline creation failed");
        nvrhi::TextureDesc desc;
        desc.width = desc.height = 2048;
        desc.format = nvrhi::Format::RGBA8_UNORM;
        desc.isRenderTarget = true;
        desc.initialState = nvrhi::ResourceStates::RenderTarget;
        desc.keepInitialState = true;
        auto texture = f.device->createTexture(desc);
        require(texture != nullptr, "Timer workload render target creation failed");
        std::array<std::array<nvrhi::TimerQueryHandle, 65>, 3> queries;
        std::array<nvrhi::CommandListHandle, 3> commands;
        for (unsigned frame = 0; frame < commands.size(); ++frame)
        {
            commands[frame] = f.list();
            for (auto& query : queries[frame])
            {
                query = f.device->createTimerQuery();
                require(query != nullptr, "Mixed workload timer allocation failed");
            }
        }
        uint32_t expected = 0x55555555u;
        for (unsigned cycle = 0; cycle < 3; ++cycle)
        {
            for (unsigned frame = 0; frame < commands.size(); ++frame)
            {
                auto& command = commands[frame];
                command->open();
                for (auto& query : queries[frame])
                    f.device->resetTimerQuery(query);
                command->beginTimerQuery(queries[frame][0]);
                for (unsigned index = 1; index < queries[frame].size(); ++index)
                {
                    command->beginTimerQuery(queries[frame][index]);
                    if (index % 3 == 0)
                        command->clearTextureFloat(texture, nvrhi::AllSubresources, nvrhi::Color(0.25f));
                    else if (index % 3 == 1)
                    {
                        auto native = static_cast<nvrhi::metal3::ICommandList*>(command.Get())->getNativeCommandBuffer();
                        id<MTLComputeCommandEncoder> encoder = [native computeCommandEncoder];
                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:(__bridge id<MTLBuffer>)
                            f.buffer->getNativeObject(nvrhi::ObjectTypes::MTL3_Buffer).pointer offset:0 atIndex:0];
                        [encoder dispatchThreads:MTLSizeMake(1024, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                        [encoder endEncoding];
                        for (unsigned iteration = 0; iteration < 512; ++iteration)
                            expected = expected * 1664525u + 1013904223u;
                    }
                    else
                    {
                        const uint32_t value = cycle + index;
                        command->writeBuffer(f.buffer, &value, sizeof(value));
                    }
                    command->endTimerQuery(queries[frame][index]);
                }
                command->endTimerQuery(queries[frame][0]);
                command->close();
                f.execute(command);
            }
            f.device->waitForIdle();
            for (const auto& frameQueries : queries)
            {
                const float outer = f.device->getTimerQueryTime(frameQueries[0]);
                require(std::isfinite(outer) && outer > 0.f && outer < 5.f, "Outer GPU timer duration is invalid");
                for (unsigned index = 1; index < frameQueries.size(); ++index)
                {
                    const float duration = f.device->getTimerQueryTime(frameQueries[index]);
                    require(std::isfinite(duration) && duration > 0.f && duration <= outer,
                        "Nested GPU timer is missing, invalid, or outside its enclosing interval");
                }
            }
            for (unsigned index = 1; index < 1024; ++index)
                require(f.words[index] == expected, "Mixed workload timer instrumentation corrupted compute results");
            f.device->runGarbageCollection();
        }
        f.checkErrors();
    }

    nvrhi::ShaderHandle compileRegressionShader(nvrhi::IDevice* device, IDxcCompiler3* compiler,
        const char* source, nvrhi::ShaderType type)
    {
        const char* entry = type == nvrhi::ShaderType::Vertex ? "vsMain"
            : type == nvrhi::ShaderType::Pixel ? "psMain" : "csMain";
        const wchar_t* wideEntry = type == nvrhi::ShaderType::Vertex ? L"vsMain"
            : type == nvrhi::ShaderType::Pixel ? L"psMain" : L"csMain";
        const wchar_t* profile = type == nvrhi::ShaderType::Vertex ? L"vs_6_6"
            : type == nvrhi::ShaderType::Pixel ? L"ps_6_6" : L"cs_6_6";
        DxcBuffer input{source, std::strlen(source), DXC_CP_UTF8};
        const wchar_t* args[] = {L"-E", wideEntry, L"-T", profile};
        nvrhi::RefCountPtr<IDxcResult> result;
        require(SUCCEEDED(compiler->Compile(&input, args, 4, nullptr, IID_PPV_ARGS(result.GetAddressOf()))),
            "Indirect regression shader compilation failed");
        HRESULT status = E_FAIL;
        require(SUCCEEDED(result->GetStatus(&status)) && SUCCEEDED(status), "Indirect regression shader is invalid");
        nvrhi::RefCountPtr<IDxcBlob> bytes;
        require(SUCCEEDED(result->GetResult(bytes.GetAddressOf())), "Indirect regression shader bytecode missing");
        auto shader = device->createShader(nvrhi::ShaderDesc(type).setEntryName(entry),
            bytes->GetBufferPointer(), bytes->GetBufferSize());
        require(shader != nullptr, "Indirect regression shader creation failed");
        return shader;
    }

    void uavTextureClears()
    {
        Fixture f;
        nvrhi::TextureDesc desc;
        desc.width = desc.height = 8;
        desc.arraySize = desc.mipLevels = 2;
        desc.dimension = nvrhi::TextureDimension::Texture2DArray;
        desc.format = nvrhi::Format::RGBA32_FLOAT;
        desc.isUAV = true;
        auto texture = f.device->createTexture(desc);
        require(texture != nullptr, "UAV texture allocation failed");
        nvrhi::TextureSubresourceSet selected;
        selected.baseMipLevel = selected.baseArraySlice = 1;
        selected.numMipLevels = selected.numArraySlices = 1;
        auto commands = f.list();
        commands->open();
        commands->clearTextureFloat(texture, nvrhi::AllSubresources, nvrhi::Color(0.25f));
        commands->clearTextureFloat(texture, selected, nvrhi::Color(1.f, 2.f, 3.f, 4.f));
        commands->close();
        f.execute(commands);
        f.device->waitForIdle();
        id<MTLTexture> nativeTexture = (__bridge id<MTLTexture>)texture->getNativeObject(nvrhi::ObjectTypes::MTL3_Texture).pointer;
        id<MTLBuffer> readback = [f.desc.pDevice newBufferWithLength:2048 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> transfer = [f.desc.commonQueue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [transfer blitCommandEncoder];
        for (NSUInteger slice = 0; slice < 2; ++slice)
            [blit copyFromTexture:nativeTexture sourceSlice:slice sourceLevel:1 sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(4, 4, 1) toBuffer:readback destinationOffset:slice * 1024
                destinationBytesPerRow:256 destinationBytesPerImage:1024];
        [blit endEncoding];
        [transfer commit];
        [transfer waitUntilCompleted];
        require(transfer.status == MTLCommandBufferStatusCompleted, "UAV clear readback failed");
        for (unsigned slice = 0; slice < 2; ++slice)
            for (unsigned y = 0; y < 4; ++y)
                for (unsigned x = 0; x < 4; ++x)
                    for (unsigned channel = 0; channel < 4; ++channel)
                    {
                        const auto* row = reinterpret_cast<const float*>(
                            static_cast<const uint8_t*>(readback.contents) + slice * 1024 + y * 256);
                        require(row[x * 4 + channel] == (slice ? float(channel + 1) : 0.25f),
                            "UAV clear lost a channel or modified an unselected array slice");
                    }
        f.checkErrors();
    }

    void textureViewResults()
    {
        Fixture f;
        nvrhi::TextureDesc desc;
        desc.width = desc.height = 2;
        desc.arraySize = desc.mipLevels = 2;
        desc.dimension = nvrhi::TextureDimension::Texture2DArray;
        desc.format = nvrhi::Format::RGBA8_UNORM;
        desc.initialState = nvrhi::ResourceStates::ShaderResource;
        desc.keepInitialState = true;
        auto typed = f.device->createTexture(desc);
        desc.width = desc.height = desc.arraySize = desc.mipLevels = 1;
        desc.dimension = nvrhi::TextureDimension::Texture2D;
        desc.isTypeless = true;
        auto typeless = f.device->createTexture(desc);
        desc.format = nvrhi::Format::D32S8;
        desc.isTypeless = false;
        desc.isRenderTarget = true;
        auto depthStencil = f.device->createTexture(desc);
        desc.format = nvrhi::Format::D32;
        desc.isTypeless = true;
        auto depthOnly = f.device->createTexture(desc);
        require(typed != nullptr && typeless != nullptr && depthStencil != nullptr && depthOnly != nullptr,
            "View texture creation failed");
        auto uploads = f.list();
        uploads->open();
        const uint32_t background[] = {0x11223344u, 0x11223344u, 0x11223344u, 0x11223344u};
        const uint32_t selected = 0x80402080u;
        const uint32_t reinterpreted = 0x12345678u;
        for (unsigned slice = 0; slice < 2; ++slice)
        {
            uploads->writeTexture(typed, slice, 0, background, 2 * sizeof(uint32_t));
            uploads->writeTexture(typed, slice, 1, slice == 1 ? &selected : background, sizeof(uint32_t));
        }
        uploads->writeTexture(typeless, 0, 0, &reinterpreted, sizeof(uint32_t));
        uploads->clearDepthStencilTexture(depthStencil, nvrhi::AllSubresources, true, 0.375f, true, 0x6d);
        uploads->clearDepthStencilTexture(depthOnly, nvrhi::AllSubresources, true, 0.625f, false, 0);
        uploads->close();
        f.execute(uploads);
        f.device->waitForIdle();
        const nvrhi::TextureSubresourceSet selectedSubresource(1, 1, 1, 1);
        const auto swizzle = nvrhi::ComponentMapping()
            .setR(nvrhi::ComponentSwizzle::B).setG(nvrhi::ComponentSwizzle::R)
            .setB(nvrhi::ComponentSwizzle::One).setA(nvrhi::ComponentSwizzle::Zero);
        std::array<id<MTLTexture>, 7> views = {
            (__bridge id<MTLTexture>)typed->getNativeView(nvrhi::ObjectTypes::MTL3_Texture,
                nvrhi::Format::RGBA8_UNORM, selectedSubresource, nvrhi::TextureDimension::Texture2D).pointer,
            (__bridge id<MTLTexture>)typed->getNativeView(nvrhi::ObjectTypes::MTL3_Texture,
                nvrhi::Format::SRGBA8_UNORM, selectedSubresource, nvrhi::TextureDimension::Texture2D).pointer,
            (__bridge id<MTLTexture>)typed->getNativeView(nvrhi::ObjectTypes::MTL3_Texture,
                nvrhi::Format::RGBA8_UNORM, selectedSubresource, nvrhi::TextureDimension::Texture2D, false, swizzle).pointer,
            (__bridge id<MTLTexture>)typeless->getNativeView(nvrhi::ObjectTypes::MTL3_Texture,
                nvrhi::Format::R32_UINT).pointer,
            (__bridge id<MTLTexture>)depthStencil->getNativeView(nvrhi::ObjectTypes::MTL3_Texture).pointer,
            nil,
            (__bridge id<MTLTexture>)depthOnly->getNativeView(nvrhi::ObjectTypes::MTL3_Texture).pointer
        };
        views[5] = [views[4] newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
        for (auto view : views)
            require(view != nil, "Required texture view creation failed");
        NSError* error = nil;
        NSString* source = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void readViews(texture2d<float, access::read> linear [[texture(0)]],"
            "texture2d<float, access::read> srgb [[texture(1)]],"
            "texture2d<float, access::read> swizzled [[texture(2)]],"
            "texture2d<uint, access::read> integer [[texture(3)]],"
            "depth2d<float, access::read> depth [[texture(4)]],"
            "texture2d<uint, access::read> stencil [[texture(5)]],"
            "depth2d<float, access::read> depthOnly [[texture(6)]], device uint4* output [[buffer(0)]]) {"
            "output[0] = as_type<uint4>(linear.read(uint2(0)));"
            "output[1] = as_type<uint4>(srgb.read(uint2(0)));"
            "output[2] = as_type<uint4>(swizzled.read(uint2(0)));"
            "output[3] = uint4(integer.read(uint2(0)).r, stencil.read(uint2(0)).r, 0, 0);"
            "output[4] = as_type<uint4>(float4(depth.read(uint2(0)), depthOnly.read(uint2(0)), 0, 0)); }";
        id<MTLLibrary> library = [f.desc.pDevice newLibraryWithSource:source options:nil error:&error];
        require(library != nil, "View shader compilation failed");
        id<MTLComputePipelineState> pipeline = [f.desc.pDevice
            newComputePipelineStateWithFunction:[library newFunctionWithName:@"readViews"] error:&error];
        require(pipeline != nil, "View pipeline creation failed");
        id<MTLCommandBuffer> commands = [f.desc.commonQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (unsigned index = 0; index < views.size(); ++index)
            [encoder setTexture:views[index] atIndex:index];
        [encoder setBuffer:(__bridge id<MTLBuffer>)
            f.buffer->getNativeObject(nvrhi::ObjectTypes::MTL3_Buffer).pointer offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        require(commands.status == MTLCommandBufferStatusCompleted, "Texture view GPU reads failed");
        const float linear[] = {128.f / 255.f, 32.f / 255.f, 64.f / 255.f, 128.f / 255.f};
        const float swizzled[] = {linear[2], linear[0], 1.f, 0.f};
        for (unsigned view = 0; view < 3; ++view)
        {
            float actual[4];
            std::memcpy(actual, f.words + view * 4, sizeof(actual));
            for (unsigned channel = 0; channel < 4; ++channel)
            {
                const float expected = view == 2 ? swizzled[channel]
                    : view == 1 && channel < 3 ? std::pow((linear[channel] + 0.055f) / 1.055f, 2.4f)
                    : linear[channel];
                require(std::abs(actual[channel] - expected) < 0.002f,
                    "Typed subresource, sRGB, or swizzled view returned incorrect components");
            }
        }
        require(f.words[12] == reinterpreted && f.words[13] == 0x6d,
            "Typeless reinterpretation or stencil aspect returned incorrect data");
        float depth;
        std::memcpy(&depth, f.words + 16, sizeof(depth));
        require(depth == 0.375f, "Combined depth/stencil texture lost its depth aspect");
        std::memcpy(&depth, f.words + 17, sizeof(depth));
        require(depth == 0.625f, "Typeless single-aspect depth texture returned incorrect depth");
        f.checkErrors();
    }

    void fragmentTimerOrdering()
    {
        Fixture f;
        nvrhi::RefCountPtr<IDxcCompiler3> compiler;
        require(SUCCEEDED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(compiler.GetAddressOf()))),
            "DXC compiler creation failed");
        const char* source =
            "float4 vsMain(uint id : SV_VertexID) : SV_Position {"
            "float2 p = float2((id << 1) & 2, id & 2); return float4(p * 2 - 1, 0, 1); }"
            "float4 psMain() : SV_Target { return float4(1, 2, 3, 1) / 64; }";
        constexpr unsigned width = 2252;
        constexpr unsigned height = 1672;
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = nvrhi::Format::RGBA16_FLOAT;
        desc.isRenderTarget = true;
        desc.initialState = nvrhi::ResourceStates::RenderTarget;
        desc.keepInitialState = true;
        auto texture = f.device->createTexture(desc);
        require(texture != nullptr, "Fragment timer texture creation failed");
        auto framebuffer = f.device->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(texture));
        require(framebuffer != nullptr, "Fragment timer framebuffer creation failed");
        nvrhi::GraphicsPipelineDesc pipelineDesc;
        pipelineDesc.VS = compileRegressionShader(f.device, compiler, source, nvrhi::ShaderType::Vertex);
        pipelineDesc.PS = compileRegressionShader(f.device, compiler, source, nvrhi::ShaderType::Pixel);
        pipelineDesc.renderState.depthStencilState.depthTestEnable = false;
        pipelineDesc.renderState.depthStencilState.depthWriteEnable = false;
        pipelineDesc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;
        auto& blend = pipelineDesc.renderState.blendState.targets[0];
        blend.blendEnable = true;
        blend.srcBlend = blend.destBlend = nvrhi::BlendFactor::One;
        blend.srcBlendAlpha = blend.destBlendAlpha = nvrhi::BlendFactor::One;
        auto pipeline = f.device->createGraphicsPipeline(pipelineDesc, framebuffer->getFramebufferInfo());
        require(pipeline != nullptr, "Fragment timer pipeline creation failed");
        auto initialize = f.list();
        initialize->open();
        initialize->clearTextureFloat(texture, nvrhi::AllSubresources, nvrhi::Color(0.f));
        initialize->close();
        f.execute(initialize);
        f.device->waitForIdle();
        auto query = f.device->createTimerQuery();
        auto trailingQuery = f.device->createTimerQuery();
        require(query != nullptr && trailingQuery != nullptr, "Fragment timer allocation failed");
        auto commands = f.list();
        commands->open();
        auto native = static_cast<nvrhi::metal3::ICommandList*>(commands.Get())->getNativeCommandBuffer();
        commands->beginTimerQuery(query);
        nvrhi::GraphicsState state;
        state.pipeline = pipeline;
        state.framebuffer = framebuffer;
        state.viewport.addViewport(nvrhi::Viewport(float(width), float(height)));
        state.viewport.addScissorRect(nvrhi::Rect(0, width, 0, height));
        commands->setGraphicsState(state);
        for (unsigned draw = 0; draw < 64; ++draw)
            commands->draw(nvrhi::DrawArguments().setVertexCount(3));
        commands->beginTimerQuery(trailingQuery);
        commands->endTimerQuery(trailingQuery);
        commands->endTimerQuery(query);
        commands->close();
        f.execute(commands);
        [native waitUntilCompleted];
        require(native.status == MTLCommandBufferStatusCompleted, "Fragment timer GPU work failed");
        const double enclosing = native.GPUEndTime - native.GPUStartTime;
        const double measured = f.device->getTimerQueryTime(query);
        require(std::isfinite(enclosing) && enclosing > 0.0 && std::isfinite(measured),
            "Fragment timer GPU durations are unavailable");
        require(measured >= enclosing * 0.5 && measured <= enclosing * 1.1,
            "GPU timer did not enclose its command buffer's fragment workload");
        const double trailing = f.device->getTimerQueryTime(trailingQuery);
        require(std::isfinite(trailing) && trailing > 0.0 && trailing < measured * 0.5,
            "Empty trailing timer included fragment work recorded before its begin");
        id<MTLCommandBuffer> readback = [f.desc.commonQueue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [readback blitCommandEncoder];
        const MTLOrigin origins[] = {
            MTLOriginMake(0, 0, 0), MTLOriginMake(width / 2, height / 2, 0),
            MTLOriginMake(width - 1, height - 1, 0)
        };
        for (unsigned index = 0; index < 3; ++index)
            [blit copyFromTexture:(__bridge id<MTLTexture>)
                texture->getNativeObject(nvrhi::ObjectTypes::MTL3_Texture).pointer
                sourceSlice:0 sourceLevel:0 sourceOrigin:origins[index] sourceSize:MTLSizeMake(1, 1, 1)
                toBuffer:(__bridge id<MTLBuffer>)f.buffer->getNativeObject(nvrhi::ObjectTypes::MTL3_Buffer).pointer
                destinationOffset:index * 256 destinationBytesPerRow:256 destinationBytesPerImage:256];
        [blit endEncoding];
        [readback commit];
        [readback waitUntilCompleted];
        require(readback.status == MTLCommandBufferStatusCompleted, "Fragment timer readback failed");
        const uint16_t expected[] = {0x3c00, 0x4000, 0x4200, 0x3c00};
        for (unsigned index = 0; index < 3; ++index)
            require(std::memcmp(f.words + index * 64, expected, sizeof(expected)) == 0,
                "Fragment timer instrumentation lost blended fullscreen draws");
        f.checkErrors();
    }

    void sparseDescriptorSnapshots()
    {
        Fixture f;
        nvrhi::RefCountPtr<IDxcCompiler3> compiler;
        require(SUCCEEDED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(compiler.GetAddressOf()))),
            "DXC compiler creation failed");
        const char* source =
            "Texture2D<float4> textures[] : register(t0, space1);"
            "RWByteAddressBuffer output : register(u0);"
            "[numthreads(1, 1, 1)] void csMain() {"
            "uint low = uint(textures[3].Load(int3(0, 0, 0)).r * 255);"
            "uint high = uint(textures[65535].Load(int3(0, 0, 0)).g * 255);"
            "output.Store(0, low | (high << 8)); }";
        auto layout = f.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(0)));
        nvrhi::BindlessLayoutDesc heapDesc;
        heapDesc.visibility = nvrhi::ShaderType::All;
        heapDesc.maxCapacity = 65536;
        heapDesc.registerSpaces = {nvrhi::BindingLayoutItem::Texture_SRV(1)};
        auto heapLayout = f.device->createBindlessLayout(heapDesc);
        auto heap = f.device->createDescriptorTable(heapLayout);
        require(layout != nullptr && heap != nullptr, "Sparse descriptor layout creation failed");
        f.device->resizeDescriptorTable(heap, heapDesc.maxCapacity, false);
        auto shader = compileRegressionShader(f.device, compiler, source, nvrhi::ShaderType::Compute);
        auto pipeline = f.device->createComputePipeline(nvrhi::ComputePipelineDesc()
            .setComputeShader(shader).addBindingLayout(layout).addBindingLayout(heapLayout));
        require(pipeline != nullptr, "Sparse descriptor pipeline creation failed");
        nvrhi::TextureDesc textureDesc;
        textureDesc.width = textureDesc.height = 1;
        textureDesc.format = nvrhi::Format::RGBA8_UNORM;
        textureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        textureDesc.keepInitialState = true;
        std::array<nvrhi::TextureHandle, 3> textures;
        const uint32_t texels[] = {0xff0000ffu, 0xff00ff00u, 0xffff0000u};
        auto uploads = f.list();
        uploads->open();
        for (unsigned index = 0; index < textures.size(); ++index)
        {
            textures[index] = f.device->createTexture(textureDesc);
            require(textures[index] != nullptr, "Sparse descriptor texture creation failed");
            uploads->writeTexture(textures[index], 0, 0, &texels[index], sizeof(uint32_t));
        }
        uploads->close();
        f.execute(uploads);
        f.device->waitForIdle();
        uploads = nullptr;
        require(f.device->writeDescriptorTable(heap, nvrhi::BindingSetItem::Texture_SRV(3, textures[0])) &&
            f.device->writeDescriptorTable(heap, nvrhi::BindingSetItem::Texture_SRV(65535, textures[1])),
            "Sparse descriptor writes failed");
        std::array<nvrhi::CommandListHandle, 2> commands;
        std::array<nvrhi::BufferHandle, 2> outputs;
        for (unsigned index = 0; index < commands.size(); ++index)
        {
            nvrhi::BufferDesc outputDesc;
            outputDesc.byteSize = sizeof(uint32_t);
            outputDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
            outputDesc.canHaveUAVs = outputDesc.canHaveRawViews = true;
            outputDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            outputDesc.keepInitialState = true;
            outputs[index] = f.device->createBuffer(outputDesc);
            require(outputs[index] != nullptr, "Sparse descriptor output creation failed");
            nvrhi::BindingSetDesc bindings;
            bindings.bindings = {nvrhi::BindingSetItem::RawBuffer_UAV(0, outputs[index])};
            auto set = f.device->createBindingSet(bindings, layout);
            require(set != nullptr, "Sparse descriptor binding set creation failed");
            commands[index] = f.list();
            commands[index]->open();
            nvrhi::ComputeState state;
            state.pipeline = pipeline;
            state.bindings = {set, heap};
            commands[index]->setComputeState(state);
            commands[index]->dispatch(1);
            commands[index]->close();
            if (index == 0)
            {
                require(f.device->writeDescriptorTable(heap, nvrhi::BindingSetItem::Texture_SRV(3, textures[2])) &&
                    f.device->writeDescriptorTable(heap, nvrhi::BindingSetItem::Texture_SRV(65535, textures[2])),
                    "Sparse descriptor replacement failed");
                textures[0] = textures[1] = nullptr;
            }
        }
        nvrhi::ICommandList* reversed[] = {commands[1], commands[0]};
        require(f.device->executeCommandLists(reversed, 2) != 0, "Sparse descriptor submission failed");
        f.device->waitForIdle();
        for (unsigned index = 0; index < outputs.size(); ++index)
        {
            id<MTLBuffer> native = (__bridge id<MTLBuffer>)
                outputs[index]->getNativeObject(nvrhi::ObjectTypes::MTL3_Buffer).pointer;
            require(*static_cast<const uint32_t*>(native.contents) == (index == 0 ? 0xffffu : 0u),
                "Sparse descriptor indices or recorded snapshot contents changed");
        }
        f.checkErrors();
    }

    void interruptedIndirectDraws()
    {
        Fixture f;
        nvrhi::RefCountPtr<IDxcCompiler3> compiler;
        require(SUCCEEDED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(compiler.GetAddressOf()))),
            "DXC compiler creation failed");
        const char* source =
            "float4 vsMain(uint id : SV_VertexID) : SV_Position {"
            "float2 p = float2((id << 1) & 2, id & 2); return float4(p * 2 - 1, 0, 1); }"
            "float4 psMain() : SV_Target { return float4(1, 0, 0, 1); }";
        nvrhi::TextureDesc textureDesc;
        textureDesc.width = 64;
        textureDesc.height = 16;
        textureDesc.format = nvrhi::Format::RGBA8_UNORM;
        textureDesc.isRenderTarget = true;
        textureDesc.initialState = nvrhi::ResourceStates::RenderTarget;
        textureDesc.keepInitialState = true;
        auto texture = f.device->createTexture(textureDesc);
        auto framebuffer = f.device->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(texture));
        nvrhi::GraphicsPipelineDesc pipelineDesc;
        pipelineDesc.VS = compileRegressionShader(f.device, compiler, source, nvrhi::ShaderType::Vertex);
        pipelineDesc.PS = compileRegressionShader(f.device, compiler, source, nvrhi::ShaderType::Pixel);
        pipelineDesc.renderState.depthStencilState.depthTestEnable = false;
        pipelineDesc.renderState.depthStencilState.depthWriteEnable = false;
        pipelineDesc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;
        auto pipeline = f.device->createGraphicsPipeline(pipelineDesc, framebuffer->getFramebufferInfo());
        require(pipeline != nullptr, "Indirect regression pipeline creation failed");
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = 64;
        bufferDesc.isDrawIndirectArgs = true;
        bufferDesc.initialState = nvrhi::ResourceStates::IndirectArgument;
        bufferDesc.keepInitialState = true;
        auto arguments = f.device->createBuffer(bufferDesc);
        bufferDesc.isDrawIndirectArgs = false;
        bufferDesc.isIndexBuffer = true;
        bufferDesc.initialState = nvrhi::ResourceStates::IndexBuffer;
        auto indices = f.device->createBuffer(bufferDesc);
        auto commands = f.list();
        auto query = f.device->createTimerQuery();
        for (bool indexed : {false, true})
        {
            for (unsigned interruption = 0; interruption < 3; ++interruption)
            {
                f.device->resetTimerQuery(query);
                commands->open();
                nvrhi::DrawIndirectArguments draw;
                draw.vertexCount = 3;
                nvrhi::DrawIndexedIndirectArguments indexedDraw;
                indexedDraw.indexCount = 3;
                if (indexed)
                    commands->writeBuffer(arguments, &indexedDraw, sizeof(indexedDraw));
                else
                    commands->writeBuffer(arguments, &draw, sizeof(draw));
                const uint32_t indexData[] = {0, 1, 2};
                commands->writeBuffer(indices, indexData, sizeof(indexData));
                commands->clearTextureFloat(texture, nvrhi::AllSubresources, nvrhi::Color(0.f));
                nvrhi::GraphicsState state;
                state.pipeline = pipeline;
                state.framebuffer = framebuffer;
                state.viewport.addViewport(nvrhi::Viewport(64.f, 16.f));
                state.viewport.addScissorRect(nvrhi::Rect(0, 64, 0, 16));
                state.indirectParams = arguments;
                state.indexBuffer = {indices, nvrhi::Format::R32_UINT, 0};
                commands->setGraphicsState(state);
                if (interruption < 2)
                {
                    commands->beginTimerQuery(query);
                    if (interruption == 1)
                        commands->endTimerQuery(query);
                }
                else
                    commands->writeBuffer(f.buffer, indexData, sizeof(indexData));
                if (indexed)
                    commands->drawIndexedIndirect(0, 1);
                else
                    commands->drawIndirect(0, 1);
                if (interruption == 0)
                    commands->endTimerQuery(query);
                commands->setTextureState(texture, nvrhi::AllSubresources, nvrhi::ResourceStates::CopySource);
                commands->commitBarriers();
                auto native = static_cast<nvrhi::metal3::ICommandList*>(commands.Get())->getNativeCommandBuffer();
                id<MTLBlitCommandEncoder> blit = [native blitCommandEncoder];
                [blit copyFromTexture:(__bridge id<MTLTexture>)
                    texture->getNativeObject(nvrhi::ObjectTypes::MTL3_Texture).pointer
                    sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(64, 16, 1)
                    toBuffer:(__bridge id<MTLBuffer>)f.buffer->getNativeObject(nvrhi::ObjectTypes::MTL3_Buffer).pointer
                    destinationOffset:0 destinationBytesPerRow:256 destinationBytesPerImage:4096];
                [blit endEncoding];
                commands->close();
                f.execute(commands);
                f.device->waitForIdle();
                for (unsigned pixel = 0; pixel < 1024; ++pixel)
                    require(f.words[pixel] == 0xff0000ffu, "Indirect draw lost graphics state after encoder interruption");
                if (interruption < 2)
                    require(f.device->getTimerQueryTime(query) > 0.f, "Interrupted indirect draw timer is invalid");
            }
        }
        f.checkErrors();
    }

    void countedIndirectBindings()
    {
        Fixture f;
        nvrhi::RefCountPtr<IDxcCompiler3> compiler;
        require(SUCCEEDED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(compiler.GetAddressOf()))),
            "DXC compiler creation failed");
        const char* source =
            "cbuffer VertexConstants : register(b0) { float4 transform; };"
            "cbuffer PixelConstants : register(b1) { float4 leftColor; float4 rightColor; };"
            "Texture2D<float4> pattern : register(t0); SamplerState patternSampler : register(s0);"
            "struct V { float4 p : SV_Position; nointerpolation uint instance : INSTANCE;"
            "nointerpolation uint expected : EXPECTED; };"
            "V vsMain(float2 position : POSITION, uint expected : TEXCOORD0, uint instance : INSTANCEINPUT) {"
            "V v; v.p = float4(position * transform.xy + transform.zw, 0, 1);"
            "v.instance = instance; v.expected = expected; return v; }"
            "float4 psMain(V v) : SV_Target {"
            "if (v.instance != v.expected) return float4(0, 0, 1, 1);"
            "return (v.expected == 7 ? leftColor : rightColor)"
            " * pattern.SampleLevel(patternSampler, float2(1.25, 0.5), 0); }";
        const char* countSource =
            "RWByteAddressBuffer count : register(u0);"
            "[numthreads(1, 1, 1)] void csMain() { uint previous; count.InterlockedAdd(12, 1, previous); }";
        auto graphicsLayout = f.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel)
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(1))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0)));
        auto computeLayout = f.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(0)));
        require(graphicsLayout != nullptr && computeLayout != nullptr, "Counted indirect binding layout creation failed");
        nvrhi::TextureDesc textureDesc;
        textureDesc.width = 64;
        textureDesc.height = 16;
        textureDesc.format = nvrhi::Format::RGBA8_UNORM;
        textureDesc.isRenderTarget = true;
        textureDesc.initialState = nvrhi::ResourceStates::RenderTarget;
        textureDesc.keepInitialState = true;
        auto texture = f.device->createTexture(textureDesc);
        auto framebuffer = f.device->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(texture));
        textureDesc.width = 2;
        textureDesc.height = 1;
        textureDesc.isRenderTarget = false;
        textureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        auto pattern = f.device->createTexture(textureDesc);
        auto sampler = f.device->createSampler(nvrhi::SamplerDesc()
            .setAllFilters(false).setAllAddressModes(nvrhi::SamplerAddressMode::Wrap));
        require(texture != nullptr && framebuffer != nullptr && pattern != nullptr && sampler != nullptr,
            "Counted indirect texture or sampler creation failed");
        const nvrhi::VertexAttributeDesc attributes[] = {
            nvrhi::VertexAttributeDesc().setName("POSITION").setFormat(nvrhi::Format::RG32_FLOAT)
                .setOffset(0).setElementStride(12),
            nvrhi::VertexAttributeDesc().setName("TEXCOORD").setFormat(nvrhi::Format::R32_UINT)
                .setOffset(8).setElementStride(12),
            nvrhi::VertexAttributeDesc().setName("INSTANCEINPUT").setFormat(nvrhi::Format::R32_UINT)
                .setBufferIndex(1).setElementStride(sizeof(uint32_t)).setIsInstanced(true)
        };
        nvrhi::GraphicsPipelineDesc pipelineDesc;
        pipelineDesc.VS = compileRegressionShader(f.device, compiler, source, nvrhi::ShaderType::Vertex);
        pipelineDesc.PS = compileRegressionShader(f.device, compiler, source, nvrhi::ShaderType::Pixel);
        pipelineDesc.inputLayout = f.device->createInputLayout(attributes, 3, pipelineDesc.VS);
        require(pipelineDesc.inputLayout != nullptr, "Counted indirect input layout creation failed");
        pipelineDesc.bindingLayouts.push_back(graphicsLayout);
        pipelineDesc.renderState.depthStencilState.depthTestEnable = false;
        pipelineDesc.renderState.depthStencilState.depthWriteEnable = false;
        pipelineDesc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;
        auto pipeline = f.device->createGraphicsPipeline(pipelineDesc, framebuffer->getFramebufferInfo());
        auto countShader = compileRegressionShader(f.device, compiler, countSource, nvrhi::ShaderType::Compute);
        auto computePipeline = f.device->createComputePipeline(nvrhi::ComputePipelineDesc()
            .setComputeShader(countShader).addBindingLayout(computeLayout));
        require(pipeline != nullptr && computePipeline != nullptr, "Counted indirect pipeline creation failed");
        nvrhi::BufferDesc argumentDesc;
        argumentDesc.byteSize = 128;
        argumentDesc.isDrawIndirectArgs = true;
        argumentDesc.initialState = nvrhi::ResourceStates::IndirectArgument;
        argumentDesc.keepInitialState = true;
        auto arguments = f.device->createBuffer(argumentDesc);
        nvrhi::BufferDesc countDesc = argumentDesc;
        countDesc.byteSize = 16;
        countDesc.canHaveUAVs = true;
        countDesc.canHaveRawViews = true;
        auto count = f.device->createBuffer(countDesc);
        nvrhi::BufferDesc vertexDesc;
        vertexDesc.byteSize = 16 + 8 * 12;
        vertexDesc.isVertexBuffer = true;
        vertexDesc.initialState = nvrhi::ResourceStates::VertexBuffer;
        vertexDesc.keepInitialState = true;
        auto vertices = f.device->createBuffer(vertexDesc);
        vertexDesc.byteSize = 16 + 12 * sizeof(uint32_t);
        auto instances = f.device->createBuffer(vertexDesc);
        nvrhi::BufferDesc indexDesc;
        indexDesc.byteSize = 16 + 12 * sizeof(uint32_t);
        indexDesc.isIndexBuffer = true;
        indexDesc.initialState = nvrhi::ResourceStates::IndexBuffer;
        indexDesc.keepInitialState = true;
        auto indices = f.device->createBuffer(indexDesc);
        nvrhi::BufferDesc constantDesc;
        constantDesc.byteSize = 16;
        constantDesc.isConstantBuffer = true;
        constantDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        constantDesc.keepInitialState = true;
        auto vertexConstants = f.device->createBuffer(constantDesc);
        constantDesc.byteSize = 32;
        auto pixelConstants = f.device->createBuffer(constantDesc);
        require(arguments != nullptr && count != nullptr && vertices != nullptr && instances != nullptr
            && indices != nullptr && vertexConstants != nullptr && pixelConstants != nullptr,
            "Counted indirect buffer creation failed");
        nvrhi::BindingSetDesc graphicsBindings;
        graphicsBindings.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, vertexConstants),
            nvrhi::BindingSetItem::ConstantBuffer(1, pixelConstants),
            nvrhi::BindingSetItem::Texture_SRV(0, pattern),
            nvrhi::BindingSetItem::Sampler(0, sampler)
        };
        auto graphicsSet = f.device->createBindingSet(graphicsBindings, graphicsLayout);
        nvrhi::BindingSetDesc computeBindings;
        computeBindings.bindings = {nvrhi::BindingSetItem::RawBuffer_UAV(0, count)};
        auto computeSet = f.device->createBindingSet(computeBindings, computeLayout);
        require(graphicsSet != nullptr && computeSet != nullptr, "Counted indirect binding set creation failed");
        struct Vertex
        {
            float x, y;
            uint32_t expectedInstance;
        };
        const Vertex vertexData[] = {
            {-2.f, -2.f, 7}, {0.f, -2.f, 7}, {-2.f, 2.f, 7}, {0.f, 2.f, 7},
            {0.f, -2.f, 11}, {2.f, -2.f, 11}, {0.f, 2.f, 11}, {2.f, 2.f, 11}
        };
        const uint32_t indexData[] = {0, 1, 2, 2, 1, 3, 0, 1, 2, 2, 1, 3};
        const uint32_t instanceData[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        const float transform[] = {0.5f, 0.5f, 0.f, 0.f};
        const float colors[] = {1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f};
        const uint32_t texels[] = {0xffffffffu, 0xff000000u};
        const nvrhi::DrawIndexedIndirectArguments draws[] = {
            nvrhi::DrawIndexedIndirectArguments().setIndexCount(6).setStartInstanceLocation(7),
            nvrhi::DrawIndexedIndirectArguments().setIndexCount(6).setStartIndexLocation(6)
                .setBaseVertexLocation(4).setStartInstanceLocation(11),
            nvrhi::DrawIndexedIndirectArguments().setIndexCount(6)
        };
        nvrhi::GraphicsState state;
        state.pipeline = pipeline;
        state.framebuffer = framebuffer;
        state.viewport.addViewport(nvrhi::Viewport(64.f, 16.f));
        state.viewport.addScissorRect(nvrhi::Rect(0, 64, 0, 16));
        state.bindings.push_back(graphicsSet);
        state.vertexBuffers.push_back({vertices, 0, 16});
        state.vertexBuffers.push_back({instances, 1, 16});
        state.indexBuffer = {indices, nvrhi::Format::R32_UINT, 16};
        state.indirectParams = arguments;
        state.indirectCountBuffer = count;
        nvrhi::ComputeState computeState;
        computeState.pipeline = computePipeline;
        computeState.bindings.push_back(computeSet);
        auto commands = f.list();
        for (unsigned drawCount : {3u, 1u, 0u})
        {
            commands->open();
            if (drawCount == 3)
            {
                commands->clearBufferUInt(arguments, 0);
                commands->clearBufferUInt(vertices, 0);
                commands->clearBufferUInt(instances, 0);
                commands->clearBufferUInt(indices, 0);
                commands->writeBuffer(arguments, draws, sizeof(draws), 16);
                commands->writeBuffer(vertices, vertexData, sizeof(vertexData), 16);
                commands->writeBuffer(instances, instanceData, sizeof(instanceData), 16);
                commands->writeBuffer(indices, indexData, sizeof(indexData), 16);
                commands->writeBuffer(vertexConstants, transform, sizeof(transform));
                commands->writeBuffer(pixelConstants, colors, sizeof(colors));
                commands->writeTexture(pattern, 0, 0, texels, sizeof(texels));
            }
            commands->clearBufferUInt(count, 0);
            if (drawCount != 0)
            {
                commands->setComputeState(computeState);
                commands->dispatch(drawCount, 1, 1);
            }
            commands->clearTextureFloat(texture, nvrhi::AllSubresources, nvrhi::Color(0.f));
            commands->setGraphicsState(state);
            commands->drawIndexedIndirectCount(16, 12, 2);
            commands->setTextureState(texture, nvrhi::AllSubresources, nvrhi::ResourceStates::CopySource);
            commands->commitBarriers();
            auto native = static_cast<nvrhi::metal3::ICommandList*>(commands.Get())->getNativeCommandBuffer();
            id<MTLBlitCommandEncoder> blit = [native blitCommandEncoder];
            [blit copyFromTexture:(__bridge id<MTLTexture>)
                texture->getNativeObject(nvrhi::ObjectTypes::MTL3_Texture).pointer
                sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(64, 16, 1)
                toBuffer:(__bridge id<MTLBuffer>)f.buffer->getNativeObject(nvrhi::ObjectTypes::MTL3_Buffer).pointer
                destinationOffset:0 destinationBytesPerRow:256 destinationBytesPerImage:4096];
            [blit endEncoding];
            commands->close();
            f.execute(commands);
            require(f.device->waitForIdle(), "Counted indirect GPU submission failed");
            for (unsigned y = 0; y < 16; ++y)
            {
                for (unsigned x = 0; x < 64; ++x)
                {
                    const uint32_t expected = drawCount == 0 ? 0u
                        : x < 32 ? 0xff0000ffu : drawCount > 1 ? 0xff00ff00u : 0u;
                    require(f.words[y * 64 + x] == expected,
                        "Counted indirect draw lost per-command parameters, graphics bindings, or GPU count clipping");
                }
            }
            f.device->runGarbageCollection();
        }
        f.checkErrors();
    }

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
        f.expectError([&] { require(std::isnan(f.device->getTimerQueryTime(nullptr)), "Invalid timer returned a fabricated duration"); });
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
            timerQueries();
            reusedTimerQueriesAcrossGpuWaits();
            mixedTimerQueries();
            fragmentTimerOrdering();
            textureViewResults();
            uavTextureClears();
            sparseDescriptorSnapshots();
            interruptedIndirectDraws();
            countedIndirectBindings();
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
