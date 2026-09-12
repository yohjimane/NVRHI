#include "metal3-backend.h"
#include "nvrhi/common/misc.h"
#include "nvrhi/common/resource.h"
#include "nvrhi/utils.h"
#include <Metal/Metal.h>
#include <atomic>
#include <cstdio>

namespace nvrhi::metal3
{
    void MTL3Context::error(const std::string& message) const
    {
        if (messageCallback)
            messageCallback->message(MessageSeverity::Error, message.c_str());
    }
    void MTL3Context::warning(const std::string& message) const
    {
        if (messageCallback)
            messageCallback->message(MessageSeverity::Warning, message.c_str());
    }

    void MTL3Context::info(const std::string& message) const
    {
        if (messageCallback)
            messageCallback->message(MessageSeverity::Info, message.c_str());
    }

    // device creation
    DeviceHandle createDevice(const DeviceDesc& desc)
    {
        Device* device = new Device(desc);
        return DeviceHandle::Create(device);
    }
    Device::Device(const DeviceDesc& desc)
        : m_AftermathEnabled(false)
    {
        m_Context.device = desc.pDevice;
        m_Context.logBufferLifetime = desc.logBufferLifetime;
        m_Context.messageCallback = desc.errorCB;

        if([m_Context.device supportsFamily:MTLGPUFamilyMetal3] == NO)
        {
            m_Context.error("[nvrhi] Metal 3 unsupported!");
        }
        else m_Context.info("[nvrhi] Metal 3 supported");

        // queues, resoureces reserve, allocation, etc...
        m_Context.commonQueue = desc.commonQueue;
        for (uint32_t index = 0; index < uint32_t(CommandQueue::Count); ++index)
            m_DefaultLifetimeTrackers[index] = RefCountPtr<CommandListLifetimeTracker>::Create(
                new CommandListLifetimeTracker(this, CommandQueue(index), true));
    }

    Device::~Device()
    {
        waitForIdle();
    }

    Object Device::getNativeObject(ObjectType objectType)
    {
        switch (objectType)
        {
        case ObjectTypes::MTL3_Device:
            return Object((__bridge void*)m_Context.device);
        case ObjectTypes::Nvrhi_Metal3_Device:
            return Object(this);
        default:
            return nullptr;
        }
    }

    GraphicsAPI Device::getGraphicsAPI()
    {
        return GraphicsAPI::METAL3;
    }
    Object Device::getNativeQueue(ObjectType objectType, CommandQueue queue)
    {
        if (objectType == ObjectTypes::MTL3_CommandQueue && uint32_t(queue) < uint32_t(CommandQueue::Count))
            return Object((__bridge void*)m_Context.commonQueue);
        return nullptr;
    }

    bool Device::waitForIdle()
    {
        id<MTLCommandBuffer> commandBuffer;
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            commandBuffer = [m_Context.commonQueue commandBuffer];
            if (!commandBuffer)
            {
                m_Context.error("[nvrhi] Failed to allocate a Metal idle marker.");
                return false;
            }
            [commandBuffer commit];
        }
        [commandBuffer waitUntilCompleted];
        std::lock_guard<std::mutex> lock(m_Mutex);
        updateCompletedSubmissions();
        bool success = commandBuffer.status == MTLCommandBufferStatusCompleted;
        for (uint32_t index = 0; index < uint32_t(CommandQueue::Count); ++index)
        {
            success = success && m_Queues[index].firstFailure == 0;
            m_DefaultLifetimeTrackers[index]->collect();
        }
        if (!success)
            m_Context.error("[nvrhi] Metal queue idle wait detected failed GPU work.");
        return success;
    }

    void Device::updateCompletedSubmissions()
    {
        for (QueueState& queue : m_Queues)
        {
            while (!queue.pending.empty())
            {
                const SubmittedCommandBuffer& submitted = queue.pending.front();
                const MTLCommandBufferStatus status = submitted.commandBuffer.status;
                if (status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError)
                    break;
                if (status == MTLCommandBufferStatusError && queue.firstFailure == 0)
                {
                    queue.firstFailure = submitted.instance;
                    const char* message = submitted.commandBuffer.error.localizedDescription.UTF8String;
                    m_Context.error(std::string("[nvrhi] Metal GPU submission failed: ") + (message ? message : "unknown error"));
                }
                if (submitted.lastInBatch)
                    queue.completed = submitted.instance;
                queue.pending.pop_front();
            }
        }
    }

    bool Device::submissionsSucceeded(const std::array<uint64_t, uint32_t(CommandQueue::Count)>& submissions) const
    {
        for (uint32_t index = 0; index < uint32_t(CommandQueue::Count); ++index)
        {
            const QueueState& queue = m_Queues[index];
            if (queue.completed < submissions[index]
                || (queue.firstFailure != 0 && queue.firstFailure <= submissions[index]))
                return false;
        }
        return true;
    }

    void Device::runGarbageCollection()
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        updateCompletedSubmissions();
        for (const auto& tracker : m_DefaultLifetimeTrackers)
            tracker->collect();
    }

    CommandListLifetimeTracker::CommandListLifetimeTracker(Device* device, CommandQueue queue, bool isDefault)
        : m_Device(device), m_Queue(queue), m_IsDefault(isDefault)
    {
    }

    CommandListLifetimeTracker::~CommandListLifetimeTracker()
    {
        if (m_IsDefault || m_CommandBuffers.empty())
            return;
        std::lock_guard<std::mutex> lock(m_Device->m_Mutex);
        auto& retained = m_Device->m_DefaultLifetimeTrackers[uint32_t(m_Queue)]->m_CommandBuffers;
        for (auto& commandBuffer : m_CommandBuffers)
            retained.push_back(std::move(commandBuffer));
    }

    Object CommandListLifetimeTracker::getNativeObject(ObjectType objectType)
    {
        return objectType == ObjectTypes::Nvrhi_Metal3_LifetimeTracker ? Object(this) : Object(nullptr);
    }

    void CommandListLifetimeTracker::collect()
    {
        auto end = std::remove_if(m_CommandBuffers.begin(), m_CommandBuffers.end(), [](const TrackedCommandBuffer& tracked) {
            const MTLCommandBufferStatus status = tracked.commandBuffer.status;
            return status == MTLCommandBufferStatusCompleted || status == MTLCommandBufferStatusError;
        });
        m_CommandBuffers.erase(end, m_CommandBuffers.end());
    }

    void CommandListLifetimeTracker::runGarbageCollection()
    {
        std::lock_guard<std::mutex> lock(m_Device->m_Mutex);
        m_Device->updateCompletedSubmissions();
        collect();
    }

    HeapHandle Device::createHeap(const HeapDesc& d)
    {
        (void)d;
        m_Context.warning("[nvrhi] Metal3 heaps are not implemented; using placed resources is unsupported.");
        return nullptr;
    }

    MemoryRequirements Device::getTextureMemoryRequirements(ITexture* texture)
    {
        MemoryRequirements result{};
        if (texture)
        {
            Texture* t = checked_cast<Texture*>(texture);
            result.size = t->memSize;
            result.alignment = t->memAlign;
        }
        return result;
    }

    bool Device::bindTextureMemory(ITexture* texture, IHeap* heap, uint64_t offset)
    {
        (void)texture; (void)heap; (void)offset;
        return false;
    }

    StagingTextureHandle Device::createStagingTexture(const TextureDesc& d, CpuAccessMode cpuAccess)
    {
        // TODO: stub
    }

    void* Device::mapStagingTexture(IStagingTexture* tex, const TextureSlice& slice, CpuAccessMode cpuAccess, size_t* outRowPitch)
    {
        // TODO: stub
    }

    void Device::unmapStagingTexture(IStagingTexture* tex)
    {
        // TODO: stub
    }

    void Device::getTextureTiling(ITexture* texture, uint32_t* numTiles, PackedMipDesc* desc, TileShape* tileShape, uint32_t* subresourceTilingsNum, SubresourceTiling* subresourceTilings)
    {
        (void)texture;
        (void)numTiles;
        (void)desc;
        (void)tileShape;
        (void)subresourceTilingsNum;
        (void)subresourceTilings;

        utils::NotSupported();
    }

    void Device::updateTextureTileMappings(ITexture* texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings, CommandQueue executionQueue)
    {
        (void)texture; (void)tileMappings; (void)numTileMappings; (void)executionQueue;
        utils::NotSupported();
    }

    SamplerFeedbackTextureHandle Device::createSamplerFeedbackTexture(ITexture* pairedTexture, const SamplerFeedbackTextureDesc& desc)
    {
        (void)pairedTexture;
        (void)desc;

        utils::NotSupported();
        return nullptr;
    }

    SamplerFeedbackTextureHandle Device::createSamplerFeedbackForNativeTexture(ObjectType objectType, Object texture, ITexture* pairedTexture)
    {
        (void)objectType;
        (void)texture;
        (void)pairedTexture;

        utils::NotSupported();
        return nullptr;
    }

    void* Device::mapBuffer(IBuffer* b, CpuAccessMode mapFlags)
    {
        Buffer* buffer = static_cast<Buffer*>(b);
        if (!buffer || !buffer->buffer)
            return nullptr;

        if (mapFlags == CpuAccessMode::None)
        {
            utils::InvalidEnum();
            return nullptr;
        }

        if (buffer->desc.cpuAccess == CpuAccessMode::None)
        {
            m_Context.error("[nvrhi] Cannot map Metal buffer without CPU access.");
            return nullptr;
        }

        if (buffer->desc.cpuAccess != mapFlags)
        {
            m_Context.error("[nvrhi] Metal buffer mapped with incompatible CPU access mode.");
            return nullptr;
        }

        if (buffer->buffer.storageMode == MTLStorageModePrivate)
        {
            m_Context.error("[nvrhi] Cannot map private Metal buffer.");
            return nullptr;
        }

        return [buffer->buffer contents];
    }

    void Device::unmapBuffer(IBuffer* b)
    {
        Buffer* buffer = static_cast<Buffer*>(b);
        if (!buffer || !buffer->buffer)
            return;

        if (buffer->buffer.storageMode == MTLStorageModeManaged &&
            buffer->desc.cpuAccess == CpuAccessMode::Write)
        {
            [buffer->buffer didModifyRange:NSMakeRange(0, NSUInteger(buffer->desc.byteSize))];
        }
    }

    MemoryRequirements Device::getBufferMemoryRequirements(IBuffer* b)
    {
        MemoryRequirements result{};

        Buffer* buffer = static_cast<Buffer*>(b);
        if (!buffer)
            return result;

        const BufferDesc& desc = buffer->desc;
        if (desc.byteSize == 0)
            return result;

        MTLResourceOptions options = convertCpuAccess(desc.cpuAccess);
        MTLSizeAndAlign sizeAndAlign =
            [m_Context.device heapBufferSizeAndAlignWithLength:NSUInteger(desc.byteSize)
                                                    options:options];

        result.size = sizeAndAlign.size;
        result.alignment = sizeAndAlign.align;
        return result;
    }

    bool Device::bindBufferMemory(IBuffer *buffer, IHeap *heap, uint64_t offset)
    {
        (void)buffer;
        (void)heap;
        (void)offset;
        utils::NotSupported();
        return false;
    }

    BufferHandle Device::createHandleForNativeBuffer(ObjectType objectType, Object buffer, const BufferDesc &desc)
    {
        if (!buffer.pointer)
            return nullptr;

        if (objectType != ObjectTypes::MTL3_Buffer)
            return nullptr;

        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer.pointer;
        if (!mtlBuffer)
            return nullptr;

        if (desc.byteSize > 0 && desc.byteSize > [mtlBuffer length])
        {
            m_Context.error("[nvrhi] Native Metal buffer is smaller than BufferDesc::byteSize.");
            return nullptr;
        }
        Buffer *result	   = new Buffer();
        result->desc	   = desc;
        result->buffer	   = (__bridge id<MTLBuffer>)buffer.pointer;
        result->ownsBuffer = false;
        return BufferHandle::Create(result);
    }

    ShaderHandle Device::createShaderSpecialization(IShader *baseShader, const ShaderSpecialization *constants,
												uint32_t numConstants)
    {
        (void)constants;
        (void)numConstants;
        utils::NotSupported();
        return baseShader;
    }
    EventQueryHandle Device::createEventQuery()
    {
        return EventQueryHandle::Create(new EventQuery(this));
    }

    EventQuery* Device::getEventQuery(IEventQuery* query)
    {
        auto* event = query ? static_cast<EventQuery*>(query->getNativeObject(ObjectTypes::Nvrhi_Metal3_EventQuery).pointer) : nullptr;
        if (!event || event->device != this)
        {
            m_Context.error("[nvrhi] Metal event query belongs to a different device or backend.");
            return nullptr;
        }
        return event;
    }

    void Device::setEventQuery(IEventQuery* query, CommandQueue queue)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        if (uint32_t(queue) >= uint32_t(CommandQueue::Count))
        {
            m_Context.error("[nvrhi] Invalid Metal event-query queue.");
            return;
        }
        EventQuery* event = getEventQuery(query);
        if (!event)
            return;
        id<MTLCommandBuffer> commandBuffer = [m_Context.commonQueue commandBuffer];
        if (!commandBuffer)
        {
            m_Context.error("[nvrhi] Failed to allocate a Metal event marker.");
            return;
        }
        event->commandBuffer = commandBuffer;
        event->failureReported = false;
        for (uint32_t index = 0; index < uint32_t(CommandQueue::Count); ++index)
            event->submissions[index] = m_Queues[index].submitted;
        [commandBuffer commit];
    }

    bool Device::pollEventQuery(IEventQuery* query)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        EventQuery* event = getEventQuery(query);
        if (!event)
            return false;
        if (!event->commandBuffer)
            return true;
        updateCompletedSubmissions();
        if (event->commandBuffer.status == MTLCommandBufferStatusError && !event->failureReported)
        {
            event->failureReported = true;
            m_Context.error("[nvrhi] Metal event marker failed.");
        }
        return event->commandBuffer.status == MTLCommandBufferStatusCompleted
            && submissionsSucceeded(event->submissions);
    }

    void Device::waitEventQuery(IEventQuery* query)
    {
        id<MTLCommandBuffer> commandBuffer;
        std::array<uint64_t, uint32_t(CommandQueue::Count)> submissions;
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            EventQuery* event = getEventQuery(query);
            if (!event || !event->commandBuffer)
                return;
            commandBuffer = event->commandBuffer;
            submissions = event->submissions;
        }
        [commandBuffer waitUntilCompleted];
        std::lock_guard<std::mutex> lock(m_Mutex);
        updateCompletedSubmissions();
        if (commandBuffer.status != MTLCommandBufferStatusCompleted || !submissionsSucceeded(submissions))
            m_Context.error("[nvrhi] Metal event wait detected failed GPU work.");
    }

    void Device::resetEventQuery(IEventQuery* query)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        EventQuery* event = getEventQuery(query);
        if (!event)
            return;
        event->commandBuffer = nil;
        event->submissions.fill(0);
        event->failureReported = false;
    }

    bool Device::queryFeatureSupport(Feature feature, void* pInfo, size_t infoSize)
    {
        (void)pInfo; (void)infoSize;
        switch (feature)
        {
        case Feature::ComputeQueue:
        case Feature::CopyQueue:
        case Feature::ConstantBufferRanges:
            return true;
        default:
            return false;
        }
    }
    MTLPixelFormat convertFormat(nvrhi::Format format)
    {
        switch (format)
        {
        case Format::RGBA8_UNORM:
            return MTLPixelFormatRGBA8Unorm;
        case Format::BGRA8_UNORM:
            return MTLPixelFormatBGRA8Unorm;
        case Format::SRGBA8_UNORM:
            return MTLPixelFormatRGBA8Unorm_sRGB;
        case Format::SBGRA8_UNORM:
            return MTLPixelFormatBGRA8Unorm_sRGB;
        case Format::R8_UNORM:
            return MTLPixelFormatR8Unorm;
        case Format::RG8_UNORM:
            return MTLPixelFormatRG8Unorm;
        case Format::R32_UINT:
            return MTLPixelFormatR32Uint;
        case Format::R16_FLOAT:
            return MTLPixelFormatR16Float;
        case Format::RG16_FLOAT:
            return MTLPixelFormatRG16Float;
        case Format::RGBA16_FLOAT:
            return MTLPixelFormatRGBA16Float;
        case Format::RGBA16_UNORM:
            return MTLPixelFormatRGBA16Unorm;
        case Format::R32_FLOAT:
            return MTLPixelFormatR32Float;
        case Format::RG32_FLOAT:
            return MTLPixelFormatRG32Float;
        case Format::RGBA32_FLOAT:
            return MTLPixelFormatRGBA32Float;
        case Format::BC1_UNORM:
            return MTLPixelFormatBC1_RGBA;
        case Format::BC1_UNORM_SRGB:
            return MTLPixelFormatBC1_RGBA_sRGB;
        case Format::BC2_UNORM:
            return MTLPixelFormatBC2_RGBA;
        case Format::BC2_UNORM_SRGB:
            return MTLPixelFormatBC2_RGBA_sRGB;
        case Format::BC3_UNORM:
            return MTLPixelFormatBC3_RGBA;
        case Format::BC3_UNORM_SRGB:
            return MTLPixelFormatBC3_RGBA_sRGB;
        case Format::BC4_UNORM:
            return MTLPixelFormatBC4_RUnorm;
        case Format::BC4_SNORM:
            return MTLPixelFormatBC4_RSnorm;
        case Format::BC5_UNORM:
            return MTLPixelFormatBC5_RGUnorm;
        case Format::BC5_SNORM:
            return MTLPixelFormatBC5_RGSnorm;
        case Format::D32:
            return MTLPixelFormatDepth32Float;
        case Format::D16:
            return MTLPixelFormatDepth16Unorm;
        case Format::D24S8:
            return MTLPixelFormatDepth24Unorm_Stencil8;
        case Format::D32S8:
            return MTLPixelFormatDepth32Float_Stencil8;
        default:
            return MTLPixelFormatInvalid;
        }
    }

    // ---- stubs ----
    TimerQueryHandle Device::createTimerQuery()
    {
        return TimerQueryHandle::Create(new TimerQuery());
    }

    bool Device::pollTimerQuery(ITimerQuery* query)
    {
        auto* timer = static_cast<TimerQuery*>(query);
        return timer && timer->resolved;
    }

    float Device::getTimerQueryTime(ITimerQuery* query)
    {
        auto* timer = static_cast<TimerQuery*>(query);
        return timer ? timer->time : 0.f;
    }

    void Device::resetTimerQuery(ITimerQuery* query)
    {
        auto* timer = static_cast<TimerQuery*>(query);
        if (!timer) return;
        timer->resolved = true;
        timer->time = 0.f;
    }

    MeshletPipelineHandle Device::createMeshletPipeline(const MeshletPipelineDesc& desc, FramebufferInfo const& fbinfo)
    {
        (void)desc;
        (void)fbinfo;
        return nullptr;
    }

    MeshletPipelineHandle Device::createMeshletPipeline(const MeshletPipelineDesc& desc, IFramebuffer* fb)
    {
        (void)desc;
        (void)fb;
        return nullptr;
    }

    BindingLayoutHandle Device::createBindlessLayout(const BindlessLayoutDesc& desc)
    {
        (void)desc;
        return nullptr;
    }

    DescriptorTableHandle Device::createDescriptorTable(IBindingLayout* layout)
    {
        (void)layout;
        return nullptr;
    }

    void Device::resizeDescriptorTable(IDescriptorTable* descriptorTable, uint32_t newSize, bool keepContents)
    {
        (void)descriptorTable;
        (void)newSize;
        (void)keepContents;
    }

    bool Device::writeDescriptorTable(IDescriptorTable* descriptorTable, const BindingSetItem& item)
    {
        (void)descriptorTable;
        (void)item;
        return false;
    }

    rt::OpacityMicromapHandle Device::createOpacityMicromap(const rt::OpacityMicromapDesc& desc)
    {
        auto* omm = new DummyOpacityMicromap();
        omm->desc = desc;
        return rt::OpacityMicromapHandle::Create(omm);
    }

    rt::AccelStructHandle Device::createAccelStruct(const rt::AccelStructDesc& desc)
    {
        auto* accel = new DummyAccelStruct();
        accel->desc = desc;
        return rt::AccelStructHandle::Create(accel);
    }

    MemoryRequirements Device::getAccelStructMemoryRequirements(rt::IAccelStruct* as)
    {
        (void)as;
        return MemoryRequirements{};
    }

    rt::cluster::OperationSizeInfo Device::getClusterOperationSizeInfo(const rt::cluster::OperationParams& params)
    {
        (void)params;
        return rt::cluster::OperationSizeInfo{};
    }

    bool Device::bindAccelStructMemory(rt::IAccelStruct* as, IHeap* heap, uint64_t offset)
    {
        (void)as;
        (void)heap;
        (void)offset;
        return false;
    }

    nvrhi::CommandListHandle Device::createCommandList(const CommandListParameters& params)
    {
        if (uint32_t(params.queueType) >= uint32_t(CommandQueue::Count))
        {
            m_Context.error("[nvrhi] Invalid Metal command-list queue type.");
            return nullptr;
        }
        if (params.lifetimeTracker)
        {
            auto* tracker = static_cast<CommandListLifetimeTracker*>(
                params.lifetimeTracker->getNativeObject(ObjectTypes::Nvrhi_Metal3_LifetimeTracker).pointer);
            if (!tracker || tracker->m_Device != this || tracker->m_Queue != params.queueType)
            {
                m_Context.error("[nvrhi] Metal lifetime tracker belongs to a different device, queue, or backend.");
                return nullptr;
            }
        }
        return nvrhi::CommandListHandle::Create(new CommandList(this, m_Context, params));
    }

    uint64_t Device::executeCommandLists(nvrhi::ICommandList* const* pCommandLists, size_t numCommandLists, CommandQueue executionQueue)
    {
        if (numCommandLists == 0)
            return 0;
        if (!pCommandLists || uint32_t(executionQueue) >= uint32_t(CommandQueue::Count))
        {
            m_Context.error("[nvrhi] Invalid Metal command-list submission array or queue.");
            return 0;
        }

        std::lock_guard<std::mutex> lock(m_Mutex);
        size_t prepared = 0;
        for (; prepared < numCommandLists; ++prepared)
        {
            nvrhi::ICommandList* list = pCommandLists[prepared];
            auto* commandList = list ? static_cast<CommandList*>(
                list->getNativeObject(ObjectTypes::Nvrhi_Metal3_CommandList).pointer) : nullptr;
            if (!commandList || commandList->m_Device != this
                || commandList->m_Desc.queueType != executionQueue
                || commandList->m_RecordingState != CommandList::RecordingState::Closed
                || commandList->trackedCmdBuffer.status != MTLCommandBufferStatusNotEnqueued)
                break;
            commandList->m_RecordingState = CommandList::RecordingState::PendingSubmission;
        }

        if (prepared != numCommandLists)
        {
            for (size_t index = 0; index < prepared; ++index)
            {
                auto* commandList = static_cast<CommandList*>(
                    pCommandLists[index]->getNativeObject(ObjectTypes::Nvrhi_Metal3_CommandList).pointer);
                commandList->m_RecordingState = CommandList::RecordingState::Closed;
            }
            m_Context.error("[nvrhi] Metal submission requires unique, closed, unsubmitted command lists from this device and queue.");
            return 0;
        }

        QueueState& queue = m_Queues[uint32_t(executionQueue)];
        const uint64_t instance = ++queue.submitted;
        for (size_t index = 0; index < numCommandLists; ++index)
        {
            auto* commandList = static_cast<CommandList*>(
                pCommandLists[index]->getNativeObject(ObjectTypes::Nvrhi_Metal3_CommandList).pointer);
            queue.pending.push_back({commandList->trackedCmdBuffer, instance, index + 1 == numCommandLists});
            commandList->submit();
        }
        return instance;
    }

    void Device::queueWaitForCommandList(CommandQueue waitQueue, CommandQueue executionQueue, uint64_t instance)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        if (uint32_t(waitQueue) >= uint32_t(CommandQueue::Count)
            || uint32_t(executionQueue) >= uint32_t(CommandQueue::Count))
        {
            m_Context.error("[nvrhi] Invalid Metal dependency queue.");
            return;
        }
        updateCompletedSubmissions();
        const QueueState& producer = m_Queues[uint32_t(executionQueue)];
        if (instance > producer.submitted)
            m_Context.error("[nvrhi] Metal queue dependency references an unsubmitted execution identifier.");
        else if (producer.firstFailure != 0 && producer.firstFailure <= instance)
            m_Context.error("[nvrhi] Metal queue dependency references failed GPU work.");
    }

    CommandListLifetimeTrackerHandle Device::createCommandListLifetimeTracker(CommandQueue executionQueue)
    {
        if (uint32_t(executionQueue) >= uint32_t(CommandQueue::Count))
        {
            m_Context.error("[nvrhi] Invalid Metal lifetime-tracker queue.");
            return nullptr;
        }
        return CommandListLifetimeTrackerHandle::Create(new CommandListLifetimeTracker(this, executionQueue));
    }

    FormatSupport Device::queryFormatSupport(Format format)
    {
        if (convertFormat(format) == MTLPixelFormatInvalid)
            return FormatSupport::None;

        const FormatInfo& info = getFormatInfo(format);
        FormatSupport support = FormatSupport::Texture | FormatSupport::ShaderLoad | FormatSupport::ShaderSample;
        if (info.hasDepth || info.hasStencil)
            support = support | FormatSupport::DepthStencil;
        else
            support = support | FormatSupport::RenderTarget | FormatSupport::ShaderUavLoad | FormatSupport::ShaderUavStore;
        return support;
    }

    coopvec::DeviceFeatures Device::queryCoopVecFeatures()
    {
        return coopvec::DeviceFeatures{};
    }

    coopvec::MatMulFormatSupport Device::queryCoopVecMatMulFormatSupport(const coopvec::MatMulFormatCombo& combination)
    {
        (void)combination;
        return coopvec::MatMulFormatSupport{};
    }

    coopvec::TrainingFormatSupport Device::queryCoopVecTrainingFormatSupport(coopvec::DataType componentType)
    {
        (void)componentType;
        return coopvec::TrainingFormatSupport{};
    }

    size_t Device::getCoopVecMatrixSize(coopvec::DataType type, coopvec::MatrixLayout layout, int rows, int columns)
    {
        (void)type;
        (void)layout;
        (void)rows;
        (void)columns;
        return 0;
    }
}
