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
    }

    Device::~Device() = default;

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
    // placeholders
    Object Device::getNativeQueue(ObjectType objectType, CommandQueue queue)
    {
        (void)queue;
        if (objectType == ObjectTypes::MTL3_CommandQueue)
            return Object((__bridge void*)m_Context.commonQueue);
        return nullptr;
    }

    bool Device::waitForIdle()
    {
        id<MTLCommandBuffer> commandBuffer = [m_Context.commonQueue commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        return true;
    }

    void Device::runGarbageCollection()
    {
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
        return EventQueryHandle::Create(new EventQuery());
    }

    void Device::setEventQuery(IEventQuery* query, CommandQueue queue)
    {
        (void)queue;

        EventQuery* event = static_cast<EventQuery*>(query);
        if (!event || !event->semaphore || !m_Context.commonQueue)
            return;

        event->signaled.store(false, std::memory_order_release);

        event->AddRef();

        // The renderer uses event queries as per-frame fences. Put a marker
        // command buffer after the already-submitted frame work on the same
        // Metal queue; when this marker completes, all earlier frame commands
        // are complete too and the frame slot can be reused.

        id<MTLCommandBuffer> commandBuffer = [m_Context.commonQueue commandBuffer];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
            event->signaled.store(true, std::memory_order_release);
            dispatch_semaphore_signal(event->semaphore);
            event->Release();
        }];
        [commandBuffer commit];
    }

    bool Device::pollEventQuery(IEventQuery* query)
    {
        EventQuery* event = static_cast<EventQuery*>(query);
        return event && event->signaled.load(std::memory_order_acquire);
    }

    void Device::waitEventQuery(IEventQuery* query)
    {
        EventQuery* event = static_cast<EventQuery*>(query);
        if (!event || !event->semaphore)
            return;

        if (!event->signaled.load(std::memory_order_acquire))
            dispatch_semaphore_wait(event->semaphore, DISPATCH_TIME_FOREVER);

        event->signaled.store(true, std::memory_order_release);
    }

    void Device::resetEventQuery(IEventQuery* query)
    {
        EventQuery* event = static_cast<EventQuery*>(query);
        if (!event || !event->semaphore)
            return;

        while (dispatch_semaphore_wait(event->semaphore, DISPATCH_TIME_NOW) == 0) {}
        event->signaled.store(false, std::memory_order_release);
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
        return nvrhi::CommandListHandle::Create(new CommandList(this, m_Context, params));
    }

    uint64_t Device::executeCommandLists(nvrhi::ICommandList* const* pCommandLists, size_t numCommandLists, CommandQueue executionQueue)
    {
        (void)pCommandLists;
        (void)numCommandLists;
        (void)executionQueue;
        static std::atomic<uint64_t> serial{1};
        return serial.fetch_add(1, std::memory_order_relaxed);
    }

    void Device::queueWaitForCommandList(CommandQueue waitQueue, CommandQueue executionQueue, uint64_t instance)
    {
        (void)waitQueue;
        (void)executionQueue;
        (void)instance;
    }

    CommandListLifetimeTrackerHandle Device::createCommandListLifetimeTracker(CommandQueue executionQueue)
    {
        (void)executionQueue;
        return nullptr;
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
