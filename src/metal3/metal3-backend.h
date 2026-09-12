#pragma once

#include <mutex>

#include "nvrhi/common/aftermath.h"
#include "nvrhi/common/resource.h"
#include "nvrhi/nvrhi.h"
#include <nvrhi/metal3.h>
#include <nvrhi/utils.h>
#include <dispatch/dispatch.h>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct IRDescriptorTableEntry;

#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
#include <tracy/TracyMetal.hmm>
#endif

namespace nvrhi::metal3 
{
    class Texture;
    class Buffer;
    class Shader;
    class Sampler;
    class InputLayout;
    class Framebuffer;
    class GraphicsPipeline;
    class ComputePipeline;
    class BindingLayout;
    class BindingSet;
    class MeshletPipeline;
    class RayTracingPipeline;
    class EventQuery;
    class TimerQuery;
    // dummy classes for non essential or planned for future features
    // Not planned
    class DummyOpacityMicromap;
    // TODO: for future when RT backend will be added
    class DummyAccelStruct;

    // useful for shader reflection
    enum class MscArgumentType : uint8_t
    {
        SRV,
        UAV,
        CBV,
        Sampler
    };

    struct MscArgumentBinding
    {
        uint32_t index = 0;
        uint32_t slot = 0;
        uint32_t space = 0;
        MscArgumentType type = MscArgumentType::SRV;
    };

    struct MscShaderReflection
    {
        bool valid = false;
        bool needsFunctionConstants = false;
        uint32_t resourceCount = 0;
        uint32_t vertexOutputSizeInBytes = 0;
        uint32_t maxInputPrimitivesPerMeshThreadgroup = 0;
        uint32_t geometryInstanceCount = 1;
        std::string shaderType;
        std::string inputPrimitive;
        std::vector<MscArgumentBinding> topLevelArgumentBuffer;
        std::unordered_map<std::string, uint32_t> vertexInputAttributes;
    };
    // shader reflected data, first part basically the MscArgumentBinding i,e per resource
    struct MetalBindingPlanEntry
    {
        uint32_t argumentIndex = 0;
        uint32_t slot = 0;
        uint32_t space = 0;
        MscArgumentType argumentType = MscArgumentType::SRV;

        uint32_t layoutIndex = ~0u;
        uint32_t layoutItemIndex = ~0u;
        ResourceType layoutType = ResourceType::None;
        bool layoutMatched = false;
    };
    // per shader stage binding plan; entries -> per resource plan
    struct MetalStageBindingPlan
    {
        ShaderType stage = ShaderType::None;
        std::string debugName;
        bool valid = false;
        uint32_t resourceCount = 0;
        std::vector<MetalBindingPlanEntry> entries;
    };

    struct MetalArgumentTableCacheKey
    {
        const MetalStageBindingPlan* plan = nullptr;
        std::vector<const BindingSet*> bindingSets;
        std::vector<uint64_t> bindingSetVersions;

        bool operator==(const MetalArgumentTableCacheKey& other) const
        {
            return plan == other.plan &&
                bindingSets == other.bindingSets &&
                bindingSetVersions == other.bindingSetVersions;
        }
    };

    struct ArgumentTableAllocation
    {
        id<MTLBuffer> buffer = nil;
        NSUInteger offset = 0;
        void* cpuAddress = nullptr;
    };

    struct MetalArgumentTableCacheEntry
    {
        MetalArgumentTableCacheKey key;
        ArgumentTableAllocation allocation;
    };

    // implementation in metal3-constants.cpp
    MTLTextureType convertTextureDimension(TextureDimension dimension, uint32_t sampleCount);
    MTLResourceOptions convertCpuAccess(CpuAccessMode cpuAccess);
    MTLVertexFormat convertVertexFormat(Format format);
    MTLIndexType convertIndexFormat(Format format);
    MTLPrimitiveType convertPrimitiveType(PrimitiveType primitiveType);
    MTLCullMode convertCullMode(RasterCullMode cullMode);
    MTLWinding convertWinding(bool frontCounterClockwise);
    MTLCompareFunction convertCompareFunction(ComparisonFunc func);
    MTLSamplerAddressMode convertSamplerAddressMode(SamplerAddressMode mode);
    MTLBlendFactor convertBlendFactor(BlendFactor factor);
    MTLBlendOperation convertBlendOp(BlendOp op);

    // metal 3 context
    struct MTL3Context
    {
        id<MTLDevice> device;
        id<MTLCommandQueue> commonQueue;

        bool logBufferLifetime = false;
        IMessageCallback* messageCallback = nullptr;
        void error(const std::string& message) const;
        void warning(const std::string& message) const;
        void info(const std::string& message) const;
    };
    // created at time of shader creation, using reflection data
    MetalStageBindingPlan createMetalStageBindingPlan(ShaderType stage, const MscShaderReflection& reflection);
    // resolve at pipeline creation per stage
    MetalStageBindingPlan resolveMetalStageBindingPlan(const MetalStageBindingPlan& reflectedPlan, const BindingLayoutVector& pipelineLayouts);
    
    // basically a little allocator for temp CPU written, GPU read upload memory
    struct UploadAllocation
    {
        id<MTLBuffer> buffer = nil;
        NSUInteger offset = 0;
        void* cpuAddress = nullptr;
    };

    struct VolatileBufferAllocation
    {
        UploadAllocation allocation;
        size_t writtenSize = 0;
    };

    // metal resource cache per nvrhi binding item
    struct MetalBindingResource
    {
        ResourceType type = ResourceType::None;
        uint32_t slot = 0;
        uint32_t arrayElement = 0;
        uint32_t registerSpace = 0;

        ResourceHandle resource;

        id<MTLTexture> texture = nil;
        id<MTLBuffer> buffer = nil;
        NSUInteger bufferOffset = 0;
        NSUInteger bufferSize = 0;
        id<MTLSamplerState> sampler = nil;
        float samplerMipBias = 0.f;

        MTLResourceUsage usage = MTLResourceUsageRead;
    };

    class UploadManager
    {
    public:
        UploadManager(const MTL3Context& context, size_t uploadChunkSize, size_t scratchMaxMem,
            bool isScratchBuffer, size_t minimumChunkSize = 4 * 1024 * 1024,
            size_t initialChunkCount = 0);
        void beginCommandBuffer();
        void submitCommandBuffer(id<MTLCommandBuffer> commandBuffer);
        UploadAllocation suballocate(size_t size, size_t alignment);
        size_t getChunkCount() const { return m_Chunks.size(); }

    private:
        struct Chunk
        {
            id<MTLBuffer> buffer = nil;
            uint8_t* cpuAddress = nullptr;
            size_t size = 0;
            size_t offset = 0;
            uint64_t lastUsedSerial = 0;
        };

        const MTL3Context& m_Context;
        size_t m_DefaultChunkSize = 0;
        uint64_t m_SubmittedSerial = 0;
        uint64_t m_ActiveSerial = 0;
        size_t m_CurrentChunk = size_t(-1);
        std::shared_ptr<std::atomic<uint64_t>> m_CompletedSerial;
        std::vector<Chunk> m_Chunks;

        Chunk* findOrCreateChunk(size_t size, size_t alignment);
    };

    struct TransientBufferAllocation
    {
        id<MTLBuffer> buffer = nil;
        NSUInteger offset = 0;
        NSUInteger size = 0;
        // null for private-storage pages
        void* cpuAddress = nullptr;
    };

    // Per-command-buffer temporary storage for indirect draw helpers. A slot is
    // reused only after Metal reports completion for the command buffer that
    // used it, so page contents and ICB commands stay valid for the GPU.
    class TransientIndirectResourcePool
    {
    public:
        struct Stats
        {
            size_t sharedPageCount = 0;
            size_t privatePageCount = 0;
            size_t sharedPagesCreated = 0;
            size_t privatePagesCreated = 0;
            size_t bytesReserved = 0;
            size_t indirectCommandBuffersReused = 0;
            size_t indirectCommandBuffersCreated = 0;
            bool usedOverflowSlot = false;
        };

        explicit TransientIndirectResourcePool(const MTL3Context& context);
        void beginCommandBuffer();
        void submitCommandBuffer(id<MTLCommandBuffer> commandBuffer);
        Stats getActiveStats() const;
        TransientBufferAllocation allocateShared(NSUInteger size, NSUInteger alignment = 256);
        TransientBufferAllocation allocatePrivate(NSUInteger size, NSUInteger alignment = 256);
        id<MTLIndirectCommandBuffer> acquireIndirectCommandBuffer(
            MTLIndirectCommandBufferDescriptor* descriptor, NSUInteger requiredCapacity,
            bool* wasCreated = nullptr, NSUInteger* capacity = nullptr);

    private:
        struct BufferPage
        {
            id<MTLBuffer> buffer = nil;
            NSUInteger capacity = 0;
            NSUInteger writeOffset = 0;
            uint8_t* cpuAddress = nullptr;  // only for shared pages
        };

        struct IndirectCommandBufferKey
        {
            // every descriptor property that changes ICB compatibility:
            MTLIndirectCommandType commandTypes{};
            bool inheritPipelineState = false;
            bool inheritBuffers = false;
            NSUInteger maxVertexBufferBindCount = 0;
            NSUInteger maxFragmentBufferBindCount = 0;
            NSUInteger maxObjectBufferBindCount = 0;
            NSUInteger maxMeshBufferBindCount = 0;

            bool operator==(const IndirectCommandBufferKey& other) const;
        };

        struct PooledIndirectCommandBuffer
        {
            IndirectCommandBufferKey key;
            NSUInteger capacity = 0;    // maxCommandCount used at creation
            id<MTLIndirectCommandBuffer> commandBuffer = nil;
        };

        struct FrameSlot
        {
            bool inFlight = false;
            bool isOverflow = false;
            Stats stats;
            std::vector<BufferPage> sharedPages;
            std::vector<BufferPage> privatePages;
            std::vector<PooledIndirectCommandBuffer> indirectCommandBuffers;
        };

        struct State
        {
            std::mutex mutex;
            std::vector<FrameSlot> slots;
        };

        const MTL3Context& m_Context;
        // Completion handlers retain State, not CommandList, so a submitted
        // command buffer cannot update a destroyed CommandList instance.
        std::shared_ptr<State> m_State;
        size_t m_ActiveSlot = size_t(-1);

        TransientBufferAllocation allocate(NSUInteger size, NSUInteger alignment,
            MTLResourceOptions options, std::vector<BufferPage> FrameSlot::*pages);
    };

    class Texture : public RefCounter<ITexture>
    {
    public:
        TextureDesc desc;
        id<MTLTexture> texture = nil;
        NSUInteger memSize;
        NSUInteger memAlign;
        bool ownsTexture = true;

        const TextureDesc& getDesc() const override { return desc; }
        Object getNativeObject(ObjectType objectType) override;
        Object getNativeView(ObjectType objectType, Format format = Format::UNKNOWN, TextureSubresourceSet subresources = AllSubresources, TextureDimension dimension = TextureDimension::Unknown, bool isReadOnlyDSV = false) override;
    };

    //TODO: stub
    class StagingTexture : public RefCounter<IStagingTexture>
    {
    public:
        TextureDesc desc;
        const TextureDesc& getDesc() const override { return desc; }
    };

    class Buffer : public RefCounter<IBuffer>
    {
    public:
        BufferDesc desc;
        id<MTLBuffer> buffer = nil;
        bool ownsBuffer = true;

        const BufferDesc& getDesc() const override { return desc; }
        GpuVirtualAddress getGpuVirtualAddress() const override { return buffer ? buffer.gpuAddress : 0; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class Shader : public RefCounter<IShader>
    {
    public:
        ShaderDesc desc;
        id<MTLLibrary> library = nil;
        id<MTLLibrary> stageInLibrary = nil;
        id<MTLFunction> function = nil;
        std::vector<uint8_t> bytecode;
        MscShaderReflection mscReflection;
        MetalStageBindingPlan reflectedBindingPlan;
        MTLSize computeThreadsPerGroup = MTLSizeMake(1, 1, 1);
        bool computeThreadsPerGroupValid = false;

        const ShaderDesc& getDesc() const override { return desc; }
        void getBytecode(const void** ppBytecode, size_t* pSize) const override;
    };

    class ShaderLibrary : public RefCounter<IShaderLibrary>
    {
    public:
        id<MTLLibrary> library = nil;
        std::vector<uint8_t> bytecode;

        void getBytecode(const void** ppBytecode, size_t* pSize) const override;
        ShaderHandle getShader(const char* entryName, ShaderType shaderType) override;
    };

    class Sampler : public RefCounter<ISampler>
    {
    public:
        SamplerDesc desc;
        id<MTLSamplerState> sampler = nil;
        const SamplerDesc& getDesc() const override { return desc; }
    };

    class InputLayout : public RefCounter<IInputLayout>
    {
    public:
        std::vector<VertexAttributeDesc> attributes;
        MTLVertexDescriptor* vertexDescriptor = nil;
        uint32_t getNumAttributes() const override { return uint32_t(attributes.size()); }
        const VertexAttributeDesc* getAttributeDesc(uint32_t index) const override;
    };

    class Framebuffer : public RefCounter<IFramebuffer>
    {
    public:
        FramebufferDesc desc;
        FramebufferInfoEx framebufferInfo;
        const FramebufferDesc& getDesc() const override { return desc; }
        const FramebufferInfoEx& getFramebufferInfo() const override { return framebufferInfo; }
    };

    class GraphicsPipeline : public RefCounter<IGraphicsPipeline>
    {
    public:
        GraphicsPipelineDesc desc;
        FramebufferInfo framebufferInfo;
        id<MTLRenderPipelineState> pipeline = nil;
        id<MTLDepthStencilState> depthStencilState = nil;
        MTLPrimitiveType primitiveType = MTLPrimitiveTypeTriangle;
        MTLCullMode cullMode = MTLCullModeNone;
        MTLWinding frontWinding = MTLWindingClockwise;
        bool usesGeometryEmulation = false;
        uint32_t geometryVertexSizeInBytes = 0;
        uint32_t geometryMaxInputPrimitivesPerMeshThreadgroup = 0;
        uint32_t geometryInstanceCount = 1;
        MetalStageBindingPlan vertexBindingPlan;
        // binding plan for object stage
        // for emulated geometry pipeline, the pipeline transforms to:
        // VS (object stage) -> GS -> PS
        MetalStageBindingPlan objectBindingPlan;
        MetalStageBindingPlan fragmentBindingPlan;
        MetalStageBindingPlan meshBindingPlan;
        const GraphicsPipelineDesc& getDesc() const override { return desc; }
        const FramebufferInfo& getFramebufferInfo() const override { return framebufferInfo; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class ComputePipeline : public RefCounter<IComputePipeline>
    {
    public:
        ComputePipelineDesc desc;
        id<MTLComputePipelineState> pipeline = nil;
        MTLSize threadsPerGroup = MTLSizeMake(1, 1, 1);
        MetalStageBindingPlan computeBindingPlan;
        const ComputePipelineDesc& getDesc() const override { return desc; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class BindingLayout : public RefCounter<IBindingLayout>
    {
    public:
        BindingLayoutDesc desc;
        BindlessLayoutDesc bindlessDesc;
        bool isBindless = false;
        const BindingLayoutDesc* getDesc() const override { return isBindless ? nullptr : &desc; }
        const BindlessLayoutDesc* getBindlessDesc() const override { return isBindless ? &bindlessDesc : nullptr; }
    };

    class BindingSet : public RefCounter<IBindingSet>
    {
    public:
        BindingSetDesc desc;
        BindingLayoutHandle layout;
        std::vector<ResourceHandle> resources;
        std::vector<MetalBindingResource> entries;
        uint64_t version = 1;
        const BindingSetDesc* getDesc() const override { return &desc; }
        IBindingLayout* getLayout() const override { return layout; }
    };

    class EventQuery : public RefCounter<IEventQuery>
    {
    public:
        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
        std::atomic<bool> signaled{ true };
    };
    // TODO: stubs
    class TimerQuery : public RefCounter<ITimerQuery> { public: bool resolved = true; float time = 0.f; };

    class ShaderTable : public RefCounter<rt::IShaderTable>
    {
    public:
        rt::ShaderTableDesc desc;
        rt::IPipeline* pipeline = nullptr;
        uint32_t numEntries = 0;
        rt::ShaderTableDesc const& getDesc() const override { return desc; }
        uint32_t getNumEntries() const override { return numEntries; }
        rt::IPipeline* getPipeline() const override { return pipeline; }
        void setRayGenerationShader(const char* exportName, IBindingSet* bindings = nullptr) override { (void)exportName; (void)bindings; numEntries = std::max(numEntries, 1u); }
        int addMissShader(const char* exportName, IBindingSet* bindings = nullptr) override { (void)exportName; (void)bindings; return int(numEntries++); }
        int addHitGroup(const char* exportName, IBindingSet* bindings = nullptr) override { (void)exportName; (void)bindings; return int(numEntries++); }
        int addCallableShader(const char* exportName, IBindingSet* bindings = nullptr) override { (void)exportName; (void)bindings; return int(numEntries++); }
        void clearMissShaders() override {}
        void clearHitShaders() override {}
        void clearCallableShaders() override {}
    };

    class RayTracingPipeline : public RefCounter<rt::IPipeline>
    {
    public:
        rt::PipelineDesc desc;
        const rt::PipelineDesc& getDesc() const override { return desc; }
        rt::ShaderTableHandle createShaderTable(rt::ShaderTableDesc const& tableDesc = rt::ShaderTableDesc()) override
        {
            ShaderTable* table = new ShaderTable();
            table->desc = tableDesc;
            table->pipeline = this;
            return rt::ShaderTableHandle::Create(table);
        }
    };

    class DummyOpacityMicromap : public RefCounter<rt::IOpacityMicromap>
    {
    public:
        rt::OpacityMicromapDesc desc;
        const rt::OpacityMicromapDesc& getDesc() const override { return desc; }
        bool isCompacted() const override { return false; }
        uint64_t getDeviceAddress() const override { return 0; }
    };

    class DummyAccelStruct : public RefCounter<rt::IAccelStruct>
    {
    public:
        rt::AccelStructDesc desc;
        const rt::AccelStructDesc& getDesc() const override { return desc; }
        bool isCompacted() const override { return false; }
        uint64_t getDeviceAddress() const override { return 0; }
    };

    // commandList impl
    class CommandList final : public RefCounter<nvrhi::metal3::ICommandList>
    {
    public:

        // TODO: Internal interface functions (metal 3 specific)
        CommandList(class Device* device, const MTL3Context& context, const CommandListParameters& params);
        ~CommandList() override;

        // IResource implementation
        Object getNativeObject(ObjectType objectType) override;

        // ICommandList implementation
        void open() override;
        void close() override;
        void clearState() override;
        
        void clearTextureFloat(ITexture* t, TextureSubresourceSet subresources, const Color& clearColor) override;
        void clearDepthStencilTexture(ITexture* t, TextureSubresourceSet subresources, bool clearDepth, float depth, bool clearStencil, uint8_t stencil) override;
        void clearTextureUInt(ITexture* t, TextureSubresourceSet subresources, uint32_t clearColor) override;
        void clearSamplerFeedbackTexture(ISamplerFeedbackTexture* texture) override;
        void decodeSamplerFeedbackTexture(IBuffer* buffer, ISamplerFeedbackTexture* texture, Format format) override;
        void setSamplerFeedbackTextureState(ISamplerFeedbackTexture* texture, ResourceStates stateBits) override;

        void copyTexture(ITexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice) override;
        //// staging texture path DNE ////
        void copyTexture(IStagingTexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice) override;
        void copyTexture(ITexture* dest, const TextureSlice& destSlice, IStagingTexture* src, const TextureSlice& srcSlice) override;
        //// staging texture path DNE /////
        void writeTexture(ITexture* dest, uint32_t arraySlice, uint32_t mipLevel, const void* data, size_t rowPitch, size_t depthPitch) override;
        void resolveTexture(ITexture* dest, const TextureSubresourceSet& dstSubresources, ITexture* src, const TextureSubresourceSet& srcSubresources) override;

        void writeBuffer(IBuffer* b, const void* data, size_t dataSize, uint64_t destOffsetBytes = 0) override;
        void clearBufferUInt(IBuffer* b, uint32_t clearValue) override;
        void copyBuffer(IBuffer* dest, uint64_t destOffsetBytes, IBuffer* src, uint64_t srcOffsetBytes, uint64_t dataSizeBytes) override;

        void setPushConstants(const void* data, size_t byteSize) override;

        void setGraphicsState(const GraphicsState& state) override;
        // for draw* commands:
        /* 
        // Metal Shader Converter vertex shaders read draw parameters from
        // the IR runtime bind points. Use the runtime wrapper for indirect
        // draws too; raw Metal draws do not populate those translated
        // D3D-style draw parameter bindings.
        */
        void draw(const DrawArguments& args) override;
        void drawIndexed(const DrawArguments& args) override;

        // supplying with draw count, just loops over the draw indices and issue
        // the draw calls one at a time, cpu side
        void drawIndirect(uint32_t offsetBytes, uint32_t drawCount) override;
        void drawIndexedIndirect(uint32_t offsetBytes, uint32_t drawCount) override;

        void drawIndexedIndirectCount(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount) override;

        void setComputeState(const ComputeState& state) override;
        void dispatch(uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1) override;
        void dispatchIndirect(uint32_t offsetBytes) override;

        void setMeshletState(const MeshletState& state) override;
        void dispatchMesh(uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1) override;
        void dispatchMeshIndirect(uint32_t offsetBytes, uint32_t maxDrawCount) override;
        void dispatchMeshIndirectCount(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount) override;

        void setRayTracingState(const rt::State& state) override;
        void dispatchRays(const rt::DispatchRaysArguments& args) override;

        void buildOpacityMicromap(rt::IOpacityMicromap* omm, const rt::OpacityMicromapDesc& desc) override;
        void copyRaytracingAccelerationStructure(rt::IAccelStruct* destination, rt::IAccelStruct* source) override;
        void buildBottomLevelAccelStruct(rt::IAccelStruct* as, const rt::GeometryDesc* pGeometries, size_t numGeometries, rt::AccelStructBuildFlags buildFlags) override;
        void compactBottomLevelAccelStructs() override;
        void buildTopLevelAccelStruct(rt::IAccelStruct* as, const rt::InstanceDesc* pInstances, size_t numInstances, rt::AccelStructBuildFlags buildFlags) override;
        void buildTopLevelAccelStructFromBuffer(rt::IAccelStruct* as, nvrhi::IBuffer* instanceBuffer, uint64_t instanceBufferOffset, size_t numInstances,
            rt::AccelStructBuildFlags buildFlags = rt::AccelStructBuildFlags::None) override;
        void executeMultiIndirectClusterOperation(const rt::cluster::OperationDesc& desc) override;

        void convertCoopVecMatrices(coopvec::ConvertMatrixLayoutDesc const* convertDescs, size_t numDescs) override;

        void beginTimerQuery(ITimerQuery* query) override;
        void endTimerQuery(ITimerQuery* query) override;

        void beginMarker(const char *name) override;
        void endMarker() override;

        void setEnableAutomaticBarriers(bool enable) override;
        void setResourceStatesForBindingSet(IBindingSet* bindingSet) override;

        void setEnableUavBarriersForTexture(ITexture* texture, bool enableBarriers) override;
        void setEnableUavBarriersForBuffer(IBuffer* buffer, bool enableBarriers) override;

        void beginTrackingTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates stateBits) override;
        void beginTrackingBufferState(IBuffer* buffer, ResourceStates stateBits) override;

        void setTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates stateBits) override;
        void setBufferState(IBuffer* buffer, ResourceStates stateBits) override;
        void setAccelStructState(rt::IAccelStruct* as, ResourceStates stateBits) override;
        
        void setPermanentTextureState(ITexture* texture, ResourceStates stateBits) override;
        void setPermanentBufferState(IBuffer* buffer, ResourceStates stateBits) override;

        void commitBarriers() override;

        ResourceStates getTextureSubresourceState(ITexture* texture, ArraySlice arraySlice, MipLevel mipLevel) override;
        ResourceStates getBufferState(IBuffer* buffer) override;

        nvrhi::IDevice* getDevice() override;
        const CommandListParameters& getDesc() override { return m_Desc; }

        // Metal3 specific methods
        id<MTLCommandBuffer> getNativeCommandBuffer() override;
        void setTracyGpuScope(const char* name, const char* file, const char* function, uint32_t line, void* context) override;
        void clearTracyGpuScope() override;
    private:
        struct TracyGpuScopeDesc
        {
            const char* name = nullptr;
            const char* file = nullptr;
            const char* function = nullptr;
            uint32_t line = 0;
            void* context = nullptr;
            bool active = false;
        };
        const MTL3Context& m_Context;

        IDevice* m_Device;
        
        // i can prolly have Queue struct and pointer to object here
        // but its not really needed, and neither does metal 3 demand it.
        // would rather just have it inside m_Context like i have right now
        //Queue* m_Queue;
        UploadManager m_UploadManager;
        // Argument tables must use stable memory until command-buffer
        // completion, but must not create a native MTLBuffer per draw.
        UploadManager m_ArgumentTableManager;
        TransientIndirectResourcePool m_TransientIndirectResources;
        
        CommandListParameters m_Desc;

        // Cache for user-provided state
        GraphicsState m_CurrentGraphicsState;
        ComputeState m_CurrentComputeState;
        MeshletState m_CurrentMeshletState;
        rt::State m_CurrentRayTracingState;
        bool m_CurrentGraphicsStateValid = false;
        bool m_CurrentComputeStateValid = false;
        bool m_CurrentMeshletStateValid = false;
        bool m_CurrentRayTracingStateValid = false;
        bool m_GeometryEmulationDrawStateValid = false;
        // Holds the uploaded IRRuntimeVertexBuffers table for the active
        // geometry-emulation draw; object shaders read this to fetch app vertex buffers.
        id<MTLBuffer> m_GeometryEmulationVertexBuffers = nil;
        NSUInteger m_GeometryEmulationVertexBuffersOffset = 0;

        bool m_BindingStatesDirty = false;

        // Cache for internal state

        // not really tracked, cuz metal buffers expire once they commit
        // kept for convenience (TODO?)
        id<MTLCommandBuffer> trackedCmdBuffer;
        id<MTLRenderCommandEncoder> m_RenderEncoder = nil;
        id<MTLComputeCommandEncoder> m_ComputeEncoder = nil;

        std::vector<BindingSetHandle> m_ReferencedBindingSets;
        std::vector<id<MTLBuffer>> m_ReferencedNativeBuffers;
        std::vector<id<MTLResource>> m_ReferencedNativeResources;

        std::vector<MetalArgumentTableCacheEntry> m_ArgumentTableCache;
        uint64_t m_ArgumentTableAllocationCount = 0;
        size_t m_ArgumentTablePageCountAtOpen = 0;

        std::array<uint8_t, c_MaxPushConstantSize> m_PushConstants{};
        size_t m_PushConstantSize = 0;
        std::unordered_map<Buffer*, VolatileBufferAllocation> m_VolatileBufferAllocations;

        TracyGpuScopeDesc m_TracyGpuScope;
#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
        std::optional<tracy::MetalZoneScope> m_TracyEncoderScope;
        std::vector<std::unique_ptr<tracy::SourceLocationData>> m_TracySourceLocations;
#endif

        void endEncoding();
        id<MTLRenderCommandEncoder> getOrCreateRenderEncoder();
        id<MTLComputeCommandEncoder> getOrCreateComputeEncoder();
        ArgumentTableAllocation getOrCreateArgumentTable(const MetalStageBindingPlan& plan, const BindingSetVector& bindingSets);
        ArgumentTableAllocation createArgumentTable(const MetalStageBindingPlan& plan, const BindingSetVector& bindingSets);
        void bindGraphicsArgumentTable(id<MTLRenderCommandEncoder> encoder, const BindingSetVector& bindingSets, const MetalStageBindingPlan& plan, MTLRenderStages stages);
        bool encodeArgumentTableEntry(IRDescriptorTableEntry* entry, const MetalBindingResource& resource);
        void useArgumentTableResource(id<MTLComputeCommandEncoder> encoder, const MetalBindingResource& resource);
        void useArgumentTableResource(id<MTLRenderCommandEncoder> encoder, const MetalBindingResource& resource, MTLRenderStages stages);
        void useArgumentTableResources(id<MTLComputeCommandEncoder> encoder, const BindingSetVector& bindingSets, const MetalStageBindingPlan& plan);
        void useArgumentTableResources(id<MTLRenderCommandEncoder> encoder, const BindingSetVector& bindingSets, const MetalStageBindingPlan& plan, MTLRenderStages stages);
        bool bindGeometryEmulationVertexBuffers(id<MTLRenderCommandEncoder> encoder, const GraphicsPipeline& pipeline, const GraphicsState& state);
        void drawIndirectGeometryEmulation(uint32_t offsetBytes, uint32_t drawCount);
        void drawIndexedIndirectGeometryEmulation(uint32_t offsetBytes, uint32_t drawCount);
        void drawIndexedIndirectCountGeometryEmulation(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount);
        void applyGraphicsStateToEncoder(id<MTLRenderCommandEncoder> encoder, const GraphicsState& state);
        void applyGraphicsBindings(id<MTLRenderCommandEncoder> encoder, const GraphicsState& state);
        void applyComputeBindings(id<MTLComputeCommandEncoder> encoder, const ComputeState& state);
        void referenceBindingSet(BindingSet* bindingSet);
#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
        tracy::SourceLocationData* getOrCreateTracySourceLocation();
        void beginTracyRenderEncoderZone(MTLRenderPassDescriptor* desc);
        void beginTracyComputeEncoderZone(MTLComputePassDescriptor* desc);
#endif
    };

    // metal3 device
    /* 
     * device functions declarations of virtual nvrhi functions declared in nvrhi.h
     * 
    */

    class Device : public RefCounter<nvrhi::metal3::IDevice>
    {
    public:
        explicit Device(const DeviceDesc& desc);
        ~Device() override;
        
        // IResource impl
        Object getNativeObject(ObjectType objectType) override;

        // IDevice impl

        // Metal3: MTLHeap or resource heaps (closest to dx12/vulkan type descriptor heaps) not implemented yet, does nothing
        HeapHandle createHeap(const HeapDesc& d) override;

        TextureHandle createTexture(const TextureDesc& d) override;

        // Metal3: useful for texture allocated inside heaps, for manual/placed allocation; we create committed textures directly currently
        MemoryRequirements getTextureMemoryRequirements(ITexture* texture) override;
        // Metal 3: nothing like binding a texture memory in metal, does nothing, a stub
        bool bindTextureMemory(ITexture* texture, IHeap* heap, uint64_t offset) override;

        TextureHandle createHandleForNativeTexture(ObjectType objectType, Object texture, const TextureDesc& desc) override;

        StagingTextureHandle createStagingTexture(const TextureDesc& d, CpuAccessMode cpuAccess) override;
        void *mapStagingTexture(IStagingTexture* tex, const TextureSlice& slice, CpuAccessMode cpuAccess, size_t *outRowPitch) override;
        void unmapStagingTexture(IStagingTexture* tex) override;

        void getTextureTiling(ITexture* texture, uint32_t* numTiles, PackedMipDesc* desc, TileShape* tileShape, uint32_t* subresourceTilingsNum, SubresourceTiling* subresourceTilings) override;
        void updateTextureTileMappings(ITexture* texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings, CommandQueue executionQueue = CommandQueue::Graphics) override;

        SamplerFeedbackTextureHandle createSamplerFeedbackTexture(ITexture* pairedTexture, const SamplerFeedbackTextureDesc& desc) override;
        SamplerFeedbackTextureHandle createSamplerFeedbackForNativeTexture(ObjectType objectType, Object texture, ITexture* pairedTexture) override;

        BufferHandle createBuffer(const BufferDesc& d) override;
        void *mapBuffer(IBuffer* b, CpuAccessMode mapFlags) override;
        void unmapBuffer(IBuffer* b) override;

        // Metal3: useful for buffers allocated inside heaps, for manual/placed allocation; we create committed buffers directly currently
        MemoryRequirements getBufferMemoryRequirements(IBuffer* buffer) override;
        bool bindBufferMemory(IBuffer* buffer, IHeap* heap, uint64_t offset) override;

        BufferHandle createHandleForNativeBuffer(ObjectType objectType, Object buffer, const BufferDesc& desc) override;

        // Metal3: -> the way the current pipeline will handle shaders is by processing *.metallib files by shader reflection and 
        // storing arguments (index, slots, types, etc) for use later at resource binding/pipeline creation, execution stage. 
        // -> it also expects a reflection .json file for every shader. 
        // The path supported is using *metalshader-converter* for generating shader reflection data. The pipeline is as follows:
        // - *.hlsl -> *.dxil  (with dxc)
        // - *.dxil -> *.metallib (metalshader-converter)
        // - *.metallib -> *.reflection.json (with msc's reflection aergument)

        ShaderHandle createShader(const ShaderDesc& d, const void* binary, size_t binarySize) override;
        ShaderHandle createShaderSpecialization(IShader* baseShader, const ShaderSpecialization* constants, uint32_t numConstants) override;
        ShaderLibraryHandle createShaderLibrary(const void* binary, size_t binarySize) override;

        SamplerHandle createSampler(const SamplerDesc& d) override;

        InputLayoutHandle createInputLayout(const VertexAttributeDesc* d, uint32_t attributeCount, IShader* vertexShader) override;

        EventQueryHandle createEventQuery() override;
        void setEventQuery(IEventQuery* query, CommandQueue queue) override;
        bool pollEventQuery(IEventQuery* query) override;
        void waitEventQuery(IEventQuery* query) override;
        void resetEventQuery(IEventQuery* query) override;

        TimerQueryHandle createTimerQuery() override;
        bool pollTimerQuery(ITimerQuery* query) override;
        float getTimerQueryTime(ITimerQuery* query) override;
        void resetTimerQuery(ITimerQuery* query) override;

        GraphicsAPI getGraphicsAPI() override;

        FramebufferHandle createFramebuffer(const FramebufferDesc& desc) override;
        
        GraphicsPipelineHandle createGraphicsPipeline(const GraphicsPipelineDesc& desc, FramebufferInfo const& fbinfo) override;
        
        GraphicsPipelineHandle createGraphicsPipeline(const GraphicsPipelineDesc& desc, IFramebuffer* fb) override;
        
        ComputePipelineHandle createComputePipeline(const ComputePipelineDesc& desc) override;

        MeshletPipelineHandle createMeshletPipeline(const MeshletPipelineDesc& desc, FramebufferInfo const& fbinfo) override;

        MeshletPipelineHandle createMeshletPipeline(const MeshletPipelineDesc& desc, IFramebuffer* fb) override;

        // Unsupported: RT backend is WIP
        rt::PipelineHandle createRayTracingPipeline(const rt::PipelineDesc& desc) override;

        BindingLayoutHandle createBindingLayout(const BindingLayoutDesc& desc) override;
        BindingLayoutHandle createBindlessLayout(const BindlessLayoutDesc& desc) override;

        BindingSetHandle createBindingSet(const BindingSetDesc& desc, IBindingLayout* layout) override;
        DescriptorTableHandle createDescriptorTable(IBindingLayout* layout) override;

        void resizeDescriptorTable(IDescriptorTable* descriptorTable, uint32_t newSize, bool keepContents = true) override;
        bool writeDescriptorTable(IDescriptorTable* descriptorTable, const BindingSetItem& item) override;

        rt::OpacityMicromapHandle createOpacityMicromap(const rt::OpacityMicromapDesc& desc) override;
        rt::AccelStructHandle createAccelStruct(const rt::AccelStructDesc& desc) override;
        MemoryRequirements getAccelStructMemoryRequirements(rt::IAccelStruct* as) override;
        rt::cluster::OperationSizeInfo getClusterOperationSizeInfo(const rt::cluster::OperationParams& params) override;

        bool bindAccelStructMemory(rt::IAccelStruct* as, IHeap* heap, uint64_t offset) override;

        nvrhi::CommandListHandle createCommandList(const CommandListParameters& params = CommandListParameters()) override;
        uint64_t executeCommandLists(nvrhi::ICommandList* const* pCommandLists, size_t numCommandLists, CommandQueue executionQueue = CommandQueue::Graphics) override;
        void queueWaitForCommandList(CommandQueue waitQueue, CommandQueue executionQueue, uint64_t instance) override;
        bool waitForIdle() override;

        CommandListLifetimeTrackerHandle createCommandListLifetimeTracker(CommandQueue executionQueue) override;

        // Metal3: does nothing
        void runGarbageCollection() override;
        bool queryFeatureSupport(Feature feature, void* pInfo = nullptr, size_t infoSize = 0) override;
        FormatSupport queryFormatSupport(Format format) override;
        coopvec::DeviceFeatures queryCoopVecFeatures() override;
        coopvec::MatMulFormatSupport queryCoopVecMatMulFormatSupport(const coopvec::MatMulFormatCombo& combination) override;
        coopvec::TrainingFormatSupport queryCoopVecTrainingFormatSupport(coopvec::DataType componentType) override;
        size_t getCoopVecMatrixSize(coopvec::DataType type, coopvec::MatrixLayout layout, int rows, int columns) override;
        Object getNativeQueue(ObjectType objectType, CommandQueue queue) override;
        IMessageCallback* getMessageCallback() override { return m_Context.messageCallback; }

        // unused funcs, no aftermath sdk here :/
        bool isAftermathEnabled() override { return m_AftermathEnabled; }
        AftermathCrashDumpHelper& getAftermathCrashDumpHelper() override { return m_AftermathCrashDumpHelper; }

        // TODO: metal.h virtual funs implementation if any
        
    private:
        bool m_AftermathEnabled;
        AftermathCrashDumpHelper m_AftermathCrashDumpHelper;

        MTL3Context m_Context;

        std::mutex m_Mutex;
    };
}
