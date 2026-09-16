#include "metal3-backend.h"
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>

namespace nvrhi::metal3
{
    struct IcbMeshIndirectParams
    {
        uint32_t maxDrawCount = 0;
        uint32_t paramOffsetBytes = 0;
        uint32_t countOffsetBytes = 0;
        uint32_t objectThreadsX = 1;
        uint32_t objectThreadsY = 1;
        uint32_t objectThreadsZ = 1;
        uint32_t meshThreadsX = 1;
        uint32_t meshThreadsY = 1;
        uint32_t meshThreadsZ = 1;
    };

    struct MeshIndirectIcbFillState
    {
        id<MTLComputePipelineState> pipeline = nil;
        id<MTLArgumentEncoder> icbEncoder = nil;
    };

    static const MeshIndirectIcbFillState& getMeshIndirectIcbFillState(const MTL3Context& context)
    {
        static std::mutex mutex;
        static MeshIndirectIcbFillState state;
        static bool attempted = false;

        std::lock_guard<std::mutex> lock(mutex);
        if (attempted)
            return state;
        attempted = true;

        static constexpr const char* source = R"(
            #include <metal_stdlib>
            using namespace metal;

            struct IcbMeshIndirectParams
            {
                uint maxDrawCount;
                uint paramOffsetBytes;
                uint countOffsetBytes;
                uint objectThreadsX;
                uint objectThreadsY;
                uint objectThreadsZ;
                uint meshThreadsX;
                uint meshThreadsY;
                uint meshThreadsZ;
            };

            struct IcbArgumentBuffer
            {
                command_buffer icb [[id(0)]];
            };

            kernel void nvrhi_metal3_fill_mesh_indirect_icb(
                device const uchar* indirectParams [[buffer(0)]],
                device const uchar* rawCount [[buffer(1)]],
                device uint2* executionRange [[buffer(2)]],
                constant IcbArgumentBuffer& icbArgumentBuffer [[buffer(3)]],
                constant IcbMeshIndirectParams& params [[buffer(4)]],
                uint tid [[thread_position_in_grid]])
            {
                device const uint* countPtr = reinterpret_cast<device const uint*>(rawCount + params.countOffsetBytes);
                const uint gpuCount = min(*countPtr, params.maxDrawCount);
                if (tid == 0)
                    executionRange[0] = uint2(0, gpuCount);
                if (tid >= gpuCount)
                    return;

                device const uint* args = reinterpret_cast<device const uint*>(indirectParams + params.paramOffsetBytes) + tid * 3;
                render_command command(icbArgumentBuffer.icb, tid);
                command.draw_mesh_threadgroups(uint3(args[0], args[1], args[2]),
                    uint3(params.objectThreadsX, params.objectThreadsY, params.objectThreadsZ),
                    uint3(params.meshThreadsX, params.meshThreadsY, params.meshThreadsZ));
            }
        )";

        if (@available(macOS 14.0, *))
        {
            NSError* error = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.languageVersion = MTLLanguageVersion3_1;
            id<MTLLibrary> library = [context.device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                                   options:options
                                                                     error:&error];
            if (!library)
            {
                std::string message = "[metal3] failed to compile mesh-indirect ICB helper";
                if (error)
                    message += std::string(": ") + [[error localizedDescription] UTF8String];
                context.error(message);
                return state;
            }
            id<MTLFunction> function = [library newFunctionWithName:@"nvrhi_metal3_fill_mesh_indirect_icb"];
            if (!function)
            {
                context.error("[metal3] mesh-indirect ICB helper function missing");
                return state;
            }
            state.icbEncoder = [function newArgumentEncoderWithBufferIndex:3];
            if (!state.icbEncoder)
            {
                context.error("[metal3] failed to create mesh-indirect ICB argument encoder");
                return state;
            }
            MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
            descriptor.computeFunction = function;
            descriptor.supportIndirectCommandBuffers = YES;
            state.pipeline = [context.device newComputePipelineStateWithDescriptor:descriptor
                                                                           options:MTLPipelineOptionNone
                                                                        reflection:nil
                                                                             error:&error];
            if (!state.pipeline)
            {
                std::string message = "[metal3] failed to create mesh-indirect ICB helper pipeline";
                if (error)
                    message += std::string(": ") + [[error localizedDescription] UTF8String];
                context.error(message);
                state.icbEncoder = nil;
            }
        }
        else
        {
            context.error("[metal3] counted mesh dispatches require macOS 14.0 or newer");
        }
        return state;
    }

    void CommandList::setMeshletState(const MeshletState& state)
    {
        if (m_CurrentFramebuffer != state.framebuffer)
            endEncoding();
        m_CurrentMeshletState = state;
        m_CurrentFramebuffer = state.framebuffer;
        m_CurrentMeshletStateValid = true;
        m_CurrentGraphicsStateValid = false;
        if (state.pipeline)
            m_ReferencedStateResources.emplace_back(state.pipeline);
        if (state.framebuffer)
            m_ReferencedStateResources.emplace_back(state.framebuffer);
        referenceBuffer(state.indirectParams);
        referenceBuffer(state.indirectCountBuffer);
        if (m_EnableAutomaticBarriers)
        {
            for (IBindingSet* bindingSet : state.bindings)
                setResourceStatesForBindingSet(bindingSet);
            if (state.indirectParams)
                setBufferState(state.indirectParams, ResourceStates::IndirectArgument);
            if (state.indirectCountBuffer)
                setBufferState(state.indirectCountBuffer, ResourceStates::IndirectArgument);
            if (state.framebuffer)
            {
                for (const FramebufferAttachment& attachment : state.framebuffer->getDesc().colorAttachments)
                    setTextureState(attachment.texture, attachment.subresources, ResourceStates::RenderTarget);
                const FramebufferAttachment& depth = state.framebuffer->getDesc().depthAttachment;
                setTextureState(depth.texture, depth.subresources, depth.isReadOnly ? ResourceStates::DepthRead : ResourceStates::DepthWrite);
            }
            commitBarriers();
        }
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<MeshletPipeline*>(state.pipeline);
        if (!encoder || !pipeline)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] setMeshletState skipped: encoder=" +
                    std::string(encoder ? "yes" : "no") + " pipeline=" + (pipeline ? "yes" : "no"));
            return;
        }
        applyMeshletStateToEncoder(encoder, state);
    }

    void CommandList::applyMeshletStateToEncoder(id<MTLRenderCommandEncoder> encoder, const MeshletState& state)
    {
        auto* pipeline = static_cast<MeshletPipeline*>(state.pipeline);
        if (!encoder || !pipeline)
            return;

        m_GeometryEmulationDrawStateValid = false;
        m_GeometryEmulationVertexBuffers = nil;
        m_GeometryEmulationVertexBuffersOffset = 0;
        m_VertexBufferBindings.fill({});
        m_FragmentBufferBindings.fill({});

        [encoder setRenderPipelineState:pipeline->pipeline];
        [encoder setDepthStencilState:pipeline->depthStencilState];
        [encoder setCullMode:pipeline->cullMode];
        [encoder setFrontFacingWinding:pipeline->frontWinding];
        const DepthStencilState& depthStencil = pipeline->desc.renderState.depthStencilState;
        [encoder setStencilReferenceValue:depthStencil.dynamicStencilRef ? state.dynamicStencilRefValue : depthStencil.stencilRefValue];
        [encoder setBlendColorRed:state.blendConstantColor.r green:state.blendConstantColor.g
            blue:state.blendConstantColor.b alpha:state.blendConstantColor.a];
        m_BindingStatesDirty = false;
        m_ArgumentTablesDirty = false;

        for (const Viewport& vp : state.viewport.viewports)
            [encoder setViewport:MTLViewport{ vp.minX, vp.minY, vp.width(), vp.height(), vp.minZ, vp.maxZ }];
        for (const Rect& rect : state.viewport.scissorRects)
            [encoder setScissorRect:MTLScissorRect{ NSUInteger(rect.minX), NSUInteger(rect.minY), NSUInteger(rect.width()), NSUInteger(rect.height()) }];

        applyBindingSets(state.bindings);
        if (pipeline->hasObjectStage)
            bindGraphicsArgumentTable(encoder, state.bindings, pipeline->objectBindingPlan, MTLRenderStageObject);
        bindGraphicsArgumentTable(encoder, state.bindings, pipeline->meshBindingPlan, MTLRenderStageMesh);
        if (pipeline->desc.PS)
            bindGraphicsArgumentTable(encoder, state.bindings, pipeline->fragmentBindingPlan, MTLRenderStageFragment);
    }

    void CommandList::dispatchMesh(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ)
    {
        if (!m_CurrentMeshletStateValid)
            return;
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<MeshletPipeline*>(m_CurrentMeshletState.pipeline);
        if (!encoder || !pipeline)
            return;
        if (m_BindingStatesDirty)
            applyMeshletStateToEncoder(encoder, m_CurrentMeshletState);
        else if (m_ArgumentTablesDirty)
            rebindArgumentTables();
        [encoder drawMeshThreadgroups:MTLSizeMake(groupsX, groupsY, groupsZ)
            threadsPerObjectThreadgroup:pipeline->threadsPerObjectThreadgroup
              threadsPerMeshThreadgroup:pipeline->threadsPerMeshThreadgroup];
    }

    void CommandList::dispatchMeshIndirect(uint32_t offsetBytes, uint32_t maxDrawCount)
    {
        if (!m_CurrentMeshletStateValid)
            return;
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<MeshletPipeline*>(m_CurrentMeshletState.pipeline);
        auto* indirectParams = static_cast<Buffer*>(m_CurrentMeshletState.indirectParams);
        if (!encoder || !pipeline || !indirectParams || !indirectParams->buffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] dispatchMeshIndirect skipped: encoder/pipeline/args missing");
            return;
        }
        if (m_BindingStatesDirty)
            applyMeshletStateToEncoder(encoder, m_CurrentMeshletState);
        else if (m_ArgumentTablesDirty)
            rebindArgumentTables();
        for (uint32_t drawIndex = 0; drawIndex < maxDrawCount; ++drawIndex)
        {
            [encoder drawMeshThreadgroupsWithIndirectBuffer:indirectParams->buffer
                                       indirectBufferOffset:offsetBytes
                                threadsPerObjectThreadgroup:pipeline->threadsPerObjectThreadgroup
                                  threadsPerMeshThreadgroup:pipeline->threadsPerMeshThreadgroup];
            offsetBytes += sizeof(DispatchIndirectArguments);
        }
    }

    void CommandList::dispatchMeshIndirectCount(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount)
    {
        if (!m_CurrentMeshletStateValid || maxDrawCount == 0)
            return;
        auto* pipeline = static_cast<MeshletPipeline*>(m_CurrentMeshletState.pipeline);
        auto* indirectParams = static_cast<Buffer*>(m_CurrentMeshletState.indirectParams);
        auto* indirectCount = static_cast<Buffer*>(m_CurrentMeshletState.indirectCountBuffer);
        if (!pipeline || !indirectParams || !indirectParams->buffer || !indirectCount || !indirectCount->buffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] dispatchMeshIndirectCount skipped: missing pipeline/args/count buffer");
            return;
        }
        const MeshIndirectIcbFillState& fillState = getMeshIndirectIcbFillState(m_Context);
        if (!fillState.pipeline || !fillState.icbEncoder)
        {
            m_Context.warning("[metal3] counted mesh dispatch skipped because the ICB helper pipeline is unavailable");
            return;
        }
        if (@available(macOS 14.0, *))
        {
            id<MTLRenderCommandEncoder> renderEncoder = getOrCreateRenderEncoder();
            if (!renderEncoder)
                return;
            if (m_BindingStatesDirty)
                applyMeshletStateToEncoder(renderEncoder, m_CurrentMeshletState);
            if (m_RecordingFailed)
                return;

            MTLIndirectCommandBufferDescriptor* icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
            icbDesc.commandTypes = MTLIndirectCommandTypeDrawMeshThreadgroups;
            icbDesc.inheritPipelineState = YES;
            icbDesc.inheritBuffers = YES;

            NSUInteger icbCapacity = 0;
            id<MTLIndirectCommandBuffer> icb =
                m_TransientIndirectResources.acquireIndirectCommandBuffer(icbDesc, NSUInteger(maxDrawCount), nullptr, &icbCapacity);
            const TransientBufferAllocation executionRange =
                m_TransientIndirectResources.allocatePrivate(sizeof(MTLIndirectCommandBufferExecutionRange));
            const TransientBufferAllocation paramsAllocation =
                m_TransientIndirectResources.allocateShared(sizeof(IcbMeshIndirectParams));
            const TransientBufferAllocation icbArgumentBuffer =
                m_TransientIndirectResources.allocateShared(fillState.icbEncoder.encodedLength);
            if (!icb || !executionRange.buffer || !paramsAllocation.buffer || !paramsAllocation.cpuAddress ||
                !icbArgumentBuffer.buffer || !icbArgumentBuffer.cpuAddress)
            {
                m_Context.error("[metal3] failed to allocate counted mesh dispatch ICB resources");
                return;
            }

            [fillState.icbEncoder setArgumentBuffer:icbArgumentBuffer.buffer offset:icbArgumentBuffer.offset];
            [fillState.icbEncoder setIndirectCommandBuffer:icb atIndex:0];

            auto* params = reinterpret_cast<IcbMeshIndirectParams*>(paramsAllocation.cpuAddress);
            params->maxDrawCount = maxDrawCount;
            params->paramOffsetBytes = paramOffsetBytes;
            params->countOffsetBytes = countOffsetBytes;
            params->objectThreadsX = uint32_t(pipeline->threadsPerObjectThreadgroup.width);
            params->objectThreadsY = uint32_t(pipeline->threadsPerObjectThreadgroup.height);
            params->objectThreadsZ = uint32_t(pipeline->threadsPerObjectThreadgroup.depth);
            params->meshThreadsX = uint32_t(pipeline->threadsPerMeshThreadgroup.width);
            params->meshThreadsY = uint32_t(pipeline->threadsPerMeshThreadgroup.height);
            params->meshThreadsZ = uint32_t(pipeline->threadsPerMeshThreadgroup.depth);

            m_ReferencedNativeResources.push_back(icb);
            m_ReferencedNativeBuffers.push_back(executionRange.buffer);
            m_ReferencedNativeBuffers.push_back(paramsAllocation.buffer);
            m_ReferencedNativeBuffers.push_back(icbArgumentBuffer.buffer);

            id<MTLBlitCommandEncoder> blit = beginBlitEncoder();
            beginEncoding(blit, "reset mesh ICB");
            [blit resetCommandsInBuffer:icb withRange:NSMakeRange(0, icbCapacity)];
            endEncoding(blit);

            id<MTLComputeCommandEncoder> compute = beginTransientComputeEncoder();
            beginEncoding(compute, "fill mesh ICB");
            [compute setComputePipelineState:fillState.pipeline];
            [compute setBuffer:indirectParams->buffer offset:0 atIndex:0];
            [compute setBuffer:indirectCount->buffer offset:0 atIndex:1];
            [compute setBuffer:executionRange.buffer offset:executionRange.offset atIndex:2];
            [compute setBuffer:icbArgumentBuffer.buffer offset:icbArgumentBuffer.offset atIndex:3];
            [compute setBuffer:paramsAllocation.buffer offset:paramsAllocation.offset atIndex:4];
            const NSUInteger threads = std::max<NSUInteger>(1, fillState.pipeline.threadExecutionWidth);
            const NSUInteger groups = (NSUInteger(maxDrawCount) + threads - 1) / threads;
            [compute dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            endEncoding(compute);

            renderEncoder = getOrCreateRenderEncoder();
            if (!renderEncoder)
                return;
            applyMeshletStateToEncoder(renderEncoder, m_CurrentMeshletState);
            [renderEncoder executeCommandsInBuffer:icb indirectBuffer:executionRange.buffer
                              indirectBufferOffset:executionRange.offset];
        }
    }

    Object MeshletPipeline::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::MTL3_RenderPipeline)
            return Object((__bridge void*)pipeline);
        return nullptr;
    }
}
