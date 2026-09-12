#include "metal3-backend.h"
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>
#include <cctype>

namespace nvrhi::metal3
{
    static std::string normalizeMscVertexInputName(const std::string& value)
    {
        std::string result;
        result.reserve(value.size());
        for (unsigned char ch : value)
        {
            if (std::isalnum(ch))
                result.push_back(static_cast<char>(std::tolower(ch)));
        }

        while (!result.empty() && std::isdigit(static_cast<unsigned char>(result.back())))
            result.pop_back();

        return result;
    }

    // c_MscVertexBufferBindPoint -> MSC's vertex-buffer binding slot where we start binding actual app vertex buffers
    // so vertex buffer slot 0 gets bound at index 6
    static constexpr uint32_t c_MscVertexBufferBindPoint = 6;
    // c_MscVertexAttributeBase -> the Metal vertex attribute index where MSC expects user vertex attributes to begin
    static constexpr uint32_t c_MscVertexAttributeBase = 11;
    // c_MetalMaxVertexAttributes -> just a guard for Metal’s vertex attribute descriptor array
    static constexpr uint32_t c_MetalMaxVertexAttributes = 31;

    static void setVertexAttribute(MTLVertexDescriptor* vertexDescriptor, uint32_t attributeIndex, const VertexAttributeDesc& attr)
    {
        vertexDescriptor.attributes[attributeIndex].format = convertVertexFormat(attr.format);
        vertexDescriptor.attributes[attributeIndex].offset = attr.offset;
        vertexDescriptor.attributes[attributeIndex].bufferIndex = c_MscVertexBufferBindPoint + attr.bufferIndex;
    }

    InputLayoutHandle Device::createInputLayout(const VertexAttributeDesc* d, uint32_t attributeCount, IShader* vertexShader)
    {
        auto* shader = static_cast<Shader*>(vertexShader);
        InputLayout* layout = new InputLayout();
        layout->attributes.assign(d, d + attributeCount);
        layout->vertexDescriptor = [[MTLVertexDescriptor alloc] init];

        for (uint32_t index = 0; index < attributeCount; ++index)
        {
            const VertexAttributeDesc& attr = d[index];
            uint32_t attributeIndex = index;
            if (shader && !shader->mscReflection.vertexInputAttributes.empty())
            {
                const std::string semantic = normalizeMscVertexInputName(attr.name);
                auto it = shader->mscReflection.vertexInputAttributes.find(semantic);
                if (it != shader->mscReflection.vertexInputAttributes.end())
                {
                    /* 
                    // MSC reflection reports the HLSL input-signature index.
                    // The generated Metal entry point places user attributes
                    // after its reserved ABI inputs, so Metal validation names
                    // POSITION0 as attribute 11, RECT0 as 12, etc.
                    */
                    attributeIndex = c_MscVertexAttributeBase + it->second;
                }
                else
                {
                    static int missingSemanticLogCount = 0;
                    if (missingSemanticLogCount++ < 32)
                    {
                        m_Context.warning("[metal3] vertex attribute '" + attr.name +
                            "' was not found in MSC reflection for shader '" +
                            (shader ? shader->desc.debugName : std::string("<null>")) +
                            "'; using input-layout index " + std::to_string(index));
                    }
                }
            }

            if (attributeIndex < c_MetalMaxVertexAttributes)
                setVertexAttribute(layout->vertexDescriptor, attributeIndex, attr);

            uint32_t bufferIndex = c_MscVertexBufferBindPoint + attr.bufferIndex;
            layout->vertexDescriptor.layouts[bufferIndex].stride = attr.elementStride;
            layout->vertexDescriptor.layouts[bufferIndex].stepFunction = attr.isInstanced ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
            layout->vertexDescriptor.layouts[bufferIndex].stepRate = 1;
        }

        return InputLayoutHandle::Create(layout);
    }

    const VertexAttributeDesc* InputLayout::getAttributeDesc(uint32_t index) const
    {
        return index < attributes.size() ? &attributes[index] : nullptr;
    }

    FramebufferHandle Device::createFramebuffer(const FramebufferDesc& desc)
    {
        Framebuffer* framebuffer = new Framebuffer();
        framebuffer->desc = desc;
        framebuffer->framebufferInfo = FramebufferInfoEx(desc);
        return FramebufferHandle::Create(framebuffer);
    }

    static void applyColorBlend(MTLRenderPipelineColorAttachmentDescriptor* attachment, const BlendState::RenderTarget& blend)
    {
        attachment.blendingEnabled = blend.blendEnable;
        attachment.sourceRGBBlendFactor = convertBlendFactor(blend.srcBlend);
        attachment.destinationRGBBlendFactor = convertBlendFactor(blend.destBlend);
        attachment.rgbBlendOperation = convertBlendOp(blend.blendOp);
        attachment.sourceAlphaBlendFactor = convertBlendFactor(blend.srcBlendAlpha);
        attachment.destinationAlphaBlendFactor = convertBlendFactor(blend.destBlendAlpha);
        attachment.alphaBlendOperation = convertBlendOp(blend.blendOpAlpha);
        attachment.writeMask = MTLColorWriteMaskNone;
        if (!!(blend.colorWriteMask & ColorMask::Red)) attachment.writeMask |= MTLColorWriteMaskRed;
        if (!!(blend.colorWriteMask & ColorMask::Green)) attachment.writeMask |= MTLColorWriteMaskGreen;
        if (!!(blend.colorWriteMask & ColorMask::Blue)) attachment.writeMask |= MTLColorWriteMaskBlue;
        if (!!(blend.colorWriteMask & ColorMask::Alpha)) attachment.writeMask |= MTLColorWriteMaskAlpha;
    }

    static const char* shaderEntryName(const Shader* shader)
    {
        return shader && !shader->desc.entryName.empty() ? shader->desc.entryName.c_str() : "main";
    }

    static bool geometryInputMatchesPrimitive(const std::string& inputPrimitive, PrimitiveType primitiveType)
    {
        if (inputPrimitive == "Point")
            return primitiveType == PrimitiveType::PointList;
        if (inputPrimitive == "Line")
            return primitiveType == PrimitiveType::LineList || primitiveType == PrimitiveType::LineStrip;
        if (inputPrimitive == "Triangle")
            return primitiveType == PrimitiveType::TriangleList || primitiveType == PrimitiveType::TriangleStrip ||
                primitiveType == PrimitiveType::TriangleFan;
        if (inputPrimitive == "LineWithAdj")
            return false;
        if (inputPrimitive == "TriangleWithAdj")
            return primitiveType == PrimitiveType::TriangleListWithAdjacency ||
                primitiveType == PrimitiveType::TriangleStripWithAdjacency;

        return false;
    }

    static void applyFramebufferFormats(MTLRenderPipelineDescriptor* pd, FramebufferInfo const& fbinfo, const BlendState& blendState)
    {
        for (NSUInteger i = 0; i < fbinfo.colorFormats.size(); ++i)
        {
            pd.colorAttachments[i].pixelFormat = convertFormat(fbinfo.colorFormats[i]);
            applyColorBlend(pd.colorAttachments[i], blendState.targets[i]);
        }
        if (fbinfo.depthFormat != Format::UNKNOWN)
            pd.depthAttachmentPixelFormat = convertFormat(fbinfo.depthFormat);
        pd.rasterSampleCount = fbinfo.sampleCount;
    }

    static void applyFramebufferFormats(MTLMeshRenderPipelineDescriptor* pd, FramebufferInfo const& fbinfo, const BlendState& blendState)
    {
        for (NSUInteger i = 0; i < fbinfo.colorFormats.size(); ++i)
        {
            pd.colorAttachments[i].pixelFormat = convertFormat(fbinfo.colorFormats[i]);
            applyColorBlend(pd.colorAttachments[i], blendState.targets[i]);
        }
        if (fbinfo.depthFormat != Format::UNKNOWN)
            pd.depthAttachmentPixelFormat = convertFormat(fbinfo.depthFormat);
        pd.rasterSampleCount = fbinfo.sampleCount;
    }

    static bool validateGeometryEmulationShader(
        const MTL3Context& context,
        const GraphicsPipelineDesc& desc,
        const Shader* vs,
        const Shader* gs,
        const Shader* ps,
        IRRuntimeGeometryPipelineConfig& outConfig)
    {
        if (!vs || !gs || !ps)
        {
            context.error("[metal3] geometry emulation pipelines require vertex, geometry, and pixel shaders");
            return false;
        }
        if (!vs->library || !gs->library || !ps->library)
        {
            context.error("[metal3] geometry emulation pipeline has a shader without a Metal library");
            return false;
        }
        if (!vs->stageInLibrary)
        {
            context.error("[metal3] geometry emulation VS '" + vs->desc.debugName +
                "' has no MSC stage-in library");
            return false;
        }
        if (!vs->function || !gs->function || !ps->function)
        {
            context.error("[metal3] geometry emulation pipeline has a shader without a Metal entry function");
            return false;
        }
        if (vs->stageInLibrary.functionNames.count == 0)
        {
            context.error("[metal3] geometry emulation vertex shader has no stage-in function candidates");
            return false;
        }
        if (vs->mscReflection.vertexOutputSizeInBytes == 0)
        {
            context.error("[metal3] geometry emulation VS '" + vs->desc.debugName +
                "' has no reflected vertex_output_size_in_bytes");
            return false;
        }
        if (gs->mscReflection.maxInputPrimitivesPerMeshThreadgroup == 0)
        {
            context.error("[metal3] geometry emulation GS '" + gs->desc.debugName +
                "' has no reflected max_input_primitives_per_mesh_threadgroup");
            return false;
        }
        if (gs->mscReflection.inputPrimitive.empty())
        {
            context.error("[metal3] geometry emulation GS '" + gs->desc.debugName +
                "' has no reflected input_primitive");
            return false;
        }
        if (!geometryInputMatchesPrimitive(gs->mscReflection.inputPrimitive, desc.primType))
        {
            context.error("[metal3] geometry emulation GS '" + gs->desc.debugName +
                "' input primitive '" + gs->mscReflection.inputPrimitive +
                "' does not match the graphics pipeline primitive topology");
            return false;
        }

        outConfig.gsVertexSizeInBytes = vs->mscReflection.vertexOutputSizeInBytes;
        outConfig.gsMaxInputPrimitivesPerMeshThreadgroup =
            gs->mscReflection.maxInputPrimitivesPerMeshThreadgroup;
        return true;
    }

    static id<MTLRenderPipelineState> createGeometryEmulationPipelineState(
        const MTL3Context& context,
        const GraphicsPipelineDesc& desc,
        FramebufferInfo const& fbinfo,
        Shader* vs,
        Shader* gs,
        Shader* ps,
        IRRuntimeGeometryPipelineConfig& outConfig)
    {
        if (!validateGeometryEmulationShader(context, desc, vs, gs, ps, outConfig))
            return nil;

        if (@available(macOS 14.0, *))
        {
            MTLMeshRenderPipelineDescriptor* pd = [[MTLMeshRenderPipelineDescriptor alloc] init];
            pd.label = @"nvrhi geometry emulation pipeline";
            pd.supportIndirectCommandBuffers = YES;
            applyFramebufferFormats(pd, fbinfo, desc.renderState.blendState);

            /* The MSC converter runtime creates the object/mesh functions from the
               converted VS/GS libraries, links the generated stage-in helper,
               and returns a regular MTLRenderPipelineState backed by a Metal
               mesh render pipeline. */
            const std::string vsEntry = shaderEntryName(vs);
            const std::string gsEntry = shaderEntryName(gs);
            const std::string psEntry = shaderEntryName(ps);

            IRGeometryEmulationPipelineDescriptor irDesc{};
            irDesc.stageInLibrary = vs->stageInLibrary;
            irDesc.vertexLibrary = vs->library;
            irDesc.vertexFunctionName = vsEntry.c_str();
            irDesc.geometryLibrary = gs->library;
            irDesc.geometryFunctionName = gsEntry.c_str();
            irDesc.fragmentLibrary = ps->library;
            irDesc.fragmentFunctionName = psEntry.c_str();
            irDesc.basePipelineDescriptor = pd;
            irDesc.pipelineConfig = outConfig;

            NSError* error = nil;
            id<MTLRenderPipelineState> nativePipeline =
                IRRuntimeNewGeometryEmulationPipeline(context.device, &irDesc, &error);
            if (!nativePipeline)
            {
                std::string message = "[nvrhi] Failed to create Metal geometry emulation pipeline";
                if (error)
                    message += std::string(": ") + [[error localizedDescription] UTF8String];
                context.error(message);
                return nil;
            }

            return nativePipeline;
        }

        context.error("[metal3] geometry emulation pipelines require macOS 14.0 or newer");
        return nil;
    }

    GraphicsPipelineHandle Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc, FramebufferInfo const& fbinfo)
    {
        auto* vs = static_cast<Shader*>(desc.VS.Get());
        auto* gs = static_cast<Shader*>(desc.GS.Get());
        auto* ps = static_cast<Shader*>(desc.PS.Get());
        if (!vs || !vs->function)
            return nullptr;

        const bool usesGeometryEmulation = gs != nullptr;
        IRRuntimeGeometryPipelineConfig geometryConfig{};
        NSError* error = nil;
        id<MTLRenderPipelineState> nativePipeline = nil;
        if (usesGeometryEmulation)
        {
            nativePipeline = createGeometryEmulationPipelineState(m_Context, desc, fbinfo, vs, gs, ps, geometryConfig);
            if (!nativePipeline)
                return nullptr;
        }
        else
        {
            MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
            pd.vertexFunction = vs->function;
            pd.fragmentFunction = ps ? ps->function : nil;  // can be nil for depth pass pipelines
            pd.supportIndirectCommandBuffers = YES;
            if (desc.inputLayout)
            {
                auto* inputLayout = static_cast<InputLayout*>(desc.inputLayout.Get());
                pd.vertexDescriptor = inputLayout ? inputLayout->vertexDescriptor : nil;
            }

            applyFramebufferFormats(pd, fbinfo, desc.renderState.blendState);
            nativePipeline = [m_Context.device newRenderPipelineStateWithDescriptor:pd error:&error];
            if (!nativePipeline)
            {
                std::string message = "[nvrhi] Failed to create Metal render pipeline";
                if (error)
                    message += std::string(": ") + [[error localizedDescription] UTF8String];
                m_Context.error(message);
                return nullptr;
            }
        }

        MTLDepthStencilDescriptor* dd = [[MTLDepthStencilDescriptor alloc] init];
        dd.depthCompareFunction = desc.renderState.depthStencilState.depthTestEnable
            ? convertCompareFunction(desc.renderState.depthStencilState.depthFunc)
            : MTLCompareFunctionAlways;
        dd.depthWriteEnabled = desc.renderState.depthStencilState.depthWriteEnable;

        GraphicsPipeline* pipeline = new GraphicsPipeline();
        pipeline->desc = desc;
        pipeline->framebufferInfo = fbinfo;
        pipeline->pipeline = nativePipeline;
        pipeline->depthStencilState = [m_Context.device newDepthStencilStateWithDescriptor:dd];
        pipeline->primitiveType = convertPrimitiveType(desc.primType);
        pipeline->cullMode = convertCullMode(desc.renderState.rasterState.cullMode);
        pipeline->frontWinding = convertWinding(desc.renderState.rasterState.frontCounterClockwise);
        pipeline->usesGeometryEmulation = usesGeometryEmulation;
        if (pipeline->usesGeometryEmulation)
        {
            pipeline->geometryVertexSizeInBytes = geometryConfig.gsVertexSizeInBytes;
            pipeline->geometryMaxInputPrimitivesPerMeshThreadgroup =
                geometryConfig.gsMaxInputPrimitivesPerMeshThreadgroup;
            pipeline->geometryInstanceCount = gs->mscReflection.geometryInstanceCount;
            pipeline->objectBindingPlan = resolveMetalStageBindingPlan(vs->reflectedBindingPlan, desc.bindingLayouts);
            pipeline->meshBindingPlan = resolveMetalStageBindingPlan(gs->reflectedBindingPlan, desc.bindingLayouts);
        }

        // create vertex and fragment plan
        pipeline->vertexBindingPlan = resolveMetalStageBindingPlan(vs->reflectedBindingPlan, desc.bindingLayouts);
        if (ps)
            pipeline->fragmentBindingPlan = resolveMetalStageBindingPlan(ps->reflectedBindingPlan, desc.bindingLayouts);
        return GraphicsPipelineHandle::Create(pipeline);
    }

    GraphicsPipelineHandle Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc, IFramebuffer* fb)
    {
        return createGraphicsPipeline(desc, fb ? FramebufferInfo(fb->getDesc()) : FramebufferInfo());
    }

    Object GraphicsPipeline::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::MTL3_RenderPipeline)
            return Object((__bridge void*)pipeline);
        return nullptr;
    }
}
