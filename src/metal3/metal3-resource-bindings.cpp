#include "metal3-backend.h"
#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace nvrhi::metal3
{
    static bool argumentTypeMatchesResourceType(MscArgumentType argumentType, ResourceType resourceType)
    {
        switch (argumentType)
        {
        case MscArgumentType::SRV:
            return resourceType == ResourceType::Texture_SRV ||
                resourceType == ResourceType::TypedBuffer_SRV ||
                resourceType == ResourceType::StructuredBuffer_SRV ||
                resourceType == ResourceType::RawBuffer_SRV ||
                resourceType == ResourceType::RayTracingAccelStruct;
        case MscArgumentType::UAV:
            return resourceType == ResourceType::Texture_UAV ||
                resourceType == ResourceType::TypedBuffer_UAV ||
                resourceType == ResourceType::StructuredBuffer_UAV ||
                resourceType == ResourceType::RawBuffer_UAV ||
                resourceType == ResourceType::SamplerFeedbackTexture_UAV;
        case MscArgumentType::CBV:
            return resourceType == ResourceType::ConstantBuffer ||
                resourceType == ResourceType::VolatileConstantBuffer;
        case MscArgumentType::Sampler:
            return resourceType == ResourceType::Sampler;
        default:
            return false;
        }
    }

    static bool layoutVisibleToStage(const BindingLayoutDesc& desc, ShaderType stage)
    {
        return (desc.visibility & stage) != ShaderType::None;
    }

    // the way it should map is follows:
    // the default register space is space0 both GPU and CPU side code
    // if we provide the space with setRegisterSpaceAndDescriptorSet(val) explicitly,
    // the GPU reflected shader should match the same space too. else error out
    static bool layoutSpaceMayMatch(const BindingLayoutDesc& desc, uint32_t layoutIndex, uint32_t reflectedSpace)
    {
        if (desc.registerSpace == reflectedSpace)
            return true;
        return false;
    }

    MetalStageBindingPlan createMetalStageBindingPlan(ShaderType stage, const MscShaderReflection& reflection)
    {
        MetalStageBindingPlan plan;
        plan.stage = stage;
        plan.valid = reflection.valid;
        plan.resourceCount = reflection.resourceCount;
        plan.entries.reserve(reflection.topLevelArgumentBuffer.size());

        for (const MscArgumentBinding& binding : reflection.topLevelArgumentBuffer)
        {
            MetalBindingPlanEntry entry;
            entry.argumentIndex = binding.index;
            entry.slot = binding.slot;
            entry.space = binding.space;
            entry.argumentType = binding.type;
            plan.entries.push_back(entry);
        }

        return plan;
    }

    // for every reflected resource, loop over binding layouts, reject ones with stage visibility
    // and/or layout space mismatch. per binding layout, loop over bindings, and assign the resource
    // cpu side, and move to the next after filling a MetalBindingPlanEntry struct
    MetalStageBindingPlan resolveMetalStageBindingPlan(const MetalStageBindingPlan& reflectedPlan, const BindingLayoutVector& pipelineLayouts)
    {
        MetalStageBindingPlan plan = reflectedPlan;

        for (MetalBindingPlanEntry& entry : plan.entries)
        {
            for (uint32_t layoutIndex = 0; layoutIndex < pipelineLayouts.size() && !entry.layoutMatched; ++layoutIndex)
            {
                auto* layout = static_cast<BindingLayout*>(pipelineLayouts[layoutIndex].Get());
                if (!layout || layout->isBindless)
                    continue;

                const BindingLayoutDesc& layoutDesc = layout->desc;
                if (!layoutVisibleToStage(layoutDesc, plan.stage) ||
                    !layoutSpaceMayMatch(layoutDesc, layoutIndex, entry.space))
                    continue;

                for (uint32_t itemIndex = 0; itemIndex < layoutDesc.bindings.size(); ++itemIndex)
                {
                    const BindingLayoutItem& item = layoutDesc.bindings[itemIndex];
                    const uint32_t itemArraySize = std::max(1u, item.getArraySize());
                    // check slot range AND type to find a match, else reject and move to next item
                    // in the binding set
                    if (entry.slot < item.slot ||
                        entry.slot >= item.slot + itemArraySize ||
                        !argumentTypeMatchesResourceType(entry.argumentType, item.type))
                        continue;

                    entry.layoutIndex = layoutIndex;
                    entry.layoutItemIndex = itemIndex;
                    entry.layoutType = item.type;
                    entry.layoutMatched = true;
                    break;
                }
            }
        }

        return plan;
    }
    static MTLResourceUsage usageForBinding(ResourceType type)
    {
        switch (type)
        {
        case ResourceType::Texture_UAV:
        case ResourceType::TypedBuffer_UAV:
        case ResourceType::StructuredBuffer_UAV:
        case ResourceType::RawBuffer_UAV:
            return MTLResourceUsageRead | MTLResourceUsageWrite;
        default:
            return MTLResourceUsageRead;
        }
    }

    // create a metal resource cache per nvrhi binding item
    static MetalBindingResource normalizeBindingResource(const BindingSetItem& item, uint32_t registerSpace)
    {
        MetalBindingResource entry;
        entry.type = item.type;
        entry.slot = item.slot;
        entry.arrayElement = item.arrayElement;
        entry.registerSpace = registerSpace;
        entry.usage = usageForBinding(item.type);

        if (item.resourceHandle)
            entry.resource = item.resourceHandle;

        switch (item.type)
        {
        // for textures, store id<MTLTexture>
        case ResourceType::Texture_SRV:
        case ResourceType::Texture_UAV:
        {
            auto* texture = static_cast<Texture*>(item.resourceHandle);
            entry.texture = texture ? texture->texture : nil;
            break;
        }
        // for samplers, extract and store id<MTLSamplerState + mip bias
        case ResourceType::Sampler:
        {
            auto* sampler = static_cast<Sampler*>(item.resourceHandle);
            entry.sampler = sampler ? sampler->sampler : nil;
            entry.samplerMipBias = sampler ? sampler->desc.mipBias : 0.f;
            break;
        }

        // for all types of buffers, extract id<MTLBuffer>, offset, size
        case ResourceType::ConstantBuffer:
        case ResourceType::VolatileConstantBuffer:
        case ResourceType::StructuredBuffer_SRV:
        case ResourceType::StructuredBuffer_UAV:
        case ResourceType::TypedBuffer_SRV:
        case ResourceType::TypedBuffer_UAV:
        case ResourceType::RawBuffer_SRV:
        case ResourceType::RawBuffer_UAV:
        {
            auto* buffer = static_cast<Buffer*>(item.resourceHandle);
            if (buffer)
            {
                const BufferRange range = item.range.resolve(buffer->desc);
                entry.buffer = buffer->buffer;
                entry.bufferOffset = NSUInteger(range.byteOffset);
                entry.bufferSize = NSUInteger(range.byteSize);
            }
            break;
        }
        default:
            break;
        }

        return entry;
    }
    SamplerHandle Device::createSampler(const SamplerDesc& d)
    {
        MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
        sd.minFilter = d.minFilter ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        sd.magFilter = d.magFilter ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        sd.mipFilter = d.mipFilter ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
        sd.sAddressMode = convertSamplerAddressMode(d.addressU);
        sd.tAddressMode = convertSamplerAddressMode(d.addressV);
        sd.rAddressMode = convertSamplerAddressMode(d.addressW);
        sd.lodBias = d.mipBias;
        sd.maxAnisotropy = NSUInteger(std::max(1.f, d.maxAnisotropy));
        sd.supportArgumentBuffers = YES;

        Sampler* sampler = new Sampler();
        sampler->desc = d;
        sampler->sampler = [m_Context.device newSamplerStateWithDescriptor:sd];
        if (!sampler->sampler)
        {
            delete sampler;
            return nullptr;
        }
        return SamplerHandle::Create(sampler);
    }

    BindingLayoutHandle Device::createBindingLayout(const BindingLayoutDesc& desc)
    {
        BindingLayout* layout = new BindingLayout();
        layout->desc = desc;
        return BindingLayoutHandle::Create(layout);
    }

    BindingSetHandle Device::createBindingSet(const BindingSetDesc& desc, IBindingLayout* layout)
    {
        BindingSet* set = new BindingSet();
        set->desc = desc;
        set->layout = layout;
        set->version = 1;

        uint32_t registerSpace = 0;
        if (layout && layout->getDesc())
            registerSpace = layout->getDesc()->registerSpace;

        set->resources.reserve(desc.bindings.size());
        set->entries.reserve(desc.bindings.size());
        for (const BindingSetItem& item : desc.bindings)
        {
            if (item.resourceHandle)
                set->resources.emplace_back(item.resourceHandle);

            set->entries.push_back(normalizeBindingResource(item, registerSpace));
        }

        return BindingSetHandle::Create(set);
    }
}
