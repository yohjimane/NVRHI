#include "metal3-backend.h"
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <cstring>
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>

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
        plan.directlyIndexedResourceHeap = reflection.directlyIndexedResourceHeap;
        plan.directlyIndexedSamplerHeap = reflection.directlyIndexedSamplerHeap;
        plan.resourceCount = reflection.resourceCount;
        plan.argumentBufferSize = reflection.argumentBufferSize;
        plan.descriptorTables = reflection.descriptorTables;
        for (auto& table : plan.descriptorTables)
        {
            if (table.descriptorCount)
            {
                plan.argumentBufferSize = (plan.argumentBufferSize + alignof(IRDescriptorTableEntry) - 1) &
                    ~(uint64_t(alignof(IRDescriptorTableEntry)) - 1);
                if (plan.argumentBufferSize > UINT32_MAX ||
                    uint64_t(table.descriptorCount) * sizeof(IRDescriptorTableEntry) > UINT32_MAX - plan.argumentBufferSize)
                {
                    plan.valid = false;
                    return plan;
                }
                table.descriptorOffset = uint32_t(plan.argumentBufferSize);
                plan.argumentBufferSize += uint64_t(table.descriptorCount) * sizeof(IRDescriptorTableEntry);
            }
        }
        plan.entries.reserve(reflection.topLevelArgumentBuffer.size());

        for (const MscArgumentBinding& binding : reflection.topLevelArgumentBuffer)
        {
            uint32_t offset = binding.byteOffset == ~0u ? binding.index * sizeof(IRDescriptorTableEntry) : binding.byteOffset;
            if (binding.tableIndex != ~0u)
            {
                if (binding.tableIndex >= plan.descriptorTables.size())
                {
                    plan.valid = false;
                    return plan;
                }
                offset += plan.descriptorTables[binding.tableIndex].descriptorOffset;
            }
            for (uint32_t element = 0; element < binding.sizeBytes / 24; ++element)
            {
                MetalBindingPlanEntry entry;
                entry.argumentIndex = binding.index + element;
                entry.slot = binding.slot + element;
                entry.space = binding.space;
                entry.argumentType = binding.type;
                entry.byteOffset = offset + element * 24;
                entry.sizeBytes = 24;
                plan.entries.push_back(entry);
            }
            plan.argumentBufferSize = std::max(plan.argumentBufferSize, uint64_t(offset) + binding.sizeBytes);
        }

        return plan;
    }

    // for every reflected resource, loop over binding layouts, reject ones with stage visibility
    // and/or layout space mismatch. per binding layout, loop over bindings, and assign the resource
    // cpu side, and move to the next after filling a MetalBindingPlanEntry struct
    MetalStageBindingPlan resolveMetalStageBindingPlan(const MetalStageBindingPlan& reflectedPlan, const BindingLayoutVector& pipelineLayouts)
    {
        MetalStageBindingPlan plan = reflectedPlan;
        for (auto& table : plan.descriptorTables)
        {
            if (table.descriptorCount)
                continue;
            for (uint32_t layoutIndex = 0; layoutIndex < pipelineLayouts.size(); ++layoutIndex)
            {
                const auto* desc = pipelineLayouts[layoutIndex]->getBindlessDesc();
                if (!desc || desc->layoutType != BindlessLayoutDesc::LayoutType::Immutable ||
                    (desc->visibility & plan.stage) == ShaderType::None || desc->firstSlot != table.slot)
                    continue;
                for (const auto& item : desc->registerSpaces)
                {
                    if (item.slot == table.space && argumentTypeMatchesResourceType(table.type, item.type))
                    {
                        table.layoutIndex = layoutIndex;
                        break;
                    }
                }
                if (table.layoutIndex != ~0u)
                    break;
            }
        }

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
            entry.texture = texture ? texture->getView(item.format, item.subresources, item.dimension,
                resolveComponentMapping(item.overrideComponentMapping, texture->desc.defaultComponentMapping)) : nil;
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
                if (item.type == ResourceType::TypedBuffer_SRV || item.type == ResourceType::TypedBuffer_UAV)
                {
                    const Format format = item.format == Format::UNKNOWN ? buffer->desc.format : item.format;
                    const FormatInfo& info = getFormatInfo(format);
                    const MTLPixelFormat pixelFormat = convertFormat(format);
                    const NSUInteger alignment = pixelFormat == MTLPixelFormatInvalid ? 0 :
                        [buffer->buffer.device minimumLinearTextureAlignmentForPixelFormat:pixelFormat];
                    if (info.bytesPerBlock && pixelFormat != MTLPixelFormatInvalid && alignment &&
                        range.byteOffset % info.bytesPerBlock == 0 && range.byteSize % info.bytesPerBlock == 0)
                    {
                        const NSUInteger alignedOffset = entry.bufferOffset / alignment * alignment;
                        entry.textureViewOffsetInElements = uint32_t((entry.bufferOffset - alignedOffset) / info.bytesPerBlock);
                        const NSUInteger width = (entry.bufferOffset - alignedOffset + entry.bufferSize) / info.bytesPerBlock;
                        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:pixelFormat
                            width:width resourceOptions:buffer->buffer.resourceOptions usage:MTLTextureUsageShaderRead |
                            (item.type == ResourceType::TypedBuffer_UAV ? MTLTextureUsageShaderWrite : MTLTextureUsageUnknown)];
                        entry.texture = [buffer->buffer newTextureWithDescriptor:descriptor offset:alignedOffset
                            bytesPerRow:width * info.bytesPerBlock];
                    }
                }
            }
            break;
        }
        default:
            break;
        }

        return entry;
    }

    bool encodeMetalBindingResource(IRDescriptorTableEntry* entry, const MetalBindingResource& resource)
    {
        if (resource.type == ResourceType::Texture_SRV || resource.type == ResourceType::Texture_UAV)
        {
            if (!resource.texture)
                return false;
            IRDescriptorTableSetTexture(entry, resource.texture, 0.f, 0);
            return true;
        }
        if (resource.type == ResourceType::Sampler)
        {
            if (!resource.sampler)
                return false;
            IRDescriptorTableSetSampler(entry, resource.sampler, resource.samplerMipBias);
            return true;
        }
        if (!resource.buffer)
            return false;
        IRBufferView view{};
        view.buffer = resource.buffer;
        view.bufferOffset = resource.bufferOffset;
        view.bufferSize = resource.bufferSize;
        view.textureBufferView = resource.texture;
        view.textureViewOffsetInElements = resource.textureViewOffsetInElements;
        view.typedBuffer = resource.type == ResourceType::TypedBuffer_SRV || resource.type == ResourceType::TypedBuffer_UAV;
        if (view.typedBuffer && !view.textureBufferView)
            return false;
        IRDescriptorTableSetBufferView(entry, &view);
        return true;
    }

    static bool supportsDescriptorType(ResourceType type)
    {
        switch (type)
        {
        case ResourceType::Texture_SRV:
        case ResourceType::Texture_UAV:
        case ResourceType::TypedBuffer_SRV:
        case ResourceType::TypedBuffer_UAV:
        case ResourceType::StructuredBuffer_SRV:
        case ResourceType::StructuredBuffer_UAV:
        case ResourceType::RawBuffer_SRV:
        case ResourceType::RawBuffer_UAV:
        case ResourceType::ConstantBuffer:
        case ResourceType::Sampler:
            return true;
        default:
            return false;
        }
    }

    BindingLayoutHandle Device::createBindlessLayout(const BindlessLayoutDesc& desc)
    {
        if (!desc.maxCapacity || desc.layoutType == BindlessLayoutDesc::LayoutType::MutableCounters ||
            (desc.layoutType == BindlessLayoutDesc::LayoutType::Immutable && desc.registerSpaces.empty()))
        {
            m_Context.error("[nvrhi] Invalid or unsupported Metal bindless layout.");
            return nullptr;
        }
        for (const BindingLayoutItem& item : desc.registerSpaces)
        {
            if (!supportsDescriptorType(item.type) ||
                ((item.type == ResourceType::Sampler) != (desc.registerSpaces.front().type == ResourceType::Sampler)))
            {
                m_Context.error("[nvrhi] Unsupported resource type in Metal bindless layout.");
                return nullptr;
            }
        }
        auto* layout = new BindingLayout();
        layout->isBindless = true;
        layout->bindlessDesc = desc;
        return BindingLayoutHandle::Create(layout);
    }

    DescriptorTableHandle Device::createDescriptorTable(IBindingLayout* layout)
    {
        if (!layout || !layout->getBindlessDesc())
            return nullptr;
        auto* table = new DescriptorTable();
        table->layout = layout;
        return DescriptorTableHandle::Create(table);
    }

    uint32_t DescriptorTable::getCapacity() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return uint32_t(entries.size());
    }

    void Device::resizeDescriptorTable(IDescriptorTable* descriptorTable, uint32_t newSize, bool keepContents)
    {
        auto* table = static_cast<DescriptorTable*>(descriptorTable);
        if (!table || newSize > table->layout->getBindlessDesc()->maxCapacity ||
            uint64_t(newSize) * sizeof(IRDescriptorTableEntry) > m_Context.device.maxBufferLength)
        {
            m_Context.error("[nvrhi] Invalid Metal descriptor table capacity.");
            return;
        }
        std::lock_guard<std::mutex> lock(table->mutex);
        if (!keepContents)
            table->entries.clear();
        table->entries.resize(newSize);
        table->snapshot.reset();
        ++table->version;
    }

    bool Device::writeDescriptorTable(IDescriptorTable* descriptorTable, const BindingSetItem& item)
    {
        auto* table = static_cast<DescriptorTable*>(descriptorTable);
        if (!table || item.arrayElement || (!supportsDescriptorType(item.type) && item.type != ResourceType::None))
            return false;
        const BindlessLayoutDesc& desc = *table->layout->getBindlessDesc();
        if (item.type != ResourceType::None)
        {
            bool allowed = desc.layoutType == BindlessLayoutDesc::LayoutType::MutableSampler
                ? item.type == ResourceType::Sampler
                : desc.layoutType == BindlessLayoutDesc::LayoutType::MutableSrvUavCbv && item.type != ResourceType::Sampler;
            for (const BindingLayoutItem& space : desc.registerSpaces)
                allowed |= space.type == item.type;
            if (!allowed)
                return false;
        }
        MetalBindingResource entry = normalizeBindingResource(item, 0);
        IRDescriptorTableEntry encoded{};
        if (item.resourceHandle && !encodeMetalBindingResource(&encoded, entry))
        {
            m_Context.error("[nvrhi] Failed to create Metal descriptor resource view.");
            return false;
        }
        std::lock_guard<std::mutex> lock(table->mutex);
        if (item.slot >= table->entries.size())
            return false;
        table->entries[item.slot] = std::move(entry);
        table->snapshot.reset();
        ++table->version;
        return true;
    }

    std::shared_ptr<DescriptorTableSnapshot> DescriptorTable::getSnapshot(const MTL3Context& context)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (snapshot || entries.empty())
            return snapshot;
        auto result = std::make_shared<DescriptorTableSnapshot>();
        result->buffer = [context.device newBufferWithLength:entries.size() * sizeof(IRDescriptorTableEntry)
            options:MTLResourceStorageModeShared];
        if (!result->buffer)
        {
            context.error("[nvrhi] Failed to allocate Metal descriptor table.");
            return nullptr;
        }
        result->entries.reserve(std::count_if(entries.begin(), entries.end(),
            [](const MetalBindingResource& entry) { return entry.resource != nullptr; }));
        auto* encoded = static_cast<IRDescriptorTableEntry*>(result->buffer.contents);
        std::memset(encoded, 0, entries.size() * sizeof(IRDescriptorTableEntry));
        for (size_t index = 0; index < entries.size(); ++index)
            if (entries[index].resource)
            {
                encodeMetalBindingResource(encoded + index, entries[index]);
                result->entries.push_back(entries[index]);
            }
        snapshot = result;
        return result;
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
        if (!layout || !layout->getDesc())
            return nullptr;
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

            MetalBindingResource resource = normalizeBindingResource(item, registerSpace);
            IRDescriptorTableEntry encoded{};
            if (item.resourceHandle && item.type != ResourceType::VolatileConstantBuffer &&
                !encodeMetalBindingResource(&encoded, resource))
            {
                m_Context.error("[nvrhi] Failed to create Metal binding-set resource view.");
                delete set;
                return nullptr;
            }
            set->entries.push_back(std::move(resource));
        }

        return BindingSetHandle::Create(set);
    }
}
