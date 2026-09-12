#include "metal3-backend.h"

namespace nvrhi::metal3
{
    static MTLTextureUsage textureUsageFromDesc(const TextureDesc& desc)
    {
        MTLTextureUsage usage = MTLTextureUsagePixelFormatView;
        if (desc.isShaderResource)
            usage |= MTLTextureUsageShaderRead;
        if (desc.isUAV)
            usage |= MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
        if (desc.isRenderTarget)
            usage |= MTLTextureUsageRenderTarget;
        return usage == MTLTextureUsageUnknown ? MTLTextureUsageShaderRead : usage;
    }
    // other APIs (vk/dx11/dx12) all imply using arraysize = 6 for texturecube explicitly
    // in case of metal3, the TextureCube, value of arraysize = 1, implies 6 textures implicitly
    static NSUInteger metalArrayLengthFromDesc(const TextureDesc& desc)
    {
        switch (desc.dimension)
        {
        case TextureDimension::TextureCube:
            return 1;
        case TextureDimension::TextureCubeArray:
            return desc.arraySize / 6u;
        default:
            return desc.arraySize;
        }
    }

    TextureHandle Device::createTexture(const TextureDesc& d)
    {
        MTLPixelFormat pixelFormat = convertFormat(d.format);
        const FormatSupport support = queryFormatSupport(d.format);
        if ((support & FormatSupport::Texture) == FormatSupport::None
            || (d.isRenderTarget && (support & (FormatSupport::DepthStencil | FormatSupport::RenderTarget)) == FormatSupport::None)
            || (d.isUAV && (support & FormatSupport::ShaderUavStore) == FormatSupport::None))
        {
            m_Context.error("[nvrhi] Unsupported Metal texture format for texture '" + d.debugName +
                "' (format=" + std::to_string(static_cast<int>(d.format)) + ").");
            return nullptr;
        }

        if (d.isTiled || d.isVirtual || d.isShadingRateSurface || d.sharedResourceFlags != SharedResourceFlags::None)
        {
            m_Context.error("[nvrhi] Metal tiled, virtual, shading-rate, and shared textures are unsupported.");
            return nullptr;
        }

        const uint32_t maxDimension = d.dimension == TextureDimension::Texture3D ? 2048 : m_Context.maxTextureDimension;
        if (!d.width || !d.height || !d.depth || !d.arraySize || !d.mipLevels
            || d.width > maxDimension || d.height > maxDimension || d.depth > 2048 || d.arraySize > 2048
            || ![m_Context.device supportsTextureSampleCount:d.sampleCount]
            || (d.sampleCount > 1 && (d.mipLevels != 1 || d.isUAV)))
        {
            m_Context.error("[nvrhi] Metal texture dimensions, sample count, or multisample usage exceed supported limits.");
            return nullptr;
        }

        MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
        td.textureType = convertTextureDimension(d.dimension, d.sampleCount);
        td.pixelFormat = pixelFormat;
        td.width = d.width;
        td.height = d.height;
        td.depth = d.depth;
        td.mipmapLevelCount = d.mipLevels;
        td.arrayLength = metalArrayLengthFromDesc(d);
        td.sampleCount = d.sampleCount;
        td.usage = textureUsageFromDesc(d);
        
        // always create private textures, fill them using command lists if CPU data needs be written
        td.storageMode = MTLStorageModePrivate;

        MTLSizeAndAlign sizeAndAlign = [m_Context.device heapTextureSizeAndAlignWithDescriptor:td];

        id<MTLTexture> nativeTexture = [m_Context.device newTextureWithDescriptor:td];
        if (!nativeTexture)
        {
            m_Context.error("[nvrhi] Failed to create Metal texture.");
            return nullptr;
        }

        Texture* texture = new Texture();
        texture->memSize = sizeAndAlign.size;
        texture->memAlign = sizeAndAlign.align;
        if (!d.debugName.empty())
            nativeTexture.label = [NSString stringWithUTF8String:d.debugName.c_str()];

        texture->desc = d;
        texture->texture = nativeTexture;
        return TextureHandle::Create(texture);
    }
    
    // useful to create handle for textures created with native metal3, like for swapchains, etc
    TextureHandle Device::createHandleForNativeTexture(ObjectType objectType, Object nativeTexture, const TextureDesc& desc)
    {
        if (objectType != ObjectTypes::MTL3_Texture)
            return nullptr;

        Texture* texture = new Texture();
        texture->desc = desc;
        texture->texture = (__bridge id<MTLTexture>)nativeTexture.pointer;
        texture->ownsTexture = false;
        return TextureHandle::Create(texture);
    }

    Object Texture::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::MTL3_Texture)
            return Object((__bridge void*)texture);
        return nullptr;
    }

    static MTLTextureSwizzle convertSwizzle(ComponentSwizzle swizzle)
    {
        switch (swizzle)
        {
        case ComponentSwizzle::R: return MTLTextureSwizzleRed;
        case ComponentSwizzle::G: return MTLTextureSwizzleGreen;
        case ComponentSwizzle::B: return MTLTextureSwizzleBlue;
        case ComponentSwizzle::A: return MTLTextureSwizzleAlpha;
        case ComponentSwizzle::Zero: return MTLTextureSwizzleZero;
        case ComponentSwizzle::One: return MTLTextureSwizzleOne;
        }
        return MTLTextureSwizzleZero;
    }

    id<MTLTexture> Texture::getView(Format format, TextureSubresourceSet subresources, TextureDimension dimension,
        std::optional<ComponentMapping> componentMapping)
    {
        if (!texture)
            return nil;
        const ComponentMapping mapping = resolveComponentMapping(componentMapping, desc.defaultComponentMapping);
        format = format == Format::UNKNOWN ? desc.format : format;
        dimension = dimension == TextureDimension::Unknown ? desc.dimension : dimension;
        if (format == desc.format && dimension == desc.dimension && subresources.isEntireTexture(desc) && mapping.isIdentity())
            return texture;
        subresources = subresources.resolve(desc, false);
        if (!subresources.numMipLevels || !subresources.numArraySlices)
            return nil;
        const MTLPixelFormat pixelFormat = convertFormat(format);
        if (pixelFormat == MTLPixelFormatInvalid)
            return nil;
        const MTLTextureType type = convertTextureDimension(dimension, desc.sampleCount);
        const MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsMake(convertSwizzle(mapping.r),
            convertSwizzle(mapping.g), convertSwizzle(mapping.b), convertSwizzle(mapping.a));
        const NSUInteger sliceCount = dimension == TextureDimension::Texture3D ? 1 : subresources.numArraySlices;
        std::lock_guard<std::mutex> lock(viewMutex);
        for (id<MTLTexture> view : views)
        {
            const MTLTextureSwizzleChannels existing = view.swizzle;
            const NSUInteger viewSlices = (type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray) ? view.arrayLength * 6 : view.arrayLength;
            if (view.pixelFormat == pixelFormat && view.textureType == type &&
                view.parentRelativeLevel == subresources.baseMipLevel &&
                view.mipmapLevelCount == subresources.numMipLevels &&
                view.parentRelativeSlice == subresources.baseArraySlice && viewSlices == sliceCount &&
                existing.red == swizzle.red && existing.green == swizzle.green &&
                existing.blue == swizzle.blue && existing.alpha == swizzle.alpha)
                return view;
        }
        id<MTLTexture> view = [texture newTextureViewWithPixelFormat:pixelFormat textureType:type
            levels:NSMakeRange(subresources.baseMipLevel, subresources.numMipLevels)
            slices:NSMakeRange(subresources.baseArraySlice, sliceCount) swizzle:swizzle];
        if (view)
            views.push_back(view);
        return view;
    }

    Object Texture::getNativeView(ObjectType objectType, Format format, TextureSubresourceSet subresources, TextureDimension dimension, bool, std::optional<ComponentMapping> overrideComponentMapping)
    {
        if (objectType != ObjectTypes::MTL3_Texture)
            return nullptr;
        return Object((__bridge void*)getView(format, subresources, dimension, overrideComponentMapping));
    }
}
