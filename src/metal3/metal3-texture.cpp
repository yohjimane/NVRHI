#include "metal3-backend.h"

namespace nvrhi::metal3
{
    static MTLTextureUsage textureUsageFromDesc(const TextureDesc& desc)
    {
        MTLTextureUsage usage = MTLTextureUsageUnknown;
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
        if (pixelFormat == MTLPixelFormatInvalid)
        {
            m_Context.error("[nvrhi] Unsupported Metal texture format for texture '" + d.debugName +
                "' (format=" + std::to_string(static_cast<int>(d.format)) + ").");
            return nullptr;
        }

        if (d.isTiled || d.isVirtual)
        {
            m_Context.error("[nvrhi] Metal3 tiled/virtual textures are not implemented.");
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

        Texture* texture = new Texture();
        // cache tex descriptor size and alignment for use for device.getTextureMemoryRequirements(...)
        texture->memSize = sizeAndAlign.size;
        texture->memAlign = sizeAndAlign.align;

        id<MTLTexture> nativeTexture = [m_Context.device newTextureWithDescriptor:td];
        if (!nativeTexture)
        {
            m_Context.error("[nvrhi] Failed to create Metal texture.");
            return nullptr;
        }

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

    Object Texture::getNativeView(ObjectType objectType, Format format, TextureSubresourceSet subresources, TextureDimension dimension, bool isReadOnlyDSV)
    {
        (void)format; (void)subresources; (void)dimension; (void)isReadOnlyDSV;
        return getNativeObject(objectType);
    }
}
