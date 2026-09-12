#include "metal3-backend.h"

namespace nvrhi::metal3
{
    BufferHandle Device::createBuffer(const BufferDesc& d)
    {
        if (d.byteSize == 0)
            return nullptr;

        Buffer* buffer = new Buffer();
        buffer->desc = d;

        if (d.isVolatile)
            return BufferHandle::Create(buffer);

        MTLResourceOptions options = convertCpuAccess(d.cpuAccess);
        buffer->buffer = [m_Context.device newBufferWithLength:NSUInteger(d.byteSize) options:options];
        if (!buffer->buffer)
        {
            delete buffer;
            m_Context.error("[nvrhi] Failed to create Metal buffer.");
            return nullptr;
        }

        if (!d.debugName.empty())
            buffer->buffer.label = [NSString stringWithUTF8String:d.debugName.c_str()];

        return BufferHandle::Create(buffer);
    }

    Object Buffer::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::MTL3_Buffer)
            return Object((__bridge void*)buffer);
        return nullptr;
    }
}
