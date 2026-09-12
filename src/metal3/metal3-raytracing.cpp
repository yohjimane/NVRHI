#include "metal3-backend.h"

namespace nvrhi::metal3
{

    rt::PipelineHandle Device::createRayTracingPipeline(const rt::PipelineDesc& desc)
    {
        m_Context.unsupported(__func__);
        return nullptr;
    }
}
