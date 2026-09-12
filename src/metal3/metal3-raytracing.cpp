#include "metal3-backend.h"

namespace nvrhi::metal3
{

    rt::PipelineHandle Device::createRayTracingPipeline(const rt::PipelineDesc& desc)
    {
        RayTracingPipeline* pipeline = new RayTracingPipeline();
        pipeline->desc = desc;
        return rt::PipelineHandle::Create(pipeline);
    }
}
