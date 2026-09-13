#include "metal3-backend.h"
#include <limits>

namespace nvrhi::metal3
{
    TimerQuery::~TimerQuery()
    {
        if (!pool)
            return;
        std::lock_guard<std::mutex> lock(pool->mutex);
        auto& page = pool->pages[pageIndex];
        page.freeSamples.push_back(sampleIndex);
        if (pageIndex != 0 && page.freeSamples.size() == TimerQueryPool::QueriesPerPage)
        {
            page.samples = nil;
            page.markers = nil;
            page.freeSamples.clear();
        }
    }

    TimerQueryHandle Device::createTimerQuery()
    {
        @autoreleasepool
        {
            const auto& pool = m_TimerQueryPool;
            std::lock_guard<std::mutex> lock(pool->mutex);
            if (!pool->markerPipeline)
            {
                if (![m_Context.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
                {
                    m_Context.error("[nvrhi] Native GPU timer queries require encoder-boundary sampling support.");
                    return nullptr;
                }
                for (id<MTLCounterSet> counterSet in m_Context.device.counterSets)
                {
                    if ([counterSet.name isEqualToString:MTLCommonCounterSetTimestamp])
                    {
                        pool->counterSet = counterSet;
                        break;
                    }
                }
                if (!pool->counterSet)
                {
                    m_Context.error("[nvrhi] The Metal device does not expose GPU timestamp counters.");
                    return nullptr;
                }
                NSError* error = nil;
                NSString* source = @"#include <metal_stdlib>\n"
                    "using namespace metal; kernel void nvrhi_timer_marker(device uint* marker [[buffer(0)]])"
                    "{ marker[0] = 1; }";
                id<MTLLibrary> library = [m_Context.device newLibraryWithSource:source options:nil error:&error];
                id<MTLFunction> function = [library newFunctionWithName:@"nvrhi_timer_marker"];
                id<MTLComputePipelineState> pipeline = function
                    ? [m_Context.device newComputePipelineStateWithFunction:function error:&error] : nil;
                if (!pipeline)
                {
                    m_Context.error(std::string("[nvrhi] Cannot create GPU timestamp boundary pipeline: ") +
                        (error ? error.localizedDescription.UTF8String : "missing marker function"));
                    return nullptr;
                }
                pool->markerPipeline = pipeline;
                pool->timestampsInNanoseconds = [m_Context.device supportsFamily:MTLGPUFamilyApple1];
            }
            size_t pageIndex = 0;
            size_t emptyPageIndex = pool->pages.size();
            for (; pageIndex < pool->pages.size(); ++pageIndex)
            {
                const auto& page = pool->pages[pageIndex];
                if (!page.freeSamples.empty())
                    break;
                if (!page.samples && emptyPageIndex == pool->pages.size())
                    emptyPageIndex = pageIndex;
            }
            if (pageIndex == pool->pages.size())
            {
                MTLCounterSampleBufferDescriptor* desc = [[MTLCounterSampleBufferDescriptor alloc] init];
                desc.counterSet = pool->counterSet;
                desc.storageMode = MTLStorageModeShared;
                desc.sampleCount = TimerQueryPool::SamplesPerQuery * TimerQueryPool::QueriesPerPage;
                desc.label = @"NVRHI GPU timer page";
                NSError* error = nil;
                id<MTLCounterSampleBuffer> samples = [m_Context.device newCounterSampleBufferWithDescriptor:desc error:&error];
                if (!samples)
                {
                    m_Context.error(std::string("[nvrhi] Cannot allocate GPU timestamp samples: ") +
                        (error ? error.localizedDescription.UTF8String : "unknown Metal error"));
                    return nullptr;
                }
                id<MTLBuffer> markers = [m_Context.device newBufferWithLength:TimerQueryPool::QueriesPerPage * sizeof(uint32_t)
                    options:MTLResourceStorageModePrivate];
                if (!markers)
                {
                    m_Context.error("[nvrhi] Cannot allocate GPU timestamp marker storage.");
                    return nullptr;
                }
                markers.label = @"NVRHI GPU timer markers";
                pageIndex = emptyPageIndex;
                if (pageIndex == pool->pages.size())
                    pool->pages.emplace_back();
                auto& page = pool->pages[pageIndex];
                page.samples = samples;
                page.markers = markers;
                page.freeSamples.reserve(TimerQueryPool::QueriesPerPage);
                for (NSUInteger index = TimerQueryPool::QueriesPerPage; index != 0; --index)
                    page.freeSamples.push_back((index - 1) * TimerQueryPool::SamplesPerQuery);
            }
            auto& page = pool->pages[pageIndex];
            auto* query = new TimerQuery();
            query->pool = pool;
            query->pageIndex = pageIndex;
            query->sampleIndex = page.freeSamples.back();
            page.freeSamples.pop_back();
            query->samples = page.samples;
            query->markers = page.markers;
            return TimerQueryHandle::Create(query);
        }
    }

    bool Device::pollTimerQuery(ITimerQuery* abstractQuery)
    {
        auto* query = static_cast<TimerQuery*>(abstractQuery);
        return query && query->ended && query->commandBuffer &&
            query->commandBuffer.status == MTLCommandBufferStatusCompleted;
    }

    float Device::getTimerQueryTime(ITimerQuery* abstractQuery)
    {
        @autoreleasepool
        {
            if (!pollTimerQuery(abstractQuery))
            {
                m_Context.error("[nvrhi] GPU timer results requested before completion.");
                return std::numeric_limits<float>::quiet_NaN();
            }
            auto* query = static_cast<TimerQuery*>(abstractQuery);
            NSData* data = [query->samples resolveCounterRange:NSMakeRange(query->sampleIndex, TimerQueryPool::SamplesPerQuery)];
            if (data.length != TimerQueryPool::SamplesPerQuery * sizeof(MTLCounterResultTimestamp))
            {
                m_Context.error("[nvrhi] Cannot resolve completed GPU timestamp samples.");
                return std::numeric_limits<float>::quiet_NaN();
            }
            const auto* values = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
            for (NSUInteger index = 0; index < TimerQueryPool::SamplesPerQuery; ++index)
            {
                if (!values[index].timestamp || values[index].timestamp == MTLCounterErrorValue ||
                    (index && values[index].timestamp < values[index - 1].timestamp))
                {
                    m_Context.error("[nvrhi] GPU timestamp samples are invalid: index=" + std::to_string(query->sampleIndex + index) +
                        " start=" + std::to_string(values[0].timestamp) +
                        " end=" + std::to_string(values[TimerQueryPool::SamplesPerQuery - 1].timestamp));
                    return std::numeric_limits<float>::quiet_NaN();
                }
            }
            double elapsed = double(values[TimerQueryPool::SamplesPerQuery - 1].timestamp - values[1].timestamp);
            if (!query->pool->timestampsInNanoseconds)
            {
                MTLTimestamp cpuTimestamp = 0;
                MTLTimestamp gpuTimestamp = 0;
                [m_Context.device sampleTimestamps:&cpuTimestamp gpuTimestamp:&gpuTimestamp];
                if (cpuTimestamp <= query->cpuStartTimestamp || gpuTimestamp <= query->gpuStartTimestamp)
                {
                    m_Context.error("[nvrhi] Cannot calibrate GPU timestamp samples.");
                    return std::numeric_limits<float>::quiet_NaN();
                }
                elapsed *= double(cpuTimestamp - query->cpuStartTimestamp) /
                    double(gpuTimestamp - query->gpuStartTimestamp);
            }
            return float(elapsed * 1e-9);
        }
    }

    void Device::resetTimerQuery(ITimerQuery* abstractQuery)
    {
        auto* query = static_cast<TimerQuery*>(abstractQuery);
        if (!query)
            return;
        if (query->started && (!query->ended || (query->commandBuffer &&
            query->commandBuffer.status != MTLCommandBufferStatusCompleted &&
            query->commandBuffer.status != MTLCommandBufferStatusError)))
        {
            m_Context.error("[nvrhi] Cannot reset an active GPU timer query.");
            return;
        }
        query->commandBuffer = nil;
        query->started = false;
        query->ended = false;
    }

    bool CommandList::encodeTimerBoundary(TimerQuery* query, bool ending)
    {
        endEncoding();
        if (m_RecordingFailed)
            return false;
        id<MTLFence> boundaryFence = m_TimerBoundaryFence;
        if (!boundaryFence)
            boundaryFence = createTimerFence();
        if (!boundaryFence)
            return false;
        const NSUInteger firstSample = query->sampleIndex + (ending ? 2 : 0);
        MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
        pass.sampleBufferAttachments[0].sampleBuffer = query->samples;
        pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = firstSample;
        pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = firstSample + 1;
        id<MTLComputeCommandEncoder> encoder = [trackedCmdBuffer computeCommandEncoderWithDescriptor:pass];
        if (!encoder)
        {
            m_RecordingFailed = true;
            m_Context.error("[nvrhi] Cannot encode GPU timestamp boundary.");
            return false;
        }
        annotateEncoder(encoder, ending ? "endTimerQuery" : "beginTimerQuery");
        if (m_WorkloadCompletionFence)
            [encoder waitForFence:m_WorkloadCompletionFence];
        if (m_TimerBoundaryFence)
            [encoder waitForFence:m_TimerBoundaryFence];
        [encoder setComputePipelineState:query->pool->markerPipeline];
        [encoder setBuffer:query->markers
            offset:(query->sampleIndex / TimerQueryPool::SamplesPerQuery) * sizeof(uint32_t) atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder updateFence:boundaryFence];
        [encoder endEncoding];
        m_TimerBoundaryFence = boundaryFence;
        return true;
    }

    void CommandList::beginTimerQuery(ITimerQuery* abstractQuery)
    {
        auto* query = static_cast<TimerQuery*>(abstractQuery);
        if (!query || query->started || m_RecordingState != RecordingState::Open || !trackedCmdBuffer)
        {
            m_RecordingFailed = true;
            m_Context.error("[nvrhi] GPU timer begin requires an open command list and an unused query.");
            return;
        }
        if (!query->pool->timestampsInNanoseconds)
            [m_Context.device sampleTimestamps:&query->cpuStartTimestamp gpuTimestamp:&query->gpuStartTimestamp];
        if (!encodeTimerBoundary(query, false))
            return;
        query->commandBuffer = trackedCmdBuffer;
        query->started = true;
        m_ReferencedStateResources.emplace_back(query);
    }

    void CommandList::endTimerQuery(ITimerQuery* abstractQuery)
    {
        auto* query = static_cast<TimerQuery*>(abstractQuery);
        if (!query || !query->started || query->ended || query->commandBuffer != trackedCmdBuffer ||
            m_RecordingState != RecordingState::Open)
        {
            m_RecordingFailed = true;
            m_Context.error("[nvrhi] GPU timer end requires the matching active query and command list.");
            return;
        }
        if (!encodeTimerBoundary(query, true))
            return;
        query->ended = true;
    }
}
