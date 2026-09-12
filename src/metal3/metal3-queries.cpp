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
            page.freeSamples.clear();
        }
    }

    TimerQueryHandle Device::createTimerQuery()
    {
        @autoreleasepool
        {
            const auto& pool = m_TimerQueryPool;
            std::lock_guard<std::mutex> lock(pool->mutex);
            if (!pool->frequency)
            {
                uint64_t frequency = 0;
                if (@available(macOS 26.0, *))
                    frequency = [m_Context.device queryTimestampFrequency];
                if (!frequency || ![m_Context.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
                {
                    m_Context.error("[nvrhi] Native GPU timer queries require timestamp frequency and encoder-boundary sampling support.");
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
                pool->frequency = frequency;
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
                desc.sampleCount = 2 * TimerQueryPool::QueriesPerPage;
                desc.label = @"NVRHI GPU timer page";
                NSError* error = nil;
                id<MTLCounterSampleBuffer> samples = [m_Context.device newCounterSampleBufferWithDescriptor:desc error:&error];
                if (!samples)
                {
                    m_Context.error(std::string("[nvrhi] Cannot allocate GPU timestamp samples: ") +
                        (error ? error.localizedDescription.UTF8String : "unknown Metal error"));
                    return nullptr;
                }
                pageIndex = emptyPageIndex;
                if (pageIndex == pool->pages.size())
                    pool->pages.emplace_back();
                auto& page = pool->pages[pageIndex];
                page.samples = samples;
                page.freeSamples.reserve(TimerQueryPool::QueriesPerPage);
                for (NSUInteger index = TimerQueryPool::QueriesPerPage; index != 0; --index)
                    page.freeSamples.push_back((index - 1) * 2);
            }
            auto& page = pool->pages[pageIndex];
            auto* query = new TimerQuery();
            query->pool = pool;
            query->pageIndex = pageIndex;
            query->sampleIndex = page.freeSamples.back();
            page.freeSamples.pop_back();
            query->samples = page.samples;
            query->frequency = pool->frequency;
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
            NSData* result = [query->samples resolveCounterRange:NSMakeRange(query->sampleIndex, 2)];
            if (result.length != 2 * sizeof(MTLCounterResultTimestamp))
            {
                m_Context.error("[nvrhi] Cannot resolve GPU timestamp samples.");
                return std::numeric_limits<float>::quiet_NaN();
            }
            const auto* values = static_cast<const MTLCounterResultTimestamp*>(result.bytes);
            if (values[0].timestamp == MTLCounterErrorValue || values[1].timestamp == MTLCounterErrorValue ||
                values[1].timestamp < values[0].timestamp)
            {
                m_Context.error("[nvrhi] GPU timestamp samples are invalid.");
                return std::numeric_limits<float>::quiet_NaN();
            }
            return float(double(values[1].timestamp - values[0].timestamp) / double(query->frequency));
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

    void CommandList::beginTimerQuery(ITimerQuery* abstractQuery)
    {
        auto* query = static_cast<TimerQuery*>(abstractQuery);
        if (!query || query->started || m_RecordingState != RecordingState::Open || !trackedCmdBuffer)
        {
            m_RecordingFailed = true;
            m_Context.error("[nvrhi] GPU timer begin requires an open command list and an unused query.");
            return;
        }
        endEncoding();
        MTLBlitPassDescriptor* pass = [MTLBlitPassDescriptor blitPassDescriptor];
        pass.sampleBufferAttachments[0].sampleBuffer = query->samples;
        pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = query->sampleIndex;
        pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = MTLCounterDontSample;
        id<MTLBlitCommandEncoder> encoder = [trackedCmdBuffer blitCommandEncoderWithDescriptor:pass];
        if (!encoder)
        {
            m_RecordingFailed = true;
            m_Context.error("[nvrhi] Cannot encode GPU timer begin sample.");
            return;
        }
        [encoder endEncoding];
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
        endEncoding();
        MTLBlitPassDescriptor* pass = [MTLBlitPassDescriptor blitPassDescriptor];
        pass.sampleBufferAttachments[0].sampleBuffer = query->samples;
        pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = MTLCounterDontSample;
        pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = query->sampleIndex + 1;
        id<MTLBlitCommandEncoder> encoder = [trackedCmdBuffer blitCommandEncoderWithDescriptor:pass];
        if (!encoder)
        {
            m_RecordingFailed = true;
            m_Context.error("[nvrhi] Cannot encode GPU timer end sample.");
            return;
        }
        [encoder endEncoding];
        query->ended = true;
    }
}
