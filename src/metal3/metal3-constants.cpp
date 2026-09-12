#include "metal3-backend.h"

namespace nvrhi::metal3
{
    MTLTextureType convertTextureDimension(TextureDimension dimension, uint32_t sampleCount)
    {
        switch (dimension)
        {
        case TextureDimension::Texture1D: return MTLTextureType1D;
        case TextureDimension::Texture1DArray: return MTLTextureType1DArray;
        case TextureDimension::Texture2D: return sampleCount > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
        case TextureDimension::Texture2DArray: return sampleCount > 1 ? MTLTextureType2DMultisampleArray : MTLTextureType2DArray;
        case TextureDimension::TextureCube: return MTLTextureTypeCube;
        case TextureDimension::TextureCubeArray: return MTLTextureTypeCubeArray;
        case TextureDimension::Texture2DMS: return MTLTextureType2DMultisample;
        case TextureDimension::Texture2DMSArray: return MTLTextureType2DMultisampleArray;
        case TextureDimension::Texture3D: return MTLTextureType3D;
        default: return MTLTextureType2D;
        }
    }

    MTLResourceOptions convertCpuAccess(CpuAccessMode cpuAccess)
    {
        switch (cpuAccess)
        {
        case CpuAccessMode::None:
            return MTLResourceStorageModePrivate;
        case CpuAccessMode::Read:
        case CpuAccessMode::Write:
            return MTLResourceStorageModeShared;
        default:
            return MTLResourceStorageModeShared;
        }
    }

    MTLVertexFormat convertVertexFormat(Format format)
    {
        switch (format)
        {
        case Format::R32_FLOAT: return MTLVertexFormatFloat;
        case Format::RG32_FLOAT: return MTLVertexFormatFloat2;
        case Format::RGB32_FLOAT: return MTLVertexFormatFloat3;
        case Format::RGBA32_FLOAT: return MTLVertexFormatFloat4;
        case Format::R16_FLOAT: return MTLVertexFormatHalf;
        case Format::RG16_FLOAT: return MTLVertexFormatHalf2;
        case Format::RGBA16_FLOAT: return MTLVertexFormatHalf4;
        case Format::RGBA8_UNORM: return MTLVertexFormatUChar4Normalized;
        case Format::RGBA8_UINT: return MTLVertexFormatUChar4;
        case Format::R16_UINT: return MTLVertexFormatUShort;
        case Format::RG16_UINT: return MTLVertexFormatUShort2;
        case Format::R32_UINT: return MTLVertexFormatUInt;
        case Format::RG32_UINT: return MTLVertexFormatUInt2;
        case Format::RGB32_UINT: return MTLVertexFormatUInt3;
        case Format::RGBA32_UINT: return MTLVertexFormatUInt4;
        default: return MTLVertexFormatInvalid;
        }
    }

    MTLIndexType convertIndexFormat(Format format)
    {
        return format == Format::R32_UINT ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
    }

    MTLPrimitiveType convertPrimitiveType(PrimitiveType primitiveType)
    {
        switch (primitiveType)
        {
        case PrimitiveType::PointList: return MTLPrimitiveTypePoint;
        case PrimitiveType::LineList: return MTLPrimitiveTypeLine;
        case PrimitiveType::LineStrip: return MTLPrimitiveTypeLineStrip;
        case PrimitiveType::TriangleStrip: return MTLPrimitiveTypeTriangleStrip;
        default: return MTLPrimitiveTypeTriangle;
        }
    }

    MTLCullMode convertCullMode(RasterCullMode cullMode)
    {
        switch (cullMode)
        {
        case RasterCullMode::Back: return MTLCullModeBack;
        case RasterCullMode::Front: return MTLCullModeFront;
        case RasterCullMode::None:
        default: return MTLCullModeNone;
        }
    }

    MTLWinding convertWinding(bool frontCounterClockwise)
    {
        return frontCounterClockwise ? MTLWindingCounterClockwise : MTLWindingClockwise;
    }

    MTLCompareFunction convertCompareFunction(ComparisonFunc func)
    {
        switch (func)
        {
        case ComparisonFunc::Never: return MTLCompareFunctionNever;
        case ComparisonFunc::Less: return MTLCompareFunctionLess;
        case ComparisonFunc::Equal: return MTLCompareFunctionEqual;
        case ComparisonFunc::LessOrEqual: return MTLCompareFunctionLessEqual;
        case ComparisonFunc::Greater: return MTLCompareFunctionGreater;
        case ComparisonFunc::NotEqual: return MTLCompareFunctionNotEqual;
        case ComparisonFunc::GreaterOrEqual: return MTLCompareFunctionGreaterEqual;
        case ComparisonFunc::Always:
        default: return MTLCompareFunctionAlways;
        }
    }

    MTLSamplerAddressMode convertSamplerAddressMode(SamplerAddressMode mode)
    {
        switch (mode)
        {
        case SamplerAddressMode::Wrap: return MTLSamplerAddressModeRepeat;
        case SamplerAddressMode::Mirror: return MTLSamplerAddressModeMirrorRepeat;
        case SamplerAddressMode::Border: return MTLSamplerAddressModeClampToBorderColor;
        case SamplerAddressMode::MirrorOnce: return MTLSamplerAddressModeMirrorClampToEdge;
        case SamplerAddressMode::Clamp:
        default: return MTLSamplerAddressModeClampToEdge;
        }
    }

    MTLBlendFactor convertBlendFactor(BlendFactor factor)
    {
        switch (factor)
        {
        case BlendFactor::Zero: return MTLBlendFactorZero;
        case BlendFactor::One: return MTLBlendFactorOne;
        case BlendFactor::SrcColor: return MTLBlendFactorSourceColor;
        case BlendFactor::InvSrcColor: return MTLBlendFactorOneMinusSourceColor;
        case BlendFactor::SrcAlpha: return MTLBlendFactorSourceAlpha;
        case BlendFactor::InvSrcAlpha: return MTLBlendFactorOneMinusSourceAlpha;
        case BlendFactor::DstAlpha: return MTLBlendFactorDestinationAlpha;
        case BlendFactor::InvDstAlpha: return MTLBlendFactorOneMinusDestinationAlpha;
        case BlendFactor::DstColor: return MTLBlendFactorDestinationColor;
        case BlendFactor::InvDstColor: return MTLBlendFactorOneMinusDestinationColor;
        case BlendFactor::SrcAlphaSaturate: return MTLBlendFactorSourceAlphaSaturated;
        case BlendFactor::ConstantColor: return MTLBlendFactorBlendColor;
        case BlendFactor::InvConstantColor: return MTLBlendFactorOneMinusBlendColor;
        default: return MTLBlendFactorOne;
        }
    }

    MTLBlendOperation convertBlendOp(BlendOp op)
    {
        switch (op)
        {
        case BlendOp::Subtract: return MTLBlendOperationSubtract;
        case BlendOp::ReverseSubtract: return MTLBlendOperationReverseSubtract;
        case BlendOp::Min: return MTLBlendOperationMin;
        case BlendOp::Max: return MTLBlendOperationMax;
        case BlendOp::Add:
        default: return MTLBlendOperationAdd;
        }
    }
}
