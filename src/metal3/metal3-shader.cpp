#include "metal3-backend.h"
#include <dispatch/dispatch.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <limits>
#include <memory>
#include <metal_irconverter/metal_irconverter.h>
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>
#pragma push_macro("BOOL")
#undef BOOL
#define BOOL MetalDxcBOOL
#include <dxcapi.h>
#pragma push_macro("interface")
#undef interface
#define interface struct
#include <d3d12shader.h>
#pragma pop_macro("interface")
#pragma pop_macro("BOOL")

namespace nvrhi::metal3
{
    static std::string trimCopy(std::string value)
    {
        auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) { return std::isspace(ch); });
        auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) { return std::isspace(ch); }).base();
        if (first >= last)
            return {};
        return std::string(first, last);
    }

    static bool readTextFile(const std::string& path, std::string& out)
    {
        std::ifstream file(path);
        if (!file)
            return false;

        std::ostringstream ss;
        ss << file.rdbuf();
        out = ss.str();
        return true;
    }

    static bool parseMscArgumentType(const std::string& value, MscArgumentType& out)
    {
        if (value == "SRV")
        {
            out = MscArgumentType::SRV;
            return true;
        }
        if (value == "UAV")
        {
            out = MscArgumentType::UAV;
            return true;
        }
        if (value == "CBV")
        {
            out = MscArgumentType::CBV;
            return true;
        }
        if (value == "Sampler")
        {
            out = MscArgumentType::Sampler;
            return true;
        }

        return false;
    }

    static std::string normalizeMscVertexInputName(const std::string& value)
    {
        std::string result;
        result.reserve(value.size());
        for (unsigned char ch : value)
        {
            if (std::isalnum(ch))
                result.push_back(static_cast<char>(std::tolower(ch)));
        }

        if (!result.empty() && !std::isdigit(static_cast<unsigned char>(result.back())))
            result.push_back('0');

        return result;
    }

    static bool parseMscReflectionJson(const std::string& json, MscShaderReflection& reflection, MTLSize* computeThreads)
    {
        std::smatch match;
        bool hasResourceCount = false;

        // top-level MSC metadata describes the original HLSL stage and whether
        // the generated Metal entry point needs function constants at creation.
        std::regex shaderTypeRegex(R"name("ShaderType"\s*:\s*"([^"]+)")name");
        if (std::regex_search(json, match, shaderTypeRegex))
            reflection.shaderType = match[1].str();

        std::regex needsFunctionConstantsRegex(R"("NeedsFunctionConstants"\s*:\s*(true|false))");
        if (std::regex_search(json, match, needsFunctionConstantsRegex))
            reflection.needsFunctionConstants = match[1].str() == "true";

        std::regex resourceCountRegex(R"("ResourceCount"\s*:\s*([0-9]+))");
        if (std::regex_search(json, match, resourceCountRegex))
        {
            reflection.resourceCount = static_cast<uint32_t>(std::stoul(match[1].str()));
            hasResourceCount = true;
        }

        // MSC emits one compact object per top-level argument-buffer entry. The
        // entry order is the physical descriptor-table index used by the runtime.
        std::regex entryRegex(
            R"msc(\{"EltOffset"\s*:\s*([0-9]+)\s*,\s*"Name"\s*:\s*"[^"]*"\s*,\s*"Size"\s*:\s*([0-9]+)\s*,\s*"Slot"\s*:\s*([0-9]+)\s*,\s*"Space"\s*:\s*([0-9]+)\s*,\s*"Type"\s*:\s*"([^"]+)"\})msc");
        for (std::sregex_iterator it(json.begin(), json.end(), entryRegex), end; it != end; ++it)
        {
            /*
             * *it[5] corresponds to 5th argument for a resource in top level argument buffer
             * Ex: a single resource can have this reflection: {"EltOffset":120,"Name":"","Size":24,"Slot":1,"Space":0,"Type":"CBV"}
             * regex capture groups:
             [0] whole matched object
             * [1] EltOffset  -> "120"
             * [2] Size       -> "24"
             * [3] Slot       -> "1"
             * [4] Space      -> "0"
             * [5] Type       -> "CBV"
             * the 5th arg here is the "Type". so default is to SRV, and then check for actual type, and change to the type post that
            */
            MscArgumentType type = MscArgumentType::SRV;
            if (!parseMscArgumentType((*it)[5].str(), type))
                continue;

            const uint32_t elementOffset = static_cast<uint32_t>(std::stoul((*it)[1].str()));
            const uint32_t elementSize = static_cast<uint32_t>(std::stoul((*it)[2].str()));

            MscArgumentBinding binding;
            binding.index = elementSize ? elementOffset / elementSize : static_cast<uint32_t>(reflection.topLevelArgumentBuffer.size());
            binding.slot = static_cast<uint32_t>(std::stoul((*it)[3].str()));
            binding.space = static_cast<uint32_t>(std::stoul((*it)[4].str()));
            binding.type = type;
            reflection.topLevelArgumentBuffer.push_back(binding);
        }

        std::regex vertexInputsRegex(R"("vertex_inputs"\s*:\s*\[([^\]]*)\])");
        if (std::regex_search(json, match, vertexInputsRegex))
        {
            const std::string vertexInputs = match[1].str();
            std::regex objectRegex(R"(\{[^\}]*\})");
            std::regex indexRegex(R"("index"\s*:\s*([0-9]+))");
            std::regex nameRegex(R"name("name"\s*:\s*"([^"]+)")name");
            for (std::sregex_iterator it(vertexInputs.begin(), vertexInputs.end(), objectRegex), end; it != end; ++it)
            {
                const std::string object = it->str();
                std::smatch indexMatch;
                std::smatch nameMatch;
                if (!std::regex_search(object, indexMatch, indexRegex) ||
                    !std::regex_search(object, nameMatch, nameRegex))
                    continue;

                const std::string name = normalizeMscVertexInputName(nameMatch[1].str());
                if (!name.empty())
                    reflection.vertexInputAttributes[name] =
                        static_cast<uint32_t>(std::stoul(indexMatch[1].str()));
            }
        }

        // size of one VS output record consumed by the emulated GS mesh stage.
        std::regex vertexOutputSizeRegex(R"("vertex_output_size_in_bytes"\s*:\s*([0-9]+))");
        if (std::regex_search(json, match, vertexOutputSizeRegex))
            reflection.vertexOutputSizeInBytes = static_cast<uint32_t>(std::stoul(match[1].str()));

        // max number of source primitives the generated mesh threadgroup processes.
        std::regex maxInputPrimsRegex(R"("max_input_primitives_per_mesh_threadgroup"\s*:\s*([0-9]+))");
        if (std::regex_search(json, match, maxInputPrimsRegex))
            reflection.maxInputPrimitivesPerMeshThreadgroup = static_cast<uint32_t>(std::stoul(match[1].str()));

        // GS instance count used by the IRConverter emulation config.
        std::regex instanceCountRegex(R"("instance_count"\s*:\s*([0-9]+))");
        if (std::regex_search(json, match, instanceCountRegex))
            reflection.geometryInstanceCount = static_cast<uint32_t>(std::stoul(match[1].str()));

        // source primitive topology expected by the converted geometry shader.
        std::regex inputPrimitiveRegex(R"name("input_primitive"\s*:\s*"([^"]+)")name");
        if (std::regex_search(json, match, inputPrimitiveRegex))
            reflection.inputPrimitive = match[1].str();
        
        // parse compute threads from reflection data, *tg_size*
        if (computeThreads)
        {
            std::regex tgSizeRegex(R"("tg_size"\s*:\s*\[\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\])");
            if (std::regex_search(json, match, tgSizeRegex))
            {
                computeThreads->width = static_cast<NSUInteger>(std::stoul(match[1].str()));
                computeThreads->height = static_cast<NSUInteger>(std::stoul(match[2].str()));
                computeThreads->depth = static_cast<NSUInteger>(std::stoul(match[3].str()));
            }
        }

        if (reflection.resourceCount == 0)
            reflection.resourceCount = static_cast<uint32_t>(reflection.topLevelArgumentBuffer.size());

        // Shaders that consume only vertex/pixel inputs legitimately have an
        // empty top-level argument buffer. Treat explicit ResourceCount:0 as
        // valid reflection instead of falling back to legacy ordering.
        reflection.valid = !reflection.topLevelArgumentBuffer.empty() ||
            (hasResourceCount && reflection.resourceCount == 0);
        return reflection.valid;
    }

    static std::vector<std::filesystem::path> makeMscReflectionCandidates(const std::string& debugName)
    {
        std::vector<std::filesystem::path> candidates;
        if (debugName.empty())
            return candidates;

        std::filesystem::path path(debugName);
        if (path.has_extension())
        {
            std::filesystem::path reflectionPath = path;
            reflectionPath.replace_extension(".reflection.json");
            candidates.push_back(reflectionPath);

            reflectionPath = path;
            reflectionPath.replace_extension(".json");
            candidates.push_back(reflectionPath);
        }

        candidates.emplace_back(debugName + ".reflection.json");
        candidates.emplace_back(debugName + ".json");
        return candidates;
    }

    static bool loadMscReflection(const std::string& debugName, MscShaderReflection& reflection, MTLSize* computeThreads, std::string& loadedPath)
    {
        for (const std::filesystem::path& path : makeMscReflectionCandidates(debugName))
        {
            std::string json;
            if (!readTextFile(path.string(), json))
                continue;

            MscShaderReflection parsed;
            if (!parseMscReflectionJson(json, parsed, computeThreads))
                continue;

            reflection = std::move(parsed);
            loadedPath = path.string();
            return true;
        }

        return false;
    }

    static std::filesystem::path makeMscStageInLibraryPath(const std::string& debugName)
    {
        if (debugName.empty())
            return {};

        std::filesystem::path path(debugName);
        if (path.extension() == ".metallib")
        {
            std::string value = path.string();
            constexpr const char* suffix = ".metallib";
            value.resize(value.size() - std::strlen(suffix));
            value += ".stageIn.metallib";
            return value;
        }

        return {};
    }

    static std::string converterErrorDescription(const IRError* error)
    {
        if (!error)
            return "no converter diagnostic";
        const uint32_t code = IRErrorGetCode(error);
        std::string message = "MSC error " + std::to_string(code);
        switch (code)
        {
        case IRErrorCodeShaderRequiresRootSignature:
            return message + ": shader requires an explicit root signature with directly indexed heap flags";
        case IRErrorCodeUnrecognizedRootSignatureDescriptor:
            return message + ": converter rejected the root signature descriptor version or flags";
        case IRErrorCodeUnrecognizedParameterTypeInRootSignature:
            return message + ": converter rejected a root parameter type";
        case IRErrorCodeResourceNotReferencedByRootSignature:
            return message + ": a shader register, space, or descriptor array extent is absent from the root signature";
        case IRErrorCodeUnsupportedInstruction:
            return message + ": DXIL contains an instruction unsupported by this converter";
        case IRErrorCodeCompilationError:
            return message + ": converter compilation failed";
        case IRErrorCodeUnableToVerifyModule:
            return message + ": converter could not verify the DXIL module";
        case IRErrorCodeUnableToLinkModule:
            return message + ": converter could not link the module; check SDK and deployment target compatibility";
        case IRErrorCodeUnrecognizedDXILHeader:
            return message + ": input is not a supported DXIL container";
        default:
            return message;
        }
    }

    template<typename T>
    struct DxcRelease
    {
        void operator()(T* object) const
        {
            if (object)
                object->Release();
        }
    };

    static IRRootSignature* createDxilRootSignature(const MTL3Context& context, Shader& shader,
        const void* binary, size_t binarySize)
    {
        IDxcUtils* rawUtils = nullptr;
        HRESULT result = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&rawUtils));
        std::unique_ptr<IDxcUtils, DxcRelease<IDxcUtils>> utils(rawUtils);
        if (FAILED(result) || !utils)
        {
            context.error("[metal3] Cannot initialize DXC resource reflection for '" + shader.desc.debugName +
                "', HRESULT " + std::to_string(uint32_t(result)));
            return nullptr;
        }
        const DxcBuffer buffer{binary, binarySize, 0};
        ID3D12ShaderReflection* rawReflection = nullptr;
        result = utils->CreateReflection(&buffer, IID_PPV_ARGS(&rawReflection));
        std::unique_ptr<ID3D12ShaderReflection, DxcRelease<ID3D12ShaderReflection>> reflection(rawReflection);
        D3D12_SHADER_DESC desc{};
        if (FAILED(result) || !reflection || FAILED(reflection->GetDesc(&desc)))
        {
            context.error("[metal3] Cannot reflect DXIL resources for '" + shader.desc.debugName +
                "', HRESULT " + std::to_string(uint32_t(result)) + "; preserve DXIL reflection data");
            return nullptr;
        }

        std::vector<std::vector<IRDescriptorRange1>> ranges(2);
        std::vector<MscDescriptorTable> tables(2);
        tables[1].type = MscArgumentType::Sampler;
        auto& metadata = shader.mscReflection;
        for (uint32_t i = 0; i < desc.BoundResources; ++i)
        {
            D3D12_SHADER_INPUT_BIND_DESC resource{};
            if (FAILED(reflection->GetResourceBindingDesc(i, &resource)))
            {
                context.error("[metal3] Cannot reflect DXIL resource " + std::to_string(i) + " for '" + shader.desc.debugName + "'");
                return nullptr;
            }
            MscArgumentBinding binding;
            IRDescriptorRange1 range{};
            switch (resource.Type)
            {
            case D3D_SIT_CBUFFER:
                binding.type = MscArgumentType::CBV;
                range.RangeType = IRDescriptorRangeTypeCBV;
                break;
            case D3D_SIT_TEXTURE:
            case D3D_SIT_TBUFFER:
            case D3D_SIT_STRUCTURED:
            case D3D_SIT_BYTEADDRESS:
                binding.type = MscArgumentType::SRV;
                range.RangeType = IRDescriptorRangeTypeSRV;
                break;
            case D3D_SIT_UAV_RWTYPED:
            case D3D_SIT_UAV_RWSTRUCTURED:
            case D3D_SIT_UAV_RWBYTEADDRESS:
                binding.type = MscArgumentType::UAV;
                range.RangeType = IRDescriptorRangeTypeUAV;
                break;
            case D3D_SIT_SAMPLER:
                binding.type = MscArgumentType::Sampler;
                range.RangeType = IRDescriptorRangeTypeSampler;
                break;
            default:
                context.error("[metal3] Unsupported DXIL resource type " + std::to_string(resource.Type) +
                    " for '" + shader.desc.debugName + "', resource '" + (resource.Name ? resource.Name : "") +
                    "', slot " + std::to_string(resource.BindPoint) + ", space " + std::to_string(resource.Space) +
                    "; UAV counters and ray tracing root resources are not supported");
                return nullptr;
            }
            const bool unbounded = resource.BindCount == 0 || resource.BindCount == UINT32_MAX;
            uint32_t tableIndex = binding.type == MscArgumentType::Sampler ? 1 : 0;
            if (unbounded)
            {
                tableIndex = uint32_t(tables.size());
                tables.emplace_back();
                tables.back().slot = resource.BindPoint;
                tables.back().space = resource.Space;
                tables.back().type = binding.type;
                ranges.emplace_back();
            }
            auto& table = tables[tableIndex];
            range.NumDescriptors = unbounded ? UINT32_MAX : resource.BindCount;
            range.BaseShaderRegister = resource.BindPoint;
            range.RegisterSpace = resource.Space;
            range.Flags = static_cast<IRDescriptorRangeFlags>(IRDescriptorRangeFlagDescriptorsVolatile |
                (binding.type == MscArgumentType::Sampler ? 0 : IRDescriptorRangeFlagDataVolatile));
            range.OffsetInDescriptorsFromTableStart = table.descriptorCount;
            ranges[tableIndex].push_back(range);
            if (!unbounded)
            {
                if (resource.BindCount > UINT32_MAX / sizeof(IRDescriptorTableEntry) - table.descriptorCount ||
                    resource.BindCount - 1 > UINT32_MAX - resource.BindPoint)
                {
                    context.error("[metal3] DXIL descriptor range exceeds the Metal argument buffer limit: " + shader.desc.debugName);
                    return nullptr;
                }
                binding.tableIndex = tableIndex;
                binding.index = table.descriptorCount;
                binding.byteOffset = table.descriptorCount * sizeof(IRDescriptorTableEntry);
                binding.sizeBytes = uint64_t(resource.BindCount) * sizeof(IRDescriptorTableEntry);
                binding.slot = resource.BindPoint;
                binding.space = resource.Space;
                metadata.topLevelArgumentBuffer.push_back(binding);
                table.descriptorCount += resource.BindCount;
            }
        }
        std::vector<IRRootParameter1> parameters;
        std::vector<uint32_t> tableIndices;
        for (uint32_t i = 0; i < tables.size(); ++i)
        {
            if (ranges[i].empty())
                continue;
            IRRootParameter1 parameter{};
            parameter.ParameterType = IRRootParameterTypeDescriptorTable;
            parameter.ShaderVisibility = IRShaderVisibilityAll;
            parameter.DescriptorTable.NumDescriptorRanges = uint32_t(ranges[i].size());
            parameter.DescriptorTable.pDescriptorRanges = ranges[i].data();
            parameters.push_back(parameter);
            tableIndices.push_back(i);
        }
        IRVersionedRootSignatureDescriptor rootDesc{};
        rootDesc.version = IRRootSignatureVersion_1_1;
        rootDesc.desc_1_1.NumParameters = uint32_t(parameters.size());
        rootDesc.desc_1_1.pParameters = parameters.data();
        uint32_t flags = IRRootSignatureFlagAllowInputAssemblerInputLayout;
        const uint64_t requiresFlags = reflection->GetRequiresFlags();
        metadata.directlyIndexedResourceHeap = (requiresFlags & D3D_SHADER_REQUIRES_RESOURCE_DESCRIPTOR_HEAP_INDEXING) != 0;
        metadata.directlyIndexedSamplerHeap = (requiresFlags & D3D_SHADER_REQUIRES_SAMPLER_DESCRIPTOR_HEAP_INDEXING) != 0;
        if (requiresFlags & D3D_SHADER_REQUIRES_RESOURCE_DESCRIPTOR_HEAP_INDEXING)
            flags |= IRRootSignatureFlagCBVSRVUAVHeapDirectlyIndexed;
        if (requiresFlags & D3D_SHADER_REQUIRES_SAMPLER_DESCRIPTOR_HEAP_INDEXING)
            flags |= IRRootSignatureFlagSamplerHeapDirectlyIndexed;
        rootDesc.desc_1_1.Flags = static_cast<IRRootSignatureFlags>(flags);
        IRError* error = nullptr;
        std::unique_ptr<IRRootSignature, decltype(&IRRootSignatureDestroy)> root(
            IRRootSignatureCreateFromDescriptor(&rootDesc, &error), IRRootSignatureDestroy);
        if (!root)
            context.error("[metal3] Cannot create root signature for '" + shader.desc.debugName + "': " + converterErrorDescription(error));
        if (error)
            IRErrorDestroy(error);
        if (!root)
            return nullptr;
        std::vector<IRResourceLocation> locations(IRRootSignatureGetResourceCount(root.get()));
        IRRootSignatureGetResourceLocations(root.get(), locations.data());
        if (locations.size() != parameters.size())
        {
            context.error("[metal3] MSC root reflection does not match the descriptor table count: " + shader.desc.debugName);
            return nullptr;
        }
        for (size_t i = 0; i < locations.size(); ++i)
        {
            const auto& location = locations[i];
            if (location.resourceType != IRResourceTypeTable || location.sizeBytes != sizeof(uint64_t) ||
                location.topLevelOffset > UINT32_MAX - sizeof(uint64_t))
            {
                context.error("[metal3] Unsupported MSC descriptor table pointer layout: " + shader.desc.debugName);
                return nullptr;
            }
            tables[tableIndices[i]].byteOffset = location.topLevelOffset;
            metadata.argumentBufferSize = std::max(metadata.argumentBufferSize, location.topLevelOffset + location.sizeBytes);
        }
        for (auto& binding : metadata.topLevelArgumentBuffer)
            binding.tableIndex = uint32_t(std::find(tableIndices.begin(), tableIndices.end(), binding.tableIndex) - tableIndices.begin());
        for (uint32_t index : tableIndices)
            metadata.descriptorTables.push_back(tables[index]);
        metadata.resourceCount = desc.BoundResources;
        return root.release();
    }

    static bool convertDxilShader(const MTL3Context& context, Shader& shader, const void* binary, size_t binarySize,
        std::vector<uint8_t>& metallib, std::string& entryName)
    {
        IRShaderStage stage = IRShaderStageInvalid;
        switch (shader.desc.shaderType)
        {
        case ShaderType::Vertex: stage = IRShaderStageVertex; break;
        case ShaderType::Pixel: stage = IRShaderStageFragment; break;
        case ShaderType::Compute: stage = IRShaderStageCompute; break;
        default:
            context.error("[metal3] DXIL conversion supports vertex, pixel and compute shaders: " + shader.desc.debugName);
            return false;
        }
        std::unique_ptr<IRRootSignature, decltype(&IRRootSignatureDestroy)> root(
            createDxilRootSignature(context, shader, binary, binarySize), IRRootSignatureDestroy);
        if (!root)
            return false;

        std::unique_ptr<IRCompiler, decltype(&IRCompilerDestroy)> compiler(IRCompilerCreate(), IRCompilerDestroy);
        std::unique_ptr<IRObject, decltype(&IRObjectDestroy)> input(
            IRObjectCreateFromDXIL(static_cast<const uint8_t*>(binary), binarySize, IRBytecodeOwnershipNone), IRObjectDestroy);
        if (!compiler || !input)
        {
            context.error("[metal3] Unable to initialize DXIL conversion: " + shader.desc.debugName);
            return false;
        }

        IRCompilerSetMinimumGPUFamily(compiler.get(), IRGPUFamilyMetal3);
        IRCompilerSetMinimumDeploymentTarget(compiler.get(), IROperatingSystem_macOS, "14.0.0");
        IRCompilerSetStageInGenerationMode(compiler.get(), IRStageInCodeGenerationModeUseMetalVertexFetch);
        IRCompilerSetFunctionConstantResourceSpace(compiler.get(), UINT32_MAX);
        IRCompilerSetFramebufferFetchResourceSpace(compiler.get(), UINT32_MAX);
        IRCompilerIgnoreRootSignature(compiler.get(), true);
        IRCompilerSetGlobalRootSignature(compiler.get(), root.get());
        IRCompilerSetEntryPointName(compiler.get(), entryName.c_str());
        IRError* error = nullptr;
        std::unique_ptr<IRObject, decltype(&IRObjectDestroy)> output(
            IRCompilerAllocCompileAndLink(compiler.get(), entryName.c_str(), input.get(), &error), IRObjectDestroy);
        if (!output)
        {
            context.error("[metal3] DXIL conversion failed for '" + shader.desc.debugName + "': " + converterErrorDescription(error));
            if (error)
                IRErrorDestroy(error);
            return false;
        }
        if (error)
            IRErrorDestroy(error);

        std::unique_ptr<IRShaderReflection, decltype(&IRShaderReflectionDestroy)> reflection(
            IRShaderReflectionCreate(), IRShaderReflectionDestroy);
        std::unique_ptr<IRMetalLibBinary, decltype(&IRMetalLibBinaryDestroy)> library(
            IRMetalLibBinaryCreate(), IRMetalLibBinaryDestroy);
        if (!reflection || !library || !IRObjectGetReflection(output.get(), stage, reflection.get()) ||
            !IRObjectGetMetalLibBinary(output.get(), stage, library.get()))
        {
            context.error("[metal3] Missing MSC library or stage reflection: " + shader.desc.debugName);
            return false;
        }

        auto& metadata = shader.mscReflection;
        metadata.needsFunctionConstants = IRShaderReflectionNeedsFunctionConstants(reflection.get());
        if (metadata.needsFunctionConstants)
        {
            context.error("[metal3] Shader requires unsupported MSC function constants: " + shader.desc.debugName);
            return false;
        }
        const char* reflectedEntry = IRShaderReflectionGetEntryPointFunctionName(reflection.get());
        if (!reflectedEntry || !*reflectedEntry)
        {
            context.error("[metal3] Missing MSC entry point: " + shader.desc.debugName);
            return false;
        }
        entryName = reflectedEntry;

        if (stage == IRShaderStageVertex)
        {
            IRVersionedVSInfo info{};
            if (!IRShaderReflectionCopyVertexInfo(reflection.get(), IRReflectionVersion_1_0, &info))
            {
                context.error("[metal3] Missing MSC vertex inputs: " + shader.desc.debugName);
                return false;
            }
            metadata.vertexOutputSizeInBytes = info.info_1_0.vertex_output_size_in_bytes;
            for (size_t i = 0; i < info.info_1_0.num_vertex_inputs; ++i)
            {
                const auto& attribute = info.info_1_0.vertex_inputs[i];
                if (attribute.name)
                    metadata.vertexInputAttributes[normalizeMscVertexInputName(attribute.name)] = attribute.attributeIndex;
            }
            IRShaderReflectionReleaseVertexInfo(&info);
        }
        if (stage == IRShaderStageCompute)
        {
            IRVersionedCSInfo info{};
            if (!IRShaderReflectionCopyComputeInfo(reflection.get(), IRReflectionVersion_1_0, &info))
            {
                context.error("[metal3] Missing MSC compute dimensions: " + shader.desc.debugName);
                return false;
            }
            shader.computeThreadsPerGroup = MTLSizeMake(info.info_1_0.tg_size[0], info.info_1_0.tg_size[1], info.info_1_0.tg_size[2]);
            IRShaderReflectionReleaseComputeInfo(&info);
            shader.computeThreadsPerGroupValid = shader.computeThreadsPerGroup.width != 0 &&
                shader.computeThreadsPerGroup.height != 0 && shader.computeThreadsPerGroup.depth != 0;
            if (!shader.computeThreadsPerGroupValid)
            {
                context.error("[metal3] Invalid MSC compute dimensions: " + shader.desc.debugName);
                return false;
            }
        }
        metallib.resize(IRMetalLibGetBytecodeSize(library.get()));
        if (metallib.empty() || IRMetalLibGetBytecode(library.get(), metallib.data()) != metallib.size())
        {
            context.error("[metal3] Failed to extract MSC bytecode: " + shader.desc.debugName);
            return false;
        }
        metadata.valid = true;
        return true;
    }

    ShaderHandle Device::createShader(const ShaderDesc& d, const void* binary, size_t binarySize)
    {
        if (!binary || binarySize == 0)
            return nullptr;

        Shader* shader = new Shader();
        shader->desc = d;
        shader->bytecode.assign(static_cast<const uint8_t*>(binary), static_cast<const uint8_t*>(binary) + binarySize);
        const bool isDxil = binarySize >= 4 && std::memcmp(binary, "DXBC", 4) == 0;
        std::string entryName = d.entryName.empty() ? "main" : d.entryName;
        std::vector<uint8_t> metallib;
        if (isDxil)
        {
            if (!convertDxilShader(m_Context, *shader, binary, binarySize, metallib, entryName))
            {
                delete shader;
                return nullptr;
            }
            binary = metallib.data();
            binarySize = metallib.size();
        }

        dispatch_data_t data = dispatch_data_create(binary, binarySize, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError* error = nil;
        shader->library = [m_Context.device newLibraryWithData:data error:&error];
        if (!shader->library)
        {
            std::string message = "[nvrhi] Failed to load Metal library";
            if (error)
                message += std::string(": ") + [[error localizedDescription] UTF8String];
            m_Context.error(message);
            delete shader;
            return nullptr;
        }

        NSString* entry = [NSString stringWithUTF8String:entryName.c_str()];
        shader->function = [shader->library newFunctionWithName:entry];
        if (!shader->function)
        {
            NSString* objectEntry = [NSString stringWithFormat:@"%@.dxil_irconverter_object_shader", entry];
            shader->function = [shader->library newFunctionWithName:objectEntry];
        }
        if (!shader->function)
        {
            m_Context.error("[nvrhi] Failed to find Metal shader entry: " + d.entryName);
            delete shader;
            return nullptr;
        }

        std::filesystem::path stageInPath = isDxil ? std::filesystem::path() : makeMscStageInLibraryPath(d.debugName);
        if (!stageInPath.empty() && std::filesystem::exists(stageInPath))
        {
            NSError* stageInError = nil;
            shader->stageInLibrary = [m_Context.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:stageInPath.string().c_str()]]
                                                                    error:&stageInError];
            if (!shader->stageInLibrary)
            {
                std::string message = "[metal3] failed to load MSC stage-in library '" + stageInPath.string() + "'";
                if (stageInError)
                    message += std::string(": ") + [[stageInError localizedDescription] UTF8String];
                m_Context.warning(message);
            }
        }

        if (!d.debugName.empty())
            shader->function.label = [NSString stringWithUTF8String:d.debugName.c_str()];

        MTLSize reflectionThreads = MTLSizeMake(0, 0, 0);
        std::string reflectionPath;
        if (isDxil)
        {
            reflectionThreads = shader->computeThreadsPerGroupValid ? shader->computeThreadsPerGroup : reflectionThreads;
        }
        else if (loadMscReflection(d.debugName, shader->mscReflection, &reflectionThreads, reflectionPath))
        {
            // uncomment for debugging
            // m_Context.info("[metal3] shader '" + d.debugName + "' reflection='" + reflectionPath +
            //     "' resources=" + std::to_string(shader->mscReflection.resourceCount) +
            //     " args=" + std::to_string(shader->mscReflection.topLevelArgumentBuffer.size()));
        }
        else
        {
            m_Context.warning("[metal3] shader '" + d.debugName + "' has no MSC reflection; using legacy argument-buffer ordering");
        }
        shader->reflectedBindingPlan = createMetalStageBindingPlan(d.shaderType, shader->mscReflection);

        if (d.shaderType == ShaderType::Compute)
        {
            if (reflectionThreads.width != 0 && reflectionThreads.height != 0 && reflectionThreads.depth != 0)
            {
                shader->computeThreadsPerGroup = reflectionThreads;
                shader->computeThreadsPerGroupValid = true;
            }

            if (shader->computeThreadsPerGroupValid)
            {
                // uncomment for debugging
                // m_Context.info("[metal3] compute shader '" + d.debugName + "' numthreads=" +
                //     std::to_string(shader->computeThreadsPerGroup.width) + "x" +
                //     std::to_string(shader->computeThreadsPerGroup.height) + "x" +
                //     std::to_string(shader->computeThreadsPerGroup.depth));
            }
        }

        return ShaderHandle::Create(shader);
    }

    // not really used anywhere, currently implemented cuz override
    // TODO: useful for creating a single shader bundle metallib file(s), path can (?) be added later
    ShaderLibraryHandle Device::createShaderLibrary(const void* binary, size_t binarySize)
    {
        if (!binary || binarySize == 0)
            return nullptr;

        ShaderLibrary* library = new ShaderLibrary();
        library->bytecode.assign(static_cast<const uint8_t*>(binary), static_cast<const uint8_t*>(binary) + binarySize);

        dispatch_data_t data = dispatch_data_create(binary, binarySize, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError* error = nil;
        library->library = [m_Context.device newLibraryWithData:data error:&error];
        if (!library->library)
        {
            delete library;
            return nullptr;
        }

        return ShaderLibraryHandle::Create(library);
    }

    void Shader::getBytecode(const void** ppBytecode, size_t* pSize) const
    {
        if (ppBytecode) *ppBytecode = bytecode.data();
        if (pSize) *pSize = bytecode.size();
    }

    void ShaderLibrary::getBytecode(const void** ppBytecode, size_t* pSize) const
    {
        if (ppBytecode) *ppBytecode = bytecode.data();
        if (pSize) *pSize = bytecode.size();
    }

    ShaderHandle ShaderLibrary::getShader(const char* entryName, ShaderType shaderType)
    {
        Shader* shader = new Shader();
        shader->desc.entryName = entryName ? entryName : "main";
        shader->desc.shaderType = shaderType;
        shader->bytecode = bytecode;
        shader->library = library;
        shader->function = [library newFunctionWithName:[NSString stringWithUTF8String:shader->desc.entryName.c_str()]];
        if (!shader->function)
        {
            delete shader;
            return nullptr;
        }
        return ShaderHandle::Create(shader);
    }
}
