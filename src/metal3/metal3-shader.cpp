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

        while (!result.empty() && std::isdigit(static_cast<unsigned char>(result.back())))
            result.pop_back();

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

    ShaderHandle Device::createShader(const ShaderDesc& d, const void* binary, size_t binarySize)
    {
        if (!binary || binarySize == 0)
            return nullptr;

        Shader* shader = new Shader();
        shader->desc = d;
        shader->bytecode.assign(static_cast<const uint8_t*>(binary), static_cast<const uint8_t*>(binary) + binarySize);

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

        NSString* entry = [NSString stringWithUTF8String:d.entryName.empty() ? "main" : d.entryName.c_str()];
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

        std::filesystem::path stageInPath = makeMscStageInLibraryPath(d.debugName);
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
        if (loadMscReflection(d.debugName, shader->mscReflection, &reflectionThreads, reflectionPath))
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
