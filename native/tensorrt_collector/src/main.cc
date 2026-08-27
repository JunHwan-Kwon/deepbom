#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#else
#include <dlfcn.h>
#include <openssl/evp.h>
#endif

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, char const* message) noexcept override {
    if (severity <= Severity::kWARNING && message != nullptr) messages.emplace_back(message);
  }
  std::vector<std::string> messages;
};

std::string json_escape(std::string const& value) {
  std::ostringstream out;
  for (unsigned char character : value) {
    switch (character) {
      case '\"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (character < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(character);
        else out << character;
    }
  }
  return out.str();
}

std::vector<std::uint8_t> read_bytes(std::filesystem::path const& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  auto const end = stream.tellg();
  if (end < 0) throw std::runtime_error("cannot size " + path.string());
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  stream.seekg(0);
  if (!bytes.empty() && !stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("cannot read " + path.string());
  }
  return bytes;
}

std::string read_text(std::filesystem::path const& path) {
  auto const bytes = read_bytes(path);
  return std::string(reinterpret_cast<char const*>(bytes.data()), bytes.size());
}

std::string sha256(std::vector<std::uint8_t> const& bytes) {
#ifdef _WIN32
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_length = 0;
  DWORD result_length = 0;
  std::vector<std::uint8_t> object;
  std::vector<std::uint8_t> digest(32);
  auto cleanup = [&]() {
    if (hash != nullptr) BCryptDestroyHash(hash);
    if (algorithm != nullptr) BCryptCloseAlgorithmProvider(algorithm, 0);
  };
  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0
      || BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_length),
          sizeof(object_length), &result_length, 0) < 0
      || result_length != sizeof(object_length)) {
    cleanup();
    throw std::runtime_error("SHA-256 initialization failed");
  }
  object.resize(object_length);
  if (BCryptCreateHash(algorithm, &hash, object.data(), static_cast<ULONG>(object.size()), nullptr, 0, 0) < 0) {
    cleanup();
    throw std::runtime_error("SHA-256 initialization failed");
  }
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    auto const remaining = bytes.size() - offset;
    auto const chunk = static_cast<ULONG>(std::min<std::size_t>(remaining, std::numeric_limits<ULONG>::max()));
    if (BCryptHashData(hash, const_cast<PUCHAR>(bytes.data() + offset), chunk, 0) < 0) {
      cleanup();
      throw std::runtime_error("SHA-256 update failed");
    }
    offset += chunk;
  }
  if (BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0) < 0) {
    cleanup();
    throw std::runtime_error("SHA-256 finalization failed");
  }
  cleanup();
  std::ostringstream out;
  for (auto const value : digest) out << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(value);
  return out.str();
#else
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1
      || (!bytes.empty() && EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1)) {
    throw std::runtime_error("SHA-256 initialization failed");
  }
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest, &length) != 1 || length != 32) throw std::runtime_error("SHA-256 finalization failed");
  std::ostringstream out;
  for (unsigned int index = 0; index < length; ++index) out << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(digest[index]);
  return out.str();
#endif
}

struct Arguments {
  std::unordered_map<std::string, std::string> values;
  std::vector<std::string> plugins;
};

struct ObservedSubgraph {
  bool supported;
  bool sdk_reported_flag;
  std::vector<std::int64_t> node_indices;
};

Arguments arguments(int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) throw std::runtime_error("arguments must be --name value pairs");
    auto const name = std::string(argv[index]).substr(2);
    if (name == "plugin") result.plugins.emplace_back(argv[index + 1]);
    else if (!result.values.emplace(name, argv[index + 1]).second) throw std::runtime_error("duplicate --" + name);
  }
  for (auto const* required : {"model", "profile", "profile-sha256", "device-id", "collector-binary-sha256", "collector-source-set-sha256", "collector-git-commit", "collector-git-state"}) {
    if (!result.values.count(required) || result.values.at(required).empty()) throw std::runtime_error(std::string("missing --") + required);
  }
  return result;
}

int nonnegative_int(std::string const& value, char const* label) {
  std::size_t consumed = 0;
  long long parsed = 0;
  try { parsed = std::stoll(value, &consumed, 10); }
  catch (...) { throw std::runtime_error(std::string(label) + " must be a non-negative integer"); }
  if (consumed != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string(label) + " must be a non-negative integer");
  }
  return static_cast<int>(parsed);
}

void load_plugin(std::filesystem::path const& path) {
#ifdef _WIN32
  if (LoadLibraryW(path.wstring().c_str()) == nullptr) throw std::runtime_error("cannot load TensorRT plugin " + path.string());
#else
  if (dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr) throw std::runtime_error("cannot load TensorRT plugin " + path.string());
#endif
}

std::string version_string(int value) {
  std::ostringstream out;
  out << value / 1000 << '.' << (value % 1000) / 10;
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto const args = arguments(argc, argv);
    auto const& values = args.values;
    std::filesystem::path const model_path = std::filesystem::absolute(values.at("model"));
    std::filesystem::path const profile_path = std::filesystem::absolute(values.at("profile"));
    auto const model = read_bytes(model_path);
    auto const profile_text = read_text(profile_path);
    auto const model_sha = sha256(model);
    auto const profile_file_sha = sha256(read_bytes(profile_path));
    int const selected_device = nonnegative_int(values.at("device-id"), "device-id");

    if (cudaSetDevice(selected_device) != cudaSuccess) throw std::runtime_error("CUDA selected device could not be activated");

    for (auto const& plugin : args.plugins) load_plugin(std::filesystem::absolute(plugin));

    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("TensorRT builder creation failed");
    auto const flags = 1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    if (!network) throw std::runtime_error("TensorRT network creation failed");
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) throw std::runtime_error("TensorRT ONNX parser creation failed");

    std::vector<ObservedSubgraph> observed_subgraphs;
#if NV_TENSORRT_MAJOR >= 10
    bool const supported = parser->supportsModelV2(model.data(), model.size(), model_path.string().c_str());
    char const* api_method = "supportsModelV2";
    char const* support_semantics = "per_subgraph_api_flag";
    for (std::int64_t index = 0; index < parser->getNbSubgraphs(); ++index) {
      std::int64_t length = 0;
      auto* nodes = parser->getSubgraphNodes(index, length);
      if (length <= 0 || nodes == nullptr) throw std::runtime_error("TensorRT parser returned an empty subgraph node list");
      auto const sdk_reported = parser->isSubgraphSupported(index);
      observed_subgraphs.push_back({sdk_reported, sdk_reported, {nodes, nodes + length}});
    }
#else
    SubGraphCollection_t legacy_subgraphs;
    bool const supported = parser->supportsModel(model.data(), model.size(), legacy_subgraphs, model_path.string().c_str());
    char const* api_method = "supportsModel";
    char const* support_semantics = "legacy_supported_collection_membership";
    for (auto const& subgraph : legacy_subgraphs) {
      if (subgraph.first.empty()) throw std::runtime_error("TensorRT parser returned an empty subgraph node list");
      ObservedSubgraph row{true, subgraph.second, {}};
      row.node_indices.reserve(subgraph.first.size());
      for (auto const node : subgraph.first) row.node_indices.push_back(static_cast<std::int64_t>(node));
      observed_subgraphs.push_back(std::move(row));
    }
#endif
    int runtime_version = 0;
    int driver_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess || cudaDriverGetVersion(&driver_version) != cudaSuccess) {
      throw std::runtime_error("CUDA version query failed");
    }
    cudaDeviceProp properties{};
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
      throw std::runtime_error("CUDA device identity query failed");
    }

    std::ostringstream out;
    out << '{'
        << "\"schema\":\"deepbom.tensorrt_parser_observation.v1\","
        << "\"artifact_sha256\":\"" << model_sha << "\","
        << "\"build_profile_sha256\":\"" << json_escape(values.at("profile-sha256")) << "\","
        << "\"build_profile_file_sha256\":\"" << profile_file_sha << "\","
        << "\"build_profile\":" << profile_text << ','
        << "\"execution_path\":\"native_tensorrt\","
        << "\"tensorrt_version\":\"" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH << "\","
        << "\"onnx_parser_version\":" << getNvOnnxParserVersion() << ','
        << "\"cuda_version\":\"runtime " << version_string(runtime_version) << "; driver " << version_string(driver_version) << "\","
        << "\"device_id\":" << device << ','
        << "\"device_compute_capability\":\"" << properties.major << '.' << properties.minor << "\","
        << "\"device_identity\":\"" << json_escape(properties.name) << " / CC " << properties.major << '.' << properties.minor << "\","
        << "\"api_method\":\"" << api_method << "\","
        << "\"subgraph_support_semantics\":\"" << support_semantics << "\","
        << "\"parser_returned\":" << (supported ? "true" : "false") << ','
        << "\"collector\":{"
        << "\"binary_sha256\":\"" << json_escape(values.at("collector-binary-sha256")) << "\","
        << "\"source_set_sha256\":\"" << json_escape(values.at("collector-source-set-sha256")) << "\","
        << "\"git_commit\":\"" << json_escape(values.at("collector-git-commit")) << "\","
        << "\"git_state\":\"" << json_escape(values.at("collector-git-state")) << "\"},"
        << "\"subgraphs\":[";
    for (std::size_t index = 0; index < observed_subgraphs.size(); ++index) {
      if (index) out << ',';
      auto const& subgraph = observed_subgraphs[index];
      out << "{\"subgraph_index\":" << index << ",\"supported\":" << (subgraph.supported ? "true" : "false")
          << ",\"sdk_reported_flag\":" << (subgraph.sdk_reported_flag ? "true" : "false") << ",\"node_indices\":[";
      for (std::size_t node = 0; node < subgraph.node_indices.size(); ++node) {
        if (node) out << ',';
        out << subgraph.node_indices[node];
      }
      out << "]}";
    }
    out << "],\"errors\":[";
    for (std::int32_t index = 0; index < parser->getNbErrors(); ++index) {
      if (index) out << ',';
      auto const* error = parser->getError(index);
      if (error == nullptr) throw std::runtime_error("TensorRT parser returned a null error record");
      out << "{\"code\":" << static_cast<int>(error->code())
          << ",\"message\":\"" << json_escape(error->desc() == nullptr ? "unknown parser error" : error->desc()) << "\"}";
    }
    out << "],\"collector_log\":[";
    for (std::size_t index = 0; index < logger.messages.size(); ++index) {
      if (index) out << ',';
      out << '\"' << json_escape(logger.messages[index]) << '\"';
    }
    out << "]}\n";
    std::cout << out.str();
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "deepbom-tensorrt-parser-collector: " << error.what() << '\n';
    return 1;
  }
}
