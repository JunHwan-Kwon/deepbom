#include "src/xnnpack/deepbom.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <dbghelp.h>
#if defined(_MSC_VER)
#pragma comment(lib, "dbghelp.lib")
#endif
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include "src/deepbom_runtime_trace.h"
#include "src/deepbom-symbol-catalog.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/subgraph.h"

namespace {
constexpr size_t kNameCapacity = 64;

struct Binding {
  uint32_t op_index;
  std::string op_name;
  std::string lowering_id;
  size_t runtime_node_id;
  size_t compute_invocation_id;
};

struct DispatchIdentity {
  uint32_t op_index;
  size_t runtime_node_id;
  size_t compute_invocation_id;
  uintptr_t function_pointer;

  bool operator==(const DispatchIdentity& other) const {
    return op_index == other.op_index &&
        runtime_node_id == other.runtime_node_id &&
        compute_invocation_id == other.compute_invocation_id &&
        function_pointer == other.function_pointer;
  }
};

struct DispatchIdentityHash {
  size_t operator()(const DispatchIdentity& value) const {
    size_t hash = std::hash<uintptr_t>{}(value.function_pointer);
    hash ^= std::hash<uint32_t>{}(value.op_index) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
    hash ^= std::hash<size_t>{}(value.runtime_node_id) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
    hash ^= std::hash<size_t>{}(value.compute_invocation_id) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
    return hash;
  }
};

std::mutex binding_mutex;
std::unordered_map<const void*, Binding> bindings;
std::mutex dispatch_identity_mutex;
std::unordered_set<DispatchIdentity, DispatchIdentityHash> observed_dispatches;
std::once_flag symbol_once;
std::mutex symbol_mutex;
std::unordered_map<uintptr_t, std::pair<const char*, const char*>> symbols;
std::unordered_map<std::string, std::pair<const char*, const char*>> symbols_by_name;
std::atomic<bool> trace_enabled{false};
#if defined(_WIN32)
bool dbghelp_ready = false;
#endif

const char* DeepBomNodeTypeName(enum xnn_node_type type) {
  switch (type) {
#define XNN_ENUM_ITEM(enum_name, enum_string) case enum_name: return #enum_name;
#include "src/xnnpack/node-type-defs.inc"
#undef XNN_ENUM_ITEM
    default: return "<invalid-node-type>";
  }
}

bool HasTracePath() {
#if defined(_WIN32)
  char* value = nullptr;
  size_t length = 0;
  const bool present = _dupenv_s(&value, &length, "DEEPBOM_RUNTIME_EVENTS_PATH") == 0 &&
      value != nullptr && value[0] != '\0';
  std::free(value);
  return present;
#else
  const char* value = std::getenv("DEEPBOM_RUNTIME_EVENTS_PATH");
  return value != nullptr && value[0] != '\0';
#endif
}

void CopyName(char* output, const char* input) {
  std::memset(output, 0, kNameCapacity);
  if (input != nullptr) std::strncpy(output, input, kNameCapacity - 1);
}

void CloseTrace() {
  if (trace_enabled.exchange(false)) deepbom_runtime_trace_close();
}

void BuildSymbolMap() {
  for (size_t i = 0; i < deepbom_xnn_symbol_catalog_size; ++i) {
    const DeepBomXnnSymbolCatalogEntry& entry = deepbom_xnn_symbol_catalog[i];
    symbols_by_name.emplace(entry.symbol,
                            std::make_pair(entry.symbol, entry.source_path));
    void* address = reinterpret_cast<void*>(entry.address);
#if defined(_WIN32)
    if (address == nullptr) {
      HMODULE module = GetModuleHandleW(nullptr);
      if (module != nullptr) address = reinterpret_cast<void*>(GetProcAddress(module, entry.symbol));
    }
#elif defined(__unix__) || defined(__APPLE__)
    if (address == nullptr) address = dlsym(RTLD_DEFAULT, entry.symbol);
#endif
    if (address != nullptr) {
      symbols.emplace(reinterpret_cast<uintptr_t>(address),
                      std::make_pair(entry.symbol, entry.source_path));
    }
  }
#if defined(_WIN32)
  SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
  dbghelp_ready = SymInitialize(GetCurrentProcess(), nullptr, TRUE) == TRUE;
#endif
}

std::pair<const char*, const char*> ResolveSymbol(uintptr_t address) {
  std::lock_guard<std::mutex> lock(symbol_mutex);
  const auto known = symbols.find(address);
  if (known != symbols.end()) return known->second;
#if defined(_WIN32)
  if (dbghelp_ready) {
    std::array<unsigned char, sizeof(SYMBOL_INFO) + MAX_SYM_NAME> storage{};
    auto* info = reinterpret_cast<SYMBOL_INFO*>(storage.data());
    info->SizeOfStruct = sizeof(SYMBOL_INFO);
    info->MaxNameLen = MAX_SYM_NAME;
    DWORD64 displacement = 0;
    if (SymFromAddr(GetCurrentProcess(), static_cast<DWORD64>(address),
                    &displacement, info) == TRUE && displacement == 0) {
      const auto source = symbols_by_name.find(info->Name);
      if (source != symbols_by_name.end()) {
        symbols.emplace(address, source->second);
        return source->second;
      }
    }
  }
#endif
  return {nullptr, nullptr};
}

void Emit(const Binding& binding, const char* symbol, const char* source_path) {
  if (!trace_enabled.load()) return;
  std::string source_ref = "google/XNNPACK@23a67314f7afdbb76191589ae090d82bf55afbfa/";
  source_ref += source_path;
  DeepBomRuntimeTraceEvent event{};
  event.op_index = binding.op_index;
  event.op_name = binding.op_name.c_str();
  event.provider = "XNNPACK";
  event.delegated = 1;
  event.partition_id = nullptr;
  event.lowering_id = binding.lowering_id.c_str();
  event.kernel_id = symbol;
  event.kernel = symbol;
  event.kernel_source_ref = source_ref.c_str();
  event.duration_us = -1.0;
  event.event_kind = "dispatch";
  event.runtime_node_id = binding.runtime_node_id;
  event.has_runtime_node_id = 1;
  event.compute_invocation_id = binding.compute_invocation_id;
  event.has_compute_invocation_id = 1;
  deepbom_runtime_trace_emit(&event);
}
}  // namespace

extern "C" enum xnn_status xnn_deepbom_trace_open(void) {
  if (trace_enabled.load() || !HasTracePath()) return xnn_status_success;
  if (deepbom_runtime_trace_open() != 0) return xnn_status_invalid_state;
  trace_enabled.store(true);
  std::atexit(CloseTrace);
  return xnn_status_success;
}

extern "C" uint32_t xnn_deepbom_no_operator_fusion_flag(void) {
  return XNN_FLAG_NO_OPERATOR_FUSION;
}

extern "C" void xnn_deepbom_trace_placement(uint32_t op_index,
                                               const char* op_name,
                                               bool delegated,
                                               const char* partition_id) {
  if (!trace_enabled.load() && xnn_deepbom_trace_open() != xnn_status_success) return;
  DeepBomRuntimeTraceEvent event{};
  event.op_index = op_index;
  event.op_name = op_name;
  event.provider = delegated ? "XNNPACK" : "TFLite CPU";
  event.delegated = delegated ? 1 : 0;
  event.partition_id = partition_id;
  event.duration_us = -1.0;
  event.event_kind = "placement";
  deepbom_runtime_trace_emit(&event);
}

extern "C" size_t xnn_deepbom_subgraph_node_count(xnn_subgraph_t subgraph) {
  return subgraph == nullptr ? 0 : subgraph->num_nodes;
}

extern "C" void xnn_deepbom_tag_subgraph_nodes(xnn_subgraph_t subgraph,
                                                  size_t first_node_id,
                                                  uint32_t op_index,
                                                  const char* op_name) {
  if (subgraph == nullptr || first_node_id > subgraph->num_nodes) return;
  xnn_deepbom_trace_open();
  for (size_t node_id = first_node_id; node_id < subgraph->num_nodes; ++node_id) {
    struct xnn_node* node = &subgraph->nodes[node_id];
    node->deepbom_source_op_index = op_index;
    node->deepbom_source_valid = true;
    CopyName(node->deepbom_source_op_name, op_name);
    if (!trace_enabled.load()) continue;
    const char* lowering = DeepBomNodeTypeName(node->type);
    DeepBomRuntimeTraceEvent event{};
    event.op_index = op_index;
    event.op_name = op_name;
    event.provider = "XNNPACK";
    event.delegated = 1;
    event.lowering_id = lowering;
    event.duration_us = -1.0;
    event.event_kind = "lowering";
    event.runtime_node_id = node_id;
    event.has_runtime_node_id = 1;
    deepbom_runtime_trace_emit(&event);
  }
}

extern "C" void xnn_deepbom_bind_operator(
    const struct xnn_operator_data* opdata, xnn_operator_t op,
    size_t runtime_node_id) {
  if (opdata == nullptr || op == nullptr || !opdata->deepbom_source_valid) return;
  std::lock_guard<std::mutex> lock(binding_mutex);
  const void* context_root = op->dynamic_context.gemm != nullptr
      ? static_cast<const void*>(op->dynamic_context.gemm)
      : static_cast<const void*>(&op->context);
  const uint8_t* context_base = reinterpret_cast<const uint8_t*>(context_root);
  for (size_t i = 0; i < static_cast<size_t>(op->num_compute_invocations); ++i) {
    const void* context = context_base + op->compute[i].context_offset;
    bindings.insert_or_assign(context, Binding{
        opdata->deepbom_source_op_index,
        opdata->deepbom_source_op_name,
        DeepBomNodeTypeName(opdata->type),
        runtime_node_id,
        i,
    });
  }
}

extern "C" void xnn_deepbom_record_dispatch(const void* context,
                                               uintptr_t function_pointer) {
  if (!trace_enabled.load() || context == nullptr || function_pointer == 0) return;
  Binding binding;
  {
    std::lock_guard<std::mutex> lock(binding_mutex);
    const auto found = bindings.find(context);
    if (found == bindings.end()) return;
    binding = found->second;
  }
  {
    std::lock_guard<std::mutex> lock(dispatch_identity_mutex);
    if (!observed_dispatches.insert(DispatchIdentity{
            binding.op_index, binding.runtime_node_id,
            binding.compute_invocation_id, function_pointer}).second) {
      return;
    }
  }
  std::call_once(symbol_once, BuildSymbolMap);
  const auto symbol = ResolveSymbol(function_pointer);
  if (symbol.first == nullptr || symbol.second == nullptr) return;
  Emit(binding, symbol.first, symbol.second);
}
