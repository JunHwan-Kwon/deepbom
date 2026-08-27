#include "deepbom_runtime_trace.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>

namespace {
std::mutex trace_mutex;
std::ofstream trace_stream;
std::string last_error;
std::string build_identifier_sha256;
uint64_t next_memory_snapshot_id = 0;

void SetError(const std::string& message) { last_error = message; }

bool ReadEnvironment(const char* name, std::string* value) {
#ifdef _WIN32
  char* buffer = nullptr;
  size_t length = 0;
  if (_dupenv_s(&buffer, &length, name) != 0) return false;
  *value = buffer == nullptr ? "" : buffer;
  std::free(buffer);
  return length > 0;
#else
  const char* raw_value = std::getenv(name);
  *value = raw_value == nullptr ? "" : raw_value;
  return raw_value != nullptr;
#endif
}

bool IsLowerHexIdentity(const std::string& value, size_t length) {
  if (value.size() != length) return false;
  for (const char value_char : value) {
    if (!((value_char >= '0' && value_char <= '9') ||
          (value_char >= 'a' && value_char <= 'f'))) {
      return false;
    }
  }
  return true;
}

bool IsEventKind(const char* value, const char* expected) {
  return value != nullptr && std::string(value) == expected;
}

void WriteJsonString(std::ostream& output, const char* value) {
  output << '"';
  for (const unsigned char value_char : std::string(value == nullptr ? "" : value)) {
    switch (value_char) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (value_char < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(value_char) << std::dec << std::setfill(' ');
        } else {
          output << value_char;
        }
    }
  }
  output << '"';
}

void WriteOptionalString(std::ostream& output, const char* name,
                         const char* value) {
  output << ",\"" << name << "\":";
  if (value == nullptr || value[0] == '\0') {
    output << "null";
  } else {
    WriteJsonString(output, value);
  }
}
}  // namespace

extern "C" int deepbom_runtime_trace_open(void) {
  std::lock_guard<std::mutex> lock(trace_mutex);
  if (trace_stream.is_open()) {
    SetError("trace stream is already open");
    return 1;
  }
  std::string event_path;
  std::string build_id;
  if (!ReadEnvironment("DEEPBOM_RUNTIME_EVENTS_PATH", &event_path) ||
      event_path.empty()) {
    SetError("DEEPBOM_RUNTIME_EVENTS_PATH is required");
    return 1;
  }
  if (!ReadEnvironment("DEEPBOM_MICROKERNEL_BUILD_ID_SHA256", &build_id)) {
    build_id.clear();
  }
  build_identifier_sha256 = build_id;
  if (!IsLowerHexIdentity(build_identifier_sha256, 64)) {
    SetError("DEEPBOM_MICROKERNEL_BUILD_ID_SHA256 must be a lowercase SHA-256");
    return 1;
  }
  trace_stream.open(event_path, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!trace_stream.is_open()) {
    SetError("cannot open DEEPBOM_RUNTIME_EVENTS_PATH");
    return 1;
  }
  next_memory_snapshot_id = 0;
  last_error.clear();
  return 0;
}

extern "C" int deepbom_runtime_trace_emit(
    const DeepBomRuntimeTraceEvent* event) {
  std::lock_guard<std::mutex> lock(trace_mutex);
  if (!trace_stream.is_open() || event == nullptr || event->op_name == nullptr ||
      event->op_name[0] == '\0' || event->provider == nullptr ||
      event->provider[0] == '\0') {
    SetError("trace stream and non-empty op_name/provider are required");
    return 1;
  }
  if (event->delegated < -1 || event->delegated > 1 ||
      !std::isfinite(event->duration_us)) {
    SetError("delegated or duration_us is invalid");
    return 1;
  }
  const bool has_kernel_id = event->kernel_id != nullptr && event->kernel_id[0] != '\0';
  const bool has_kernel_symbol = event->kernel != nullptr && event->kernel[0] != '\0';
  const bool has_kernel_source =
      event->kernel_source_ref != nullptr && event->kernel_source_ref[0] != '\0';
  const bool has_kernel = has_kernel_id || has_kernel_symbol || has_kernel_source;
  const bool complete_kernel = has_kernel_id && has_kernel_symbol &&
      has_kernel_source && event->lowering_id != nullptr &&
      event->lowering_id[0] != '\0';
  if (has_kernel && !complete_kernel) {
    SetError("kernel evidence requires lowering, stable ID, symbol, and source ref");
    return 1;
  }
  const char* event_kind = event->event_kind == nullptr || event->event_kind[0] == '\0'
      ? "observation" : event->event_kind;
  const bool known_kind = IsEventKind(event_kind, "observation") ||
      IsEventKind(event_kind, "placement") || IsEventKind(event_kind, "lowering") ||
      IsEventKind(event_kind, "dispatch") || IsEventKind(event_kind, "execution");
  if (!known_kind || (event->has_runtime_node_id != 0 && event->has_runtime_node_id != 1) ||
      (event->has_compute_invocation_id != 0 && event->has_compute_invocation_id != 1)) {
    SetError("event kind or optional runtime identifier flag is invalid");
    return 1;
  }
  const bool has_lowering = event->lowering_id != nullptr && event->lowering_id[0] != '\0';
  if ((IsEventKind(event_kind, "placement") &&
       (has_lowering || has_kernel || event->duration_us >= 0.0 ||
        event->has_runtime_node_id || event->has_compute_invocation_id)) ||
      (IsEventKind(event_kind, "lowering") &&
       (!has_lowering || has_kernel || event->duration_us >= 0.0 ||
        !event->has_runtime_node_id || event->has_compute_invocation_id)) ||
      (IsEventKind(event_kind, "dispatch") &&
       (!complete_kernel || !event->has_runtime_node_id || !event->has_compute_invocation_id)) ||
      (IsEventKind(event_kind, "execution") &&
       (has_lowering || has_kernel || event->duration_us < 0.0 ||
        event->has_runtime_node_id || event->has_compute_invocation_id))) {
    SetError("event fields do not satisfy event-kind semantics");
    return 1;
  }

  std::ostringstream row;
  row << "{\"event_kind\":";
  WriteJsonString(row, event_kind);
  row << ",\"op_index\":" << event->op_index << ",\"op_name\":";
  WriteJsonString(row, event->op_name);
  row << ",\"provider\":";
  WriteJsonString(row, event->provider);
  row << ",\"delegated\":";
  if (event->delegated < 0) row << "null";
  else row << (event->delegated == 1 ? "true" : "false");
  WriteOptionalString(row, "partition_id", event->partition_id);
  WriteOptionalString(row, "lowering_id", event->lowering_id);
  WriteOptionalString(row, "kernel_id", event->kernel_id);
  WriteOptionalString(row, "kernel", event->kernel);
  WriteOptionalString(row, "kernel_source_ref", event->kernel_source_ref);
  row << ",\"kernel_build_identifier_sha256\":";
  if (has_kernel) WriteJsonString(row, build_identifier_sha256.c_str());
  else row << "null";
  row << ",\"duration_us\":";
  if (event->duration_us < 0.0) row << "null";
  else row << std::setprecision(std::numeric_limits<double>::max_digits10)
            << event->duration_us;
  row << ",\"runtime_node_id\":";
  if (event->has_runtime_node_id) row << event->runtime_node_id;
  else row << "null";
  row << ",\"compute_invocation_id\":";
  if (event->has_compute_invocation_id) row << event->compute_invocation_id;
  else row << "null";
  row << "}\n";
  trace_stream << row.str();
  trace_stream.flush();
  if (!trace_stream.good()) {
    SetError("runtime event write failed");
    return 1;
  }
  return 0;
}

extern "C" uint64_t deepbom_runtime_trace_begin_memory_snapshot(
    size_t non_persistent_arena_bytes, size_t persistent_arena_bytes,
    size_t tensor_count, size_t execution_node_count, size_t allocation_count,
    size_t alias_count) {
  std::lock_guard<std::mutex> lock(trace_mutex);
  if (!trace_stream.is_open()) {
    SetError("trace stream is not open for a memory snapshot");
    return std::numeric_limits<uint64_t>::max();
  }
  if (next_memory_snapshot_id == std::numeric_limits<uint64_t>::max()) {
    SetError("memory snapshot identifier overflow");
    return std::numeric_limits<uint64_t>::max();
  }
  const uint64_t snapshot_id = next_memory_snapshot_id++;
  std::ostringstream row;
  row << "{\"event_kind\":\"memory_snapshot\",\"memory_snapshot_id\":"
      << snapshot_id << ",\"non_persistent_arena_bytes\":"
      << non_persistent_arena_bytes << ",\"persistent_arena_bytes\":"
      << persistent_arena_bytes << ",\"tensor_count\":" << tensor_count
      << ",\"execution_node_count\":" << execution_node_count
      << ",\"allocation_count\":" << allocation_count
      << ",\"alias_count\":" << alias_count << "}\n";
  trace_stream << row.str();
  trace_stream.flush();
  if (!trace_stream.good()) {
    SetError("memory snapshot write failed");
    return std::numeric_limits<uint64_t>::max();
  }
  return snapshot_id;
}

extern "C" int deepbom_runtime_trace_emit_memory_allocation(
    uint64_t snapshot_id, size_t tensor_index, const char* arena,
    size_t offset_bytes, size_t size_bytes, int32_t first_node,
    int32_t last_node) {
  std::lock_guard<std::mutex> lock(trace_mutex);
  const bool known_arena = arena != nullptr &&
      (std::string(arena) == "kTfLiteArenaRw" ||
       std::string(arena) == "kTfLiteArenaRwPersistent");
  if (!trace_stream.is_open() || snapshot_id >= next_memory_snapshot_id ||
      !known_arena || size_bytes == 0 || first_node < 0 ||
      last_node < first_node ||
      offset_bytes > std::numeric_limits<size_t>::max() - size_bytes) {
    SetError("memory allocation event is invalid");
    return 1;
  }
  std::ostringstream row;
  row << "{\"event_kind\":\"memory_allocation\",\"memory_snapshot_id\":"
      << snapshot_id << ",\"tensor_index\":" << tensor_index
      << ",\"arena\":";
  WriteJsonString(row, arena);
  row << ",\"offset_bytes\":" << offset_bytes
      << ",\"size_bytes\":" << size_bytes
      << ",\"first_node\":" << first_node
      << ",\"last_node\":" << last_node << "}\n";
  trace_stream << row.str();
  trace_stream.flush();
  if (!trace_stream.good()) {
    SetError("memory allocation event write failed");
    return 1;
  }
  return 0;
}

extern "C" int deepbom_runtime_trace_emit_memory_alias(
    uint64_t snapshot_id, size_t tensor_index,
    size_t shared_with_tensor_index) {
  std::lock_guard<std::mutex> lock(trace_mutex);
  if (!trace_stream.is_open() || snapshot_id >= next_memory_snapshot_id ||
      tensor_index == shared_with_tensor_index) {
    SetError("memory alias event is invalid");
    return 1;
  }
  trace_stream << "{\"event_kind\":\"memory_alias\",\"memory_snapshot_id\":"
               << snapshot_id << ",\"tensor_index\":" << tensor_index
               << ",\"shared_with_tensor_index\":"
               << shared_with_tensor_index << "}\n";
  trace_stream.flush();
  if (!trace_stream.good()) {
    SetError("memory alias event write failed");
    return 1;
  }
  return 0;
}

extern "C" int deepbom_runtime_trace_close(void) {
  std::lock_guard<std::mutex> lock(trace_mutex);
  if (!trace_stream.is_open()) return 0;
  trace_stream.flush();
  const bool failed = !trace_stream.good();
  trace_stream.close();
  if (failed) {
    SetError("runtime event flush failed");
    return 1;
  }
  return 0;
}

extern "C" const char* deepbom_runtime_trace_last_error(void) {
  thread_local std::string error_snapshot;
  std::lock_guard<std::mutex> lock(trace_mutex);
  error_snapshot = last_error;
  return error_snapshot.c_str();
}
