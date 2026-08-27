#include "deepbom_runtime_trace.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

namespace {
bool SetEnvironment(const char* name, const char* value) {
#ifdef _WIN32
  return _putenv_s(name, value) == 0;
#else
  return setenv(name, value, 1) == 0;
#endif
}

int Fail(const char* message) {
  std::fprintf(stderr, "%s: %s\n", message, deepbom_runtime_trace_last_error());
  return 1;
}
}  // namespace

int main() {
  const char* output_path = "deepbom-runtime-trace-test.ndjson";
  const std::string build_id(64, 'a');
  if (!SetEnvironment("DEEPBOM_RUNTIME_EVENTS_PATH", output_path) ||
      !SetEnvironment("DEEPBOM_MICROKERNEL_BUILD_ID_SHA256", build_id.c_str())) {
    return Fail("cannot configure test environment");
  }
  if (deepbom_runtime_trace_open() != 0) return Fail("open failed");
  const DeepBomRuntimeTraceEvent event = {
      7,
      "CONV_2D",
      "XNNPACK",
      1,
      "xnn-0",
      "convolution_to_igemm",
      "f32-igemm-4x8-scalar",
      "xnn_f32_igemm_minmax_ukernel_4x8__scalar",
      "google/XNNPACK@23a67314f7afdbb76191589ae090d82bf55afbfa/src/f32-igemm/gen/f32-igemm-4x8-minmax.c",
      12.5,
      "dispatch",
      3,
      1,
      0,
      1,
  };
  if (deepbom_runtime_trace_emit(&event) != 0) return Fail("emit failed");
  DeepBomRuntimeTraceEvent invalid = event;
  invalid.kernel_id = nullptr;
  if (deepbom_runtime_trace_emit(&invalid) == 0) return Fail("incomplete kernel identity was accepted");
  const uint64_t memory_snapshot = deepbom_runtime_trace_begin_memory_snapshot(
      4096, 256, 3, 2, 2, 1);
  if (memory_snapshot == UINT64_MAX) return Fail("memory snapshot failed");
  if (deepbom_runtime_trace_emit_memory_allocation(
          memory_snapshot, 0, "kTfLiteArenaRw", 0, 1024, 0, 1) != 0 ||
      deepbom_runtime_trace_emit_memory_allocation(
          memory_snapshot, 1, "kTfLiteArenaRwPersistent", 0, 256, 0,
          INT32_MAX) != 0 ||
      deepbom_runtime_trace_emit_memory_alias(memory_snapshot, 2, 0) != 0) {
    return Fail("memory ledger emit failed");
  }
  if (deepbom_runtime_trace_emit_memory_allocation(
          memory_snapshot, 3, "unknown", 0, 32, 0, 0) == 0) {
    return Fail("unknown arena was accepted");
  }
  if (deepbom_runtime_trace_close() != 0) return Fail("close failed");

  std::ifstream input(output_path, std::ios::binary);
  const std::string row((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  std::remove(output_path);
  if (row.find("\"op_index\":7") == std::string::npos ||
      row.find("\"duration_us\":12.5") == std::string::npos ||
      row.find("\"event_kind\":\"dispatch\"") == std::string::npos ||
      row.find("\"event_kind\":\"memory_snapshot\"") == std::string::npos ||
      row.find("\"non_persistent_arena_bytes\":4096") == std::string::npos ||
      row.find("\"event_kind\":\"memory_allocation\"") == std::string::npos ||
      row.find("\"event_kind\":\"memory_alias\"") == std::string::npos ||
      row.find("\"runtime_node_id\":3") == std::string::npos ||
      row.find(build_id) == std::string::npos ||
      row.find("xnn_f32_igemm_minmax_ukernel_4x8__scalar") == std::string::npos) {
    std::fprintf(stderr, "unexpected trace row: %s\n", row.c_str());
    return 1;
  }
  return 0;
}
