#ifndef DEEPBOM_RUNTIME_TRACE_H_
#define DEEPBOM_RUNTIME_TRACE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DeepBomRuntimeTraceEvent {
  size_t op_index;
  const char* op_name;
  const char* provider;
  int delegated;
  const char* partition_id;
  const char* lowering_id;
  const char* kernel_id;
  const char* kernel;
  const char* kernel_source_ref;
  double duration_us;
  // Null or empty preserves the legacy "observation" event. New instrumented
  // runtimes use placement, lowering, dispatch, or execution.
  const char* event_kind;
  size_t runtime_node_id;
  int has_runtime_node_id;
  size_t compute_invocation_id;
  int has_compute_invocation_id;
} DeepBomRuntimeTraceEvent;

// Opens the path in DEEPBOM_RUNTIME_EVENTS_PATH and truncates any prior stream.
// Returns zero on success. A caller must open once before worker threads emit.
int deepbom_runtime_trace_open(void);

// Emits one original-graph-op event. delegated uses -1 for unknown, 0 for false,
// and 1 for true. duration_us uses a negative value when timing is not collected.
int deepbom_runtime_trace_emit(const DeepBomRuntimeTraceEvent* event);

// Starts one post-commit TFLite ArenaPlanner snapshot. The returned identifier
// is process-local and monotonically increasing. UINT64_MAX indicates failure.
uint64_t deepbom_runtime_trace_begin_memory_snapshot(
    size_t non_persistent_arena_bytes, size_t persistent_arena_bytes,
    size_t tensor_count, size_t execution_node_count, size_t allocation_count,
    size_t alias_count);

// Adds one owning allocation to a memory snapshot. arena must be exactly
// kTfLiteArenaRw or kTfLiteArenaRwPersistent.
int deepbom_runtime_trace_emit_memory_allocation(
    uint64_t snapshot_id, size_t tensor_index, const char* arena,
    size_t offset_bytes, size_t size_bytes, int32_t first_node,
    int32_t last_node);

// Adds one ArenaPlanner in-place tensor mapping to a memory snapshot.
int deepbom_runtime_trace_emit_memory_alias(
    uint64_t snapshot_id, size_t tensor_index,
    size_t shared_with_tensor_index);

// Flushes and closes the stream. Returns zero when all writes completed.
int deepbom_runtime_trace_close(void);

// Returns a process-lifetime diagnostic string for the last failed operation.
const char* deepbom_runtime_trace_last_error(void);

#ifdef __cplusplus
}
#endif

#endif  // DEEPBOM_RUNTIME_TRACE_H_
