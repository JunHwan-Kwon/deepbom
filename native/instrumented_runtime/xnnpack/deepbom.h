#ifndef XNNPACK_SRC_XNNPACK_DEEPBOM_H_
#define XNNPACK_SRC_XNNPACK_DEEPBOM_H_

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include "include/xnnpack.h"

#ifdef __cplusplus
extern "C" {
#endif

struct xnn_operator_data;

enum xnn_status xnn_deepbom_trace_open(void);
uint32_t xnn_deepbom_no_operator_fusion_flag(void);
void xnn_deepbom_trace_placement(uint32_t op_index, const char* op_name,
                                 bool delegated, const char* partition_id);
size_t xnn_deepbom_subgraph_node_count(xnn_subgraph_t subgraph);
void xnn_deepbom_tag_subgraph_nodes(xnn_subgraph_t subgraph,
                                    size_t first_node_id, uint32_t op_index,
                                    const char* op_name);
void xnn_deepbom_bind_operator(const struct xnn_operator_data* opdata,
                               xnn_operator_t op, size_t runtime_node_id);
void xnn_deepbom_record_dispatch(const void* context,
                                 uintptr_t function_pointer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_DEEPBOM_H_
