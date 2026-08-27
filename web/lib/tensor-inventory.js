export const TENSOR_ROLE_ORDER = ["kernel", "bias", "activation", "container_tensor", "metadata"];
export const TENSOR_ROLE_MAP_VERSION = "deepbom.tensor_role_map.2026-08-04.1";

const TFLITE_INPUT_ROLES = {
  CONV_2D: { kernel: [1], bias: [2] },
  DEPTHWISE_CONV_2D: { kernel: [1], bias: [2] },
  FULLY_CONNECTED: { kernel: [1], bias: [2] },
  TRANSPOSE_CONV: { kernel: [1], bias: [3] },
  BATCH_MATMUL: { kernel: [1] },
  SVDF: { kernel: [1, 2], bias: [3] },
  RNN: { kernel: [1, 2], bias: [3] },
  UNIDIRECTIONAL_SEQUENCE_RNN: { kernel: [1, 2], bias: [3] },
  LSTM: {
    kernel: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 20, 21, 22, 23],
    bias: [12, 13, 14, 15, 17],
  },
  UNIDIRECTIONAL_SEQUENCE_LSTM: {
    kernel: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 20, 21, 22, 23],
    bias: [12, 13, 14, 15, 17],
  },
  EMBEDDING_LOOKUP: { kernel: [1] },
  PRELU: { kernel: [1] },
};

const ONNX_INPUT_ROLES = {
  CONV: { kernel: [1], bias: [2] },
  CONVTRANSPOSE: { kernel: [1], bias: [2] },
  GEMM: { kernel: [1], bias: [2] },
  MATMUL: { kernel: [1] },
  MATMULINTEGER: { kernel: [1] },
  QLINEARCONV: { kernel: [3], bias: [8] },
  QLINEARMATMUL: { kernel: [3] },
  RNN: { kernel: [1, 2], bias: [3] },
  GRU: { kernel: [1, 2], bias: [3] },
  LSTM: { kernel: [1, 2], bias: [3] },
  BATCHNORMALIZATION: { kernel: [1], bias: [2] },
  INSTANCENORMALIZATION: { kernel: [1], bias: [2] },
  LAYERNORMALIZATION: { kernel: [1], bias: [2] },
  PRELU: { kernel: [1] },
};

function normalizedOpName(name) {
  return String(name || "").replace(/[^A-Za-z0-9]+/g, "_").toUpperCase();
}

export function tensorQuantizationMode(tensor) {
  const scales = Number(tensor?.quant_scales || 0);
  const parameterization = String(tensor?.quantization_parameterization || tensor?.scale_mode || "").toLowerCase();
  const quantized = scales > 0 || !["", "none", "not_bound"].includes(parameterization);
  if (!quantized) return "none";
  if (scales > 1 || /per.?axis|per.?channel|block/.test(parameterization)) return "per_channel";
  return "per_tensor";
}

export function classifyTensorRoles(analysis = {}) {
  const tensors = Array.isArray(analysis.tensors) ? analysis.tensors : [];
  const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
  const format = String(analysis.format || "tflite").toLowerCase();
  const onnx = format === "onnx";
  if (["gguf", "safetensors"].includes(format)) {
    return tensors.map((tensor, position) => ({
      index: Number.isInteger(tensor?.index) ? tensor.index : position,
      tensor,
      role: "container_tensor",
    }));
  }
  const roleSpecs = onnx ? ONNX_INPUT_ROLES : TFLITE_INPUT_ROLES;
  const tensorByIndex = new Map(tensors.map((tensor, position) => [
    Number.isInteger(tensor?.index) ? tensor.index : position,
    tensor,
  ]));
  const roles = new Map();

  const bindRole = (tensorIndex, role) => {
    const tensor = tensorByIndex.get(tensorIndex);
    if (!tensor?.constant_buffer) return;
    const assigned = roles.get(tensorIndex) || new Set();
    assigned.add(role);
    roles.set(tensorIndex, assigned);
  };

  for (const op of ops) {
    const spec = roleSpecs[normalizedOpName(op.name)];
    if (!spec) continue;
    for (const role of ["kernel", "bias"]) {
      for (const position of spec[role] || []) bindRole(Number(op.inputs?.[position]), role);
    }
  }

  return tensors.map((tensor, position) => {
    const index = Number.isInteger(tensor?.index) ? tensor.index : position;
    const assignedRoles = [...(roles.get(index) || [])];
    const role = tensor?.constant_buffer && assignedRoles.length === 1
      ? assignedRoles[0]
      : tensor?.constant_buffer ? "metadata" : "activation";
    return { index, tensor, role };
  });
}

export function buildTensorInventory(analysis = {}) {
  const tensors = Array.isArray(analysis.tensors) ? analysis.tensors : [];
  const classified = classifyTensorRoles(analysis);
  const grouped = new Map();
  for (const { tensor, role } of classified) {
    const dtype = String(tensor?.dtype || "UNKNOWN").toUpperCase();
    const mode = tensorQuantizationMode(tensor);
    const key = `${role}\u0000${dtype}`;
    const row = grouped.get(key) || {
      role,
      dtype,
      total: 0,
      quantized: 0,
      per_tensor: 0,
      per_channel: 0,
    };
    row.total += 1;
    if (mode !== "none") row.quantized += 1;
    if (mode === "per_tensor") row.per_tensor += 1;
    if (mode === "per_channel") row.per_channel += 1;
    grouped.set(key, row);
  }

  const rows = [...grouped.values()].sort((left, right) => {
    const roleDelta = TENSOR_ROLE_ORDER.indexOf(left.role) - TENSOR_ROLE_ORDER.indexOf(right.role);
    return roleDelta || left.dtype.localeCompare(right.dtype);
  });
  const sum = (field) => rows.reduce((total, row) => total + row[field], 0);
  return {
    schema: "deepbom.tensor_inventory.v1",
    role_map_version: TENSOR_ROLE_MAP_VERSION,
    evidence_class: "DERIVED",
    status: "assessed",
    method: "GGUF/SafeTensors directory entries are container tensors. For graph formats, non-constant graph tensors are activations; constant tensors at format-specific operator-signature kernel/bias input positions are classified by role, and unmatched or conflicting constants are conservatively classified as metadata/parameters. Quantization mode is derived from scale cardinality or ONNX parameterization.",
    tensor_count: tensors.length,
    quantized_tensors: sum("quantized"),
    per_tensor_tensors: sum("per_tensor"),
    per_channel_tensors: sum("per_channel"),
    rows,
  };
}

export function tensorInventoryConserves(inventory) {
  const rows = Array.isArray(inventory?.rows) ? inventory.rows : [];
  const sum = (field) => rows.reduce((total, row) => total + Number(row[field] || 0), 0);
  return sum("total") === Number(inventory?.tensor_count || 0)
    && sum("quantized") === Number(inventory?.quantized_tensors || 0)
    && sum("per_tensor") === Number(inventory?.per_tensor_tensors || 0)
    && sum("per_channel") === Number(inventory?.per_channel_tensors || 0)
    && sum("per_tensor") + sum("per_channel") === sum("quantized");
}
