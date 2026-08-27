const XNNP_CODES = {
  "XNNP:Q:OK": "Quantized op — delegated (INT8 path)",
  "XNNP:Q:DTYPE": "Quantized delegation requires 8-bit input/output",
  "XNNP:Q:4D": "Quantized MEAN requires 4D input tensor",
  "XNNP:Q:POOL": "INT8 AVERAGE_POOL_2D not assumed delegated (conservative)",
  "XNNP:Q:SOFTMAX": "SOFTMAX not listed in XNNPACK quantized op table",
  "XNNP:Q:MM": "Quantized BATCH_MATMUL not in XNNPACK table",
  "XNNP:Q:GATHER": "GATHER is index-driven; not delegated in quantized path",
  "XNNP:Q:REDUCE": "Reduction op: axis/shape constraints prevent delegation",
  "XNNP:Q:META": "Shape/metadata op — CPU fallback (zero-MAC)",
  "XNNP:Q:REJECT": "No matching quantized support rule was found in the pinned rulepack; conservatively predicted non-delegated",
  "XNNP:F:OK": "FP32 op — delegated",
  "XNNP:F:DTYPE": "FP32 delegation requires FLOAT32 input/output",
  "XNNP:F:BIAS": "XNNPACK requires bias tensor (3rd input); bias-free export falls back",
  "XNNP:F:4D": "MEAN requires 4D [N,H,W,C] input for XNNPACK delegation",
  "XNNP:F:MM": "FP32 BATCH_MATMUL — shape/batch constraints; not assumed delegated",
  "XNNP:F:GATHER": "FP32 GATHER — index-driven access; not assumed delegated",
  "XNNP:F:REDUCE": "Reduction op: scalar/dynamic reductions fall back to CPU",
  "XNNP:F:META": "Shape metadata op — CPU (zero-MAC)",
  "XNNP:F:REJECT": "No matching FP32 support rule was found in the pinned rulepack; conservatively predicted non-delegated",
  "XNNP:PRED:STATIC": "Static prediction based on the pinned TensorFlow Lite XNNPACK README supported-op tables; actual partitioning depends on runtime",
};

const XNNP_PREFIX = {
  "XNNP:Q:COND:": (code) => `Quantized XNNPACK documentary constraint failed: ${formatConstraintId(code.slice("XNNP:Q:COND:".length))}`,
  "XNNP:F:COND:": (code) => `FP32 XNNPACK documentary constraint failed: ${formatConstraintId(code.slice("XNNP:F:COND:".length))}`,
  "XNNP:Q:CNT:": (code) => `Quantized CONCATENATION: XNNPACK supports 2–4 inputs (got ${code.split(":")[3]})`,
  "XNNP:F:CNT:": (code) => `FP32 op: XNNPACK supports 2–4 inputs/outputs (got ${code.split(":")[3]})`,
  "XNNP:F:REJECT:": (code) => `${code.split(":")[3]} has no matching FP32 support rule in the pinned rulepack; conservatively predicted non-delegated`,
};

function formatConstraintId(value) {
  return String(value || "unknown constraint").replaceAll("_", " ");
}

export function decodeXnnpReason(code) {
  if (!code) return "";
  if (XNNP_CODES[code]) return XNNP_CODES[code];
  for (const [prefix, fn] of Object.entries(XNNP_PREFIX)) {
    if (code.startsWith(prefix)) return fn(code);
  }
  return code;
}

export function decodeRoofReason(code) {
  if (!code) return "";
  if (!code.startsWith("ROOF:")) return code;
  const parts = code.split(":");
  if (parts[1] === "MEM" && parts[2] === "0") {
    return `Zero-MAC shape/structural operation / low-intensity posture (target: ${parts[3] || "?"})`;
  }
  const dtype = parts[1] === "F" ? "FP32" : "INT8";
  const intensity = parts[2];
  const memTh = parts[3];
  const cmpTh = parts[4];
  const target = parts[5];
  const flags = parts.slice(6).join(":");
  let msg = `${dtype} intensity ${intensity} ops/byte; thresholds: low <${memTh}, high ≥${cmpTh}; target: ${target}`;
  if (flags === "GATHER") msg += "; depthwise layout-sensitive on x86 SIMD";
  if (flags === "SVE2") msg += "; SVE2 high throughput shifts ridge";
  return msg;
}
