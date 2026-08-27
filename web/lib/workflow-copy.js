import { formatEvidenceScope } from "./format-evidence-scope.js";

export function auditFocusCopyFor(tabId, format = "tflite") {
  const normalized = String(format).toLowerCase();
  const scope = formatEvidenceScope(normalized);
  const onnx = normalized === "onnx";
  const gguf = normalized === "gguf";
  const safeTensors = normalized === "safetensors";
  const coreMl = normalized === "coreml";
  const copy = {
    overview: {
      title: "Overview",
      detail: `${scope.assessed}. ${scope.runtimeBoundary}.`,
    },
    xnnpack: {
      title: "TFLite / XNNPACK",
      detail: "Source-pinned predicted partition view: conditional chain continuity, classified breaks, fallback traffic, packing overhead, and zero-MAC copy pressure. Executed assignment remains separate.",
    },
    quant: {
      title: gguf ? "GGUF Encodings" : safeTensors ? "Tensor Storage" : coreMl ? "Core ML Numerics" : "Quantization",
      detail: gguf
        ? "Source-pinned GGML block types, stored bits per element, exact tensor payload ranges, and unsupported encoding coverage."
        : safeTensors
          ? "Declared tensor dtypes, shapes, exact byte cardinality, and payload range conservation; execution quantization is not inferred."
          : coreMl
            ? "Core ML numerical payload coverage with explicit separation between assessed weight encodings and unassessed program contracts."
            : "Numerical contract view: quantization coverage, per-tensor/per-channel scales, scale dispersion, zero-point status, and INT8 risk.",
    },
    accelerator: {
      title: "Accelerator Placement",
      detail: "Source-backed eligibility, conditional segments, fallback islands, boundary payload, build conditions, and unresolved runtime evidence. This is not an observed assignment or a GPU timing model.",
    },
    "quant-labs": {
      title: "Quant Labs",
      detail: "Detailed numerical evidence: Q/DQ flow, evidence-bounded PTQ/QAT experiment posture, residual contracts, integer safety, numerical ABI, and preprocessing.",
    },
    llm: {
      title: "On-device LLM",
      detail: "Artifact-bound architecture, tensor storage, tokenizer and chat-template identity, dense/MoE/SSM state and compute scenarios, validated runtime residency/offload/paging when supplied, plus explicit medical-AI evidence boundaries.",
    },
    roofline: {
      title: onnx ? "Intensity" : "Roofline",
      detail: onnx
        ? "Static arithmetic-intensity posture from assessed MACs and logical tensor bytes; no target throughput model is applied."
        : "Target-performance view: arithmetic intensity, low/mixed/high-intensity posture, and selected target assumptions.",
    },
    stage: {
      title: "Stage",
      detail: "Topology view: spatial stages, repeated blocks, MAC concentration, and where delegate or memory risk clusters.",
    },
  };
  return copy[tabId] || copy.overview;
}

export function performanceVisualCopyFor(tabId, format = "tflite") {
  const normalized = String(format).toLowerCase();
  const onnx = normalized === "onnx";
  const gguf = normalized === "gguf";
  const safeTensors = normalized === "safetensors";
  const coreMl = normalized === "coreml";
  const copy = {
    overview: {
      title: "Performance Visuals",
      subtitle: "Compute concentration, estimated bottleneck contribution, and target fit for the selected artifact.",
      status: "overview",
    },
    xnnpack: {
      title: "TFLite / XNNPACK Predicted Partition Flow",
      subtitle: "Horizontal source-pinned candidates show where conditional partitions continue, break, and return to XNNPACK eligibility; selected-build and runtime assignment remain separate evidence.",
      status: "conditional placement",
    },
    quant: {
      title: gguf ? "GGUF Tensor Encoding" : safeTensors ? "SafeTensors Storage Contract" : coreMl ? "Core ML Numerical Contract" : "Quantization Summary",
      subtitle: gguf
        ? "Serialized GGML block encodings and payload bytes are reconciled per tensor without inventing an operator graph."
        : safeTensors
          ? "Tensor dtype, shape, and byte ranges are reconciled exactly; affine execution metadata is reported as absent from the format."
          : coreMl
            ? "Only numerical contracts decoded from the selected Core ML representation are classified; skipped payloads remain explicitly unassessed."
            : "Graph-op state, MAC-bearing coverage, scale/zero-point checks, and stored-kernel warnings with explicit denominators.",
      status: gguf ? "tensor encodings" : safeTensors ? "storage evidence" : coreMl ? "Core ML evidence" : "quant summary",
    },
    accelerator: {
      title: "Accelerator Eligibility and Boundary Evidence",
      subtitle: "Backend and precision conditions are separated from static exclusions, conditional segments, transfer exposure, and observed runtime assignment.",
      status: "source eligibility",
    },
    "quant-labs": {
      title: "Quantization Labs",
      subtitle: onnx
        ? "Explicit ONNX Q/DQ flow and evidence-bounded intervention posture; TFLite-only numerical ledgers remain unavailable."
        : "Q/DQ boundaries, intervention posture, and independently validated numerical evidence ledgers.",
      status: "quant evidence",
    },
    llm: {
      title: "On-device LLM Evidence",
      subtitle: "Architecture and repository facts are separated from conditional state equations, runtime bindings, and medical deployment claims.",
      status: "LLM static contract",
    },
    roofline: {
      title: onnx ? "Static Intensity View" : "Roofline Pressure View",
      subtitle: onnx
        ? "Assessed MAC intensity and logical tensor traffic are shown without an ORT execution-provider throughput model."
        : "Stage-level memory pressure and cross-target comparison expose where the selected backend may bottleneck.",
      status: onnx ? "ONNX static assessment" : "roofline",
    },
    stage: {
      title: "Stage Performance View",
      subtitle: "Stage memory mix and MAC concentration reveal repeated blocks that deserve target profiling.",
      status: "stage view",
    },
  };
  return copy[tabId] || {
    title: "Performance Visuals",
    subtitle: "Static performance visualizations for the selected audit tab.",
    status: "static preflight",
  };
}

export function workflowActionCopyFor(workspace, fallback = {}, format = "tflite") {
  const scope = formatEvidenceScope(format);
  if (workspace === "input" && Number(fallback.availableIndex) > 0) {
    return {
      action: "Replace artifact or target",
      detail: "The current audit remains available until a replacement is selected and analyzed.",
    };
  }
  const copy = {
    input: ["Select artifact", ["tflite", "onnx"].includes(scope.id) ? "Choose a local artifact and reference target." : "Choose a local artifact or package."],
    audit: ["Review static evidence", `${scope.assessed}.`],
    findings: ["Review artifact findings", "Review format-applicable integrity, numerical, interface, package, and deployment-contract findings."],
    graph: ["Inspect serialized graph", "Search decoded ops, tensors, and producer/consumer links without treating unobserved runtime order as fact."],
    redesign: ["Review projection", "Compare an isolated, untrained structural scenario against the immutable source audit."],
    runtime: ["Run benchmark", "Prepare local inputs and measure browser runtime timing."],
    deepbom: ["Run Artifact Geometry", "Execute the controlled local WASM module and review the artifact-derived weight and topology descriptors."],
    perturbation: ["Run Perturbation", "Execute local input perturbation and output drift evidence for the deployment artifact."],
    runtime_basin: ["Run Backend Consistency", "Execute local backend availability and output drift checks."],
    deployment_sensitivity: ["Run Deployment Sensitivity Proxy", "Execute research-stage finite-difference deployment-function probes."],
    output: ["Review reports and evidence", "Download a login-free, watermarked Engineering or Regulatory Support Report and its verification manifest. Sign-in remains limited to reusable raw derivatives and controlled research bundles."],
  }[workspace];
  return {
    action: copy?.[0] || fallback.action || "",
    detail: copy?.[1] || fallback.detail || "",
  };
}

export function workflowStepIndexFor(order, step) {
  return order.indexOf(step);
}

export function preferredModuleWorkspaceFor({ requested, activeWorkspace, order }) {
  if (workflowStepIndexFor(order, requested) >= 0) return requested;
  if (workflowStepIndexFor(order, activeWorkspace) >= workflowStepIndexFor(order, "deepbom")) return activeWorkspace;
  return "deepbom";
}

export function moduleWorkspaceIdFor(workspace, moduleWorkspaces) {
  return moduleWorkspaces?.has?.(workspace) ? workspace : "";
}

export function workflowConfigForState(state, {
  selected = "selected artifact",
  detail = {},
  activeWorkspace = "input",
  order = [],
  format = "tflite",
} = {}) {
  const scope = formatEvidenceScope(format);
  const preferredModuleWorkspace = preferredModuleWorkspaceFor({ requested: detail.active, activeWorkspace, order });
  const states = {
    idle: {
      mode: "No artifact selected",
      active: "input",
      activeIndex: 0,
      availableIndex: 0,
      action: "Select artifact",
      detail: "Choose a local artifact or package; deployment context follows its resolved format.",
    },
    selected: {
      mode: "Artifact staged",
      active: "input",
      activeIndex: 0,
      availableIndex: 0,
      action: scope.runLabel,
      detail: `${selected} is staged locally for a ${scope.depth.toLowerCase()}. Review the estimate, then run the audit.`,
    },
    running: {
      mode: "Static audit running",
      active: "audit",
      activeIndex: 1,
      availableIndex: 1,
      action: "Parsing artifact",
      detail: "The browser is reading the local artifact and building analysis evidence.",
    },
    audited: {
      mode: scope.completion,
      active: "audit",
      activeIndex: 1,
      availableIndex: order.length - 1,
      action: "Review assessed evidence and boundaries",
      detail: `${scope.assessed}. ${scope.runtimeBoundary}.`,
    },
    runtime: {
      mode: "Runtime evidence updated",
      active: "runtime",
      activeIndex: workflowStepIndexFor(order, "runtime"),
      availableIndex: order.length - 1,
      action: "Review benchmark evidence",
      detail: "Compare local backend timing and inspect output status before exporting.",
    },
    module: {
      mode: "Advanced analysis complete",
      active: preferredModuleWorkspace,
      activeIndex: workflowStepIndexFor(order, preferredModuleWorkspace),
      availableIndex: order.length - 1,
      action: "Review module result",
      detail: "The selected module result is ready. Use Output for final report preview, downloads, and evidence bundle.",
    },
    locked: {
      mode: detail.title || "Runtime locked",
      active: "input",
      activeIndex: 0,
      availableIndex: 0,
      action: "Open latest deployment",
      detail: detail.detail || "This build cannot execute until the runtime guard passes.",
    },
  };
  return states[state] || states.idle;
}
