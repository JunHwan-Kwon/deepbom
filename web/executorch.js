import { BoundedFlatBufferReader } from "./lib/flatbuffer-reader.js";
import { convTransposeAxisPairs } from "./lib/guarded-integer-expression.js";
import {
  EXECUTORCH_OPERATOR_SIGNATURE_SOURCE,
  EXECUTORCH_PORTABLE_OPERATOR_SIGNATURES,
} from "./lib/executorch-operator-signatures.generated.js";
import {
  assessExecuTorchProcessedPayload,
  buildExecuTorchSelectedBuildBinding,
} from "./lib/executorch-build-binding.js";

export const EXECUTORCH_SCHEMA_SOURCE = Object.freeze({
  repository: "pytorch/executorch",
  release: "v1.4.1",
  commit: "e4d02f41f7909e8ed5bf4a14ffc520d733453d9f",
  program_schema_path: "schema/program.fbs",
  program_schema_sha256: "da7b809b497abea70e813cb23f39e46b6c76e2501d66ef3e91d4f9e1c9cfd62f",
  scalar_type_schema_path: "schema/scalar_type.fbs",
  scalar_type_schema_sha256: "a4c83c25ee7da8eedf61f04fe3df979e5866035763f0ee8d3ed68463e3baad8f",
  flat_tensor_schema_path: "schema/flat_tensor.fbs",
  flat_tensor_schema_sha256: "c010f3f7a5afdb1118d3690248c399fc5b39577d6b3e17b3e436d7e199271db6",
  extended_header_path: "schema/extended_header.cpp",
  extended_header_sha256: "fee4a644efb026c8cf72847ad34ed4cce37fb7ac207b9e26c77059941855ba0a",
  schema_version_path: "exir/version.py",
  schema_version_sha256: "d1853272c0ed0cf026ecec49f2ad6932d924cbca7b03a46d2ed16e73227a2047",
  runtime_loader_path: "runtime/executor/program.cpp",
  runtime_loader_sha256: "d38be8eeec0fac0cea8f25d61820bc6f6d2bac4f07a89f3cb9ce175649260ca9",
  exporter_schema_version: 0,
});

const LIMITS = Object.freeze({
  maxVectorElements: 2_000_000,
  maxStringBytes: 16 * 1024 * 1024,
  maxPlans: 4096,
  maxValues: 2_000_000,
  maxInstructions: 5_000_000,
  maxRank: 64,
});

const KERNEL_TYPES = Object.freeze({
  0: "NONE", 1: "Null", 2: "Int", 3: "Bool", 4: "Double", 5: "Tensor", 6: "String",
  7: "IntList", 8: "DoubleList", 9: "BoolList", 10: "TensorList", 11: "OptionalTensorList",
});
const INSTRUCTION_TYPES = Object.freeze({ 0: "NONE", 1: "KernelCall", 2: "DelegateCall", 3: "MoveCall", 4: "JumpFalseCall", 5: "FreeCall" });
const SHAPE_DYNAMISM = Object.freeze({ 0: "STATIC", 1: "DYNAMIC_BOUND", 2: "DYNAMIC_UNBOUND" });
const DEVICE_TYPES = Object.freeze({ 0: "CPU", 1: "CUDA" });
const DATA_LOCATIONS = Object.freeze({ 0: "INLINE", 1: "SEGMENT" });
const TENSOR_DATA_LOCATIONS = Object.freeze({ 0: "SEGMENT", 1: "EXTERNAL" });
const SCALAR_TYPES = Object.freeze({
  0: ["UINT8", 8], 1: ["INT8", 8], 2: ["INT16", 16], 3: ["INT32", 32], 4: ["INT64", 64],
  5: ["FLOAT16", 16], 6: ["FLOAT32", 32], 7: ["FLOAT64", 64], 11: ["BOOL", 8],
  12: ["QINT8", 8], 13: ["QUINT8", 8], 14: ["QINT32", 32], 15: ["BFLOAT16", 16],
  16: ["QUINT4X2", 4], 17: ["QUINT2X4", 2], 22: ["BITS16", 16],
  23: ["FLOAT8E5M2", 8], 24: ["FLOAT8E4M3FN", 8], 25: ["FLOAT8E5M2FNUZ", 8],
  26: ["FLOAT8E4M3FNUZ", 8], 27: ["UINT16", 16], 28: ["UINT32", 32], 29: ["UINT64", 64],
});

export function analyzeExecuTorchModel(bytes, filename = "model.pte", options = {}) {
  if (!(bytes instanceof Uint8Array)) throw new TypeError("ExecuTorch analysis requires a Uint8Array.");
  const identifier = readIdentifier(bytes);
  if (identifier === "ET12") return analyzeProgram(bytes, filename, options);
  if (identifier === "FT01") return analyzeFlatTensor(bytes, filename);
  throw new Error(`ExecuTorch artifact identifier ${JSON.stringify(identifier)} is unsupported; expected ET12 (.pte) or FT01 (.ptd).`);
}

function analyzeProgram(bytes, filename, options) {
  const extendedHeader = parseExtendedHeader(bytes);
  const programBytes = boundedProgramBytes(bytes, extendedHeader);
  const reader = new BoundedFlatBufferReader(programBytes, LIMITS);
  const root = reader.root("ET12");
  const version = reader.scalar(root, 0, "u32", 0);
  const segments = parseSegments(reader, root, 4, extendedHeader, bytes.byteLength, "Program.segments");
  const constantBuffers = reader.tableVector(root, 2, "Program.constant_buffer").map((table, index) => {
    const storage = reader.vector(table, 0, 1, `Program.constant_buffer[${index}].storage`);
    return { index, byte_length: storage?.length || 0 };
  });
  const delegateInlineData = reader.tableVector(root, 3, "Program.backend_delegate_data").map((table, index) => {
    const data = reader.byteVector(table, 0, `Program.backend_delegate_data[${index}].data`);
    return { index, byte_length: data.length, bytes: data };
  });
  const constantSegment = parseSubsegment(reader, reader.tableField(root, 5, "Program.constant_segment"), "Program.constant_segment", segments);
  const mutableSegments = reader.tableVector(root, 6, "Program.mutable_data_segments")
    .map((table, index) => parseSubsegment(reader, table, `Program.mutable_data_segments[${index}]`, segments));
  assertUnique(mutableSegments.map((item) => item.segment_index), "Program.mutable_data_segments segment_index");
  const namedData = parseNamedData(reader, root, 7, segments, "Program.named_data");
  assertUnique(namedData.map((item) => item.key), "Program.named_data key");
  if (constantBuffers.length > 0 && constantSegment?.offsets.length) {
    throw new Error("Program.constant_buffer and Program.constant_segment.offsets are mutually exclusive but both are populated.");
  }

  const plans = reader.tableVector(root, 1, "Program.execution_plan");
  if (plans.length > LIMITS.maxPlans) throw new Error(`Program.execution_plan count ${plans.length} exceeds ${LIMITS.maxPlans}.`);
  const state = { tensors: [], ops: [], plans: [], delegates: [], issues: [] };
  plans.forEach((table, index) => parseExecutionPlan(reader, table, index, state, {
    constantBuffers, delegateInlineData, constantSegment, mutableSegments, segments,
  }));
  assertUnique(state.plans.map((plan) => plan.name), "ExecutionPlan name");

  const processedPayloadEvidence = state.delegates.map((delegate) => {
    const payload = materializeDelegatePayload(delegate, delegateInlineData, segments, bytes);
    return {
      plan_index: delegate.plan_index,
      delegate_index: delegate.index,
      processed_location: delegate.processed_location,
      processed_index: delegate.processed_index,
      ...assessExecuTorchProcessedPayload(delegate.backend_id, payload),
    };
  });
  for (const row of processedPayloadEvidence) {
    if (row.structural_status === "CONTRADICTION_SOURCE_DECLARED_FLATBUFFER_ENVELOPE_INVALID") {
      state.issues.push({
        code: "EXECUTORCH_PROCESSED_BACKEND_PAYLOAD_CONTRADICTION",
        plan_index: row.plan_index,
        delegate_index: row.delegate_index,
        backend_id: row.backend_id,
        reason: row.structural_error,
      });
    }
  }
  const selectedBuildBinding = buildExecuTorchSelectedBuildBinding(
    state.delegates,
    state.ops,
    options.selectedBuildAttestation || null,
    options.selectedBuildInput || null,
  );
  if (selectedBuildBinding.status === "CONTRADICTION_SELECTED_BUILD_CANNOT_SATISFY_SERIALIZED_PROGRAM") {
    state.issues.push({
      code: "EXECUTORCH_SELECTED_BUILD_BINDING_CONTRADICTION",
      delegate_contradiction_count: selectedBuildBinding.delegate_contradiction_count,
      kernel_contradiction_count: selectedBuildBinding.kernel_contradiction_count,
    });
  }
  if (version !== EXECUTORCH_SCHEMA_SOURCE.exporter_schema_version) {
    state.issues.push({
      code: "EXECUTORCH_SCHEMA_VERSION_PIN_CONTRADICTION",
      observed_schema_version: version,
      pinned_exporter_schema_version: EXECUTORCH_SCHEMA_SOURCE.exporter_schema_version,
    });
  }

  const externalTensors = state.tensors.filter((tensor) => tensor.external_data_name);
  const externalResolution = resolveExternalTensorData(externalTensors, options.externalDataFiles || []);
  state.issues.push(...externalResolution.contract_mismatches.map((row) => ({
    code: "EXECUTORCH_EXTERNAL_TENSOR_CONTRACT_MISMATCH",
    tensor_name: row.name,
    reasons: row.reasons,
  })));
  const histogram = countRows(state.ops, (item) => item.name);
  const delegatedInstructions = state.ops.filter((item) => item.instruction_kind === "DelegateCall").length;
  const kernelInstructions = state.ops.filter((item) => item.instruction_kind === "KernelCall").length;
  const assessedKernelInstructions = state.ops.filter((item) => item.instruction_kind === "KernelCall" && item.macs_decimal != null).length;
  const signatureBoundKernelInstructions = state.ops.filter((item) => item.instruction_kind === "KernelCall" && item.signature_status === "source_bound").length;
  const unknownKernelInstructions = kernelInstructions - assessedKernelInstructions;
  const unknownComputeInstructions = delegatedInstructions + unknownKernelInstructions;
  const nonComputeInstructions = state.ops.length - kernelInstructions - delegatedInstructions;
  const assessedMacs = sumBigInt(state.ops.filter((item) => item.macs_decimal != null).map((item) => BigInt(item.macs_decimal)));
  const completeMacAssessment = unknownComputeInstructions === 0;
  const plannedMemory = buildPlannedMemory(state.plans);
  const plannedInstructionLiveness = buildInstructionPlannedLiveness(state);
  const constantBytes = sumBigInt([
    ...constantBuffers.map((item) => BigInt(item.byte_length)),
    ...(constantSegment ? [BigInt(constantSegment.segment_size)] : []),
  ]);
  const segmentBytes = sumBigInt(segments.map((item) => BigInt(item.size)));
  const inputs = state.tensors.filter((tensor) => tensor.role === "input" || tensor.role === "input_output");
  const outputs = state.tensors.filter((tensor) => tensor.role === "output" || tensor.role === "input_output");

  return {
    schema: "deepbom.static_analysis.executorch.v1.1",
    format: "executorch",
    executorch_container: "pte",
    filename,
    file_size: bytes.byteLength,
    version,
    subgraphs: state.plans.length,
    operator_count: state.ops.length,
    tensor_count: state.tensors.length,
    input_tensor_indices: inputs.map((tensor) => tensor.index),
    output_tensor_indices: outputs.map((tensor) => tensor.index),
    inputs,
    outputs,
    tensors: state.tensors,
    ops: state.ops,
    histogram,
    stages: state.plans.map((plan, index) => {
      const planOps = state.ops.filter((op) => op.plan_index === plan.index);
      const assessed = planOps.filter((op) => op.macs_decimal != null);
      const planMacs = sumBigInt(assessed.map((op) => BigInt(op.macs_decimal || 0)));
      const unassessed = planOps.filter((op) => op.instruction_kind === "DelegateCall" || (op.instruction_kind === "KernelCall" && op.macs_decimal == null));
      return {
      index,
      key: plan.name,
      name: plan.name,
      op_count: plan.instruction_count,
      first_op: plan.first_op,
      last_op: plan.last_op,
      channels: [],
      macs: unassessed.length ? null : safeNumber(planMacs),
      macs_decimal: unassessed.length ? null : planMacs.toString(),
      mac_percent: null,
      mac_assessed_ops: assessed.length,
      mac_not_assessed_ops: unassessed.length,
      summary_text: `instructions ${plan.instruction_count} (${plan.kernel_instruction_count} kernel + ${plan.delegate_instruction_count} delegate + ${plan.non_compute_instruction_count} move/control/free) / chains ${plan.chain_count} / tensors ${plan.tensor_count} / serialized operators ${plan.operator_registry_count} / source-bound portable signatures ${planOps.filter((op) => op.signature_status === "source_bound").length}/${plan.kernel_instruction_count} / delegates ${plan.delegate_count} / planned non-constant memory ${plan.non_const_memory_bytes_decimal} B`,
    };
    }),
    total_macs: completeMacAssessment ? safeNumber(assessedMacs) : null,
    total_macs_decimal: completeMacAssessment ? assessedMacs.toString() : null,
    total_ops: kernelInstructions || delegatedInstructions ? null : 0,
    mac_assessment: {
      status: completeMacAssessment
        ? kernelInstructions ? "assessed_source_bound_portable_kernel_signatures" : "assessed_no_compute_instructions"
        : assessedKernelInstructions ? "partial_source_bound_kernel_signatures" : "not_assessed_operator_signature_or_delegate_semantics_unbound",
      complete: completeMacAssessment,
      assessed_ops: nonComputeInstructions + assessedKernelInstructions,
      total_ops: state.ops.length,
      unknown_compute_instruction_count: unknownComputeInstructions,
      kernel_instruction_count: kernelInstructions,
      source_bound_kernel_instruction_count: signatureBoundKernelInstructions,
      mac_assessed_kernel_instruction_count: assessedKernelInstructions,
      mac_unassessed_kernel_instruction_count: unknownKernelInstructions,
      delegate_instruction_count: delegatedInstructions,
      non_compute_instruction_count: nonComputeInstructions,
      total_assessed_macs: safeNumber(assessedMacs),
      total_assessed_macs_decimal: assessedMacs.toString(),
      detail: unknownComputeInstructions
        ? `${signatureBoundKernelInstructions}/${kernelInstructions} KernelCall instruction(s) bind to the pinned ${EXECUTORCH_OPERATOR_SIGNATURE_SOURCE.portable_operator_count}-operator portable registry; ${assessedKernelInstructions} have exact nominal tensor-contraction MAC results. ${unknownKernelInstructions} kernel and ${delegatedInstructions} delegate call(s) remain unassessed, so no complete total is emitted.`
        : kernelInstructions
          ? `All ${kernelInstructions} KernelCall instruction(s) bind to the pinned portable registry and have exact nominal tensor-contraction MAC results. Non-contraction operators contribute zero to this explicitly bounded MAC metric.`
          : "Every serialized instruction is a move, control-flow, or free operation with no nominal tensor-contraction MAC semantics; the exact nominal MAC total is zero.",
    },
    tensor_liveness: {
      schema: "deepbom.executorch_planned_memory.v1",
      status: plannedInstructionLiveness.complete ? plannedInstructionLiveness.status : "observed_aot_plan",
      peak_live_payload_bytes: null,
      peak_live_payload_status: "not_promoted_aot_address_occupancy_is_not_runtime_logical_payload",
      peak_planned_live_allocation_bytes: plannedInstructionLiveness.complete ? safeNumber(plannedInstructionLiveness.peak) : null,
      peak_planned_live_allocation_decimal: plannedInstructionLiveness.complete ? plannedInstructionLiveness.peak.toString() : null,
      peak_planned_live_allocation_status: plannedInstructionLiveness.status,
      per_plan_instruction_liveness: plannedInstructionLiveness.plans,
      planned_non_const_memory_bytes: safeNumber(plannedMemory.total),
      planned_non_const_memory_decimal: plannedMemory.total.toString(),
      per_device: plannedMemory.perDevice,
      detail: plannedInstructionLiveness.detail,
    },
    size_breakdown: {
      model_size: bytes.byteLength,
      constants: safeNumber(constantBytes),
      constant_bytes_decimal: constantBytes.toString(),
      appended_segments: safeNumber(segmentBytes),
      appended_segment_bytes_decimal: segmentBytes.toString(),
      structural_and_padding: bytes.byteLength - Number(segmentBytes <= BigInt(Number.MAX_SAFE_INTEGER) ? segmentBytes : 0n),
      status: segmentBytes <= BigInt(bytes.byteLength) ? "assessed" : "not_assessed_safe_integer_overflow",
    },
    weight_integrity: {
      status: state.issues.length ? "warn" : "pass",
      assessed_tensors: state.tensors.filter((tensor) => tensor.constant_buffer).length,
      issues: state.issues,
      detail: "Serialized tensor shape/storage spans, allocation offsets, segment ranges, dim-order permutations, and supplied PTD external tensor contracts were checked. Numerical payload finiteness is not inferred for opaque delegate blobs.",
    },
    metadata_presence: buildMetadataPresence("pte", version, state.plans),
    executorch_program: {
      schema: "deepbom.executorch_program.v1",
      evidence_class: "OBSERVED_DERIVED",
      source: EXECUTORCH_SCHEMA_SOURCE,
      operator_signature_registry: EXECUTORCH_OPERATOR_SIGNATURE_SOURCE,
      identifier: "ET12",
      extended_header: extendedHeader,
      plans: state.plans,
      delegates: state.delegates,
      processed_backend_payloads: processedPayloadEvidence,
      selected_build_binding: selectedBuildBinding,
      kernel_instruction_count: kernelInstructions,
      delegate_instruction_count: delegatedInstructions,
      segments,
      constant_buffers: constantBuffers,
      constant_segment: constantSegment,
      mutable_data_segments: mutableSegments,
      named_data: namedData,
      external_tensor_data: externalResolution,
      graph_boundary: "Instruction order and EValue identity are serialized. Portable KernelCall direction is reconstructed only when the serialized operator and argument vector match the pinned source registry. DelegateCall internals and unmatched/custom kernel semantics remain unbound and are not guessed.",
    },
    runtime_compat: {
      min_runtime_version: null,
      derived_min_runtime_version: null,
      effective_min_runtime_version: null,
      min_runtime_version_status: "NOT_DERIVABLE_SCHEMA_VERSION_NOT_RELEASE_MONOTONIC",
      schema_version_status: version === EXECUTORCH_SCHEMA_SOURCE.exporter_schema_version
        ? "MATCHES_PINNED_EXPORTER_SCHEMA_VERSION"
        : "CONTRADICTION_PINNED_EXPORTER_SCHEMA_VERSION_MISMATCH",
      max_op_version: 0,
      version_driving_ops: [`ET12 schema version ${version}`],
      runtime_version_basis: `ExecuTorch schema and loader are pinned to ${EXECUTORCH_SCHEMA_SOURCE.release} @ ${EXECUTORCH_SCHEMA_SOURCE.commit}. Program schema version ${version} is observed; the pinned source does not expose a monotonic schema-version-to-minimum-release mapping, so no runtime floor is emitted.`,
      detail: "Serialized delegate IDs and compile specs are observed; backend availability and execution require the matching ExecuTorch runtime build.",
    },
    executorch_sections_suppressed: [
      ...(unknownKernelInstructions ? ["Nominal operator MAC totals for KernelCall rows not matching the pinned portable signature registry"] : []),
      "Delegate-internal graph and kernel selection hidden inside processed backend blobs",
      "Observed runtime allocation, placement, latency, and physical data movement",
    ],
    xnnpack_assumption: "NOT_APPLICABLE_EXECUTORCH: ET12 serialized BackendDelegate IDs are reported directly; XNNPACK eligibility is not inferred from a TFLite rulepack.",
    xnnpack_chains: [],
    xnnpack_chain_breaks: 0,
    markdown: buildProgramMarkdown(filename, version, state, plannedMemory, segmentBytes),
  };
}

function analyzeFlatTensor(bytes, filename) {
  const extendedHeader = parseExtendedHeader(bytes);
  const flatBytes = boundedProgramBytes(bytes, extendedHeader);
  const reader = new BoundedFlatBufferReader(flatBytes, LIMITS);
  const root = reader.root("FT01");
  const version = reader.scalar(root, 0, "u32", 0);
  const segments = parseSegments(reader, root, 1, extendedHeader, bytes.byteLength, "FlatTensor.segments");
  const namedTables = reader.tableVector(root, 2, "FlatTensor.named_data");
  const tensors = namedTables.map((table, index) => parseFlatNamedData(reader, table, index, segments));
  assertUnique(tensors.map((tensor) => tensor.name), "FlatTensor.named_data key");
  const segmentBytes = sumBigInt(segments.map((item) => BigInt(item.size)));
  return {
    schema: "deepbom.static_analysis.executorch.v1.1",
    format: "executorch",
    executorch_container: "ptd",
    filename,
    file_size: bytes.byteLength,
    version,
    subgraphs: 0,
    operator_count: 0,
    tensor_count: tensors.length,
    inputs: [],
    outputs: [],
    input_tensor_indices: [],
    output_tensor_indices: [],
    tensors,
    ops: [],
    histogram: [],
    stages: [],
    total_macs: null,
    total_ops: null,
    mac_assessment: { status: "not_applicable_data_container", complete: true, assessed_ops: 0, total_ops: 0 },
    tensor_liveness: { status: "not_applicable_data_container", peak_live_payload_bytes: null },
    size_breakdown: {
      model_size: bytes.byteLength,
      constants: safeNumber(segmentBytes),
      constant_bytes_decimal: segmentBytes.toString(),
      appended_segments: safeNumber(segmentBytes),
      appended_segment_bytes_decimal: segmentBytes.toString(),
      status: "assessed",
    },
    weight_integrity: {
      status: "pass",
      assessed_tensors: tensors.filter((tensor) => tensor.layout_status === "assessed").length,
      issues: [],
      detail: "FT01 named-data uniqueness, segment ranges, tensor cardinality, storage size, and dim-order permutations were checked.",
    },
    metadata_presence: buildMetadataPresence("ptd", version, []),
    executorch_flat_tensor: {
      schema: "deepbom.executorch_flat_tensor.v1",
      evidence_class: "OBSERVED_DERIVED",
      source: EXECUTORCH_SCHEMA_SOURCE,
      identifier: "FT01",
      extended_header: extendedHeader,
      segments,
      named_data: tensors.map((tensor) => ({ name: tensor.name, segment_index: tensor.segment_index, layout_status: tensor.layout_status })),
    },
    runtime_compat: {
      min_runtime_version: "",
      runtime_version_basis: `ExecuTorch FT01 schema pinned to ${EXECUTORCH_SCHEMA_SOURCE.release} @ ${EXECUTORCH_SCHEMA_SOURCE.commit}.`,
      detail: "PTD is named external tensor/blob data and does not serialize an execution plan.",
    },
    executorch_sections_suppressed: ["Execution graph, operator cost, placement, and latency: FT01 is a tensor-data container."],
    xnnpack_assumption: "NOT_APPLICABLE_EXECUTORCH_PTD",
    xnnpack_chains: [],
    xnnpack_chain_breaks: 0,
    markdown: buildFlatTensorMarkdown(filename, version, tensors, segmentBytes),
  };
}

function parseExecutionPlan(reader, table, planIndex, state, storage) {
  const name = reader.stringField(table, 0, `ExecutionPlan[${planIndex}].name`) || `method_${planIndex}`;
  const inputRefs = reader.scalarVector(table, 3, "i32", `${name}.inputs`);
  const outputRefs = reader.scalarVector(table, 4, "i32", `${name}.outputs`);
  const valueTables = reader.tableVector(table, 2, `${name}.values`);
  if (valueTables.length > LIMITS.maxValues) throw new Error(`${name}.values count ${valueTables.length} exceeds ${LIMITS.maxValues}.`);
  validateIndices(inputRefs, valueTables.length, `${name}.inputs`);
  validateIndices(outputRefs, valueTables.length, `${name}.outputs`);
  const inputSet = new Set(inputRefs);
  const outputSet = new Set(outputRefs);
  const valueMap = valueTables.map((valueTable, valueIndex) => parseEValue(reader, valueTable, planIndex, name, valueIndex, inputSet, outputSet, state, storage));

  const operators = reader.tableVector(table, 6, `${name}.operators`).map((operator, index) => ({
    index,
    name: reader.stringField(operator, 0, `${name}.operators[${index}].name`) || "UNKNOWN",
    overload: reader.stringField(operator, 1, `${name}.operators[${index}].overload`),
  }));
  const delegates = reader.tableVector(table, 7, `${name}.delegates`).map((delegate, index) => parseDelegate(reader, delegate, planIndex, name, index, storage));
  state.delegates.push(...delegates);
  const nonConstSizes = reader.scalarVector(table, 8, "i64", `${name}.non_const_buffer_sizes`);
  if (nonConstSizes.some((value) => value < 0n)) throw new Error(`${name}.non_const_buffer_sizes contains a negative value.`);
  const deviceRows = reader.tableVector(table, 9, `${name}.non_const_buffer_device`).map((row, index) => {
    const bufferIndex = reader.scalar(row, 0, "i32", 0);
    if (bufferIndex <= 0 || bufferIndex >= nonConstSizes.length) throw new Error(`${name}.non_const_buffer_device[${index}] references reserved or absent buffer ${bufferIndex}; planned memory IDs are 1..${Math.max(0, nonConstSizes.length - 1)}.`);
    const deviceType = reader.scalar(row, 1, "i8", 0);
    const deviceIndex = reader.scalar(row, 2, "i8", 0);
    if (!(deviceType in DEVICE_TYPES)) throw new Error(`${name}.non_const_buffer_device[${index}] has unknown device type ${deviceType}.`);
    if (deviceIndex < 0) throw new Error(`${name}.non_const_buffer_device[${index}] has negative device index ${deviceIndex}.`);
    return { buffer_index: bufferIndex, device_type: DEVICE_TYPES[deviceType], device_index: deviceIndex };
  });
  assertUnique(deviceRows.map((row) => row.buffer_index), `${name}.non_const_buffer_device buffer_index`);
  const deviceByBuffer = new Map(deviceRows.map((row) => [row.buffer_index, row]));
  const memoryBuffers = nonConstSizes.slice(1).map((size, offset) => {
    const index = offset + 1;
    const device = deviceByBuffer.get(index) || { device_type: "CPU", device_index: 0 };
    return { index, size_bytes: safeNumber(size), size_bytes_decimal: size.toString(), device_type: device.device_type, device_index: device.device_index };
  });
  validatePlannedTensorAllocations(state.tensors.filter((tensor) => tensor.plan_index === planIndex), nonConstSizes, name);

  let instructionCount = 0;
  const firstOp = state.ops.length;
  const chains = reader.tableVector(table, 5, `${name}.chains`).map((chain, chainIndex) => {
    const chainInputs = reader.scalarVector(chain, 0, "i32", `${name}.chains[${chainIndex}].inputs`);
    const chainOutputs = reader.scalarVector(chain, 1, "i32", `${name}.chains[${chainIndex}].outputs`);
    validateIndices(chainInputs, valueTables.length, `${name}.chains[${chainIndex}].inputs`);
    validateIndices(chainOutputs, valueTables.length, `${name}.chains[${chainIndex}].outputs`);
    const instructions = reader.tableVector(chain, 2, `${name}.chains[${chainIndex}].instructions`);
    instructionCount += instructions.length;
    if (instructionCount > LIMITS.maxInstructions) throw new Error(`${name} instruction count exceeds ${LIMITS.maxInstructions}.`);
    instructions.forEach((instruction, instructionIndex) => parseInstruction(reader, instruction, {
      planIndex, planName: name, chainIndex, instructionIndex, operators, delegates, values: valueMap, state,
    }));
    return { index: chainIndex, input_value_indices: chainInputs, output_value_indices: chainOutputs, instruction_count: instructions.length };
  });
  const lastOp = state.ops.length ? state.ops.length - 1 : firstOp - 1;
  const planOps = state.ops.slice(firstOp, lastOp + 1);
  const kernelInstructionCount = planOps.filter((op) => op.instruction_kind === "KernelCall").length;
  const delegateInstructionCount = planOps.filter((op) => op.instruction_kind === "DelegateCall").length;
  state.plans.push({
    index: planIndex,
    name,
    value_count: valueTables.length,
    tensor_count: valueMap.filter((value) => value.kind === "Tensor").length,
    input_value_indices: inputRefs,
    output_value_indices: outputRefs,
    operator_registry_count: operators.length,
    delegate_count: delegates.length,
    chain_count: chains.length,
    chains,
    instruction_count: instructionCount,
    kernel_instruction_count: kernelInstructionCount,
    delegate_instruction_count: delegateInstructionCount,
    non_compute_instruction_count: instructionCount - kernelInstructionCount - delegateInstructionCount,
    first_op: firstOp,
    last_op: lastOp,
    non_const_memory_buffers: memoryBuffers,
    reserved_non_const_buffer_zero_decimal: nonConstSizes[0]?.toString() ?? null,
    non_const_memory_bytes_decimal: sumBigInt(nonConstSizes.slice(1)).toString(),
  });
}

function parseEValue(reader, table, planIndex, planName, valueIndex, inputSet, outputSet, state, storage) {
  const type = reader.scalar(table, 0, "u8", 0);
  if (!(type in KERNEL_TYPES)) throw new Error(`${planName}.values[${valueIndex}] has unknown KernelTypes discriminator ${type}.`);
  const valueTable = reader.tableField(table, 1, `${planName}.values[${valueIndex}].${KERNEL_TYPES[type]}`);
  if (type !== 0 && !valueTable) throw new Error(`${planName}.values[${valueIndex}] discriminator ${KERNEL_TYPES[type]} lacks its union value.`);
  if (type !== 5) return parseNonTensorEValue(reader, valueTable, type, valueIndex, planName);
  const tensor = parseTensor(reader, valueTable, planIndex, planName, valueIndex, inputSet, outputSet, storage);
  tensor.index = state.tensors.length;
  state.tensors.push(tensor);
  return { index: valueIndex, kind: "Tensor", tensor_index: tensor.index };
}

function parseNonTensorEValue(reader, table, type, valueIndex, planName) {
  const base = { index: valueIndex, kind: KERNEL_TYPES[type] };
  if (type === 0 || type === 1) return { ...base, value: null };
  if (type === 2) {
    const value = reader.scalar(table, 0, "i64", 0n);
    return { ...base, value: safeNumber(value), value_decimal: value.toString() };
  }
  if (type === 3) return { ...base, value: reader.scalar(table, 0, "bool", false) };
  if (type === 4) return { ...base, value: reader.scalar(table, 0, "f64", 0) };
  if (type === 6) return { ...base, value: reader.stringField(table, 0, `${planName}.values[${valueIndex}].String`) || "" };
  if (type === 7) {
    const values = reader.scalarVector(table, 0, "i64", `${planName}.values[${valueIndex}].IntList`);
    return { ...base, values: values.map((value) => safeNumber(value)), values_decimal: values.map(String) };
  }
  if (type === 8) return { ...base, values: reader.scalarVector(table, 0, "f64", `${planName}.values[${valueIndex}].DoubleList`) };
  if (type === 9) return { ...base, values: reader.scalarVector(table, 0, "bool", `${planName}.values[${valueIndex}].BoolList`) };
  const refs = reader.scalarVector(table, 0, "i32", `${planName}.values[${valueIndex}].${KERNEL_TYPES[type]}`);
  return { ...base, evalue_indices: refs };
}

function parseTensor(reader, table, planIndex, planName, valueIndex, inputSet, outputSet, storage) {
  const scalarType = reader.scalar(table, 0, "i8", 0);
  const scalar = SCALAR_TYPES[scalarType];
  if (!scalar) throw new Error(`${planName}.values[${valueIndex}] has unsupported ScalarType ${scalarType}.`);
  const storageOffset = reader.scalar(table, 1, "i32", 0);
  const sizes = reader.scalarVector(table, 2, "i32", `${planName}.values[${valueIndex}].sizes`);
  const dimOrder = reader.scalarVector(table, 3, "u8", `${planName}.values[${valueIndex}].dim_order`);
  if (sizes.length > LIMITS.maxRank) throw new Error(`${planName}.values[${valueIndex}] rank ${sizes.length} exceeds ${LIMITS.maxRank}.`);
  validateDimOrder(dimOrder, sizes.length, `${planName}.values[${valueIndex}].dim_order`);
  const dataBufferIndex = reader.scalar(table, 5, "u32", 0);
  const allocation = reader.tableField(table, 6, `${planName}.values[${valueIndex}].allocation_info`);
  const allocationInfo = allocation ? {
    memory_id: reader.scalar(allocation, 0, "u32", 0),
    memory_offset_decimal: combineU32(reader.scalar(allocation, 1, "u32", 0), reader.scalar(allocation, 2, "u32", 0)).toString(),
  } : null;
  const dynamismValue = reader.scalar(table, 8, "i8", 0);
  if (!(dynamismValue in SHAPE_DYNAMISM)) throw new Error(`${planName}.values[${valueIndex}] has unknown shape dynamism ${dynamismValue}.`);
  const extra = reader.tableField(table, 9, `${planName}.values[${valueIndex}].extra_tensor_info`);
  const extraInfo = extra ? parseExtraTensorInfo(reader, extra, `${planName}.values[${valueIndex}]`) : null;
  const role = inputSet.has(valueIndex) && outputSet.has(valueIndex) ? "input_output" : inputSet.has(valueIndex) ? "input" : outputSet.has(valueIndex) ? "output" : dataBufferIndex > 0 && !allocation ? "initializer" : "intermediate";
  const shapeContract = deriveExecuTorchTensorShapeContract(sizes, SHAPE_DYNAMISM[dynamismValue], scalar[1], `${planName}.values[${valueIndex}]`);
  const { shape_status: shapeStatus, logical_elements: elements, logical_bytes: logicalBytes } = shapeContract;
  const storageContract = validateTensorStorage({
    planName, valueIndex, dataBufferIndex, allocationInfo, extraInfo, logicalBytes, storageOffset, storage,
  });
  return {
    plan_index: planIndex,
    evalue_index: valueIndex,
    name: extraInfo?.fully_qualified_name || `${planName}/value_${valueIndex}`,
    dtype: scalar[0],
    scalar_type_value: scalarType,
    storage_bits_per_element: scalar[1],
    shape: [...sizes],
    shape_signature: shapeContract.shape_signature,
    shape_declared: true,
    shape_dynamism: SHAPE_DYNAMISM[dynamismValue],
    shape_status: shapeStatus,
    shape_upper_bound: shapeContract.shape_upper_bound,
    dim_order: [...dimOrder],
    role,
    constant_buffer: dataBufferIndex > 0,
    data_buffer_index: dataBufferIndex,
    allocation_info: allocationInfo,
    storage_offset: storageOffset,
    logical_elements: safeNumber(elements),
    logical_elements_decimal: elements?.toString() ?? null,
    buffer_data_length: safeNumber(logicalBytes),
    buffer_data_length_decimal: logicalBytes?.toString() ?? null,
    buffer_data_length_status: shapeContract.logical_bytes_status,
    buffer_data_status: storageContract.status,
    serialized_storage_span_bytes: safeNumber(storageContract.span),
    serialized_storage_span_decimal: storageContract.span?.toString() ?? null,
    external_data_name: extraInfo?.location === "EXTERNAL" ? extraInfo.fully_qualified_name : "",
    device_type: extraInfo?.device_type || "CPU",
    device_index: extraInfo?.device_index || 0,
    quant_scales: 0,
    quant_zero_points: 0,
    quantization_parameterization: "not_serialized_in_program_tensor_contract",
  };
}

export function deriveExecuTorchTensorShapeContract(sizes, dynamism, bitsPerElement, label = "ExecuTorch tensor") {
  if (!Array.isArray(sizes) || !["STATIC", "DYNAMIC_BOUND", "DYNAMIC_UNBOUND"].includes(dynamism)
    || !Number.isSafeInteger(bitsPerElement) || bitsPerElement <= 0) throw new Error(`${label} shape contract is invalid.`);
  if (dynamism !== "DYNAMIC_UNBOUND" && sizes.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new Error(`${label} has a negative or non-integer ${dynamism} shape dimension.`);
  }
  const elements = dynamism === "DYNAMIC_UNBOUND" ? null : productBigInt(sizes, `${label} shape bound`);
  const logicalBytes = elements == null ? null : ceilBitsToBytes(elements, bitsPerElement);
  return {
    shape_status: dynamism === "STATIC" ? "assessed" : dynamism === "DYNAMIC_BOUND" ? "assessed_upper_bound" : "not_assessed_dynamic_unbound",
    shape_signature: dynamism === "STATIC" ? [...sizes] : sizes.map(() => -1),
    shape_upper_bound: dynamism === "DYNAMIC_BOUND" ? [...sizes] : null,
    logical_elements: elements,
    logical_bytes: logicalBytes,
    logical_bytes_status: dynamism === "STATIC" ? "exact_static"
      : dynamism === "DYNAMIC_BOUND" ? "exact_serialized_upper_bound" : "not_assessed_dynamic_unbound",
  };
}

function parseInstruction(reader, instruction, context) {
  const type = reader.scalar(instruction, 0, "u8", 0);
  if (!(type in INSTRUCTION_TYPES) || type === 0) throw new Error(`${context.planName}.chains[${context.chainIndex}].instructions[${context.instructionIndex}] has invalid discriminator ${type}.`);
  const argsTable = reader.tableField(instruction, 1, `${context.planName}.${INSTRUCTION_TYPES[type]}[${context.instructionIndex}]`);
  if (!argsTable) throw new Error(`${context.planName} instruction ${context.instructionIndex} lacks ${INSTRUCTION_TYPES[type]} payload.`);
  const base = {
    index: context.state.ops.length,
    plan_index: context.planIndex,
    plan_name: context.planName,
    chain_index: context.chainIndex,
    instruction_index: context.instructionIndex,
    instruction_kind: INSTRUCTION_TYPES[type],
    inputs: [], outputs: [], estimated_bytes: null, estimated_bytes_status: "not_assessed_argument_direction_unbound",
    macs: null, macs_status: "not_assessed_operator_semantics_unbound", mac_percent: null,
    static_bound_guess: "not-assessed", quantized_path: false, quantized_compute_path: false,
  };
  if (type === 1) {
    const operatorIndex = reader.scalar(argsTable, 0, "i32", 0);
    if (operatorIndex < 0 || operatorIndex >= context.operators.length) throw new Error(`${context.planName} KernelCall references operator ${operatorIndex} outside ${context.operators.length}.`);
    const operator = context.operators[operatorIndex];
    const refs = reader.scalarVector(argsTable, 1, "i32", `${context.planName}.KernelCall.args`);
    validateIndices(refs, context.values.length, `${context.planName}.KernelCall.args`);
    const name = operator.overload ? `${operator.name}.${operator.overload}` : operator.name;
    context.state.ops.push(bindKernelCall({ ...base, name, operator_index: operatorIndex, evalue_args: refs, tensor_args: tensorRefs(refs, context.values) }, refs, context));
  } else if (type === 2) {
    const delegateIndex = reader.scalar(argsTable, 0, "i32", 0);
    if (delegateIndex < 0 || delegateIndex >= context.delegates.length) throw new Error(`${context.planName} DelegateCall references delegate ${delegateIndex} outside ${context.delegates.length}.`);
    const refs = reader.scalarVector(argsTable, 1, "i32", `${context.planName}.DelegateCall.args`);
    validateIndices(refs, context.values.length, `${context.planName}.DelegateCall.args`);
    const delegate = context.delegates[delegateIndex];
    context.state.ops.push({ ...base, name: `DELEGATE:${delegate.backend_id}`, delegate_index: delegateIndex, evalue_args: refs, tensor_args: tensorRefs(refs, context.values), placement_evidence: "OBSERVED_SERIALIZED_DELEGATE_CALL" });
  } else if (type === 3) {
    const from = reader.scalar(argsTable, 0, "i32", 0);
    const to = reader.scalar(argsTable, 1, "i32", 0);
    validateIndices([from, to], context.values.length, `${context.planName}.MoveCall`);
    context.state.ops.push({ ...base, name: "MOVE", evalue_args: [from, to], tensor_args: tensorRefs([from, to], context.values), inputs: tensorRefs([from], context.values), outputs: tensorRefs([to], context.values), estimated_bytes_status: "not_applicable_move_semantics_do_not_prove_physical_copy", macs: 0, macs_status: "not_applicable" });
  } else if (type === 4) {
    const condition = reader.scalar(argsTable, 0, "i32", 0);
    const destination = reader.scalar(argsTable, 1, "i32", 0);
    validateIndices([condition], context.values.length, `${context.planName}.JumpFalseCall condition`);
    context.state.ops.push({ ...base, name: "JUMP_FALSE", evalue_args: [condition], tensor_args: tensorRefs([condition], context.values), inputs: tensorRefs([condition], context.values), destination_instruction: destination, estimated_bytes_status: "not_applicable_control_flow", macs: 0, macs_status: "not_applicable" });
  } else {
    const value = reader.scalar(argsTable, 0, "i32", 0);
    validateIndices([value], context.values.length, `${context.planName}.FreeCall`);
    context.state.ops.push({ ...base, name: "FREE", evalue_args: [value], tensor_args: tensorRefs([value], context.values), inputs: tensorRefs([value], context.values), estimated_bytes_status: "not_applicable_lifetime_control", macs: 0, macs_status: "not_applicable" });
  }
}

function bindKernelCall(base, refs, context) {
  const signature = EXECUTORCH_PORTABLE_OPERATOR_SIGNATURES[base.name];
  if (!signature) return {
    ...base,
    signature_status: "not_bound_operator_outside_pinned_portable_registry",
    signature_source: null,
  };
  const expectedLength = signature.argument_count + signature.appended_return_count;
  if (refs.length !== expectedLength) return signatureConflict(base, signature, context, "argument_count_mismatch", {
    expected_argument_vector_length: expectedLength,
    observed_argument_vector_length: refs.length,
  });
  const argumentRefs = refs.slice(0, signature.argument_count);
  const returnRefs = refs.slice(signature.argument_count);
  const outputPositions = signature.tensor_output_argument_positions.length
    ? signature.tensor_output_argument_positions : signature.tensor_inout_argument_positions;
  const declaredOutputRefs = outputPositions.map((position) => argumentRefs[position]);
  if (declaredOutputRefs.length !== returnRefs.length
    || declaredOutputRefs.some((value, index) => value !== returnRefs[index])) {
    return signatureConflict(base, signature, context, "output_alias_mismatch", {
      declared_output_evalue_indices: declaredOutputRefs,
      appended_return_evalue_indices: returnRefs,
    });
  }
  const inputEvalues = unique(signature.tensor_input_argument_positions.map((position) => argumentRefs[position]));
  const outputEvalues = unique([...declaredOutputRefs, ...returnRefs]);
  const mac = assessKernelMac(signature, argumentRefs, context.values, context.state.tensors);
  return {
    ...base,
    inputs: tensorRefs(inputEvalues, context.values),
    outputs: tensorRefs(outputEvalues, context.values),
    input_evalue_indices: inputEvalues,
    output_evalue_indices: outputEvalues,
    signature_status: "source_bound",
    signature_basis: signature.basis,
    signature_source_schema: signature.source_schema,
    signature_argument_count: signature.argument_count,
    signature_appended_return_count: signature.appended_return_count,
    nominal_mac_rule: signature.nominal_mac_rule,
    estimated_bytes_status: "not_assessed_physical_transfer_not_serialized",
    macs: mac.macs,
    macs_decimal: mac.decimal,
    macs_status: mac.status,
  };
}

function signatureConflict(base, signature, context, reason, detail) {
  context.state.issues.push({
    code: "EXECUTORCH_OPERATOR_SIGNATURE_CONFLICT",
    op_index: base.index,
    operator: base.name,
    reason,
    ...detail,
  });
  return {
    ...base,
    signature_status: `conflict_${reason}`,
    signature_basis: signature.basis,
    signature_source_schema: signature.source_schema,
    ...detail,
  };
}

function assessKernelMac(signature, argumentRefs, values, tensors) {
  const rule = signature.nominal_mac_rule;
  if (rule === "zero_nominal_tensor_contraction_macs") return exactMac(0n, "assessed_zero_nominal_tensor_contraction_macs");
  const argumentTensor = (position) => {
    const value = values[argumentRefs[position]];
    return value?.kind === "Tensor" ? tensors[value.tensor_index] : null;
  };
  try {
    if (rule === "mm" || rule === "bmm") return matrixMac(rule, argumentTensor(0), argumentTensor(1), argumentTensor(signature.tensor_output_argument_positions[0]));
    if (rule === "addmm") return matrixMac("mm", argumentTensor(1), argumentTensor(2), argumentTensor(signature.tensor_output_argument_positions[0]));
    if (rule === "convolution") {
      const input = argumentTensor(0);
      const weight = argumentTensor(1);
      const output = argumentTensor(signature.tensor_output_argument_positions[0]);
      const transposed = evalueBoolean(values[argumentRefs[6]]);
      const groups = evaluePositiveInteger(values[argumentRefs[8]]);
      if (transposed == null || groups == null) return unassessedMac("not_assessed_convolution_scalar_contract_unbound");
      if (transposed) {
        const stride = evaluePositiveIntegerList(values[argumentRefs[3]]);
        const padding = evalueNonnegativeIntegerList(values[argumentRefs[4]]);
        const dilation = evaluePositiveIntegerList(values[argumentRefs[5]]);
        const outputPadding = evalueNonnegativeIntegerList(values[argumentRefs[7]]);
        if (!stride || !padding || !dilation || !outputPadding) return unassessedMac("not_assessed_transposed_convolution_spatial_contract_unbound");
        return transposedConvolutionMac(input, weight, output, groups, stride, padding, dilation, outputPadding);
      }
      return convolutionMac(input, weight, output, groups);
    }
    if (rule === "convolution_backward") {
      const gradOutput = argumentTensor(0);
      const input = argumentTensor(1);
      const weight = argumentTensor(2);
      const transposed = evalueBoolean(values[argumentRefs[7]]);
      const groups = evaluePositiveInteger(values[argumentRefs[9]]);
      const outputMask = evalueBooleanList(values[argumentRefs[10]], 3);
      if (transposed == null || groups == null || !outputMask) {
        return unassessedMac("not_assessed_convolution_backward_scalar_contract_unbound");
      }
      const outputs = [argumentTensor(11), argumentTensor(12), argumentTensor(13)];
      return convolutionBackwardMac({
        gradOutput, input, weight, outputs, outputMask, transposed, groups,
        stride: evaluePositiveIntegerList(values[argumentRefs[4]]),
        padding: evalueNonnegativeIntegerList(values[argumentRefs[5]]),
        dilation: evaluePositiveIntegerList(values[argumentRefs[6]]),
        outputPadding: evalueNonnegativeIntegerList(values[argumentRefs[8]]),
      });
    }
  } catch (error) {
    return unassessedMac(`not_assessed_shape_contract_conflict:${error?.message || error}`);
  }
  return unassessedMac("not_assessed_nominal_mac_rule_unregistered");
}

export function assessExecuTorchPortableKernelMac(signatureKey, argumentRefs, values, tensors) {
  const signature = EXECUTORCH_PORTABLE_OPERATOR_SIGNATURES[signatureKey];
  if (!signature) return unassessedMac("not_assessed_operator_outside_pinned_portable_registry");
  if (!Array.isArray(argumentRefs) || argumentRefs.length !== signature.argument_count) {
    return unassessedMac("not_assessed_argument_count_mismatch");
  }
  return assessKernelMac(signature, argumentRefs, values, tensors);
}

function matrixMac(rule, left, right, output) {
  const leftShape = staticShape(left, `${rule} left`);
  const rightShape = staticShape(right, `${rule} right`);
  const outputShape = staticShape(output, `${rule} output`);
  if (rule === "mm") {
    if (leftShape.length !== 2 || rightShape.length !== 2 || outputShape.length !== 2) throw new Error("rank_mismatch");
    if (leftShape[1] !== rightShape[0] || outputShape[0] !== leftShape[0] || outputShape[1] !== rightShape[1]) throw new Error("dimension_mismatch");
    return exactMac(BigInt(leftShape[0]) * BigInt(leftShape[1]) * BigInt(rightShape[1]), "assessed_source_bound_mm");
  }
  if (leftShape.length !== 3 || rightShape.length !== 3 || outputShape.length !== 3) throw new Error("rank_mismatch");
  if (leftShape[0] !== rightShape[0] || leftShape[1] !== outputShape[1] || leftShape[2] !== rightShape[1]
    || outputShape[0] !== leftShape[0] || outputShape[2] !== rightShape[2]) throw new Error("dimension_mismatch");
  return exactMac(BigInt(leftShape[0]) * BigInt(leftShape[1]) * BigInt(leftShape[2]) * BigInt(rightShape[2]), "assessed_source_bound_bmm");
}

function convolutionMac(input, weight, output, groups) {
  const inputShape = staticShape(input, "convolution input");
  const weightShape = staticShape(weight, "convolution weight");
  const outputShape = staticShape(output, "convolution output");
  if (inputShape.length < 3 || inputShape.length !== weightShape.length || inputShape.length !== outputShape.length) throw new Error("rank_mismatch");
  if (inputShape[0] !== outputShape[0] || outputShape[1] !== weightShape[0]
    || inputShape[1] !== weightShape[1] * groups) throw new Error("channel_or_batch_mismatch");
  const outputElements = productBigInt(outputShape, "convolution output");
  const kernelInputElements = productBigInt(weightShape.slice(1), "convolution weight kernel");
  return exactMac(outputElements * kernelInputElements, "assessed_source_bound_convolution");
}

function transposedConvolutionMac(input, weight, output, groups, stride, padding, dilation, outputPadding) {
  const inputShape = staticShape(input, "transposed convolution input");
  const weightShape = staticShape(weight, "transposed convolution weight");
  const outputShape = staticShape(output, "transposed convolution output");
  if (inputShape.length < 3 || inputShape.length !== weightShape.length || inputShape.length !== outputShape.length) throw new Error("rank_mismatch");
  const spatialRank = inputShape.length - 2;
  if (![stride, padding, dilation, outputPadding].every((row) => row.length === spatialRank)) throw new Error("spatial_parameter_rank_mismatch");
  if (inputShape[0] !== outputShape[0] || inputShape[1] !== weightShape[0]
    || outputShape[1] !== weightShape[1] * groups || inputShape[1] % groups !== 0) throw new Error("channel_or_batch_mismatch");
  const pairCounts = [];
  for (let axis = 0; axis < spatialRank; axis += 1) {
    if (outputPadding[axis] >= Math.max(stride[axis], dilation[axis])) throw new Error("output_padding_outside_source_bound");
    const expected = (inputShape[axis + 2] - 1) * stride[axis] - 2 * padding[axis]
      + dilation[axis] * (weightShape[axis + 2] - 1) + outputPadding[axis] + 1;
    if (outputShape[axis + 2] !== expected) throw new Error("output_shape_mismatch");
    const pairs = convTransposeAxisPairs(
      BigInt(inputShape[axis + 2]), BigInt(weightShape[axis + 2]), BigInt(stride[axis]),
      BigInt(dilation[axis]), BigInt(padding[axis]), BigInt(outputShape[axis + 2]),
    );
    if (pairs == null) throw new Error("overlap_pair_count_exceeds_bound");
    pairCounts.push(pairs);
  }
  const value = [BigInt(inputShape[0]), BigInt(inputShape[1]), BigInt(weightShape[1]), ...pairCounts]
    .reduce((product, factor) => product * factor, 1n);
  return exactMac(value, "assessed_source_bound_transposed_convolution_overlap");
}

function convolutionBackwardMac({ gradOutput, input, weight, outputs, outputMask, transposed, groups, stride, padding, dilation, outputPadding }) {
  const inputShape = staticShape(input, "convolution backward input");
  const weightShape = staticShape(weight, "convolution backward weight");
  const gradOutputShape = staticShape(gradOutput, "convolution backward grad_output");
  if (outputMask[0] && !sameShape(staticShape(outputs[0], "convolution backward grad_input"), inputShape)) throw new Error("grad_input_shape_mismatch");
  if (outputMask[1] && !sameShape(staticShape(outputs[1], "convolution backward grad_weight"), weightShape)) throw new Error("grad_weight_shape_mismatch");
  const outputChannels = transposed ? weightShape[1] * groups : weightShape[0];
  if (outputMask[2] && !sameShape(staticShape(outputs[2], "convolution backward grad_bias"), [outputChannels])) throw new Error("grad_bias_shape_mismatch");
  if (!outputMask[0] && !outputMask[1]) return exactMac(0n, "assessed_source_bound_convolution_backward_bias_only");
  const forward = transposed
    ? stride && padding && dilation && outputPadding
      ? transposedConvolutionMac(input, weight, gradOutput, groups, stride, padding, dilation, outputPadding)
      : null
    : convolutionMac(input, weight, gradOutput, groups);
  if (!forward) return unassessedMac("not_assessed_transposed_convolution_backward_spatial_contract_unbound");
  const contractions = Number(outputMask[0]) + Number(outputMask[1]);
  return exactMac(BigInt(forward.decimal) * BigInt(contractions), transposed
    ? "assessed_source_bound_transposed_convolution_backward_overlap"
    : "assessed_source_bound_convolution_backward");
}

function staticShape(tensor, label) {
  if (!tensor || tensor.shape_status !== "assessed" || !Array.isArray(tensor.shape)
    || tensor.shape.some((value) => !Number.isSafeInteger(value) || value < 0)) throw new Error(`${label}_not_static`);
  return tensor.shape;
}

function evalueBoolean(value) { return value?.kind === "Bool" && typeof value.value === "boolean" ? value.value : null; }
function evalueBooleanList(value, expectedLength = null) {
  if (value?.kind !== "BoolList" || !Array.isArray(value.values)
    || value.values.some((item) => typeof item !== "boolean")
    || expectedLength != null && value.values.length !== expectedLength) return null;
  return [...value.values];
}
function evaluePositiveInteger(value) {
  if (value?.kind !== "Int" || !/^[1-9]\d*$/.test(String(value.value_decimal || ""))) return null;
  const result = BigInt(value.value_decimal);
  return result <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(result) : null;
}
function evalueIntegerList(value, predicate) {
  if (value?.kind !== "IntList" || !Array.isArray(value.values_decimal) || value.values_decimal.length !== value.values?.length) return null;
  const result = [];
  for (const decimal of value.values_decimal) {
    if (!/^-?\d+$/.test(String(decimal))) return null;
    const integer = BigInt(decimal);
    if (integer > BigInt(Number.MAX_SAFE_INTEGER) || integer < BigInt(Number.MIN_SAFE_INTEGER)) return null;
    const number = Number(integer);
    if (!predicate(number)) return null;
    result.push(number);
  }
  return result;
}
function evaluePositiveIntegerList(value) { return evalueIntegerList(value, (item) => item > 0); }
function evalueNonnegativeIntegerList(value) { return evalueIntegerList(value, (item) => item >= 0); }
function exactMac(value, status) { return { macs: safeNumber(value), decimal: value.toString(), status }; }
function unassessedMac(status) { return { macs: null, decimal: null, status }; }
function unique(values) { return [...new Set(values)]; }
function sameShape(left, right) { return left.length === right.length && left.every((value, index) => value === right[index]); }

function parseDelegate(reader, table, planIndex, planName, index, storage) {
  const backendId = reader.stringField(table, 0, `${planName}.delegates[${index}].id`);
  if (!backendId) throw new Error(`${planName}.delegates[${index}] has an empty backend id.`);
  const processed = reader.tableField(table, 1, `${planName}.delegates[${index}].processed`);
  if (!processed) throw new Error(`${planName}.delegates[${index}] lacks processed-data reference.`);
  const locationValue = reader.scalar(processed, 0, "i8", 0);
  if (!(locationValue in DATA_LOCATIONS)) throw new Error(`${planName}.delegates[${index}] has invalid processed-data location ${locationValue}.`);
  const dataIndex = reader.scalar(processed, 1, "u32", 0);
  const rows = locationValue === 0 ? storage.delegateInlineData || [] : storage.segments;
  if (dataIndex >= rows.length) throw new Error(`${planName}.delegates[${index}] references ${DATA_LOCATIONS[locationValue]} data ${dataIndex} outside ${rows.length}.`);
  const compileSpecs = reader.tableVector(table, 2, `${planName}.delegates[${index}].compile_specs`).map((spec, specIndex) => {
    const key = reader.stringField(spec, 0, `${planName}.delegates[${index}].compile_specs[${specIndex}].key`);
    const value = reader.byteVector(spec, 1, `${planName}.delegates[${index}].compile_specs[${specIndex}].value`);
    return { key, value_bytes: value.length, value_hex_prefix: hexPrefix(value) };
  });
  assertUnique(compileSpecs.map((item) => item.key), `${planName}.delegates[${index}] compile-spec key`);
  return { plan_index: planIndex, index, backend_id: backendId, processed_location: DATA_LOCATIONS[locationValue], processed_index: dataIndex, compile_specs: compileSpecs };
}

function materializeDelegatePayload(delegate, inlineData, segments, artifactBytes) {
  if (delegate.processed_location === "INLINE") {
    const payload = inlineData[delegate.processed_index]?.bytes;
    if (!(payload instanceof Uint8Array)) throw new Error(`ExecuTorch inline delegate payload ${delegate.processed_index} is unavailable.`);
    return payload;
  }
  const segment = segments[delegate.processed_index];
  if (!segment) throw new Error(`ExecuTorch delegate segment ${delegate.processed_index} is unavailable.`);
  const start = BigInt(segment.absolute_offset);
  const end = BigInt(segment.absolute_end);
  if (start > BigInt(Number.MAX_SAFE_INTEGER) || end > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("ExecuTorch delegate payload range exceeds the JavaScript safe integer range.");
  return artifactBytes.subarray(Number(start), Number(end));
}

function parseSegments(reader, root, fieldIndex, header, fileSize, label) {
  const tables = reader.tableVector(root, fieldIndex, label);
  const segments = tables.map((table, index) => {
    const offset = readCompatibleUnsigned(reader, table, 0, `${label}[${index}].offset`);
    const size = readCompatibleUnsigned(reader, table, 1, `${label}[${index}].size`);
    return { index, offset: offset.value.toString(), size: size.value.toString(), offset_wire_bits: offset.wireBits, size_wire_bits: size.wireBits };
  });
  const materialized = segments.some((segment) => BigInt(segment.size) > 0n);
  if (materialized && (!header.present || BigInt(header.segment_base_offset) === 0n)) throw new Error(`${label} contains materialized payload but no valid extended-header segment base is present.`);
  let previousEnd = 0n;
  for (const segment of segments) {
    const offset = BigInt(segment.offset);
    const size = BigInt(segment.size);
    if (offset < previousEnd) throw new Error(`${label}[${segment.index}] overlaps or is not sorted after the previous segment.`);
    const relativeEnd = offset + size;
    const absoluteEnd = BigInt(header.segment_base_offset) + relativeEnd;
    if (absoluteEnd > BigInt(fileSize)) throw new Error(`${label}[${segment.index}] ends at ${absoluteEnd}, beyond file size ${fileSize}.`);
    if (BigInt(header.segment_data_size || "0") > 0n && relativeEnd > BigInt(header.segment_data_size)) throw new Error(`${label}[${segment.index}] exceeds extended-header segment_data_size.`);
    segment.absolute_offset = (BigInt(header.segment_base_offset) + offset).toString();
    segment.absolute_end = absoluteEnd.toString();
    previousEnd = relativeEnd;
  }
  return segments;
}

function readCompatibleUnsigned(reader, table, fieldIndex, label) {
  const position = reader.field(table, fieldIndex, 1);
  if (position == null) return { value: 0n, wireBits: 0 };
  const relative = position - table.position;
  if (relative + 8 <= table.objectLength) return { value: reader.u64(position, label), wireBits: 64 };
  if (relative + 4 <= table.objectLength) return { value: BigInt(reader.u32(position, label)), wireBits: 32 };
  throw new Error(`${label} has neither a bounded uint64 nor legacy uint32 wire span.`);
}

function parseSubsegment(reader, table, label, segments) {
  if (!table) return null;
  const segmentIndex = reader.scalar(table, 0, "u32", 0);
  const offsets = reader.scalarVector(table, 1, "u64", `${label}.offsets`).map((value) => value.toString());
  if (segmentIndex >= segments.length && offsets.length) throw new Error(`${label} references segment ${segmentIndex} outside ${segments.length}.`);
  let previous = 0n;
  const segmentSize = segmentIndex < segments.length ? BigInt(segments[segmentIndex].size) : 0n;
  offsets.forEach((text, index) => {
    const value = BigInt(text);
    if (index > 0 && value < previous) throw new Error(`${label}.offsets is not monotonic at index ${index}.`);
    if (value > segmentSize) throw new Error(`${label}.offsets[${index}] exceeds segment size ${segmentSize}.`);
    previous = value;
  });
  return { segment_index: segmentIndex, segment_size: segmentSize.toString(), offsets };
}

function parseNamedData(reader, root, fieldIndex, segments, label) {
  return reader.tableVector(root, fieldIndex, label).map((table, index) => {
    const key = reader.stringField(table, 0, `${label}[${index}].key`);
    const segmentIndex = reader.scalar(table, 1, "u32", 0);
    if (!key) throw new Error(`${label}[${index}] has an empty key.`);
    if (segmentIndex >= segments.length) throw new Error(`${label}[${index}] references segment ${segmentIndex} outside ${segments.length}.`);
    return { key, segment_index: segmentIndex, segment_size: segments[segmentIndex].size };
  });
}

function parseFlatNamedData(reader, table, index, segments) {
  const name = reader.stringField(table, 0, `FlatTensor.named_data[${index}].key`);
  const segmentIndex = reader.scalar(table, 1, "u32", 0);
  if (!name) throw new Error(`FlatTensor.named_data[${index}] has an empty key.`);
  if (segmentIndex >= segments.length) throw new Error(`FlatTensor.named_data[${index}] references segment ${segmentIndex} outside ${segments.length}.`);
  const layout = reader.tableField(table, 2, `FlatTensor.named_data[${index}].tensor_layout`);
  if (!layout) return { index, name, role: "named_blob", dtype: "OPAQUE", shape: [], shape_declared: false, segment_index: segmentIndex, buffer_data_length: Number(segments[segmentIndex].size), layout_status: "not_applicable_blob" };
  const scalarType = reader.scalar(layout, 0, "i8", 0);
  const scalar = SCALAR_TYPES[scalarType];
  if (!scalar) throw new Error(`FlatTensor.named_data[${index}] has unsupported ScalarType ${scalarType}.`);
  const shape = reader.scalarVector(layout, 1, "i32", `FlatTensor.named_data[${index}].sizes`);
  if (shape.length > LIMITS.maxRank || shape.some((value) => value < 0)) throw new Error(`FlatTensor.named_data[${index}] has an invalid shape.`);
  const dimOrder = reader.scalarVector(layout, 2, "u8", `FlatTensor.named_data[${index}].dim_order`);
  validateDimOrder(dimOrder, shape.length, `FlatTensor.named_data[${index}].dim_order`);
  const elements = productBigInt(shape, `FlatTensor.named_data[${index}] shape`);
  const logicalBytes = ceilBitsToBytes(elements, scalar[1]);
  const segmentBytes = BigInt(segments[segmentIndex].size);
  if (logicalBytes > segmentBytes) throw new Error(`FlatTensor.named_data[${index}] requires ${logicalBytes} B but segment ${segmentIndex} contains ${segmentBytes} B.`);
  return {
    index, name, role: "initializer", dtype: scalar[0], scalar_type_value: scalarType, storage_bits_per_element: scalar[1],
    shape, shape_signature: [...shape], shape_declared: true, dim_order: dimOrder, segment_index: segmentIndex,
    logical_elements: safeNumber(elements), logical_elements_decimal: elements.toString(),
    buffer_data_length: safeNumber(logicalBytes), buffer_data_length_decimal: logicalBytes.toString(),
    serialized_storage_span_bytes: safeNumber(segmentBytes), serialized_storage_span_decimal: segmentBytes.toString(),
    buffer_data_status: logicalBytes === segmentBytes ? "observed_exact_segment_payload" : "observed_segment_with_padding_or_shared_span",
    layout_status: "assessed", constant_buffer: true, quant_scales: 0, quant_zero_points: 0,
  };
}

function parseExtraTensorInfo(reader, table, label) {
  const mutableIndexField = reader.field(table, 0, 8);
  const fullyQualifiedName = reader.stringField(table, 1, `${label}.fully_qualified_name`);
  const locationValue = reader.scalar(table, 2, "i8", 0);
  const deviceType = reader.scalar(table, 3, "i8", 0);
  const deviceIndex = reader.scalar(table, 4, "i8", 0);
  if (!(locationValue in TENSOR_DATA_LOCATIONS)) throw new Error(`${label} has unknown tensor data location ${locationValue}.`);
  if (!(deviceType in DEVICE_TYPES)) throw new Error(`${label} has unknown device type ${deviceType}.`);
  if (deviceIndex < 0) throw new Error(`${label} has negative device index ${deviceIndex}.`);
  if (locationValue === 1 && !fullyQualifiedName) throw new Error(`${label} is EXTERNAL but has no fully qualified name.`);
  return {
    mutable_data_segments_index: mutableIndexField == null ? null : reader.u64(mutableIndexField, `${label}.mutable_data_segments_idx`).toString(),
    fully_qualified_name: fullyQualifiedName,
    location: TENSOR_DATA_LOCATIONS[locationValue],
    device_type: DEVICE_TYPES[deviceType],
    device_index: deviceIndex,
  };
}

function validateTensorStorage({ planName, valueIndex, dataBufferIndex, allocationInfo, extraInfo, logicalBytes, storageOffset, storage }) {
  if (storageOffset !== 0) return { status: "not_runnable_nonzero_storage_offset", span: null };
  if (extraInfo?.location === "EXTERNAL") return { status: "external_named_data_required", span: null };
  if (dataBufferIndex === 0) return { status: allocationInfo ? "planned_mutable_allocation" : "runtime_or_input_allocation", span: logicalBytes };
  let span = null;
  if (!allocationInfo) {
    if (storage.constantBuffers.length) {
      if (dataBufferIndex >= storage.constantBuffers.length) throw new Error(`${planName}.values[${valueIndex}] references constant buffer ${dataBufferIndex} outside ${storage.constantBuffers.length}.`);
      span = BigInt(storage.constantBuffers[dataBufferIndex].byte_length);
    } else if (storage.constantSegment?.offsets.length) {
      span = subsegmentSpan(storage.constantSegment, dataBufferIndex, `${planName}.values[${valueIndex}] constant segment`);
    } else throw new Error(`${planName}.values[${valueIndex}] references constant data ${dataBufferIndex}, but no constant storage table is populated.`);
  } else {
    const mutableIndex = extraInfo?.mutable_data_segments_index == null ? 0 : toSafeIndex(BigInt(extraInfo.mutable_data_segments_index), `${planName}.values[${valueIndex}] mutable segment index`);
    if (mutableIndex >= storage.mutableSegments.length) throw new Error(`${planName}.values[${valueIndex}] references mutable data segment ${mutableIndex} outside ${storage.mutableSegments.length}.`);
    span = subsegmentSpan(storage.mutableSegments[mutableIndex], dataBufferIndex, `${planName}.values[${valueIndex}] mutable segment`);
  }
  if (logicalBytes != null && logicalBytes > span) throw new Error(`${planName}.values[${valueIndex}] logical payload ${logicalBytes} B exceeds serialized storage span ${span} B.`);
  return { status: logicalBytes == null ? "serialized_span_observed_logical_dynamic" : logicalBytes === span ? "observed_exact_serialized_payload" : "observed_serialized_span_with_padding", span };
}

function validatePlannedTensorAllocations(tensors, bufferSizes, planName) {
  for (const tensor of tensors) {
    const allocation = tensor.allocation_info;
    if (!allocation) continue;
    const memoryId = allocation.memory_id;
    if (!Number.isSafeInteger(memoryId) || memoryId <= 0 || memoryId >= bufferSizes.length) {
      throw new Error(`${planName} tensor ${tensor.evalue_index} references planned memory_id ${memoryId} outside 1..${Math.max(0, bufferSizes.length - 1)}.`);
    }
    const offset = BigInt(allocation.memory_offset_decimal);
    const span = tensor.buffer_data_length_decimal == null ? null : BigInt(tensor.buffer_data_length_decimal);
    if (span != null && offset + span > bufferSizes[memoryId]) {
      throw new Error(`${planName} tensor ${tensor.evalue_index} planned allocation [${offset}, ${offset + span}) exceeds memory buffer ${memoryId} size ${bufferSizes[memoryId]}.`);
    }
    tensor.planned_allocation_status = span == null
      ? "observed_offset_dynamic_unbound_span_not_assessed" : tensor.shape_dynamism === "DYNAMIC_BOUND"
        ? "validated_upper_bound_within_aot_buffer" : "validated_exact_static_span_within_aot_buffer";
    tensor.planned_memory_buffer_size_decimal = bufferSizes[memoryId].toString();
  }
}

function subsegmentSpan(subsegment, index, label) {
  if (index >= subsegment.offsets.length) throw new Error(`${label} index ${index} is outside ${subsegment.offsets.length} offsets.`);
  const start = BigInt(subsegment.offsets[index]);
  const end = index + 1 < subsegment.offsets.length ? BigInt(subsegment.offsets[index + 1]) : BigInt(subsegment.segment_size);
  if (end < start) throw new Error(`${label} has a negative span at index ${index}.`);
  return end - start;
}

function resolveExternalTensorData(requiredTensors, files) {
  const supplied = [];
  for (const file of files || []) {
    const bytes = file?.bytes instanceof Uint8Array ? file.bytes : null;
    if (!bytes || readIdentifier(bytes) !== "FT01") continue;
    const filename = file.name || file.path || "external.ptd";
    const analysis = analyzeFlatTensor(bytes, filename);
    supplied.push({ filename, analysis });
  }
  const owner = new Map();
  for (const item of supplied) {
    for (const tensor of item.analysis.tensors) {
      if (owner.has(tensor.name)) throw new Error(`External tensor ${tensor.name} is defined by more than one supplied PTD file.`);
      owner.set(tensor.name, { filename: item.filename, tensor });
    }
  }
  const required = [...requiredTensors].sort((left, right) => left.external_data_name.localeCompare(right.external_data_name));
  assertUnique(required.map((tensor) => tensor.external_data_name), "Program external tensor name");
  const requiredNames = required.map((tensor) => tensor.external_data_name);
  const missing = requiredNames.filter((name) => !owner.has(name));
  const contractRows = required.filter((tensor) => owner.has(tensor.external_data_name)).map((tensor) => {
    const suppliedTensor = owner.get(tensor.external_data_name).tensor;
    const comparison = compareExecuTorchExternalTensorContract(tensor, suppliedTensor);
    return { name: tensor.external_data_name, filename: owner.get(tensor.external_data_name).filename, ...comparison };
  });
  const mismatches = contractRows.filter((row) => row.status !== "matched");
  const matched = contractRows.filter((row) => row.status === "matched");
  const verifiedBytes = sumBigInt(matched.map((row) => BigInt(row.logical_bytes_decimal)));
  return {
    status: required.length === 0 ? "not_applicable" : missing.length ? "not_assessed_missing_ptd"
      : mismatches.length ? "invalid_external_tensor_contract" : "verified_complete",
    required_name_count: required.length,
    supplied_ptd_count: supplied.length,
    resolved_name_count: contractRows.length,
    verified_contract_count: matched.length,
    verified_logical_bytes_decimal: verifiedBytes.toString(),
    missing_names: missing,
    bindings: contractRows,
    contract_mismatches: mismatches.map((row) => ({ name: row.name, filename: row.filename, reasons: row.reasons })),
  };
}

export function compareExecuTorchExternalTensorContract(expected, supplied) {
  const reasons = [];
  if (!expected || !supplied) reasons.push("tensor_contract_missing");
  const expectedShape = Array.isArray(expected?.shape) ? expected.shape.map(Number) : null;
  const suppliedShape = Array.isArray(supplied?.shape) ? supplied.shape.map(Number) : null;
  if (expected?.shape_status !== "assessed" || !expectedShape || expectedShape.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    reasons.push("expected_tensor_shape_not_static");
  } else if (!supplied?.shape_declared || !suppliedShape || suppliedShape.length !== expectedShape.length
    || suppliedShape.some((value, index) => value !== expectedShape[index])) reasons.push("shape_mismatch");
  if (String(expected?.dtype || "") !== String(supplied?.dtype || "")) reasons.push("dtype_mismatch");
  const expectedBytes = exactDecimal(expected?.buffer_data_length_decimal);
  const suppliedBytes = exactDecimal(supplied?.buffer_data_length_decimal);
  if (expectedBytes == null || suppliedBytes == null || expectedBytes !== suppliedBytes) reasons.push("logical_byte_length_mismatch");
  if (supplied?.layout_status !== "assessed") reasons.push("supplied_ptd_tensor_layout_not_assessed");
  return {
    status: reasons.length ? "mismatch" : "matched",
    reasons,
    dtype: supplied?.dtype || null,
    shape: suppliedShape || [],
    logical_bytes_decimal: suppliedBytes == null ? null : suppliedBytes.toString(),
    serialized_span_bytes_decimal: exactDecimal(supplied?.serialized_storage_span_decimal)?.toString() ?? null,
  };
}

function parseExtendedHeader(bytes) {
  if (bytes.length < 12 || String.fromCharCode(...bytes.subarray(8, 12)) !== "eh00") {
    return { present: false, magic: "", header_length: 0, program_size: String(bytes.length), segment_base_offset: "0", segment_data_size: "0" };
  }
  if (bytes.length < 32) throw new Error("ExecuTorch extended header is truncated below 32 bytes.");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = view.getUint32(12, true);
  if (headerLength < 24) throw new Error(`ExecuTorch extended header length ${headerLength} is below 24 bytes.`);
  if (8 + headerLength > bytes.length) throw new Error(`ExecuTorch extended header length ${headerLength} exceeds file size.`);
  const programSize = view.getBigUint64(16, true);
  const segmentBase = view.getBigUint64(24, true);
  const segmentDataSize = headerLength >= 32 ? view.getBigUint64(32, true) : 0n;
  if (programSize < 8n || programSize > BigInt(bytes.length)) throw new Error(`ExecuTorch program_size ${programSize} is outside file size ${bytes.length}.`);
  if (segmentBase > 0n && (segmentBase < programSize || segmentBase > BigInt(bytes.length))) throw new Error(`ExecuTorch segment_base_offset ${segmentBase} is outside [program_size, file_size].`);
  if (segmentDataSize > 0n && segmentBase + segmentDataSize > BigInt(bytes.length)) throw new Error("ExecuTorch segment_data_size exceeds the file boundary.");
  return { present: true, magic: "eh00", header_length: headerLength, program_size: programSize.toString(), segment_base_offset: segmentBase.toString(), segment_data_size: segmentDataSize.toString() };
}

function boundedProgramBytes(bytes, header) {
  const size = toSafeIndex(BigInt(header.program_size), "ExecuTorch program_size");
  return bytes.subarray(0, size);
}

function readIdentifier(bytes) {
  if (!(bytes instanceof Uint8Array) || bytes.length < 8) return "";
  return String.fromCharCode(...bytes.subarray(4, 8));
}

function buildPlannedMemory(plans) {
  const perDeviceMap = new Map();
  let total = 0n;
  for (const plan of plans) {
    for (const buffer of plan.non_const_memory_buffers || []) {
      const value = BigInt(buffer.size_bytes_decimal);
      total += value;
      const key = `${buffer.device_type}:${buffer.device_index}`;
      perDeviceMap.set(key, (perDeviceMap.get(key) || 0n) + value);
    }
  }
  return {
    total,
    perDevice: [...perDeviceMap.entries()].map(([device, bytes]) => ({ device, bytes: safeNumber(bytes), bytes_decimal: bytes.toString() })),
  };
}

function buildInstructionPlannedLiveness(state) {
  const rows = [];
  let aggregatePeak = 0n;
  let complete = true;
  let bounded = false;
  for (const plan of state.plans) {
    const ops = state.ops.filter((op) => op.plan_index === plan.index);
    const tensors = state.tensors.filter((tensor) => tensor.plan_index === plan.index && tensor.allocation_info);
    let reason = null;
    if (plan.chain_count !== 1) reason = "multiple_execution_chains";
    else if (ops.some((op) => op.instruction_kind === "JumpFalseCall")) reason = "runtime_control_flow";
    else if (ops.some((op) => op.instruction_kind === "DelegateCall")) reason = "delegate_argument_direction_unbound";
    else if (ops.some((op) => op.instruction_kind === "KernelCall" && op.signature_status !== "source_bound")) reason = "kernel_argument_direction_unbound";
    else if (tensors.some((tensor) => tensor.buffer_data_length_decimal == null)) reason = "dynamic_unbound_allocation_span";
    if (reason) {
      complete = false;
      rows.push({ plan_index: plan.index, plan_name: plan.name, status: `not_assessed_${reason}`, peak_bytes: null, peak_bytes_decimal: null, instruction_count: ops.length });
      continue;
    }
    const planInputs = new Set(plan.input_value_indices || []);
    const planOutputs = new Set(plan.output_value_indices || []);
    const intervals = [];
    for (const tensor of tensors) {
      const producers = ops.flatMap((op, index) => (op.outputs || []).includes(tensor.index) ? [index] : []);
      const consumers = ops.flatMap((op, index) => (op.inputs || []).includes(tensor.index) ? [index] : []);
      const frees = ops.flatMap((op, index) => op.instruction_kind === "FreeCall" && (op.inputs || []).includes(tensor.index) ? [index] : []);
      const born = planInputs.has(tensor.evalue_index) ? -1 : producers.length ? Math.min(...producers) : null;
      const lastUse = consumers.length ? Math.max(...consumers) : null;
      const dies = frees.length ? Math.min(...frees) : planOutputs.has(tensor.evalue_index) ? ops.length : lastUse;
      if (born == null || dies == null || dies < born) { reason = "tensor_lifetime_not_reconstructable"; break; }
      bounded ||= tensor.shape_dynamism === "DYNAMIC_BOUND";
      intervals.push({
        tensor_index: tensor.index,
        memory_id: tensor.allocation_info.memory_id,
        start: BigInt(tensor.allocation_info.memory_offset_decimal),
        end: BigInt(tensor.allocation_info.memory_offset_decimal) + BigInt(tensor.buffer_data_length_decimal),
        born,
        dies,
      });
    }
    if (reason) {
      complete = false;
      rows.push({ plan_index: plan.index, plan_name: plan.name, status: `not_assessed_${reason}`, peak_bytes: null, peak_bytes_decimal: null, instruction_count: ops.length });
      continue;
    }
    let peak = 0n;
    let peakInstruction = ops.length ? 0 : -1;
    for (let instruction = -1; instruction <= ops.length; instruction += 1) {
      const active = intervals.filter((row) => row.born <= instruction && instruction <= row.dies);
      const bytes = unionAllocationBytes(active);
      if (bytes > peak) { peak = bytes; peakInstruction = instruction; }
    }
    if (peak > aggregatePeak) aggregatePeak = peak;
    rows.push({
      plan_index: plan.index,
      plan_name: plan.name,
      status: bounded ? "derived_exact_aot_upper_bound_address_liveness" : "derived_exact_aot_static_address_liveness",
      peak_bytes: safeNumber(peak),
      peak_bytes_decimal: peak.toString(),
      peak_instruction_index: peakInstruction,
      instruction_count: ops.length,
      planned_tensor_count: intervals.length,
    });
  }
  return {
    complete,
    peak: aggregatePeak,
    plans: rows,
    status: complete ? bounded ? "derived_exact_aot_upper_bound_address_liveness" : "derived_exact_aot_static_address_liveness" : "partial_aot_address_liveness",
    detail: complete
      ? "For each single straight-line execution chain with source-bound argument direction, active AOT allocation ranges are reconstructed from producer/consumer/FreeCall order and unioned by memory_id and offset. DYNAMIC_BOUND spans remain upper bounds. This is planned address occupancy, not runtime RSS, allocator-private memory, or observed physical liveness."
      : "ExecutionPlan.non_const_buffer_sizes remains exact serialized AOT planner evidence. Instruction-level planned address occupancy is emitted only for single straight-line chains with source-bound argument direction and bounded tensor spans; runtime RSS and allocator-private memory are never inferred.",
  };
}

function unionAllocationBytes(rows) {
  const byMemory = new Map();
  for (const row of rows) {
    if (!byMemory.has(row.memory_id)) byMemory.set(row.memory_id, []);
    byMemory.get(row.memory_id).push([row.start, row.end]);
  }
  let total = 0n;
  for (const intervals of byMemory.values()) {
    intervals.sort((left, right) => left[0] < right[0] ? -1 : left[0] > right[0] ? 1 : 0);
    let start = null;
    let end = null;
    for (const [nextStart, nextEnd] of intervals) {
      if (start == null) { start = nextStart; end = nextEnd; continue; }
      if (nextStart <= end) { if (nextEnd > end) end = nextEnd; }
      else { total += end - start; start = nextStart; end = nextEnd; }
    }
    if (start != null) total += end - start;
  }
  return total;
}

function buildMetadataPresence(container, version, plans) {
  return {
    format: "executorch",
    schema: "deepbom.artifact_metadata.v1.4",
    status: "assessed",
    has_model_metadata: true,
    metadata_entries: ["container", "schema_version", ...(plans.length ? ["execution_plan_names"] : [])],
    preprocessing_contract_status: "absent_no_standard_preprocessing_contract",
    output_semantics_documented: false,
    detail: `ExecuTorch ${container.toUpperCase()} version ${version}; ${plans.length} serialized execution plan(s). Preprocessing and task semantics are not standardized by this container.`,
  };
}

function buildProgramMarkdown(filename, version, state, plannedMemory, segmentBytes) {
  return `# ExecuTorch Static Audit: ${filename}\n\n- ET12 schema version: ${version}\n- Execution plans: ${state.plans.length}\n- Serialized instructions: ${state.ops.length}\n- Tensor EValues: ${state.tensors.length}\n- Backend delegates: ${state.delegates.length}\n- Planned non-constant memory: ${plannedMemory.total} B\n- Appended segment payload: ${segmentBytes} B\n\nInstruction order and EValue argument identity are observed. Operator argument direction, delegate-internal graphs, runtime allocation, kernels, and latency are not inferred.`;
}

function buildFlatTensorMarkdown(filename, version, tensors, segmentBytes) {
  return `# ExecuTorch FlatTensor Audit: ${filename}\n\n- FT01 schema version: ${version}\n- Named entries: ${tensors.length}\n- Appended segment payload: ${segmentBytes} B\n\nFT01 stores named tensor/blob data and does not serialize an execution graph.`;
}

function tensorRefs(refs, values) { return refs.map((index) => values[index]).filter((value) => value?.kind === "Tensor").map((value) => value.tensor_index); }
function validateIndices(values, length, label) { values.forEach((value, index) => { if (!Number.isInteger(value) || value < 0 || value >= length) throw new Error(`${label}[${index}] references ${value} outside [0, ${length}).`); }); }
function validateDimOrder(order, rank, label) { if (order.length !== rank || new Set(order).size !== rank || order.some((value) => value >= rank)) throw new Error(`${label} is not a permutation of rank ${rank}.`); }
function assertUnique(values, label) { const seen = new Set(); for (const value of values) { if (seen.has(value)) throw new Error(`${label} contains duplicate ${JSON.stringify(value)}.`); seen.add(value); } }
function combineU32(low, high) { return BigInt(low) | BigInt(high) << 32n; }
function productBigInt(values, label) { let total = 1n; for (const value of values) { if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} contains invalid dimension ${value}.`); total *= BigInt(value); } return total; }
function ceilBitsToBytes(elements, bits) { return (elements * BigInt(bits) + 7n) / 8n; }
function sumBigInt(values) { return values.reduce((sum, value) => sum + BigInt(value), 0n); }
function exactDecimal(value) {
  if (typeof value === "bigint") return value >= 0n ? value : null;
  const text = String(value ?? "");
  return /^(?:0|[1-9]\d*)$/.test(text) ? BigInt(text) : null;
}
function safeNumber(value) { return value == null || BigInt(value) > BigInt(Number.MAX_SAFE_INTEGER) ? null : Number(value); }
function toSafeIndex(value, label) { if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} ${value} exceeds the JavaScript address range.`); return Number(value); }
function hexPrefix(bytes) { return [...bytes.subarray(0, 32)].map((value) => value.toString(16).padStart(2, "0")).join(""); }
function countRows(items, keyFn) { const counts = new Map(); for (const item of items) { const key = keyFn(item); counts.set(key, (counts.get(key) || 0) + 1); } return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([name, count]) => ({ name, count })); }
