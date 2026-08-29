const BAZEL_DEFINE = /^--define\s+([^=\s;]+)=([^\s;]+)$/;

const SATISFIED_TFLITE_DELEGATE_BINDINGS = new Set([
  "declared_present",
  "observed_runtime_capability_present",
]);

export function bindTfliteBuildRequirement(runtimeEvidence, requirement) {
  if (!runtimeEvidence) {
    return { status: "pending_runtime_evidence_not_imported", evidence_class: "NOT_ASSESSED", value: null };
  }
  const token = String(requirement || "").split(";")[0].trim();
  if (!token) {
    return { status: "pending_required_configuration_not_declared", evidence_class: "NOT_ASSESSED", value: null };
  }

  const expectedDefine = parseBazelDefine(token);
  const definitions = runtimeEvidence?.selector_context?.build?.compile_definitions;
  if (expectedDefine && Array.isArray(definitions)) {
    const declared = definitions.find((row) => String(row?.name || "") === expectedDefine.name);
    if (declared) {
      const actual = String(declared.value ?? "").trim();
      return actual === expectedDefine.value
        ? { status: "declared_present", evidence_class: "DECLARED_RUNTIME_BUILD", value: token }
        : { status: "contradiction_required_configuration_value", evidence_class: "DECLARED_RUNTIME_BUILD", value: `--define ${expectedDefine.name}=${actual}` };
    }
  }

  const build = String(runtimeEvidence?.runtime?.build || "").trim();
  if (!build) {
    return { status: "pending_imported_runtime_build_not_declared", evidence_class: "NOT_ASSESSED", value: null };
  }
  if (expectedDefine) {
    const observed = parseBazelDefines(build).filter((row) => row.name === expectedDefine.name);
    if (observed.some((row) => row.value === expectedDefine.value)) {
      return { status: "declared_present", evidence_class: "DECLARED_RUNTIME_BUILD", value: token };
    }
    if (observed.length) {
      return { status: "contradiction_required_configuration_value", evidence_class: "DECLARED_RUNTIME_BUILD", value: observed.map((row) => `--define ${row.name}=${row.value}`).join("; ") };
    }
  } else if (containsToken(build, token)) {
    return { status: "declared_present", evidence_class: "DECLARED_RUNTIME_BUILD", value: token };
  }
  return { status: "contradiction_required_configuration_absent", evidence_class: "DECLARED_RUNTIME_BUILD", value: build };
}

export function tfliteBuildRequirementsBound(runtimeEvidence, requirements) {
  const rows = Array.isArray(requirements) ? requirements : [];
  return rows.length > 0 && rows.every((row) => bindTfliteBuildRequirement(
    runtimeEvidence,
    row?.required_build_configuration,
  ).status === "declared_present");
}

export function bindTfliteDelegateRequirement(runtimeEvidence, requirement) {
  if (Number(requirement?.affected_source_candidate_op_count || 0) === 0) {
    return binding("not_applicable_no_affected_source_candidates", "NOT_APPLICABLE", null);
  }
  if (!runtimeEvidence) {
    return binding("pending_runtime_evidence_not_imported", "NOT_ASSESSED", null);
  }
  const inventory = runtimeEvidence.tflite_delegate_build_inventory;
  if (!inventory) {
    return binding("pending_delegate_build_inventory_not_imported", "NOT_ASSESSED", null);
  }

  switch (requirement?.id) {
    case "tflite_gpu_delegate_compiled":
      if (inventory.gpu?.compiled_status === "enabled_by_declared_cmake_option") {
        return binding(
          "partial_gpu_delegate_compiled_target_backend_unobserved",
          inventory.evidence_class,
          inventory.gpu.compiled_status,
        );
      }
      if (String(inventory.gpu?.compiled_status || "").startsWith("disabled_")) {
        return binding("contradiction_gpu_delegate_not_compiled", inventory.evidence_class, inventory.gpu.compiled_status);
      }
      return binding("pending_gpu_build_option_not_declared", "NOT_ASSESSED", inventory.gpu?.compiled_status ?? null);
    case "tflite_gpu_allow_quant_ops":
      if (inventory.gpu?.quantized_model_flag_status === "enabled_by_declared_runtime_option") {
        return binding("declared_present", inventory.evidence_class, inventory.gpu.experimental_flags);
      }
      if (inventory.gpu?.quantized_model_flag_status === "disabled_by_declared_runtime_option") {
        return binding("contradiction_quantized_model_flag_disabled", inventory.evidence_class, inventory.gpu.experimental_flags);
      }
      return binding("pending_gpu_runtime_option_not_declared", "NOT_ASSESSED", null);
    case "tflite_nnapi_delegate_compiled":
      if (inventory.nnapi?.compiled_status === "enabled_by_declared_cmake_option_and_android_gate") {
        return binding("declared_present", inventory.evidence_class, inventory.nnapi.compiled_status);
      }
      if (String(inventory.nnapi?.compiled_status || "").startsWith("disabled_")) {
        return binding("contradiction_nnapi_delegate_not_compiled", inventory.evidence_class, inventory.nnapi.compiled_status);
      }
      return binding("pending_nnapi_build_gate_unresolved", "NOT_ASSESSED", inventory.nnapi?.compiled_status ?? null);
    case "tflite_nnapi_feature_level": {
      const featureLevel = inventory.nnapi?.runtime_feature_level;
      const accelerator = String(inventory.nnapi?.accelerator_identity || "").trim();
      if (featureLevel == null || !accelerator) {
        return binding("pending_nnapi_runtime_capability_query", "NOT_ASSESSED", null);
      }
      const observed = inventory.nnapi.capability_source === "android_nnapi_runtime_query";
      return binding(
        observed ? "observed_runtime_capability_present" : "partial_declared_runtime_capability_not_observed",
        observed ? "OBSERVED_RUNTIME_CAPABILITY" : inventory.evidence_class,
        `${featureLevel}; ${accelerator}`,
      );
    }
    default:
      return binding("unsupported_delegate_requirement", "NOT_ASSESSED", requirement?.id ?? null);
  }
}

export function summarizeTfliteDelegateBuildBinding(runtimeEvidence, requirements) {
  const rows = (Array.isArray(requirements) ? requirements : []).map((requirement) => ({
    id: requirement?.id ?? null,
    profile: requirement?.profile ?? null,
    affected_source_candidate_op_count: Number(requirement?.affected_source_candidate_op_count || 0),
    binding: bindTfliteDelegateRequirement(runtimeEvidence, requirement),
  }));
  const applicable = rows.filter((row) => row.binding.status !== "not_applicable_no_affected_source_candidates");
  return {
    rows,
    applicable_requirement_count: applicable.length,
    satisfied_requirement_count: applicable.filter((row) => isTfliteDelegateBindingSatisfied(row.binding)).length,
    contradiction_count: applicable.filter((row) => row.binding.status.startsWith("contradiction_")).length,
    pending_or_partial_count: applicable.filter((row) => !isTfliteDelegateBindingSatisfied(row.binding)
      && !row.binding.status.startsWith("contradiction_")).length,
    all_applicable_requirements_bound: applicable.length > 0
      && applicable.every((row) => isTfliteDelegateBindingSatisfied(row.binding)),
  };
}

export function isTfliteDelegateBindingSatisfied(value) {
  return SATISFIED_TFLITE_DELEGATE_BINDINGS.has(String(value?.status || ""));
}

function parseBazelDefine(value) {
  const match = BAZEL_DEFINE.exec(String(value || "").trim());
  return match ? { name: match[1], value: match[2] } : null;
}

function parseBazelDefines(value) {
  const rows = [];
  const pattern = /(?:^|[\s;])--define\s+([^=\s;]+)=([^\s;]+)/g;
  for (const match of String(value || "").matchAll(pattern)) rows.push({ name: match[1], value: match[2] });
  return rows;
}

function containsToken(value, token) {
  return String(value || "").split(/[\s;,]+/).includes(String(token || ""));
}

function binding(status, evidenceClass, value) {
  return { status, evidence_class: evidenceClass || "NOT_ASSESSED", value };
}
