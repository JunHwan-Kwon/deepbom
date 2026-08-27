// Custom target profiles: take a built-in profile as a base, retune its
// planning numbers, and keep the result as a first-class target.
//
// The analyzer accepts a JSON specification anywhere a target id is accepted,
// so a custom target only needs an id the UI can carry (`custom:*`) plus a
// resolver that swaps that id for its specification at the analyzer boundary.
// Every value is validated here as well as in the analyzer, so the editor can
// report a problem before an audit is started.

const STORAGE_KEY = "deepbom.custom_target_profiles.v1";
export const CUSTOM_TARGET_PREFIX = "custom:";
export const EVIDENCE_CLASSES = Object.freeze(["MEASURED", "VENDOR_DECLARED", "USER_DECLARED"]);
const MAX_CUSTOM_TARGETS = 32;

/// Field metadata drives the editor, so a new tunable field needs no new UI.
/// `group` orders the form; `hint` states what the number means for the model.
export const TUNABLE_FIELDS = Object.freeze([
  { key: "core_count_min", label: "System core count (minimum variant)", unit: "cores", kind: "integer", min: 1, max: 1024, group: "compute",
    hint: "Use the same minimum and maximum for one exact device. A product family range produces only its documented endpoint variants." },
  { key: "core_count_max", label: "System core count (maximum variant)", unit: "cores", kind: "integer", min: 1, max: 1024, group: "compute",
    hint: "Must not be below the minimum. Static isolation scenarios are suppressed until both endpoints are bound." },
  { key: "performance_reference_core_count", label: "Peak-throughput reference", unit: "cores", kind: "integer", min: 1, max: 1024, group: "compute",
    hint: "Exact number of homogeneous cores represented by Peak throughput. This denominator is required before per-core scaling is calculated." },
  { key: "effective_peak_gops", label: "Peak throughput", unit: "GOPS", kind: "number", min: 0.1, max: 1e6, group: "compute",
    hint: "Theoretical peak. The bound divides by this times the utilization." },
  { key: "compute_utilization_factor", label: "Compute utilization", unit: "fraction", kind: "number", min: 0.01, max: 1, group: "compute",
    hint: "Fraction of peak a kernel actually reaches. 1.0 makes the estimate a best-case bound, not a latency." },
  { key: "effective_memory_bandwidth_gbps", label: "Memory bandwidth", unit: "GB/s", kind: "number", min: 0.25, max: 1e4, group: "memory" },
  { key: "weight_packing_bandwidth_gbps", label: "Weight packing bandwidth", unit: "GB/s", kind: "number", min: 0.1, max: 1e4, group: "memory" },
  { key: "l1_data_bytes", label: "L1 data cache", unit: "bytes", kind: "integer", min: 1024, max: 1 << 24, group: "memory" },
  { key: "l2_bytes", label: "L2 cache", unit: "bytes", kind: "integer", min: 4096, max: 1 << 30, group: "memory" },
  { key: "int8_speedup_estimate", label: "INT8 speedup", unit: "x", kind: "number", min: 0.1, max: 64, group: "compute" },
  { key: "fp32_compute_factor", label: "FP32 compute factor", unit: "x", kind: "number", min: 0.1, max: 64, group: "compute" },
  { key: "simd_width_bits", label: "SIMD width", unit: "bits", kind: "integer", min: 32, max: 2048, group: "isa" },
  { key: "fp32_lanes", label: "FP32 lanes", unit: "", kind: "integer", min: 1, max: 256, group: "isa" },
  { key: "fp16_lanes", label: "FP16 lanes", unit: "", kind: "integer", min: 1, max: 512, group: "isa" },
  { key: "int8_lanes", label: "INT8 lanes", unit: "", kind: "integer", min: 1, max: 1024, group: "isa" },
  { key: "channel_alignment_multiple", label: "Channel alignment", unit: "", kind: "integer", min: 1, max: 256, group: "isa" },
  { key: "in_order", label: "In-order core", unit: "", kind: "boolean", group: "isa" },
  { key: "dot_product", label: "Dot-product instructions", unit: "", kind: "boolean", group: "isa" },
  { key: "sve2", label: "SVE2", unit: "", kind: "boolean", group: "isa" },
  { key: "chain_break_overhead_us_low", label: "Predicted partition-break overhead (low)", unit: "us", kind: "number", min: 0, max: 1e5, group: "delegate" },
  { key: "chain_break_overhead_us_high", label: "Predicted partition-break overhead (high)", unit: "us", kind: "number", min: 0, max: 1e5, group: "delegate" },
  { key: "xnnpack_kernel_family", label: "XNNPACK kernel family", unit: "", kind: "text", maxLength: 64, group: "delegate" },
  { key: "architecture", label: "Architecture", unit: "", kind: "text", maxLength: 128, group: "identity" },
  { key: "l2_capacity_scope", label: "L2 capacity scope", unit: "", kind: "text", maxLength: 128, group: "identity" },
  { key: "cache_assumption", label: "Cache assumption", unit: "", kind: "text", maxLength: 1024, group: "identity" },
  { key: "cache_source_url", label: "Cache source URL", unit: "", kind: "text", maxLength: 512, group: "identity" },
]);

const FIELD_BY_KEY = new Map(TUNABLE_FIELDS.map((field) => [field.key, field]));

export function isCustomTargetId(id) {
  return String(id || "").startsWith(CUSTOM_TARGET_PREFIX);
}

export function loadCustomTargets(storage = safeStorage()) {
  if (!storage) return [];
  let parsed;
  try { parsed = JSON.parse(storage.getItem(STORAGE_KEY) || "[]"); }
  catch { return []; }
  if (!Array.isArray(parsed)) return [];
  return parsed.filter((entry) => entry && typeof entry === "object" && isCustomTargetId(entry.id)).slice(0, MAX_CUSTOM_TARGETS);
}

export function saveCustomTarget(spec, storage = safeStorage(), baseProfile = null) {
  const validated = validateCustomTargetSpec(spec, baseProfile);
  if (!storage) return [validated];
  const existing = loadCustomTargets(storage).filter((entry) => entry.id !== validated.id);
  if (existing.length >= MAX_CUSTOM_TARGETS) {
    throw new Error(`At most ${MAX_CUSTOM_TARGETS} custom targets can be stored`);
  }
  const next = [...existing, validated].sort((a, b) => a.id.localeCompare(b.id));
  storage.setItem(STORAGE_KEY, JSON.stringify(next));
  return next;
}

export function deleteCustomTarget(id, storage = safeStorage()) {
  if (!storage) return [];
  const next = loadCustomTargets(storage).filter((entry) => entry.id !== id);
  storage.setItem(STORAGE_KEY, JSON.stringify(next));
  return next;
}

/// The analyzer boundary: a built-in id passes through, a custom id becomes the
/// specification the analyzer parses. An unknown custom id is an error rather
/// than a silent fall back to a built-in, which would analyze the wrong target.
export function resolveTargetSpec(id, customTargets = loadCustomTargets()) {
  if (!isCustomTargetId(id)) return String(id || "");
  const found = customTargets.find((entry) => entry.id === id);
  if (!found) throw new Error(`Custom target ${id} is not available in this browser`);
  return JSON.stringify(found);
}

/// A profile-shaped object so a custom target can sit in the same list as the
/// built-ins for selection and labelling before any analysis has run.
export function customTargetStub(spec, baseProfile) {
  return {
    ...(baseProfile || {}),
    ...(spec.overrides || {}),
    id: spec.id,
    label: spec.label,
    profile_sha256: "",
    performance_model_evidence_class: spec.evidence_class,
    derived_from: {
      base_profile_id: spec.base,
      base_profile_sha256: baseProfile?.profile_sha256 || "",
      overridden_fields: Object.keys(spec.overrides || {}).sort(),
      evidence_note: spec.evidence_note || "",
    },
  };
}

export function validateCustomTargetSpec(spec, baseProfile = null) {
  if (!spec || typeof spec !== "object") throw new Error("Custom target must be an object");
  const base = String(spec.base || "").trim();
  if (!base || base.startsWith("{")) throw new Error("Custom target requires a built-in base profile");
  const id = String(spec.id || "").trim();
  if (!isCustomTargetId(id) || id.length <= CUSTOM_TARGET_PREFIX.length) {
    throw new Error(`Custom target id must start with '${CUSTOM_TARGET_PREFIX}'`);
  }
  if (id.length > 128 || ![...id].every((character) => {
    const code = character.codePointAt(0);
    return code >= 0x20 && code <= 0x7e;
  })) {
    throw new Error("Custom target id must be at most 128 printable ASCII characters");
  }
  const label = String(spec.label || "").trim();
  if (!label || label.length > 128) throw new Error("Custom target requires a label of 1-128 characters");
  const evidenceClass = String(spec.evidence_class || "USER_DECLARED");
  if (!EVIDENCE_CLASSES.includes(evidenceClass)) {
    throw new Error(`Evidence class must be one of: ${EVIDENCE_CLASSES.join(", ")}`);
  }
  const evidenceNote = String(spec.evidence_note || "").trim();
  if (evidenceNote.length > 1024) throw new Error("Evidence note exceeds 1024 characters");
  if (evidenceClass === "MEASURED" && !evidenceNote) {
    throw new Error("A MEASURED custom target requires a note naming the measurement");
  }

  const overrides = {};
  for (const [key, value] of Object.entries(spec.overrides || {})) {
    if (key === "compute_utilization_by_kernel_class") {
      overrides[key] = validateKernelClassMap(value);
      continue;
    }
    const field = FIELD_BY_KEY.get(key);
    if (!field) throw new Error(`'${key}' is not a tunable field`);
    overrides[key] = validateField(field, value);
  }
  const resolvedLow = overrides.chain_break_overhead_us_low
    ?? baseProfile?.chain_break_overhead_us_low;
  const resolvedHigh = overrides.chain_break_overhead_us_high
    ?? baseProfile?.chain_break_overhead_us_high;
  if (Number.isFinite(Number(resolvedLow)) && Number.isFinite(Number(resolvedHigh))
    && Number(resolvedHigh) < Number(resolvedLow)) {
    throw new Error("Predicted partition-break overhead high must not be below low");
  }
  const resolvedCoreMin = overrides.core_count_min ?? baseProfile?.core_count_min;
  const resolvedCoreMax = overrides.core_count_max ?? baseProfile?.core_count_max;
  if (Number.isInteger(Number(resolvedCoreMin)) && Number.isInteger(Number(resolvedCoreMax))
    && Number(resolvedCoreMax) < Number(resolvedCoreMin)) {
    throw new Error("System core-count maximum must not be below the minimum");
  }
  const resolvedReferenceCores = overrides.performance_reference_core_count ?? baseProfile?.performance_reference_core_count;
  if (Number(resolvedReferenceCores) > 0 && Number(resolvedCoreMin) > 0 && Number(resolvedCoreMax) > 0
    && ![Number(resolvedCoreMin), Number(resolvedCoreMax)].includes(Number(resolvedReferenceCores))) {
    throw new Error("Peak-throughput reference cores must match one declared system core-count variant");
  }
  return { base, id, label, evidence_class: evidenceClass, evidence_note: evidenceNote, overrides };
}

function validateField(field, value) {
  if (field.kind === "boolean") {
    if (typeof value !== "boolean") throw new Error(`${field.label} must be true or false`);
    return value;
  }
  if (field.kind === "text") {
    const text = String(value ?? "");
    if (text.length > field.maxLength) throw new Error(`${field.label} exceeds ${field.maxLength} characters`);
    return text;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${field.label} must be a number`);
  if (field.kind === "integer" && !Number.isInteger(parsed)) {
    throw new Error(`${field.label} must be a whole number`);
  }
  if (parsed < field.min || parsed > field.max) {
    throw new Error(`${field.label} must be between ${field.min} and ${field.max}`);
  }
  return parsed;
}

function validateKernelClassMap(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Per-kernel utilization must be an object of family to fraction");
  }
  const entries = Object.entries(value);
  if (entries.length > 64) throw new Error("At most 64 kernel families can be tuned");
  const resolved = {};
  for (const [family, raw] of entries) {
    if (!/^[a-z0-9_]{1,64}$/.test(family)) {
      throw new Error(`Kernel family '${family}' must be lower_snake_case of at most 64 characters`);
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0.01 || parsed > 1) {
      throw new Error(`Utilization for '${family}' must be between 0.01 and 1`);
    }
    resolved[family] = parsed;
  }
  return resolved;
}

/// Only fields that actually differ from the base are recorded, so a profile
/// never claims to have retuned a number it left alone.
export function overridesFromForm(baseProfile, values) {
  const overrides = {};
  for (const field of TUNABLE_FIELDS) {
    if (!(field.key in values)) continue;
    if (field.kind !== "boolean" && field.kind !== "text" && String(values[field.key] ?? "").trim() === "") continue;
    const next = field.kind === "boolean" ? Boolean(values[field.key])
      : field.kind === "text" ? String(values[field.key] ?? "")
      : Number(values[field.key]);
    const current = baseProfile?.[field.key];
    if (field.kind === "text" ? String(current ?? "") === next : current === next) continue;
    overrides[field.key] = validateField(field, next);
  }
  if (values.compute_utilization_by_kernel_class
    && Object.keys(values.compute_utilization_by_kernel_class).length) {
    overrides.compute_utilization_by_kernel_class =
      validateKernelClassMap(values.compute_utilization_by_kernel_class);
  }
  return overrides;
}

function safeStorage() {
  try { return typeof localStorage === "undefined" ? null : localStorage; }
  catch { return null; }
}
