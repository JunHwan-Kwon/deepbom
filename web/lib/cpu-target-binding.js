const SHA256 = /^[a-f0-9]{64}$/;
const SOURCES = new Set(["default_assumption", "explicit_id", "profile_file"]);

export function buildCpuCostTargetBinding(targetProfile, { bindingSource, sourceInput = null } = {}) {
  const profileId = String(targetProfile?.id || "").trim();
  const profileSha256 = String(targetProfile?.profile_sha256 || "").trim().toLowerCase();
  if (!profileId) throw new Error("CPU cost target profile id is required.");
  if (!SHA256.test(profileSha256)) throw new Error("CPU cost target profile SHA-256 is required.");
  if (!SOURCES.has(bindingSource)) throw new Error("CPU cost target binding source is invalid.");
  const binding = {
    schema: "deepbom.cpu_cost_target_binding.v1",
    profile_id: profileId,
    profile_sha256: profileSha256,
    binding_source: bindingSource,
    host_observed: false,
    source_input: bindingSource === "profile_file" ? normalizeSourceInput(sourceInput) : null,
  };
  validateCpuCostTargetBinding(binding);
  return binding;
}

export function validateCpuCostTargetBinding(binding) {
  if (binding?.schema !== "deepbom.cpu_cost_target_binding.v1") throw new Error("CPU cost target binding schema is invalid.");
  if (!String(binding.profile_id || "").trim()) throw new Error("CPU cost target binding profile id is missing.");
  if (!SHA256.test(String(binding.profile_sha256 || ""))) throw new Error("CPU cost target binding profile SHA-256 is invalid.");
  if (!SOURCES.has(binding.binding_source)) throw new Error("CPU cost target binding source is invalid.");
  if (binding.host_observed !== false) throw new Error("CPU cost target profiles are planning inputs and cannot claim host observation.");
  if (binding.binding_source === "profile_file") normalizeSourceInput(binding.source_input);
  else if (binding.source_input != null) throw new Error("Built-in CPU target bindings cannot carry a profile-file source input.");
  return binding;
}

function normalizeSourceInput(input) {
  const value = {
    filename: String(input?.filename || "").trim(),
    byte_length: Number(input?.byte_length),
    source_sha256: String(input?.source_sha256 || "").toLowerCase(),
    normalized_profile_sha256: String(input?.normalized_profile_sha256 || "").toLowerCase(),
    duplicate_key_validation: String(input?.duplicate_key_validation || ""),
  };
  if (!value.filename || !Number.isSafeInteger(value.byte_length) || value.byte_length <= 0
      || !SHA256.test(value.source_sha256) || !SHA256.test(value.normalized_profile_sha256)
      || value.duplicate_key_validation !== "complete") {
    throw new Error("CPU cost target profile-file source input is invalid.");
  }
  return value;
}
