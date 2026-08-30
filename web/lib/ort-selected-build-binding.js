import { assessOrtReducedOperatorConfig } from "./ort-reduced-operator-config.js";

export const ORT_SELECTED_BUILD_BINDING_SCHEMA = "deepbom.ort_selected_build_provider_binding.v1.2";

const BACKEND_TO_SOURCE_PROFILE = Object.freeze({
  cpu: "wasm_cpu",
  cuda: "cuda",
  dml: "directml",
  webgpu: "webgpu",
  webnn: "webnn",
  qnn: "qnn",
  coreml: "coreml",
  nnapi: "nnapi",
  xnnpack: "xnnpack",
});

export function buildOrtSelectedBuildProviderBinding(analysis, runtime) {
  const compatibility = analysis?.ort_compatibility_evidence;
  const providers = compatibility?.execution_providers || [];
  const protectedRulepackAvailable = compatibility != null;
  if (protectedRulepackAvailable && (compatibility.schema !== "deepbom.ort_source_compatibility.v1.15"
    || compatibility.source_commit !== runtime?.source_commit || !Array.isArray(providers) || !providers.length)) {
    throw new Error("Selected ORT build cannot be cross-referenced against mismatched protected source compatibility evidence.");
  }
  const inventory = compatibility?.source_condition_inventory?.execution_providers || [];
  const backends = runtime.supported_backends || [];
  const bindings = backends.map((backend) => {
    const sourceProfile = protectedRulepackAvailable ? BACKEND_TO_SOURCE_PROFILE[backend.name] || null : null;
    const source = sourceProfile ? providers.find((provider) => provider.execution_provider === sourceProfile) : null;
    const counts = sourceProfile ? inventory.find((row) => row.execution_provider === sourceProfile) : null;
    return {
      backend_name: backend.name,
      bundled: backend.bundled,
      source_profile: sourceProfile,
      source_profile_rule_count: counts?.source_rule_count ?? null,
      binding_status: source
        ? backend.bundled ? "BUNDLED_BACKEND_WITH_PINNED_SOURCE_PROFILE" : "DISCOVERED_BACKEND_WITH_PINNED_SOURCE_PROFILE_NOT_BUNDLED"
        : protectedRulepackAvailable
          ? backend.bundled ? "BUNDLED_BACKEND_WITHOUT_STATIC_SOURCE_PROFILE" : "DISCOVERED_BACKEND_WITHOUT_STATIC_SOURCE_PROFILE"
          : "PROTECTED_SOURCE_RULEPACK_NOT_LOADED",
    };
  });
  const boundProfiles = new Set(bindings.map((row) => row.source_profile).filter(Boolean));
  const reducedConfig = runtime.reduced_operator_config || null;
  const reducedAssessment = reducedConfig
    ? assessOrtReducedOperatorConfig(analysis, reducedConfig.normalized_config) : null;
  return {
    schema: ORT_SELECTED_BUILD_BINDING_SCHEMA,
    evidence_class: "OBSERVED_BUILD_INVENTORY_PLUS_SOURCE_PROFILE_CROSS_REFERENCE",
    runtime_source_commit: runtime.source_commit,
    rulepack_source_commit: compatibility?.source_commit || null,
    source_commit_match: protectedRulepackAvailable ? true : null,
    supported_backends_sha256: runtime.supported_backends_sha256,
    provider_inventory_status: runtime.provider_inventory_status,
    reduced_operator_inventory_status: runtime.reduced_operator_inventory_status,
    reduced_operator_config_identity: reducedConfig ? {
      schema: reducedConfig.schema,
      source_name: reducedConfig.source_name,
      source_sha256: reducedConfig.source_sha256,
      normalized_sha256: reducedConfig.normalized_sha256,
      binary_binding_status: reducedConfig.binary_binding_status,
    } : null,
    source_build_attestation: runtime.build_attestation ? {
      schema: runtime.build_attestation.schema,
      attestation_sha256: runtime.build_attestation.attestation_sha256,
      distribution_identity: runtime.distribution_identity,
      package_manifest_sha256: runtime.build_attestation.runtime_package.package_manifest_sha256,
      binary_inventory_sha256: runtime.build_attestation.runtime_package.binary_inventory_sha256,
      primary_binary_sha256: runtime.build_attestation.runtime_package.primary_binary_sha256,
      reduced_operator_config_sha256: runtime.build_attestation.reduced_operator_config?.source_sha256 || null,
    } : null,
    reduced_operator_assessment: reducedAssessment,
    bindings,
    source_profiles_not_listed_by_selected_build: providers
      .map((provider) => provider.execution_provider)
      .filter((profile) => !boundProfiles.has(profile)),
    interpretation_boundary: reducedConfig
      ? runtime.build_attestation
        ? "A listed backend is a selected-package observation. The exact reduced-operator config is an attested input to the selected source build and is compared with every serialized graph scope. This establishes configured operator inclusion intent for that build, not GetCapability acceptance, optimized assignment, successful execution, or correctness."
        : "A listed backend is a selected-package observation. The imported reduced-operator config is compared with every serialized graph scope, but is not proof of selected-binary inclusion because no build attestation binds the config digest to that binary. GetCapability and execution remain separate ledgers."
      : "A listed backend is a selected-package build observation, not proof that any operator is compiled into a reduced build, accepted by GetCapability, assigned, or executed. Source-profile candidates and runtime assignment remain separate ledgers.",
  };
}

export function validateOrtSelectedBuildProviderBinding(binding, analysis, runtime) {
  const expected = buildOrtSelectedBuildProviderBinding(analysis, runtime);
  if (JSON.stringify(binding) !== JSON.stringify(expected)) throw new Error("Selected ORT build/provider binding does not reproduce from the protected rulepack and native runtime inventory.");
  return true;
}
