import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

const SHA256 = /^[a-f0-9]{64}$/;

export const CYCLONEDX_DRAFT_PROFILE_REGISTRY = deepFreeze({
  schema: "deepbom.cyclonedx_draft_profile_registry.v1",
  captured_at: "2026-09-04",
  active_profile_id: "cyclonedx-2.0-ai-ml-integration-2026-09-04",
  profiles: {
    "cyclonedx-2.0-ai-ml-integration-2026-09-04": {
      specification_version: "2.0-draft",
      integration_status: "UNRESOLVED_INTEGRATION",
      export_allowed: false,
      evidence_class: "SOURCE_BACKED",
      sources: [
        source(990, "CycloneDX/specification", "f0834e1520ddf1a0641b1b3d7d74655293a6b2ff", [
          content("schema/2.0/cyclonedx-2.0.schema.json", "cb299dd805d9b2afbc88489ab70b9e33c67021c474f104da8ba8bf5553a292b2"),
          content("schema/2.0/model/cyclonedx-ai-ml-2.0.schema.json", "9b346b22f6115c08a6d63cbb04b0774b93aa769dec08015e303b3956363dfa53"),
        ], "typed_ai_ml_model"),
        source(175, "CycloneDX/cyclonedx-property-taxonomy", "f5f956dd7fd0c312f05520ce7aecc3746d099645", [
          content("cdx/ai-ml.md", "63fa274daee550a70439b5ec18007b456cd393d4a31e4d8e7ada31f330d298db"),
        ], "ai_ml_property_taxonomy"),
        source(1067, "CycloneDX/specification", "ebb5184e91a1dbcdb2009f4c5f19b1f439473058", [
          content("perspectives/model-card-perspective.json", "30b518a141d3b541d713aa581610798be372a8ae8e8f3f5c409853615f66f553"),
        ], "predefined_model_card_perspective"),
        source(1075, "CycloneDX/specification", "24b8d8ef60965aea9b50d7a3fce16ca5117b278b", [
          content("schema/2.0/model/cyclonedx-component-2.0.schema.json", "9f20417fabb485a20445fc1be5c45d705bd4f018acb562a0a7ddc9810f1c66ba"),
          content("schema/2.0/model/cyclonedx-definition-2.0.schema.json", "dbcf6f638b40db691a1ec1bb369036ba36c61ff3e15d39351d14e5bca478ff7f"),
          content("schema/2.0/model/cyclonedx-perspective-2.0.schema.json", "3092870f3cbc205d0906945691f7f33a5f6592600c4bb708afc3004da4807feb"),
          content("tools/src/test/resources/2.0/valid-perspective-2.0.json", "bfb2d8de6149d1d876d1cdc85795cf563399e6ce2008875a1960db90233d4e83"),
        ], "inventory_definitions_perspective_integration"),
      ],
      unresolved_conditions: [
        {
          reason_code: "TYPED_TAXONOMY_OWNERSHIP_UNRESOLVED",
          detail: "The typed AI/ML quantization object and the taxonomy affine properties do not yet have an integrated ownership rule.",
        },
        {
          reason_code: "PERSPECTIVE_INVENTORY_SCOPE_UNRESOLVED",
          detail: "The predefined model-card mappings and the inventories/definitions refactor have not been integrated into one settled processing model.",
        },
      ],
    },
    "legacy-parameter-contract-2026-08-06": {
      specification_version: "2.0-draft",
      integration_status: "LEGACY_PINNED_NON_CONFORMANT_FIXTURE",
      export_allowed: false,
      evidence_class: "SOURCE_BACKED",
      sources: [
        source(990, "CycloneDX/specification", "58a7cc2d04105e7525b0ed369ccf0a4325dc34b2", [], "historical_typed_ai_ml_model"),
        source(175, "CycloneDX/cyclonedx-property-taxonomy", "1b380dfae8bf4a83646ae59ea3d3b42d466f3858", [], "historical_ai_ml_property_taxonomy"),
      ],
      unresolved_conditions: [{
        reason_code: "SCHEMA_CLOSURE_UNAVAILABLE",
        detail: "This historical proposal fixture is retained for regression evidence only and is not a CycloneDX 2.0 conformance claim.",
      }],
    },
  },
});

export const ACTIVE_CYCLONEDX_DRAFT_PROFILE_ID = CYCLONEDX_DRAFT_PROFILE_REGISTRY.active_profile_id;

export function cycloneDxDraftProfile(profileId = ACTIVE_CYCLONEDX_DRAFT_PROFILE_ID) {
  const profile = CYCLONEDX_DRAFT_PROFILE_REGISTRY.profiles[profileId];
  if (!profile) throw new Error(`Unknown CycloneDX draft profile: ${profileId}`);
  validateProfile(profileId, profile);
  return profile;
}

export function buildCycloneDxDraftCompatibilityRecord(analysis = {}, options = {}) {
  const profileId = options.profileId || ACTIVE_CYCLONEDX_DRAFT_PROFILE_ID;
  const profile = cycloneDxDraftProfile(profileId);
  const artifactSha256 = normalizeSha256(options.hash || analysis.model_sha256);
  const artifact = {
    name: String(analysis.filename || "model"),
    format: String(analysis.format || "unknown").toLowerCase(),
    sha256: artifactSha256 || null,
  };
  const registrySha256 = sha256TextHex(canonicalJson(CYCLONEDX_DRAFT_PROFILE_REGISTRY));
  return {
    schema: "deepbom.cyclonedx_draft_compatibility.v1",
    generated_at: options.generatedAt || new Date().toISOString(),
    evidence_class: "SOURCE_BACKED",
    artifact,
    profile_id: profileId,
    profile_registry_sha256: registrySha256,
    specification_version: profile.specification_version,
    integration_status: profile.integration_status,
    export_allowed: profile.export_allowed,
    decision: profile.export_allowed ? "EXPORT_ALLOWED" : "EXPORT_REFUSED",
    sources: profile.sources,
    unresolved_conditions: profile.unresolved_conditions,
    stable_export: {
      specification_version: "1.7",
      status: "AVAILABLE",
      recommendation: "Use the validated CycloneDX 1.7 artifact evidence document while the 2.0 drafts remain unresolved.",
    },
    claim_boundary: "This record identifies pinned draft sources and their integration status. It is not a CycloneDX 2.0 BOM or a conformance claim.",
  };
}

export function assertCycloneDxDraftExportAllowed(profileId = ACTIVE_CYCLONEDX_DRAFT_PROFILE_ID) {
  const profile = cycloneDxDraftProfile(profileId);
  if (!profile.export_allowed) {
    const error = new Error(`CycloneDX 2.0 draft export refused: ${profile.integration_status}. Use the stable CycloneDX 1.7 export or inspect the draft compatibility record.`);
    error.code = profile.integration_status === "UNRESOLVED_INTEGRATION"
      ? "UNRESOLVED_INTEGRATION"
      : "STALE_DRAFT_PROFILE";
    error.profile_id = profileId;
    throw error;
  }
  return profile;
}

function source(pullRequest, repository, commit, contents, role) {
  return {
    repository,
    pull_request: `https://github.com/${repository}/pull/${pullRequest}`,
    commit,
    commit_url: `https://github.com/${repository}/commit/${commit}`,
    role,
    contents,
  };
}

function content(path, sha256) {
  return { path, sha256 };
}

function validateProfile(profileId, profile) {
  if (!profile || typeof profile !== "object") throw new Error(`CycloneDX draft profile ${profileId} is invalid.`);
  if (typeof profile.export_allowed !== "boolean") throw new Error(`CycloneDX draft profile ${profileId} lacks an export decision.`);
  for (const sourceEntry of profile.sources || []) {
    if (!/^[a-f0-9]{40}$/.test(sourceEntry.commit)) throw new Error(`CycloneDX draft profile ${profileId} has an invalid commit pin.`);
    for (const row of sourceEntry.contents || []) {
      if (!row.path || !SHA256.test(row.sha256)) throw new Error(`CycloneDX draft profile ${profileId} has an invalid content pin.`);
    }
  }
}

function normalizeSha256(value) {
  const digest = String(value || "").trim().toLowerCase();
  return SHA256.test(digest) ? digest : "";
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}
