import { createCheck } from "./check-assert.mjs";
import {
  ACTIVE_CYCLONEDX_DRAFT_PROFILE_ID,
  assertCycloneDxDraftExportAllowed,
  buildCycloneDxDraftCompatibilityRecord,
  cycloneDxDraftProfile,
  CYCLONEDX_DRAFT_PROFILE_REGISTRY,
} from "../web/lib/cyclonedx-draft-profiles.js";
import {
  buildCycloneDx20ParameterContractPreview,
  validateCycloneDx20ParameterContractPreview,
} from "../web/lib/cyclonedx-20-preview.js";

const { done, expect, expectEqual } = createCheck("CycloneDX draft profile check");
const active = cycloneDxDraftProfile();
expectEqual(ACTIVE_CYCLONEDX_DRAFT_PROFILE_ID, "cyclonedx-2.0-ai-ml-integration-2026-09-04", "active profile identity");
expectEqual(active.integration_status, "UNRESOLVED_INTEGRATION", "current integration remains unresolved");
expectEqual(active.export_allowed, false, "unresolved draft export fails closed");
expectEqual(active.sources.length, 4, "all interdependent PR sources are pinned");
expect(active.sources.every((source) => /^[a-f0-9]{40}$/.test(source.commit)), "every source has an exact commit");
expect(active.sources.flatMap((source) => source.contents).every((row) => /^[a-f0-9]{64}$/.test(row.sha256)), "every material source has a SHA-256");
expectEqual(Object.isFrozen(CYCLONEDX_DRAFT_PROFILE_REGISTRY), true, "registry is immutable");

let refused = null;
try { assertCycloneDxDraftExportAllowed(); } catch (error) { refused = error; }
expectEqual(refused?.code, "UNRESOLVED_INTEGRATION", "current export refusal reason");

const analysis = {
  filename: "interface.tflite",
  format: "tflite",
  model_sha256: "a".repeat(64),
  inputs: [{ index: 0, name: "image", dtype: "UINT8", shape: [1, 1], scale_sample: [0.5], zero_point_sample: [128] }],
  outputs: [],
};
const status = buildCycloneDxDraftCompatibilityRecord(analysis, { generatedAt: "2026-09-04T00:00:00.000Z" });
expectEqual(status.schema, "deepbom.cyclonedx_draft_compatibility.v1", "status schema");
expectEqual(status.decision, "EXPORT_REFUSED", "status decision");
expectEqual(status.stable_export.status, "AVAILABLE", "stable 1.7 fallback");
expect(!Object.hasOwn(status, "specFormat"), "status record cannot be mistaken for a CycloneDX BOM");

let implicitLegacy = null;
try { buildCycloneDx20ParameterContractPreview(analysis); } catch (error) { implicitLegacy = error; }
expectEqual(implicitLegacy?.code, "STALE_DRAFT_PROFILE", "historical fixture requires explicit opt-in");
const legacy = buildCycloneDx20ParameterContractPreview(analysis, {
  generatedAt: "2026-08-06T00:00:00.000Z",
  profileId: "legacy-parameter-contract-2026-08-06",
  allowHistoricalFixture: true,
});
const validation = validateCycloneDx20ParameterContractPreview(legacy);
expect(validation.valid, validation.errors.join("; "));

done("CycloneDX draft profiles passed (four-source pins, immutable registry, fail-closed current export, and explicit historical fixture boundary).");
