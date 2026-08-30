import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const REFERENCE_RULEPACK_BASIS_SHA256 = "551296e4c38fc9b69a99467de0c4075ec9fe1a2b6bf8598b50f6ccaf34262c05";
const POST_REFERENCE_RULE_ENTRIES = new Set([
  "web/lib/onnx-ort-matmul-shape-inference.js (ONNX rule/reconstruction)",
  "web/lib/onnx-ort-recurrent-shape-inference.js (ONNX rule/reconstruction)",
  "web/lib/onnx-ort-transformer-shape-inference.js (ONNX rule/reconstruction)",
]);

function referenceRulepackHashBasis() {
  const current = Array.isArray(ANALYZER_METADATA.rulepackHashBasis)
    ? ANALYZER_METADATA.rulepackHashBasis
    : [];
  const targetProfileEntry = "src/target_profiles.rs (target profiles and target-specific planning contracts)";
  const planningRuleEntry = "src/lib.rs (quantization classification and XNNPACK planning rules)";
  if (!current.includes(targetProfileEntry) || !current.includes(planningRuleEntry)) {
    throw new Error("Reference rulepack basis migration requires an explicit clean-release provenance rotation.");
  }
  const referenceBasis = current.flatMap((entry) => {
    if (entry === targetProfileEntry) return ["src/lib.rs (target profiles and XNNPACK support rules)"];
    if (entry === planningRuleEntry) return [];
    if (POST_REFERENCE_RULE_ENTRIES.has(entry)) return [];
    return [entry];
  });
  if (sha256TextHex(JSON.stringify(referenceBasis)) !== REFERENCE_RULEPACK_BASIS_SHA256) {
    throw new Error("Reference rulepack basis changed; regenerate from a clean release and rotate the pinned analyzer identity and reference documents together.");
  }
  return Object.freeze(referenceBasis);
}

// These fields identify the clean analyzer snapshot that generated the
// checked-in reference documents. Regenerate the documents from a clean tree
// and rotate this pin together with them; do not silently inherit the current
// working tree's identity.
export const REFERENCE_CONTRACT_ANALYZER_METADATA = Object.freeze({
  ...ANALYZER_METADATA,
  version: "2026-08-03",
  semanticVersion: "1.95.0",
  buildCommit: "077ceb7533cdaafe9389634f3c7d36ede346b2dd",
  buildSourceState: "clean",
  buildContentSha256: "7fba3e04cd19fe4a5eab2c8bb4d639fd1b9e72e39d89bff6f4fe3b07b96299cc",
  buildContentManifestSha256: "3c5f08161b2271e2f3818dbe0d693fa4bc2e58917a39cb6432c7a01d8e110c28",
  rulepackVersion: "deepbom.rulepack.2026-07-24.63",
  rulepackSha256: "0b74cf5f8157f2aa14c0b26098ff36427c043bc7fc14f517aac9115ac52ae46a",
  rulepackHashBasis: referenceRulepackHashBasis(),
});
