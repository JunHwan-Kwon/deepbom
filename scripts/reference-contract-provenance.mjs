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
  semanticVersion: "1.94.0",
  buildCommit: "684260dddf632240c31bee646581d9c4ae32623a",
  buildSourceState: "clean",
  buildContentSha256: "5c03ec024b99e9a73ac3b3f098cebdd8dc18ae5ec67d740de97dd21df4e5b34d",
  buildContentManifestSha256: "d8c13a2a8b44844a88d90bd651f48ae97ab7e5bdcc6109746afd803116dd6ac6",
  rulepackVersion: "deepbom.rulepack.2026-07-24.63",
  rulepackSha256: "a1e0dc58a9dc742b4532448190410d89e04d56ab134a16a72cda6cae8b6a6d9f",
  rulepackHashBasis: referenceRulepackHashBasis(),
});
