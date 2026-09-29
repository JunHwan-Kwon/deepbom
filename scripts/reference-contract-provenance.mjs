import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const REFERENCE_RULEPACK_BASIS_SHA256 = "c9f8e0d16dd918d4b48cf2fbb2636a8fcd98eb5492d4a76826ecca887cd84dc9";
if (sha256TextHex(JSON.stringify(ANALYZER_METADATA.rulepackHashBasis)) !== REFERENCE_RULEPACK_BASIS_SHA256) {
  throw new Error("Reference rulepack basis changed; regenerate from a clean release and rotate the pinned analyzer identity and reference documents together.");
}

// Pinned to the clean 2.0.0 source snapshot that generated the reference set.
// Rotate these fields and the documents together; never relabel old evidence.
export const REFERENCE_CONTRACT_ANALYZER_METADATA = Object.freeze({
  ...ANALYZER_METADATA,
  version: "2026-08-03",
  semanticVersion: "2.0.0",
  buildCommit: "3b0c62caeec91c6875f724310c2c7c2bc94733d4",
  buildSourceState: "clean",
  buildContentSha256: "529d2222dff3890467e4de1c76d1e8fe91034daf41a0d93674c7afd70e41d1aa",
  buildContentManifestSha256: "ea75bf3e2381aaa4facdee6122732d814488dbf01630d569c135e075553add8b",
  rulepackVersion: "deepbom.rulepack.2026-07-24.63",
  rulepackSha256: "771d28776486275b1eda24095fce6cdb307cdb9cbc2df449a9e765f2a5afa8de",
});
