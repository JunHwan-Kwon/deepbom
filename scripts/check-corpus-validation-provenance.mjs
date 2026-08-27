import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { createCheck } from "./check-assert.mjs";
import {
  INTERFACE_CORPUS_VALIDATION,
  interfaceCorpusValidationExternalReference,
  interfaceCorpusValidationProperties,
} from "../web/lib/corpus-validation-provenance.js";

const check = createCheck("corpus-validation-provenance");
const profilePath = "web/reference/quant-policy-boundary-validation.v1.json";
const profileBytes = await readFile(profilePath);
const profile = JSON.parse(profileBytes);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

check.expectEqual(sha256(profileBytes), INTERFACE_CORPUS_VALIDATION.profileSha256, "deployed validation profile SHA-256");
check.expectEqual(sha256(await readFile("corpus/quant_policy/manifest.v1.json")), profile.source.manifest_sha256, "corpus manifest SHA-256");
check.expectEqual(sha256(await readFile("scripts/verify-interface-boundary-corpus.mjs")), profile.verifier.implementation_sha256, "verifier implementation SHA-256");
check.expectEqual(profile.results.public_artifact_count, 50, "artifact denominator");
check.expectEqual(profile.results.external_parameter_count, 114, "external-parameter denominator");
check.expectEqual(profile.results.complete_affine_parameter_count + profile.results.explicitly_unquantized_parameter_count, 114, "parameter-state conservation");
check.expectEqual(profile.results.invalid_or_incomplete_parameter_count, 0, "invalid parameter count");

const reference = interfaceCorpusValidationExternalReference();
check.expectEqual(reference.hashes[0].content, INTERFACE_CORPUS_VALIDATION.profileSha256, "CycloneDX reference hash");
check.expectEqual(reference.url, INTERFACE_CORPUS_VALIDATION.profileUrl, "CycloneDX reference URL");
const properties = new Map(interfaceCorpusValidationProperties());
check.expectEqual(properties.get("deepbom:validation:interfaceCorpusReviewSha256"), profile.source.review_sha256, "review digest property");
check.expectEqual(Number(properties.get("deepbom:validation:interfaceCorpusArtifactCount")), 50, "artifact count property");
check.expect(profile.interpretation_boundary.includes("not a random estimate") && profile.interpretation_boundary.includes("does not validate the currently analyzed artifact"), "interpretation boundary");

check.done("Corpus validation provenance passed (profile, manifest, verifier, result, and export binding hashes).");
