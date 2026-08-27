import { webcrypto } from "node:crypto";
import { readFile } from "node:fs/promises";

import { createCheck } from "./check-assert.mjs";
import { buildCanonicalPackageDigest, zipTextFile } from "../web/lib/report-utils.js";
import { signPackageDigest, verifyPackageSignature, verifyPackageSignatureWithTrust } from "../web/lib/public-key-signature.js";
import { OFFICIAL_SIGNING_KEY_REGISTRY_URL, SIGNER_CLASSES, resolveSignatureTrust, validateSigningKeyRegistry } from "../web/lib/signing-key-trust.js";

const check = createCheck("public-key-package-signature");
const files = [
  zipTextFile("a.json", "{\"a\":1}"),
  zipTextFile("nested/b.txt", "evidence"),
];
const digest = await buildCanonicalPackageDigest(files);
const signature = await signPackageDigest(digest, { scope: "test-package", cryptoProvider: webcrypto });

check.expectEqual(signature.schema, "deepbom.detached_package_signature.v2", "signature schema");
check.expectEqual(signature.algorithm, "ES256", "signature algorithm");
check.expectEqual(signature.signed_payload.signer.class, SIGNER_CLASSES.local, "local signer class");
check.expect(/^[a-f0-9]{64}$/.test(signature.public_key.rfc7638_thumbprint_sha256), "public-key thumbprint");
check.expect(await verifyPackageSignature(signature, digest, webcrypto), "signature verifies with included public key");

const tampered = structuredClone(digest);
tampered.package_hash_sha256 = "0".repeat(64);
check.expect(!(await verifyPackageSignature(signature, tampered, webcrypto)), "tampered digest is rejected");

const tamperedSignature = structuredClone(signature);
tamperedSignature.signature = `${tamperedSignature.signature.startsWith("A") ? "B" : "A"}${tamperedSignature.signature.slice(1)}`;
check.expect(!(await verifyPackageSignature(tamperedSignature, digest, webcrypto)), "tampered signature is rejected");

const checkedInRegistry = JSON.parse(await readFile(new URL("../web/.well-known/deepbom-signing-keys.json", import.meta.url), "utf8"));
const registryValidation = validateSigningKeyRegistry(checkedInRegistry);
check.expect(registryValidation.valid, `checked-in official signing registry is valid: ${registryValidation.errors.join(", ")}`);
const invalidIntervalRegistry = structuredClone(checkedInRegistry);
invalidIntervalRegistry.keys[0].valid_until = "2026-08-14T23:59:59Z";
check.expect(!validateSigningKeyRegistry(invalidIntervalRegistry).valid, "reversed key-validity interval is rejected");
const invalidAuthorityRegistry = structuredClone(checkedInRegistry);
invalidAuthorityRegistry.authority.zenodo_doi = "https://doi.org/10.5281/zenodo.0";
check.expect(!validateSigningKeyRegistry(invalidAuthorityRegistry).valid, "unbound authority DOI is rejected");
const localTrust = await verifyPackageSignatureWithTrust(signature, digest, checkedInRegistry, { cryptoProvider: webcrypto });
check.expect(localTrust.cryptographically_valid && !localTrust.trusted && localTrust.signer_class === SIGNER_CLASSES.local, "local signature remains cryptographically valid without official-authority promotion");

const officialKeys = await webcrypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, true, ["sign", "verify"]);
const officialPublicJwk = await webcrypto.subtle.exportKey("jwk", officialKeys.publicKey);
const officialProbe = await signPackageDigest(digest, {
  scope: "official_release_evidence",
  keyPair: { ...officialKeys, publicJwk: officialPublicJwk },
  signerClass: SIGNER_CLASSES.official,
  keyId: "rotation-test-v2",
  issuedAt: "2026-08-15T12:00:00Z",
  authorityRegistryUrl: OFFICIAL_SIGNING_KEY_REGISTRY_URL,
  cryptoProvider: webcrypto,
});
const rotationRegistry = structuredClone(checkedInRegistry);
rotationRegistry.keys = [{
  key_id: "rotation-test-v2",
  signer_class: SIGNER_CLASSES.official,
  algorithm: "ES256",
  curve: "P-256",
  status: "active",
  valid_from: "2026-08-01T00:00:00Z",
  valid_until: null,
  revoked_at: null,
  allowed_scopes: ["official_release_evidence"],
  public_jwk: officialPublicJwk,
  rfc7638_thumbprint_sha256: officialProbe.public_key.rfc7638_thumbprint_sha256,
}];
check.expect(validateSigningKeyRegistry(rotationRegistry).valid, "rotated official registry validates");
const officialTrust = await verifyPackageSignatureWithTrust(officialProbe, digest, rotationRegistry, { cryptoProvider: webcrypto });
check.expect(officialTrust.cryptographically_valid && officialTrust.trusted && officialTrust.status === "trusted_official_release_key", "active registered official key is trusted");

const retiredRegistry = structuredClone(rotationRegistry);
retiredRegistry.keys[0].status = "retired";
retiredRegistry.keys[0].valid_until = "2026-08-31T23:59:59Z";
const retiredTrust = await verifyPackageSignatureWithTrust(officialProbe, digest, retiredRegistry, { cryptoProvider: webcrypto });
check.expect(retiredTrust.trusted && retiredTrust.status === "trusted_historical_official_release_key", "historical signature remains trusted after orderly key rotation");

const revokedRegistry = structuredClone(rotationRegistry);
revokedRegistry.keys[0].status = "revoked";
revokedRegistry.keys[0].revoked_at = "2026-08-16T00:00:00Z";
const revokedTrust = await verifyPackageSignatureWithTrust(officialProbe, digest, revokedRegistry, { cryptoProvider: webcrypto });
check.expect(!revokedTrust.trusted && revokedTrust.status === "key_revoked", "revoked key is rejected");

const wrongScope = structuredClone(rotationRegistry);
wrongScope.keys[0].allowed_scopes = ["official_verified_example"];
const scopeTrust = await verifyPackageSignatureWithTrust(officialProbe, digest, wrongScope, { cryptoProvider: webcrypto });
check.expect(!scopeTrust.trusted && scopeTrust.status === "scope_not_authorized", "registry scope mismatch is rejected");

const wrongOriginTrust = await verifyPackageSignatureWithTrust(officialProbe, digest, rotationRegistry, {
  cryptoProvider: webcrypto,
  registryUrl: "https://example.invalid/deepbom-signing-keys.json",
});
check.expect(!wrongOriginTrust.trusted && wrongOriginTrust.status === "untrusted_registry_location", "look-alike registry origin is rejected");

const wrongRecordRegistryBinding = structuredClone(officialProbe);
wrongRecordRegistryBinding.signed_payload.signer.authority_registry_url = "https://example.invalid/deepbom-signing-keys.json";
const wrongRecordResolution = resolveSignatureTrust(wrongRecordRegistryBinding, rotationRegistry);
check.expect(!wrongRecordResolution.trusted && wrongRecordResolution.status === "signature_registry_binding_invalid", "signed record must bind the fixed registry URL");
const wrongRecordTrust = await verifyPackageSignatureWithTrust(wrongRecordRegistryBinding, digest, rotationRegistry, { cryptoProvider: webcrypto });
check.expect(!wrongRecordTrust.cryptographically_valid && wrongRecordTrust.status === "signature_invalid", "tampered signed registry binding is rejected cryptographically");

check.done("Public-key package signature passed (ES256 generation, detached verification, and tamper rejection).");
