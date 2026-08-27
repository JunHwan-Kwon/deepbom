import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const SIGNING_KEY_REGISTRY_SCHEMA = "deepbom.signing_key_registry.v1";
export const OFFICIAL_SIGNING_KEY_REGISTRY_URL = "https://deepbom.org/.well-known/deepbom-signing-keys.json";
export const OFFICIAL_RELEASE_AUTHORITY_DOI = "https://doi.org/10.5281/zenodo.21834509";
export const SIGNER_CLASSES = Object.freeze({
  local: "LOCAL_BROWSER_KEY",
  official: "OFFICIAL_RELEASE_KEY",
});

const KEY_STATUSES = new Set(["active", "retired", "revoked"]);
const OFFICIAL_SCOPES = new Set(["official_release_evidence", "official_verified_example"]);

function parseTime(value) {
  if (typeof value !== "string" || !value) return null;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : null;
}

function publicJwkIdentity(jwk) {
  if (!jwk || jwk.kty !== "EC" || jwk.crv !== "P-256" || typeof jwk.x !== "string" || typeof jwk.y !== "string") return null;
  return canonicalJson({ crv: jwk.crv, kty: jwk.kty, x: jwk.x, y: jwk.y });
}

function publicJwkThumbprint(jwk) {
  const identity = publicJwkIdentity(jwk);
  return identity ? sha256TextHex(identity) : null;
}

export function validateSigningKeyRegistry(registry) {
  const errors = [];
  if (registry?.schema !== SIGNING_KEY_REGISTRY_SCHEMA) errors.push("registry_schema_invalid");
  if (!Number.isSafeInteger(registry?.registry_version) || registry.registry_version < 1) errors.push("registry_version_invalid");
  if (registry?.authority?.canonical_origin !== "https://deepbom.org") errors.push("authority_origin_invalid");
  if (registry?.authority?.registry_url !== OFFICIAL_SIGNING_KEY_REGISTRY_URL) errors.push("authority_registry_url_invalid");
  if (registry?.authority?.zenodo_doi !== OFFICIAL_RELEASE_AUTHORITY_DOI) errors.push("authority_doi_invalid");
  if (parseTime(registry?.updated_at) == null) errors.push("registry_updated_at_invalid");
  if (!Array.isArray(registry?.keys) || !registry.keys.length) errors.push("registry_keys_empty");
  const ids = new Set();
  const thumbprints = new Set();
  for (const [index, key] of (registry?.keys || []).entries()) {
    const prefix = `keys/${index}`;
    if (typeof key?.key_id !== "string" || !key.key_id) errors.push(`${prefix}/key_id_invalid`);
    else if (ids.has(key.key_id)) errors.push(`${prefix}/key_id_duplicate`);
    else ids.add(key.key_id);
    if (key?.signer_class !== SIGNER_CLASSES.official) errors.push(`${prefix}/signer_class_invalid`);
    if (key?.algorithm !== "ES256" || key?.curve !== "P-256") errors.push(`${prefix}/algorithm_invalid`);
    if (!KEY_STATUSES.has(key?.status)) errors.push(`${prefix}/status_invalid`);
    const validFrom = parseTime(key?.valid_from);
    const validUntil = key?.valid_until == null ? null : parseTime(key.valid_until);
    if (validFrom == null) errors.push(`${prefix}/valid_from_invalid`);
    if (key?.valid_until != null && validUntil == null) errors.push(`${prefix}/valid_until_invalid`);
    if (validFrom != null && validUntil != null && validUntil < validFrom) errors.push(`${prefix}/validity_interval_invalid`);
    if (key?.revoked_at != null && parseTime(key.revoked_at) == null) errors.push(`${prefix}/revoked_at_invalid`);
    if (key?.status === "revoked" && parseTime(key?.revoked_at) == null) errors.push(`${prefix}/revocation_time_missing`);
    if (key?.status === "retired" && parseTime(key?.valid_until) == null) errors.push(`${prefix}/retired_valid_until_missing`);
    if (!Array.isArray(key?.allowed_scopes) || !key.allowed_scopes.length || key.allowed_scopes.some((scope) => !OFFICIAL_SCOPES.has(scope))) errors.push(`${prefix}/allowed_scopes_invalid`);
    if (!publicJwkIdentity(key?.public_jwk)) errors.push(`${prefix}/public_jwk_invalid`);
    if (!/^[a-f0-9]{64}$/.test(String(key?.rfc7638_thumbprint_sha256 || ""))) errors.push(`${prefix}/thumbprint_invalid`);
    else if (publicJwkThumbprint(key.public_jwk) !== key.rfc7638_thumbprint_sha256) errors.push(`${prefix}/thumbprint_jwk_mismatch`);
    else if (thumbprints.has(key.rfc7638_thumbprint_sha256)) errors.push(`${prefix}/thumbprint_duplicate`);
    else thumbprints.add(key.rfc7638_thumbprint_sha256);
  }
  return { valid: errors.length === 0, errors };
}

export function resolveSignatureTrust(record, registry, { registryUrl = OFFICIAL_SIGNING_KEY_REGISTRY_URL } = {}) {
  const signer = record?.signed_payload?.signer || null;
  if (signer?.class !== SIGNER_CLASSES.official) return {
    trusted: false,
    signer_class: SIGNER_CLASSES.local,
    status: "local_key_cryptographic_integrity_only",
    key_id: signer?.key_id || null,
    registry_url: null,
  };
  if (registryUrl !== OFFICIAL_SIGNING_KEY_REGISTRY_URL) return {
    trusted: false,
    signer_class: signer.class,
    status: "untrusted_registry_location",
    key_id: signer.key_id || null,
    registry_url: registryUrl,
  };
  if (signer?.authority_registry_url !== OFFICIAL_SIGNING_KEY_REGISTRY_URL) return {
    trusted: false,
    signer_class: signer.class,
    status: "signature_registry_binding_invalid",
    key_id: signer.key_id || null,
    registry_url: registryUrl,
  };
  const validation = validateSigningKeyRegistry(registry);
  if (!validation.valid) return {
    trusted: false,
    signer_class: signer.class,
    status: "registry_invalid",
    key_id: signer.key_id || null,
    registry_url: registryUrl,
    errors: validation.errors,
  };
  const key = registry.keys.find((candidate) => candidate.key_id === signer.key_id);
  if (!key) return { trusted: false, signer_class: signer.class, status: "key_not_registered", key_id: signer.key_id || null, registry_url: registryUrl };
  if (key.status === "revoked") return { trusted: false, signer_class: signer.class, status: "key_revoked", key_id: key.key_id, registry_url: registryUrl };
  if (record?.public_key?.rfc7638_thumbprint_sha256 !== key.rfc7638_thumbprint_sha256
    || publicJwkIdentity(record?.public_key?.jwk) !== publicJwkIdentity(key.public_jwk)) {
    return { trusted: false, signer_class: signer.class, status: "registered_key_mismatch", key_id: key.key_id, registry_url: registryUrl };
  }
  if (!key.allowed_scopes.includes(record?.signed_payload?.scope)) return { trusted: false, signer_class: signer.class, status: "scope_not_authorized", key_id: key.key_id, registry_url: registryUrl };
  const issuedAt = parseTime(signer.issued_at);
  const validFrom = parseTime(key.valid_from);
  const validUntil = key.valid_until == null ? null : parseTime(key.valid_until);
  if (issuedAt == null) return { trusted: false, signer_class: signer.class, status: "signature_time_invalid", key_id: key.key_id, registry_url: registryUrl };
  if (issuedAt < validFrom || (validUntil != null && issuedAt > validUntil)) return { trusted: false, signer_class: signer.class, status: "signature_outside_key_validity", key_id: key.key_id, registry_url: registryUrl };
  return {
    trusted: true,
    signer_class: signer.class,
    status: key.status === "retired" ? "trusted_historical_official_release_key" : "trusted_official_release_key",
    key_id: key.key_id,
    registry_url: registryUrl,
    zenodo_doi: registry.authority.zenodo_doi,
  };
}
