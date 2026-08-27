import { sha256TextHex } from "./sha256-sync.js";

export function cycloneDxSerialNumber({ artifactSha256 = "", generatedAt = "", profile = "document" } = {}) {
  const digest = sha256TextHex(`${normalizeSha256(artifactSha256) || "unbound"}\n${String(generatedAt)}\n${String(profile)}`);
  return `urn:uuid:${uuidFromDigest(digest)}`;
}

export function artifactBomRef(artifactSha256 = "", fallbackName = "model") {
  const digest = normalizeSha256(artifactSha256);
  return digest ? `deepbom-model-sha256-${digest}` : `deepbom-model-${safeToken(fallbackName)}`;
}

export function analyzerBomRef(name, semanticVersion, buildIdentity = "") {
  return `deepbom-analyzer-${safeToken(name)}-${safeToken(semanticVersion)}${buildIdentity ? `-${safeToken(buildIdentity)}` : ""}`;
}

export function analyzerContentVersion(semanticVersion, buildCommit = "", bundleSha256 = "") {
  const base = String(semanticVersion || "unversioned").trim();
  const commit = String(buildCommit || "").trim().toLowerCase();
  const bundle = normalizeSha256(bundleSha256);
  const identifiers = [];
  if (/^[a-f0-9]{7,64}$/.test(commit)) identifiers.push(`git.${commit.slice(0, 12)}`);
  if (bundle) identifiers.push(`bundle.${bundle.slice(0, 12)}`);
  if (!identifiers.length) return base;
  return `${base}${base.includes("+") ? "." : "+"}${identifiers.join(".")}`;
}

export function artifactContentVersion(artifactSha256 = "", declaredVersion = "") {
  const declared = String(declaredVersion || "").trim();
  if (declared && !/^(unknown|unbound|not[_ -]?declared|n\/a)$/i.test(declared)) {
    return { version: declared, basis: "artifact_embedded_or_supplied_version" };
  }
  const digest = normalizeSha256(artifactSha256);
  return digest ? { version: `sha256-${digest}`, basis: "full_artifact_sha256" } : { version: null, basis: "not_available" };
}

function uuidFromDigest(value) {
  const hex = String(value).slice(0, 32).split("");
  hex[12] = "8";
  hex[16] = ((Number.parseInt(hex[16], 16) & 0x3) | 0x8).toString(16);
  const joined = hex.join("");
  return `${joined.slice(0, 8)}-${joined.slice(8, 12)}-${joined.slice(12, 16)}-${joined.slice(16, 20)}-${joined.slice(20)}`;
}

function normalizeSha256(value) {
  const digest = String(value || "").trim().toLowerCase();
  return /^[a-f0-9]{64}$/.test(digest) ? digest : "";
}

function safeToken(value) {
  return encodeURIComponent(String(value || "unbound").trim().toLowerCase()).replaceAll("%", "_");
}
