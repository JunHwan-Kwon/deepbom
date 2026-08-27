import {
  buildCanonicalPackageDigest,
  canonicalJson,
  jsonForDownload,
  zipTextFile,
} from "./report-utils.js";
import { OFFICIAL_SIGNING_KEY_REGISTRY_URL, SIGNER_CLASSES, resolveSignatureTrust } from "./signing-key-trust.js";

export const PUBLIC_KEY_SIGNATURE_SCHEMA = "deepbom.detached_package_signature.v2";
export const LEGACY_PUBLIC_KEY_SIGNATURE_SCHEMA = "deepbom.detached_package_signature.v1";
const DB_NAME = "deepbom-local-signing-keys";
const STORE_NAME = "keys";
const LOCAL_KEY_SLOT = "deepbom-local-browser-p256-v1";
let volatileKeyPair = null;

export async function appendPublicKeySignature(files, scope) {
  if (!files?.length) return files;
  const packageDigest = await buildCanonicalPackageDigest(files);
  const signature = await signPackageDigest(packageDigest, { scope });
  if (!(await verifyPackageSignature(signature, packageDigest))) throw new Error("Local public-key signature verification failed");
  files.push(zipTextFile("deepbom_public_key_signature.json", jsonForDownload(signature)));
  return files;
}

function bytesToBase64Url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  const base64 = typeof btoa === "function" ? btoa(binary) : Buffer.from(bytes).toString("base64");
  return base64.replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

function base64UrlToBytes(value) {
  const normalized = String(value).replaceAll("-", "+").replaceAll("_", "/");
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
  const binary = typeof atob === "function" ? atob(padded) : Buffer.from(padded, "base64").toString("binary");
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

function hex(bytes) {
  return [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function assertDigest(digest) {
  if (!digest || !/^[a-f0-9]{64}$/.test(String(digest.package_hash_sha256 || ""))) throw new Error("Package digest SHA-256 is invalid");
  if (!Array.isArray(digest.files) || !digest.files.length) throw new Error("Package digest member ledger is empty");
}

function signaturePayload(digest, scope, signer = null) {
  return {
    schema: signer ? "deepbom.detached_package_signature_payload.v2" : "deepbom.detached_package_signature_payload.v1",
    scope: String(scope || "artifact_package"),
    ...(signer ? { signer } : {}),
    package_hash_method: digest.package_hash_method,
    package_hash_sha256: digest.package_hash_sha256,
    canonicalization: digest.canonicalization,
    files: digest.files,
  };
}

async function sha256Hex(provider, value) {
  return hex(await provider.subtle.digest("SHA-256", new TextEncoder().encode(value)));
}

async function publicJwkThumbprint(provider, jwk) {
  const payload = canonicalJson({ crv: jwk.crv, kty: jwk.kty, x: jwk.x, y: jwk.y });
  return sha256Hex(provider, payload);
}

async function generatedKeyPair(provider) {
  const generated = await provider.subtle.generateKey(
    { name: "ECDSA", namedCurve: "P-256" },
    true,
    ["sign", "verify"],
  );
  const [privateJwk, publicJwk] = await Promise.all([
    provider.subtle.exportKey("jwk", generated.privateKey),
    provider.subtle.exportKey("jwk", generated.publicKey),
  ]);
  const [privateKey, publicKey] = await Promise.all([
    provider.subtle.importKey("jwk", privateJwk, { name: "ECDSA", namedCurve: "P-256" }, false, ["sign"]),
    provider.subtle.importKey("jwk", publicJwk, { name: "ECDSA", namedCurve: "P-256" }, true, ["verify"]),
  ]);
  return { privateKey, publicKey, publicJwk };
}

function openKeyDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE_NAME);
    request.onerror = () => reject(request.error || new Error("Signing-key database failed to open"));
    request.onsuccess = () => resolve(request.result);
  });
}

async function storedKeyPair() {
  if (typeof indexedDB === "undefined") return null;
  const database = await openKeyDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const request = database.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(LOCAL_KEY_SLOT);
      request.onerror = () => reject(request.error || new Error("Signing key lookup failed"));
      request.onsuccess = () => resolve(request.result || null);
    });
  } finally {
    database.close();
  }
}

async function persistKeyPair(keyPair) {
  if (typeof indexedDB === "undefined") return;
  const database = await openKeyDatabase();
  try {
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).put(keyPair, LOCAL_KEY_SLOT);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error || new Error("Signing key persistence failed"));
      transaction.onabort = () => reject(transaction.error || new Error("Signing key persistence aborted"));
    });
  } finally {
    database.close();
  }
}

export async function getOrCreateLocalSigningKey(cryptoProvider = globalThis.crypto) {
  if (!cryptoProvider?.subtle) throw new Error("WebCrypto is required for public-key package signing");
  if (volatileKeyPair) return volatileKeyPair;
  try {
    const stored = await storedKeyPair();
    if (stored?.privateKey && stored?.publicKey && stored?.publicJwk) return (volatileKeyPair = stored);
  } catch (error) {
    console.warn("[public-key-signature] persisted key unavailable; using an in-memory key", error);
  }
  volatileKeyPair = await generatedKeyPair(cryptoProvider);
  try {
    await persistKeyPair(volatileKeyPair);
  } catch (error) {
    console.warn("[public-key-signature] key persistence failed; the key is session-scoped", error);
  }
  return volatileKeyPair;
}

export async function signPackageDigest(digest, {
  scope = "artifact_package",
  keyPair = null,
  signerClass = SIGNER_CLASSES.local,
  keyId = null,
  issuedAt = new Date().toISOString(),
  authorityRegistryUrl = null,
  cryptoProvider = globalThis.crypto,
} = {}) {
  assertDigest(digest);
  if (![SIGNER_CLASSES.local, SIGNER_CLASSES.official].includes(signerClass)) throw new Error("Package signer class is invalid");
  if (signerClass === SIGNER_CLASSES.official && (!keyPair || !keyId || authorityRegistryUrl !== OFFICIAL_SIGNING_KEY_REGISTRY_URL)) {
    throw new Error("Official package signing requires an explicit key pair, registered key ID, and fixed authority registry URL");
  }
  const keys = keyPair || await getOrCreateLocalSigningKey(cryptoProvider);
  const publicJwk = keys.publicJwk || await cryptoProvider.subtle.exportKey("jwk", keys.publicKey);
  const thumbprint = await publicJwkThumbprint(cryptoProvider, publicJwk);
  const signer = {
    class: signerClass,
    key_id: keyId || `local-browser:${thumbprint.slice(0, 24)}`,
    issued_at: issuedAt,
    authority_registry_url: signerClass === SIGNER_CLASSES.official ? authorityRegistryUrl : null,
  };
  const payload = signaturePayload(digest, scope, signer);
  const canonicalPayload = canonicalJson(payload);
  const signature = await cryptoProvider.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    keys.privateKey,
    new TextEncoder().encode(canonicalPayload),
  );
  return {
    schema: PUBLIC_KEY_SIGNATURE_SCHEMA,
    algorithm: "ES256",
    curve: "P-256",
    signature_encoding: "IEEE-P1363",
    signed_payload: payload,
    signed_payload_sha256: await sha256Hex(cryptoProvider, canonicalPayload),
    public_key: {
      format: "JWK",
      jwk: publicJwk,
      rfc7638_thumbprint_sha256: thumbprint,
    },
    signature: bytesToBase64Url(new Uint8Array(signature)),
    verification_boundary: signerClass === SIGNER_CLASSES.official
      ? "Cryptographic validity and official authority are separate checks. Official authority requires the fixed HTTPS registry, matching key ID/JWK/thumbprint, authorized scope, validity interval, and non-revoked status."
      : "The signature proves possession of a browser-local key for the signed package digest. It does not establish DEEPBOM authorship, organization identity, or official release status.",
  };
}

export async function verifyPackageSignature(record, expectedDigest, cryptoProvider = globalThis.crypto) {
  assertDigest(expectedDigest);
  if (![PUBLIC_KEY_SIGNATURE_SCHEMA, LEGACY_PUBLIC_KEY_SIGNATURE_SCHEMA].includes(record?.schema) || record?.algorithm !== "ES256" || record?.curve !== "P-256") return false;
  const signer = record.schema === PUBLIC_KEY_SIGNATURE_SCHEMA ? record?.signed_payload?.signer || null : null;
  if (record.schema === PUBLIC_KEY_SIGNATURE_SCHEMA && (!signer || ![SIGNER_CLASSES.local, SIGNER_CLASSES.official].includes(signer.class))) return false;
  const expectedPayload = signaturePayload(expectedDigest, record?.signed_payload?.scope, signer);
  if (canonicalJson(record.signed_payload) !== canonicalJson(expectedPayload)) return false;
  const canonicalPayload = canonicalJson(record.signed_payload);
  if (record.signed_payload_sha256 !== await sha256Hex(cryptoProvider, canonicalPayload)) return false;
  const jwk = record?.public_key?.jwk;
  if (!jwk || record.public_key.rfc7638_thumbprint_sha256 !== await publicJwkThumbprint(cryptoProvider, jwk)) return false;
  const publicKey = await cryptoProvider.subtle.importKey("jwk", jwk, { name: "ECDSA", namedCurve: "P-256" }, false, ["verify"]);
  return cryptoProvider.subtle.verify(
    { name: "ECDSA", hash: "SHA-256" },
    publicKey,
    base64UrlToBytes(record.signature),
    new TextEncoder().encode(canonicalPayload),
  );
}

export async function verifyPackageSignatureWithTrust(record, expectedDigest, registry, {
  cryptoProvider = globalThis.crypto,
  registryUrl = OFFICIAL_SIGNING_KEY_REGISTRY_URL,
} = {}) {
  const cryptographicallyValid = await verifyPackageSignature(record, expectedDigest, cryptoProvider);
  if (!cryptographicallyValid) return {
    cryptographically_valid: false,
    trusted: false,
    signer_class: record?.signed_payload?.signer?.class || null,
    status: "signature_invalid",
  };
  return {
    cryptographically_valid: true,
    ...resolveSignatureTrust(record, registry, { registryUrl }),
  };
}
