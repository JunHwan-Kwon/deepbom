import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { canonicalJson, validatePackageMemberName } from "../web/lib/report-utils.js";

const ATTESTATION_MEMBER = "attestation.json";

export function parseStoredZip(input) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input || []);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const eocd = findSignature(view, 0x06054b50, Math.max(0, bytes.length - 65_557), bytes.length - 22);
  if (eocd < 0) throw new Error("ZIP end-of-central-directory record not found.");
  const entryCount = view.getUint16(eocd + 10, true);
  let offset = view.getUint32(eocd + 16, true);
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const members = new Map();
  const seenNames = new Set();
  const seenCaseFolded = new Set();
  for (let index = 0; index < entryCount; index += 1) {
    if (view.getUint32(offset, true) !== 0x02014b50) throw new Error(`Invalid ZIP central-directory entry ${index}.`);
    const method = view.getUint16(offset + 10, true);
    const compressedSize = view.getUint32(offset + 20, true);
    const uncompressedSize = view.getUint32(offset + 24, true);
    const nameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    const localOffset = view.getUint32(offset + 42, true);
    const name = validatePackageMemberName(decoder.decode(bytes.subarray(offset + 46, offset + 46 + nameLength)), seenNames, seenCaseFolded);
    seenNames.add(name);
    seenCaseFolded.add(name.toLocaleLowerCase("en-US"));
    if (method !== 0 || compressedSize !== uncompressedSize) throw new Error(`Unsupported compressed ZIP member ${name}; DEEPBOM package verification expects stored entries.`);
    if (view.getUint32(localOffset, true) !== 0x04034b50) throw new Error(`Invalid local ZIP header for ${name}.`);
    const localNameLength = view.getUint16(localOffset + 26, true);
    const localExtraLength = view.getUint16(localOffset + 28, true);
    const localName = decoder.decode(bytes.subarray(localOffset + 30, localOffset + 30 + localNameLength));
    if (localName !== name) throw new Error(`ZIP local and central-directory names differ for ${name}.`);
    const dataStart = localOffset + 30 + localNameLength + localExtraLength;
    const dataEnd = dataStart + uncompressedSize;
    if (dataEnd > bytes.length) throw new Error(`ZIP member ${name} extends beyond the archive.`);
    members.set(name, bytes.slice(dataStart, dataEnd));
    offset += 46 + nameLength + extraLength + commentLength;
  }
  return members;
}

export function verifyAttestedMemberMap(members) {
  const seenNames = new Set();
  const seenCaseFolded = new Set();
  for (const rawName of members.keys()) {
    const name = validatePackageMemberName(rawName, seenNames, seenCaseFolded);
    seenNames.add(name);
    seenCaseFolded.add(name.toLocaleLowerCase("en-US"));
  }
  const attestationBytes = members.get(ATTESTATION_MEMBER);
  if (!attestationBytes) throw new Error(`${ATTESTATION_MEMBER} is missing.`);
  let attestation;
  try {
    attestation = JSON.parse(new TextDecoder().decode(attestationBytes));
  } catch {
    throw new Error(`${ATTESTATION_MEMBER} is not valid JSON.`);
  }
  if (attestation.schema !== "deepbom.attestation.v2.2") throw new Error(`Unsupported attestation schema: ${attestation.schema || "missing"}`);
  const payload = attestation.signed_payload || {};
  if (payload.schema !== "deepbom.attestation_payload.v2.2") throw new Error(`Unsupported signed payload schema: ${payload.schema || "missing"}`);
  const declared = payload.package?.unsigned_package_members || [];
  const actualNames = [...members.keys()].filter((name) => name !== ATTESTATION_MEMBER).sort();
  const declaredNames = declared.map((item) => item.name).sort();
  if (canonicalJson(actualNames) !== canonicalJson(declaredNames)) throw new Error("Unsigned package-member names do not match the attested digest set.");
  const verifiedFiles = declared.map((item) => {
    const bytes = members.get(item.name);
    if (!bytes) throw new Error(`Attested member is missing: ${item.name}`);
    const sha256 = sha256Hex(bytes);
    if (bytes.byteLength !== item.size) throw new Error(`Size mismatch for ${item.name}: expected ${item.size}, observed ${bytes.byteLength}.`);
    if (sha256 !== item.sha256) throw new Error(`SHA-256 mismatch for ${item.name}.`);
    return { name: item.name, size: item.size, sha256 };
  }).sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  const digestSet = { schema: "deepbom.package_member_digest_set.v1", files: verifiedFiles };
  const packageHash = sha256Hex(new TextEncoder().encode(canonicalJson(digestSet)));
  if (packageHash !== payload.package?.package_hash_sha256) throw new Error("Canonical package-member digest-set SHA-256 mismatch.");
  if (attestation.verification_status?.package_hash_sha256 !== packageHash) throw new Error("Verification receipt package hash mismatch.");
  if (payload.package?.path_normalization !== "relative_posix_nfc" || payload.package?.duplicate_member_policy !== "reject_after_nfc_and_case_fold" || payload.package?.undeclared_member_policy !== "reject") throw new Error("Attestation package-member safety policy is missing or unsupported.");
  const kid = String(attestation.signature?.kid || "");
  if (!kid || /(^|[-_.])(current|latest|active)([-_.]|$)/i.test(kid)) throw new Error("Signing key ID is missing or is a mutable alias.");
  if (payload.signing_key?.kid !== kid || payload.signing_key?.alg !== attestation.signature?.alg) throw new Error("Signed key metadata does not match the signature envelope.");
  return {
    status: "member_digests_verified",
    package_hash_sha256: packageHash,
    member_count: verifiedFiles.length,
    signing_key_id: kid,
    signature_verification: "not_performed_server_only_hs256",
    attestation_scope: payload.attestation_scope || {},
    checks: {
      PACKAGE_MEMBERS: "PASS",
      MEMBER_PATH_SAFETY: "PASS",
      MEMBER_DIGESTS: "PASS",
      PACKAGE_HASH: "PASS",
      MODEL_HASH: "NOT_CHECKED",
      SERVER_REGISTRATION: "PRESENT",
      PUBLIC_KEY_SIGNATURE: "NOT_AVAILABLE",
      ANALYSIS_CORRECTNESS: "NOT_ASSESSED",
      DEPLOYMENT_READINESS: "NOT_ASSESSED",
    },
    attested_model_sha256: payload.subject?.model_sha256 || null,
  };
}

export function verifyPackageBytes(bytes) {
  return verifyAttestedMemberMap(parseStoredZip(bytes));
}

function findSignature(view, signature, start, end) {
  for (let offset = end; offset >= start; offset -= 1) {
    if (view.getUint32(offset, true) === signature) return offset;
  }
  return -1;
}

function sha256Hex(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function main() {
  const args = process.argv.slice(2);
  const packagePath = args.find((arg) => !arg.startsWith("--"));
  const modelFlag = args.indexOf("--model");
  const modelPath = modelFlag >= 0 ? args[modelFlag + 1] : null;
  if (!packagePath || (modelFlag >= 0 && !modelPath)) throw new Error("Usage: npm run verify:package -- <deepbom-package.zip> [--model <model.tflite|model.onnx>]");
  const result = verifyPackageBytes(await readFile(packagePath));
  if (modelPath) {
    const observedModelSha256 = sha256Hex(await readFile(modelPath));
    if (!result.attested_model_sha256) throw new Error("Attestation does not contain a model SHA-256 subject.");
    if (observedModelSha256 !== result.attested_model_sha256) throw new Error(`Model SHA-256 mismatch: expected ${result.attested_model_sha256}, observed ${observedModelSha256}.`);
    result.checks.MODEL_HASH = "PASS";
    result.model = { path: path.resolve(modelPath), sha256: observedModelSha256 };
  }
  process.stdout.write(`${JSON.stringify({ package: path.resolve(packagePath), ...result }, null, 2)}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => {
    process.stderr.write(`Package verification failed: ${error.message}\n`);
    process.exitCode = 1;
  });
}
