import { createHash } from "node:crypto";
import { canonicalJson } from "../web/lib/report-utils.js";
import { createZipBlob } from "../web/lib/zip.js";
import { parseStoredZip, verifyAttestedMemberMap, verifyPackageBytes } from "./verify-package.mjs";

const encoder = new TextEncoder();
const unsigned = [
  { name: "engineering_report.md", data: "report" },
  { name: "manifest.json", data: "{}" },
];
const files = unsigned.map((file) => {
  const bytes = encoder.encode(file.data);
  return { name: file.name, size: bytes.byteLength, sha256: sha256Hex(bytes) };
}).sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
const digestSet = { schema: "deepbom.package_member_digest_set.v1", files };
const packageHash = sha256Hex(encoder.encode(canonicalJson(digestSet)));
const kid = "deepbom-hs256-0123456789abcdefabcd";
const attestation = {
  schema: "deepbom.attestation.v2.2",
  signed_payload: {
    schema: "deepbom.attestation_payload.v2.2",
    signing_key: { kid, alg: "HS256" },
    package: {
      package_hash_sha256: packageHash,
      unsigned_package_members: files,
      path_normalization: "relative_posix_nfc",
      duplicate_member_policy: "reject_after_nfc_and_case_fold",
      undeclared_member_policy: "reject",
    },
    attestation_scope: {},
  },
  signature: { alg: "HS256", kid, value: "test-only" },
  verification_status: { package_hash_sha256: packageHash },
};
const zipFiles = [...unsigned, { name: "attestation.json", data: JSON.stringify(attestation) }];
const zipBytes = new Uint8Array(await createZipBlob(zipFiles).arrayBuffer());
const parsed = parseStoredZip(zipBytes);
if (parsed.size !== 3) throw new Error(`Expected 3 parsed ZIP members, received ${parsed.size}.`);
const result = verifyPackageBytes(zipBytes);
if (result.package_hash_sha256 !== packageHash || result.member_count !== 2) throw new Error("Valid package did not verify.");

const tampered = new Map(parsed);
tampered.set("engineering_report.md", encoder.encode("tampered"));
let rejected = false;
try {
  verifyAttestedMemberMap(tampered);
} catch {
  rejected = true;
}
if (!rejected) throw new Error("Tampered package member was not rejected.");

for (const unsafeNames of [
  ["../report.md"],
  ["/absolute/report.md"],
  ["Report.md", "report.md"],
  ["e\u0301vidence.json", "évidence.json"],
]) {
  const unsafe = new Map(parsed);
  unsafeNames.forEach((name, index) => unsafe.set(name, encoder.encode(`unsafe-${index}`)));
  let unsafeRejected = false;
  try {
    verifyAttestedMemberMap(unsafe);
  } catch {
    unsafeRejected = true;
  }
  if (!unsafeRejected) throw new Error(`Unsafe or colliding package member was not rejected: ${unsafeNames.join(", ")}`);
}

console.log("Package verifier check passed.");

function sha256Hex(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
