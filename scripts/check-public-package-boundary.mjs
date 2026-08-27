import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const releaseRoot = path.join(root, ".local-validation", "channel-release");
const packageRoot = path.join(releaseRoot, "npm", "package");
const expectedMembers = [
  "LICENSE",
  "README.md",
  "bin/deepbom.mjs",
  "package.json",
  "pkg/release-manifest.json",
  "pkg/tflite_wasm_audit_bg.wasm",
];

if (!existsSync(packageRoot)) {
  throw new Error("Public package boundary check requires npm run build:channels first.");
}

const members = (await collectFiles(packageRoot)).map((file) => relative(packageRoot, file)).sort();
assertEqual(members, expectedMembers, "npm package member allowlist");

const packageDocument = JSON.parse(await readFile(path.join(packageRoot, "package.json"), "utf8"));
assert(packageDocument.name === "deepbom", "Unexpected npm package name.");
assert(packageDocument.license === "Apache-2.0", "Public npm package must use Apache-2.0.");
assert(packageDocument.private !== true, "Generated npm package must not inherit the private monorepo publication guard.");
assert(packageDocument.repository?.url === "git+https://github.com/JunHwan-Kwon/deepbom.git", "npm repository identity drifted.");
assert(packageDocument.publishConfig?.access === "public", "npm package access must be explicit.");
assert(packageDocument.bin?.deepbom === "bin/deepbom.mjs", "npm CLI entry point drifted.");

const publicLicense = await readFile(path.join(root, "channels", "LICENSE"));
const packagedLicense = await readFile(path.join(packageRoot, "LICENSE"));
assert(sha256(publicLicense) === sha256(packagedLicense), "Packaged license differs from the public channel license.");

const releaseManifest = JSON.parse(await readFile(path.join(packageRoot, "pkg", "release-manifest.json"), "utf8"));
assert(releaseManifest.schema === "deepbom.npm_release.v1", "Unexpected npm release-manifest schema.");
assert(releaseManifest.license?.spdx === "Apache-2.0", "Release manifest is missing the public SPDX license.");
assert(releaseManifest.source?.distribution === "public_channel", "Release manifest is missing its public source-distribution identity.");
assert(Number.isInteger(releaseManifest.public_bundle_input_count) && releaseManifest.public_bundle_input_count > 0,
  "Release manifest is missing the verified public bundle input count.");

const wasm = await readFile(path.join(packageRoot, "pkg", "tflite_wasm_audit_bg.wasm"));
assert(sha256(wasm) === releaseManifest.runtime?.tflite_wasm_sha256, "Packaged WASM digest does not reproduce.");
assertWasmHasNoDebugCustomSections(wasm);

const bundle = await readFile(path.join(packageRoot, "bin", "deepbom.mjs"), "utf8");
const forbiddenText = [
  [/(?:^|[^A-Za-z])(?:[A-Za-z]:\\\\(?:Users|consistency|workspace)|\/home\/runner\/work\/)/, "absolute build path"],
  [/(?:sourceMappingURL|sourcesContent)/, "source-map marker"],
  [/(?:protected\/deepbom_wasm|web\/protected\/deepbom|deepbom_wasm_bg\.wasm)/, "protected analyzer path or payload"],
  [/scripts\/generate-(?:ort-rulepack|tflite-delegate-rulepack|xnnpack-delegate-rulepack)\.mjs/, "private rulepack generator path"],
  [/(?:ORT_EP_RULES|TFLITE_DELEGATE_RULES|fn\s+a55_kernel_candidates)/, "private generated-rule symbol"],
  [/(?:npm_[A-Za-z0-9]{20,}|pypi-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)/, "publication credential or private key"],
];
for (const [pattern, label] of forbiddenText) {
  assert(!pattern.test(bundle), `Public JavaScript bundle contains ${label}.`);
}

const forbiddenMemberPatterns = [/\.map$/i, /(?:^|\/)\.env(?:\.|$)/i, /(?:^|\/)\.git(?:\/|$)/i, /(?:^|\/)protected(?:\/|$)/i];
for (const member of members) {
  assert(!forbiddenMemberPatterns.some((pattern) => pattern.test(member)), `Forbidden public package member: ${member}`);
}

console.log(`Public package boundary passed (${members.length} exact members; ${releaseManifest.public_bundle_input_count} bundled source inputs; license, secrets, source maps, protected paths, and WASM custom sections checked).`);

async function collectFiles(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await collectFiles(file));
    else if (entry.isFile()) files.push(file);
  }
  return files;
}

function assertWasmHasNoDebugCustomSections(bytes) {
  assert(bytes.length >= 8 && bytes.subarray(0, 4).equals(Buffer.from([0x00, 0x61, 0x73, 0x6d])), "Packaged WASM magic is invalid.");
  let offset = 8;
  const forbidden = new Set(["name", "producers", "sourceMappingURL", "external_debug_info"]);
  while (offset < bytes.length) {
    const id = bytes[offset];
    offset += 1;
    const sectionSize = readUleb(bytes, offset);
    offset = sectionSize.next;
    const end = offset + sectionSize.value;
    assert(end <= bytes.length, "Packaged WASM section extends beyond the payload.");
    if (id === 0) {
      const nameLength = readUleb(bytes, offset);
      const nameStart = nameLength.next;
      const nameEnd = nameStart + nameLength.value;
      assert(nameEnd <= end, "Packaged WASM custom-section name is truncated.");
      const name = bytes.subarray(nameStart, nameEnd).toString("utf8");
      assert(!forbidden.has(name) && !name.startsWith(".debug_"), `Packaged WASM exposes debug custom section ${name}.`);
    }
    offset = end;
  }
  assert(offset === bytes.length, "Packaged WASM section ledger did not conserve byte length.");
}

function readUleb(bytes, start) {
  let value = 0;
  let shift = 0;
  let offset = start;
  for (let index = 0; index < 5; index += 1) {
    assert(offset < bytes.length, "Truncated WASM ULEB128 value.");
    const byte = bytes[offset];
    offset += 1;
    value |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return { value: value >>> 0, next: offset };
    shift += 7;
  }
  throw new Error("Oversized WASM ULEB128 value.");
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function relative(base, file) {
  return path.relative(base, file).replaceAll(path.sep, "/");
}

function assertEqual(actual, expected, label) {
  assert(JSON.stringify(actual) === JSON.stringify(expected), `${label} mismatch:\nexpected ${JSON.stringify(expected)}\nobserved ${JSON.stringify(actual)}`);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
