import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile, copyFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = process.cwd();
const option = name => { const index = process.argv.indexOf(name); return index < 0 ? null : process.argv[index + 1]; };
const packageRoot = path.resolve(option("--package") || root);
const python = option("--python");
const sdk = await import(pathToFileURL(path.join(packageRoot, "sdk/index.mjs")));
const artifact = path.join(root, "web/samples/gpu_partition_probe.onnx");
const sha256 = "82a2feef00eb6ab03d82f2b30cd17f4d826e2d8307cb059eccd6a0f3120059b2";
const temp = await mkdtemp(path.join(tmpdir(), "deepbom-sdk-test-"));
const run = (command, args, options = {}) => {
  const result = spawnSync(command, args, { encoding: "utf8", timeout: 60_000, maxBuffer: 16 * 1024 * 1024, ...options });
  assert.equal(result.status, 0, result.stderr || result.error?.message);
  return result.stdout;
};
const canonical = args => JSON.parse(run(process.execPath, [path.join(packageRoot, "bin/deepbom.mjs"), ...args, "--compact"]));
try {
  const summary = await sdk.inspect(artifact, { expectedSha256: sha256 });
  assert.deepEqual(summary, canonical(["audit", artifact, "--scan", "auto", "--section", "summary"]).sections.summary);
  assert.deepEqual(await sdk.audit(artifact), canonical(["audit", artifact, "--scan", "auto", "--output-format", "envelope"]));
  assert.equal(summary.artifact.sha256, sha256);
  assert.equal(summary.verdict.artifact_defect_count, 0);
  assert(summary.verdict.evidence_needed_count > 0);
  assert.equal(summary.coverage.declared, summary.coverage.assessed + summary.coverage.partial + summary.coverage.unavailable);
  assert.equal(summary.coverage.unavailable, summary.coverage.needs_external_evidence, "retain the legacy alias without calling it an evidence-gap count");
  assert.equal(summary.reproduction.expected_sha256, sha256);
  assert.deepEqual(await sdk.capabilities(), canonical(["capabilities"]));
  assert.deepEqual(await sdk.diff(artifact, artifact), canonical(["diff", artifact, artifact]));
  const contract = path.join(temp, "interface.json");
  await writeFile(contract, JSON.stringify(await sdk.captureContract(artifact)));
  await sdk.verifyContract(artifact, contract);
  const bom = await sdk.audit(artifact, { output: "cyclonedx" });
  const bomPath = path.join(temp, "bom.json");
  await writeFile(bomPath, JSON.stringify(bom));
  await sdk.verifyBom(artifact, bomPath);
  const hash = bom.metadata.component.hashes.find(row => row.alg === "SHA-256");
  hash.content = "0".repeat(64);
  await writeFile(bomPath, JSON.stringify(bom));
  await assert.rejects(sdk.verifyBom(artifact, bomPath), error => error instanceof sdk.DeepBomIncompleteBinding && error.exitCode === 3 && !!error.document);
  await assert.rejects(sdk.inspect(artifact, { expectedSha256: "0".repeat(64) }), error => error instanceof sdk.DeepBomIdentityMismatch && error.exitCode === 4 && error.code === "IDENTITY_MISMATCH");
  await assert.rejects(sdk.audit(artifact, { maxOutputBytes: 1 }), sdk.DeepBomOutputTooLarge);
  await assert.rejects(sdk.audit(artifact, { timeoutMs: 1 }), sdk.DeepBomTimeout);
  await assert.rejects(sdk.audit(path.join(temp, "missing.onnx")), sdk.DeepBomInvocationError);
  for (const options of [{ scann: "full" }, { expectedSha256: "bad" }, { gate: "defects", policy: "regulatory" }, { timeoutMs: NaN }, { timeoutMs: 2 ** 32 }, { sections: "summary" }, { maxOutputBytes: true }]) {
    await assert.rejects(sdk.audit(artifact, options), TypeError);
  }
  await assert.rejects(sdk.inspect(artifact, { output: "analysis" }), TypeError);
  await assert.rejects(sdk.audit(artifact, Object.create({ expectedSha256: "0".repeat(64) })), TypeError);
  await assert.rejects(sdk.inspect(artifact, Object.defineProperty({}, "expectedSha256", { value: "0".repeat(64) })), TypeError);
  // Exercise the real process/read boundary with malformed engine responses.
  const fake = path.join(temp, "malformed-engine");
  await mkdir(path.join(fake, "sdk"), { recursive: true }); await mkdir(path.join(fake, "bin"));
  await copyFile(path.join(packageRoot, "sdk/index.mjs"), path.join(fake, "sdk/index.mjs"));
  const boundary = await import(pathToFileURL(path.join(fake, "sdk/index.mjs")));
  for (const bytes of [Buffer.from('{"value":1e999}'), Buffer.from([123,34,118,34,58,34,255,34,125]), Buffer.from('[]')]) {
    await writeFile(path.join(fake, "bin/deepbom.mjs"), `import { writeFileSync } from 'node:fs'; writeFileSync(process.argv[process.argv.indexOf('--output')+1], Buffer.from('${bytes.toString("base64")}', 'base64'));`);
    await assert.rejects(boundary.capabilities(), boundary.DeepBomInvocationError);
  }
  const spaced = path.join(temp, "- model ; literal $.onnx");
  await copyFile(artifact, spaced);
  assert.equal((await sdk.inspect(spaced)).artifact.sha256, sha256);

  let header = JSON.stringify({ weight: { dtype: "F32", shape: [1], data_offsets: [0, 4] } });
  header = header.padEnd(Math.ceil(header.length / 8) * 8, " ");
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(header.length));
  const payload = Buffer.alloc(4); payload.writeFloatLE(NaN);
  const defective = path.join(temp, "nan.safetensors");
  await writeFile(defective, Buffer.concat([prefix, Buffer.from(header), payload]));
  const storageOnly = await sdk.inspect(defective, { scan: "structure" });
  assert.equal(storageOnly.graph.mac_confidence, "not_applicable");
  assert.equal(storageOnly.graph.total_macs, null, "storage-only cost must not become zero");
  const structureEnvelope = await sdk.audit(defective, { scan: "structure" });
  assert(structureEnvelope.capabilities.unavailable.includes("tensor_payloads"), "an intentionally skipped payload scan is not partial execution");
  assert(!structureEnvelope.capabilities.assessed.includes("tensor_payloads"));
  for (const method of ["audit", "inspect"]) {
    await assert.rejects(sdk[method](defective, { scan: "integrity", gate: "defects" }), error => {
      assert(error instanceof sdk.DeepBomPolicyBlocked);
      assert.equal(error.exitCode, 2);
      assert.equal(error.code, "POLICY_BLOCKED");
      assert.equal(error.document.schema, method === "inspect" ? "deepbom.review_summary.v1" : "deepbom.artifact_evidence_envelope.v1");
      if (method === "inspect") assert(error.document.verdict.artifact_defect_count > 0);
      return true;
    });
  }

  // Resolve the published exports map from an installed consumer, not a source fallback.
  const consumer = option("--consumer") || root;
  assert.equal(run(process.execPath, ["--input-type=module", "-e", "import { inspect } from 'deepbom'; console.log(typeof inspect)"], { cwd: consumer }).trim(), "function");
  const typeFile = path.join(consumer, `.deepbom-sdk-types-${process.pid}.mts`);
  try {
    await writeFile(typeFile, `import { audit, inspect, diff, type EvidenceEnvelope, type ReviewSummary } from 'deepbom';
const envelope: EvidenceEnvelope = await audit('model.onnx');
const summary: ReviewSummary = await inspect('model.onnx', { expectedSha256: 'a'.repeat(64) });
const count: number = summary.verdict.artifact_defect_count;
const cost: number | null = summary.graph.total_macs;
const digest: string = envelope.identity.sha256;
await diff('a.onnx', 'b.onnx', { tensorsOnly: true });
// @ts-expect-error Unknown option must be rejected.
await inspect('model.onnx', { scann: 'full' });
// @ts-expect-error Alternative projection must not masquerade as a canonical envelope.
const wrong: EvidenceEnvelope = await audit('model.onnx', { output: 'sarif' });
`);
    run(process.execPath, [path.join(root, "node_modules/typescript/bin/tsc"), "--strict", "--noEmit", "--module", "NodeNext", "--target", "ES2022", typeFile]);
  } finally { await rm(typeFile, { force: true }); }
  if (python) {
    const restored = JSON.parse(run(python, ["-c", "import deepbom,json,sys; print(json.dumps(deepbom.inspect(sys.argv[1],expected_sha256=sys.argv[2])))", artifact, sha256]));
    assert.deepEqual(restored, summary, "Python and Node must preserve the exact common summary");
    const result = run(python, ["-c", `import deepbom,sys
try:
    deepbom.inspect(sys.argv[1],scan='integrity',gate='defects')
except deepbom.DeepBomPolicyBlocked as e:
    assert e.document['schema']=='deepbom.review_summary.v1' and e.exit_code==2 and e.code=='POLICY_BLOCKED'
else:
    raise AssertionError('defect gate should block')
`, defective]);
    assert.equal(result, "");
  }
  console.log("SDK checks passed: engine parity, identity, coverage, unknowns, comparison, BOM/interface verification, typed errors, process bounds, imports and TypeScript declarations" + (python ? ", installed Python parity." : "."));
} finally { await rm(temp, { recursive: true, force: true }); }
