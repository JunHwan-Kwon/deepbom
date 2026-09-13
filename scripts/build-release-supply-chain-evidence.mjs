import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const options = parseOptions(process.argv.slice(2));
const root = process.cwd();
const packageDocument = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));
const sourceCommit = options.commit || process.env.GITHUB_SHA || gitHeadFromEnvironment() || "unbound";
const sbomPath = path.resolve(options.sbom || "deepbom-self-sbom.cdx.json");
await mkdir(path.dirname(sbomPath), { recursive: true });

// npm is deliberately the primary SBOM generator. DeepBOM only normalizes the
// root release identity because npm derives its display name from the checkout
// directory for private source worktrees.
const npmInvocation = resolveNpmInvocation();
const sbom = JSON.parse(execFileSync(npmInvocation.executable, [...npmInvocation.prefix, "sbom", "--sbom-format", "cyclonedx", "--omit", "dev", "--package-lock-only"], {
  cwd: root,
  encoding: "utf8",
  maxBuffer: 32 * 1024 * 1024,
}));
normalizeReleaseIdentity(sbom, packageDocument, sourceCommit);
if (process.env.SOURCE_DATE_EPOCH) {
  const seconds = Number.parseInt(process.env.SOURCE_DATE_EPOCH, 10);
  if (!Number.isSafeInteger(seconds) || seconds < 0) throw new Error("SOURCE_DATE_EPOCH must be a non-negative integer.");
  sbom.metadata.timestamp = new Date(seconds * 1000).toISOString();
}
await writeFile(sbomPath, `${JSON.stringify(sbom, null, 2)}\n`, "utf8");

let provenance = null;
if (options.provenance) {
  const provenancePath = path.resolve(options.provenance);
  await mkdir(path.dirname(provenancePath), { recursive: true });
  const subjectRoot = path.resolve(options["subject-dir"] || path.dirname(sbomPath));
  const excluded = new Set([provenancePath, sbomPath]);
  const subjects = [];
  for (const file of await regularFiles(subjectRoot)) {
    if (excluded.has(file)) continue;
    const bytes = await readFile(file);
    subjects.push({
      name: path.relative(subjectRoot, file).replaceAll(path.sep, "/"),
      digest: { sha256: createHash("sha256").update(bytes).digest("hex") },
    });
  }
  provenance = {
    _type: "https://in-toto.io/Statement/v1",
    subject: subjects.sort((a, b) => a.name.localeCompare(b.name)),
    predicateType: "https://slsa.dev/provenance/v1",
    predicate: {
      buildDefinition: {
        buildType: "https://deepbom.org/build-types/release-channels/v1",
        externalParameters: { version: packageDocument.version, channel: "release" },
        internalParameters: { source_commit: sourceCommit },
        resolvedDependencies: [{
          uri: "pkg:npm/deepbom",
          digest: { sha256: createHash("sha256").update(await readFile(path.join(root, "package-lock.json"))).digest("hex") },
        }],
      },
      runDetails: {
        builder: { id: process.env.GITHUB_WORKFLOW_REF || "https://github.com/JunHwan-Kwon/deepbom/.github/workflows/release-channels.yml" },
        metadata: { invocationId: process.env.GITHUB_RUN_ID || "local-untrusted-build" },
      },
    },
  };
  await writeFile(provenancePath, `${JSON.stringify(provenance)}\n`, "utf8");
}

console.log(`Release supply-chain evidence written: ${sbomPath}${provenance ? ` and ${path.resolve(options.provenance)}` : ""}. Local provenance is unsigned; release workflows add identity-backed attestations.`);

function normalizeReleaseIdentity(document, project, commit) {
  document.metadata ||= {};
  document.metadata.component ||= {};
  Object.assign(document.metadata.component, {
    type: "application",
    name: "deepbom",
    version: project.version,
    "bom-ref": `pkg:npm/deepbom@${project.version}`,
    purl: `pkg:npm/deepbom@${project.version}`,
    licenses: [{ license: { id: "Apache-2.0" } }],
  });
  document.metadata.properties = [
    ...(document.metadata.properties || []).filter((item) => item?.name !== "deepbom:sbom:normalization"),
    { name: "deepbom:sbom:normalization", value: "npm-generated dependency graph; root identity normalized to the public DeepBOM release" },
    { name: "deepbom:sbom:source-commit", value: commit },
  ];
  document.serialNumber = `urn:uuid:${uuidFromSha256(createHash("sha256").update(`${project.version}:${commit}`).digest("hex"))}`;
}

function uuidFromSha256(hex) {
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-4${hex.slice(13, 16)}-a${hex.slice(17, 20)}-${hex.slice(20, 32)}`;
}

async function regularFiles(directory) {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) result.push(...await regularFiles(file));
    else if (entry.isFile() && (await stat(file)).isFile()) result.push(file);
  }
  return result;
}

function parseOptions(args) {
  const parsed = {};
  for (let index = 0; index < args.length; index += 1) {
    const token = args[index];
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const [name, inline] = token.slice(2).split("=", 2);
    const value = inline ?? args[++index];
    if (!value || value.startsWith("--")) throw new Error(`--${name} requires a value.`);
    parsed[name] = value;
  }
  return parsed;
}

function gitHeadFromEnvironment() {
  return /^[a-f0-9]{40}$/i.test(process.env.DEEPBOM_SOURCE_COMMIT || "") ? process.env.DEEPBOM_SOURCE_COMMIT : "";
}

function resolveNpmInvocation() {
  const candidates = [
    process.env.npm_execpath,
    path.join(path.dirname(process.execPath), "node_modules", "npm", "bin", "npm-cli.js"),
    path.resolve(path.dirname(process.execPath), "..", "lib", "node_modules", "npm", "bin", "npm-cli.js"),
  ].filter(Boolean);
  const cli = candidates.find((candidate) => existsSync(candidate));
  if (cli) return { executable: process.execPath, prefix: [cli] };
  return { executable: "npm", prefix: [] };
}
