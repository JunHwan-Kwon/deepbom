import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { canonicalJson } from "../web/lib/report-utils.js";
import {
  createTensorRtBuildProfile,
  TENSORRT_BUILD_PROFILE_SCHEMA,
  TENSORRT_PARSER_OBSERVATION_SCHEMA,
} from "../web/lib/tensorrt-static-preflight.js";

const execFileAsync = promisify(execFile);
const SCRIPT_ROOT = path.dirname(fileURLToPath(import.meta.url));
export const REPOSITORY_ROOT = path.resolve(SCRIPT_ROOT, "..");
export const COLLECTOR_SOURCE_PATHS = Object.freeze([
  "native/tensorrt_collector/CMakeLists.txt",
  "native/tensorrt_collector/src/main.cc",
]);
const SHA256 = /^[a-f0-9]{64}$/;
const NATIVE_PARSER_METHODS = new Set(["supportsModel", "supportsModelV2"]);

export async function sha256File(filePath) {
  return createHash("sha256").update(await readFile(filePath)).digest("hex");
}

export async function tensorRtCollectorSourceIdentity(repositoryRoot = REPOSITORY_ROOT) {
  const files = [];
  for (const relativePath of COLLECTOR_SOURCE_PATHS) {
    const absolutePath = path.join(repositoryRoot, relativePath);
    const info = await stat(absolutePath);
    if (!info.isFile()) throw new Error(`TensorRT collector source is not a file: ${relativePath}`);
    files.push({ path: relativePath.replaceAll("\\", "/"), byte_length: info.size, sha256: await sha256File(absolutePath) });
  }
  return { files, source_set_sha256: sha256Text(canonicalJson(files)) };
}

export async function repositoryGitIdentity(repositoryRoot = REPOSITORY_ROOT) {
  const commit = (await run("git", ["rev-parse", "HEAD"], { cwd: repositoryRoot })).stdout.trim();
  const status = (await run("git", ["status", "--porcelain=v1", "--untracked-files=all"], { cwd: repositoryRoot })).stdout;
  if (!commit) throw new Error("TensorRT collector Git commit could not be resolved.");
  return { git_commit: commit, git_state: status.trim() ? "dirty" : "clean" };
}

export async function buildTensorRtCollector({ tensorRtRoot, buildDir = null, configuration = "Release", repositoryRoot = REPOSITORY_ROOT } = {}) {
  const sdkRoot = path.resolve(String(tensorRtRoot || process.env.TENSORRT_ROOT || ""));
  if (!String(tensorRtRoot || process.env.TENSORRT_ROOT || "").trim()) {
    throw new Error("TensorRT collector build requires --tensorrt-root or TENSORRT_ROOT; no SDK is selected implicitly.");
  }
  const resolvedBuildDir = path.resolve(buildDir || path.join(repositoryRoot, ".local-validation", "tensorrt-collector-build"));
  await mkdir(resolvedBuildDir, { recursive: true });
  await run("cmake", ["-S", path.join(repositoryRoot, "native", "tensorrt_collector"), "-B", resolvedBuildDir, `-DTENSORRT_ROOT=${sdkRoot}`], { cwd: repositoryRoot });
  await run("cmake", ["--build", resolvedBuildDir, "--config", configuration], { cwd: repositoryRoot });
  const filename = process.platform === "win32" ? "deepbom-tensorrt-parser-collector.exe" : "deepbom-tensorrt-parser-collector";
  const candidates = [path.join(resolvedBuildDir, configuration, filename), path.join(resolvedBuildDir, filename)];
  for (const candidate of candidates) {
    try { if ((await stat(candidate)).isFile()) return candidate; } catch { /* try the next generator layout */ }
  }
  throw new Error(`TensorRT collector binary was not found after build: ${candidates.join(", ")}`);
}

export async function captureTensorRtParser({
  modelPath,
  profile,
  collectorPath,
  outputPath,
  externalComponents = [],
  repositoryRoot = REPOSITORY_ROOT,
  collectorCommand = null,
  collectorArgumentPrefix = [],
} = {}) {
  const absoluteModel = path.resolve(String(modelPath || ""));
  const absoluteCollector = path.resolve(String(collectorPath || ""));
  if (path.extname(absoluteModel).toLowerCase() !== ".onnx") throw new Error("TensorRT parser capture requires an ONNX artifact.");
  if (!(await stat(absoluteModel)).isFile() || !(await stat(absoluteCollector)).isFile()) throw new Error("TensorRT model or collector is not a file.");
  const normalizedProfile = normalizeBuildProfile(profile);
  if (normalizedProfile.execution_path !== "native_tensorrt") {
    throw new Error("The native TensorRT collector cannot stand in for ORT TensorRT EP GetCapability evidence.");
  }
  const modelSha256 = await sha256File(absoluteModel);
  const collectorBinarySha256 = await sha256File(absoluteCollector);
  const sourceIdentity = await tensorRtCollectorSourceIdentity(repositoryRoot);
  const gitIdentity = await repositoryGitIdentity(repositoryRoot);
  const plugins = await verifyPlugins(normalizedProfile.plugins, path.dirname(absoluteModel));
  const components = await normalizeExternalComponents(externalComponents);
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "deepbom-tensorrt-"));
  try {
    const stagedModel = path.join(temporaryRoot, path.basename(absoluteModel));
    await copyFile(absoluteModel, stagedModel);
    for (const component of components) {
      const target = path.join(temporaryRoot, ...component.relative_path.split("/"));
      await mkdir(path.dirname(target), { recursive: true });
      await copyFile(component.source_path, target);
    }
    const profilePath = path.join(temporaryRoot, "deepbom-tensorrt-build-profile.json");
    const profileFileText = `${canonicalJson(normalizedProfile)}\n`;
    await writeFile(profilePath, profileFileText, "utf8");
    const args = [
      ...collectorArgumentPrefix,
      "--model", stagedModel,
      "--profile", profilePath,
      "--profile-sha256", normalizedProfile.profile_sha256,
      "--device-id", String(normalizedProfile.device_id),
      "--collector-binary-sha256", collectorBinarySha256,
      "--collector-source-set-sha256", sourceIdentity.source_set_sha256,
      "--collector-git-commit", gitIdentity.git_commit,
      "--collector-git-state", gitIdentity.git_state,
      ...plugins.flatMap((plugin) => ["--plugin", plugin.absolute_path]),
    ];
    const command = collectorCommand || absoluteCollector;
    const { stdout, stderr } = await run(command, args, { cwd: temporaryRoot, maxBuffer: 64 * 1024 * 1024 });
    let observation;
    try { observation = JSON.parse(stdout); } catch (error) { throw new Error(`TensorRT collector did not emit one JSON document: ${error.message}`); }
    observation.collector.source_files = sourceIdentity.files;
    observation.collector.stderr = stderr.trim() || null;
    observation.artifact_components = {
      binding_method: "isolated_staging_directory_with_explicit_relative_sidecars",
      main: { relative_path: path.basename(absoluteModel), byte_length: (await stat(absoluteModel)).size, sha256: modelSha256 },
      external: components.map(({ relative_path, byte_length, sha256 }) => ({ relative_path, byte_length, sha256 })),
    };
    observation.plugins = plugins.map(({ profile_path, byte_length, sha256 }) => ({ profile_path, byte_length, sha256 }));
    verifyTensorRtParserObservation(observation, {
      modelSha256,
      profile: normalizedProfile,
      profileFileSha256: sha256Text(profileFileText),
      collectorBinarySha256,
      sourceIdentity,
      gitIdentity,
      components,
      plugins,
    });
    if (outputPath) await atomicWriteJson(path.resolve(outputPath), observation);
    return observation;
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
}

export function verifyTensorRtParserObservation(observation, expected) {
  if (!observation || observation.schema !== TENSORRT_PARSER_OBSERVATION_SCHEMA
    || observation.artifact_sha256 !== expected.modelSha256
    || observation.build_profile_sha256 !== expected.profile.profile_sha256
    || observation.build_profile_file_sha256 !== expected.profileFileSha256
    || canonicalJson(observation.build_profile) !== canonicalJson(expected.profile)
    || observation.execution_path !== "native_tensorrt"
    || !NATIVE_PARSER_METHODS.has(observation.api_method)
    || typeof observation.parser_returned !== "boolean"
    || !String(observation.tensorrt_version || "").trim()
    || !String(observation.cuda_version || "").trim()
    || Number(observation.device_id) !== expected.profile.device_id
    || (expected.profile.device_compute_capability != null
      && String(observation.device_compute_capability || "") !== String(expected.profile.device_compute_capability))
    || !String(observation.device_identity || "").trim()
    || observation.collector?.binary_sha256 !== expected.collectorBinarySha256
    || observation.collector?.source_set_sha256 !== expected.sourceIdentity.source_set_sha256
    || observation.collector?.git_commit !== expected.gitIdentity.git_commit
    || observation.collector?.git_state !== expected.gitIdentity.git_state
    || !Array.isArray(observation.subgraphs) || !Array.isArray(observation.errors)) {
    throw new Error("TensorRT parser observation failed identity verification.");
  }
  if (!versionMatches(observation.tensorrt_version, expected.profile.expected_tensorrt_version)
    || !versionMatches(observation.cuda_version, expected.profile.expected_cuda_version)) {
    throw new Error("TensorRT parser observation runtime version differs from the bound profile.");
  }
  if (!SHA256.test(observation.collector.binary_sha256) || !SHA256.test(observation.collector.source_set_sha256)) {
    throw new Error("TensorRT collector digest is invalid.");
  }
  const componentRows = observation.artifact_components?.external || [];
  const expectedComponents = expected.components.map(({ relative_path, byte_length, sha256 }) => ({ relative_path, byte_length, sha256 }));
  if (canonicalJson(componentRows) !== canonicalJson(expectedComponents)) throw new Error("TensorRT external component manifest diverges.");
  const pluginRows = observation.plugins || [];
  const expectedPlugins = expected.plugins.map(({ profile_path, byte_length, sha256 }) => ({ profile_path, byte_length, sha256 }));
  if (canonicalJson(pluginRows) !== canonicalJson(expectedPlugins)) throw new Error("TensorRT plugin manifest diverges.");
  const seen = new Set();
  const expectedSemantics = observation.api_method === "supportsModel"
    ? "legacy_supported_collection_membership" : "per_subgraph_api_flag";
  if (observation.subgraph_support_semantics !== expectedSemantics) {
    throw new Error("TensorRT parser observation subgraph semantics do not match its API generation.");
  }
  for (const [position, subgraph] of observation.subgraphs.entries()) {
    if (subgraph.subgraph_index !== position || typeof subgraph.supported !== "boolean"
      || typeof subgraph.sdk_reported_flag !== "boolean" || !Array.isArray(subgraph.node_indices) || !subgraph.node_indices.length) {
      throw new Error(`TensorRT parser subgraph ${position} is malformed.`);
    }
    for (const node of subgraph.node_indices) {
      if (!Number.isSafeInteger(node) || node < 0 || seen.has(node)) throw new Error(`TensorRT parser node index ${node} is invalid or duplicated.`);
      seen.add(node);
    }
  }
  return observation;
}

function normalizeBuildProfile(value) {
  const source = typeof value === "string" ? JSON.parse(value) : value;
  if (!source || typeof source !== "object") throw new Error("TensorRT build profile JSON is required.");
  const normalized = createTensorRtBuildProfile(source);
  if (source.schema && source.schema !== TENSORRT_BUILD_PROFILE_SCHEMA) throw new Error("TensorRT build profile schema is invalid.");
  if (source.profile_sha256 && source.profile_sha256 !== normalized.profile_sha256) throw new Error("TensorRT build profile SHA-256 does not reproduce.");
  return normalized;
}

function versionMatches(observed, expected) {
  if (!expected) return true;
  const wanted = String(expected).trim();
  const actual = String(observed || "").trim();
  if (actual === wanted) return true;
  return (actual.match(/\d+(?:\.\d+){1,3}/g) || []).includes(wanted);
}

async function verifyPlugins(plugins, modelDirectory) {
  const rows = [];
  for (const plugin of plugins || []) {
    const absolutePath = path.resolve(modelDirectory, plugin.path);
    const actual = await sha256File(absolutePath);
    if (actual !== plugin.sha256) throw new Error(`TensorRT plugin SHA-256 diverges: ${plugin.path}`);
    rows.push({ profile_path: plugin.path, absolute_path: absolutePath, byte_length: (await stat(absolutePath)).size, sha256: actual });
  }
  return rows;
}

async function normalizeExternalComponents(values) {
  const rows = [];
  const names = new Set();
  for (const value of values || []) {
    const relativePath = String(value?.relative_path || "").replaceAll("\\", "/");
    const sourcePath = path.resolve(String(value?.source_path || ""));
    if (!relativePath || path.posix.isAbsolute(relativePath) || relativePath.split("/").some((part) => !part || part === "." || part === "..")) {
      throw new Error(`TensorRT external component path is unsafe: ${relativePath || "empty"}`);
    }
    if (names.has(relativePath)) throw new Error(`TensorRT external component path duplicates ${relativePath}.`);
    names.add(relativePath);
    const info = await stat(sourcePath);
    if (!info.isFile()) throw new Error(`TensorRT external component is not a file: ${sourcePath}`);
    rows.push({ relative_path: relativePath, source_path: sourcePath, byte_length: info.size, sha256: await sha256File(sourcePath) });
  }
  return rows.sort((a, b) => a.relative_path.localeCompare(b.relative_path));
}

async function atomicWriteJson(outputPath, value) {
  await mkdir(path.dirname(outputPath), { recursive: true });
  const temporaryPath = `${outputPath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, `${canonicalJson(value)}\n`, "utf8");
  await rename(temporaryPath, outputPath);
}

function sha256Text(value) { return createHash("sha256").update(value, "utf8").digest("hex"); }

async function run(command, args, options = {}) {
  try {
    return await execFileAsync(command, args, { windowsHide: true, encoding: "utf8", maxBuffer: 16 * 1024 * 1024, ...options });
  } catch (error) {
    const detail = String(error.stderr || error.stdout || error.message || error).trim();
    throw new Error(`${path.basename(command)} failed${detail ? `: ${detail}` : ""}`);
  }
}
