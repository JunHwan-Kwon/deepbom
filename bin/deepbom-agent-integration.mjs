import { createHash } from "node:crypto";
import { lstat, mkdir, readFile, rename, rm, rmdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { buildAgentSkillFiles } from "./deepbom-agent-skill.mjs";

export const AGENT_INTEGRATION_SCHEMA = "deepbom.agent_integration.v1";

const TARGETS = Object.freeze({
  codex: ".agents/skills/deepbom",
  "claude-code": ".claude/skills/deepbom",
  generic: "skills/deepbom",
});
const MANIFEST_FILE = ".deepbom-integration.json";
export function agentIntegrationTargets() {
  return Object.entries(TARGETS).map(([id, destination]) => ({ id, destination }));
}

export async function manageAgentIntegration({ action, target = null, apply = false, root = process.cwd(), version }) {
  const projectRoot = path.resolve(root);
  const skillFiles = buildAgentSkillFiles(version);
  const targets = target ? [normalizeTarget(target)] : Object.keys(TARGETS);
  if (action === "status") {
    return {
      schema: AGENT_INTEGRATION_SCHEMA,
      action,
      applied: false,
      project_root: projectRoot,
      integrations: await Promise.all(targets.map((id) => inspectIntegration(projectRoot, skillFiles, id, version))),
    };
  }
  if (targets.length !== 1) throw new Error(`The ${action} action requires one integration target.`);
  const id = targets[0];
  if (action === "install") return installIntegration(projectRoot, skillFiles, id, version, apply);
  if (action === "remove") return removeIntegration(projectRoot, id, apply);
  throw new Error(`Unknown agent integration action: ${action}`);
}

async function installIntegration(projectRoot, skillFiles, target, version, apply) {
  const inspection = await inspectIntegration(projectRoot, skillFiles, target, version);
  if (inspection.status === "conflict") {
    throw new Error(`Agent integration would overwrite unmanaged changes: ${inspection.conflicts.join(", ")}.`);
  }
  const destination = path.resolve(projectRoot, TARGETS[target]);
  await assertSafeDestination(projectRoot, destination);
  const changes = inspection.files.filter((row) => row.status !== "unchanged");
  const result = {
    schema: AGENT_INTEGRATION_SCHEMA,
    action: "install",
    target,
    applied: Boolean(apply),
    status: changes.length ? (apply ? "installed" : "changes_pending") : "current",
    project_root: projectRoot,
    destination: portable(path.relative(projectRoot, destination)),
    changes,
    next_command: apply ? null : `deepbom integrate ${target} --apply`,
  };
  if (!apply || !changes.length) return result;

  await mkdir(destination, { recursive: true });
  const records = [];
  for (const [relative, content] of skillFiles) {
    const bytes = Buffer.from(content, "utf8");
    const targetPath = path.join(destination, relative);
    await mkdir(path.dirname(targetPath), { recursive: true });
    await writeAtomically(targetPath, bytes);
    records.push({ path: relative, sha256: sha256(bytes) });
  }
  const manifest = {
    schema: AGENT_INTEGRATION_SCHEMA,
    owner: "deepbom",
    target,
    version,
    managed_files: records,
  };
  await writeAtomically(path.join(destination, MANIFEST_FILE), Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`));
  return result;
}

async function removeIntegration(projectRoot, target, apply) {
  const destination = path.resolve(projectRoot, TARGETS[target]);
  await assertSafeDestination(projectRoot, destination);
  const manifest = await readManifest(destination);
  if (!manifest) {
    return {
      schema: AGENT_INTEGRATION_SCHEMA,
      action: "remove",
      target,
      applied: false,
      status: "not_installed",
      destination: portable(path.relative(projectRoot, destination)),
      changes: [],
    };
  }
  const changed = [];
  for (const row of manifest.managed_files || []) {
    const bytes = await readOptional(path.join(destination, row.path));
    if (!bytes || sha256(bytes) !== row.sha256) changed.push(row.path);
  }
  if (changed.length) throw new Error(`Refusing to remove modified agent integration files: ${changed.join(", ")}.`);
  const result = {
    schema: AGENT_INTEGRATION_SCHEMA,
    action: "remove",
    target,
    applied: Boolean(apply),
    status: apply ? "removed" : "changes_pending",
    destination: portable(path.relative(projectRoot, destination)),
    changes: [...(manifest.managed_files || []).map((row) => ({ path: row.path, status: "remove" })), { path: MANIFEST_FILE, status: "remove" }],
    next_command: apply ? null : `deepbom integrate remove ${target} --apply`,
  };
  if (!apply) return result;
  for (const row of manifest.managed_files || []) await rm(path.join(destination, row.path), { force: true });
  await rm(path.join(destination, MANIFEST_FILE), { force: true });
  const directories = [...new Set((manifest.managed_files || []).map((row) => path.dirname(path.join(destination, row.path))))]
    .sort((left, right) => right.length - left.length);
  for (const directory of directories) await removeEmptyParents(directory, projectRoot);
  await removeEmptyParents(destination, projectRoot);
  return result;
}

async function inspectIntegration(projectRoot, skillFiles, target, version) {
  const destination = path.resolve(projectRoot, TARGETS[target]);
  await assertSafeDestination(projectRoot, destination);
  const manifest = await readManifest(destination);
  const prior = new Map((manifest?.managed_files || []).map((row) => [row.path, row.sha256]));
  const files = [];
  const conflicts = [];
  for (const [relative, content] of skillFiles) {
    const expected = Buffer.from(content, "utf8");
    const observed = await readOptional(path.join(destination, relative));
    let status = "add";
    if (observed && sha256(observed) === sha256(expected)) status = "unchanged";
    else if (observed && prior.get(relative) === sha256(observed)) status = "update_managed";
    else if (observed) {
      status = "conflict";
      conflicts.push(relative);
    }
    files.push({ path: relative, status, sha256: sha256(expected) });
  }
  const unchanged = files.every((row) => row.status === "unchanged");
  return {
    target,
    destination: portable(path.relative(projectRoot, destination)),
    status: conflicts.length ? "conflict" : unchanged ? "current" : manifest ? "outdated" : "not_installed",
    installed_version: manifest?.version || null,
    available_version: version,
    conflicts,
    files,
  };
}

function normalizeTarget(value) {
  const target = String(value || "").trim().toLowerCase();
  if (!Object.hasOwn(TARGETS, target)) throw new Error("Agent integration target must be codex, claude-code, or generic.");
  return target;
}

async function readManifest(destination) {
  const bytes = await readOptional(path.join(destination, MANIFEST_FILE));
  if (!bytes) return null;
  const manifest = JSON.parse(bytes.toString("utf8"));
  if (manifest.schema !== AGENT_INTEGRATION_SCHEMA || manifest.owner !== "deepbom" || !Array.isArray(manifest.managed_files)) {
    throw new Error(`Invalid DEEPBOM integration manifest: ${path.join(destination, MANIFEST_FILE)}`);
  }
  return manifest;
}

async function assertSafeDestination(projectRoot, destination) {
  const relative = path.relative(projectRoot, destination);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) throw new Error("Agent integration destination must stay below the current directory.");
  let cursor = projectRoot;
  for (const segment of relative.split(path.sep)) {
    cursor = path.join(cursor, segment);
    try {
      const entry = await lstat(cursor);
      if (entry.isSymbolicLink()) throw new Error(`Agent integration refuses symbolic-link path components: ${cursor}`);
      if (!entry.isDirectory()) throw new Error(`Agent integration path component is not a directory: ${cursor}`);
    } catch (error) {
      if (error?.code === "ENOENT") break;
      throw error;
    }
  }
}

async function writeAtomically(destination, bytes) {
  const temporary = `${destination}.${process.pid}.tmp`;
  const backup = `${destination}.${process.pid}.bak`;
  await writeFile(temporary, bytes, { mode: 0o600 });
  const existed = await pathExists(destination);
  try {
    if (existed) await rename(destination, backup);
    await rename(temporary, destination);
    if (existed) await rm(backup, { force: true });
  } catch (error) {
    await rm(temporary, { force: true });
    if (existed && await pathExists(backup) && !await pathExists(destination)) await rename(backup, destination);
    throw error;
  }
}

async function pathExists(file) {
  try { await lstat(file); return true; } catch (error) { if (error?.code === "ENOENT") return false; throw error; }
}

async function removeEmptyParents(start, stop) {
  let cursor = start;
  while (cursor.startsWith(`${stop}${path.sep}`)) {
    try { await rmdir(cursor); } catch { return; }
    cursor = path.dirname(cursor);
  }
}

async function readOptional(file) {
  try { return await readFile(file); } catch (error) { if (error?.code === "ENOENT") return null; throw error; }
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function portable(value) {
  return value.replaceAll(path.sep, "/");
}
