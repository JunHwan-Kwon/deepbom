import { open, realpath, stat } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import { normalizeMetadataInput } from "../web/lib/provenance/omop.js";
import { PROVENANCE_LIMITS, requireProvenance } from "../web/lib/provenance/contracts.js";
import { observeEvidenceBytes } from "../web/lib/provenance/files.js";

export async function observeLocalEvidence(input, directory) {
  const { nodes } = normalizeMetadataInput(input);
  if (!directory) return [];
  const root = await realpath(directory), observations = []; let total = 0;
  for (const node of nodes) {
    if (!node.file) continue;
    const candidate = path.resolve(root, node.file);
    let resolved;
    try { resolved = await realpath(candidate); } catch (error) { if (error.code === "ENOENT") continue; throw error; }
    requireProvenance(path.relative(root, resolved) !== "" && !path.relative(root, resolved).startsWith(".." + path.sep) && path.relative(root, resolved) !== ".." && !path.isAbsolute(path.relative(root, resolved)), "evidence file resolves outside the selected evidence directory.");
    const handle = await open(resolved, constants.O_RDONLY | (constants.O_NOFOLLOW || 0));
    try {
      // Check the opened object before reading, including a parent-directory
      // symlink replacement between realpath() and open().
      const openedIdentity = await handle.stat({ bigint: true });
      const currentPath = await realpath(candidate), relative = path.relative(root, currentPath);
      requireProvenance(relative && relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative), "evidence file moved outside the selected directory.");
      const currentIdentity = await stat(currentPath, { bigint: true });
      requireProvenance(openedIdentity.dev === currentIdentity.dev && openedIdentity.ino === currentIdentity.ino, "evidence path changed while opening.");
      const before = await handle.stat();
      requireProvenance(before.isFile() && before.size <= PROVENANCE_LIMITS.file_bytes, "evidence input must be a regular file of at most 32 MiB.");
      total += before.size; requireProvenance(total <= PROVENANCE_LIMITS.total_file_bytes, "evidence files exceed 64 MiB in total.");
      const bytes = Buffer.alloc(before.size); let offset = 0;
      while (offset < bytes.length) { const { bytesRead } = await handle.read(bytes, offset, bytes.length - offset, offset); requireProvenance(bytesRead > 0, "evidence file changed while reading."); offset += bytesRead; }
      const after = await handle.stat();
      requireProvenance(before.size === after.size && before.mtimeMs === after.mtimeMs && before.ctimeMs === after.ctimeMs, "evidence file changed while reading.");
      observations.push(observeEvidenceBytes(node, bytes));
    } finally { await handle.close(); }
  }
  return observations;
}
