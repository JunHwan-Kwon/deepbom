import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";

import { CYCLONEDX_DRAFT_PROFILE_REGISTRY } from "../web/lib/cyclonedx-draft-profiles.js";

const profile = CYCLONEDX_DRAFT_PROFILE_REGISTRY.profiles[CYCLONEDX_DRAFT_PROFILE_REGISTRY.active_profile_id];
let checked = 0;
for (const source of profile.sources) {
  const pullNumber = Number(new URL(source.pull_request).pathname.split("/").at(-1));
  const remote = `https://github.com/${source.repository}.git`;
  const headLine = execFileSync("git", ["ls-remote", remote, `refs/pull/${pullNumber}/head`], {
    encoding: "utf8",
    timeout: 30_000,
  }).trim();
  const observedHead = headLine.split(/\s+/)[0] || "";
  if (observedHead !== source.commit) {
    const error = new Error(`STALE_DRAFT_PROFILE: ${source.repository}#${pullNumber} moved from ${source.commit} to ${observedHead || "unresolved"}.`);
    error.code = "STALE_DRAFT_PROFILE";
    throw error;
  }
  for (const content of source.contents) {
    const url = `https://raw.githubusercontent.com/${source.repository}/${source.commit}/${content.path}`;
    const response = await fetch(url, { redirect: "error" });
    if (!response.ok) throw new Error(`Cannot fetch pinned CycloneDX source (${response.status}): ${url}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const observed = createHash("sha256").update(bytes).digest("hex");
    if (observed !== content.sha256) throw new Error(`CycloneDX source pin mismatch for ${url}: expected ${content.sha256}, observed ${observed}`);
    checked += 1;
  }
}
console.log(`CycloneDX draft source pins passed (${profile.sources.length} current PR heads, ${checked} immutable source objects).`);
