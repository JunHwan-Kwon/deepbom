import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

// An unrelated deployment is not a content update. Omit the optional field
// when a clean source revision cannot be established (exports, dirty trees,
// missing Git, or a history truncated at the relevant commit).
export function sitemapLastmod(source, { cwd = process.cwd() } = {}) {
  const git = (...args) => spawnSync("git", args, {
    cwd, encoding: "utf8", timeout: 5000, windowsHide: true,
  });
  const status = git("status", "--porcelain", "--", source);
  if (status.status !== 0 || status.stdout.trim()) return [];
  const log = git("log", "-1", "--format=%H %cs", "--", source);
  if (log.status !== 0) return [];
  const match = log.stdout.trim().match(/^([a-f0-9]{40,64}) (\d{4}-\d{2}-\d{2})$/);
  if (!match) return [];
  const shallow = git("rev-parse", "--git-path", "shallow");
  if (shallow.status !== 0) return [];
  const boundary = path.resolve(cwd, shallow.stdout.trim());
  if (existsSync(boundary) && readFileSync(boundary, "utf8").split(/\s+/).includes(match[1])) return [];
  return [`    <lastmod>${match[2]}</lastmod>`];
}
