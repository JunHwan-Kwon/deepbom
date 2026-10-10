import { existsSync } from "node:fs";
import path from "node:path";
import { collectFileSizes, formatBytes, mibToBytes } from "./size-utils.mjs";

const dist = path.resolve("dist");
// The public multi-format corpus accounts for 7.3 MiB of the deploy artifact.
// Keep a small explicit ceiling above the measured release instead of requiring
// deploy-time overrides that would make the budget non-reproducible.
// The compatibility snapshot, schema and standalone explorer add about 70 KiB.
// Retain a bounded 128 KiB increment; the 16 MiB per-file ceiling is unchanged.
// Conversational queries add 99,113 bytes to the widget bundle and 27,968
// bytes of shared query modules. Reserve a further 128 KiB for these features.
// Workflow navigation and reused weight visual encoders raise measured dist to
// 66.76 MiB; reserve 128 KiB. The per-file ceiling remains unchanged.
// Native/numerical schemas, views and preserved 0.3–0.11 compatibility archives
// add 604,459 bytes before the shared native engine and updated bundles.
// Measured dist is 67.45 MiB; allocate a bounded 768 KiB feature increment.
// Saved-report workspace, compiled schema validator and immutable 0.12–0.13.1
// catalogs add about 380 KiB; retain the existing per-file ceiling.
// Native model/report support and immutable 0.14–0.15 catalogs bring the
// measured 2.2.0 release to 71,323,627 bytes (68.02 MiB). Reserve 128 KiB
// above the previous total budget; keep the 16 MiB per-file limit unchanged.
// Snapshot/workflow modules, schemas, the 0.16 catalog and public review page
// bring the measured 2.3.0 artifact to 71,532,074 bytes (68.22 MiB).
// Reserve a bounded 256 KiB increment; preserve the 16 MiB per-file ceiling.
const totalBudgetMiB = Number(process.env.DIST_BUDGET_MIB || 68.375);
const fileBudgetMiB = Number(process.env.DIST_FILE_BUDGET_MIB || 16);

if (!existsSync(dist)) {
  throw new Error("dist is missing. Run `npm run build:worker` before checking dist budget.");
}

const totalBudgetBytes = mibToBytes(totalBudgetMiB);
const fileBudgetBytes = mibToBytes(fileBudgetMiB);
const files = await collectFileSizes(dist);
const total = files.reduce((sum, file) => sum + file.bytes, 0);
const largest = [...files].sort((a, b) => b.bytes - a.bytes)[0];
const oversizedFiles = files.filter((file) => file.bytes > fileBudgetBytes).sort((a, b) => b.bytes - a.bytes);

if (total > totalBudgetBytes) {
  throw new Error(`dist budget exceeded: ${formatBytes(total)} > ${formatBytes(totalBudgetBytes)}. Set DIST_BUDGET_MIB to adjust intentionally.`);
}

if (oversizedFiles.length) {
  throw new Error(
    `dist per-file budget exceeded: ${oversizedFiles.map((file) => `${file.path}=${formatBytes(file.bytes)}`).join(", ")} > ${formatBytes(fileBudgetBytes)}. Set DIST_FILE_BUDGET_MIB to adjust intentionally.`,
  );
}

console.log(
  `Dist budget check passed (total ${formatBytes(total)} / ${formatBytes(totalBudgetBytes)}, largest ${largest ? `${largest.path} ${formatBytes(largest.bytes)}` : "n/a"} / ${formatBytes(fileBudgetBytes)}).`,
);
