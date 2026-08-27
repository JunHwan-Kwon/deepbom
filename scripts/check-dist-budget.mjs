import { existsSync } from "node:fs";
import path from "node:path";
import { collectFileSizes, formatBytes, mibToBytes } from "./size-utils.mjs";

const dist = path.resolve("dist");
// The public multi-format corpus accounts for 7.3 MiB of the deploy artifact.
// Keep a small explicit ceiling above the measured release instead of requiring
// deploy-time overrides that would make the budget non-reproducible.
const totalBudgetMiB = Number(process.env.DIST_BUDGET_MIB || 64);
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
