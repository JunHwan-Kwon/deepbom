// Preserve compatibility 0.45.0 (~118 KiB) plus the expanded common WASM; total allowance +128 KiB.
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
// Structure IDE, its shared editor and immutable 0.17–0.20 compatibility
// snapshots bring the measured Web artifact to 68.68 MiB. Allocate 384 KiB
// for this delivery; retain the 16 MiB per-file limit and explicit total gate.
// Orthogonal routing and immutable 0.21 catalog bring the artifact to 68.78 MiB.
// Reserve 128 KiB; the per-file cap is unchanged.
// Source hierarchy modules/schema and the immutable 0.22 catalog bring the
// measured release to 69.03 MiB. Reserve 256 KiB; initial scripts remain 2,475 gzip bytes.
// Portable numerical/evidence owners, HDF5 decoding and archived compatibility
// catalogs bring this development artifact to 85,192,073 bytes (81.25 MiB).
// The app reuses the copied HDF5 vendor module, eliminating a 5,594,575-byte
// duplicate. The ChatGPT audit worker remains self-contained by contract.
// Keep a bounded 81.5 MiB total and the unchanged 16 MiB per-file gate.
// Portable IR owners plus immutable 0.46–0.49 catalogs require a bounded 512 KiB increment.
// Preserve all runtime and per-file ceilings.
// Final immutable 0.54/0.55 catalogs and portable rules measure 86,135,852 bytes.
// Add 64 KiB to the total only; keep the 16 MiB per-file and startup limits.
// Release 2.5.0 retains the 0.56.0 and 0.57.0 catalogs and rebuilt companion WASM.
// Measured assembly is 82.29 MiB; add 128 KiB total while retaining all per-file/startup limits.
// Bilingual query guidance, widget navigation and immutable 0.57.1 total 82.57 MiB.
// Allocate 256 KiB; preserve the existing per-file and startup limits.
// Selection context, follow-up controls and the immutable 0.57.2 catalog bring
// measured dist to 86,714,220 bytes (82.70 MiB). Add a bounded 64 KiB;
// preserve the 16 MiB per-file and 2,475-byte gzip startup ceilings.
// Local exploration, shared hierarchy UI and immutable 0.57.3 measure 86882348 bytes.
// Add 128 KiB total; preserve per-file and startup ceilings.
const totalBudgetMiB = Number(process.env.DIST_BUDGET_MIB || 82.875);
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
