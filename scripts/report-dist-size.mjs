import path from "node:path";
import { collectFileSizes, formatBytes } from "./size-utils.mjs";

const dist = path.resolve("dist");
const topCount = Number(process.argv[2] || 15);

const files = await collectFileSizes(dist);
const total = files.reduce((sum, file) => sum + file.bytes, 0);
const largest = [...files].sort((a, b) => b.bytes - a.bytes).slice(0, Math.max(1, topCount));

console.log(`dist_files=${files.length}`);
console.log(`dist_total=${formatBytes(total)} (${total} bytes)`);
console.log(`largest_files_top=${largest.length}`);
for (const file of largest) {
  console.log(`${formatBytes(file.bytes).padStart(10)}  ${file.path}`);
}
