import { readFile } from "node:fs/promises";
import process from "node:process";

const defaultUrl = "https://deepbom.org/web/sw.js";
const options = parseArgs(process.argv.slice(2));
const liveUrl = options.url || defaultUrl;
const versionPattern = /tflite-wasm-static-audit-v\d+/;

function parseArgs(args) {
  const parsed = {
    url: "",
    waitMs: 0,
    intervalMs: 15000,
  };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--url") {
      parsed.url = args[index + 1] || "";
      index += 1;
    } else if (arg === "--wait-ms") {
      parsed.waitMs = Number(args[index + 1] || 0);
      index += 1;
    } else if (arg === "--interval-ms") {
      parsed.intervalMs = Number(args[index + 1] || parsed.intervalMs);
      index += 1;
    } else if (!arg.startsWith("--") && !parsed.url) {
      parsed.url = arg;
    }
  }
  parsed.waitMs = Math.max(0, Number.isFinite(parsed.waitMs) ? parsed.waitMs : 0);
  parsed.intervalMs = Math.max(1000, Number.isFinite(parsed.intervalMs) ? parsed.intervalMs : 15000);
  return parsed;
}

function extractVersion(text, label) {
  const match = versionPattern.exec(text || "");
  if (!match) throw new Error(`Could not find service worker cache version in ${label}.`);
  return match[0];
}

async function readLocalVersion() {
  const source = await readFile(new URL("../web/sw.js", import.meta.url), "utf8");
  return extractVersion(source, "web/sw.js");
}

async function readLiveVersion() {
  const url = new URL(liveUrl);
  url.searchParams.set("check", String(Date.now()));
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  return extractVersion(await response.text(), url.href);
}

const local = await readLocalVersion();
const startedAt = Date.now();
let live = await readLiveVersion();
let status = local === live ? "match" : "pending";

while (status === "pending" && options.waitMs > 0 && Date.now() - startedAt < options.waitMs) {
  const elapsed = Date.now() - startedAt;
  console.log(`status=pending elapsed_ms=${elapsed}; waiting ${options.intervalMs} ms`);
  await sleep(options.intervalMs);
  live = await readLiveVersion();
  status = local === live ? "match" : "pending";
}

console.log(`local=${local}`);
console.log(`live=${live}`);
console.log(`status=${status}`);

if (status === "pending") {
  console.log("Live deployment has not caught up to the local service worker version yet.");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
