import os from "node:os";
import { readFile, writeFile } from "node:fs/promises";
import { runCommand } from "./run-utils.mjs";
import { hardenWasmFile } from "./wasm-binary-hardening.mjs";

// Wrap wasm-pack so release binaries never embed absolute build-machine paths.
// Rust panic-location strings include dependency source paths; without
// remapping, the public .wasm leaks the builder's home directory (user name)
// via cargo-registry paths. Function names/debug sections are already absent
// in release builds — this closes the remaining string leak.
const home = os.homedir();
const remap = [
  `--remap-path-prefix=${home}=~`,
  `--remap-path-prefix=${process.cwd()}=.`,
].join(" ");
process.env.RUSTFLAGS = [process.env.RUSTFLAGS, remap].filter(Boolean).join(" ");

const target = process.argv[2] || "main";
if (target === "main") {
  // Re-stamp the build-expiry (TTL) for each UTC build day so manual local
  // deploys cannot retain an old expiry while concurrent/same-day builds remain
  // byte-reproducible. Formal releases supply explicit epochs through CI.
  if (!process.env.APP_EXPIRES_AT_EPOCH_MS && !process.env.APP_EXPIRES_AT) {
    const days = Number(process.env.APP_EXPIRY_DAYS) || 30;
    const utcBuildDay = Math.floor(Date.now() / 86400_000) * 86400_000;
    process.env.APP_EXPIRES_AT_EPOCH_MS = String(utcBuildDay + days * 86400_000);
    process.env.APP_NOT_BEFORE_EPOCH_MS = String(utcBuildDay - 86400_000);
  }
  await normalizeGeneratedRepository("pkg/package.json");
  await runCommand("wasm-pack", ["build", "--target", "web", "--release"]);
  reportHardening("pkg/tflite_wasm_audit_bg.wasm");
} else if (target === "deepbom") {
  await normalizeGeneratedRepository("protected/deepbom_wasm/pkg/package.json");
  await runCommand("wasm-pack", [
    "build", "protected/deepbom_wasm", "--target", "web", "--release",
    "--out-dir", "../../web/protected/deepbom/pkg", "--out-name", "deepbom_wasm",
  ]);
  reportHardening("web/protected/deepbom/pkg/deepbom_wasm_bg.wasm");
} else {
  throw new Error(`unknown wasm build target: ${target}`);
}

async function normalizeGeneratedRepository(filePath) {
  let source;
  try {
    source = await readFile(filePath, "utf8");
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  const manifest = JSON.parse(source);
  if (!manifest.repository || typeof manifest.repository === "string") return;
  if (typeof manifest.repository !== "object" || typeof manifest.repository.url !== "string") {
    throw new Error(`${filePath}: unsupported generated repository metadata`);
  }
  manifest.repository = manifest.repository.url;
  await writeFile(filePath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
}

function reportHardening(filePath) {
  const report = hardenWasmFile(filePath);
  console.log(`WASM hardening: ${filePath}; custom=${report.strippedCustomSections.length}; paths=${report.redactedPathCount}; bytes=${report.beforeBytes}->${report.afterBytes}.`);
}
