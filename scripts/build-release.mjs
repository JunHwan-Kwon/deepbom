import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { RELEASE_GENERATED_TRACKED_ARTIFACTS } from "./release-generated-artifacts.mjs";

const statusArgs = [
  "status",
  "--porcelain=v1",
  "--untracked-files=all",
  "--",
  ".",
  ":!web/lib/build-metadata.js",
  ":!dist",
  ...RELEASE_GENERATED_TRACKED_ARTIFACTS.map((file) => `:!${file}`),
];
const status = execFileSync("git", statusArgs, { encoding: "utf8" }).trim();
if (status) {
  throw new Error(`Formal release builds require a clean source tree. Commit or remove these changes first:\n${status}`);
}

for (const name of ["APP_EXPIRES_AT_EPOCH_MS", "APP_NOT_BEFORE_EPOCH_MS"]) {
  if (!/^\d{13}$/.test(String(process.env[name] || ""))) {
    throw new Error(`Formal release builds require an explicit 13-digit ${name} so runtime-guard bytes are reproducible.`);
  }
}

const delegateRulepackVerification = spawnSync(process.execPath, ["scripts/generate-xnnpack-delegate-rulepack.mjs", "--check"], {
  stdio: "inherit",
});
if (delegateRulepackVerification.error) throw delegateRulepackVerification.error;
if (delegateRulepackVerification.status !== 0) process.exit(delegateRulepackVerification.status || 1);

const sourcePinVerification = spawnSync(process.execPath, ["scripts/verify-xnnpack-source-pin.mjs"], {
  stdio: "inherit",
});
if (sourcePinVerification.error) throw sourcePinVerification.error;
if (sourcePinVerification.status !== 0) process.exit(sourcePinVerification.status || 1);

const onnxShapeSchemaVerification = spawnSync(process.execPath, ["scripts/generate-onnx-shape-schema.mjs", "--check"], {
  stdio: "inherit",
});
if (onnxShapeSchemaVerification.error) throw onnxShapeSchemaVerification.error;
if (onnxShapeSchemaVerification.status !== 0) process.exit(onnxShapeSchemaVerification.status || 1);

const onnxContainerSourceVerification = spawnSync(process.execPath, ["scripts/verify-onnx-container-source-pin.mjs"], {
  stdio: "inherit",
});
if (onnxContainerSourceVerification.error) throw onnxContainerSourceVerification.error;
if (onnxContainerSourceVerification.status !== 0) process.exit(onnxContainerSourceVerification.status || 1);

const onnxControlFlowSourceVerification = spawnSync(process.execPath, ["scripts/verify-onnx-controlflow-source-pin.mjs"], {
  stdio: "inherit",
});
if (onnxControlFlowSourceVerification.error) throw onnxControlFlowSourceVerification.error;
if (onnxControlFlowSourceVerification.status !== 0) process.exit(onnxControlFlowSourceVerification.status || 1);

const onnxMlValueSourceVerification = spawnSync(process.execPath, ["scripts/verify-onnx-ml-value-source-pin.mjs"], {
  stdio: "inherit",
});
if (onnxMlValueSourceVerification.error) throw onnxMlValueSourceVerification.error;
if (onnxMlValueSourceVerification.status !== 0) process.exit(onnxMlValueSourceVerification.status || 1);

const onnxTfIdfSourceVerification = spawnSync(process.execPath, ["scripts/verify-onnx-tfidf-source-pin.mjs"], {
  stdio: "inherit",
});
if (onnxTfIdfSourceVerification.error) throw onnxTfIdfSourceVerification.error;
if (onnxTfIdfSourceVerification.status !== 0) process.exit(onnxTfIdfSourceVerification.status || 1);

const runtimeInfoPinVerification = spawnSync(process.execPath, ["scripts/verify-tflite-runtime-info-pin.mjs"], {
  stdio: "inherit",
});
if (runtimeInfoPinVerification.error) throw runtimeInfoPinVerification.error;
if (runtimeInfoPinVerification.status !== 0) process.exit(runtimeInfoPinVerification.status || 1);

const npmCli = process.env.npm_execpath;
if (!npmCli) throw new Error("npm_execpath is unavailable; run this release wrapper through `npm run build:release`.");
const generatedArtifactSnapshot = new Map(RELEASE_GENERATED_TRACKED_ARTIFACTS.map((file) => [file, readFileSync(file)]));
const sourceMetadataPath = "web/lib/build-metadata.js";
const sourceMetadataSnapshot = existsSync(sourceMetadataPath) ? readFileSync(sourceMetadataPath) : null;
let result;
try {
  result = spawnSync(process.execPath, [npmCli, "run", "build:worker"], {
    stdio: "inherit",
    env: {
      ...process.env,
      DEEPBOM_RELEASE_BUILD: "1",
    },
  });
} finally {
  // dist already contains the release bytes. Restore tracked generated files so
  // a formal build is repeatable and does not turn its clean source into a dirty tree.
  for (const [file, bytes] of generatedArtifactSnapshot) writeFileSync(file, bytes);
  if (sourceMetadataSnapshot) writeFileSync(sourceMetadataPath, sourceMetadataSnapshot);
  else rmSync(sourceMetadataPath, { force: true });
}

if (result?.error) throw result.error;
if (result?.status !== 0) process.exit(result?.status || 1);
