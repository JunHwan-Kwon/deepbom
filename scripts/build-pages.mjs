import { readFileSync } from "node:fs";
import { copyFile, mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";
import { build, transform } from "esbuild";
import { readSwRuntimeCacheableSuffixes } from "./sw-utils.mjs";
import { hardenWasmFile } from "./wasm-binary-hardening.mjs";
import { writeBuildMetadata } from "./write-build-metadata.mjs";
import { buildCliCapabilities } from "../bin/deepbom-automation.mjs";
import { buildAgentCapabilities } from "../bin/deepbom-agent-contract.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
const packageDocument = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8"));
const protectedDeepBomPackage = ["web", "protected", "deepbom", "pkg"];

const pkgRuntimeFiles = [
  "tflite_wasm_audit.js",
  "tflite_wasm_audit_bg.wasm",
];
const deploymentExcludedWebFiles = [
  "samples/mobilenet_v1_025_224_float.tflite",
  "samples/sample_cnn_float.onnx",
  "protected/deepbom/pkg/deepbom_wasm.d.ts",
  "protected/deepbom/pkg/deepbom_wasm_bg.wasm.d.ts",
  "protected/deepbom/pkg/package.json",
];
const deploymentExcludedWebPatterns = [
  /^lib\/cyclonedx-(?:20|draft|perspective)(?:-|\.|$)/i,
  /^vendor\/jsonpath-rfc95\d+(?:\.|$)/i,
];

// Treat deploy assembly as the final WASM byte boundary. This second,
// idempotent pass prevents a late build-tool write from bypassing the earlier
// per-crate hardening step before metadata hashing and dist copying.
for (const relativePath of [
  "pkg/tflite_wasm_audit_bg.wasm",
  "web/protected/deepbom/pkg/deepbom_wasm_bg.wasm",
]) hardenWasmFile(path.join(root, relativePath));

const buildMetadata = writeBuildMetadata();

await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });
await rm(path.join(root, "pkg", ".gitignore"), { force: true });
await rm(path.join(root, ...protectedDeepBomPackage, ".gitignore"), { force: true });

await copyDir(path.join(root, "web"), path.join(dist, "web"));
await copyDir(path.join(root, "web", "evaluate"), path.join(dist, "evaluate"));
await copyDir(path.join(root, "web", "for-agents"), path.join(dist, "for-agents"));
await rm(path.join(dist, "web", "evaluate"), { recursive: true, force: true });
await rm(path.join(dist, "web", "for-agents"), { recursive: true, force: true });
for (const file of deploymentExcludedWebFiles) {
  await rm(path.join(dist, "web", file), { force: true });
}
for (const file of await collectFiles(path.join(dist, "web"))) {
  const relative = path.relative(path.join(dist, "web"), file).replaceAll(path.sep, "/");
  if (deploymentExcludedWebPatterns.some((pattern) => pattern.test(relative))) await rm(file, { force: true });
}

for (const file of pkgRuntimeFiles) {
  await copyProjectFile(path.join("pkg", file));
}
await stripTypeSelfReference(path.join(dist, "pkg", "tflite_wasm_audit.js"));
await stripTypeSelfReference(path.join(dist, ...protectedDeepBomPackage, "deepbom_wasm.js"));

for (const suffix of readSwRuntimeCacheableSuffixes(path.join(root, "web", "sw.js"))) {
  const runtimeFile = runtimeFileFromNodeModulesSuffix(suffix);
  await copyRuntimeFile(runtimeFile.packageName, runtimeFile.packageRelative);
}

for (const file of [
  "web/index.html",
  "web/medical.html",
  "web/app.js",
  "web/onnx.js",
  "web/sw.js",
  "web/lib/runtime-module-loader.js",
]) {
  await rewriteNodeModulesPath(path.join(dist, file));
}
const applicationBundle = await bundleApplicationEntry();
for (const file of applicationBundle.outputFiles) await rewriteNodeModulesPath(file);
await appendServiceWorkerAssets(
  path.join(dist, "web", "sw.js"),
  applicationBundle.outputFiles.map((file) => `./${path.relative(path.join(dist, "web"), file).replaceAll(path.sep, "/")}`),
);
await stampServiceWorkerBuild(path.join(dist, "web", "sw.js"), buildMetadata.bundleContentSha256);

await writeFile(path.join(dist, ".nojekyll"), "");
await mkdir(path.join(dist, ".well-known"), { recursive: true });
await copyFile(
  path.join(root, "web", ".well-known", "deepbom-signing-keys.json"),
  path.join(dist, ".well-known", "deepbom-signing-keys.json"),
);
await mkdir(path.join(dist, "schemas"), { recursive: true });
await copyFile(
  path.join(root, "docs", "schemas", "deepbom-artifact-ir-v2.schema.json"),
  path.join(dist, "schemas", "deepbom-artifact-ir-v2.schema.json"),
);
const webIndexHtml = await readFile(path.join(dist, "web", "index.html"), "utf8");
await writeFile(path.join(dist, "index.html"), shellHtml(webIndexHtml));
await writeFile(path.join(dist, "medical.html"), shellHtml(webIndexHtml, "Medical Evidence Workspace / DEEPBOM"));

// Standalone report-verification page served at the clean URL /verify
await writeFile(path.join(dist, "verify.html"), await readFile(path.join(dist, "web", "verify.html"), "utf8"));

// Private-link gateway served at /test. It activates an expiring access grant,
// then redirects into the single canonical Workbench instead of cloning it.
await writeFile(path.join(dist, "test.html"), await readFile(path.join(dist, "web", "test.html"), "utf8"));

// SEO: robots.txt and sitemap.xml at domain root
const today = new Date().toISOString().slice(0, 10);
const agentCapabilities = buildAgentCapabilities(buildCliCapabilities(packageDocument.version, {
  defaultTarget: "android_mid_a55",
  deltaTargets: ["android_mid_a55", "rpi4_a72", "x86_avx2", "wasm_simd"],
}));
await writeFile(path.join(dist, "agent-capabilities.json"), `${JSON.stringify(agentCapabilities, null, 2)}\n`);
await writeFile(path.join(dist, "robots.txt"), [
  "User-agent: OAI-SearchBot",
  "Allow: /",
  "Disallow: /test",
  "",
  "User-agent: *",
  "Allow: /",
  "Disallow: /test",
  "",
  "Sitemap: https://deepbom.org/sitemap.xml",
  "",
  "# Plain-language entry points and the local-analysis boundary for assistants.",
  "# https://deepbom.org/llms.txt",
].join("\n") + "\n");
// An assistant that reaches this domain can read the linked data, but nothing
// tells it that the analysis runs on the reader's own machine and never in a
// hosted endpoint. State the entry points and that boundary in plain text.
await writeFile(path.join(dist, "llms.txt"), [
  "# DEEPBOM",
  "",
  "> Deployment artifact evidence for on-device AI. DEEPBOM identifies a model",
  "> artifact by hash, records what its serialized contracts actually state, and",
  "> keeps static derivations, target assumptions, and imported runtime",
  "> observations separate so each claim carries its own evidence class.",
  "",
  "Analysis runs locally, in the reader's browser or CLI process. Artifact bytes",
  "are not uploaded, and there is no hosted analysis endpoint to call. To analyse",
  "a model on behalf of a user, run the CLI on their machine.",
  "",
  "## Command line",
  "",
  "```",
  `npx -y deepbom@${packageDocument.version} capabilities --format agent-json`,
  `npx -y deepbom@${packageDocument.version} self-test --compact`,
  `npx -y deepbom@${packageDocument.version} audit model.tflite --summary`,
  `npx -y deepbom@${packageDocument.version} explain-rule <rule-id> --json`,
  "```",
  "",
  `For tool-call access instead of shell invocation, \`npx -y deepbom@${packageDocument.version} mcp\` speaks the`,
  "Model Context Protocol over stdio and exposes `deepbom_capabilities`,",
  "`deepbom_audit`, `deepbom_diff`, and `deepbom_explain_rule`. It runs locally",
  "on the same terms. A plain chat without shell or local MCP access cannot run",
  "the analysis; it can only provide the pinned command.",
  "",
  "Formats: .tflite, .onnx, .gguf, .safetensors, .mlmodel, .pte, .ptd.",
  "Outputs: analysis JSON, evidence envelope, CycloneDX 1.7, SARIF 2.1.0.",
  "Exit codes: 0 pass, 1 invocation or analysis failure, 2 policy or verification",
  "block, 3 incomplete verification binding.",
  "",
  "## Reading a result",
  "",
  "Findings carry three distinct kinds and must not be merged. `artifact_defect`",
  "is a problem in the artifact. `caution` is something a reviewer should look",
  "at. `evidence_gap` is a claim the artifact cannot settle on its own, such as",
  "runtime placement or measured latency; it is not a defect, and closing it",
  "needs an imported runtime, lineage, or build-evidence document. The browser",
  "review summary reports the same group under the field name",
  "`evidence_needed`. Static output never establishes executed accelerator",
  "assignment, latency, energy, accuracy, or device fit.",
  "",
  "## Pages",
  "",
  "- [Workspace](https://deepbom.org/): browser-local audit of one artifact",
  "- [AI agent setup](https://deepbom.org/for-agents/): local Agent Skill, npx, and stdio MCP paths",
  "- [Regulatory brief](https://deepbom.org/evaluate/regulatory/): what the records can support in a controlled process, and where they stop",
  "- [Quality brief](https://deepbom.org/evaluate/quality/): validating an installed analyzer before relying on it",
  "- [Engineering brief](https://deepbom.org/evaluate/engineering/): architecture, CI entry point, and what it will not infer",
  "- [Report verifier](https://deepbom.org/verify): check a generated report against its artifact",
  "",
  "## Project",
  "",
  "- Source: https://github.com/JunHwan-Kwon/deepbom (Apache-2.0)",
  "- Record: https://doi.org/10.5281/zenodo.21834508",
  "- Author: Jun-Hwan Kwon, Ph.D. (ORCID 0000-0002-6464-3895)",
  "- Agent skill file: https://github.com/JunHwan-Kwon/deepbom/blob/main/skills/deepbom/SKILL.md",
  `- Claude Desktop extension: https://github.com/JunHwan-Kwon/deepbom/releases/download/channels-v${packageDocument.version}/deepbom-${packageDocument.version}.mcpb`,
  "",
].join("\n"));
await writeFile(path.join(dist, "sitemap.xml"), [
  '<?xml version="1.0" encoding="UTF-8"?>',
  '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
  "  <url>",
  "    <loc>https://deepbom.org/</loc>",
  `    <lastmod>${today}</lastmod>`,
  "    <changefreq>weekly</changefreq>",
  "    <priority>1.0</priority>",
  "  </url>",
  "  <url>",
  "    <loc>https://deepbom.org/for-agents/</loc>",
  `    <lastmod>${today}</lastmod>`,
  "    <changefreq>monthly</changefreq>",
  "    <priority>0.7</priority>",
  "  </url>",
  "  <url>",
  "    <loc>https://deepbom.org/verify</loc>",
  `    <lastmod>${today}</lastmod>`,
  "    <changefreq>monthly</changefreq>",
  "    <priority>0.6</priority>",
  "  </url>",
  ...["regulatory", "quality", "engineering"].flatMap((brief) => [
    "  <url>",
    `    <loc>https://deepbom.org/evaluate/${brief}/</loc>`,
    `    <lastmod>${today}</lastmod>`,
    "    <changefreq>monthly</changefreq>",
    "    <priority>0.7</priority>",
    "  </url>",
  ]),
  "</urlset>",
].join("\n") + "\n");

const customDomain = process.env.CUSTOM_DOMAIN?.trim();
if (customDomain) {
  await writeFile(path.join(dist, "CNAME"), `${customDomain}\n`);
}

const deploymentHardening = await minifyProjectAssets(dist);
const frontendDelivery = await measureFrontendDelivery();
await writeFile(
  path.join(dist, "deployment-hardening.json"),
  `${JSON.stringify(deploymentHardening, null, 2)}\n`,
);
await writeFile(
  path.join(dist, "frontend-delivery.json"),
  `${JSON.stringify(frontendDelivery, null, 2)}\n`,
);

// Verify the release manifest while generated WASM bytes still match the
// source tree. The formal-build wrapper restores tracked artifacts afterward.
await import("./check-build-metadata.mjs?build-pages-verification");

console.log(
  `Static deploy artifact ready: ${dist} `
  + `(${deploymentHardening.javascript_files} project JS and ${deploymentHardening.css_files} CSS files minified; `
  + `${frontendDelivery.pre_interaction.javascript_requests} pre-interaction JS requests / ${frontendDelivery.pre_interaction.gzip_bytes} gzip bytes).`,
);

async function bundleApplicationEntry() {
  const result = await build({
    absWorkingDir: root,
    entryPoints: ["web/app.js"],
    outdir: path.join(dist, "web"),
    outbase: "web",
    entryNames: "[name].bundle",
    chunkNames: "chunks/[name]-[hash]",
    assetNames: "chunks/[name]-[hash]",
    bundle: true,
    charset: "ascii",
    external: [
      "../pkg/*",
      "@litertjs/*",
      "onnxruntime-web/*",
    ],
    format: "esm",
    legalComments: "none",
    metafile: true,
    minify: true,
    platform: "browser",
    sourcemap: false,
    splitting: true,
    target: "es2022",
    write: true,
  });
  const outputFiles = Object.keys(result.metafile.outputs)
    .filter((file) => file.endsWith(".js"))
    .map((file) => path.resolve(root, file))
    .sort();
  const entry = path.join(dist, "web", "app.bundle.js");
  if (!outputFiles.includes(entry)) throw new Error("esbuild did not emit dist/web/app.bundle.js.");
  await writeFile(path.join(dist, "web", "app.js"), 'import "./app.bundle.js";\n');
  return { outputFiles };
}

async function appendServiceWorkerAssets(file, assets) {
  const source = await readFile(file, "utf8");
  const unique = [...new Set(assets)].sort();
  const marker = "const APP_ASSETS = [";
  if (!source.includes(marker)) throw new Error("Service worker APP_ASSETS declaration is missing.");
  const injected = source.replace(marker, `${marker}\n${unique.map((asset) => `  ${JSON.stringify(asset)},`).join("\n")}`);
  await writeFile(file, injected);
}

async function measureFrontendDelivery() {
  const html = await readFile(path.join(dist, "web", "index.html"), "utf8");
  const scriptPaths = [...html.matchAll(/<script\b[^>]*\bsrc="(\.\/[^\"]+)"[^>]*><\/script>/g)]
    .map((match) => match[1]);
  const uniqueScripts = [...new Set(scriptPaths)];
  const gzipBytes = uniqueScripts.reduce((sum, relative) => {
    const bytes = readFileSync(path.join(dist, "web", relative.slice(2)));
    return sum + gzipSync(bytes, { level: 9 }).byteLength;
  }, 0);
  const javascriptRequests = uniqueScripts.length;
  const gzipBudgetBytes = 600 * 1024;
  const requestBudget = 25;
  if (gzipBytes > gzipBudgetBytes) throw new Error(`Pre-interaction JS gzip budget exceeded: ${gzipBytes} > ${gzipBudgetBytes}.`);
  if (javascriptRequests > requestBudget) throw new Error(`Pre-interaction JS request budget exceeded: ${javascriptRequests} > ${requestBudget}.`);
  return {
    schema: "deepbom.frontend_delivery.v1",
    budget_scope: "document scripts before artifact or tool interaction",
    pre_interaction: {
      javascript_requests: javascriptRequests,
      gzip_bytes: gzipBytes,
      javascript_request_budget: requestBudget,
      gzip_byte_budget: gzipBudgetBytes,
      scripts: uniqueScripts,
    },
    application_entry: "web/app.js -> web/app.bundle.js",
    application_load_trigger: "artifact, verified sample, or analysis-tool interaction",
    tflite_wasm_preloaded: false,
  };
}

async function copyDir(source, target) {
  await mkdir(target, { recursive: true });
  const entries = await readdir(source, { withFileTypes: true });
  for (const entry of entries) {
    const sourcePath = path.join(source, entry.name);
    const targetPath = path.join(target, entry.name);
    if (entry.isDirectory()) {
      await copyDir(sourcePath, targetPath);
    } else if (entry.isFile()) {
      await copyFile(sourcePath, targetPath);
    }
  }
}

async function copyRuntimeFile(packageName, packageRelative) {
  const source = path.join(root, "node_modules", packageName, packageRelative);
  const fileStat = await stat(source);
  if (!fileStat.isFile()) {
    throw new Error(`Expected runtime file: ${source}`);
  }
  const target = path.join(dist, "vendor", packageName, packageRelative);
  await mkdir(path.dirname(target), { recursive: true });
  await copyFile(source, target);
}

async function copyProjectFile(relativePath) {
  const source = path.join(root, relativePath);
  const target = path.join(dist, relativePath);
  await mkdir(path.dirname(target), { recursive: true });
  await copyFile(source, target);
}

async function rewriteNodeModulesPath(file) {
  const source = await readFile(file, "utf8");
  const rewritten = source
    .replaceAll("../node_modules/@litertjs/", "../vendor/@litertjs/")
    .replaceAll("/node_modules/@litertjs/", "/vendor/@litertjs/")
    .replaceAll("../node_modules/onnxruntime-web/", "../vendor/onnxruntime-web/")
    .replaceAll("/node_modules/onnxruntime-web/", "/vendor/onnxruntime-web/");
  await writeFile(file, rewritten);
}

async function stripTypeSelfReference(file) {
  const source = await readFile(file, "utf8");
  const rewritten = source.replace(/^\/\* @ts-self-types=.*?\*\/\r?\n/, "");
  await writeFile(file, rewritten);
}

async function minifyProjectAssets(rootDir) {
  const files = await collectFiles(rootDir);
  const javascript = files.filter((file) => path.extname(file).toLowerCase() === ".js"
    && !file.includes(`${path.sep}vendor${path.sep}`)
    && path.basename(file) !== "sw.js");
  const css = files.filter((file) => path.extname(file).toLowerCase() === ".css"
    && !file.includes(`${path.sep}vendor${path.sep}`));
  let sourceBytes = 0;
  let outputBytes = 0;
  for (const file of javascript) {
    const source = await readFile(file, "utf8");
    sourceBytes += Buffer.byteLength(source);
    const result = await transform(source, {
      charset: "ascii",
      format: "esm",
      legalComments: "none",
      loader: "js",
      minify: true,
      minifyIdentifiers: true,
      minifySyntax: true,
      minifyWhitespace: true,
      sourcefile: path.relative(rootDir, file).replaceAll(path.sep, "/"),
      sourcemap: false,
      target: "es2022",
    });
    const output = stripSourceMapReference(result.code);
    outputBytes += Buffer.byteLength(output);
    await writeFile(file, output);
  }
  for (const file of css) {
    const source = await readFile(file, "utf8");
    sourceBytes += Buffer.byteLength(source);
    const result = await transform(source, {
      charset: "ascii",
      legalComments: "none",
      loader: "css",
      minify: true,
      sourcefile: path.relative(rootDir, file).replaceAll(path.sep, "/"),
      sourcemap: false,
      target: "es2022",
    });
    const output = stripSourceMapReference(result.code);
    outputBytes += Buffer.byteLength(output);
    await writeFile(file, output);
  }
  const transformed = new Set([...javascript, ...css]);
  for (const file of files.filter((item) => [".js", ".css"].includes(path.extname(item).toLowerCase()))) {
    if (transformed.has(file)) continue;
    const source = await readFile(file, "utf8");
    const output = stripSourceMapReference(source);
    if (output !== source) await writeFile(file, output);
  }
  return {
    schema: "deepbom.deployment_hardening.v1",
    javascript_files: javascript.length,
    css_files: css.length,
    source_maps: "forbidden",
    javascript_transform: "esbuild minify syntax, whitespace, and identifiers; ESM exports preserved",
    css_transform: "esbuild minify",
    source_bytes: sourceBytes,
    output_bytes: outputBytes,
    byte_reduction_ratio: sourceBytes > 0 ? Number((1 - outputBytes / sourceBytes).toFixed(6)) : 0,
    security_boundary: "Copy-resistance only. Browser-delivered code remains inspectable. TFLite analysis and selected protected projections execute in stripped WASM; ONNX, ExecuTorch, Core ML, GGUF, and SafeTensors analysis uses minified browser modules with isolated Worker execution where payload processing is heavy.",
  };
}

async function collectFiles(rootDir) {
  const files = [];
  for (const entry of await readdir(rootDir, { withFileTypes: true })) {
    const file = path.join(rootDir, entry.name);
    if (entry.isDirectory()) files.push(...await collectFiles(file));
    else if (entry.isFile()) files.push(file);
  }
  return files;
}

function stripSourceMapReference(source) {
  return source
    .replace(/\/\/[#@]\s*sourceMappingURL=.*?(?:\r?\n|$)/g, "")
    .replace(/\/\*[#@]\s*sourceMappingURL=.*?\*\//g, "");
}

async function stampServiceWorkerBuild(file, buildContentSha256) {
  const source = await readFile(file, "utf8");
  const suffix = String(buildContentSha256 || "").slice(0, 16);
  if (!suffix) throw new Error("Build-content SHA-256 is required to stamp the service-worker cache.");
  const rewritten = source.replace(
    /const CACHE_NAME = "([^"]+)";/,
    (_match, cacheName) => `const CACHE_NAME = "${cacheName}-${suffix}";`,
  );
  if (rewritten === source) throw new Error("Could not stamp service-worker CACHE_NAME.");
  await writeFile(file, rewritten);
}

function shellHtml(indexHtml, title = "DEEPBOM | Deployment Artifact Evidence for On-Device AI") {
  const withBase = indexHtml.includes("<base ")
    ? indexHtml
    : indexHtml.replace("<head>", '<head>\n    <base href="/web/" />');
  return withBase.replace(/<title>[^<]*<\/title>/, `<title>${title}</title>`);
}

function runtimeFileFromNodeModulesSuffix(suffix) {
  const prefix = "/node_modules/";
  if (!suffix.startsWith(prefix)) {
    throw new Error(`Unsupported runtime cache suffix: ${suffix}`);
  }

  const parts = suffix.slice(prefix.length).split("/");
  const packageName = parts[0].startsWith("@") ? parts.slice(0, 2).join("/") : parts[0];
  const packageRelative = parts.slice(packageName.startsWith("@") ? 2 : 1).join("/");

  if (!packageName || !packageRelative) {
    throw new Error(`Invalid runtime cache suffix: ${suffix}`);
  }

  return { packageName, packageRelative };
}
