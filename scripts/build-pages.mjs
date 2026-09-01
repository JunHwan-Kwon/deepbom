import { copyFile, mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transform } from "esbuild";
import { readSwRuntimeCacheableSuffixes } from "./sw-utils.mjs";
import { hardenWasmFile } from "./wasm-binary-hardening.mjs";
import { writeBuildMetadata } from "./write-build-metadata.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
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
for (const file of deploymentExcludedWebFiles) {
  await rm(path.join(dist, "web", file), { force: true });
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
await writeFile(path.join(dist, "robots.txt"), [
  "User-agent: *",
  "Allow: /",
  "Disallow: /test",
  "",
  "Sitemap: https://deepbom.org/sitemap.xml",
].join("\n") + "\n");
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
  "    <loc>https://deepbom.org/verify</loc>",
  `    <lastmod>${today}</lastmod>`,
  "    <changefreq>monthly</changefreq>",
  "    <priority>0.6</priority>",
  "  </url>",
  "</urlset>",
].join("\n") + "\n");

const customDomain = process.env.CUSTOM_DOMAIN?.trim();
if (customDomain) {
  await writeFile(path.join(dist, "CNAME"), `${customDomain}\n`);
}

const deploymentHardening = await minifyProjectAssets(dist);
await writeFile(
  path.join(dist, "deployment-hardening.json"),
  `${JSON.stringify(deploymentHardening, null, 2)}\n`,
);

// Verify the release manifest while generated WASM bytes still match the
// source tree. The formal-build wrapper restores tracked artifacts afterward.
await import("./check-build-metadata.mjs?build-pages-verification");

console.log(
  `Static deploy artifact ready: ${dist} `
  + `(${deploymentHardening.javascript_files} project JS and ${deploymentHardening.css_files} CSS files minified; source maps disabled).`,
);

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
    security_boundary: "Copy-resistance only. Browser-delivered code remains inspectable; authoritative analysis and projections execute in stripped WASM.",
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
