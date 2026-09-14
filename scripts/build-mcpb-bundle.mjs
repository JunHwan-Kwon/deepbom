import { createHash } from "node:crypto";
import { readFile, readdir, rm, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import AdmZip from "adm-zip";

const FIXED_ZIP_TIME = new Date(Date.UTC(1980, 0, 1, 0, 0, 0));

export async function buildMcpbBundle({ npmPackageRoot, outputRoot, version }) {
  const packageRoot = path.join(outputRoot, "package");
  await rm(outputRoot, { recursive: true, force: true });
  await mkdir(path.join(packageRoot, "server", "bin"), { recursive: true });
  await mkdir(path.join(packageRoot, "server", "pkg"), { recursive: true });

  for (const relative of [
    "bin/deepbom.mjs",
    "bin/deepbom-self-test.onnx",
    "pkg/release-manifest.json",
    "pkg/tflite_wasm_audit_bg.wasm",
  ]) {
    await writeFile(path.join(packageRoot, "server", relative), await readFile(path.join(npmPackageRoot, relative)));
  }
  await writeFile(path.join(packageRoot, "LICENSE"), await readFile(path.join(npmPackageRoot, "LICENSE")));
  const npmReadme = await readFile(path.join(npmPackageRoot, "README.md"), "utf8");
  await writeFile(path.join(packageRoot, "README.md"), `${npmReadme.trimEnd()}\n\n${mcpbPrivacySection()}\n`);

  const manifest = buildManifest(version);
  await writeFile(path.join(packageRoot, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);

  const bundlePath = path.join(outputRoot, `deepbom-${version}.mcpb`);
  const zip = new AdmZip();
  for (const file of await collectFiles(packageRoot)) {
    const relative = path.relative(packageRoot, file).replaceAll(path.sep, "/");
    zip.addFile(relative, await readFile(file));
    zip.getEntry(relative).header.time = FIXED_ZIP_TIME;
  }
  zip.writeZip(bundlePath);
  return {
    packageRoot,
    bundlePath,
    manifest,
    sha256: createHash("sha256").update(await readFile(bundlePath)).digest("hex"),
  };
}

function buildManifest(version) {
  return {
    $schema: "https://raw.githubusercontent.com/modelcontextprotocol/mcpb/main/schemas/mcpb-manifest-v0.4.schema.json",
    manifest_version: "0.4",
    name: "deepbom",
    display_name: "DEEPBOM Artifact Evidence",
    version,
    description: "Local static analysis and evidence generation for deployed AI model artifacts.",
    long_description: "Analyze supported model artifacts on the local machine. Artifact bytes remain local. Results distinguish observed facts, deterministic derivations, predictions, and evidence gaps.",
    author: { name: "Jun-Hwan Kwon", url: "https://github.com/JunHwan-Kwon" },
    repository: { type: "git", url: "https://github.com/JunHwan-Kwon/deepbom.git" },
    homepage: "https://deepbom.org/for-agents/",
    documentation: "https://deepbom.org/for-agents/",
    support: "https://github.com/JunHwan-Kwon/deepbom/issues",
    license: "Apache-2.0",
    privacy_policies: ["https://deepbom.org/privacy"],
    server: {
      type: "node",
      entry_point: "server/bin/deepbom.mjs",
      mcp_config: {
        command: "node",
        args: ["${__dirname}/server/bin/deepbom.mjs", "mcp"],
        env: { DEEPBOM_MCP_ALLOWED_ROOTS: "${user_config.allowed_root}" },
      },
    },
    user_config: {
      allowed_root: {
        type: "directory",
        title: "Allowed model directory",
        description: "DEEPBOM may read model artifacts only under this directory.",
        required: true,
        default: "${HOME}",
      },
    },
    tools: [
      { name: "deepbom_capabilities", description: "Inspect supported formats, targets, outputs, and evidence boundaries." },
      { name: "deepbom_audit", description: "Statically audit one local deployed-model artifact." },
      { name: "deepbom_diff", description: "Compare two artifacts of the same supported format." },
      { name: "deepbom_explain_rule", description: "Explain a finding or analysis rule without guessing its meaning." },
    ],
    tools_generated: false,
    keywords: ["model-context-protocol", "static-analysis", "tflite", "onnx", "gguf", "ml-bom"],
    compatibility: {
      claude_desktop: ">=1.0.0",
      platforms: ["darwin", "win32"],
      runtimes: { node: ">=20" },
    },
  };
}

function mcpbPrivacySection() {
  return [
    "## Privacy Policy",
    "",
    "This desktop extension runs DEEPBOM on the user's computer. It reads only",
    "artifact paths under the directory selected as `allowed_root`. Local artifact",
    "bytes, filenames, tensor payloads, and analysis results are not uploaded by",
    "the extension. A user may deliberately provide an immutable remote source to",
    "a tool; in that case DEEPBOM downloads it into a content-addressed local cache",
    "under an allowed root and does not send local artifacts to that source.",
    "",
    "The extension does not collect telemetry, share artifact data with third",
    "parties, or retain data on a DEEPBOM service. Local files and caches remain",
    "under the user's control and follow the user's own deletion and retention",
    "practices. Tool results are returned to Claude and are then handled under the",
    "user's Claude account and organization policies.",
    "",
    "Full policy and contact information: https://deepbom.org/privacy",
  ].join("\n");
}

async function collectFiles(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await collectFiles(file));
    else if (entry.isFile()) files.push(file);
  }
  return files.sort();
}
