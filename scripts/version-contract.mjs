import { readFile } from "node:fs/promises";
import path from "node:path";

const BASE_VERSION = /^\d+\.\d+\.\d+$/;
const PUBLISHED_VERSION = /^(\d+\.\d+\.\d+)(?:-(alpha|beta|rc)\.(\d+))?$/;
const PRERELEASE_IDENTIFIER = /^(alpha|beta|rc)\.(\d+)$/;
const CHANNELS = new Set(["dev", "prerelease", "release"]);

export async function readVersionContract(root, { releaseVersion = process.env.DEEPBOM_RELEASE_VERSION || "" } = {}) {
  const sourcePath = path.join(root, "release", "version.json");
  const source = JSON.parse(await readFile(sourcePath, "utf8"));
  return resolveVersionContract(source, { sourcePath, releaseVersion });
}

export function resolveVersionContract(source, { sourcePath = "release/version.json", releaseVersion = "" } = {}) {
  if (source?.schema !== "deepbom.release_version.v1") throw new Error("Unsupported release/version.json schema.");
  if (!BASE_VERSION.test(String(source.base_version || ""))) throw new Error("release/version.json base_version must be x.y.z.");
  if (!CHANNELS.has(source.channel)) throw new Error("release/version.json channel must be dev, prerelease, or release.");
  const sourcePrerelease = String(source.prerelease || "");
  if (source.channel === "prerelease" && !PRERELEASE_IDENTIFIER.test(sourcePrerelease)) {
    throw new Error("release/version.json prerelease channel requires prerelease=(alpha|beta|rc).N.");
  }
  if (source.channel !== "prerelease" && sourcePrerelease) throw new Error("Only the prerelease channel may declare prerelease.");

  const overrideMatch = releaseVersion ? PUBLISHED_VERSION.exec(releaseVersion) : null;
  if (releaseVersion && !overrideMatch) throw new Error("The release override must be x.y.z or x.y.z-(alpha|beta|rc).N.");
  if (overrideMatch && overrideMatch[1] !== source.base_version) {
    throw new Error(`Release override ${releaseVersion} does not match canonical base ${source.base_version}.`);
  }
  const overridePrerelease = overrideMatch?.[2] ? `${overrideMatch[2]}.${overrideMatch[3]}` : "";
  const channel = overrideMatch ? (overridePrerelease ? "prerelease" : "release") : source.channel;
  const prerelease = overrideMatch ? overridePrerelease : sourcePrerelease;
  if (channel === "prerelease" && !prerelease) throw new Error("Prerelease publication requires an explicit identifier.");
  const displayVersion = channel === "release"
    ? source.base_version
    : channel === "prerelease"
      ? `${source.base_version}-${prerelease}`
      : `${source.base_version}-dev`;
  if (releaseVersion && displayVersion !== releaseVersion) throw new Error(`Release override ${releaseVersion} disagrees with resolved version ${displayVersion}.`);
  return Object.freeze({
    sourcePath,
    baseVersion: source.base_version,
    channel,
    prerelease: prerelease || null,
    displayVersion,
    npmVersion: displayVersion,
    cargoVersion: displayVersion,
    pythonVersion: channel === "release"
      ? source.base_version
      : channel === "prerelease"
        ? `${source.base_version}${pythonPrerelease(prerelease)}`
        : `${source.base_version}.dev0`,
    npmDistTag: channel === "prerelease" ? "next" : channel === "release" ? "latest" : null,
    publishable: channel !== "dev",
  });
}

export function assertPublishableVersion(contract) {
  if (!contract?.publishable || !["release", "prerelease"].includes(contract.channel)) {
    throw new Error(`Refusing to publish development version ${contract?.displayVersion || "unknown"}.`);
  }
}

function pythonPrerelease(identifier) {
  const match = PRERELEASE_IDENTIFIER.exec(identifier);
  if (!match) throw new Error(`Invalid prerelease identifier ${identifier}.`);
  return `${match[1] === "alpha" ? "a" : match[1] === "beta" ? "b" : "rc"}${match[2]}`;
}
