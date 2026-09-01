import { readFile } from "node:fs/promises";
import path from "node:path";

const RELEASE_VERSION = /^\d+\.\d+\.\d+$/;
const CHANNELS = new Set(["dev", "release"]);

export async function readVersionContract(root, { releaseVersion = process.env.DEEPBOM_RELEASE_VERSION || "" } = {}) {
  const sourcePath = path.join(root, "release", "version.json");
  const source = JSON.parse(await readFile(sourcePath, "utf8"));
  if (source?.schema !== "deepbom.release_version.v1") throw new Error("Unsupported release/version.json schema.");
  if (!RELEASE_VERSION.test(String(source.base_version || ""))) throw new Error("release/version.json base_version must be x.y.z.");
  if (!CHANNELS.has(source.channel)) throw new Error("release/version.json channel must be dev or release.");
  if (releaseVersion && !RELEASE_VERSION.test(releaseVersion)) throw new Error("The release override must be an exact x.y.z version.");
  if (releaseVersion && releaseVersion !== source.base_version) {
    throw new Error(`Release override ${releaseVersion} does not match canonical base ${source.base_version}.`);
  }
  const release = Boolean(releaseVersion) || source.channel === "release";
  const displayVersion = release ? source.base_version : `${source.base_version}-dev`;
  return Object.freeze({
    sourcePath,
    baseVersion: source.base_version,
    channel: release ? "release" : "dev",
    displayVersion,
    npmVersion: displayVersion,
    cargoVersion: displayVersion,
    pythonVersion: release ? source.base_version : `${source.base_version}.dev0`,
    publishable: release,
  });
}

export function assertPublishableVersion(contract) {
  if (!contract?.publishable || contract.channel !== "release") {
    throw new Error(`Refusing to publish development version ${contract?.displayVersion || "unknown"}.`);
  }
}
