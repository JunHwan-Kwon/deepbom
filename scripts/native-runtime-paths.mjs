import os from "node:os";
import path from "node:path";
import process from "node:process";

export function defaultNativeRuntimeCacheDir() {
  if (process.platform === "win32" && process.env.LOCALAPPDATA) {
    return path.join(process.env.LOCALAPPDATA, "DeepBOM", "native-runtime");
  }
  const cacheRoot = process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache");
  return path.join(cacheRoot, "deepbom", "native-runtime");
}
