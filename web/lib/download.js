import { safeStem } from "./format.js";

export function artifactFilename(baseFilename, suffix) {
  return `${safeStem(baseFilename || "model")}_${suffix}`;
}

export function downloadText(filename, text, type) {
  const blob = new Blob([text], { type });
  downloadBlob(filename, blob);
}

export async function downloadTextArtifact({
  artifact,
  buildText,
  getFilename,
  isReady = () => true,
  ensureAllowed = async () => true,
  ensureHash = async () => {},
}) {
  if (!artifact || !buildText || !getFilename || !isReady(artifact)) return false;
  if (!(await ensureAllowed(artifact))) return false;
  if (artifact.ensureHash) await ensureHash();
  downloadText(getFilename(artifact.suffix), await buildText(), artifact.type);
  return true;
}

export function registerTextExport(button, artifact, buildText, options = {}) {
  if (!button) return;
  button.addEventListener("click", async () => {
    await downloadTextArtifact({ artifact, buildText, ...options });
  });
}

export function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
