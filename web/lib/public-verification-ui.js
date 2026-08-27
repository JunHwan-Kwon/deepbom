export function syncPublicVerificationButton(button, { hasAnalysis, reportTargetReady } = {}) {
  if (!button) return;
  button.disabled = !hasAnalysis || !reportTargetReady;
  button.title = reportTargetReady
    ? "Download a login-free detached manifest binding report, artifact, analyzer, target, and runtime identity hashes."
    : "Run or load the selected report target analysis first.";
}

export function bindPublicVerificationButton(button, {
  getAnalysis,
  getBinding,
  bindingMatches,
  ensureHash,
  getContext,
  getScope,
  getRuntimeEvidence,
  origin,
  filename,
  download,
  serialize,
  setStatus,
} = {}) {
  button?.addEventListener("click", async () => {
    try {
      const analysis = getAnalysis();
      const binding = getBinding();
      if (!analysis || !binding?.canCopy || !bindingMatches(binding, analysis)) return;
      await ensureHash();
      const { buildPublicVerificationManifest, validatePublicVerificationManifest } = await import("./public-verification-manifest.js");
      const manifest = buildPublicVerificationManifest({
        analysis,
        context: getContext(),
        scope: getScope(analysis),
        runtimeEvidence: getRuntimeEvidence(),
        origin,
      });
      if (!validatePublicVerificationManifest(manifest)) throw new Error("Generated public verification manifest failed self-validation.");
      download(filename(), serialize(manifest), "application/json");
      setStatus("Public verification manifest downloaded", "ok");
    } catch (error) {
      setStatus(error?.message || String(error), "error");
    }
  });
}
