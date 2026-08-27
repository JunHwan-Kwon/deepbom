import { validateRoundingEquivalenceDigests } from "./rounding-equivalence.js";

if (typeof self !== "undefined") {
  self.onmessage = async (event) => {
    try {
      const { analysis, modelBytes } = event.data || {};
      await validateRoundingEquivalenceDigests(analysis, modelBytes);
      self.postMessage({ ok: true });
    } catch (error) {
      self.postMessage({ ok: false, error: error?.message || String(error) });
    }
  };
}
