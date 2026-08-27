import { validateKernelWitnessDigests } from "./kernel-witness.js";

if (typeof self !== "undefined") {
  self.onmessage = async (event) => {
    try {
      const { analysis, modelBytes } = event.data || {};
      await validateKernelWitnessDigests(analysis, modelBytes);
      self.postMessage({ ok: true });
    } catch (error) {
      self.postMessage({ ok: false, error: error?.message || String(error) });
    }
  };
}
