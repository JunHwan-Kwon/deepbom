import { validateResidualStepResponseDigests } from "./residual-step-response.js";

if (typeof self !== "undefined") {
  self.onmessage = async (event) => {
    try {
      const { result, analysis } = event.data || {};
      await validateResidualStepResponseDigests(result, analysis);
      self.postMessage({ ok: true });
    } catch (error) {
      self.postMessage({ ok: false, error: error?.message || String(error) });
    }
  };
}
