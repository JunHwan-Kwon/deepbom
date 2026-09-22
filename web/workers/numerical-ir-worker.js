import { buildWeightIr } from "../lib/weight-ir.js";
self.onmessage = async event => {
  try {
    const { model, analysis, source } = event.data;
    const result = await buildWeightIr(model, analysis, source, { onProgress: progress => self.postMessage({ progress }) });
    self.postMessage({ result });
  } catch (error) { self.postMessage({ error: String(error?.message || error) }); }
};
