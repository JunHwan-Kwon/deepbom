import { collectWeightAnalysis, compareWeightAnalyses } from "../lib/weight-analysis.js";
import { readWeightModelFile } from "../lib/numerical-ir/weight-browser-input.js";
self.onmessage = async event => {
  try {
    const { model, analysis, source, options={}, baseline=null, mapping=[] } = event.data;
    const onProgress=progress=>self.postMessage({progress});
    const current=await collectWeightAnalysis(model,analysis,source,{options,onProgress});
    let comparison=null;
    if(baseline) {
      const context=await readWeightModelFile(baseline);
      const prior=await collectWeightAnalysis(context.model,context.analysis,context.source,{options:{...options,tensor_ids:[],axes:{}},onProgress});
      comparison=compareWeightAnalyses(prior,current,{mapping});
    }
    self.postMessage({ result:current.weight_ir, advanced:current.weight_analysis, comparison });
  } catch (error) { self.postMessage({ error: String(error?.message || error) }); }
};
