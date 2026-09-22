import { detectModelFormat } from "../model-file.js";
import { getArtifactIrContext } from "../artifact-ir-context.js";
import { hashSource, readBytes, MAX_IN_MEMORY_SOURCE } from "./weight-sources.js";
import { requireCondition } from "./common.js";

export async function readWeightModelFile(file) {
  const filename=file.name || "model",format=detectModelFormat(filename,await readBytes(file,0,Math.min(file.size,4096)));
  let analysis;
  if(["gguf","safetensors"].includes(format)) {
    const {readMetadataModelFile}=await import("../metadata-model-adapters.js");analysis=(await readMetadataModelFile(file,format,{scanMode:"structure"})).analysis;
  } else if(format==="coreml") {
    const {readCoreMlModelFile}=await import("../coreml-metadata-adapter.js");analysis=(await readCoreMlModelFile(file)).analysis;
  } else {
    requireCondition(file.size<=MAX_IN_MEMORY_SOURCE,"baseline requires too much browser memory");
    const bytes=await readBytes(file);
    if(format==="onnx") {const {analyzeOnnxModel}=await import("../../onnx.js");analysis=analyzeOnnxModel(bytes,filename);}
    else if(format==="executorch") {const {analyzeExecuTorchModel}=await import("../../executorch.js");analysis=analyzeExecuTorchModel(bytes,filename);}
    else if(format==="tflite") {
      const wasm=await import("../../../pkg/tflite_wasm_audit.js");await wasm.default({module_or_path:new URL("../../../pkg/tflite_wasm_audit_bg.wasm",import.meta.url)});analysis=wasm.analyze_tflite(bytes,filename);
    } else throw new Error("Unsupported numerical baseline format: "+format);
  }
  const sha256=await hashSource(file);analysis.model_sha256=sha256;
  const model=getArtifactIrContext(analysis,{filename,format,sha256,size:file.size}).model_ir;
  return {analysis,model,source:file};
}
