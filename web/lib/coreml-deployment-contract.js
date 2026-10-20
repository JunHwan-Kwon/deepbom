import { portableContract } from "./portable-contracts.js";
export const COREML_DEPLOYMENT_SOURCE = Object.freeze({
  repository: "apple/coremltools",
  release: "9.0",
  source_commit: "428d4b2658dfc44194f27f4f36870751be402ff7",
  model_proto: "mlmodel/format/Model.proto",
  model_proto_sha256: "c731d0202acecd54f20eaf97c94a2a4764fe541ba92311485a9c36d3c3c3b544",
  coremltools_init: "coremltools/__init__.py",
  coremltools_init_sha256: "ebbb958d3bc70c16c3c1d991f46fd3d25da5f08f1c12083f0634a7293a0cca81",
  deployment_compatibility: "coremltools/converters/mil/_deployment_compatibility.py",
  deployment_compatibility_sha256: "4b4601bc4afa4b90282052d933267ffdcfa78af42472e133393640e55f200fed",
  compute_plan: "coremltools/models/compute_plan.py",
  compute_plan_sha256: "977c84697df762a1958bd1c8e562db8e9cb95270254508983e11519c57f8e497",
});

const FLOORS = Object.freeze({
  1: Object.freeze({ ios: "11", macos: "10.13", tvos: "11", watchos: "4", visionos: null, coreml: "1" }),
  2: Object.freeze({ ios: "11.2", macos: "10.13.2", tvos: "11.2", watchos: "4.2", visionos: null, coreml: "1.2" }),
  3: Object.freeze({ ios: "12", macos: "10.14", tvos: "12", watchos: "5", visionos: null, coreml: "2" }),
  4: Object.freeze({ ios: "13", macos: "10.15", tvos: "13", watchos: "6", visionos: null, coreml: "3" }),
  5: Object.freeze({ ios: "14", macos: "11", tvos: "14", watchos: "7", visionos: null, coreml: "4" }),
  6: Object.freeze({ ios: "15", macos: "12", tvos: "15", watchos: "8", visionos: null, coreml: "5" }),
  7: Object.freeze({ ios: "16", macos: "13", tvos: "16", watchos: "9", visionos: null, coreml: "6" }),
  8: Object.freeze({ ios: "17", macos: "14", tvos: "17", watchos: "10", visionos: null, coreml: "7" }),
  9: Object.freeze({ ios: "18", macos: "15", tvos: "18", watchos: "11", visionos: null, coreml: "8" }),
  10: Object.freeze({ ios: "26", macos: "26", tvos: "26", watchos: "26", visionos: "26", coreml: "9" }),
});

const MODEL_TYPE_FLOOR = Object.freeze({
  customModel: 3,
  textClassifier: 3,
  wordTagger: 3,
  visionFeaturePrint: 3,
  nonMaximumSuppression: 3,
  kNearestNeighborsClassifier: 4,
  soundAnalysisPreprocessing: 4,
  itemSimilarityRecommender: 4,
  linkedModel: 4,
  gazetteer: 4,
  wordEmbedding: 4,
  mlProgram: 6,
  audioFeaturePrint: 6,
  classConfidenceThresholding: 8,
});

const OPSET_TO_SPEC = Object.freeze({ CoreML3: 4, CoreML4: 5, CoreML5: 6, CoreML6: 7, CoreML7: 8, CoreML8: 9, CoreML9: 10 });

export function buildCoreMlDeploymentContract({ specificationVersion, modelType, isUpdatable = false, description = null, mlProgram = null } = {}) {
  return {
    ...portableContract("coreml_deployment", {specificationVersion:Number(specificationVersion),modelType:modelType ?? null,isUpdatable:!!isUpdatable,description,mlProgram,
      floors:FLOORS,model_type_floors:MODEL_TYPE_FLOOR,opset_to_spec:OPSET_TO_SPEC}),
    source: { ...COREML_DEPLOYMENT_SOURCE },
    method: "Use the exact specificationVersion-to-OS availability table serialized in pinned Model.proto, then independently derive a necessary observed-feature floor from model type, interface dtypes/flexibility, multi-function/state declarations, updatability, and MIL opset identity. The declared specification remains the runtime load floor even when observed features would permit an older format version.",
    boundary: "This is a Core ML format/OS availability floor. It does not establish device eligibility, compiled-model specialization, CPU/GPU/ANE placement, MLComputePlan cost, or successful execution on a particular Apple device.",
  };
}

export function coreMlFloorLabel(floor) {
  if (!floor) return "not mapped";
  return `iOS ${floor.ios}; macOS ${floor.macos}; tvOS ${floor.tvos}; watchOS ${floor.watchos}${floor.visionos ? `; visionOS ${floor.visionos}` : ""}; Core ML ${floor.coreml}`;
}
