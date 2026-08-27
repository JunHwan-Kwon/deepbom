import { createAccumulatorAtlasController } from "./accumulator-atlas.js";
import { createAccumulatorReachabilityController } from "./accumulator-reachability-viewer.js";
import { createChannelVitalityController } from "./channel-vitality.js";
import { createContractMigrationController } from "./contract-migration.js";
import { createKernelWitnessController } from "./kernel-witness.js";
import { createInputCounterexampleController } from "./input-counterexample-viewer.js";
import { createNumericalAbiPropagationController } from "./numerical-abi-propagation-viewer.js";
import { createPreprocessingRealizabilityController } from "./preprocessing-realizability-viewer.js";
import { createPreprocessingConsequenceController } from "./preprocessing-consequence-viewer.js";
import { createQuantizationLatticeController } from "./quantization-lattice.js";
import { createRequantizationFidelityController } from "./requantization-fidelity.js";
import { createResidualContractDistortionController } from "./residual-contract-distortion.js";
import { createResidualStepResponseController } from "./residual-step-response.js";
import { createRoundingEquivalenceController } from "./rounding-equivalence.js";
import {
  ensureQuantResearchCoverage,
  quantResearchLabCoverage,
} from "./quant-research-applicability.js";

const CONTROLLERS = Object.freeze([
  ["quantizationLatticeController", "quantizationLattice", "quantization_lattice", createQuantizationLatticeController, "analysis"],
  ["accumulatorAtlasController", "accumulatorAtlas", "accumulator_atlas", createAccumulatorAtlasController, "context"],
  ["requantizationFidelityController", "requantizationFidelity", "requantization_fidelity", createRequantizationFidelityController, "context"],
  ["kernelWitnessController", "kernelWitness", "kernel_extremum_witness", createKernelWitnessController, "context"],
  ["channelVitalityController", "channelVitality", "channel_vitality", createChannelVitalityController, "context"],
  ["roundingEquivalenceController", "roundingEquivalence", "rounding_equivalence", createRoundingEquivalenceController, "context"],
  ["accumulatorReachabilityController", "accumulatorReachability", "accumulator_reachability", createAccumulatorReachabilityController, "context"],
  ["numericalAbiPropagationController", "numericalAbiPropagation", "numerical_abi_propagation", createNumericalAbiPropagationController, "analysis"],
  ["inputCounterexampleController", "inputCounterexample", "input_counterexample", createInputCounterexampleController, "analysis"],
  ["preprocessingRealizabilityController", "preprocessingRealizability", "preprocessing_realizability", createPreprocessingRealizabilityController, "analysis"],
  ["preprocessingConsequenceController", "preprocessingConsequence", "preprocessing_consequence", createPreprocessingConsequenceController, "context"],
  ["contractMigrationController", "contractMigration", "contract_migration", createContractMigrationController, "context"],
  ["residualStepResponseController", "residualStepResponse", "residual_step_response", createResidualStepResponseController, "analysis"],
  ["residualContractDistortionController", "residualContractDistortion", "residual_contract_distortion", createResidualContractDistortionController, "analysis"],
]);

export function createQuantizationResearchSuite({
  elements,
  getContext,
  jumpToGraphOp,
  onDownload,
  onDownloadBinary,
  ensureRuntime,
  onPreprocessingConsequenceResult,
}) {
  const suite = {};
  const residualSelection = createResidualSelectionStore();
  for (const [controllerName, key, labId, factory, sourceKind] of CONTROLLERS) {
    const source = sourceKind === "context"
      ? { getContext }
      : { getAnalysis: () => getContext()?.analysis || null };
    const controller = factory({
      root: elements[`${key}Panel`],
      status: elements[`${key}Status`],
      summary: elements[`${key}Summary`],
      body: elements[`${key}Body`],
      runButton: elements[`run${upperFirst(key)}`],
      downloadButton: elements[`download${upperFirst(key)}`],
      ...source,
      jumpToGraphOp,
      onDownload,
      onDownloadBinary,
      ensureRuntime,
      residualSelection,
      onResult: key === "preprocessingConsequence" ? onPreprocessingConsequenceResult : undefined,
    });
    suite[controllerName] = withApplicabilityGate(controller, {
      labId,
      root: elements[`${key}Panel`],
      status: elements[`${key}Status`],
      summary: elements[`${key}Summary`],
      body: elements[`${key}Body`],
      runButton: elements[`run${upperFirst(key)}`],
      downloadButton: elements[`download${upperFirst(key)}`],
    });
  }
  return suite;
}

function createResidualSelectionStore() {
  let value = { opIndex: null, design: null };
  const listeners = new Set();
  return Object.freeze({
    get: () => ({ ...value }),
    set: (next, source = "") => {
      const updated = {
        opIndex: Number.isInteger(Number(next?.opIndex)) ? Number(next.opIndex) : value.opIndex,
        design: next?.design || value.design,
      };
      if (updated.opIndex === value.opIndex && updated.design === value.design) return;
      value = updated;
      for (const listener of listeners) listener({ ...value }, source);
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  });
}

function withApplicabilityGate(controller, elements) {
  const render = controller.render.bind(controller);
  controller.render = (analysis) => {
    if (!analysis) {
      clearGate(elements.root);
      return render(analysis);
    }
    ensureQuantResearchCoverage(analysis);
    const coverage = quantResearchLabCoverage(analysis, elements.labId);
    if (coverage?.render_policy === "common_empty") {
      render(null);
      renderCommonEmpty(elements, coverage);
      return;
    }
    clearGate(elements.root);
    return render(analysis);
  };
  return controller;
}

function renderCommonEmpty({ root, status, summary, body, runButton, downloadButton }, coverage) {
  root?.classList.add("quant-lab-common-empty");
  if (root) {
    root.hidden = false;
    root.dataset.quantLabApplicability = coverage.status;
    root.dataset.quantLabReasonCode = coverage.reason_code;
  }
  if (status) {
    status.textContent = coverage.status === "not_applicable" ? "not applicable to artifact class" : "not assessed";
    status.dataset.tone = "neutral";
  }
  summary?.replaceChildren();
  if (body) {
    const message = document.createElement("div");
    message.className = "quant-lab-common-empty-message";
    const title = document.createElement("strong");
    title.textContent = coverage.status === "not_applicable" ? "Outside this artifact class" : "No assessable evidence";
    const detail = document.createElement("p");
    detail.textContent = `${coverage.reason} See Quant Research Coverage for the shared denominator and excluded-lab list.`;
    const code = document.createElement("code");
    code.textContent = coverage.reason_code;
    message.append(title, detail, code);
    body.replaceChildren(message);
  }
  if (runButton) runButton.disabled = true;
  if (downloadButton) downloadButton.disabled = true;
}

function clearGate(root) {
  root?.classList.remove("quant-lab-common-empty");
  if (root) {
    delete root.dataset.quantLabApplicability;
    delete root.dataset.quantLabReasonCode;
  }
}

function upperFirst(value) {
  return `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
}
