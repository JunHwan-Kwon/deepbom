import { sha256Hex } from "./hash.js";

export const INPUT_COUNTEREXAMPLE_SCHEMA = "deepbom.input_counterexample.v1";
export const INPUT_COUNTEREXAMPLE_METHOD_VERSION = "2026-07-18.3";

const SOURCE_EVIDENCE_SCHEMA = "deepbom.accumulator_reachability.v1 + deepbom.numerical_abi_propagation.v1.1";
const WITNESS_PREFIX = "deepbom.input_counterexample.witness.v1\0";
const PORTFOLIO_PREFIX = "deepbom.input_counterexample.portfolio.v1\0";
const FILL_POLICY = "fill_every_element_with_zero_point_then_apply_sparse_overrides";
const VALID_CLASSIFICATIONS = new Set([
  "tensor_abi_constructive",
  "direct_model_input_not_constructed",
  "upstream_activation_constraint_unresolved",
]);

export function validateInputCounterexampleShape(evidence) {
  assert(evidence?.schema === INPUT_COUNTEREXAMPLE_SCHEMA, "Input counterexample schema mismatch.");
  assert(evidence.method_version === INPUT_COUNTEREXAMPLE_METHOD_VERSION, "Input counterexample method mismatch.");
  assert(evidence.evidence_class === "DERIVED", "Input counterexample evidence class mismatch.");
  assert(evidence.source_evidence_schema === SOURCE_EVIDENCE_SCHEMA, "Input counterexample source evidence mismatch.");
  assert(["assessed", "partial", "not_applicable", "not_computed_internal_scope"].includes(evidence.status), "Input counterexample status is invalid.");
  assert(Array.isArray(evidence.sources), "Input counterexample sources are missing.");
  assert(Array.isArray(evidence.witnesses), "Input counterexample witnesses are missing.");
  assert(evidence.sources.length === integer(evidence.exact_local_source_op_count, "exact-local source count"), "Input counterexample source count mismatch.");
  assert(evidence.witnesses.length === integer(evidence.representative_witness_count, "representative witness count"), "Input counterexample witness count mismatch.");
  assertDigest(evidence.portfolio_ledger_sha256, "portfolio ledger");
  for (const source of evidence.sources) {
    integer(source.op_index, "source op index");
    assert(VALID_CLASSIFICATIONS.has(source.classification), `Input counterexample classification is invalid at op #${source.op_index}.`);
    assertDigest(source.source_reachability_ledger_sha256, `source reachability ledger at op #${source.op_index}`);
    assertDigest(source.source_propagation_ledger_sha256, `source propagation ledger at op #${source.op_index}`);
    BigInt(source.exact_reachable_divergent_state_count_decimal);
    if (source.representative_witness_index == null) {
      assert(source.representative_witness_ledger_sha256 === "", `Unexpected witness ledger at op #${source.op_index}.`);
    } else {
      const index = integer(source.representative_witness_index, "representative witness index");
      assert(index >= 0 && index < evidence.witnesses.length, `Representative witness index escapes the portfolio at op #${source.op_index}.`);
      assertDigest(source.representative_witness_ledger_sha256, `representative witness ledger at op #${source.op_index}`);
    }
  }
  return evidence;
}

export function buildInputWitnessTensor(witness) {
  assert(witness && typeof witness === "object", "Input witness is missing.");
  const shape = staticShape(witness.model_input_shape);
  const elementCount = checkedProduct(shape);
  assert(elementCount === integer(witness.model_input_element_count, "model-input element count"), "Model-input shape product does not match the declared element count.");
  assert(["UINT8", "INT8"].includes(witness.model_input_dtype), "Input witness dtype must be UINT8 or INT8.");
  const [qmin, qmax] = codeRange(witness.model_input_dtype);
  assert(Array.isArray(witness.model_input_code_range)
    && witness.model_input_code_range.length === 2
    && witness.model_input_code_range[0] === qmin
    && witness.model_input_code_range[1] === qmax, "Input witness code range mismatch.");
  const fill = code(witness.full_tensor_fill_code, qmin, qmax, "full tensor fill code");
  assert(fill === code(witness.model_input_zero_point, qmin, qmax, "model-input zero point"), "Input witness fill code must equal the tensor zero point.");
  assert(witness.full_tensor_fill_policy === FILL_POLICY, "Input witness fill policy mismatch.");
  const bytes = new Uint8Array(elementCount);
  bytes.fill(encodeCode(fill, witness.model_input_dtype));
  assert(Array.isArray(witness.sparse_overrides), "Input witness sparse overrides are missing.");
  assert(witness.sparse_overrides.length === integer(witness.sparse_override_count, "sparse override count"), "Input witness sparse override count mismatch.");
  const seen = new Set();
  let previous = -1;
  for (const override of witness.sparse_overrides) {
    const index = integer(override.input_linear_index, "sparse override index");
    assert(index >= 0 && index < elementCount, `Sparse override #${index} escapes the model-input tensor.`);
    assert(index > previous && !seen.has(index), "Sparse overrides must be unique and ascending.");
    const value = code(override.input_code, qmin, qmax, `sparse override #${index}`);
    assert(value !== fill, `Sparse override #${index} redundantly stores the fill code.`);
    bytes[index] = encodeCode(value, witness.model_input_dtype);
    seen.add(index);
    previous = index;
  }
  let nonFillCount = 0;
  for (const byte of bytes) if (decodeCode(byte, witness.model_input_dtype) !== fill) nonFillCount += 1;
  assert(nonFillCount === seen.size, "Sparse overrides do not exactly describe every non-fill tensor element.");
  return { bytes, shape, elementCount, fillCode: fill, qmin, qmax };
}

export async function reconstructInputWitness(witness) {
  const tensor = buildInputWitnessTensor(witness);
  const tensorSha256 = await sha256Hex(tensor.bytes);
  assert(tensorSha256 === witness.full_model_input_tensor_sha256, "Full model-input tensor SHA-256 mismatch.");
  assertDigest(witness.source_reachability_ledger_sha256, "witness reachability ledger");
  assertDigest(witness.source_propagation_ledger_sha256, "witness propagation ledger");
  assertDigest(witness.witness_ledger_sha256, "witness ledger");
  assert(Array.isArray(witness.source_output_coordinate) && witness.source_output_coordinate.length === 4, "Source output coordinate must be rank four.");
  assert(integer(witness.source_output_coordinate[3], "source output channel coordinate") === integer(witness.source_channel_index, "source channel index"), "Source output coordinate channel mismatch.");
  assert(witness.full_valid_receptive_field === true, "Input witness is not a full-valid receptive field.");
  const patchOrigin = vector(witness.patch_origin_yx, 2, "patch origin");
  const patchShape = vector(witness.effective_patch_shape, 3, "effective patch shape", true);
  const kernelShape = vector(witness.kernel_shape, 3, "kernel shape", true);
  const dilation = vector(witness.dilation_hw, 2, "dilation", true);
  vector(witness.stride_hw, 2, "stride", true);
  assert(patchShape[0] === (kernelShape[0] - 1) * dilation[0] + 1
    && patchShape[1] === (kernelShape[1] - 1) * dilation[1] + 1
    && patchShape[2] === tensor.shape[3], "Effective patch geometry mismatch.");
  assert(patchOrigin[0] + patchShape[0] <= tensor.shape[1]
    && patchOrigin[1] + patchShape[1] <= tensor.shape[2], "Effective patch escapes the model-input tensor.");
  assert(Array.isArray(witness.patch_codes_hwc)
    && witness.patch_codes_hwc.length === checkedProduct(patchShape), "Input witness patch cardinality mismatch.");
  const extractedPatch = [];
  for (let y = 0; y < patchShape[0]; y += 1) {
    for (let x = 0; x < patchShape[1]; x += 1) {
      for (let channel = 0; channel < patchShape[2]; channel += 1) {
        const linear = nhwcLinearIndex(tensor.shape, [0, patchOrigin[0] + y, patchOrigin[1] + x, channel]);
        extractedPatch.push(decodeCode(tensor.bytes[linear], witness.model_input_dtype));
      }
    }
  }
  assert(sameArray(extractedPatch, witness.patch_codes_hwc), "Input witness patch does not match the full tensor.");
  assert(Array.isArray(witness.terms) && witness.terms.length > 0, "Input witness term ledger is missing.");
  let dot = 0n;
  const histogram = new Map();
  const termIndices = new Set();
  for (const [position, term] of witness.terms.entries()) {
    assert(integer(term.term_index, "term index") === position, `Input witness term order mismatch at ${position}.`);
    const kernelCoordinate = vector(term.kernel_coordinate, 3, `kernel coordinate ${position}`);
    assert(kernelCoordinate[0] < kernelShape[0] && kernelCoordinate[1] < kernelShape[1] && kernelCoordinate[2] < tensor.shape[3], `Kernel coordinate escapes geometry at term ${position}.`);
    const expectedCoordinate = [0, patchOrigin[0] + kernelCoordinate[0] * dilation[0], patchOrigin[1] + kernelCoordinate[1] * dilation[1], kernelCoordinate[2]];
    const inputCoordinate = vector(term.input_coordinate, 4, `input coordinate ${position}`);
    assert(sameArray(inputCoordinate, expectedCoordinate), `Input coordinate disagrees with kernel geometry at term ${position}.`);
    const linear = nhwcLinearIndex(tensor.shape, inputCoordinate);
    assert(linear === integer(term.input_linear_index, "term input linear index"), `NHWC linear index mismatch at term ${position}.`);
    assert(!termIndices.has(linear), `Input witness reuses tensor element ${linear} in the term ledger.`);
    termIndices.add(linear);
    const inputCode = decodeCode(tensor.bytes[linear], witness.model_input_dtype);
    assert(inputCode === code(term.input_code, tensor.qmin, tensor.qmax, `term input code ${position}`), `Full tensor code mismatch at term ${position}.`);
    const centered = BigInt(inputCode) - BigInt(witness.model_input_zero_point);
    assert(centered === BigInt(term.centered_input_code), `Centered input mismatch at term ${position}.`);
    const weight = BigInt(term.centered_weight);
    const product = centered * weight;
    assert(product === BigInt(term.term_product_decimal), `Term product mismatch at term ${position}.`);
    dot += product;
    histogram.set(inputCode, (histogram.get(inputCode) || 0) + 1);
  }
  assert(dot === BigInt(witness.dot_product_decimal), "Input witness dot product mismatch.");
  const postBias = dot + BigInt(witness.bias_decimal);
  assert(postBias === BigInt(witness.post_bias_accumulator_decimal), "Input witness post-bias accumulator mismatch.");
  assert(integer(witness.default_output_code, "default output code") !== integer(witness.single_rounding_output_code, "single-rounding output code"), "Input witness output paths do not diverge.");
  assert(witness.output_code_delta === witness.default_output_code - witness.single_rounding_output_code, "Input witness output-code delta mismatch.");
  assertHistogram(witness.input_code_histogram, histogram, witness.terms.length);
  const witnessLedgerSha256 = await sha256Hex(new TextEncoder().encode(witnessCanonical(witness)));
  assert(witnessLedgerSha256 === witness.witness_ledger_sha256, "Input witness ledger SHA-256 mismatch.");
  return { ...tensor, tensorSha256, witnessLedgerSha256, extractedPatch, dotProduct: dot, postBiasAccumulator: postBias };
}

export async function validateInputCounterexampleAnalysis(analysis) {
  const evidence = validateInputCounterexampleShape(analysis?.input_counterexample);
  const reachabilityByOp = new Map((analysis?.accumulator_reachability?.ops || []).map((row) => [row.op_index, row]));
  const propagationByOp = new Map((analysis?.numerical_abi_propagation?.sources || []).map((row) => [row.op_index, row]));
  const opsByIndex = new Map((analysis?.ops || []).map((op) => [op.index, op]));
  const modelInputs = new Set((analysis?.input_tensor_indices || []).filter(Number.isInteger));
  const exactRows = [...reachabilityByOp.values()].filter((row) => BigInt(row.exact_reachable_divergent_state_count_decimal || "0") > 0n);
  assert(exactRows.length === evidence.sources.length, "Input counterexample portfolio omits exact-local reachability sources.");
  const sourceByOp = new Map();
  const reconstructedWitnesses = [];
  for (const source of evidence.sources) {
    assert(!sourceByOp.has(source.op_index), `Duplicate input counterexample source op #${source.op_index}.`);
    sourceByOp.set(source.op_index, source);
    const reachability = reachabilityByOp.get(source.op_index);
    const propagation = propagationByOp.get(source.op_index);
    const op = opsByIndex.get(source.op_index);
    assert(reachability && propagation && op, `Input counterexample source evidence is missing at op #${source.op_index}.`);
    assert(source.source_reachability_ledger_sha256 === reachability.reachability_ledger_sha256, `Reachability ledger join mismatch at op #${source.op_index}.`);
    assert(source.source_propagation_ledger_sha256 === propagation.propagation_ledger_sha256, `Propagation ledger join mismatch at op #${source.op_index}.`);
    assert(source.exact_reachable_divergent_channel_count === reachability.exact_reachable_divergent_channel_count, `Exact channel count join mismatch at op #${source.op_index}.`);
    assert(source.exact_reachable_divergent_state_count_decimal === reachability.exact_reachable_divergent_state_count_decimal, `Exact state count join mismatch at op #${source.op_index}.`);
    assert(source.reachable_model_output_tensor_count === propagation.reachable_model_output_tensor_count, `Output reachability join mismatch at op #${source.op_index}.`);
    assert(source.exact_model_output_graph_route_count_decimal === propagation.exact_model_output_graph_route_count_decimal, `Output route count join mismatch at op #${source.op_index}.`);
    const firstInput = (op.inputs || [])[0];
    assert(source.input_tensor_index === firstInput, `Source input tensor join mismatch at op #${source.op_index}.`);
    const direct = modelInputs.has(firstInput);
    const hasWitness = source.representative_witness_index != null;
    const expectedClass = direct
      ? hasWitness ? "tensor_abi_constructive" : "direct_model_input_not_constructed"
      : "upstream_activation_constraint_unresolved";
    assert(source.classification === expectedClass, `Input-origin classification mismatch at op #${source.op_index}.`);
    if (hasWitness) {
      const witness = evidence.witnesses[source.representative_witness_index];
      assert(witness.source_op_index === source.op_index, `Representative witness source mismatch at op #${source.op_index}.`);
      assert(witness.model_input_tensor_index === firstInput, `Representative witness input mismatch at op #${source.op_index}.`);
      assert(witness.source_reachability_ledger_sha256 === source.source_reachability_ledger_sha256, `Witness reachability ledger join mismatch at op #${source.op_index}.`);
      assert(witness.source_propagation_ledger_sha256 === source.source_propagation_ledger_sha256, `Witness propagation ledger join mismatch at op #${source.op_index}.`);
      assert(witness.source_exact_reachable_divergent_state_count_decimal === source.exact_reachable_divergent_state_count_decimal, `Witness exact-state join mismatch at op #${source.op_index}.`);
      const reconstructed = await reconstructInputWitness(witness);
      assert(reconstructed.witnessLedgerSha256 === source.representative_witness_ledger_sha256, `Representative witness ledger mismatch at op #${source.op_index}.`);
      reconstructedWitnesses[source.representative_witness_index] = reconstructed;
    }
  }
  for (const row of exactRows) assert(sourceByOp.has(row.op_index), `Exact-local op #${row.op_index} is absent from the input portfolio.`);
  assert(reconstructedWitnesses.filter(Boolean).length === evidence.witnesses.length, "Input counterexample portfolio contains unreferenced witnesses.");
  validatePortfolioCounts(evidence);
  const portfolioLedgerSha256 = await sha256Hex(new TextEncoder().encode(portfolioCanonical(evidence.sources)));
  assert(portfolioLedgerSha256 === evidence.portfolio_ledger_sha256, "Input counterexample portfolio ledger SHA-256 mismatch.");
  return { evidence, witnesses: reconstructedWitnesses, portfolioLedgerSha256 };
}

function validatePortfolioCounts(evidence) {
  const constructive = evidence.sources.filter((source) => source.classification === "tensor_abi_constructive");
  const direct = evidence.sources.filter((source) => source.input_origin === "declared_model_input");
  const upstream = evidence.sources.filter((source) => source.classification === "upstream_activation_constraint_unresolved");
  const notAssessed = evidence.sources.length - constructive.length - upstream.length;
  assert(evidence.direct_model_input_source_op_count === direct.length, "Direct model-input source count mismatch.");
  assert(evidence.tensor_abi_constructive_source_op_count === constructive.length, "Constructive source count mismatch.");
  assert(evidence.upstream_activation_unresolved_source_op_count === upstream.length, "Upstream-unresolved source count mismatch.");
  assert(evidence.not_assessed_source_op_count === notAssessed, "Not-assessed source count mismatch.");
  assert(evidence.tensor_abi_constructive_channel_count === constructive.reduce((sum, source) => sum + source.exact_reachable_divergent_channel_count, 0), "Constructive channel count mismatch.");
  const states = constructive.reduce((sum, source) => sum + BigInt(source.exact_reachable_divergent_state_count_decimal), 0n);
  assert(BigInt(evidence.tensor_abi_constructive_divergent_state_count_decimal) === states, "Constructive divergent-state count mismatch.");
  assert(evidence.output_reachable_constructive_source_op_count === constructive.filter((source) => source.reachable_model_output_tensor_count > 0).length, "Output-reachable constructive source count mismatch.");
  const conservation = `${evidence.sources.length} = ${constructive.length} constructive + ${upstream.length} upstream-unresolved + ${notAssessed} not-assessed`;
  assert(evidence.source_classification_conservation === conservation, "Input counterexample source classification does not conserve.");
}

function witnessCanonical(witness) {
  let value = WITNESS_PREFIX;
  value += `reachability=${witness.source_reachability_ledger_sha256}\npropagation=${witness.source_propagation_ledger_sha256}\n`;
  value += `op=${witness.source_op_index};channel=${witness.source_channel_index};input=${witness.model_input_tensor_index};output=${witness.source_output_tensor_index};target=${witness.post_bias_accumulator_decimal};default=${witness.default_output_code};single=${witness.single_rounding_output_code};dot=${witness.dot_product_decimal};bias=${witness.bias_decimal};tensor_sha=${witness.full_model_input_tensor_sha256}\n`;
  value += `fill=${witness.full_tensor_fill_code};elements=${witness.model_input_element_count};shape=${witness.model_input_shape.join(",")};output_coordinate=${witness.source_output_coordinate.join(",")};patch_origin=${witness.patch_origin_yx.join(",")};patch_shape=${witness.effective_patch_shape.join(",")}\n`;
  for (const term of witness.terms) value += `term=${term.term_index};linear=${term.input_linear_index};code=${term.input_code};centered=${term.centered_input_code};weight=${term.centered_weight};product=${term.term_product_decimal}\n`;
  for (const override of witness.sparse_overrides) value += `override=${override.input_linear_index};code=${override.input_code}\n`;
  return value;
}

function portfolioCanonical(sources) {
  let value = PORTFOLIO_PREFIX;
  for (const source of sources) {
    value += `source=${source.op_index};class=${source.classification};channels=${source.exact_reachable_divergent_channel_count};states=${source.exact_reachable_divergent_state_count_decimal};outputs=${source.reachable_model_output_tensor_count};routes=${source.exact_model_output_graph_route_count_decimal ?? "none"};reachability=${source.source_reachability_ledger_sha256};propagation=${source.source_propagation_ledger_sha256};witness=${source.representative_witness_ledger_sha256}\n`;
  }
  return value;
}

function assertHistogram(rows, histogram, termCount) {
  assert(Array.isArray(rows), "Input witness histogram is missing.");
  let total = 0;
  let previous = -Infinity;
  assert(rows.length === histogram.size, "Input witness histogram cardinality mismatch.");
  for (const row of rows) {
    const value = integer(row.code, "histogram code");
    const count = integer(row.count, "histogram count");
    assert(value > previous && count > 0, "Input witness histogram must be ascending with positive counts.");
    assert(histogram.get(value) === count, `Input witness histogram count mismatch for code ${value}.`);
    previous = value;
    total += count;
  }
  assert(total === termCount, "Input witness histogram does not conserve terms.");
}

function staticShape(value) {
  assert(Array.isArray(value) && value.length === 4, "Model-input shape must be static rank-four NHWC.");
  return value.map((dimension) => {
    const parsed = integer(dimension, "model-input dimension");
    assert(parsed > 0, "Model-input dimensions must be positive.");
    return parsed;
  });
}

function checkedProduct(values) {
  let product = 1;
  for (const value of values) {
    product *= value;
    assert(Number.isSafeInteger(product), "Tensor element count exceeds JavaScript safe integer range.");
  }
  return product;
}

function nhwcLinearIndex(shape, coordinate) {
  assert(shape.length === 4 && coordinate.length === 4, "NHWC coordinate rank mismatch.");
  coordinate.forEach((value, index) => assert(Number.isInteger(value) && value >= 0 && value < shape[index], "NHWC coordinate escapes the tensor."));
  return ((coordinate[0] * shape[1] + coordinate[1]) * shape[2] + coordinate[2]) * shape[3] + coordinate[3];
}

function codeRange(dtype) {
  return dtype === "UINT8" ? [0, 255] : [-128, 127];
}

function encodeCode(value, dtype) {
  return dtype === "UINT8" ? value : value & 0xff;
}

function decodeCode(value, dtype) {
  return dtype === "UINT8" || value < 128 ? value : value - 256;
}

function code(value, minimum, maximum, label) {
  const parsed = integer(value, label);
  assert(parsed >= minimum && parsed <= maximum, `${label} escapes the quantized code range.`);
  return parsed;
}

function vector(value, length, label, positive = false) {
  assert(Array.isArray(value) && value.length === length, `${label} rank mismatch.`);
  return value.map((entry) => {
    const parsed = integer(entry, label);
    assert(parsed >= (positive ? 1 : 0), `${label} contains an invalid coordinate or extent.`);
    return parsed;
  });
}

function sameArray(left, right) {
  return Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]);
}

function integer(value, label) {
  assert(Number.isSafeInteger(value), `${label} must be a safe integer.`);
  return value;
}

function assertDigest(value, label) {
  assert(/^[a-f0-9]{64}$/.test(value || ""), `${label} SHA-256 is invalid.`);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
