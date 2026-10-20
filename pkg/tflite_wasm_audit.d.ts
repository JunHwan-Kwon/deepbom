/* tslint:disable */
/* eslint-disable */

export class NumericalComparison {
    free(): void;
    [Symbol.dispose](): void;
    add_chunk(left: Float64Array, right: Float64Array): void;
    cosine(): number | undefined;
    duplicate(): void;
    finish(): any;
    merge(other: NumericalComparison): void;
    constructor();
    regression_gain(): number | undefined;
}

export class NumericalMoments {
    free(): void;
    [Symbol.dispose](): void;
    add_chunk(values: Float64Array): void;
    coefficient_of_variation(size: number): number | undefined;
    divided_by(divisor: number, factor: string): number;
    finish(size: number): any;
    mean(size: number): number | undefined;
    constructor(squares: boolean);
    totals(): string[];
}

export class NumericalStatistics {
    free(): void;
    [Symbol.dispose](): void;
    add_chunk(values: Float64Array): void;
    add_integer(value: string): void;
    finish(): any;
    constructor();
}

export function analyze_tflite(bytes: Uint8Array, filename: string): any;

export function analyze_tflite_for_target(bytes: Uint8Array, filename: string, target_id: string): any;

/**
 * Compute activation Haar energy for each CONV/DEPTHWISE op via synthetic forward pass.
 * Uses a 32×32 (max) synthetic input with a deterministic spatial pattern.
 * Works entirely in f64; INT8 weights are dequantized using full per-channel scales.
 */
export function compute_activation_haar(bytes: Uint8Array, filename: string, target_id: string): any;

export function compute_delegation_repair(model_bytes: Uint8Array, filename: string, target_id: string): any;

export function compute_deployment_delta(baseline_bytes: Uint8Array, baseline_filename: string, candidate_bytes: Uint8Array, candidate_filename: string, target_ids_json: string): any;

export function compute_deployment_frontier(bytes: Uint8Array, filename: string, target_ids_json: string): any;

/**
 * Backward (input) influence: spatial BFS from clicked op's output → model inputs.
 */
export function compute_input_influence(bytes: Uint8Array, filename: string, op_index: number, target_id: string): any;

export function compute_kernel_haar_decomposition(bytes: Uint8Array, filename: string, target_id: string): any;

/**
 * Per-layer weight statistics + activation Haar for triplanar model viewer.
 * Returns one entry per CONV_2D / DEPTHWISE_CONV_2D / FULLY_CONNECTED op.
 */
export function compute_model_tomography(bytes: Uint8Array, filename: string, target_id: string): any;

/**
 * Forward (output) influence: spatial BFS from clicked op's output → model outputs.
 */
export function compute_output_influence(bytes: Uint8Array, filename: string, op_index: number, target_id: string): any;

/**
 * Fast low-norm-filter count for an op weight tensor (L2 < 2% of max).
 */
export function compute_quick_low_norm_stat(bytes: Uint8Array, filename: string, tensor_index: number, target_id: string): any;

/**
 * Compute correction factor between static latency estimate and WASM-measured runtime.
 * Pass measured_ms = 0.0 if no runtime data is available (returns static estimate only).
 */
export function compute_static_runtime_calibration(bytes: Uint8Array, filename: string, target_id: string, measured_ms: number): any;

/**
 * Retired ABI entry point. Numerical consumers must use the common Weight IR.
 * Keep an explicit error for cached/older clients rather than returning a
 * second statistics contract with different percentile and rank semantics.
 */
export function compute_weight_histogram(_bytes: Uint8Array, _filename: string, _tensor_index: number, _target_id: string): any;

export function evidence_sha256(bytes: Uint8Array): string;

export function explore_tflite_redesign_pareto(bytes: Uint8Array, filename: string, target_id: string, request: any): any;

/**
 * Generate two filter-normalized random directions in weight space (one per seed parameter).
 * Returns d1, d2 as flat Float32Arrays and per-tensor metadata needed for JS-side perturbation.
 * Call once per seed; JS applies alpha*d1+beta*d2 directly on the model Uint8Array.
 */
export function landscape_directions(bytes: Uint8Array, seed1: number, seed2: number): any;

/**
 * Run K independent 2D landscape projections in a single WASM call.
 * analyze_with_target is called only once; K × G² forward passes follow.
 * Each projection uses a distinct filter-normalized direction pair.
 */
export function landscape_tomography(bytes: Uint8Array, num_projections: number, grid_size: number, radius: number): any;

/**
 * Single-layer 2D landscape grid — perturbs only the weight tensor of the given op.
 * Much faster than full-model landscape for interactive per-layer exploration.
 */
export function layer_landscape_grid(bytes: Uint8Array, op_index: number, seed1: number, seed2: number, grid_size: number, radius: number): any;

export function numerical_abi_version(): string;

export function numerical_affine(values: Float64Array, shape: Float64Array, dtype: string, contract: any, order: string): any;

export function numerical_contract(operation: string, input: string): string;

export function numerical_count_ratio(a: string, b: string): number | undefined;

export function numerical_distribution(statistics: any): any;

export function numerical_distribution_comparison(baseline: any, candidate: any): any;

export function numerical_histogram_edges(): Float64Array;

export function numerical_metric_methods(): string;

export function numerical_pruning_prepare(values: Float64Array, rows: number, columns: number): any;

export function numerical_pruning_simulate(values: Float64Array, rows: number, columns: number, order: Uint32Array, percent: number): any;

export function numerical_quality_metric(metric: string, prediction: Float64Array, target: Float64Array, shape: Uint32Array, settings: string): number;

export function numerical_shape_count(shape: Float64Array): string;

export function numerical_spectrum(data: Float64Array, rows: number, columns: number, max_work: number): string;

export function numerical_spectrum_summary(sigma: Float64Array, scale: number, rows: number, columns: number): string;

export function numerical_weight_comparison(a: Float64Array, b: Float64Array): any;

export function numerical_weight_differences(a: Float64Array, b: Float64Array): Float64Array;

export function numerical_weight_magnitude(data: Float64Array): number;

export function numerical_weight_matrix(data: Float64Array, shape: Float64Array, axis: number, order: string): Float64Array;

export function numerical_weight_permute(data: Float64Array, shape: Float64Array, order: string, axes: Float64Array): Float64Array;

export function numerical_weight_similarity(data: Float64Array, rows: number, columns: number, limit: number, threshold: number): any;

export function numerical_weight_sparsity(data: Float64Array, shape: Float64Array, order: string, axis: number, block: number): any;

export function numerical_weight_tiles(data: Float64Array, rows: number, columns: number, maximum: number): any;

export function preprocess_rgba_to_float32(rgba: Uint8Array, channels: number, mean_r: number, mean_g: number, mean_b: number, std_r: number, std_g: number, std_b: number, bgr: boolean): Float32Array;

export function preprocess_rgba_to_int8(rgba: Uint8Array, channels: number, mean_r: number, mean_g: number, mean_b: number, std_r: number, std_g: number, std_b: number, scale: number, zero_point: number, bgr: boolean): Int8Array;

export function preprocess_rgba_to_uint8(rgba: Uint8Array, channels: number, mean_r: number, mean_g: number, mean_b: number, std_r: number, std_g: number, std_b: number, scale: number, zero_point: number, bgr: boolean): Uint8Array;

export function project_tflite_redesign(bytes: Uint8Array, filename: string, target_id: string, request: any): any;

export function protobuf_fields(bytes: Uint8Array): any;

export function protobuf_packed_floats(bytes: Uint8Array, width: number): Float64Array;

export function protobuf_packed_varints(bytes: Uint8Array): BigUint64Array;

export function protobuf_varint(bytes: Uint8Array): any;

export function runtime_guard(): number;

/**
 * Compute full G×G synthetic f64 loss landscape for one seed.
 * Weights perturbed in float domain (no requantization). Returns drift vs center.
 */
export function synthetic_landscape_grid(bytes: Uint8Array, seed1: number, seed2: number, grid_size: number, radius: number): any;

export function target_profiles(): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly compute_delegation_repair: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly compute_deployment_delta: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
    readonly compute_deployment_frontier: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly preprocess_rgba_to_float32: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
    readonly preprocess_rgba_to_int8: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => void;
    readonly preprocess_rgba_to_uint8: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => void;
    readonly compute_kernel_haar_decomposition: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly compute_activation_haar: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly landscape_directions: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly synthetic_landscape_grid: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly compute_model_tomography: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly landscape_tomography: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly layer_landscape_grid: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly numerical_abi_version: (a: number) => void;
    readonly numerical_count_ratio: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly numerical_shape_count: (a: number, b: number, c: number) => void;
    readonly __wbg_numericalmoments_free: (a: number, b: number) => void;
    readonly numericalmoments_new: (a: number) => number;
    readonly numericalmoments_add_chunk: (a: number, b: number, c: number, d: number) => void;
    readonly numericalmoments_totals: (a: number, b: number) => void;
    readonly numericalmoments_mean: (a: number, b: number, c: number) => void;
    readonly numericalmoments_divided_by: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly numericalmoments_coefficient_of_variation: (a: number, b: number, c: number) => void;
    readonly numericalmoments_finish: (a: number, b: number, c: number) => void;
    readonly __wbg_numericalcomparison_free: (a: number, b: number) => void;
    readonly numericalcomparison_new: () => number;
    readonly numericalcomparison_add_chunk: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly numericalcomparison_merge: (a: number, b: number, c: number) => void;
    readonly numericalcomparison_duplicate: (a: number, b: number) => void;
    readonly numericalcomparison_regression_gain: (a: number, b: number) => void;
    readonly numericalcomparison_cosine: (a: number, b: number) => void;
    readonly numericalcomparison_finish: (a: number, b: number) => void;
    readonly numerical_histogram_edges: (a: number) => void;
    readonly __wbg_numericalstatistics_free: (a: number, b: number) => void;
    readonly numericalstatistics_new: () => number;
    readonly numericalstatistics_add_chunk: (a: number, b: number, c: number, d: number) => void;
    readonly numericalstatistics_add_integer: (a: number, b: number, c: number, d: number) => void;
    readonly numericalstatistics_finish: (a: number, b: number) => void;
    readonly numerical_contract: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly evidence_sha256: (a: number, b: number, c: number) => void;
    readonly protobuf_fields: (a: number, b: number, c: number) => void;
    readonly protobuf_packed_varints: (a: number, b: number, c: number) => void;
    readonly protobuf_varint: (a: number, b: number, c: number) => void;
    readonly protobuf_packed_floats: (a: number, b: number, c: number, d: number) => void;
    readonly numerical_quality_metric: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
    readonly numerical_metric_methods: (a: number) => void;
    readonly numerical_spectrum: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly numerical_spectrum_summary: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly numerical_weight_permute: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly numerical_weight_magnitude: (a: number, b: number, c: number) => void;
    readonly numerical_weight_matrix: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly numerical_weight_similarity: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly numerical_weight_tiles: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly numerical_weight_sparsity: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly numerical_weight_comparison: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly numerical_weight_differences: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly numerical_distribution: (a: number, b: number) => void;
    readonly numerical_distribution_comparison: (a: number, b: number, c: number) => void;
    readonly numerical_affine: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => void;
    readonly numerical_pruning_prepare: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly numerical_pruning_simulate: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly runtime_guard: () => number;
    readonly analyze_tflite: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly analyze_tflite_for_target: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly project_tflite_redesign: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly explore_tflite_redesign_pareto: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly compute_weight_histogram: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly compute_input_influence: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly compute_output_influence: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly compute_quick_low_norm_stat: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly compute_static_runtime_calibration: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly target_profiles: (a: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
