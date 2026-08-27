export const SAFETENSORS_QUANTIZATION_SOURCES = Object.freeze({
  awq: Object.freeze({
    repository: "casper-hansen/AutoAWQ",
    commit: "5f3785dcaa107ca76f5fa5355f459370c86f82d6",
    path: "awq/modules/linear/gemm.py",
    sha256: "073e1faa28ce167349606ee2d0ad6362a1f978fe08ca07298da3c6b977123081",
    layout: "WQLinear_GEMM",
  }),
  gptq: Object.freeze({
    repository: "ModelCloud/GPTQModel",
    release: "7.3.5",
    commit: "a752a7cebeaa5feb499e0b038895f745572ab0fa",
    path: "gptqmodel/nn_modules/qlinear/__init__.py",
    sha256: "63c03283af6a1452b1427f88340e992795dfc8169e06f52a4500d692b01486d4",
    layout: "GPTQQuantLinear classic INT32-compatible packed state",
  }),
  hqq: Object.freeze({
    repository: "mobiusml/hqq",
    commit: "d88a488ec8aa2d58362ef2038a52bca862db2e74",
    path: "hqq/core/quantize.py",
    sha256: "a76a054ba4382d9552f563c468296771cddf53dcd9f4ce6655f94fe18e653daa",
    layout: "HQQLinear encoded SafeTensors state",
  }),
  "compressed-tensors": Object.freeze({
    repository: "vllm-project/compressed-tensors",
    release: "0.9.4",
    commit: "8aa8b822c8f6dac6272bb0749a498e3297b88510",
    path: "src/compressed_tensors/compressors/quantized_compressors/pack_quantized.py",
    sha256: "48f207964f1ec22a762dc8e0902c34d8adb51247d448549df6a42a2f43ede5e3",
    layout: "pack-quantized",
  }),
});

export const SAFETENSORS_QUANTIZATION_AUXILIARY_SOURCES = Object.freeze({
  hqq: Object.freeze([
    Object.freeze({ path: "hqq/core/bitpack.py", sha256: "19499f1c72e5f2422e7c90d83ab370e006bbbbad0ede3020b7e356688d99fca8" }),
    Object.freeze({ path: "hqq/core/utils.py", sha256: "808316e4337f6a6305b97844c729a5b70a072052ae5c1c69b793834a143fdf60" }),
    Object.freeze({ path: "hqq/models/base.py", sha256: "4823d53d2992a0eada577cc8bc5c58202eca33d15b792ec50dc03ab63fdf7dc6" }),
    Object.freeze({
      repository: "huggingface/transformers",
      commit: "8cb5963cc22174954e7dca2c0a3320b7dc2f4edc",
      path: "src/transformers/utils/quantization_config.py",
      sha256: "30af0253da4122a03c4d758cada0dfaccdd89ddb725b7f698f45be135ddb5358",
    }),
  ]),
  "compressed-tensors": Object.freeze([
    Object.freeze({ path: "src/compressed_tensors/quantization/quant_config.py", sha256: "33149471be5d3aa30de8b1720f92b687c5f44ef12d716200a22509a83d9d1c61" }),
    Object.freeze({ path: "src/compressed_tensors/quantization/quant_args.py", sha256: "b0aa5bf0370d39b89d8d78cda25d539ff222877657072712a7e7f2cfaf131a16" }),
    Object.freeze({ path: "src/compressed_tensors/quantization/quant_scheme.py", sha256: "cb3d2831b6f0a7b4195d777693928826ef90e9a4ea836f76f55754cd9eefcac3" }),
    Object.freeze({ path: "src/compressed_tensors/quantization/lifecycle/apply.py", sha256: "391fba42637da45446b5e60c0406adbb64f9a87bad42dec8c378c7e0320bbc49" }),
    Object.freeze({ path: "src/compressed_tensors/quantization/lifecycle/initialize.py", sha256: "498e3e1495a9552ba9423bafcbbaeb21cd2915853d7112f92e8d4ac9f512be85" }),
  ]),
});

export const SAFETENSORS_QUANTIZATION_CONTRACT_SCHEMA = "deepbom.safetensors_quantization_contract.v1.2";

const FIELDS = Object.freeze(["method", "bits", "group_size", "zero_point", "sym", "version", "desc_act", "checkpoint_format"]);

export function buildSafeTensorsQuantizationContract(config, tensors, { sidecars = {} } = {}) {
  const declarations = quantizationDeclarations(config, sidecars);
  if (!declarations.length) return unavailable("not_applicable_no_quantization_declaration", "No supported quantization declaration was selected with the SafeTensors repository.");
  const normalized = declarations.map(normalizeDeclaration);
  const declaredMethods = unique(normalized.map((row) => row.method).filter(Boolean));
  if (declaredMethods.length !== 1 || !SAFETENSORS_QUANTIZATION_SOURCES[declaredMethods[0]]) {
    return unavailable("not_assessed_unsupported_or_conflicting_method", `Expected one supported quantization method; observed ${declaredMethods.join(", ") || "none"}.`, { declarations: normalized });
  }
  const method = declaredMethods[0];
  if (method === "hqq") return buildHqqContract(declarations, tensors);
  if (method === "compressed-tensors") return buildCompressedTensorsContract(declarations, tensors);
  const conflicts = declarationConflicts(normalized);
  const merged = mergeDeclarations(normalized);
  const configIssues = validateConfig(method, merged);
  const modules = quantizedModules(tensors).map((module) => method === "awq" ? assessAwqModule(module, merged) : assessGptqModule(module, merged));
  if (!modules.length) configIssues.push("quantization_declaration_has_no_packed_module_tensors");
  const issueCount = conflicts.length + configIssues.length + modules.reduce((sum, row) => sum + row.issues.length, 0);
  const source = SAFETENSORS_QUANTIZATION_SOURCES[method];
  return {
    schema: SAFETENSORS_QUANTIZATION_CONTRACT_SCHEMA,
    status: issueCount ? "fail" : "assessed",
    evidence_class: "OBSERVED/DERIVED_FROM_PINNED_FORMAT_SOURCE",
    method,
    bits: merged.bits ?? null,
    group_size: merged.group_size ?? null,
    zero_point: merged.zero_point ?? null,
    symmetric: merged.sym ?? null,
    implementation_version: merged.version ?? null,
    checkpoint_format: merged.checkpoint_format ?? null,
    pack_word_bits: 32,
    pack_factor: validBits(merged.bits) ? 32 / merged.bits : null,
    storage_word_bits: 32,
    codes_per_storage_word: validBits(merged.bits) ? 32 / merged.bits : null,
    granularity: "per_group_weight",
    logical_group_axis: "input_features",
    declaration_count: normalized.length,
    declarations: normalized,
    declaration_conflicts: conflicts,
    config_issues: configIssues,
    module_count: modules.length,
    valid_module_count: modules.filter((row) => row.status === "pass").length,
    invalid_module_count: modules.filter((row) => row.status !== "pass").length,
    logical_weight_element_count: decimalSum(modules, "logical_weight_element_count"),
    packed_weight_code_capacity: decimalSum(modules, "packed_weight_code_capacity"),
    logical_weight_bits: decimalSum(modules, "logical_weight_bits"),
    packed_weight_storage_bits: decimalSum(modules, "packed_weight_storage_bits"),
    packing_padding_bits: decimalSum(modules, "packing_padding_bits"),
    packing_conservation_status: modules.every((row) => row.packing_conservation_status === "exact_no_padding") ? "exact_no_padding" : "invalid",
    scale_element_count: decimalSum(modules, "scale_element_count"),
    zero_point_code_capacity: decimalSum(modules, "zero_point_code_capacity"),
    packed_tensor_bytes: modules.reduce((sum, row) => sum + row.packed_tensor_bytes, 0),
    source,
    source_files: sourceFiles(method),
    modules,
    boundary: method === "gptq"
      ? "This contract validates the classic INT32 qweight/qzeros/scales/g_idx state against the pinned GPTQModel layout. Other GPTQModel pack dtypes, planar formats, calibration quality, reconstructed weight accuracy, kernel selection, runtime support, and task quality are not inferred."
      : "This contract validates selected config ownership, packed tensor identities, dtypes, shapes, group cardinality, and code capacity against one pinned implementation layout. It does not establish calibration quality, reconstructed weight accuracy, kernel selection, runtime support, or task quality.",
  };
}

function buildHqqContract(declarations, tensors) {
  const source = SAFETENSORS_QUANTIZATION_SOURCES.hqq;
  const parsed = declarations.map((row) => ({ path: row.path, ...hqqDeclaration(row.value) }));
  const missing = parsed.filter((row) => row.mode === "invalid").map((row) => row.path);
  if (missing.length) return unavailable("not_assessed_hqq_quantization_config_invalid", "HQQ quant_config does not contain a source-compatible static or module-tagged weight contract.", { source, source_files: sourceFiles("hqq"), declaration_paths: declarations.map((row) => row.path), unsupported_paths: missing });
  const signatures = unique(parsed.map((row) => JSON.stringify({ mode: row.mode, static_config: row.static_config, dynamic_configs: row.dynamic_configs, skip_modules: row.skip_modules })));
  const conflicts = signatures.length > 1 ? [{ field: "hqq_quant_config", values: signatures, paths: parsed.map((row) => row.path) }] : [];
  const declaration = parsed[0];
  const configIssues = [];
  const configuredRows = declaration.mode === "static" ? [declaration.static_config] : Object.values(declaration.dynamic_configs);
  configuredRows.forEach((config) => validateHqqConfig(config, configIssues));
  const serializedModules = hqqModules(tensors);
  const observedTags = new Set(serializedModules.map((module) => hqqLinearTag(module.name)));
  if (declaration.mode === "dynamic") {
    for (const tag of Object.keys(declaration.dynamic_configs)) if (!observedTags.has(tag)) configIssues.push(`dynamic_config_target_has_no_encoded_module:${tag}`);
  }
  const modules = serializedModules.map((module) => {
    const tag = hqqLinearTag(module.name);
    const config = declaration.mode === "static" ? declaration.static_config : declaration.dynamic_configs[tag] || null;
    const excluded = declaration.skip_modules.some((name) => module.name === name || module.name.endsWith(`.${name}`));
    return assessHqqModule(module, config, { tag, excluded, ownership_mode: declaration.mode });
  });
  if (!modules.length) configIssues.push("hqq_declaration_has_no_encoded_W_q_modules");
  const bits = commonModuleValue(modules, "bits");
  const groupSize = commonModuleValue(modules, "group_size");
  const axis = commonModuleValue(modules, "logical_weight_axis");
  return packedContract({ method: "hqq", declarations: parsed, conflicts, configIssues, modules, source,
    bits, groupSize, symmetric: null, zeroPoint: null,
    granularity: declaration.mode === "dynamic" ? "module_scoped_per_group_weight" : "per_group_weight", logicalGroupAxis: axis,
    storageWordBits: bits == null ? null : HQQ_PACKING[bits]?.storageBits ?? null,
    codesPerStorageWord: bits == null ? null : HQQ_PACKING[bits]?.codes ?? null,
    extra: {
      configuration_scope: declaration.mode === "dynamic" ? "source_bound_module_tag" : "global_static",
      module_tag_rule: "Remove path components equal to model or layers and every numeric component, then join the remainder with dots.",
      configured_module_tag_count: declaration.mode === "dynamic" ? Object.keys(declaration.dynamic_configs).length : null,
      shard_ownership: shardOwnership(modules),
    },
    boundary: "This contract validates HQQ encoded SafeTensors metadata, module-tag ownership, source-defined grouping, packed shape, shard ownership, and bit capacity. It does not infer reconstructed values, calibration quality, runtime backend, or task quality.",
  });
}

function buildCompressedTensorsContract(declarations, tensors) {
  const source = SAFETENSORS_QUANTIZATION_SOURCES["compressed-tensors"];
  const parsed = declarations.map((row) => ({ path: row.path, config: compressedQuantizationConfig(row.value) }));
  const unsupported = parsed.filter((row) => !row.config).map((row) => row.path);
  if (unsupported.length) return unavailable("not_assessed_compressed_tensors_config_invalid", "compressed-tensors analysis requires a pack-quantized config_groups object with named targets.", { source, source_files: sourceFiles("compressed-tensors"), declaration_paths: declarations.map((row) => row.path), unsupported_paths: unsupported });
  const signatures = unique(parsed.map((row) => JSON.stringify(row.config)));
  const conflicts = signatures.length > 1 ? [{ field: "compressed_tensors_config", values: signatures, paths: parsed.map((row) => row.path) }] : [];
  const config = parsed[0].config;
  const configIssues = [];
  const modules = compressedTensorsModules(tensors).filter((module) => module.weight_packed).map((module) => {
    const selection = selectCompressedScheme(module.name, config);
    if (selection.status !== "selected") return emptyPackedModule(module, [selection.reason], selection);
    const weights = selection.scheme.weights;
    const weightIssues = compressedWeightConfigIssues(weights);
    if (weightIssues.length) return emptyPackedModule(module, weightIssues, selection, activationContracts(module, selection.scheme));
    return assessCompressedTensorsModule(module, weights, selection);
  });
  if (!modules.length) configIssues.push("compressed_tensors_declaration_has_no_weight_packed_modules");
  const moduleActivationContracts = modules.flatMap((row) => row.activation_contracts || []);
  const standaloneActivationContracts = standaloneCompressedActivationContracts(compressedTensorsModules(tensors), modules, config);
  const allActivationContracts = [...moduleActivationContracts, ...standaloneActivationContracts];
  for (const contract of standaloneActivationContracts) {
    for (const issue of contract.issues || []) configIssues.push(`${contract.module_name}:${contract.kind}:${issue}`);
  }
  const bits = commonModuleValue(modules, "bits");
  const groupSize = commonModuleValue(modules, "group_size");
  const symmetric = commonModuleValue(modules, "symmetric");
  return packedContract({ method: "compressed-tensors", declarations: parsed.map((row) => ({ path: row.path, ...row.config })), conflicts, configIssues, modules, source,
    bits, groupSize, symmetric, zeroPoint: symmetric == null ? null : !symmetric,
    granularity: "module_scoped_weight", logicalGroupAxis: 1, storageWordBits: 32, codesPerStorageWord: null,
    extra: {
      configuration_scope: "source_bound_target_precedence",
      target_precedence: ["global ignore", "exact module name", "regex module name", "module class name"],
      config_group_count: config.groups.length,
      activation_quantization_contracts: allActivationContracts,
      activation_quantization_contract_count: allActivationContracts.length,
      shard_ownership: shardOwnership(modules, allActivationContracts),
      regex_contract: "Python re.match semantics are reproduced only for the explicitly accepted ASCII-compatible regex subset; unsupported constructs fail closed during config parsing.",
    },
    boundary: "This contract applies the pinned compressed-tensors target and ignore precedence per packed Linear module, validates source-defined pack-quantized layout, original shape metadata, group scale cardinality, optional packed zero points, activation companion ownership, shard ownership, and exact INT32 storage padding. It does not infer dynamic activation values, unpacked weights, runtime kernels, or task quality.",
  });
}

const HQQ_PACKING = Object.freeze({
  8: Object.freeze({ name: "8bit_u8", dtype: "U8", storageBits: 8, codes: 1 }),
  6: Object.freeze({ name: "8bit_u8", dtype: "U8", storageBits: 8, codes: 1 }),
  5: Object.freeze({ name: "8bit_u8", dtype: "U8", storageBits: 8, codes: 1 }),
  4: Object.freeze({ name: "4bit_u8", dtype: "U8", storageBits: 8, codes: 2 }),
  3: Object.freeze({ name: "3bit_32", dtype: "I32", storageBits: 32, codes: 10 }),
  2: Object.freeze({ name: "2bit_u8", dtype: "U8", storageBits: 8, codes: 4 }),
  1.58: Object.freeze({ name: "2bit_u8", dtype: "U8", storageBits: 8, codes: 4 }),
  1: Object.freeze({ name: "1bit_u8", dtype: "U8", storageBits: 8, codes: 8 }),
});

function hqqWeightConfig(value) {
  const candidate = object(value?.weight_quant_params) ? value.weight_quant_params : value;
  if (!object(candidate)) return null;
  const bits = finiteNumber(candidate.nbits ?? candidate.bits);
  const groupSize = candidate.group_size == null ? null : integer(candidate.group_size);
  const axis = integer(candidate.axis ?? 1);
  const viewAsFloat = boolean(candidate.view_as_float) ?? false;
  const channelWise = boolean(candidate.channel_wise) ?? true;
  if (bits == null || axis == null) return null;
  return { bits, group_size: groupSize, axis, view_as_float: viewAsFloat, channel_wise: channelWise };
}

function hqqDeclaration(value) {
  const root = object(value?.quant_config) ? value.quant_config : value;
  const skipModules = Array.isArray(value?.skip_modules) ? value.skip_modules.filter((item) => typeof item === "string") : [];
  const staticConfig = hqqWeightConfig(root);
  if (staticConfig) return { mode: "static", static_config: staticConfig, dynamic_configs: {}, skip_modules: skipModules };
  if (!object(root)) return { mode: "invalid", static_config: null, dynamic_configs: {}, skip_modules: skipModules };
  const dynamicConfigs = {};
  for (const [tag, candidate] of Object.entries(root)) {
    const config = hqqWeightConfig(candidate);
    if (!tag || !config) return { mode: "invalid", static_config: null, dynamic_configs: {}, skip_modules: skipModules };
    dynamicConfigs[tag] = config;
  }
  return Object.keys(dynamicConfigs).length
    ? { mode: "dynamic", static_config: null, dynamic_configs: dynamicConfigs, skip_modules: skipModules }
    : { mode: "invalid", static_config: null, dynamic_configs: {}, skip_modules: skipModules };
}

function validateHqqConfig(config, issues) {
  if (!HQQ_PACKING[config.bits]) issues.push("hqq_bits_not_source_registered");
  if (config.group_size != null && (!Number.isSafeInteger(config.group_size) || config.group_size < 1 || config.group_size % 8)) issues.push("hqq_group_size_must_be_null_or_positive_multiple_of_8");
  if (![0, 1].includes(config.axis)) issues.push("hqq_axis_must_be_zero_or_one");
  if (config.channel_wise !== true) issues.push("hqq_encoded_group_contract_requires_channel_wise_true");
  if (config.view_as_float === true) issues.push("hqq_float_view_storage_not_supported_without_compute_dtype_contract");
}

function hqqLinearTag(name) {
  return String(name || "").split(".").filter((part) => part !== "model" && part !== "layers" && !/^\d+$/.test(part)).join(".");
}

function compressedQuantizationArgs(value) {
  if (!object(value)) return null;
  const groupSize = value.group_size == null ? null : integer(value.group_size);
  const strategy = lower(value.strategy) ?? (groupSize > 0 ? "group" : groupSize === -1 ? "channel" : "tensor");
  return {
    bits: integer(value.num_bits ?? 8), group_size: groupSize,
    symmetric: boolean(value.symmetric) ?? true,
    strategy, type: lower(value.type || "int"), dynamic: boolean(value.dynamic) ?? false,
    actorder: lower(value.actorder),
  };
}

function compressedQuantizationConfig(value) {
  const format = lower(value?.format);
  if (format !== "pack-quantized" || !object(value?.config_groups)) return null;
  const groups = [];
  for (const [name, scheme] of Object.entries(value.config_groups)) {
    if (!object(scheme) || !Array.isArray(scheme.targets) || !scheme.targets.length || scheme.targets.some((target) => typeof target !== "string" || !target)) return null;
    for (const target of scheme.targets) if (target.startsWith("re:") && !portablePythonRegex(target.slice(3))) return null;
    groups.push({
      name,
      targets: [...scheme.targets],
      weights: scheme.weights == null ? null : compressedQuantizationArgs(scheme.weights),
      input_activations: scheme.input_activations == null ? null : compressedQuantizationArgs(scheme.input_activations),
      output_activations: scheme.output_activations == null ? null : compressedQuantizationArgs(scheme.output_activations),
    });
  }
  if (!groups.length) return null;
  if (object(value.kv_cache_scheme)) groups.push({
    name: "kv_cache",
    targets: ["re:.*self_attn$"],
    weights: null,
    input_activations: null,
    output_activations: compressedQuantizationArgs(value.kv_cache_scheme),
    kv_cache: true,
  });
  const ignore = Array.isArray(value.ignore) ? value.ignore.filter((target) => typeof target === "string" && target) : [];
  for (const target of ignore) if (target.startsWith("re:") && !portablePythonRegex(target.slice(3))) return null;
  return {
    format,
    groups,
    ignore,
  };
}

function hqqModules(tensors) {
  return collectModules(tensors, /^(.*)\.(W_q|scale|zero|shape|nbits|group_size|axis|packing|view_as_float)$/);
}

function compressedTensorsModules(tensors) {
  return collectModules(tensors, /^(.*)\.(weight_packed|weight_scale|weight_shape|weight_zero_point|weight_g_idx|input_scale|input_zero_point|output_scale|output_zero_point|k_scale|v_scale)$/);
}

function selectCompressedScheme(moduleName, config, moduleClass = "Linear") {
  const ignored = compressedMatches(moduleName, moduleClass, config.ignore);
  if (ignored.length) return { status: "ignored", reason: "packed_module_matches_global_ignore", matched_ignore_targets: ignored };
  const targetToGroup = new Map();
  for (const group of config.groups) for (const target of group.targets) targetToGroup.set(target, group);
  const targets = compressedMatches(moduleName, moduleClass, [...targetToGroup.keys()]);
  if (!targets.length) return { status: "unmatched", reason: "packed_module_has_no_matching_config_target", matched_targets: [] };
  const groups = targets.map((target) => targetToGroup.get(target));
  const nonKv = groups.filter((group) => !group.kv_cache);
  const selected = nonKv[0] || groups[0];
  const kv = groups.find((group) => group.kv_cache) || null;
  if (!selected.weights && !kv) return { status: "unmatched", reason: "matching_config_target_has_no_weight_scheme", matched_targets: targets };
  if (kv && selected !== kv && selected.output_activations && kv.output_activations) {
    return { status: "conflict", reason: "kv_cache_and_module_scheme_both_define_output_activations", matched_targets: targets };
  }
  return {
    status: "selected",
    matched_targets: targets,
    selected_group: selected.name,
    target_precedence_match: targets[0],
    scheme: kv && selected !== kv ? { ...selected, output_activations: kv.output_activations, kv_cache_merged: true } : selected,
  };
}

function compressedMatches(name, moduleClass, targets) {
  const sorted = [...(targets || [])].sort((left, right) => {
    const leftRegex = left.startsWith("re:") ? 1 : 0;
    const rightRegex = right.startsWith("re:") ? 1 : 0;
    return leftRegex - rightRegex || left.localeCompare(right);
  });
  const matchValue = (value) => sorted.filter((target) => {
    if (target.startsWith("re:")) {
      try { return new RegExp(target.slice(3)).exec(value)?.index === 0; }
      catch { return false; }
    }
    return target === value;
  });
  return [...matchValue(name), ...(moduleClass ? matchValue(moduleClass) : [])];
}

function portablePythonRegex(pattern) {
  if (!pattern || !/^[\x20-\x7e]+$/.test(pattern)) return false;
  // Python-only assertions, inline flags, named groups, conditionals, and
  // Unicode-property escapes cannot be claimed equivalent in JavaScript.
  if (/\(\?|\\[AZ]|\\[pP]|\\N\{|\\g</.test(pattern)) return false;
  try { new RegExp(pattern); return true; } catch { return false; }
}

function compressedWeightConfigIssues(config) {
  const issues = [];
  if (!config) return ["selected_config_group_has_no_weight_scheme"];
  if (!Number.isSafeInteger(config.bits) || config.bits < 1 || config.bits > 8) issues.push("compressed_tensors_bits_must_be_between_1_and_8");
  if (!Number.isSafeInteger(config.group_size) || config.group_size < 1) issues.push("compressed_tensors_group_size_must_be_positive");
  if (config.type !== "int") issues.push("compressed_tensors_pack_quantized_weight_type_must_be_int");
  if (config.strategy !== "group") issues.push("compressed_tensors_pack_quantized_weight_strategy_must_be_group");
  if (config.dynamic !== false) issues.push("compressed_tensors_packed_weight_scheme_must_be_static");
  return issues;
}

function activationContracts(module, scheme) {
  const rows = [];
  for (const [kind, config] of [["input", scheme?.input_activations], ["output", scheme?.output_activations]]) {
    if (!config) continue;
    const scale = module[`${kind}_scale`] || null;
    const zero = module[`${kind}_zero_point`] || null;
    const issues = [];
    const expectedShape = config.strategy === "token" ? [1, 1] : [1];
    if (config.dynamic) {
      if (scale || zero) issues.push("dynamic_activation_contract_has_serialized_scale_or_zero_point");
    } else {
      if (!scale) issues.push("static_activation_scale_missing");
      else if (!same(scale.shape || [], expectedShape)) issues.push("static_activation_scale_shape_mismatch");
      if (!config.symmetric && !zero) issues.push("asymmetric_static_activation_zero_point_missing");
      if (zero && !same(zero.shape || [], expectedShape)) issues.push("static_activation_zero_point_shape_mismatch");
    }
    rows.push({
      module_name: module.name,
      kind: `${kind}_activation`,
      status: issues.length ? "fail" : config.dynamic ? "assessed_config_dynamic_runtime_values" : "assessed_static_serialized_companions",
      issues,
      config,
      scale_tensor: scale ? tensorRef(scale) : null,
      zero_point_tensor: zero ? tensorRef(zero) : null,
      boundary: config.dynamic ? "Scale and zero point are computed from runtime activation values and are not serialized numerical constants." : "Serialized companion identity and shape are validated; calibration quality and runtime quantization error are not inferred.",
    });
  }
  return rows;
}

function standaloneCompressedActivationContracts(allModules, packedModules, config) {
  const packedNames = new Set(packedModules.map((module) => module.name));
  const rows = [];
  for (const module of allModules.filter((row) => !packedNames.has(row.name))) {
    if (module.k_scale || module.v_scale) {
      const kv = config.groups.find((group) => group.kv_cache && compressedMatches(module.name, null, group.targets).length);
      const issues = [];
      if (!kv) issues.push("serialized_kv_scale_has_no_matching_kv_cache_scheme");
      for (const kind of ["k_scale", "v_scale"]) if (module[kind]) {
        if (!same(module[kind].shape || [], [1])) issues.push(`${kind}_shape_must_be_scalar_cardinality_one`);
      }
      rows.push({
        module_name: module.name,
        kind: "kv_cache",
        status: issues.length ? "fail" : "assessed_static_serialized_kv_scales",
        issues,
        config: kv?.output_activations || null,
        k_scale_tensor: module.k_scale ? tensorRef(module.k_scale) : null,
        v_scale_tensor: module.v_scale ? tensorRef(module.v_scale) : null,
        boundary: "Serialized KV scale tensor identity and scalar cardinality are validated; cache allocation, runtime values, and attention execution are not inferred.",
      });
      continue;
    }
    const selection = selectCompressedScheme(module.name, config, null);
    if (selection.status === "selected") rows.push(...activationContracts(module, selection.scheme));
  }
  return rows;
}

function collectModules(tensors, pattern) {
  const map = new Map();
  for (const tensor of tensors || []) {
    const match = String(tensor.name || "").match(pattern);
    if (!match) continue;
    if (!map.has(match[1])) map.set(match[1], { name: match[1] });
    map.get(match[1])[match[2]] = tensor;
  }
  return [...map.values()].sort((left, right) => left.name.localeCompare(right.name));
}

function assessHqqModule(module, config, ownership = {}) {
  const issues = required(module, ["W_q", "scale", "zero", "shape", "nbits", "group_size", "axis", "packing"]);
  const shape = decodedIntegerVector(module.shape, "shape", issues);
  const bits = decodedScalarNumber(module.nbits, "nbits", issues);
  const groupSize = decodedScalarNumber(module.group_size, "group_size", issues);
  const axis = decodedScalarNumber(module.axis, "axis", issues);
  const packing = decodedAscii(module.packing, "packing", issues);
  const viewAsFloat = module.view_as_float ? decodedScalarNumber(module.view_as_float, "view_as_float", issues) : 0;
  if (shape && (shape.length !== 2 || shape.some((value) => !Number.isSafeInteger(value) || value < 1))) issues.push("shape_metadata_must_be_positive_rank_two");
  if (!config) issues.push("module_not_selected_by_hqq_dynamic_config");
  if (ownership.excluded) issues.push("encoded_module_matches_hqq_skip_modules");
  const expectedGroupSize = config && shape ? config.group_size ?? (config.axis === 1 ? shape[1] : shape[0]) : config?.group_size ?? null;
  if (bits != null && config && bits !== config.bits) issues.push("nbits_metadata_config_mismatch");
  if (groupSize != null && expectedGroupSize != null && groupSize !== expectedGroupSize) issues.push("group_size_metadata_config_mismatch");
  if (axis != null && config && axis !== config.axis) issues.push("axis_metadata_config_mismatch");
  if (config?.view_as_float === true || viewAsFloat !== 0) issues.push("view_as_float_metadata_not_supported");
  const effectiveBits = bits ?? config?.bits ?? null;
  const effectiveAxis = axis ?? config?.axis ?? null;
  const effectiveGroupSize = groupSize ?? expectedGroupSize;
  const layout = HQQ_PACKING[effectiveBits];
  if (packing != null && packing !== layout?.name) issues.push("packing_metadata_source_layout_mismatch");
  const logical = shape ? BigInt(shape[0]) * BigInt(shape[1]) : 0n;
  if (!Number.isSafeInteger(effectiveGroupSize) || effectiveGroupSize < 1) issues.push("effective_group_size_not_resolved");
  if (![0, 1].includes(effectiveAxis)) issues.push("effective_axis_not_resolved");
  if (!layout) issues.push("effective_hqq_bits_not_source_registered");
  if (logical && Number.isSafeInteger(effectiveGroupSize) && logical % BigInt(effectiveGroupSize)) issues.push("logical_weight_count_not_divisible_by_group_size");
  const grouped = logical && Number.isSafeInteger(effectiveGroupSize) && logical % BigInt(effectiveGroupSize) === 0n ? Number(logical / BigInt(effectiveGroupSize)) : null;
  const groupedShape = grouped == null || ![0, 1].includes(effectiveAxis) ? null : effectiveAxis === 1 ? [grouped, effectiveGroupSize] : [effectiveGroupSize, grouped];
  const scaleShape = groupedShape ? (effectiveAxis === 1 ? [groupedShape[0], 1] : [1, groupedShape[1]]) : null;
  const scale = shape2AnyFloat(module.scale, "scale", issues);
  const zero = shape2AnyFloat(module.zero, "zero", issues);
  if (scale && scaleShape && !same(scale, scaleShape)) issues.push("scale_shape_mismatch");
  if (zero && scaleShape && !same(zero, scaleShape)) issues.push("zero_shape_mismatch");
  const packedRows = groupedShape && layout ? Math.ceil(groupedShape[0] / layout.codes) : null;
  const expectedPacked = packedRows == null ? null : [packedRows, groupedShape[1]];
  const stored = shape2(module.W_q, "W_q", layout?.dtype || "unknown", issues);
  if (stored && expectedPacked && !same(stored, expectedPacked)) issues.push("W_q_shape_mismatch");
  if (layout && layout.codes !== 10 && groupedShape && groupedShape[0] % layout.codes) issues.push("source_packer_requires_first_axis_divisible_by_pack_factor");
  const physical = stored && layout ? product(stored) * BigInt(layout.storageBits) : 0n;
  const logicalBits = exactBitCount(logical, effectiveBits);
  const padding = physical - logicalBits;
  if (padding < 0n) issues.push("packed_storage_capacity_below_logical_bits");
  if (layout?.codes === 10 && padding >= BigInt(stored?.[1] || 0) * 32n) issues.push("hqq_3bit_padding_exceeds_one_padded_pack_row");
  return packedModuleResult(module, issues, {
    input: shape?.[1] ?? null, output: shape?.[0] ?? null, groupSize: effectiveGroupSize,
    groups: grouped, bits: effectiveBits, logical, physical, padding, packedCodeCapacity: stored && layout ? product(stored) * BigInt(layout.codes) : 0n,
    scales: scale ? product(scale) : 0n, zeroes: zero ? product(zero) : 0n,
    storageBits: layout?.storageBits ?? null, codesPerWord: layout?.codes ?? null, packing: layout?.name ?? null,
    tensors: ["W_q", "scale", "zero", "shape", "nbits", "group_size", "axis", "packing", "view_as_float"],
    logicalAxis: effectiveAxis, storedGroupAxis: effectiveAxis, zeroStorage: "floating_affine_zero",
    ownership,
  });
}

function assessCompressedTensorsModule(module, config, ownership = {}) {
  const requiredNames = ["weight_packed", "weight_scale", "weight_shape", ...(config.symmetric ? [] : ["weight_zero_point"]), ...(config.actorder === "group" || config.actorder === "dynamic" ? ["weight_g_idx"] : [])];
  const issues = required(module, requiredNames);
  const activations = activationContracts(module, ownership.scheme);
  for (const row of activations) for (const issue of row.issues) issues.push(`${row.kind}:${issue}`);
  const packed = shape2(module.weight_packed, "weight_packed", "I32", issues);
  const scale = shape2AnyFloat(module.weight_scale, "weight_scale", issues);
  const originalShape = decodedIntegerVector(module.weight_shape, "weight_shape", issues);
  if (module.weight_shape && !["I64", "I32"].includes(module.weight_shape.dtype)) issues.push("weight_shape_dtype_must_be_I64_or_I32");
  if (originalShape && (originalShape.length !== 2 || originalShape.some((value) => !Number.isSafeInteger(value) || value < 1))) issues.push("weight_shape_metadata_must_be_positive_rank_two");
  const output = originalShape?.[0] ?? packed?.[0] ?? null;
  const input = originalShape?.[1] ?? (scale ? scale[1] * config.group_size : null);
  const groups = input != null ? Math.ceil(input / config.group_size) : null;
  if (input != null && input % config.group_size) issues.push("input_features_not_divisible_by_group_size");
  if (packed && output != null && input != null && !same(packed, [output, Math.ceil(input * config.bits / 32)])) issues.push("weight_packed_shape_mismatch");
  if (scale && output != null && groups != null && !same(scale, [output, groups])) issues.push("weight_scale_shape_mismatch");
  if (originalShape && packed && packed[0] !== originalShape[0]) issues.push("weight_shape_output_mismatch");
  let zeroCardinality = 0n;
  if (config.symmetric && module.weight_zero_point) issues.push("weight_zero_point_present_for_symmetric_scheme");
  if (!config.symmetric && module.weight_zero_point) {
    const zero = shape2(module.weight_zero_point, "weight_zero_point", "I32", issues);
    const expected = output != null && groups != null ? [Math.ceil(output * config.bits / 32), groups] : null;
    if (zero && expected && !same(zero, expected)) issues.push("weight_zero_point_packed_shape_mismatch");
    zeroCardinality = zero ? product(zero) * BigInt(Math.floor(32 / config.bits)) : 0n;
  }
  if (module.weight_g_idx) {
    const gIdx = shape1(module.weight_g_idx, "weight_g_idx", "I32", issues);
    if (gIdx && input != null && gIdx[0] !== input) issues.push("weight_g_idx_shape_mismatch");
  }
  const logical = input != null && output != null ? BigInt(input) * BigInt(output) : 0n;
  const logicalBits = exactBitCount(logical, config.bits);
  const physical = packed ? product(packed) * 32n : 0n;
  const padding = physical - logicalBits;
  if (padding < 0n) issues.push("packed_storage_capacity_below_logical_bits");
  if (output != null && padding >= BigInt(output) * 32n) issues.push("packed_storage_padding_exceeds_one_word_per_row");
  return packedModuleResult(module, issues, {
    input, output, groupSize: config.group_size, groups, bits: config.bits, logical, physical, padding,
    packedCodeCapacity: logical, scales: scale ? product(scale) : 0n, zeroes: zeroCardinality,
    storageBits: 32, codesPerWord: null, packing: "dense_cross_element_int32",
    tensors: ["weight_packed", "weight_scale", "weight_shape", "weight_zero_point", "weight_g_idx"],
    logicalAxis: 1, storedGroupAxis: 1, zeroStorage: config.symmetric ? "omitted_symmetric_zero" : "packed_along_output_axis",
    symmetric: config.symmetric,
    ownership,
    activationContracts: activations,
  });
}

function emptyPackedModule(module, issues, ownership = {}, activations = []) {
  return packedModuleResult(module, [...issues], {
    input: null, output: null, groupSize: null, groups: null, bits: null,
    logical: 0n, physical: 0n, padding: 0n, packedCodeCapacity: 0n, scales: 0n, zeroes: 0n,
    storageBits: 32, codesPerWord: null, packing: "not_assessed",
    tensors: ["weight_packed", "weight_scale", "weight_shape", "weight_zero_point", "weight_g_idx", "input_scale", "input_zero_point", "output_scale", "output_zero_point"],
    logicalAxis: 1, storedGroupAxis: 1, zeroStorage: "not_assessed", symmetric: null,
    ownership, activationContracts: activations,
  });
}

function packedModuleResult(module, issues, values) {
  return {
    name: module.name, status: issues.length ? "fail" : "pass", issues,
    input_features: values.input, output_features: values.output,
    group_size: values.groupSize, group_count: values.groups,
    logical_weight_axis: values.logicalAxis, stored_group_axis: values.storedGroupAxis,
    bits: values.bits, storage_word_bits: values.storageBits, codes_per_storage_word: values.codesPerWord,
    symmetric: values.symmetric ?? null,
    packing_layout: values.packing,
    logical_weight_element_count: values.logical.toString(),
    packed_weight_code_capacity: values.packedCodeCapacity.toString(),
    logical_weight_bits: values.logicalBits?.toString?.() ?? exactBitCount(values.logical, values.bits).toString(),
    packed_weight_storage_bits: values.physical.toString(),
    packing_padding_bits: values.padding.toString(),
    packing_conservation_status: values.padding < 0n ? "invalid" : values.padding === 0n ? "exact_no_padding" : "exact_with_source_defined_padding",
    scale_element_count: values.scales.toString(), zero_point_code_capacity: values.zeroes.toString(),
    zero_point_storage_transform: values.zeroStorage,
    packed_tensor_bytes: values.tensors.filter((key) => module[key]).reduce((sum, key) => sum + Number(module[key].byte_length || 0), 0),
    tensors: Object.fromEntries(values.tensors.filter((key) => module[key]).map((key) => [key, tensorRef(module[key])])),
    ownership: values.ownership || null,
    activation_contracts: values.activationContracts || [],
    shard_paths: unique(values.tensors.filter((key) => module[key]?.shard_path).map((key) => module[key].shard_path)),
  };
}

function packedContract({ method, declarations, conflicts, configIssues, modules, source, bits, groupSize, symmetric, zeroPoint, granularity, logicalGroupAxis, storageWordBits, codesPerStorageWord, boundary, extra = {} }) {
  const issueCount = conflicts.length + configIssues.length + modules.reduce((sum, row) => sum + row.issues.length, 0);
  return {
    schema: SAFETENSORS_QUANTIZATION_CONTRACT_SCHEMA,
    status: issueCount ? "fail" : "assessed", evidence_class: "OBSERVED/DERIVED_FROM_PINNED_FORMAT_SOURCE",
    method, bits, group_size: groupSize, zero_point: zeroPoint, symmetric,
    implementation_version: null, checkpoint_format: method === "compressed-tensors" ? "pack-quantized" : "encoded_state_dict",
    pack_word_bits: storageWordBits, pack_factor: codesPerStorageWord,
    storage_word_bits: storageWordBits, codes_per_storage_word: codesPerStorageWord,
    granularity, logical_group_axis: logicalGroupAxis,
    declaration_count: declarations.length, declarations, declaration_conflicts: conflicts, config_issues: configIssues,
    module_count: modules.length, valid_module_count: modules.filter((row) => row.status === "pass").length,
    invalid_module_count: modules.filter((row) => row.status !== "pass").length,
    logical_weight_element_count: decimalSum(modules, "logical_weight_element_count"),
    packed_weight_code_capacity: decimalSum(modules, "packed_weight_code_capacity"),
    logical_weight_bits: decimalSum(modules, "logical_weight_bits"),
    packed_weight_storage_bits: decimalSum(modules, "packed_weight_storage_bits"),
    packing_padding_bits: decimalSum(modules, "packing_padding_bits"),
    packing_conservation_status: modules.every((row) => row.packing_conservation_status === "exact_no_padding") ? "exact_no_padding"
      : modules.every((row) => row.packing_conservation_status !== "invalid") ? "exact_with_source_defined_padding" : "invalid",
    scale_element_count: decimalSum(modules, "scale_element_count"), zero_point_code_capacity: decimalSum(modules, "zero_point_code_capacity"),
    packed_tensor_bytes: modules.reduce((sum, row) => sum + row.packed_tensor_bytes, 0),
    source, source_files: sourceFiles(method), modules, boundary, ...extra,
  };
}

function quantizationDeclarations(config, sidecars) {
  const rows = [];
  if (object(config?.quantization_config)) rows.push({ path: "config.json#/quantization_config", value: config.quantization_config });
  for (const key of ["quant_config", "quantize_config", "quantization_config"]) {
    const sidecar = sidecars?.[key];
    if (object(sidecar?.document)) rows.push({ path: sidecar.path || `${key}.json`, value: sidecar.document });
  }
  return rows;
}

function normalizeDeclaration(row) {
  const value = row.value;
  return {
    path: row.path,
    method: lower(value.quant_method),
    bits: integer(value.bits ?? value.w_bit),
    group_size: integer(value.group_size ?? value.q_group_size),
    zero_point: boolean(value.zero_point),
    sym: boolean(value.sym),
    version: lower(value.version),
    desc_act: boolean(value.desc_act),
    checkpoint_format: lower(value.checkpoint_format ?? value.format),
  };
}

function declarationConflicts(rows) {
  const issues = [];
  for (const field of FIELDS) {
    const values = unique(rows.map((row) => row[field]).filter((value) => value != null));
    if (values.length > 1) issues.push({ field, values, paths: rows.filter((row) => row[field] != null).map((row) => row.path) });
  }
  return issues;
}

function mergeDeclarations(rows) {
  const merged = {};
  for (const field of FIELDS) merged[field] = rows.map((row) => row[field]).find((value) => value != null) ?? null;
  return merged;
}

function validateConfig(method, config) {
  const issues = [];
  if (!validBits(config.bits)) issues.push("bits_must_be_a_positive_divisor_of_32");
  if (!Number.isSafeInteger(config.group_size) || config.group_size === 0 || config.group_size < -1) issues.push("group_size_must_be_positive_or_minus_one");
  if (method === "awq" && config.zero_point == null) issues.push("awq_zero_point_declaration_missing");
  if (method === "awq" && config.version && config.version !== "gemm") issues.push("only_source_pinned_awq_gemm_layout_is_assessed");
  return issues;
}

function quantizedModules(tensors) {
  const map = new Map();
  for (const tensor of tensors || []) {
    const match = String(tensor.name || "").match(/^(.*)\.(qweight|qzeros|scales|g_idx)$/);
    if (!match) continue;
    if (!map.has(match[1])) map.set(match[1], { name: match[1] });
    map.get(match[1])[match[2]] = tensor;
  }
  return [...map.values()].sort((left, right) => left.name.localeCompare(right.name));
}

function assessAwqModule(module, config) {
  const issues = required(module, ["qweight", "scales", ...(config.zero_point === false ? [] : ["qzeros"])]);
  const pack = validBits(config.bits) ? 32 / config.bits : null;
  const qweight = shape2(module.qweight, "qweight", "I32", issues);
  const scales = shape2(module.scales, "scales", "F16", issues);
  const qzeros = module.qzeros ? shape2(module.qzeros, "qzeros", "I32", issues) : null;
  if (module.g_idx) issues.push("awq_gemm_layout_does_not_define_g_idx");
  const input = qweight?.[0] ?? null;
  const output = qweight && pack ? qweight[1] * pack : null;
  const groupSize = resolvedGroupSize(config.group_size, input);
  const groups = input != null && groupSize ? Math.ceil(input / groupSize) : null;
  if (input != null && groupSize && input % groupSize) issues.push("input_features_not_divisible_by_group_size");
  if (scales && groups != null && output != null && !same(scales, [groups, output])) issues.push("scales_shape_mismatch");
  if (qzeros && groups != null && qweight && !same(qzeros, [groups, qweight[1]])) issues.push("qzeros_shape_mismatch");
  if (config.zero_point === false && module.qzeros) issues.push("qzeros_present_while_zero_point_is_false");
  return moduleResult(module, issues, { input, output, groups, groupSize, pack, bits: config.bits, qweight, scales, qzeros, zeroStorage: "packed_unsigned_zero_code" });
}

function assessGptqModule(module, config) {
  const issues = required(module, ["qweight", "qzeros", "scales", "g_idx"]);
  const pack = validBits(config.bits) ? 32 / config.bits : null;
  const qweight = shape2(module.qweight, "qweight", "I32", issues);
  const scales = shape2(module.scales, "scales", "F16", issues);
  const qzeros = shape2(module.qzeros, "qzeros", "I32", issues);
  const gIdx = shape1(module.g_idx, "g_idx", "I32", issues);
  const input = gIdx?.[0] ?? (qweight && pack ? qweight[0] * pack : null);
  const output = qweight?.[1] ?? null;
  const groupSize = resolvedGroupSize(config.group_size, input);
  const groups = input != null && groupSize ? Math.ceil(input / groupSize) : null;
  if (input != null && pack && input % pack) issues.push("input_features_not_divisible_by_pack_factor");
  if (output != null && pack && output % pack) issues.push("output_features_not_divisible_by_pack_factor");
  if (qweight && input != null && output != null && pack && !same(qweight, [input / pack, output])) issues.push("qweight_shape_mismatch");
  if (scales && groups != null && output != null && !same(scales, [groups, output])) issues.push("scales_shape_mismatch");
  if (qzeros && groups != null && output != null && pack && !same(qzeros, [groups, output / pack])) issues.push("qzeros_shape_mismatch");
  return moduleResult(module, issues, { input, output, groups, groupSize, pack, bits: config.bits, qweight, scales, qzeros, gIdx, zeroStorage: "packed_zero_code_minus_one" });
}

function moduleResult(module, issues, values) {
  const logical = values.input != null && values.output != null ? BigInt(values.input) * BigInt(values.output) : 0n;
  const packed = values.qweight && values.pack ? product(values.qweight) * BigInt(values.pack) : 0n;
  const scales = values.scales ? product(values.scales) : 0n;
  const zeroCapacity = values.qzeros && values.pack ? product(values.qzeros) * BigInt(values.pack) : 0n;
  return {
    name: module.name,
    status: issues.length ? "fail" : "pass",
    issues,
    input_features: values.input,
    output_features: values.output,
    bits: values.bits,
    group_size: values.groupSize,
    group_count: values.groups,
    logical_weight_axis: 1,
    stored_group_axis: 0,
    logical_weight_element_count: logical.toString(),
    packed_weight_code_capacity: packed.toString(),
    logical_weight_bits: (logical * BigInt(values.bits || 0)).toString(),
    packed_weight_storage_bits: values.qweight ? (product(values.qweight) * 32n).toString() : "0",
    packing_padding_bits: values.qweight ? (product(values.qweight) * 32n - logical * BigInt(values.bits || 0)).toString() : "0",
    packing_conservation_status: packed === logical ? "exact_no_padding" : "invalid",
    scale_element_count: scales.toString(),
    zero_point_code_capacity: zeroCapacity.toString(),
    zero_point_storage_transform: values.zeroStorage,
    packed_tensor_bytes: [module.qweight, module.qzeros, module.scales, module.g_idx].filter(Boolean).reduce((sum, row) => sum + Number(row.byte_length || 0), 0),
    tensors: Object.fromEntries(["qweight", "qzeros", "scales", "g_idx"].filter((key) => module[key]).map((key) => [key, tensorRef(module[key])])),
  };
}

function required(module, names) {
  return names.filter((name) => !module[name]).map((name) => `missing_${name}`);
}

function sourceFiles(method) {
  const primary = SAFETENSORS_QUANTIZATION_SOURCES[method];
  return [primary, ...(SAFETENSORS_QUANTIZATION_AUXILIARY_SOURCES[method] || [])].map((row) => ({ ...row }));
}

function declarationConflictsFor(rows, fields) {
  const issues = [];
  for (const field of fields) {
    const values = [...new Map(rows.filter((row) => row[field] != null).map((row) => [JSON.stringify(row[field]), row[field]])).values()];
    if (values.length > 1) issues.push({ field, values, paths: rows.filter((row) => row[field] != null).map((row) => row.path) });
  }
  return issues;
}

function decodedValues(tensor, name, issues) {
  if (!tensor) return null;
  const integrity = tensor.numerical_integrity || {};
  if (!String(integrity.decoded_values_status || "").startsWith("complete_") || !Array.isArray(integrity.decoded_values)) {
    issues.push(`${name}_small_value_payload_not_decoded`);
    return null;
  }
  return integrity.decoded_values;
}

function decodedIntegerVector(tensor, name, issues) {
  const values = decodedValues(tensor, name, issues);
  if (!values) return null;
  const decoded = values.map(exactSafeInteger);
  if (decoded.some((value) => value == null)) {
    issues.push(`${name}_contains_non_integer_metadata`);
    return null;
  }
  return decoded;
}

function decodedScalarNumber(tensor, name, issues) {
  const values = decodedValues(tensor, name, issues);
  if (!values) return null;
  if (values.length !== 1) {
    issues.push(`${name}_metadata_must_be_scalar`);
    return null;
  }
  const value = Number(values[0]);
  if (!Number.isFinite(value)) {
    issues.push(`${name}_metadata_must_be_finite`);
    return null;
  }
  return value;
}

function decodedAscii(tensor, name, issues) {
  const values = decodedIntegerVector(tensor, name, issues);
  if (!values) return null;
  if (tensor.dtype !== "U8" || values.some((value) => value < 0 || value > 127)) {
    issues.push(`${name}_metadata_must_be_ascii_U8`);
    return null;
  }
  return String.fromCharCode(...values);
}

function exactSafeInteger(value) {
  try {
    const parsed = BigInt(String(value));
    return parsed >= BigInt(Number.MIN_SAFE_INTEGER) && parsed <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(parsed) : null;
  } catch {
    return null;
  }
}

function shape2AnyFloat(tensor, name, issues) {
  if (!tensor) return null;
  if (!["F16", "BF16", "F32", "F64"].includes(tensor.dtype)) issues.push(`${name}_dtype_${tensor.dtype || "unknown"}_expected_floating`);
  if (!Array.isArray(tensor.shape) || tensor.shape.length !== 2 || tensor.shape.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    issues.push(`${name}_must_have_positive_rank_two_shape`);
    return null;
  }
  return tensor.shape;
}

function shape2(tensor, name, dtype, issues) {
  if (!tensor) return null;
  if (tensor.dtype !== dtype) issues.push(`${name}_dtype_${tensor.dtype || "unknown"}_expected_${dtype}`);
  if (!Array.isArray(tensor.shape) || tensor.shape.length !== 2 || tensor.shape.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    issues.push(`${name}_must_have_positive_rank_two_shape`);
    return null;
  }
  return tensor.shape;
}

function shape1(tensor, name, dtype, issues) {
  if (!tensor) return null;
  if (tensor.dtype !== dtype) issues.push(`${name}_dtype_${tensor.dtype || "unknown"}_expected_${dtype}`);
  if (!Array.isArray(tensor.shape) || tensor.shape.length !== 1 || !Number.isSafeInteger(tensor.shape[0]) || tensor.shape[0] < 1) {
    issues.push(`${name}_must_have_positive_rank_one_shape`);
    return null;
  }
  return tensor.shape;
}

function tensorRef(tensor) {
  return { tensor_index: tensor.index ?? null, tensor_name: tensor.name, dtype: tensor.dtype, shape: tensor.shape, byte_length: Number(tensor.byte_length || 0), shard_path: tensor.shard_path || null };
}

function commonModuleValue(modules, field) {
  const values = [...new Map((modules || []).map((row) => row[field]).filter((value) => value != null).map((value) => [JSON.stringify(value), value])).values()];
  return values.length === 1 ? values[0] : null;
}

function shardOwnership(modules, activationContracts = []) {
  const rows = [];
  for (const module of modules || []) for (const [role, tensor] of Object.entries(module.tensors || {})) {
    rows.push({ module_name: module.name, tensor_role: role, tensor_name: tensor.tensor_name, shard_path: tensor.shard_path || null });
  }
  for (const contract of activationContracts || []) {
    for (const role of ["scale_tensor", "zero_point_tensor", "k_scale_tensor", "v_scale_tensor"]) {
      const tensor = contract[role];
      if (!tensor) continue;
      rows.push({ module_name: contract.module_name, tensor_role: `${contract.kind}:${role}`, tensor_name: tensor.tensor_name, shard_path: tensor.shard_path || null });
    }
  }
  const uniqueRows = [...new Map(rows.map((row) => [`${row.module_name}\u0000${row.tensor_role}\u0000${row.tensor_name}`, row])).values()];
  return {
    status: uniqueRows.some((row) => row.shard_path) ? uniqueRows.every((row) => row.shard_path) ? "assessed_all_quantization_tensors_shard_bound" : "fail_partial_shard_ownership" : "not_applicable_single_file_or_unannotated_input",
    tensor_count: uniqueRows.length,
    shard_bound_tensor_count: uniqueRows.filter((row) => row.shard_path).length,
    rows: uniqueRows,
  };
}

function unavailable(status, reason, extra = {}) {
  return { schema: SAFETENSORS_QUANTIZATION_CONTRACT_SCHEMA, status, evidence_class: "NOT_ASSESSED", reason, ...extra };
}

function resolvedGroupSize(value, input) { return value === -1 ? input : value; }
function validBits(value) { return Number.isSafeInteger(value) && value > 0 && value <= 32 && 32 % value === 0; }
function finiteNumber(value) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : null; }
function exactBitCount(elements, bits) {
  if (bits === 1.58) return elements * 2n;
  if (!Number.isSafeInteger(bits)) return 0n;
  return elements * BigInt(bits);
}
function object(value) { return value && typeof value === "object" && !Array.isArray(value); }
function integer(value) { return value == null || value === "" || !Number.isSafeInteger(Number(value)) ? null : Number(value); }
function boolean(value) { return typeof value === "boolean" ? value : null; }
function lower(value) { return typeof value === "string" && value.trim() ? value.trim().toLowerCase() : null; }
function unique(values) { return [...new Set(values.map((value) => typeof value === "string" ? value : JSON.stringify(value)))].map((value) => value === "true" ? true : value === "false" ? false : /^-?\d+$/.test(value) ? Number(value) : value); }
function same(left, right) { return left.length === right.length && left.every((value, index) => value === right[index]); }
function product(shape) { return shape.reduce((value, dimension) => value * BigInt(dimension), 1n); }
function decimalSum(rows, key) { return rows.reduce((sum, row) => sum + BigInt(row[key] || 0), 0n).toString(); }
