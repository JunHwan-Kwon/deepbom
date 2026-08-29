import { ProtoReader } from "./tflite-runtime-info-adapter.js";
import {
  COREML_NEURAL_NETWORK_SOURCE, finalizeCoreMlNeuralNetwork, parseCoreMlNeuralNetwork,
  parseCoreMlNeuralNetworkLayer, parseCoreMlNeuralNetworkPreprocessing,
} from "./coreml-neural-network.js";
import {
  COREML_MIL_SOURCE, finalizeCoreMlMilProgram, graphFromCoreMlMilProgram,
  parseCoreMlMilAttributeEntry, parseCoreMlMilFunctionEntry, parseCoreMlMilProgram,
  refreshCoreMlMilCompressionEvidence,
} from "./coreml-mil-program.js";
import { buildCoreMlDeploymentContract } from "./coreml-deployment-contract.js";
import { coreMlExactLedger, multiplyCoreMlExactIntegers } from "./coreml-exact-integer.js";
import {
  COREML_CLASSICAL_FIELDS, COREML_CLASSICAL_SOURCE, COREML_PIPELINE_FIELDS,
  parseCoreMlClassicalModel, parseCoreMlPipeline,
} from "./coreml-classical-model.js";

const MAX_MODEL_DESCRIPTION_BYTES = 16 * 1024 * 1024;
const MAX_PROTO_FIELDS = 1_000_000;
const MAX_FEATURES = 100_000;
const MAX_PIPELINE_DEPTH = 32;
const MAX_PIPELINE_MODELS = 4096;
const RANGE_READ_CHUNK_BYTES = 1024 * 1024;

export const COREML_FORMAT_SOURCE = Object.freeze({
  repository: "apple/coremltools",
  release: "9.0",
  source_commit: "428d4b2658dfc44194f27f4f36870751be402ff7",
  model_proto: "mlmodel/format/Model.proto",
  model_proto_sha256: "c731d0202acecd54f20eaf97c94a2a4764fe541ba92311485a9c36d3c3c3b544",
  feature_types_proto: "mlmodel/format/FeatureTypes.proto",
  feature_types_proto_sha256: "01f12c1220ce0cb19496b0497ca4e733a37ff12294382885bd25a6903d9547b7",
  package_source: "modelpackage/src/ModelPackage.cpp",
  package_source_sha256: "b5a806722610af6d94be33e3a28e471c51bda3d33328f437a6ff21996b0e3ef8",
  coremltools_init: "coremltools/__init__.py",
  coremltools_init_sha256: "ebbb958d3bc70c16c3c1d991f46fd3d25da5f08f1c12083f0634a7293a0cca81",
  deployment_compatibility: "coremltools/converters/mil/_deployment_compatibility.py",
  deployment_compatibility_sha256: "4b4601bc4afa4b90282052d933267ffdcfa78af42472e133393640e55f200fed",
  compute_plan: "coremltools/models/compute_plan.py",
  compute_plan_sha256: "977c84697df762a1958bd1c8e562db8e9cb95270254508983e11519c57f8e497",
  neural_network_proto: COREML_NEURAL_NETWORK_SOURCE.neural_network_proto,
  neural_network_proto_sha256: COREML_NEURAL_NETWORK_SOURCE.neural_network_proto_sha256,
  quantization_implementation: COREML_NEURAL_NETWORK_SOURCE.quantization_implementation,
  quantization_implementation_sha256: COREML_NEURAL_NETWORK_SOURCE.quantization_implementation_sha256,
  mil_proto: COREML_MIL_SOURCE.mil_proto,
  mil_proto_sha256: COREML_MIL_SOURCE.mil_proto_sha256,
  mil_conv_definition: COREML_MIL_SOURCE.conv_definition,
  mil_conv_definition_sha256: COREML_MIL_SOURCE.conv_definition_sha256,
  mil_linear_definition: COREML_MIL_SOURCE.linear_definition,
  mil_linear_definition_sha256: COREML_MIL_SOURCE.linear_definition_sha256,
  mil_compression_ios18_definition: COREML_MIL_SOURCE.compression_ios18_definition,
  mil_compression_ios18_definition_sha256: COREML_MIL_SOURCE.compression_ios18_definition_sha256,
  mil_constexpr_ios16_definition: COREML_MIL_SOURCE.constexpr_ios16_definition,
  mil_constexpr_ios16_definition_sha256: COREML_MIL_SOURCE.constexpr_ios16_definition_sha256,
});

const MODEL_TYPES = Object.freeze({
  200: "pipelineClassifier", 201: "pipelineRegressor", 202: "pipeline",
  300: "glmRegressor", 301: "supportVectorRegressor", 302: "treeEnsembleRegressor",
  303: "neuralNetworkRegressor", 304: "bayesianProbitRegressor",
  400: "glmClassifier", 401: "supportVectorClassifier", 402: "treeEnsembleClassifier",
  403: "neuralNetworkClassifier", 404: "kNearestNeighborsClassifier",
  500: "neuralNetwork", 501: "itemSimilarityRecommender", 502: "mlProgram",
  555: "customModel", 556: "linkedModel", 560: "classConfidenceThresholding",
  600: "oneHotEncoder", 601: "imputer", 602: "featureVectorizer", 603: "dictVectorizer",
  604: "scaler", 606: "categoricalMapping", 607: "normalizer", 609: "arrayFeatureExtractor",
  610: "nonMaximumSuppression", 900: "identity", 2000: "textClassifier", 2001: "wordTagger",
  2002: "visionFeaturePrint", 2003: "soundAnalysisPreprocessing", 2004: "gazetteer",
  2005: "wordEmbedding", 2006: "audioFeaturePrint", 3000: "serializedModel",
});

const ARRAY_DTYPES = Object.freeze({
  65568: "FLOAT32", 65600: "FLOAT64", 131104: "INT32", 131080: "INT8", 65552: "FLOAT16",
});
const IMAGE_COLOR = Object.freeze({ 10: ["IMAGE_GRAYSCALE8", 1], 20: ["IMAGE_RGB8", 3], 30: ["IMAGE_BGR8", 3], 40: ["IMAGE_GRAYSCALE_FLOAT16", 1] });
const LEGACY_NEURAL_NETWORK_FIELDS = new Set([303, 403, 500]);

function limitedPush(rows, value, label) {
  if (rows.length >= MAX_FEATURES) throw new Error(`${label} exceeds ${MAX_FEATURES} entries`);
  rows.push(value);
}

function parseMetadata(reader) {
  const result = { short_description: null, version: null, author: null, license: null, user_defined: {} };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field >= 1 && field <= 4) {
      const value = reader.stringField(wire, singular, field, `metadata_${field}`);
      if (field === 1) result.short_description = value;
      else if (field === 2) result.version = value;
      else if (field === 3) result.author = value;
      else result.license = value;
    } else if (field === 100) {
      const entry = reader.message(wire, "CoreML.Metadata.userDefined");
      let key = null;
      let value = null;
      const fields = new Set();
      while (!entry.done) {
        const item = entry.key();
        if (item.field === 1) key = entry.stringField(item.wire, fields, 1, "key");
        else if (item.field === 2) value = entry.stringField(item.wire, fields, 2, "value");
        else entry.skip(item.wire);
      }
      if (key == null || value == null) throw new Error("Core ML metadata map entry is incomplete");
      if (Object.hasOwn(result.user_defined, key)) throw new Error(`Core ML metadata repeats key ${key}`);
      result.user_defined[key] = value;
    } else reader.skip(wire);
  }
  return result;
}

function repeatedSignedInt64(reader, wire, name) {
  const values = [];
  const read = (source) => {
    const value = BigInt.asIntN(64, source.rawVarint());
    if (value < BigInt(Number.MIN_SAFE_INTEGER) || value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${name} exceeds the safe integer range`);
    values.push(Number(value));
  };
  if (wire === 0) read(reader);
  else if (wire === 2) {
    const packed = reader.message(wire, name);
    while (!packed.done) read(packed);
  } else throw new Error(`${name} has unsupported wire type ${wire}`);
  return values;
}

function unsignedInt64Field(reader, wire, singular, field, name) {
  reader.requireWire(wire, 0, name);
  if (singular.has(field)) throw new Error(`${reader.label} repeats ${name}`);
  singular.add(field);
  const value = reader.rawVarint();
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${name} exceeds the safe integer range`);
  return Number(value);
}

function parseSizeRange(reader, label) {
  const singular = new Set();
  let lower = 0;
  let upper = 0;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) lower = unsignedInt64Field(reader, wire, singular, 1, `${label}.lowerBound`);
    else if (field === 2) upper = reader.int64Field(wire, singular, 2, `${label}.upperBound`);
    else reader.skip(wire);
  }
  if (!Number.isSafeInteger(lower) || lower < 0 || !Number.isSafeInteger(upper) || upper >= 0 && upper < lower) {
    throw new Error(`${label} has an invalid lower/upper bound`);
  }
  return {
    lower_bound: lower,
    upper_bound: upper < 0 ? -1 : upper,
    upper_bound_unbounded: upper < 0,
    serialized_upper_bound: upper,
  };
}

function parseImageSize(reader, label) {
  const singular = new Set();
  let width = null;
  let height = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) width = reader.int64Field(wire, singular, 1, `${label}.width`);
    else if (field === 2) height = reader.int64Field(wire, singular, 2, `${label}.height`);
    else reader.skip(wire);
  }
  if (![width, height].every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error(`${label} must have positive width and height`);
  return { width, height };
}

function parseImageFlexibility(reader, kind) {
  if (kind === "enumerated") {
    const sizes = [];
    while (!reader.done) {
      const { field, wire } = reader.key();
      if (field === 1) limitedPush(sizes, parseImageSize(reader.message(wire, "CoreML.ImageSize"), "Core ML enumerated image size"), "Core ML enumerated image sizes");
      else reader.skip(wire);
    }
    if (!sizes.length) throw new Error("Core ML enumerated image sizes are empty");
    return { kind: "enumerated", sizes };
  }
  const singular = new Set();
  let widthRange = null;
  let heightRange = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      if (singular.has(1)) throw new Error("Core ML image size range repeats widthRange");
      singular.add(1);
      widthRange = parseSizeRange(reader.message(wire, "CoreML.ImageSizeRange.widthRange"), "Core ML image width range");
    } else if (field === 2) {
      if (singular.has(2)) throw new Error("Core ML image size range repeats heightRange");
      singular.add(2);
      heightRange = parseSizeRange(reader.message(wire, "CoreML.ImageSizeRange.heightRange"), "Core ML image height range");
    } else reader.skip(wire);
  }
  if (!widthRange || !heightRange) throw new Error("Core ML image size range is incomplete");
  return { kind: "range", width: widthRange, height: heightRange };
}

function parseArrayShape(reader, label) {
  const shape = [];
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) for (const dimension of repeatedSignedInt64(reader, wire, label)) limitedPush(shape, dimension, "Core ML shape rank");
    else reader.skip(wire);
  }
  if (!shape.length || shape.some((dimension) => dimension <= 0)) throw new Error(`${label} must contain positive dimensions`);
  return shape;
}

function parseArrayFlexibility(reader, kind) {
  if (kind === "enumerated") {
    const shapes = [];
    while (!reader.done) {
      const { field, wire } = reader.key();
      if (field === 1) limitedPush(shapes, parseArrayShape(reader.message(wire, "CoreML.ArrayFeatureType.Shape"), "Core ML enumerated array shape"), "Core ML enumerated array shapes");
      else reader.skip(wire);
    }
    if (!shapes.length) throw new Error("Core ML enumerated array shapes are empty");
    const rank = shapes[0].length;
    if (shapes.some((shape) => shape.length !== rank)) throw new Error("Core ML enumerated array shapes do not share one rank");
    return { kind: "enumerated", shapes };
  }
  const ranges = [];
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) limitedPush(ranges, parseSizeRange(reader.message(wire, "CoreML.ArrayFeatureType.ShapeRange.sizeRanges"), "Core ML array dimension range"), "Core ML array dimension ranges");
    else reader.skip(wire);
  }
  if (!ranges.length) throw new Error("Core ML array shape range is empty");
  return { kind: "range", dimensions: ranges };
}

function parseArrayFeatureType(reader) {
  let dtype = 0;
  const shape = [];
  const singular = new Set();
  let flexibility = null;
  let defaultValue = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      for (const dimension of repeatedSignedInt64(reader, wire, "CoreML.ArrayFeatureType.shape")) limitedPush(shape, dimension, "Core ML shape rank");
    } else if (field === 2) dtype = reader.intField(wire, singular, 2, "dataType");
    else if (field === 21 || field === 31) {
      if (flexibility) throw new Error("Core ML ArrayFeatureType contains multiple ShapeFlexibility values");
      flexibility = parseArrayFlexibility(reader.message(wire, `CoreML.ArrayFeatureType.${field}`), field === 21 ? "enumerated" : "range");
    } else if ([41, 51, 61].includes(field)) {
      if (defaultValue) throw new Error("Core ML ArrayFeatureType contains multiple defaultOptionalValue values");
      if (field === 41) {
        reader.requireWire(wire, 0, "Core ML int default value");
        defaultValue = { kind: "int32", value: Number(BigInt.asIntN(32, reader.rawVarint())) };
      } else if (field === 51) defaultValue = { kind: "float32", value: reader.floatField(wire, new Set(), 51, "floatDefaultValue") };
      else {
        reader.requireWire(wire, 1, "Core ML double default value");
        if (reader.position + 8 > reader.bytes.length) throw new Error("Core ML double default value is truncated");
        const value = new DataView(reader.bytes.buffer, reader.bytes.byteOffset + reader.position, 8).getFloat64(0, true);
        reader.position += 8;
        if (!Number.isFinite(value)) throw new Error("Core ML double default value must be finite");
        defaultValue = { kind: "float64", value };
      }
    } else reader.skip(wire);
  }
  if (!ARRAY_DTYPES[dtype]) throw new Error(`Core ML ArrayFeatureType has unknown dataType ${dtype}`);
  if (shape.some((dimension) => dimension <= 0)) throw new Error("Core ML ArrayFeatureType default shape contains a non-positive dimension");
  if (flexibility?.kind === "enumerated") {
    if (shape.length && !flexibility.shapes.some((candidate) => sameShape(candidate, shape))) throw new Error("Core ML default array shape is not one of its enumerated shapes");
    if (!shape.length) shape.push(...flexibility.shapes[0]);
  } else if (flexibility?.kind === "range") {
    if (shape.length && shape.length !== flexibility.dimensions.length) throw new Error("Core ML default array shape rank does not match shape range rank");
    if (!shape.length) shape.push(...flexibility.dimensions.map((range) => range.lower_bound));
    if (shape.some((dimension, index) => dimension < flexibility.dimensions[index].lower_bound
      || flexibility.dimensions[index].upper_bound !== -1 && dimension > flexibility.dimensions[index].upper_bound)) throw new Error("Core ML default array shape is outside its shape range");
  }
  return { dtype: ARRAY_DTYPES[dtype], shape, feature_type: "multi_array", constraints: { flexibility, default_optional_value: defaultValue } };
}

function parseSequenceFeatureType(reader) {
  let elementType = null;
  let sizeRange = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1 || field === 3) {
      if (elementType) throw new Error("Core ML SequenceFeatureType contains multiple element types");
      reader.message(wire, `CoreML.SequenceFeatureType.${field}`);
      elementType = field === 1 ? "INT64" : "STRING";
    } else if (field === 101) {
      if (sizeRange) throw new Error("Core ML SequenceFeatureType repeats sizeRange");
      sizeRange = parseSizeRange(reader.message(wire, "CoreML.SequenceFeatureType.sizeRange"), "Core ML sequence size range");
    } else reader.skip(wire);
  }
  if (!elementType) throw new Error("Core ML SequenceFeatureType is missing its element type");
  return { dtype: `SEQUENCE<${elementType}>`, shape: [], feature_type: "sequence", constraints: { element_type: elementType, size_range: sizeRange } };
}

function parseFeatureType(reader) {
  const result = { dtype: "UNKNOWN", shape: [], feature_type: "unknown", optional: false, constraints: null };
  const singular = new Set();
  let typeField = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1 || field === 2 || field === 3) {
      if (typeField != null) throw new Error("Core ML FeatureType contains multiple oneof values");
      typeField = field;
      reader.message(wire, `CoreML.FeatureType.${field}`);
      const names = { 1: "INT64", 2: "FLOAT64", 3: "STRING" };
      result.dtype = names[field];
      result.feature_type = names[field].toLowerCase();
    } else if (field === 6) {
      if (typeField != null) throw new Error("Core ML FeatureType contains multiple oneof values");
      typeField = field;
      const dictionary = reader.message(wire, "CoreML.DictionaryFeatureType");
      let keyType = null;
      while (!dictionary.done) {
        const item = dictionary.key();
        if (item.field === 1 || item.field === 2) {
          if (keyType != null) throw new Error("Core ML DictionaryFeatureType contains multiple key types");
          keyType = item.field === 1 ? "INT64" : "STRING";
          dictionary.message(item.wire, `CoreML.DictionaryFeatureType.${item.field}`);
        } else dictionary.skip(item.wire);
      }
      if (!keyType) throw new Error("Core ML DictionaryFeatureType is missing its key type");
      result.dtype = "DICTIONARY";
      result.feature_type = "dictionary";
      result.constraints = { key_type: keyType, value_type: "FLOAT64" };
    } else if (field === 7) {
      if (typeField != null) throw new Error("Core ML FeatureType contains multiple oneof values");
      typeField = field;
      Object.assign(result, parseSequenceFeatureType(reader.message(wire, "CoreML.SequenceFeatureType")));
    } else if (field === 8) {
      if (typeField != null) throw new Error("Core ML FeatureType contains multiple oneof values");
      typeField = field;
      const state = reader.message(wire, "CoreML.StateFeatureType");
      let wrapped = null;
      while (!state.done) {
        const item = state.key();
        if (item.field === 1) {
          if (wrapped) throw new Error("Core ML StateFeatureType repeats arrayType");
          wrapped = parseArrayFeatureType(state.message(item.wire, "CoreML.StateFeatureType.arrayType"));
        } else state.skip(item.wire);
      }
      if (!wrapped) throw new Error("Core ML StateFeatureType is missing arrayType");
      Object.assign(result, { dtype: wrapped.dtype, shape: wrapped.shape, feature_type: "state", constraints: { wrapped_type: wrapped } });
    } else if (field === 4) {
      if (typeField != null) throw new Error("Core ML FeatureType contains multiple oneof values");
      typeField = field;
      const image = reader.message(wire, "CoreML.ImageFeatureType");
      let width = null;
      let height = null;
      let color = 0;
      let flexibility = null;
      const fields = new Set();
      while (!image.done) {
        const item = image.key();
        if (item.field === 1) width = image.int64Field(item.wire, fields, 1, "width");
        else if (item.field === 2) height = image.int64Field(item.wire, fields, 2, "height");
        else if (item.field === 3) color = image.intField(item.wire, fields, 3, "colorSpace");
        else if (item.field === 21 || item.field === 31) {
          if (flexibility) throw new Error("Core ML ImageFeatureType contains multiple SizeFlexibility values");
          flexibility = parseImageFlexibility(image.message(item.wire, `CoreML.ImageFeatureType.${item.field}`), item.field === 21 ? "enumerated" : "range");
        } else image.skip(item.wire);
      }
      if (flexibility?.kind === "enumerated") {
        if (width > 0 && height > 0 && !flexibility.sizes.some((size) => size.width === width && size.height === height)) throw new Error("Core ML default image size is not one of its enumerated sizes");
        if (!(width > 0 && height > 0)) ({ width, height } = flexibility.sizes[0]);
      } else if (flexibility?.kind === "range") {
        if (!(width > 0 && height > 0)) { width = flexibility.width.lower_bound; height = flexibility.height.lower_bound; }
        if (width < flexibility.width.lower_bound || flexibility.width.upper_bound !== -1 && width > flexibility.width.upper_bound
          || height < flexibility.height.lower_bound || flexibility.height.upper_bound !== -1 && height > flexibility.height.upper_bound) throw new Error("Core ML default image size is outside its image size range");
      }
      const [dtype, channels] = IMAGE_COLOR[color] || [`IMAGE_COLORSPACE_${color}`, null];
      result.dtype = dtype;
      result.shape = height > 0 && width > 0 ? [height, width, ...(channels ? [channels] : [])] : [];
      result.feature_type = "image";
      result.constraints = { width, height, color_space: color, shape_projection: "height_width_channels", flexibility };
    } else if (field === 5) {
      if (typeField != null) throw new Error("Core ML FeatureType contains multiple oneof values");
      typeField = field;
      Object.assign(result, parseArrayFeatureType(reader.message(wire, "CoreML.ArrayFeatureType")));
    } else if (field === 1000) {
      result.optional = reader.intField(wire, singular, 1000, "isOptional") !== 0;
    } else reader.skip(wire);
  }
  if (typeField == null) throw new Error("Core ML FeatureType is missing its type value");
  return result;
}

function parseFeature(reader, index, role, functionName = null) {
  const result = { index, name: "", short_description: null, role, function_name: functionName };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.name = reader.stringField(wire, singular, 1, "name");
    else if (field === 2) result.short_description = reader.stringField(wire, singular, 2, "shortDescription");
    else if (field === 3) Object.assign(result, parseFeatureType(reader.message(wire, "CoreML.FeatureType")));
    else reader.skip(wire);
  }
  if (!result.name) throw new Error(`Core ML ${role} feature is missing a name`);
  return result;
}

function parseFunction(reader, index) {
  const result = { index, name: "", inputs: [], outputs: [], states: [], predicted_feature_name: null, predicted_probabilities_name: null };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.name = reader.stringField(wire, singular, 1, "functionName");
    else if ([2, 3, 6].includes(field)) {
      const rows = field === 2 ? result.inputs : field === 3 ? result.outputs : result.states;
      limitedPush(rows, parseFeature(reader.message(wire, "CoreML.FunctionFeature"), rows.length, field === 2 ? "input" : field === 3 ? "output" : "state", result.name), "Core ML function features");
    } else if (field === 4) result.predicted_feature_name = reader.stringField(wire, singular, 4, "predictedFeatureName");
    else if (field === 5) result.predicted_probabilities_name = reader.stringField(wire, singular, 5, "predictedProbabilitiesName");
    else reader.skip(wire);
  }
  if (!result.name) throw new Error("Core ML function is missing a name");
  for (const item of [...result.inputs, ...result.outputs, ...result.states]) item.function_name = result.name;
  return result;
}

function parseDescription(bytes) {
  const reader = new ProtoReader(bytes, "CoreML.ModelDescription");
  const result = {
    inputs: [],
    outputs: [],
    states: [],
    training_inputs: [],
    functions: [],
    default_function_name: null,
    predicted_feature_name: null,
    predicted_probabilities_name: null,
    metadata: {},
  };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if ([1, 10, 13, 50].includes(field)) {
      const rows = field === 1 ? result.inputs : field === 10 ? result.outputs : field === 13 ? result.states : result.training_inputs;
      const role = field === 1 ? "input" : field === 10 ? "output" : field === 13 ? "state" : "training_input";
      limitedPush(rows, parseFeature(reader.message(wire, "CoreML.FeatureDescription"), rows.length, role), "Core ML model features");
    } else if (field === 11) result.predicted_feature_name = reader.stringField(wire, singular, 11, "predictedFeatureName");
    else if (field === 12) result.predicted_probabilities_name = reader.stringField(wire, singular, 12, "predictedProbabilitiesName");
    else if (field === 20) limitedPush(result.functions, parseFunction(reader.message(wire, "CoreML.FunctionDescription"), result.functions.length), "Core ML functions");
    else if (field === 21) result.default_function_name = reader.stringField(wire, singular, 21, "defaultFunctionName");
    else if (field === 100) result.metadata = parseMetadata(reader.message(wire, "CoreML.Metadata"));
    else reader.skip(wire);
  }
  return result;
}

function coreMlInputTensorContract(feature, network) {
  const externalShape = Array.isArray(feature?.shape) ? feature.shape : [];
  if (feature?.feature_type === "image") {
    const height = Number(feature.constraints?.height);
    const width = Number(feature.constraints?.width);
    const channels = Number(IMAGE_COLOR[feature.constraints?.color_space]?.[1]);
    if (![height, width, channels].every((value) => Number.isSafeInteger(value) && value > 0)) {
      return { shape: [], dtype: "UNKNOWN", shape_source: "coreml_image_mapping_unresolved" };
    }
    return {
      shape: network.image_input_shape_mapping === 1
        ? [1, channels, height, width]
        : [1, 1, channels, height, width],
      dtype: "FLOAT32",
      shape_source: network.image_input_shape_mapping === 1
        ? "coreml_rank4_image_input_mapping"
        : "coreml_rank5_image_input_mapping",
    };
  }
  if (feature?.feature_type === "multi_array") {
    if (network.array_input_shape_mapping === 1) {
      return { shape: externalShape, dtype: feature.dtype, shape_source: "coreml_exact_array_input_mapping" };
    }
    if (externalShape.length === 1) {
      return { shape: [1, 1, externalShape[0], 1, 1], dtype: feature.dtype, shape_source: "coreml_rank5_array_input_mapping" };
    }
    if (externalShape.length === 3) {
      return { shape: [1, 1, ...externalShape], dtype: feature.dtype, shape_source: "coreml_rank5_array_input_mapping" };
    }
    return { shape: [], dtype: feature.dtype, shape_source: "coreml_rank5_array_input_mapping_unresolved_rank" };
  }
  return { shape: [], dtype: "UNKNOWN", shape_source: "coreml_non_tensor_interface" };
}

function shapeProduct(shape) {
  if (!Array.isArray(shape) || !shape.length || shape.some((value) => !Number.isSafeInteger(value) || value <= 0)) return null;
  let product = 1;
  for (const value of shape) {
    product *= value;
    if (!Number.isSafeInteger(product)) return null;
  }
  return product;
}

function safeCountProduct(values) {
  let product = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0 || (value && product > Math.floor(Number.MAX_SAFE_INTEGER / value))) return null;
    product *= value;
  }
  return product;
}

const COREML_ELEMENT_BITS = Object.freeze({
  BOOL: 8, INT4: 4, UINT1: 1, UINT2: 2, UINT3: 3, UINT4: 4, UINT6: 6,
  INT8: 8, UINT8: 8, FLOAT8E4M3FN: 8, FLOAT8E5M2: 8,
  FLOAT16: 16, BFLOAT16: 16, INT16: 16, UINT16: 16,
  FLOAT32: 32, INT32: 32, UINT32: 32, FLOAT64: 64, INT64: 64, UINT64: 64,
});

function coreMlTensorBytes(tensor) {
  const count = tensor?.rank === 0 ? 1 : shapeProduct(tensor?.shape);
  const bits = COREML_ELEMENT_BITS[tensor?.dtype];
  if (count == null || bits == null || count > Math.floor(Number.MAX_SAFE_INTEGER / bits)) return null;
  const bytes = Math.ceil(count * bits / 8);
  return Number.isSafeInteger(bytes) ? bytes : null;
}

function coreMlLivenessState(ops, tensors, inputTensorIndices, outputTensorIndices) {
  const producer = new Map((inputTensorIndices || []).map((index) => [index, -1]));
  const lastUse = new Map();
  for (const op of ops || []) {
    for (const index of op.outputs || []) producer.set(index, op.index);
    for (const index of op.inputs || []) {
      if (!tensors[index]?.constant_buffer) lastUse.set(index, Math.max(lastUse.get(index) ?? -1, op.index));
    }
  }
  for (const index of outputTensorIndices || []) {
    if (producer.has(index)) lastUse.set(index, Math.max(lastUse.get(index) ?? -1, ops.length));
  }
  const liveTensors = (tensors || []).filter((tensor) => !tensor.constant_buffer && producer.has(tensor.index) && lastUse.has(tensor.index));
  return { producer, lastUse, liveTensors };
}

function coreMlLiveTensorIndicesAt(state, cursor) {
  return state.liveTensors.filter((tensor) => state.producer.get(tensor.index) <= cursor && state.lastUse.get(tensor.index) >= cursor)
    .map((tensor) => tensor.index);
}

function coreMlKnownBytesForTensorSet(indices, tensors) {
  let bytes = 0;
  const unknown = [];
  for (const index of new Set(indices || [])) {
    const width = coreMlTensorBytes(tensors[index]);
    if (width == null) unknown.push(index);
    else bytes += width;
  }
  return { bytes, unknown };
}

function buildCoreMlTensorLiveness(ops, tensors, inputTensorIndices, outputTensorIndices, nestedBlockOperationCount = 0) {
  if (nestedBlockOperationCount > 0) return {
    schema: "deepbom.coreml.tensor_liveness.v1",
    status: "not_assessed_control_flow_execution_path_not_reconstructed",
    evidence_class: "NOT_ASSESSED",
    peak_bytes: null,
    peak_bytes_status: "not_assessed_control_flow_execution_path_not_reconstructed",
    peak_at_op: null,
    peak_at_op_name: null,
    assessed: false,
    assessed_tensor_count: 0,
    unassessed_tensor_count: (tensors || []).filter((tensor) => !tensor.constant_buffer).length,
    unknown_activation_tensors: (tensors || []).filter((tensor) => !tensor.constant_buffer).length,
    unassessed_tensors: [],
    non_dense_value_count: 0,
    non_dense_values: [],
    method: `${nestedBlockOperationCount} nested MIL block operation(s) are decoded, but branch/loop execution counts and mutually exclusive lifetimes are not reconstructed. No flattened-graph peak is emitted; runtime allocator, scratch, and backend-private buffers also remain unobserved.`,
  };
  const { producer, lastUse, liveTensors } = coreMlLivenessState(ops, tensors, inputTensorIndices, outputTensorIndices);
  const unassessed = liveTensors.filter((tensor) => coreMlTensorBytes(tensor) == null).map((tensor) => ({
    tensor_index: tensor.index,
    tensor_name: tensor.name,
    reason: !shapeProduct(tensor.shape) ? "shape is not fully static and positive" : `dtype ${tensor.dtype || "UNKNOWN"} has no supported byte width`,
  }));
  const assessed = liveTensors.length - unassessed.length;
  let peakBytes = 0;
  let peakAtOp = 0;
  let peakAtOpName = "";
  let peakLiveTensorIndices = [];
  for (let cursor = 0; cursor <= (ops || []).length; cursor += 1) {
    let liveBytes = 0;
    for (const tensor of liveTensors) {
      const bytes = coreMlTensorBytes(tensor);
      if (bytes != null && producer.get(tensor.index) <= cursor && lastUse.get(tensor.index) >= cursor) liveBytes += bytes;
    }
    if (liveBytes > peakBytes) {
      peakBytes = liveBytes;
      peakAtOp = Math.min(cursor, Math.max(0, ops.length - 1));
      peakAtOpName = ops[peakAtOp]?.name || "";
      peakLiveTensorIndices = coreMlLiveTensorIndicesAt({ producer, lastUse, liveTensors }, cursor);
    }
  }
  const status = assessed === 0 ? "not_assessed" : unassessed.length ? "partial" : "assessed";
  return {
    schema: "deepbom.coreml.tensor_liveness.v1",
    status,
    evidence_class: "DERIVED",
    peak_bytes: status === "not_assessed" ? null : peakBytes,
    peak_bytes_status: status === "partial" ? "assessed_tensor_lower_bound" : status,
    peak_at_op: status === "not_assessed" ? null : peakAtOp,
    peak_at_op_name: status === "not_assessed" ? null : peakAtOpName,
    peak_live_tensor_indices: status === "not_assessed" ? [] : peakLiveTensorIndices,
    assessed: status !== "not_assessed",
    assessed_tensor_count: assessed,
    unassessed_tensor_count: unassessed.length,
    unknown_activation_tensors: unassessed.length,
    unassessed_tensors: unassessed,
    non_dense_value_count: 0,
    non_dense_values: [],
    method: "Static producer-to-last-consumer liveness sweep over decoded Core ML dense SSA/blob tensors. Unknown shape or dtype rows are excluded and make the subtotal a lower bound; weights, allocator alignment, runtime scratch, and backend-private buffers are excluded.",
  };
}

function attachCoreMlMilScopeLiveness(analysis) {
  const contract = analysis?.coreml?.mil_scope_intrinsic_cost;
  if (!contract?.scope_rows?.length) return;
  for (const row of contract.scope_rows) {
    const scopeOps = (analysis.ops || []).filter((op) => op.mil_scope === row.scope);
    const produced = new Set(scopeOps.flatMap((op) => op.outputs || []));
    const consumed = new Set(scopeOps.flatMap((op) => op.inputs || []));
    const inputs = [...new Set([
      ...(row.scope === contract.root_scope ? analysis.input_tensor_indices || [] : []),
      ...(row.block_input_tensor_indices || []),
      ...[...consumed].filter((index) => !produced.has(index)),
    ])];
    const outputs = [...new Set([
      ...(row.scope === contract.root_scope ? analysis.output_tensor_indices || [] : []),
      ...(row.block_output_tensor_indices || []),
      ...[...produced].filter((index) => !consumed.has(index) || (analysis.output_tensor_indices || []).includes(index)),
    ])];
    const localOps = scopeOps.map((op, index) => ({ ...op, index }));
    row.liveness_input_tensor_indices = inputs;
    row.liveness_output_tensor_indices = outputs;
    row.scope_local_liveness = buildCoreMlTensorLiveness(localOps, analysis.tensors || [], inputs, outputs, 0);
  }
  contract.scope_liveness_assessed_count = contract.scope_rows.filter((row) => row.scope_local_liveness.status === "assessed").length;
  contract.scope_liveness_partial_count = contract.scope_rows.filter((row) => row.scope_local_liveness.status === "partial").length;
  contract.scope_liveness_unassessed_count = contract.scope_rows.length - contract.scope_liveness_assessed_count - contract.scope_liveness_partial_count;
}

function buildCoreMlMilControlFlowLiveness(analysis) {
  const contract = analysis?.coreml?.mil_scope_intrinsic_cost;
  const nestedScopeCount = Number(contract?.nested_scope_count || 0);
  if (!contract?.scope_rows?.length || nestedScopeCount === 0) {
    return buildCoreMlTensorLiveness(analysis.ops || [], analysis.tensors || [], analysis.input_tensor_indices || [], analysis.output_tensor_indices || [], 0);
  }
  const rows = new Map(contract.scope_rows.map((row) => [row.scope, row]));
  const opsByScope = new Map(contract.scope_rows.map((row) => [row.scope, (analysis.ops || []).filter((op) => op.mil_scope === row.scope)]));
  const supported = new Set(["cond", "while_loop"]);
  const visit = (scope, stack = new Set()) => {
    if (stack.has(scope)) return { status: "not_assessed_recursive_control_flow_scope", reason: `recursive MIL scope reference at ${scope}` };
    const row = rows.get(scope);
    if (!row?.scope_local_liveness) return { status: "not_assessed_missing_scope_liveness", reason: `scope ${scope} has no local liveness contract` };
    const nextStack = new Set(stack).add(scope);
    const scopeOps = opsByScope.get(scope) || [];
    const localOps = scopeOps.map((op, index) => ({ ...op, index }));
    const state = coreMlLivenessState(localOps, analysis.tensors || [], row.liveness_input_tensor_indices || [], row.liveness_output_tensor_indices || []);
    let best = {
      peak_bytes: row.scope_local_liveness.peak_bytes ?? 0,
      peak_live_tensor_indices: row.scope_local_liveness.peak_live_tensor_indices || [],
      peak_at_op: row.scope_local_liveness.peak_at_op,
      peak_at_op_name: row.scope_local_liveness.peak_at_op_name,
      unknown_tensor_indices: (row.scope_local_liveness.unassessed_tensors || []).map((item) => item.tensor_index),
    };
    for (let localIndex = 0; localIndex < localOps.length; localIndex += 1) {
      const op = localOps[localIndex];
      const childScopes = op.mil_nested_scopes || [];
      if (!childScopes.length) continue;
      const type = String(op.mil_operation_type || "").toLowerCase();
      if (!supported.has(type) || childScopes.length !== 2) {
        return { status: "not_assessed_unsupported_nested_operation", reason: `MIL ${op.mil_operation_type || op.name} owns ${childScopes.length} nested block(s); only source-defined two-block cond and while_loop envelopes are implemented` };
      }
      const outerLive = coreMlLiveTensorIndicesAt(state, localIndex);
      for (const childScope of childScopes) {
        const child = visit(childScope, nextStack);
        if (!child.peak_live_tensor_indices) return child;
        const union = [...new Set([...outerLive, ...child.peak_live_tensor_indices])];
        const measured = coreMlKnownBytesForTensorSet(union, analysis.tensors || []);
        if (measured.bytes > best.peak_bytes) {
          best = {
            peak_bytes: measured.bytes,
            peak_live_tensor_indices: union,
            peak_at_op: op.index,
            peak_at_op_name: op.name,
            unknown_tensor_indices: [...new Set([...(child.unknown_tensor_indices || []), ...measured.unknown])],
          };
        } else if (measured.bytes === best.peak_bytes) {
          best.unknown_tensor_indices = [...new Set([...(best.unknown_tensor_indices || []), ...(child.unknown_tensor_indices || []), ...measured.unknown])];
        }
      }
    }
    return best;
  };
  const envelope = visit(contract.root_scope);
  if (!envelope.peak_live_tensor_indices) return {
    schema: "deepbom.coreml.tensor_liveness.v1",
    status: envelope.status,
    evidence_class: "NOT_ASSESSED",
    peak_bytes: null,
    peak_bytes_status: envelope.status,
    peak_at_op: null,
    peak_at_op_name: null,
    peak_live_tensor_indices: [],
    assessed: false,
    assessed_tensor_count: 0,
    unassessed_tensor_count: 0,
    unknown_activation_tensors: 0,
    unassessed_tensors: [],
    non_dense_value_count: 0,
    non_dense_values: [],
    method: `${envelope.reason}. No flattened-graph peak is emitted.`,
  };
  const unknown = [...new Set(envelope.unknown_tensor_indices || [])];
  return {
    schema: "deepbom.coreml.tensor_liveness.v1",
    status: unknown.length ? "partial_control_flow_peak_lower_bound" : "assessed_static_control_flow_peak_envelope",
    evidence_class: "SOURCE_PINNED_AND_DERIVED",
    peak_bytes: envelope.peak_bytes,
    peak_bytes_status: unknown.length ? "assessed_tensor_lower_bound" : "static_logical_payload_envelope",
    peak_at_op: envelope.peak_at_op,
    peak_at_op_name: envelope.peak_at_op_name,
    peak_live_tensor_indices: envelope.peak_live_tensor_indices,
    assessed: true,
    assessed_tensor_count: envelope.peak_live_tensor_indices.length - unknown.length,
    unassessed_tensor_count: unknown.length,
    unknown_activation_tensors: unknown.length,
    unassessed_tensors: unknown.map((index) => ({ tensor_index: index, tensor_name: analysis.tensors?.[index]?.name || "", reason: "shape or dtype does not have an exact static byte width" })),
    non_dense_value_count: 0,
    non_dense_values: [],
    control_flow_scope_count: contract.scope_rows.length,
    method: "Static logical-payload peak envelope over source-defined MIL cond and while_loop scopes. cond branches are mutually exclusive; while_loop condition and body scopes execute sequentially and typed iteration buffers are reusable, so iteration count does not multiply the peak. Runtime allocator alignment, in-place aliasing, scratch, and backend-private buffers remain excluded.",
  };
}

function buildCoreMlSizeBreakdown(analysis) {
  const integrity = analysis?.weight_integrity;
  const parameters = Array.isArray(integrity?.parameters) ? integrity.parameters : [];
  const payloadBytes = Number.isSafeInteger(integrity?.payload_bytes) ? integrity.payload_bytes : null;
  const fileBytes = Number(analysis?.file_size_bytes ?? analysis?.file_size);
  const parameterValues = parameters.map((row) => row.value_count).filter((value) => Number.isSafeInteger(value) && value >= 0);
  const allCardinalitiesKnown = parameterValues.length === parameters.length;
  const storedElements = allCardinalitiesKnown ? parameterValues.reduce((sum, value) => sum + value, 0) : null;
  const fp32Bytes = parameters.filter((row) => row.storage === "float32" || row.dtype === "FLOAT32")
    .reduce((sum, row) => sum + Number(row.byte_length || 0), 0);
  const constantCount = Number(integrity?.parameter_count ?? parameters.length);
  const structureBytes = payloadBytes != null && Number.isSafeInteger(fileBytes) && payloadBytes <= fileBytes ? fileBytes - payloadBytes : null;
  return {
    schema: "deepbom.coreml.size_breakdown.v1",
    status: payloadBytes == null ? "partial_constant_payload_not_fully_bound" : structureBytes == null ? "partial_container_conservation_unavailable" : "assessed",
    file_size: Number.isSafeInteger(fileBytes) ? fileBytes : null,
    constant_bytes: payloadBytes,
    constant_tensor_count: constantCount,
    physical_constant_buffer_count: constantCount,
    logical_constant_reference_bytes: payloadBytes,
    stored_scalar_elements: storedElements,
    float_constant_bytes: fp32Bytes,
    structure_overhead_bytes: structureBytes,
    metadata_bytes: null,
    unique_constant_bytes: null,
    duplicate_constant_bytes: null,
    theoretical_fp16_constant_bytes: payloadBytes == null ? null : payloadBytes - fp32Bytes + fp32Bytes / 2,
    theoretical_int8_constant_bytes: payloadBytes == null ? null : payloadBytes - fp32Bytes + fp32Bytes / 4,
    zero_constant_byte_ratio: null,
    metrics: {
      zero_constant_byte_ratio: { status: "not_assessed", reason: "Scalar zero counts do not determine raw 0x00 byte counts for Core ML protobuf and blob encodings" },
      duplicate_constant_bytes: { status: "not_assessed", reason: "Cross-parameter semantic duplicate equivalence is not evaluated" },
    },
    detail: payloadBytes == null
      ? "Core ML constant declarations are inventoried, but one or more external package blobs are not bound; missing bytes remain null."
      : "Exact decoded WeightParams or bound ML Program blob payload bytes are subtracted from the selected model/package byte total. The remainder combines protobuf/package structure and metadata because those classes are not separately attributable.",
  };
}

export function refreshCoreMlDerivedEvidence(analysis) {
  if (!analysis || analysis.format !== "coreml") return analysis;
  refreshCoreMlMilCompressionEvidence(analysis);
  attachCoreMlMilScopeLiveness(analysis);
  analysis.tensor_liveness = buildCoreMlMilControlFlowLiveness(analysis);
  analysis.size_breakdown = buildCoreMlSizeBreakdown(analysis);
  return analysis;
}

function sameShape(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function spatialOutput(input, kernel, stride, padding, dilation = 1) {
  if (![input, kernel, stride, dilation].every((value) => Number.isSafeInteger(value) && value > 0)) return null;
  const effectiveKernel = dilation * (kernel - 1) + 1;
  if (padding?.kind === "same") return Math.ceil(input / stride);
  if (padding?.kind === "valid") {
    const amounts = padding.amounts || [[0, 0], [0, 0]];
    const pair = amounts[padding.axis || 0] || [0, 0];
    const numerator = input + Number(pair[0] || 0) + Number(pair[1] || 0) - effectiveKernel;
    return numerator < 0 ? 0 : Math.floor(numerator / stride) + 1;
  }
  if (padding?.kind === "include_last_pixel") {
    const amount = Number(padding.amount || 0);
    let output = Math.ceil((input + 2 * amount - effectiveKernel) / stride) + 1;
    if (amount > 0 && (output - 1) * stride >= input + amount) output -= 1;
    return Math.max(0, output);
  }
  return null;
}

function coreMlAxis(value, rank, allowEnd = false) {
  if (!Number.isSafeInteger(value) || !Number.isSafeInteger(rank) || rank < 1) return null;
  const axis = value < 0 ? value + rank : value;
  return axis >= 0 && axis <= rank - (allowEnd ? 0 : 1) ? axis : null;
}

function coreMlBroadcastShape(left, right) {
  const rank = Math.max(left.length, right.length);
  const output = Array(rank).fill(1);
  for (let offset = 1; offset <= rank; offset += 1) {
    const a = left.at(-offset) ?? 1;
    const b = right.at(-offset) ?? 1;
    if (a !== b && a !== 1 && b !== 1) return null;
    output[rank - offset] = Math.max(a, b);
  }
  return output;
}

function coreMlStaticReshape(shape, inputCount) {
  if (!shape.length) return null;
  const missing = shape.indexOf(-1);
  const known = shape.filter((value) => value !== -1);
  const knownCount = shapeProduct(known.length ? known : [1]);
  if (knownCount == null || missing < 0) return knownCount === inputCount ? [...shape] : null;
  if (inputCount % knownCount !== 0 || inputCount / knownCount < 1) return null;
  const output = [...shape];
  output[missing] = inputCount / knownCount;
  return output;
}

function coreMlRankPreservingReshape(shape, input) {
  if (shape.length !== input.length || shape.filter((value) => value === -1).length > 1
    || shape.some((value) => value < -1)) return null;
  return coreMlStaticReshape(shape.map((value, index) => value === 0 ? input[index] : value), shapeProduct(input));
}

function coreMlReducedShape(input, axes, keepDims) {
  const normalized = axes.map((value) => coreMlAxis(value, input.length));
  if (!normalized.length || normalized.some((axis) => axis == null) || new Set(normalized).size !== normalized.length) return null;
  const selected = new Set(normalized);
  const output = keepDims ? input.map((value, index) => selected.has(index) ? 1 : value)
    : input.filter((_, index) => !selected.has(index));
  return output.length ? output : [1];
}

const LEGACY_SHAPE_PRESERVING = new Set([
  "ACTIVATION", "BATCHNORM", "MEAN_VARIANCE_NORMALIZE", "L2_NORMALIZE", "SOFTMAX", "LRN", "UNARY",
  "ADD", "MULTIPLY", "AVERAGE", "SCALE", "BIAS", "MAX", "MIN", "CLIP", "CEIL", "FLOOR", "SIGN",
  "ROUND", "EXP2", "SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN", "SINH", "COSH", "TANH", "ASINH",
  "ACOSH", "ATANH", "ERF", "GELU", "EQUAL", "NOT_EQUAL", "LESS_THAN", "LESS_EQUAL", "GREATER_THAN",
  "GREATER_EQUAL", "LOGICAL_OR", "LOGICAL_XOR", "LOGICAL_NOT", "LOGICAL_AND", "MOD_BROADCASTABLE",
  "MIN_BROADCASTABLE", "MAX_BROADCASTABLE", "ADD_BROADCASTABLE", "POW_BROADCASTABLE", "DIVIDE_BROADCASTABLE",
  "FLOOR_DIV_BROADCASTABLE", "MULTIPLY_BROADCASTABLE", "SUBTRACT_BROADCASTABLE", "COPY",
]);
const LEGACY_BROADCASTING = new Set([
  "EQUAL", "NOT_EQUAL", "LESS_THAN", "LESS_EQUAL", "GREATER_THAN", "GREATER_EQUAL", "LOGICAL_OR", "LOGICAL_XOR", "LOGICAL_AND",
  "MOD_BROADCASTABLE", "MIN_BROADCASTABLE", "MAX_BROADCASTABLE", "ADD_BROADCASTABLE", "POW_BROADCASTABLE", "DIVIDE_BROADCASTABLE",
  "FLOOR_DIV_BROADCASTABLE", "MULTIPLY_BROADCASTABLE", "SUBTRACT_BROADCASTABLE", "WHERE_BROADCASTABLE",
]);
const LEGACY_LIKE_FIRST = new Set(["FILL_LIKE", "RANDOM_NORMAL_LIKE", "RANDOM_UNIFORM_LIKE", "RANDOM_BERNOULLI_LIKE"]);
const LEGACY_LIKE_SECOND = new Set(["BROADCAST_TO_LIKE", "RESHAPE_LIKE"]);
const LEGACY_REDUCE_ND = new Set(["REDUCE_L1", "REDUCE_L2", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_SUM", "REDUCE_PROD", "REDUCE_MEAN", "REDUCE_LOG_SUM", "REDUCE_SUM_SQUARE", "REDUCE_LOG_SUM_EXP"]);

function deriveLegacyLayer(layer, inputShapes) {
  const knownInputs = inputShapes.every((shape) => shapeProduct(shape) != null);
  const first = inputShapes[0] || [];
  const attrs = layer.attributes || {};
  const result = { output_shapes: [], shape_status: "not_assessed", shape_reason: "operation-specific shape rule is not implemented", macs: null, macs_decimal: null, macs_status: "not_assessed", macs_reason: "operation-specific arithmetic rule or required shape is unavailable", macs_definition: null };
  const setSingle = (shape, status) => {
    if (layer.outputs.length === 1 && shapeProduct(shape) != null) {
      result.output_shapes = [shape];
      result.shape_status = status;
      result.shape_reason = null;
    }
  };
  const setMany = (shapes, status) => {
    if (shapes.length === layer.outputs.length && shapes.every((shape) => shapeProduct(shape) != null)) {
      result.output_shapes = shapes;
      result.shape_status = status;
      result.shape_reason = null;
    }
  };
  const nonMac = (reason) => {
    result.macs = 0;
    result.macs_decimal = "0";
    result.macs_status = "derived_non_mac_operation";
    result.macs_reason = reason;
  };
  if (layer.type === "CONVOLUTION" && knownInputs && inputShapes.length === 1 && first.length >= 3) {
    const c = first.length - 3;
    const h = first.length - 2;
    const w = first.length - 1;
    const inputChannels = first[c];
    const expectedChannels = attrs.is_deconvolution ? attrs.kernel_channels : attrs.kernel_channels * attrs.n_groups;
    if (inputChannels !== expectedChannels) throw new Error(`Core ML convolution ${layer.name} input has ${inputChannels} channels; serialized kernel contract expects ${expectedChannels}`);
    let outH;
    let outW;
    if (attrs.is_deconvolution) {
      if (attrs.output_shape?.length === 2) [outH, outW] = attrs.output_shape;
      else if (attrs.padding?.kind === "same") [outH, outW] = [first[h] * attrs.stride[0], first[w] * attrs.stride[1]];
      else {
        const pads = attrs.padding?.amounts || [[0, 0], [0, 0]];
        outH = (first[h] - 1) * attrs.stride[0] + attrs.kernel_size[0] - pads[0][0] - pads[0][1];
        outW = (first[w] - 1) * attrs.stride[1] + attrs.kernel_size[1] - pads[1][0] - pads[1][1];
      }
    } else {
      outH = spatialOutput(first[h], attrs.kernel_size[0], attrs.stride[0], { ...attrs.padding, axis: 0 }, attrs.dilation[0]);
      outW = spatialOutput(first[w], attrs.kernel_size[1], attrs.stride[1], { ...attrs.padding, axis: 1 }, attrs.dilation[1]);
    }
    const output = [...first];
    output[c] = attrs.output_channels;
    output[h] = outH;
    output[w] = outW;
    setSingle(output, "derived_coreml_convolution_contract");
    const macProduct = multiplyCoreMlExactIntegers(attrs.is_deconvolution
      ? [...first.slice(0, c), first[h], first[w], attrs.kernel_channels, attrs.output_channels / attrs.n_groups, ...attrs.kernel_size]
      : [...first.slice(0, c), outH, outW, attrs.output_channels, attrs.kernel_channels, ...attrs.kernel_size]);
    result.macs = macProduct?.number ?? null;
    result.macs_decimal = macProduct?.decimal ?? null;
    result.macs_status = !macProduct ? "not_assessed_invalid_exact_cardinality" : result.macs == null ? "derived_exact_decimal_only_coreml_convolution" : "derived_exact_coreml_convolution";
    result.macs_reason = macProduct ? null : "MAC factors are not exact nonnegative integers";
    result.macs_definition = "Convolution MACs count one multiply-accumulate per output element and serialized kernel coefficient; bias and activation arithmetic are excluded.";
  } else if (layer.type === "CONVOLUTION_3D" && knownInputs && inputShapes.length === 1 && first.length >= 5) {
    const c = first.length - 4;
    const d = first.length - 3;
    const h = first.length - 2;
    const w = first.length - 1;
    if (first[c] !== attrs.input_channels) throw new Error(`Core ML Conv3D ${layer.name} input has ${first[c]} channels; serialized contract expects ${attrs.input_channels}`);
    const outputSpatial = [];
    for (let axis = 0; axis < 3; axis += 1) {
      const input = first[[d, h, w][axis]];
      const effectiveKernel = attrs.dilation[axis] * (attrs.kernel_size[axis] - 1) + 1;
      if (attrs.is_deconvolution) {
        if (attrs.output_shape.length === 3) outputSpatial.push(attrs.output_shape[axis]);
        else if (attrs.padding_type === 2) outputSpatial.push(input * attrs.stride[axis]);
        else {
          const padding = attrs.padding_type === 0 ? attrs.custom_padding[axis] : [0, 0];
          outputSpatial.push((input - 1) * attrs.stride[axis] + effectiveKernel - padding[0] - padding[1]);
        }
      } else {
        const padding = attrs.padding_type === 2
          ? { kind: "same" }
          : { kind: "valid", amounts: [attrs.padding_type === 0 ? attrs.custom_padding[axis] : [0, 0]], axis: 0 };
        outputSpatial.push(spatialOutput(input, attrs.kernel_size[axis], attrs.stride[axis], padding, attrs.dilation[axis]));
      }
    }
    if (outputSpatial.some((value) => !Number.isSafeInteger(value) || value <= 0)) throw new Error(`Core ML Conv3D ${layer.name} derives a non-positive output shape`);
    const output = [...first];
    output[c] = attrs.output_channels;
    [output[d], output[h], output[w]] = outputSpatial;
    setSingle(output, "derived_coreml_convolution_3d_contract");
    const macProduct = multiplyCoreMlExactIntegers(attrs.is_deconvolution
      ? [...first.slice(0, c), first[d], first[h], first[w], attrs.input_channels, attrs.output_channels / attrs.n_groups, ...attrs.kernel_size]
      : [...first.slice(0, c), ...outputSpatial, attrs.output_channels, attrs.input_channels / attrs.n_groups, ...attrs.kernel_size]);
    result.macs = macProduct?.number ?? null;
    result.macs_decimal = macProduct?.decimal ?? null;
    result.macs_status = !macProduct ? "not_assessed_invalid_exact_cardinality" : result.macs == null ? "derived_exact_decimal_only_coreml_convolution_3d" : "derived_exact_coreml_convolution_3d";
    result.macs_reason = macProduct ? null : "MAC factors are not exact nonnegative integers";
    result.macs_definition = "Conv3D MACs count one multiply-accumulate per output element and grouped kernel coefficient; bias and activation arithmetic are excluded.";
  } else if (layer.type === "POOLING" && knownInputs && inputShapes.length === 1 && first.length >= 4) {
    const output = [...first];
    const h = first.length - 2;
    const w = first.length - 1;
    if (attrs.global_pooling) [output[h], output[w]] = [1, 1];
    else {
      const kernel = attrs.kernel_size?.length ? attrs.kernel_size : [3, 3];
      const stride = attrs.stride?.length ? attrs.stride : [1, 1];
      const padding = attrs.padding || { kind: "valid", amounts: [[0, 0], [0, 0]] };
      output[h] = spatialOutput(first[h], kernel[0], stride[0], padding.kind === "include_last_pixel" ? { ...padding, amount: padding.amounts?.[0] || 0 } : { ...padding, axis: 0 });
      output[w] = spatialOutput(first[w], kernel[1], stride[1], padding.kind === "include_last_pixel" ? { ...padding, amount: padding.amounts?.[1] || 0 } : { ...padding, axis: 1 });
    }
    setSingle(output, "derived_coreml_pooling_contract");
    nonMac("Pooling uses comparisons or reductions, not multiply-accumulates");
  } else if (layer.type === "PADDING" && knownInputs && inputShapes.length === 1 && first.length >= 2) {
    const amounts = attrs.padding_amounts;
    if (amounts?.length !== 2 || amounts.flat().some((value) => !Number.isSafeInteger(value) || value < 0)) throw new Error(`Core ML padding ${layer.name} has invalid spatial edge amounts`);
    const output = [...first];
    output[output.length - 2] += amounts[0][0] + amounts[0][1];
    output[output.length - 1] += amounts[1][0] + amounts[1][1];
    setSingle(output, "derived_coreml_padding_contract");
    nonMac("Padding materializes tensor elements and has no multiply-accumulates");
  } else if (layer.type === "UPSAMPLE" && knownInputs && inputShapes.length === 1 && first.length >= 3) {
    const integer = attrs.scaling_factor?.length ? attrs.scaling_factor : null;
    const fractional = attrs.fractional_scaling_factor?.length ? attrs.fractional_scaling_factor : null;
    if (integer && fractional || integer && integer.length !== 2 || fractional && fractional.length !== 2) throw new Error(`Core ML upsample ${layer.name} has conflicting or incomplete scaling factors`);
    const factors = integer || fractional || [1, 1];
    const output = [...first];
    output[output.length - 2] *= factors[0];
    output[output.length - 1] *= factors[1];
    if (output.slice(-2).some((value) => !Number.isSafeInteger(value) || value <= 0)) throw new Error(`Core ML upsample ${layer.name} does not derive integral positive spatial dimensions`);
    setSingle(output, "derived_coreml_upsample_contract");
    nonMac("Interpolation is excluded from the matrix MAC ledger");
  } else if (layer.type === "RESIZE_BILINEAR" && knownInputs && inputShapes.length === 1 && first.length >= 3) {
    const target = attrs.target_size?.length ? attrs.target_size : [1, 1];
    if (target.length !== 2 || target.some((value) => !Number.isSafeInteger(value) || value <= 0)) throw new Error(`Core ML resize ${layer.name} has an invalid target size`);
    setSingle([...first.slice(0, -2), ...target], "derived_coreml_resize_bilinear_contract");
    nonMac("Bilinear interpolation is excluded from the matrix MAC ledger");
  } else if (layer.type === "CROP_RESIZE" && knownInputs && inputShapes.length === 2) {
    const [feature, boxes] = inputShapes;
    const target = attrs.target_size?.length ? attrs.target_size : [1, 1];
    if (feature.length !== 5 || boxes.length !== 5 || feature[0] !== 1 || boxes[1] !== 1 || boxes[3] !== 1 || boxes[4] !== 1
      || ![4, 5].includes(boxes[2]) || target.length !== 2 || target.some((value) => !Number.isSafeInteger(value) || value <= 0)) throw new Error(`Core ML crop-resize ${layer.name} has an invalid feature, ROI, or target-shape contract`);
    setSingle([boxes[0], boxes[2] === 4 ? feature[1] : 1, feature[2], ...target], "derived_coreml_crop_resize_contract");
    nonMac("Crop-resize interpolation is excluded from the matrix MAC ledger");
  } else if (layer.type === "REDUCE" && knownInputs && inputShapes.length === 1) {
    const suffixes = [[-3, -2, -1], [-2, -1], [-3], [-2], [-1]];
    const axes = suffixes[attrs.axis];
    const output = axes && coreMlReducedShape(first, axes, true);
    if (!output || attrs.mode < 0 || attrs.mode > 9 || attrs.mode === 9 && attrs.axis < 2) throw new Error(`Core ML reduce ${layer.name} has an invalid mode, axis, or input rank`);
    setSingle(output, "derived_coreml_reduce_contract");
    nonMac("Reduction arithmetic is excluded from the matrix MAC ledger");
  } else if (layer.type === "POOLING_3D" && knownInputs && inputShapes.length === 1 && first.length === 5) {
    const output = [...first];
    for (let axis = 0; axis < 3; axis += 1) {
      const padding = attrs.padding_type === 2
        ? { kind: "same" }
        : { kind: "valid", amounts: [attrs.padding_type === 0 ? attrs.custom_padding[axis] : [0, 0]], axis: 0 };
      output[axis + 2] = spatialOutput(first[axis + 2], attrs.kernel_size[axis], attrs.stride[axis], padding);
    }
    if (output.slice(2).some((value) => !Number.isSafeInteger(value) || value <= 0)) throw new Error(`Core ML Pooling3D ${layer.name} derives a non-positive output shape`);
    setSingle(output, "derived_coreml_pooling_3d_contract");
    nonMac("Pooling3D uses comparisons or reductions, not multiply-accumulates");
  } else if (layer.type === "GLOBAL_POOLING_3D" && knownInputs && inputShapes.length === 1 && first.length === 5) {
    setSingle([...first.slice(0, 2), 1, 1, 1], "derived_coreml_global_pooling_3d_contract");
    nonMac("GlobalPooling3D uses comparisons or reductions, not multiply-accumulates");
  } else if (layer.type === "EMBEDDING" && knownInputs && inputShapes.length === 1 && [4, 5].includes(first.length)) {
    if (first.slice(-3).some((value) => value !== 1)) throw new Error(`Core ML embedding ${layer.name} input must end in three unit dimensions`);
    const output = [...first];
    output[first.length - 3] = attrs.output_channels;
    setSingle(output, "derived_coreml_embedding_contract");
    nonMac("Embedding performs indexed lookup; no dense matrix multiply is executed");
  } else if (layer.type === "EMBEDDING_ND" && knownInputs && inputShapes.length === 1 && first.length >= 2) {
    if (first.at(-1) !== 1) throw new Error(`Core ML EmbeddingND ${layer.name} input must end in a unit dimension`);
    setSingle([...first.slice(0, -1), attrs.embedding_size], "derived_coreml_embedding_nd_contract");
    nonMac("EmbeddingND performs indexed lookup; no dense matrix multiply is executed");
  } else if (LEGACY_BROADCASTING.has(layer.type) && knownInputs) {
    const expectedInputs = layer.type === "WHERE_BROADCASTABLE" ? 3 : 2;
    if (inputShapes.length !== expectedInputs) throw new Error(`Core ML ${layer.type} ${layer.name} requires ${expectedInputs} inputs`);
    const output = inputShapes.slice(1).reduce((shape, next) => shape && coreMlBroadcastShape(shape, next), first);
    if (!output) throw new Error(`Core ML ${layer.type} ${layer.name} inputs are not broadcast-compatible`);
    setSingle(output, "derived_coreml_broadcast_contract");
    nonMac("Broadcast elementwise arithmetic is excluded from the matrix MAC ledger");
  } else if (layer.type === "CONCAT" && knownInputs && inputShapes.length >= 2) {
    const axis = first.length - (attrs.sequence_concat ? 5 : 3);
    if (axis < 0 || inputShapes.some((shape) => shape.length !== first.length
      || shape.some((value, index) => index !== axis && value !== first[index]))) throw new Error(`Core ML concat ${layer.name} input ranks or non-concatenated dimensions do not match`);
    const output = [...first];
    output[axis] = inputShapes.reduce((sum, shape) => sum + shape[axis], 0);
    setSingle(output, "derived_coreml_concat_contract");
    nonMac("Concat copies tensor slices and has no multiply-accumulates");
  } else if (layer.type === "SPLIT" && knownInputs && inputShapes.length === 1) {
    const axis = first.length - 3;
    if (axis < 0 || attrs.n_outputs !== layer.outputs.length || first[axis] % attrs.n_outputs !== 0) throw new Error(`Core ML split ${layer.name} output count does not uniformly divide axis -3`);
    const width = first[axis] / attrs.n_outputs;
    setMany(layer.outputs.map(() => first.map((value, index) => index === axis ? width : value)), "derived_coreml_split_contract");
    nonMac("Split copies tensor slices and has no multiply-accumulates");
  } else if (layer.type === "GET_SHAPE" && knownInputs && inputShapes.length === 1) {
    setSingle([first.length], "derived_coreml_get_shape_contract");
    nonMac("GetShape reads tensor metadata and has no multiply-accumulates");
  } else if (layer.type === "TILE" && knownInputs && inputShapes.length === 1 && attrs.reps?.length) {
    if (attrs.reps.length > first.length || attrs.reps.some((value) => !Number.isSafeInteger(value) || value <= 0)) throw new Error(`Core ML tile ${layer.name} reps are invalid for the input rank`);
    const reps = [...Array(first.length - attrs.reps.length).fill(1), ...attrs.reps];
    setSingle(first.map((value, index) => value * reps[index]), "derived_coreml_tile_contract");
    nonMac("Tile copies tensor elements and has no multiply-accumulates");
  } else if (layer.type === "STACK" && knownInputs && inputShapes.length >= 2) {
    if (inputShapes.some((shape) => !sameShape(shape, first))) throw new Error(`Core ML stack ${layer.name} inputs do not have identical shapes`);
    const axis = coreMlAxis(attrs.axis, first.length + 1);
    if (axis == null) throw new Error(`Core ML stack ${layer.name} axis is outside the output rank`);
    const output = [...first];
    output.splice(axis, 0, inputShapes.length);
    setSingle(output, "derived_coreml_stack_contract");
    nonMac("Stack copies tensor elements and has no multiply-accumulates");
  } else if (layer.type === "GATHER" && knownInputs && inputShapes.length === 2) {
    const axis = coreMlAxis(attrs.axis, first.length);
    if (axis == null) throw new Error(`Core ML gather ${layer.name} axis is outside the data rank`);
    setSingle([...first.slice(0, axis), ...inputShapes[1], ...first.slice(axis + 1)], "derived_coreml_gather_contract");
    nonMac("Gather performs indexed movement and has no multiply-accumulates");
  } else if (layer.type === "SCATTER" && knownInputs && inputShapes.length === 3) {
    const axis = coreMlAxis(attrs.axis, first.length);
    const expected = axis == null ? null : [...first.slice(0, axis), ...inputShapes[1], ...first.slice(axis + 1)];
    if (!expected || !sameShape(expected, inputShapes[2])) throw new Error(`Core ML scatter ${layer.name} updates do not match the indexed container contract`);
    setSingle([...first], "derived_coreml_scatter_contract");
    nonMac("Scatter performs indexed updates and has no matrix multiply-accumulates");
  } else if (layer.type === "GATHER_ND" && knownInputs && inputShapes.length === 2) {
    const indices = inputShapes[1];
    const depth = indices.at(-1);
    if (!indices.length || depth < 1 || depth > first.length) throw new Error(`Core ML GatherND ${layer.name} index depth is invalid`);
    setSingle([...indices.slice(0, -1), ...first.slice(depth)], "derived_coreml_gather_nd_contract");
    nonMac("GatherND performs indexed movement and has no multiply-accumulates");
  } else if (layer.type === "SCATTER_ND" && knownInputs && inputShapes.length === 3) {
    const indices = inputShapes[1];
    const depth = indices.at(-1);
    const expected = indices.length && depth >= 1 && depth <= first.length ? [...indices.slice(0, -1), ...first.slice(depth)] : null;
    if (!expected || !sameShape(expected, inputShapes[2])) throw new Error(`Core ML ScatterND ${layer.name} updates do not match the multi-index contract`);
    setSingle([...first], "derived_coreml_scatter_nd_contract");
    nonMac("ScatterND performs indexed updates and has no matrix multiply-accumulates");
  } else if (layer.type === "GATHER_ALONG_AXIS" && knownInputs && inputShapes.length === 2) {
    const indices = inputShapes[1];
    const axis = coreMlAxis(attrs.axis, first.length);
    if (axis == null || indices.length !== first.length || indices.some((value, index) => index !== axis && value !== first[index])) throw new Error(`Core ML GatherAlongAxis ${layer.name} data and indices shapes are incompatible`);
    setSingle([...indices], "derived_coreml_gather_along_axis_contract");
    nonMac("GatherAlongAxis performs indexed movement and has no multiply-accumulates");
  } else if (layer.type === "SCATTER_ALONG_AXIS" && knownInputs && inputShapes.length === 3) {
    const [indices, updates] = inputShapes.slice(1);
    const axis = coreMlAxis(attrs.axis, first.length);
    if (axis == null || !sameShape(indices, updates) || indices.length !== first.length
      || indices.some((value, index) => index !== axis && value !== first[index])) throw new Error(`Core ML ScatterAlongAxis ${layer.name} inputs are incompatible`);
    setSingle([...first], "derived_coreml_scatter_along_axis_contract");
    nonMac("ScatterAlongAxis performs indexed updates and has no matrix multiply-accumulates");
  } else if (LEGACY_LIKE_FIRST.has(layer.type) && knownInputs && inputShapes.length === 1) {
    setSingle([...first], "derived_coreml_like_shape_contract");
    nonMac("Shape-like allocation or random generation has no matrix multiply-accumulates");
  } else if (LEGACY_LIKE_SECOND.has(layer.type) && knownInputs && inputShapes.length === 2) {
    if (layer.type === "RESHAPE_LIKE" && shapeProduct(first) !== shapeProduct(inputShapes[1])) throw new Error(`Core ML ReshapeLike ${layer.name} does not conserve input cardinality`);
    if (layer.type === "BROADCAST_TO_LIKE" && !sameShape(coreMlBroadcastShape(first, inputShapes[1]) || [], inputShapes[1])) throw new Error(`Core ML BroadcastToLike ${layer.name} cannot broadcast its first input to the second input shape`);
    setSingle([...inputShapes[1]], "derived_coreml_like_shape_contract");
    nonMac("Shape-like indexing or broadcasting has no multiply-accumulates");
  } else if (["FILL_STATIC", "BROADCAST_TO_STATIC"].includes(layer.type) && attrs.target_shape?.length) {
    if (layer.type === "BROADCAST_TO_STATIC" && (inputShapes.length !== 1
      || !sameShape(coreMlBroadcastShape(first, attrs.target_shape) || [], attrs.target_shape))) throw new Error(`Core ML BroadcastToStatic ${layer.name} target shape is not broadcast-compatible`);
    setSingle([...attrs.target_shape], "derived_coreml_static_shape_contract");
    nonMac("Static-shape allocation or broadcasting has no multiply-accumulates");
  } else if (["RANDOM_NORMAL_STATIC", "RANDOM_UNIFORM_STATIC", "RANDOM_BERNOULLI_STATIC"].includes(layer.type) && attrs.output_shape?.length) {
    setSingle([...attrs.output_shape], "derived_coreml_static_shape_contract");
    nonMac("Static random generation has no matrix multiply-accumulates");
  } else if (layer.type === "INNER_PRODUCT" && knownInputs && inputShapes.length === 1 && first.length >= 3) {
    const flattened = shapeProduct(first.slice(-3));
    if (flattened !== attrs.input_channels) throw new Error(`Core ML inner product ${layer.name} input cardinality ${flattened} does not match inputChannels ${attrs.input_channels}`);
    const output = [...first.slice(0, -3), attrs.output_channels, 1, 1];
    setSingle(output, "derived_coreml_inner_product_contract");
    const macProduct = multiplyCoreMlExactIntegers([...first.slice(0, -3), attrs.input_channels, attrs.output_channels]);
    result.macs = macProduct?.number ?? null;
    result.macs_decimal = macProduct?.decimal ?? null;
    result.macs_status = !macProduct ? "not_assessed_invalid_exact_cardinality" : result.macs == null ? "derived_exact_decimal_only_coreml_inner_product" : "derived_exact_coreml_inner_product";
    result.macs_reason = macProduct ? null : "MAC factors are not exact nonnegative integers";
  } else if (layer.type === "FLATTEN" && knownInputs && inputShapes.length === 1 && first.length >= 3) {
    setSingle([...first.slice(0, -3), shapeProduct(first.slice(-3)), 1, 1], "derived_coreml_flatten_contract");
    nonMac("Flatten changes tensor indexing and has no multiply-accumulates");
  } else if (layer.type === "RESHAPE" && knownInputs && inputShapes.length === 1 && attrs.target_shape?.length) {
    const target = attrs.target_shape.length === 3 ? [...first.slice(0, -3), ...attrs.target_shape] : [attrs.target_shape[0], first[1], ...attrs.target_shape.slice(1)];
    if (shapeProduct(target) !== shapeProduct(first)) throw new Error(`Core ML reshape ${layer.name} target cardinality does not preserve its input`);
    setSingle(target, "derived_coreml_reshape_contract");
    nonMac("Reshape changes tensor indexing and has no multiply-accumulates");
  } else if (layer.type === "PERMUTE" && knownInputs && inputShapes.length === 1 && first.length === 5) {
    const movable = [0, 2, 3, 4];
    setSingle([first[movable[attrs.axes[0]]], first[1], ...attrs.axes.slice(1).map((axis) => first[movable[axis]])], "derived_coreml_permute_contract");
    nonMac("Permute changes tensor indexing and has no multiply-accumulates");
  } else if (layer.type === "TRANSPOSE" && knownInputs && inputShapes.length === 1) {
    if (attrs.axes.length !== first.length || new Set(attrs.axes).size !== first.length
      || attrs.axes.some((axis) => axis < 0 || axis >= first.length)) throw new Error(`Core ML transpose ${layer.name} axes are not a permutation of the input rank`);
    setSingle(attrs.axes.map((axis) => first[axis]), "derived_coreml_transpose_contract");
    nonMac("Transpose changes tensor indexing and has no multiply-accumulates");
  } else if (layer.type === "CONCAT_ND" && knownInputs && inputShapes.length >= 2) {
    const axis = coreMlAxis(attrs.axis, first.length);
    if (axis == null || inputShapes.some((shape) => shape.length !== first.length)) throw new Error(`Core ML ConcatND ${layer.name} has an invalid axis or input rank`);
    if (inputShapes.some((shape) => shape.some((value, index) => index !== axis && value !== first[index]))) throw new Error(`Core ML ConcatND ${layer.name} non-axis dimensions do not match`);
    if (attrs.interleave && inputShapes.some((shape) => !sameShape(shape, first))) throw new Error(`Core ML ConcatND ${layer.name} interleave requires identical input shapes`);
    const output = [...first];
    output[axis] = inputShapes.reduce((sum, shape) => sum + shape[axis], 0);
    setSingle(output, "derived_coreml_concat_nd_contract");
    nonMac("ConcatND copies tensor slices and has no multiply-accumulates");
  } else if (layer.type === "SPLIT_ND" && knownInputs && inputShapes.length === 1) {
    const axis = coreMlAxis(attrs.axis, first.length);
    if (axis == null) throw new Error(`Core ML SplitND ${layer.name} axis is outside the input rank`);
    let sizes = attrs.split_sizes;
    if (!sizes.length) {
      if (attrs.num_splits !== layer.outputs.length || first[axis] % attrs.num_splits !== 0) throw new Error(`Core ML SplitND ${layer.name} uniform split does not conserve the input axis`);
      sizes = Array(attrs.num_splits).fill(first[axis] / attrs.num_splits);
    }
    if (sizes.length !== layer.outputs.length || sizes.reduce((sum, value) => sum + value, 0) !== first[axis]) throw new Error(`Core ML SplitND ${layer.name} splitSizes do not conserve the input axis`);
    setMany(sizes.map((size) => first.map((value, index) => index === axis ? size : value)), "derived_coreml_split_nd_contract");
    nonMac("SplitND copies tensor slices and has no multiply-accumulates");
  } else if (layer.type === "SQUEEZE" && knownInputs && inputShapes.length === 1) {
    let axes;
    if (attrs.squeeze_all) axes = first.flatMap((value, index) => value === 1 ? [index] : []);
    else {
      axes = attrs.axes.map((value) => coreMlAxis(value, first.length));
      if (axes.some((axis) => axis == null) || new Set(axes).size !== axes.length || axes.some((axis) => first[axis] !== 1)) throw new Error(`Core ML squeeze ${layer.name} axes are invalid or do not select unit dimensions`);
    }
    const selected = new Set(axes);
    const output = first.filter((_, index) => !selected.has(index));
    setSingle(output.length ? output : [1], "derived_coreml_squeeze_contract");
    nonMac("Squeeze changes tensor indexing and has no multiply-accumulates");
  } else if (layer.type === "EXPAND_DIMS" && knownInputs && inputShapes.length === 1) {
    const outputRank = first.length + attrs.axes.length;
    const axes = attrs.axes.map((value) => coreMlAxis(value, outputRank));
    if (axes.some((axis) => axis == null) || new Set(axes).size !== axes.length) throw new Error(`Core ML ExpandDims ${layer.name} axes are invalid or repeated`);
    const selected = new Set(axes);
    const output = [];
    let inputIndex = 0;
    for (let index = 0; index < outputRank; index += 1) output.push(selected.has(index) ? 1 : first[inputIndex++]);
    setSingle(output, "derived_coreml_expand_dims_contract");
    nonMac("ExpandDims changes tensor indexing and has no multiply-accumulates");
  } else if (layer.type === "FLATTEN_TO_2D" && knownInputs && inputShapes.length === 1) {
    const axis = coreMlAxis(attrs.axis, first.length, true);
    if (axis == null) throw new Error(`Core ML FlattenTo2D ${layer.name} axis is outside the valid boundary range`);
    setSingle([shapeProduct(first.slice(0, axis)) || 1, shapeProduct(first.slice(axis)) || 1], "derived_coreml_flatten_to_2d_contract");
    nonMac("FlattenTo2D changes tensor indexing and has no multiply-accumulates");
  } else if (layer.type === "RESHAPE_STATIC" && knownInputs && inputShapes.length === 1) {
    const output = coreMlStaticReshape(attrs.target_shape, shapeProduct(first));
    if (!output) throw new Error(`Core ML ReshapeStatic ${layer.name} targetShape does not conserve input cardinality`);
    setSingle(output, "derived_coreml_reshape_static_contract");
    nonMac("ReshapeStatic changes tensor indexing and has no multiply-accumulates");
  } else if (layer.type === "RANK_PRESERVING_RESHAPE" && knownInputs && inputShapes.length === 1) {
    const output = coreMlRankPreservingReshape(attrs.target_shape || [], first);
    if (!output) throw new Error(`Core ML RankPreservingReshape ${layer.name} target does not conserve rank and cardinality`);
    setSingle(output, "derived_coreml_rank_preserving_reshape_contract");
    nonMac("RankPreservingReshape changes tensor indexing and has no multiply-accumulates");
  } else if (layer.type === "CONSTANT_PADDING" && knownInputs && inputShapes.length === 1) {
    const pads = attrs.pad_amounts || [];
    if (pads.length !== first.length * 2) throw new Error(`Core ML ConstantPadding ${layer.name} requires two pad values per input dimension`);
    const output = first.map((value, index) => attrs.pad_to_output_size
      ? Math.max(value, pads[index * 2], pads[index * 2 + 1]) : value + pads[index * 2] + pads[index * 2 + 1]);
    if (attrs.pad_to_output_size && first.some((_, index) => pads[index * 2] && pads[index * 2 + 1])) throw new Error(`Core ML ConstantPadding ${layer.name} output-size mode has two nonzero values for one axis`);
    setSingle(output, "derived_coreml_constant_padding_contract");
    nonMac("ConstantPadding materializes tensor elements and has no multiply-accumulates");
  } else if (layer.type === "TOP_K" && knownInputs && inputShapes.length === 1 && layer.outputs.length === 2) {
    const axis = coreMlAxis(attrs.axis, first.length);
    if (axis == null || !Number.isSafeInteger(attrs.k) || attrs.k <= 0 || attrs.k > first[axis]) throw new Error(`Core ML TopK ${layer.name} has an invalid static K or axis`);
    const output = first.map((value, index) => index === axis ? attrs.k : value);
    setMany([output, [...output]], "derived_coreml_top_k_contract");
    nonMac("TopK selection is excluded from the matrix MAC ledger");
  } else if (["ARG_MIN", "ARG_MAX"].includes(layer.type) && knownInputs && inputShapes.length === 1) {
    const axis = coreMlAxis(attrs.axis, first.length);
    if (axis == null) throw new Error(`Core ML ${layer.type} ${layer.name} axis is outside the input rank`);
    const output = attrs.remove_dim && first.length > 1 ? first.filter((_, index) => index !== axis)
      : first.map((value, index) => index === axis ? 1 : value);
    setSingle(output, `derived_coreml_${layer.type.toLowerCase()}_contract`);
    nonMac(`${layer.type} selection is excluded from the matrix MAC ledger`);
  } else if (LEGACY_REDUCE_ND.has(layer.type) && knownInputs && inputShapes.length === 1) {
    const axes = attrs.reduce_all ? first.map((_, index) => index) : attrs.axes;
    const output = coreMlReducedShape(first, axes, attrs.keep_dims);
    if (!output) throw new Error(`Core ML ${layer.type} ${layer.name} axes are invalid or repeated`);
    setSingle(output, "derived_coreml_reduce_nd_contract");
    nonMac("Reduction arithmetic is excluded from the matrix MAC ledger");
  } else if (["LOAD_CONSTANT", "LOAD_CONSTANT_ND"].includes(layer.type) && attrs.shape?.length) {
    const output = layer.type === "LOAD_CONSTANT" ? [1, 1, ...attrs.shape] : [...attrs.shape];
    setSingle(output, "derived_coreml_constant_contract");
    nonMac("Loading a serialized constant has no multiply-accumulates");
  } else if (layer.type === "BATCHED_MATMUL" && knownInputs) {
    if (inputShapes.length === 1 && attrs.weight_first_dimension && attrs.weight_second_dimension) {
      const input = [...first];
      if (input.at(-1) !== attrs.weight_first_dimension) throw new Error(`Core ML batched matmul ${layer.name} input width does not match serialized weights`);
      input[input.length - 1] = attrs.weight_second_dimension;
      setSingle(input, "derived_coreml_batched_matmul_weight_contract");
      const macProduct = multiplyCoreMlExactIntegers([...first.slice(0, -1), attrs.weight_first_dimension, attrs.weight_second_dimension]);
      result.macs = macProduct?.number ?? null;
      result.macs_decimal = macProduct?.decimal ?? null;
      result.macs_status = !macProduct ? "not_assessed_invalid_exact_cardinality" : result.macs == null ? "derived_exact_decimal_only_coreml_batched_matmul" : "derived_exact_coreml_batched_matmul";
      result.macs_reason = macProduct ? null : "MAC factors are not exact nonnegative integers";
    } else if (inputShapes.length === 2 && first.length >= 1 && inputShapes[1].length >= 1) {
      const left = first.length === 1 ? [1, ...first] : [...first];
      const right = inputShapes[1].length === 1 ? [...inputShapes[1], 1] : [...inputShapes[1]];
      const m = attrs.transpose_a ? left.at(-1) : left.at(-2);
      const kA = attrs.transpose_a ? left.at(-2) : left.at(-1);
      const kB = attrs.transpose_b ? right.at(-1) : right.at(-2);
      const n = attrs.transpose_b ? right.at(-2) : right.at(-1);
      const batch = coreMlBroadcastShape(left.slice(0, -2), right.slice(0, -2));
      if (kA !== kB || !batch) throw new Error(`Core ML batched matmul ${layer.name} has incompatible contracted or broadcast dimensions`);
      setSingle([...batch, m, n], "derived_coreml_batched_matmul_input_contract");
      const macProduct = multiplyCoreMlExactIntegers([...batch, m, kA, n]);
      result.macs = macProduct?.number ?? null;
      result.macs_decimal = macProduct?.decimal ?? null;
      result.macs_status = !macProduct ? "not_assessed_invalid_exact_cardinality" : result.macs == null ? "derived_exact_decimal_only_coreml_batched_matmul" : "derived_exact_coreml_batched_matmul";
      result.macs_reason = macProduct ? null : "MAC factors are not exact nonnegative integers";
    }
  } else if (["SIMPLE_RECURRENT", "GRU", "UNIDIRECTIONAL_LSTM", "BIDIRECTIONAL_LSTM"].includes(layer.type)
    && shapeProduct(first) != null && first.length === 5) {
    if (first[2] !== attrs.input_vector_size || first[3] !== 1 || first[4] !== 1) {
      throw new Error(`Core ML recurrent layer ${layer.name} input shape does not match [Seq, Batch, inputVectorSize, 1, 1]`);
    }
    const directions = layer.type === "BIDIRECTIONAL_LSTM" ? 2 : 1;
    const sequenceOutput = ["UNIDIRECTIONAL_LSTM", "BIDIRECTIONAL_LSTM"].includes(layer.type)
      ? attrs.lstm_params.sequence_output : attrs.sequence_output;
    setSingle([sequenceOutput ? first[0] : 1, first[1], attrs.output_vector_size * directions, 1, 1], "derived_coreml_recurrent_contract");
    const gates = layer.type === "SIMPLE_RECURRENT" ? 1 : layer.type === "GRU" ? 3 : 4;
    const macProduct = multiplyCoreMlExactIntegers([first[0], first[1], directions, gates, attrs.output_vector_size,
      attrs.input_vector_size + attrs.output_vector_size]);
    result.macs = macProduct?.number ?? null;
    result.macs_decimal = macProduct?.decimal ?? null;
    result.macs_status = !macProduct ? "not_assessed_invalid_exact_cardinality" : result.macs == null ? `derived_exact_decimal_only_coreml_${layer.type.toLowerCase()}_matrix_macs` : `derived_exact_coreml_${layer.type.toLowerCase()}_matrix_macs`;
    result.macs_reason = macProduct ? null : "MAC factors are not exact nonnegative integers";
    result.macs_definition = "Recurrent MACs include serialized input and recursion matrix products for every sequence step, batch, gate, and direction; gate elementwise arithmetic, bias, clipping, and activation functions are excluded.";
  } else if (layer.type === "LAYER_NORMALIZATION" && knownInputs && inputShapes.length === 1) {
    if (attrs.normalized_shape.length > first.length
      || !sameShape(first.slice(first.length - attrs.normalized_shape.length), attrs.normalized_shape)) {
      throw new Error(`Core ML layer normalization ${layer.name} normalizedShape does not match the trailing input dimensions`);
    }
    setSingle([...first], "derived_coreml_layer_normalization_contract");
    result.macs = 0;
    result.macs_decimal = "0";
    result.macs_status = "derived_non_matrix_mac_operation";
    result.macs_reason = "The MAC ledger excludes scalar normalization reductions and affine elementwise arithmetic";
    result.macs_definition = "Layer normalization is shape-preserving; its scalar reductions and affine operations are not represented as convolution/matrix MACs.";
  } else if (LEGACY_SHAPE_PRESERVING.has(layer.type) && knownInputs && inputShapes.length && layer.outputs.length === 1) {
    if (inputShapes.slice(1).some((shape) => !sameShape(shape, first))) {
      result.shape_reason = "broadcast or multi-input shape rule requires operation-specific decoding";
    } else {
      setSingle([...first], "derived_coreml_shape_preserving_contract");
      nonMac("This decoded layer class has no multiply-accumulates");
    }
  }
  return result;
}

function graphFromNeuralNetwork(network, inputs, outputs) {
  if (!network) return null;
  const tensors = [];
  const tensorByName = new Map();
  const ensureTensor = (name, contract = null) => {
    if (tensorByName.has(name)) {
      const tensor = tensors[tensorByName.get(name)];
      if (contract?.shape?.length && tensor.shape.length && contract.shape.some((value, index) => value !== tensor.shape[index])
        || contract?.shape?.length && tensor.shape.length !== contract.shape.length) {
        throw new Error(`Core ML tensor ${name} has conflicting serialized shapes ${tensor.shape.join("x")} (${tensor.shape_source}) and ${contract.shape.join("x")} (${contract.shape_source || "unspecified source"})`);
      }
      if (contract?.shape?.length && !tensor.shape.length) {
        tensor.shape = [...contract.shape];
        tensor.shape_source = contract.shape_source || "coreml_layer_descriptor";
      }
      if (contract?.dtype && tensor.dtype === "UNKNOWN") tensor.dtype = contract.dtype;
      return tensor.index;
    }
    const index = tensors.length;
    tensors.push({
      index,
      name,
      shape: Array.isArray(contract?.shape) ? [...contract.shape] : [],
      dtype: contract?.dtype || "UNKNOWN",
      constant_buffer: false,
      quant_scales: 0,
      quant_zero_points: 0,
      shape_source: contract?.shape_source || (contract?.shape?.length ? "coreml_layer_descriptor" : "not_serialized"),
    });
    tensorByName.set(name, index);
    return index;
  };
  const inputTensorIndices = inputs.map((item) => ensureTensor(item.name, coreMlInputTensorContract(item, network)));
  const ops = [];
  for (const layer of network.layers) {
    const inputIds = layer.inputs.map((name, index) => ensureTensor(name, layer.input_shapes[index] ? {
      shape: layer.input_shapes[index], dtype: "UNKNOWN", shape_source: `coreml_layer_input_descriptor:${layer.name}:${layer.type}`,
    } : null));
    const inputShapes = inputIds.map((id) => tensors[id]?.shape || []);
    const derived = deriveLegacyLayer(layer, inputShapes);
    const outputIds = layer.outputs.map((name, index) => {
      const serialized = layer.output_shapes[index] || null;
      const inferred = derived.output_shapes[index] || null;
      if (serialized?.length && inferred?.length && !sameShape(serialized, inferred)) {
        throw new Error(`Core ML layer ${layer.name} serialized output shape ${serialized.join("x")} conflicts with the operation contract ${inferred.join("x")}`);
      }
      return ensureTensor(name, {
        shape: serialized?.length ? serialized : inferred || [],
        dtype: tensors[inputIds[0]]?.dtype || "UNKNOWN",
        shape_source: serialized?.length ? "coreml_layer_descriptor" : inferred?.length ? derived.shape_status : "not_serialized",
      });
    });
    const quantizedWeights = layer.weights.filter((weight) => ["raw_quantized", "int8_dynamic"].includes(weight.storage));
    const quantizationState = quantizedWeights.length ? "weight_only_or_dynamic_range" : "none";
    const outputBytes = outputIds.reduce((sum, id) => {
      if (sum == null) return null;
      const tensor = tensors[id];
      const tensorBytes = coreMlTensorBytes(tensor);
      return tensorBytes != null && sum <= Number.MAX_SAFE_INTEGER - tensorBytes ? sum + tensorBytes : null;
    }, 0);
    ops.push({
      index: layer.index,
      name: layer.type,
      coreml_layer_name: layer.name,
      coreml_layer_type_field: layer.type_field,
      coreml_attributes: layer.attributes,
      inputs: inputIds,
      outputs: outputIds,
      input_shapes: inputIds.map((id) => tensors[id]?.shape || []),
      output_shapes: outputIds.map((id) => tensors[id]?.shape || []),
      shape_status: derived.shape_status,
      shape_reason: derived.shape_reason,
      macs: derived.macs,
      macs_decimal: derived.macs_decimal,
      macs_status: derived.macs_status,
      macs_reason: derived.macs_reason,
      coreml_macs_definition: derived.macs_definition,
      estimated_bytes: outputBytes,
      estimated_bytes_reason: outputBytes == null ? "Core ML intermediate tensor shape or dtype is not completely derived" : "Exact output tensor cardinality multiplied by serialized or propagated element width",
      quantization_state: quantizationState,
      quantization: quantizationState,
      quantization_detail: quantizedWeights.length
        ? `${quantizedWeights.length}/${layer.weights.length} decoded weight parameter(s) use quantized storage`
        : layer.weight_scan_status === "assessed" ? `${layer.weights.length} decoded weight parameter(s); no quantized storage` : "weight scan is partial for this layer type",
      quant_risk: "none",
      coreml_weight_scan_status: layer.weight_scan_status,
      coreml_weights: layer.weights,
      stage_index: layer.index,
      stage_key: layer.type,
    });
  }
  const outputTensorIndices = outputs.map((item) => ensureTensor(item.name, {
    shape: [],
    dtype: item.feature_type === "multi_array" ? item.dtype : "UNKNOWN",
    shape_source: "coreml_external_output_contract_not_projected_into_dag",
  }));
  const weightParams = network.layers.flatMap((layer) => layer.weights.map((weight) => ({ ...weight, layer_index: layer.index, layer_name: layer.name, layer_type: layer.type })));
  const quantizedWeightParams = weightParams.filter((weight) => ["raw_quantized", "int8_dynamic"].includes(weight.storage));
  const fp32WeightParams = weightParams.filter((weight) => weight.storage === "float32");
  const fp16WeightParams = weightParams.filter((weight) => weight.storage === "float16");
  const weightParameterBytes = weightParams.reduce((sum, weight) => sum + Number(weight.byte_length || 0), 0);
  const quantizedWeightParameterBytes = quantizedWeightParams.reduce((sum, weight) => sum + Number(weight.byte_length || 0), 0);
  const fp32WeightParameterBytes = fp32WeightParams.reduce((sum, weight) => sum + Number(weight.byte_length || 0), 0);
  const fp16WeightParameterBytes = fp16WeightParams.reduce((sum, weight) => sum + Number(weight.byte_length || 0), 0);
  const perAxisQuantizedWeightParams = quantizedWeightParams.filter((weight) => Number(weight.quantization?.scale_count || 0) > 1);
  const scanComplete = network.layers.every((layer) => layer.weight_scan_status === "assessed");
  const numericalParams = weightParams.filter((weight) => weight.numerical_integrity?.status?.startsWith("assessed"));
  const nonfiniteWeightValues = numericalParams.reduce((sum, weight) => sum + Number(weight.numerical_integrity.nonfinite_count || 0), 0);
  const allZeroWeightParams = numericalParams.filter((weight) => weight.numerical_integrity.all_zero === true).length;
  const numericalBytes = numericalParams.reduce((sum, weight) => sum + Number(weight.byte_length || 0), 0);
  const assessedMacOps = ops.filter((op) => op.macs_decimal != null);
  const computeOps = ops.filter((op) => ["CONVOLUTION", "CONVOLUTION_3D", "INNER_PRODUCT", "BATCHED_MATMUL",
    "SIMPLE_RECURRENT", "GRU", "UNIDIRECTIONAL_LSTM", "BIDIRECTIONAL_LSTM"].includes(op.name));
  const assessedComputeOps = computeOps.filter((op) => op.macs_decimal != null);
  const macLedger = coreMlExactLedger(assessedComputeOps.map((op) => op.macs_decimal), assessedComputeOps.length === computeOps.length);
  const totalMacs = macLedger.complete_value;
  const stateCounts = new Map();
  for (const op of ops) stateCounts.set(op.quantization_state, (stateCounts.get(op.quantization_state) || 0) + 1);
  return {
    ops,
    tensors,
    input_tensor_indices: inputTensorIndices,
    output_tensor_indices: outputTensorIndices,
    quantization_status: {
      assessment_status: scanComplete ? "assessed" : "partial",
      classification: quantizedWeightParams.length ? "coreml_legacy_weight_quantization" : scanComplete ? "coreml_legacy_non_quantized_weights" : "coreml_legacy_partial_weight_scan",
      label: quantizedWeightParams.length
        ? "Quantized WeightParams detected"
        : scanComplete ? "No quantized WeightParams" : "Weight encodings partially assessed",
      summary: `${quantizedWeightParams.length}/${weightParams.length} decoded WeightParams use quantized storage; ${fp32WeightParams.length} FP32; ${fp16WeightParams.length} FP16.`,
      detail: `${network.layers.filter((layer) => layer.weight_scan_status === "assessed").length}/${network.layers.length} legacy NeuralNetwork layer WeightParams field scans complete. Parameter-shape cardinality and activation execution precision are not inferred from stored weight encoding.`,
      full_integer: false,
      compute_ops: null,
      quantized_compute_ops: null,
      quantized_compute_mac_percent: null,
      weight_parameter_count: weightParams.length,
      quantized_weight_parameter_count: quantizedWeightParams.length,
      fp32_weight_parameter_count: fp32WeightParams.length,
      fp16_weight_parameter_count: fp16WeightParams.length,
      weight_parameter_bytes: weightParameterBytes,
      quantized_weight_parameter_bytes: quantizedWeightParameterBytes,
      fp32_weight_parameter_bytes: fp32WeightParameterBytes,
      fp16_weight_parameter_bytes: fp16WeightParameterBytes,
      per_axis_quantized_weight_parameter_count: perAxisQuantizedWeightParams.length,
      scanned_layer_count: network.layers.filter((layer) => layer.weight_scan_status === "assessed").length,
      layer_count: network.layers.length,
      op_state_counts: [...stateCounts].map(([state, count]) => ({ state, count })),
    },
    mac_assessment: {
      status: assessedComputeOps.length === computeOps.length
        ? totalMacs == null ? "assessed_all_decoded_compute_ops_exact_decimal_only" : "assessed_all_decoded_compute_ops"
        : "partial_operation_shape_or_safe_integer_range",
      compute_ops: computeOps.length,
      assessed_compute_ops: assessedComputeOps.length,
      assessed_all_ops_including_non_mac: assessedMacOps.length,
      total_macs: totalMacs,
      assessed_macs: macLedger.assessed_value,
      assessed_macs_decimal: macLedger.assessed_value_decimal,
      complete_macs_decimal: macLedger.complete_value_decimal,
      safe_number_mirror_status: macLedger.safe_number_mirror_status,
    },
    total_macs: totalMacs,
    weight_integrity: {
      schema: "deepbom.coreml.legacy_weight_integrity.v1",
      status: numericalParams.length === weightParams.length ? "assessed" : "partial",
      parameter_count: weightParams.length,
      assessed_parameter_count: numericalParams.length,
      payload_bytes: weightParameterBytes,
      assessed_payload_bytes: numericalBytes,
      payload_byte_conservation: numericalBytes === weightParameterBytes,
      nonfinite_value_count: nonfiniteWeightValues,
      all_zero_parameter_count: allZeroWeightParams,
      parameters: weightParams.map((weight) => ({
        layer_index: weight.layer_index,
        layer_name: weight.layer_name,
        layer_type: weight.layer_type,
        role: weight.role,
        storage: weight.storage,
        byte_length: weight.byte_length,
        value_count: weight.value_count,
        numerical_integrity: weight.numerical_integrity,
      })),
    },
  };
}

function graphFromMilProgram(program, preferredFunction) {
  if (!program) return null;
  const graph = graphFromCoreMlMilProgram(program, preferredFunction);
  const quantOps = graph.ops.filter((op) => op.quantization_state !== "none");
  const blobTensors = graph.tensors.filter((tensor) => tensor.blob_reference);
  const immediateTensors = graph.tensors.filter((tensor) => tensor.constant_buffer && tensor.value_storage === "immediate");
  const immediateParameters = immediateTensors.map((tensor) => {
    const immediate = tensor.immediate_value;
    const numeric = immediate && ["float32", "float64", "int32", "int64", "bool"].includes(immediate.kind)
      && !immediate.truncated && immediate.values.length === immediate.count && immediate.values.every((value) => typeof value === "number");
    const numbers = numeric ? immediate.values.filter((value) => typeof value === "number") : [];
    const nan = numbers.filter(Number.isNaN).length;
    const positiveInfinity = numbers.filter((value) => value === Number.POSITIVE_INFINITY).length;
    const negativeInfinity = numbers.filter((value) => value === Number.NEGATIVE_INFINITY).length;
    const finite = numbers.filter(Number.isFinite);
    const zero = finite.filter((value) => value === 0).length;
    const integrity = {
      status: numeric ? "assessed_immediate_semantic_values" : "not_assessed_immediate_encoding_or_sample_truncated",
      decoded_value_count: numeric ? immediate.count : null,
      finite_count: numeric ? finite.length : null,
      zero_count: numeric ? zero : null,
      nan_count: numeric ? nan : null,
      positive_infinity_count: numeric ? positiveInfinity : null,
      negative_infinity_count: numeric ? negativeInfinity : null,
      nonfinite_count: numeric ? nan + positiveInfinity + negativeInfinity : null,
      finite_min: finite.length ? Math.min(...finite) : null,
      finite_max: finite.length ? Math.max(...finite) : null,
      all_zero: numeric ? immediate.count > 0 && zero === immediate.count : null,
      payload_sha256: null,
    };
    tensor.numerical_integrity = integrity;
    return {
      tensor_index: tensor.index, tensor_name: tensor.name, role: tensor.role, storage: "mil_immediate",
      byte_length: immediate?.kind === "bytes" && Number.isSafeInteger(immediate.byte_length) ? immediate.byte_length : null,
      value_count: immediate?.logical_count ?? immediate?.count ?? null, numerical_integrity: integrity,
    };
  });
  const assessedImmediate = immediateParameters.filter((row) => row.numerical_integrity.status.startsWith("assessed"));
  const immediatePayloadBytes = immediateParameters.every((row) => Number.isSafeInteger(row.byte_length))
    ? immediateParameters.reduce((sum, row) => sum + row.byte_length, 0) : null;
  const stateCounts = new Map();
  for (const op of graph.ops) stateCounts.set(op.quantization_state, (stateCounts.get(op.quantization_state) || 0) + 1);
  return {
    ...graph,
    quantization_status: {
      assessment_status: "assessed",
      classification: quantOps.length ? "coreml_mlprogram_serialized_quantization_transforms" : "coreml_mlprogram_no_serialized_quantization_transform",
      label: quantOps.length ? "Serialized ML Program quantization transforms detected" : "No serialized ML Program quantization transform",
      summary: `${formatCount(quantOps.length)}/${formatCount(graph.ops.length)} MIL operation(s) are explicit quantize/dequantize or constexpr compression transforms.`,
      detail: "Classification is based on serialized MIL operation types. Runtime activation precision, ANE/GPU/CPU lowering, and fused conversions are not inferred.",
      full_integer: false,
      compute_ops: graph.mac_assessment.compute_ops,
      quantized_compute_ops: null,
      quantized_compute_mac_percent: null,
      op_state_counts: [...stateCounts].map(([state, count]) => ({ state, count })),
      blob_constant_count: blobTensors.length,
      immediate_constant_count: immediateTensors.length,
    },
    weight_integrity: {
      schema: "deepbom.coreml.mlprogram_weight_integrity.v1",
      status: blobTensors.length ? assessedImmediate.length ? "partial_package_blob_binding_required" : "not_assessed_package_blob_binding_required"
        : immediateTensors.length ? immediateParameters.length === assessedImmediate.length ? "assessed" : "partial" : "not_applicable_no_serialized_constants",
      parameter_count: blobTensors.length + immediateTensors.length,
      assessed_parameter_count: assessedImmediate.length,
      payload_bytes: blobTensors.length ? null : immediateTensors.length ? immediatePayloadBytes : 0,
      assessed_payload_bytes: 0,
      payload_byte_conservation: blobTensors.length === 0 && immediatePayloadBytes != null,
      nonfinite_value_count: assessedImmediate.reduce((sum, row) => sum + Number(row.numerical_integrity.nonfinite_count || 0), 0),
      all_zero_parameter_count: assessedImmediate.filter((row) => row.numerical_integrity.all_zero).length,
      blob_reference_count: graph.blob_references.length,
      parameters: immediateParameters,
    },
  };
}

function coreMlFeatureTensorContract(feature) {
  if (feature?.feature_type === "multi_array" && shapeProduct(feature.shape) != null) {
    return { shape: [...feature.shape], dtype: feature.dtype, shape_source: "coreml_model_description" };
  }
  if (["int64", "float64"].includes(feature?.feature_type)) {
    return { shape: [1], dtype: feature.dtype, shape_source: "coreml_scalar_feature" };
  }
  if (feature?.feature_type === "image" && shapeProduct(feature.shape) != null) {
    return { shape: [...feature.shape], dtype: feature.dtype, shape_source: "coreml_image_feature" };
  }
  return { shape: [], dtype: feature?.dtype || "UNKNOWN", shape_source: "coreml_non_dense_or_flexible_feature" };
}

function staticVectorizedInputWidth(inputs) {
  let width = 0;
  for (const feature of inputs) {
    if (["int64", "float64"].includes(feature.feature_type)) width += 1;
    else if (feature.feature_type === "multi_array") {
      const count = shapeProduct(feature.shape);
      if (count == null || width > Number.MAX_SAFE_INTEGER - count) return null;
      width += count;
    } else return null;
  }
  return width;
}

function classicalWeightIntegrity(model) {
  const parameters = Array.isArray(model?.parameters) ? model.parameters : [];
  const payloadBytes = parameters.reduce((sum, row) => sum + Number(row.byte_length || 0), 0);
  const nonfinite = parameters.reduce((sum, row) => sum + Number(row.numerical_integrity?.nonfinite_count || 0), 0);
  return {
    schema: "deepbom.coreml.classical_weight_integrity.v1",
    status: "assessed",
    parameter_count: parameters.length,
    assessed_parameter_count: parameters.length,
    payload_bytes: payloadBytes,
    assessed_payload_bytes: payloadBytes,
    payload_byte_conservation: true,
    nonfinite_value_count: nonfinite,
    all_zero_parameter_count: parameters.filter((row) => row.numerical_integrity?.all_zero).length,
    parameters,
  };
}

function graphFromClassicalModel(model, inputs, outputs) {
  if (!model) return null;
  const tensors = [];
  const makeTensor = (feature, index) => {
    const contract = coreMlFeatureTensorContract(feature);
    return {
      index,
      name: feature.name,
      shape: contract.shape,
      dtype: contract.dtype,
      shape_source: contract.shape_source,
      constant_buffer: false,
      quant_scales: 0,
      quant_zero_points: 0,
    };
  };
  const inputTensorIndices = inputs.map((feature) => {
    const index = tensors.length; tensors.push(makeTensor(feature, index)); return index;
  });
  const outputTensorIndices = outputs.map((feature) => {
    const index = tensors.length; tensors.push(makeTensor(feature, index)); return index;
  });
  const inputWidth = staticVectorizedInputWidth(inputs);
  let macs = null;
  let macsDecimal = null;
  let macStatus = "not_assessed_dynamic_or_non_vector_input_contract";
  let macDefinition = "No matrix-MAC total is emitted without a static vectorized input contract.";
  if (model.kind.startsWith("glm")) {
    if (inputWidth != null && inputWidth !== model.coefficient_width) throw new Error(`Core ML ${model.kind} coefficient width ${model.coefficient_width} does not match static input width ${inputWidth}`);
    if (inputWidth != null) {
      const product = multiplyCoreMlExactIntegers([inputWidth, model.coefficient_row_count]);
      macs = product?.number ?? null;
      macsDecimal = product?.decimal ?? null;
      macStatus = !product ? "not_assessed_invalid_exact_cardinality" : macs == null ? "derived_exact_decimal_only_coreml_glm_dot_product_macs" : "derived_exact_coreml_glm_dot_product_macs";
      macDefinition = "One dot-product multiply-accumulate per serialized coefficient; offset and post-evaluation transform arithmetic are excluded.";
    }
  } else if (model.kind.startsWith("supportVector")) {
    const support = model.support_vectors;
    if (inputWidth != null) {
      if (support.kind === "dense" && support.width !== inputWidth) throw new Error(`Core ML ${model.kind} dense support-vector width ${support.width} does not match static input width ${inputWidth}`);
      if (support.kind === "sparse" && support.width > inputWidth) throw new Error(`Core ML ${model.kind} sparse support-vector index exceeds static input width ${inputWidth}`);
    }
    const kernelProduct = support.kind === "dense" ? multiplyCoreMlExactIntegers([support.count, support.width])
      : model.kernel.kind === "rbf" ? inputWidth == null ? null : multiplyCoreMlExactIntegers([support.count, inputWidth])
        : multiplyCoreMlExactIntegers([support.nonzero_entry_count]);
    const coefficientProduct = multiplyCoreMlExactIntegers([support.count, model.coefficient_row_count]);
    if (kernelProduct && coefficientProduct) {
      const total = coreMlExactLedger([kernelProduct.decimal, coefficientProduct.decimal], true);
      macs = total.complete_value;
      macsDecimal = total.complete_value_decimal;
      macStatus = macs == null ? "derived_exact_decimal_only_coreml_svm_kernel_reduction_and_coefficient_macs" : "derived_exact_coreml_svm_kernel_reduction_and_coefficient_macs";
      macDefinition = "Kernel reductions plus coefficient-weighted decision accumulations are counted. exp, pow, tanh, subtraction, bias/rho, voting, and probability calibration arithmetic are excluded.";
    }
  } else if (model.kind.startsWith("treeEnsemble")) {
    macs = 0;
    macsDecimal = "0";
    macStatus = "derived_non_matrix_mac_operation";
    macDefinition = "Tree traversal uses comparisons and leaf additions; the exact node, branch, leaf, and maximum-depth ledgers are retained separately.";
  }
  const outputBytes = outputTensorIndices.reduce((sum, index) => {
    const tensor = tensors[index];
    const bytes = coreMlTensorBytes(tensor);
    if (bytes == null) return null;
    return sum == null || sum > Number.MAX_SAFE_INTEGER - bytes ? null : sum + bytes;
  }, 0);
  const opName = model.kind.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toUpperCase();
  const op = {
    index: 0,
    name: opName,
    coreml_layer_name: model.kind,
    inputs: inputTensorIndices,
    outputs: outputTensorIndices,
    input_shapes: inputTensorIndices.map((index) => tensors[index].shape),
    output_shapes: outputTensorIndices.map((index) => tensors[index].shape),
    macs,
    macs_decimal: macsDecimal,
    macs_status: macStatus,
    macs_reason: macsDecimal == null ? macDefinition : null,
    coreml_macs_definition: macDefinition,
    estimated_bytes: outputBytes,
    estimated_bytes_reason: outputBytes == null ? "Output feature shape or dense element width is not static" : "Exact output feature cardinality multiplied by serialized element width",
    quantization_state: "none",
    quantization: "none",
    quantization_detail: "Classical Core ML parameters use serialized FLOAT64 coefficients or thresholds; no affine quantization contract is present.",
    quant_risk: "none",
    coreml_weight_scan_status: "assessed",
    coreml_classical_model: model,
    stage_index: 0,
    stage_key: opName,
  };
  const computeOp = !model.kind.startsWith("treeEnsemble");
  const totalMacs = macs;
  const weightIntegrity = classicalWeightIntegrity(model);
  return {
    ops: [op], tensors, input_tensor_indices: inputTensorIndices, output_tensor_indices: outputTensorIndices,
    total_macs: totalMacs,
    mac_assessment: {
      status: macsDecimal == null ? "partial_static_input_or_kernel_arithmetic_unavailable" : totalMacs == null ? "assessed_all_decoded_compute_ops_exact_decimal_only" : "assessed_all_decoded_compute_ops",
      compute_ops: computeOp ? 1 : 0,
      assessed_compute_ops: computeOp && macsDecimal != null ? 1 : 0,
      assessed_all_ops_including_non_mac: macsDecimal != null ? 1 : 0,
      total_macs: totalMacs,
      assessed_macs: totalMacs,
      assessed_macs_decimal: macsDecimal,
      complete_macs_decimal: macsDecimal,
      safe_number_mirror_status: macsDecimal != null && totalMacs == null ? "exact_decimal_only" : "safe_integer_mirror_available",
      method: macDefinition,
    },
    quantization_status: {
      assessment_status: "assessed",
      classification: "coreml_classical_float64_parameters",
      label: "Serialized FLOAT64 classical parameters",
      summary: `${weightIntegrity.parameter_count} numerical parameter group(s), ${weightIntegrity.payload_bytes} coefficient/threshold bytes, no affine quantization contract.`,
      detail: "The protobuf numerical payload is decoded exactly. Runtime floating-point reduction order and post-transform implementation remain external.",
      full_integer: false,
      compute_ops: computeOp ? 1 : 0,
      quantized_compute_ops: 0,
      quantized_compute_mac_percent: 0,
      op_state_counts: [{ state: "none", count: 1 }],
    },
    weight_integrity: weightIntegrity,
    classical_model: model,
  };
}

function selectedCoreMlInterface(header) {
  const description = header?.description || { inputs: [], outputs: [], states: [], functions: [] };
  const selectedFunction = description.functions?.find((item) => item.name === description.default_function_name)
    || description.functions?.[0] || null;
  return {
    selected_function: selectedFunction,
    inputs: selectedFunction?.inputs || description.inputs || [],
    outputs: selectedFunction?.outputs || description.outputs || [],
    states: selectedFunction?.states || description.states || [],
  };
}

function comparableFeatureContract(feature) {
  return JSON.stringify({
    feature_type: feature?.feature_type || null,
    dtype: feature?.dtype || null,
    shape: feature?.shape || [],
    optional: Boolean(feature?.optional),
    constraints: feature?.constraints || null,
  });
}

function assertEquivalentFeature(expected, actual, label) {
  if (comparableFeatureContract(expected) !== comparableFeatureContract(actual)) throw new Error(`${label} has a conflicting Core ML feature contract`);
}

function graphFromHeaderPayload(header, inputs, outputs, selectedFunction) {
  return graphFromNeuralNetwork(header.neural_network, inputs, outputs)
    || graphFromMilProgram(header.ml_program, selectedFunction?.name || header.description?.default_function_name || null)
    || graphFromClassicalModel(header.classical_model, inputs, outputs)
    || graphFromPipeline(header.pipeline, inputs, outputs, header.is_updatable);
}

const MAX_COREML_FLEXIBLE_SHAPE_RETAINED_SCENARIOS = 256;
const MAX_COREML_FLEXIBLE_SHAPE_EVALUATIONS = 65_536;

function coreMlFlexibleFeatureCases(feature) {
  const flexibility = feature?.constraints?.flexibility;
  const cases = [{ kind: "default", shape: [...(feature?.shape || [])], feature }];
  const add = (kind, shape, overrides = {}) => {
    const key = JSON.stringify(shape);
    if (cases.some((row) => JSON.stringify(row.shape) === key)) return;
    cases.push({ kind, shape, feature: { ...feature, ...overrides } });
  };
  if (feature?.feature_type === "multi_array" && flexibility?.kind === "enumerated") {
    for (const shape of flexibility.shapes || []) add("enumerated", [...shape], { shape: [...shape] });
  } else if (feature?.feature_type === "multi_array" && flexibility?.kind === "range") {
    const lower = flexibility.dimensions.map((range) => range.lower_bound);
    add("range_lower_endpoint", lower, { shape: lower });
    if (flexibility.dimensions.every((range) => !range.upper_bound_unbounded)) {
      const upper = flexibility.dimensions.map((range) => range.upper_bound);
      add("range_upper_endpoint", upper, { shape: upper });
    }
  } else if (feature?.feature_type === "image" && flexibility?.kind === "enumerated") {
    const channels = feature.shape?.[2];
    for (const size of flexibility.sizes || []) {
      const shape = [size.height, size.width, ...(channels ? [channels] : [])];
      add("enumerated", shape, {
        shape,
        constraints: { ...feature.constraints, width: size.width, height: size.height },
      });
    }
  } else if (feature?.feature_type === "image" && flexibility?.kind === "range") {
    const channels = feature.shape?.[2];
    const addImage = (kind, width, height) => {
      const shape = [height, width, ...(channels ? [channels] : [])];
      add(kind, shape, { shape, constraints: { ...feature.constraints, width, height } });
    };
    addImage("range_lower_endpoint", flexibility.width.lower_bound, flexibility.height.lower_bound);
    if (!flexibility.width.upper_bound_unbounded && !flexibility.height.upper_bound_unbounded) {
      addImage("range_upper_endpoint", flexibility.width.upper_bound, flexibility.height.upper_bound);
    }
  }
  return cases;
}

function buildCoreMlFlexibleInputScenarios(header, inputs, outputs, baselineGraph) {
  if (!header?.neural_network || !inputs.some((feature) => feature.constraints?.flexibility?.kind)) return {
    schema: "deepbom.coreml.flexible_input_scenarios.v1",
    status: "not_applicable_no_legacy_flexible_input_contract",
    evidence_class: "NOT_APPLICABLE",
    scenario_count: 0,
    scenarios: [],
  };
  const caseSets = inputs.map(coreMlFlexibleFeatureCases);
  const requestedScenarioCount = caseSets.reduce((count, rows) => {
    if (!Number.isSafeInteger(count) || !rows.length || count > Math.floor(Number.MAX_SAFE_INTEGER / rows.length)) return null;
    return count * rows.length;
  }, 1);
  if (requestedScenarioCount == null || requestedScenarioCount > MAX_COREML_FLEXIBLE_SHAPE_EVALUATIONS) return {
    schema: "deepbom.coreml.flexible_input_scenarios.v1",
    status: "not_assessed_scenario_product_exceeds_bound",
    evidence_class: "NOT_ASSESSED",
    requested_scenario_count: requestedScenarioCount,
    requested_scenario_count_lower_bound: requestedScenarioCount ?? Number.MAX_SAFE_INTEGER,
    scenario_evaluation_limit: MAX_COREML_FLEXIBLE_SHAPE_EVALUATIONS,
    scenario_count: 0,
    scenarios: [],
    reason: "The Cartesian product of serialized input-shape cases exceeds the explicit exhaustive-evaluation bound; no sampled subset is presented as complete.",
  };
  const scenarios = [];
  const statusCounts = new Map();
  const envelope = {
    total_macs_decimal_min: null,
    total_macs_decimal_max: null,
    input_logical_payload_bytes_min: null,
    input_logical_payload_bytes_max: null,
    output_logical_payload_bytes_min: null,
    output_logical_payload_bytes_max: null,
    peak_live_logical_payload_bytes_min: null,
    peak_live_logical_payload_bytes_max: null,
  };
  const updateNumericEnvelope = (name, value) => {
    if (!Number.isSafeInteger(value) || value < 0) return;
    const minKey = `${name}_min`;
    const maxKey = `${name}_max`;
    envelope[minKey] = envelope[minKey] == null ? value : Math.min(envelope[minKey], value);
    envelope[maxKey] = envelope[maxKey] == null ? value : Math.max(envelope[maxKey], value);
  };
  const updateDecimalEnvelope = (value) => {
    if (!/^\d+$/.test(String(value ?? ""))) return;
    const next = BigInt(value);
    if (envelope.total_macs_decimal_min == null || next < BigInt(envelope.total_macs_decimal_min)) envelope.total_macs_decimal_min = next.toString();
    if (envelope.total_macs_decimal_max == null || next > BigInt(envelope.total_macs_decimal_max)) envelope.total_macs_decimal_max = next.toString();
  };
  const evaluate = (combination, scenarioIndex) => {
    const scenarioInputs = inputs.map((feature, index) => combination[index].feature);
    let row;
    try {
      const graph = combination.every((row) => row.kind === "default") && baselineGraph
        ? baselineGraph : graphFromNeuralNetwork(header.neural_network, scenarioInputs, outputs);
      const liveness = buildCoreMlTensorLiveness(graph.ops || [], graph.tensors || [], graph.input_tensor_indices || [], graph.output_tensor_indices || [], 0);
      const inputPayload = coreMlKnownBytesForTensorSet(graph.input_tensor_indices || [], graph.tensors || []);
      const outputPayload = coreMlKnownBytesForTensorSet(graph.output_tensor_indices || [], graph.tensors || []);
      const complete = graph.mac_assessment?.complete_macs_decimal != null && !inputPayload.unknown.length
        && !outputPayload.unknown.length && ["assessed", "assessed_static_control_flow_peak_envelope"].includes(liveness.status);
      row = {
        scenario_index: scenarioIndex,
        scenario_kind: combination.some((row) => row.kind.startsWith("range_")) ? "range_endpoint"
          : combination.some((row) => row.kind === "enumerated") ? "enumerated" : "default",
        status: complete ? "assessed" : "partial",
        input_shapes: scenarioInputs.map((feature, index) => ({ name: feature.name, shape: [...combination[index].shape], case_kind: combination[index].kind })),
        total_macs: graph.total_macs,
        total_macs_decimal: graph.mac_assessment?.complete_macs_decimal ?? null,
        input_logical_payload_bytes: inputPayload.unknown.length ? null : inputPayload.bytes,
        output_logical_payload_bytes: outputPayload.unknown.length ? null : outputPayload.bytes,
        peak_live_logical_payload_bytes: liveness.peak_bytes,
        peak_live_status: liveness.peak_bytes_status,
        residuals: [
          ...(inputPayload.unknown.length ? [`${inputPayload.unknown.length} input tensor byte width(s) unresolved`] : []),
          ...(outputPayload.unknown.length ? [`${outputPayload.unknown.length} output tensor byte width(s) unresolved`] : []),
          ...(graph.mac_assessment?.complete_macs_decimal == null ? [graph.mac_assessment?.status || "MAC total unresolved"] : []),
          ...(!["assessed", "assessed_static_control_flow_peak_envelope"].includes(liveness.status) ? [liveness.status] : []),
        ],
      };
    } catch (error) {
      row = {
        scenario_index: scenarioIndex,
        scenario_kind: combination.some((row) => row.kind.startsWith("range_")) ? "range_endpoint"
          : combination.some((row) => row.kind === "enumerated") ? "enumerated" : "default",
        status: "not_assessed_shape_contract_conflict",
        input_shapes: scenarioInputs.map((feature, index) => ({ name: feature.name, shape: [...combination[index].shape], case_kind: combination[index].kind })),
        total_macs: null,
        total_macs_decimal: null,
        input_logical_payload_bytes: null,
        output_logical_payload_bytes: null,
        peak_live_logical_payload_bytes: null,
        peak_live_status: "not_assessed_shape_contract_conflict",
        residuals: [String(error?.message || error)],
      };
    }
    statusCounts.set(row.status, (statusCounts.get(row.status) || 0) + 1);
    updateDecimalEnvelope(row.total_macs_decimal);
    updateNumericEnvelope("input_logical_payload_bytes", row.input_logical_payload_bytes);
    updateNumericEnvelope("output_logical_payload_bytes", row.output_logical_payload_bytes);
    updateNumericEnvelope("peak_live_logical_payload_bytes", row.peak_live_logical_payload_bytes);
    if (scenarios.length < MAX_COREML_FLEXIBLE_SHAPE_RETAINED_SCENARIOS) scenarios.push(row);
  };
  const combination = new Array(caseSets.length);
  let evaluatedScenarioCount = 0;
  const visit = (depth) => {
    if (depth === caseSets.length) {
      evaluate(combination, evaluatedScenarioCount);
      evaluatedScenarioCount += 1;
      return;
    }
    for (const row of caseSets[depth]) {
      combination[depth] = row;
      visit(depth + 1);
    }
  };
  visit(0);
  if (evaluatedScenarioCount !== requestedScenarioCount) throw new Error("Core ML flexible-shape scenario enumeration did not conserve the serialized Cartesian product");
  const hasUnboundedRange = inputs.some((feature) => {
    const flex = feature.constraints?.flexibility;
    return flex?.kind === "range" && (flex.dimensions?.some((row) => row.upper_bound_unbounded)
      || flex.width?.upper_bound_unbounded || flex.height?.upper_bound_unbounded);
  });
  return {
    schema: "deepbom.coreml.flexible_input_scenarios.v1",
    status: statusCounts.size === 1 && statusCounts.get("assessed") === requestedScenarioCount && !hasUnboundedRange
      ? "assessed_all_serialized_cases" : "partial",
    evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED",
    scenario_count: requestedScenarioCount,
    evaluated_scenario_count: evaluatedScenarioCount,
    retained_scenario_count: scenarios.length,
    scenario_rows_truncated: scenarios.length < evaluatedScenarioCount,
    assessed_scenario_count: statusCounts.get("assessed") || 0,
    scenario_status_counts: [...statusCounts].map(([status, count]) => ({ status, count })),
    exact_envelope: envelope,
    has_unbounded_range: hasUnboundedRange,
    range_interpretation: "Range endpoint rows are exact evaluations of those endpoint shapes, not a proof that an interior point cannot have a larger cost or payload.",
    scenarios,
  };
}

function graphFromPipeline(pipeline, inputs, outputs, pipelineUpdatable) {
  if (!pipeline) return null;
  const typeTable = new Map(inputs.map((feature) => [feature.name, feature]));
  const nested = [];
  const names = pipeline.names.length ? pipeline.names : pipeline.models.map((_, index) => `model_${index}`);
  for (let index = 0; index < pipeline.models.length; index += 1) {
    const header = pipeline.models[index];
    const contract = selectedCoreMlInterface(header);
    for (const feature of contract.inputs) {
      const available = typeTable.get(feature.name);
      if (!available) throw new Error(`Core ML pipeline model ${index} input ${feature.name} is not produced by the pipeline input or a preceding model`);
      assertEquivalentFeature(available, feature, `Core ML pipeline model ${index} input ${feature.name}`);
    }
    for (const feature of contract.outputs) typeTable.set(feature.name, feature);
    if (pipelineUpdatable && index < pipeline.models.length - 1 && header.is_updatable) throw new Error("Only the final model in an updatable Core ML pipeline may be updatable");
    if (pipelineUpdatable && index === pipeline.models.length - 1 && !header.is_updatable) throw new Error("The final model in an updatable Core ML pipeline must be updatable");
    if (!pipelineUpdatable && header.is_updatable) throw new Error("A non-updatable Core ML pipeline contains an updatable nested model");
    const annotatedInputs = contract.inputs.map((feature, featureIndex) => ({ ...feature, index: featureIndex, quantization: null }));
    const annotatedOutputs = contract.outputs.map((feature, featureIndex) => ({ ...feature, index: featureIndex, quantization: null }));
    const graph = graphFromHeaderPayload(header, annotatedInputs, annotatedOutputs, contract.selected_function);
    nested.push({ index, name: names[index], header, contract, graph });
  }
  for (const feature of outputs) {
    const available = typeTable.get(feature.name);
    if (!available) throw new Error(`Core ML pipeline output ${feature.name} is not produced by the pipeline input or a nested model`);
    assertEquivalentFeature(available, feature, `Core ML pipeline output ${feature.name}`);
  }

  const tensors = [];
  const tensorByName = new Map();
  const ensureTensor = (name, source, external) => {
    const contract = source || { shape: [], dtype: "UNKNOWN" };
    if (tensorByName.has(name)) {
      const tensor = tensors[tensorByName.get(name)];
      if (tensor.shape.length && contract.shape?.length && !sameShape(tensor.shape, contract.shape)) throw new Error(`Core ML pipeline tensor ${name} has conflicting shapes`);
      if (tensor.dtype !== "UNKNOWN" && contract.dtype && contract.dtype !== "UNKNOWN" && tensor.dtype !== contract.dtype) throw new Error(`Core ML pipeline tensor ${name} has conflicting dtypes`);
      if (!tensor.shape.length && contract.shape?.length) { tensor.shape = [...contract.shape]; tensor.shape_source = contract.shape_source || "coreml_pipeline_contract"; }
      if (tensor.dtype === "UNKNOWN" && contract.dtype) tensor.dtype = contract.dtype;
      return tensor.index;
    }
    const index = tensors.length;
    tensors.push({
      ...contract,
      index,
      name,
      shape: Array.isArray(contract.shape) ? [...contract.shape] : [],
      dtype: contract.dtype || "UNKNOWN",
      constant_buffer: external ? false : Boolean(contract.constant_buffer),
      quant_scales: Number(contract.quant_scales || 0),
      quant_zero_points: Number(contract.quant_zero_points || 0),
      shape_source: contract.shape_source || (contract.shape?.length ? "coreml_pipeline_contract" : "not_serialized"),
    });
    tensorByName.set(name, index);
    return index;
  };
  const inputTensorIndices = inputs.map((feature) => ensureTensor(feature.name, coreMlFeatureTensorContract(feature), true));
  const ops = [];
  const pipelineBindings = [];
  const weightParameters = [];
  const blobReferences = [];
  let payloadBytes = 0;
  let payloadComplete = true;
  let nonfiniteCount = 0;
  let allZeroCount = 0;
  let totalMacs = 0;
  let totalMacsComplete = true;
  let computeOps = 0;
  let assessedComputeOps = 0;
  let computeCoverageComplete = true;
  for (const entry of nested) {
    const graph = entry.graph;
    if (!graph) {
      const placeholderInputs = entry.contract.inputs.map((feature) => ensureTensor(feature.name, coreMlFeatureTensorContract(feature), true));
      const placeholderOutputs = entry.contract.outputs.map((feature) => ensureTensor(feature.name, coreMlFeatureTensorContract(feature), true));
      ops.push({
        index: ops.length,
        name: `COREML_${String(entry.header.model_type || "MODEL").replace(/([a-z0-9])([A-Z])/g, "$1_$2").toUpperCase()}`,
        inputs: placeholderInputs,
        outputs: placeholderOutputs,
        input_shapes: placeholderInputs.map((id) => tensors[id].shape),
        output_shapes: placeholderOutputs.map((id) => tensors[id].shape),
        macs: null,
        macs_status: "not_assessed_nested_model_payload_not_decoded",
        macs_reason: "The nested Core ML model type is serialized but its operation-specific payload is not decoded.",
        estimated_bytes: null,
        quantization_state: "not_assessed",
        quantization: "not_assessed",
        quantization_detail: "Nested model payload not decoded",
        quant_risk: "none",
        pipeline_model_index: entry.index,
        pipeline_model_name: entry.name,
        pipeline_model_type: entry.header.model_type,
        stage_index: ops.length,
        stage_key: entry.header.model_type,
      });
      payloadComplete = false; totalMacsComplete = false; computeCoverageComplete = false;
      continue;
    }
    const inputFeatures = new Map(graph.input_tensor_indices.map((tensorIndex, featureIndex) => [tensorIndex, entry.contract.inputs[featureIndex]]));
    const outputFeatures = new Map(graph.output_tensor_indices.map((tensorIndex, featureIndex) => [tensorIndex, entry.contract.outputs[featureIndex]]));
    const remap = new Map();
    for (const tensor of graph.tensors) {
      const featureBindings = [
        inputFeatures.has(tensor.index) ? { direction: "input", feature: inputFeatures.get(tensor.index) } : null,
        outputFeatures.has(tensor.index) ? { direction: "output", feature: outputFeatures.get(tensor.index) } : null,
      ].filter((item) => item?.feature);
      if (!featureBindings.length) {
        remap.set(tensor.index, ensureTensor(`pipeline/${entry.index}:${entry.name}/${tensor.name}`, tensor, false));
        continue;
      }
      const contractTensorIds = featureBindings.map(({ feature }) => ensureTensor(feature.name, coreMlFeatureTensorContract(feature), true));
      const oneContractTensor = contractTensorIds.every((id) => id === contractTensorIds[0]);
      const contractTensor = oneContractTensor ? tensors[contractTensorIds[0]] : null;
      const identityBinding = contractTensor && pipelineTensorContractsCompatible(contractTensor, tensor);
      const mappedTensorId = identityBinding
        ? contractTensor.index
        : ensureTensor(`pipeline/${entry.index}:${entry.name}/${tensor.name}`, tensor, false);
      remap.set(tensor.index, mappedTensorId);
      for (let bindingIndex = 0; bindingIndex < featureBindings.length; bindingIndex += 1) {
        const { direction, feature } = featureBindings[bindingIndex];
        const featureTensorId = contractTensorIds[bindingIndex];
        pipelineBindings.push({
          pipeline_model_index: entry.index,
          pipeline_model_name: entry.name,
          direction,
          feature_name: feature.name,
          feature_tensor_index: featureTensorId,
          nested_tensor_index: mappedTensorId,
          binding: identityBinding && featureTensorId === mappedTensorId ? "identity" : "serialized_feature_adapter",
          feature_shape: [...(tensors[featureTensorId]?.shape || [])],
          nested_shape: [...(tensors[mappedTensorId]?.shape || [])],
          feature_dtype: tensors[featureTensorId]?.dtype || "UNKNOWN",
          nested_dtype: tensors[mappedTensorId]?.dtype || "UNKNOWN",
        });
      }
    }
    for (const op of graph.ops) {
      const index = ops.length;
      const mappedInputs = op.inputs.map((id) => remap.get(id));
      const mappedOutputs = op.outputs.map((id) => remap.get(id));
      ops.push({
        ...op,
        index,
        inputs: mappedInputs,
        outputs: mappedOutputs,
        input_shapes: mappedInputs.map((id) => tensors[id]?.shape || []),
        output_shapes: mappedOutputs.map((id) => tensors[id]?.shape || []),
        pipeline_model_index: entry.index,
        pipeline_model_name: entry.name,
        pipeline_model_type: entry.header.model_type,
        stage_index: index,
        stage_key: `${entry.index}:${op.stage_key || op.name}`,
      });
    }
    if (graph.total_macs == null || totalMacs > Number.MAX_SAFE_INTEGER - graph.total_macs) totalMacsComplete = false;
    else totalMacs += graph.total_macs;
    if (!Number.isSafeInteger(graph.mac_assessment?.compute_ops) || !Number.isSafeInteger(graph.mac_assessment?.assessed_compute_ops)) computeCoverageComplete = false;
    else { computeOps += graph.mac_assessment.compute_ops; assessedComputeOps += graph.mac_assessment.assessed_compute_ops; }
    const integrity = graph.weight_integrity;
    if (!integrity || !Number.isSafeInteger(integrity.payload_bytes)) payloadComplete = false;
    else payloadBytes += integrity.payload_bytes;
    nonfiniteCount += Number(integrity?.nonfinite_value_count || 0);
    allZeroCount += Number(integrity?.all_zero_parameter_count || 0);
    for (const parameter of integrity?.parameters || []) weightParameters.push({
      ...parameter,
      pipeline_model_index: entry.index,
      pipeline_model_name: entry.name,
      pipeline_model_type: entry.header.model_type,
    });
    for (const reference of graph.blob_references || []) blobReferences.push({ ...reference, tensor_index: remap.get(reference.tensor_index), pipeline_model_index: entry.index });
  }
  const outputTensorIndices = outputs.map((feature) => ensureTensor(feature.name, coreMlFeatureTensorContract(feature), true));
  const stateCounts = new Map();
  for (const op of ops) stateCounts.set(op.quantization_state, (stateCounts.get(op.quantization_state) || 0) + 1);
  const exactMacs = totalMacsComplete ? totalMacs : null;
  const exactPayloadBytes = payloadComplete ? payloadBytes : null;
  pipeline.model_summaries = nested.map((entry) => ({ index: entry.index, name: entry.name, model_type: entry.header.model_type, graph_status: entry.graph ? "decoded" : "not_decoded", op_count: entry.graph?.ops.length ?? null }));
  pipeline.feature_bindings = pipelineBindings;
  pipeline.feature_adapter_count = pipelineBindings.filter((binding) => binding.binding === "serialized_feature_adapter").length;
  return {
    ops, tensors, input_tensor_indices: inputTensorIndices, output_tensor_indices: outputTensorIndices,
    pipeline_feature_bindings: pipelineBindings,
    total_macs: exactMacs,
    mac_assessment: {
      status: totalMacsComplete && computeCoverageComplete ? "assessed_all_decoded_pipeline_models" : "partial_nested_model_payload_or_arithmetic",
      compute_ops: computeCoverageComplete ? computeOps : null,
      assessed_compute_ops: computeCoverageComplete ? assessedComputeOps : null,
      assessed_all_ops_including_non_mac: ops.filter((op) => op.macs != null).length,
      total_macs: exactMacs,
      pipeline_model_count: nested.length,
      decoded_pipeline_model_count: nested.filter((entry) => entry.graph).length,
    },
    quantization_status: {
      assessment_status: payloadComplete ? "assessed" : "partial",
      classification: "coreml_pipeline_mixed_model_contracts",
      label: payloadComplete ? "Pipeline numerical contracts assessed" : "Pipeline numerical contracts partially assessed",
      summary: `${nested.filter((entry) => entry.graph).length}/${nested.length} nested model payload(s) decoded; ${weightParameters.length} numerical parameter group(s).`,
      detail: "Nested graph and numerical evidence is merged once by named Core ML pipeline feature contracts; runtime fusion and compute-unit placement are not inferred.",
      full_integer: false,
      compute_ops: computeCoverageComplete ? computeOps : null,
      quantized_compute_ops: null,
      quantized_compute_mac_percent: null,
      op_state_counts: [...stateCounts].map(([state, count]) => ({ state, count })),
    },
    weight_integrity: {
      schema: "deepbom.coreml.pipeline_weight_integrity.v1",
      status: payloadComplete ? "assessed" : "partial",
      parameter_count: weightParameters.length,
      assessed_parameter_count: weightParameters.filter((row) => row.numerical_integrity?.status?.startsWith("assessed")).length,
      payload_bytes: exactPayloadBytes,
      assessed_payload_bytes: payloadBytes,
      payload_byte_conservation: payloadComplete,
      nonfinite_value_count: nonfiniteCount,
      all_zero_parameter_count: allZeroCount,
      parameters: weightParameters,
    },
    blob_references: blobReferences,
    pipeline,
  };
}

function pipelineTensorContractsCompatible(featureTensor, nestedTensor) {
  const featureShape = featureTensor?.shape || [];
  const nestedShape = nestedTensor?.shape || [];
  if (featureShape.length && nestedShape.length && !sameShape(featureShape, nestedShape)) return false;
  const featureDtype = featureTensor?.dtype || "UNKNOWN";
  const nestedDtype = nestedTensor?.dtype || "UNKNOWN";
  return featureDtype === "UNKNOWN" || nestedDtype === "UNKNOWN" || featureDtype === nestedDtype;
}

function formatCount(value) { return Number(value || 0).toLocaleString("en-US"); }

function analysisFromHeader(header, filename, fileSize) {
  const description = header.description || { inputs: [], outputs: [], states: [], training_inputs: [], functions: [], metadata: {} };
  if (description.functions.length && (description.inputs.length || description.outputs.length || description.states.length)) throw new Error("Core ML ModelDescription mixes function-level and model-level interfaces");
  const selectedFunction = description.functions.find((item) => item.name === description.default_function_name) || description.functions[0] || null;
  if (description.default_function_name && selectedFunction?.name !== description.default_function_name) throw new Error("Core ML defaultFunctionName does not resolve to a declared function");
  const inputs = selectedFunction?.inputs || description.inputs;
  const outputs = selectedFunction?.outputs || description.outputs;
  const states = selectedFunction?.states || description.states;
  if (states.some((item) => item.feature_type !== "state")) throw new Error("Core ML state interface contains a non-state FeatureType");
  if ([...inputs, ...outputs].some((item) => item.feature_type === "state")) throw new Error("Core ML input/output interface contains a state FeatureType outside the state list");
  const preprocessing = header.neural_network?.preprocessing || [];
  const unboundPreprocessing = preprocessing.filter((item) => !item.feature_name);
  for (const item of preprocessing) {
    if (!item.feature_name) continue;
    const input = inputs.find((candidate) => candidate.name === item.feature_name);
    if (!input) throw new Error(`Core ML preprocessing feature ${item.feature_name} does not resolve to a model input`);
    if (input.feature_type !== "image") throw new Error(`Core ML preprocessing feature ${item.feature_name} is not an image input`);
  }
  const annotatedInputs = inputs.map((item, index) => ({
    ...item,
    index,
    quantization: null,
    coreml_preprocessing: preprocessing.find((entry) => entry.feature_name === item.name) || null,
  }));
  const metadata = description.metadata || {};
  const predictedFeatureName = selectedFunction?.predicted_feature_name || description.predicted_feature_name || null;
  const predictedProbabilitiesName = selectedFunction?.predicted_probabilities_name || description.predicted_probabilities_name || null;
  if (predictedFeatureName && !outputs.some((item) => item.name === predictedFeatureName)) throw new Error("Core ML predictedFeatureName does not resolve to a selected output");
  if (predictedProbabilitiesName && !outputs.some((item) => item.name === predictedProbabilitiesName)) throw new Error("Core ML predictedProbabilitiesName does not resolve to a selected output");
  const annotatedOutputs = outputs.map((item, index) => ({
    ...item,
    index,
    semantic_role: item.name === predictedFeatureName
      ? "predicted_feature"
      : item.name === predictedProbabilitiesName
        ? "predicted_probabilities"
        : null,
    quantization: null,
  }));
  const graph = graphFromHeaderPayload(header, annotatedInputs, annotatedOutputs, selectedFunction);
  if (header.ml_program && graph) {
    const graphInputs = graph.input_tensor_indices.map((index) => graph.tensors[index]?.mil_ssa_name || graph.tensors[index]?.name);
    if (annotatedInputs.length && (annotatedInputs.length !== graphInputs.length || annotatedInputs.some((item) => !graphInputs.includes(item.name)))) {
      throw new Error("Core ML ML Program function inputs do not match the selected ModelDescription interface");
    }
  }
  const parserScope = graph
    ? header.ml_program
      ? `Top-level identity, selected MIL function ${graph.function_name}, opset ${graph.opset}, ${graph.ops.length} SSA operation(s), ${graph.tensors.length} typed value(s), nested blocks, blob references, source-backed conv/linear/matmul MAC rules, and source-backed serialized compression contracts were decoded. ${graph.mac_assessment.assessed_compute_ops}/${graph.mac_assessment.compute_ops} MAC-bearing operations were assessed; external blob payloads require package binding.`
      : header.neural_network
        ? `Top-level identity, interfaces, ${graph.ops.length} legacy NeuralNetwork layer(s), ${preprocessing.length} preprocessing contract(s), supported WeightParams encodings, payload digests, numerical integrity, and source-backed operation shape/MAC rules were decoded. WeightParams coverage is ${graph.quantization_status.assessment_status}; ${graph.mac_assessment.assessed_compute_ops}/${graph.mac_assessment.compute_ops} MAC-bearing layers were assessed.`
        : header.classical_model
          ? `Top-level identity, interfaces, ${header.classical_model.kind}, exact serialized parameter cardinalities, FLOAT64 numerical integrity, structural invariants, and source-backed arithmetic counts were decoded. MAC coverage is ${graph.mac_assessment.assessed_compute_ops}/${graph.mac_assessment.compute_ops}; non-MAC tree structure remains in its exact node ledger.`
          : `Top-level identity, named pipeline feature contracts, ${header.pipeline.model_summaries.length} nested model(s), ${graph.ops.length} merged serialized operation(s), numerical payloads, and exact nested MAC subtotals were decoded once. ${graph.mac_assessment.decoded_pipeline_model_count}/${graph.mac_assessment.pipeline_model_count} nested payloads produced decoded graph evidence; runtime fusion remains external.`
    : `Top-level model identity, description, functions, and interface FeatureTypes; unsupported or bounded-out model payload and weights were not decoded${header.neural_network_skip_reason || header.ml_program_skip_reason || header.classical_model_skip_reason || header.pipeline_skip_reason ? ` (${header.neural_network_skip_reason || header.ml_program_skip_reason || header.classical_model_skip_reason || header.pipeline_skip_reason})` : ""}.`;
  const deploymentFloor = buildCoreMlDeploymentContract({
    specificationVersion: header.specification_version,
    modelType: header.model_type,
    isUpdatable: header.is_updatable,
    description,
    mlProgram: header.ml_program,
  });
  if (deploymentFloor.status === "invalid_declared_version_below_observed_feature_floor") {
    throw new Error(`Core ML deployment contract is invalid: ${deploymentFloor.contradictions.join("; ")}`);
  }
  const flexibleInputScenarios = buildCoreMlFlexibleInputScenarios(header, annotatedInputs, annotatedOutputs, graph);
  const coremlEvidence = {
    ...header,
    source_basis: {
      ...COREML_FORMAT_SOURCE,
      classical_model_sources: COREML_CLASSICAL_SOURCE.files.map((row) => ({ path: row.path, sha256: row.sha256 })),
    },
    parser_scope: parserScope,
    deployment_floor: deploymentFloor,
    flexible_input_scenarios: flexibleInputScenarios,
    preprocessing_binding: {
      schema: "deepbom.coreml.preprocessing_binding.v1",
      serialized_entry_count: preprocessing.length,
      bound_entry_count: preprocessing.length - unboundPreprocessing.length,
      unbound_entry_count: unboundPreprocessing.length,
      status: unboundPreprocessing.length ? "partial_missing_required_feature_name" : "complete",
      evidence_class: "OBSERVED/DERIVED",
      source_rule: "NeuralNetwork.proto requires featureName to equal the input name; no implicit binding is inferred when it is omitted.",
    },
    ...(header.ml_program && graph?.scope_intrinsic_cost ? { mil_scope_intrinsic_cost: graph.scope_intrinsic_cost } : {}),
    ...(header.ml_program && graph?.compression_contract ? { mil_compression_contract: graph.compression_contract } : {}),
  };
  return refreshCoreMlDerivedEvidence({
    schema: "deepbom.static_analysis.coreml.v1.1",
    format: "coreml",
    filename,
    file_size: fileSize,
    file_size_bytes: fileSize,
    operator_count: graph?.ops.length ?? null,
    tensor_count: graph?.tensors.length ?? null,
    total_macs: graph?.total_macs ?? null,
    histogram: graph ? (() => {
      const counts = new Map();
      for (const op of graph.ops) counts.set(op.name, (counts.get(op.name) || 0) + 1);
      return [...counts].map(([name, count]) => ({ name, count }))
        .sort((left, right) => right.count - left.count || left.name.localeCompare(right.name));
    })() : [],
    mac_assessment: graph?.mac_assessment || {
      status: "not_assessed_model_program_payload_not_decoded",
      compute_ops: null,
      assessed_compute_ops: 0,
    },
    ops: graph?.ops || [], tensors: graph?.tensors || [],
    input_tensor_indices: graph?.input_tensor_indices || [],
    output_tensor_indices: graph?.output_tensor_indices || [],
    inputs: annotatedInputs,
    outputs: annotatedOutputs,
    states: states.map((item, index) => ({ ...item, index, quantization: null })),
    training_inputs: (description.training_inputs || []).map((item, index) => ({ ...item, index, quantization: null })),
    metadata_presence: {
      status: "assessed",
      producer_name: metadata.user_defined?.["com.github.apple.coremltools.source"] || null,
      producer_version: null,
      metadata_model_version: metadata.version || metadata.user_defined?.["com.apple.developer.machine-learning.models.version"] || null,
      metadata_author: metadata.author || null,
      metadata_license: metadata.license || null,
      metadata_model_description: metadata.short_description || null,
      preprocessing_contract_status: header.neural_network?.preprocessing_count
        ? unboundPreprocessing.length
          ? `partial_serialized_neural_network_preprocessing_${preprocessing.length - unboundPreprocessing.length}_bound_${unboundPreprocessing.length}_unbound`
          : `serialized_neural_network_preprocessing_decoded_${header.neural_network.preprocessing_count}_entries`
        : "not_assessed_from_coreml_feature_description",
      output_semantics_documented: annotatedOutputs.some((item) => Boolean(item.short_description || item.semantic_role)),
    },
    runtime_requirements: {
      platform: "Core ML",
      minimum_specification_version: header.specification_version,
      declared_os_floor: deploymentFloor.declared_load_floor,
      observed_feature_minimum_specification_version: deploymentFloor.observed_feature_minimum_specification_version,
      status: deploymentFloor.status,
      evidence_class: deploymentFloor.evidence_class,
    },
    quantization_status: graph?.quantization_status || {
      assessment_status: "not_assessed",
      classification: "coreml_payload_not_decoded",
      label: "Not assessed",
      summary: "Core ML weight and activation quantization were not decoded for this model representation.",
      detail: "No numerical classification is inferred without a decoded model payload.",
      compute_ops: null,
      quantized_compute_ops: null,
      op_state_counts: [],
    },
    weight_integrity: graph?.weight_integrity || null,
    coreml_blob_references: graph?.blob_references || [],
    flexible_input_scenarios: flexibleInputScenarios,
    format_extensions: { coreml: coremlEvidence },
    coreml: coremlEvidence,
  });
}

function parseTopLevelBytes(bytes, depth = 0) {
  if (depth > MAX_PIPELINE_DEPTH) throw new Error(`Core ML pipeline nesting exceeds ${MAX_PIPELINE_DEPTH} levels`);
  const reader = new ProtoReader(bytes, "CoreML.Model");
  const singular = new Set();
  const header = { specification_version: null, is_updatable: false, model_type: null, model_type_field: null, description: null, neural_network: null, ml_program: null, classical_model: null, pipeline: null };
  let fields = 0;
  while (!reader.done) {
    if (++fields > MAX_PROTO_FIELDS) throw new Error("Core ML model contains too many protobuf fields");
    const { field, wire } = reader.key();
    if (field === 1) header.specification_version = reader.intField(wire, singular, 1, "specificationVersion");
    else if (field === 2) {
      // Singular field: readCoreMlModelFile rejects a repeated description, and
      // this path must reach the same verdict rather than letting the last one win.
      if (singular.has(2)) throw new Error("Core ML model repeats singular field description");
      singular.add(2);
      const description = reader.bytesField(wire, "description");
      if (description.length > MAX_MODEL_DESCRIPTION_BYTES) throw new Error("Core ML model description exceeds the bounded parser limit");
      header.description = parseDescription(description);
    } else if (field === 10) header.is_updatable = reader.intField(wire, singular, 10, "isUpdatable") !== 0;
    else if (MODEL_TYPES[field]) {
      if (wire !== 2) throw new Error("Core ML model type has an invalid wire type");
      if (header.model_type_field != null) throw new Error("Core ML model contains multiple model types");
      header.model_type = MODEL_TYPES[field];
      header.model_type_field = field;
      if (LEGACY_NEURAL_NETWORK_FIELDS.has(field)) {
        const payload = reader.bytesField(wire, `CoreML.${header.model_type}`);
        header.neural_network = parseCoreMlNeuralNetwork(new ProtoReader(payload, `CoreML.${header.model_type}`));
      } else if (field === 502) {
        const payload = reader.bytesField(wire, "CoreML.mlProgram");
        header.ml_program = parseCoreMlMilProgram(new ProtoReader(payload, "CoreML.MIL.Program"));
      } else if (COREML_CLASSICAL_FIELDS.has(field)) {
        const payload = reader.bytesField(wire, `CoreML.${header.model_type}`);
        header.classical_model = parseCoreMlClassicalModel(field, new ProtoReader(payload, `CoreML.${header.model_type}`));
      } else if (COREML_PIPELINE_FIELDS.has(field)) {
        const payload = reader.bytesField(wire, `CoreML.${header.model_type}`);
        header.pipeline = parseCoreMlPipeline(field, new ProtoReader(payload, `CoreML.${header.model_type}`), parseTopLevelBytes, depth);
      } else reader.skip(wire);
    } else reader.skip(wire);
  }
  if (header.specification_version == null) throw new Error("Core ML model is missing specificationVersion");
  if (!header.description) throw new Error("Core ML model is missing ModelDescription");
  if (!header.model_type) throw new Error("Core ML model type is not recognized by the pinned schema");
  return header;
}

class BlobProtoCursor {
  constructor(file, start = 0, end = file.size) {
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || end > file.size) {
      throw new Error("Core ML protobuf range is invalid");
    }
    this.file = file;
    this.position = start;
    this.end = end;
    this.cache = new Uint8Array();
    this.cacheStart = start;
  }
  async byte() {
    if (this.position >= this.end) throw new Error("Core ML protobuf is truncated");
    if (this.position < this.cacheStart || this.position >= this.cacheStart + this.cache.length) {
      this.cacheStart = this.position;
      this.cache = new Uint8Array(await this.file.slice(this.position, Math.min(this.end, this.position + 64 * 1024)).arrayBuffer());
    }
    return this.cache[this.position++ - this.cacheStart];
  }
  async varint() {
    let value = 0n;
    for (let shift = 0n; shift < 70n; shift += 7n) {
      const byte = await this.byte();
      value |= BigInt(byte & 0x7f) << shift;
      if (!(byte & 0x80)) {
        if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("Core ML protobuf integer exceeds the safe integer range");
        return Number(value);
      }
    }
    throw new Error("Core ML protobuf contains an overlong varint");
  }
  async bytes(length) {
    if (!Number.isSafeInteger(length) || length < 0 || this.position + length > this.end) throw new Error("Core ML protobuf field exceeds the message boundary");
    const bytes = new Uint8Array(await this.file.slice(this.position, this.position + length).arrayBuffer());
    this.position += length;
    return bytes;
  }
  async skip(wire) {
    if (wire === 0) await this.varint();
    else if (wire === 1) this.position += 8;
    else if (wire === 2) {
      const length = await this.varint();
      this.position += length;
    }
    else if (wire === 5) this.position += 4;
    else throw new Error(`Core ML protobuf contains unsupported wire type ${wire}`);
    if (this.position > this.end) throw new Error("Core ML protobuf field exceeds the message boundary");
  }
}

async function readBlobMessageRange(cursor, wire, label) {
  if (wire !== 2) throw new Error(`${label} has an invalid wire type`);
  const length = await cursor.varint();
  const start = cursor.position;
  const end = start + length;
  if (!Number.isSafeInteger(end) || end > cursor.end) throw new Error(`${label} exceeds the message boundary`);
  cursor.position = end;
  return { start, end, length };
}

async function readRangeBytes(file, range) {
  return new Uint8Array(await file.slice(range.start, range.end).arrayBuffer());
}

async function validateUtf8Range(file, range, label) {
  const decoder = new TextDecoder("utf-8", { fatal: true });
  try {
    for (let start = range.start; start < range.end; start += RANGE_READ_CHUNK_BYTES) {
      const end = Math.min(range.end, start + RANGE_READ_CHUNK_BYTES);
      decoder.decode(new Uint8Array(await file.slice(start, end).arrayBuffer()), { stream: end < range.end });
    }
    decoder.decode();
  } catch { throw new Error(`${label} is not valid UTF-8`); }
}

async function decodeUtf8Range(file, range, label) {
  if (range.length > MAX_MODEL_DESCRIPTION_BYTES) throw new Error(`${label} exceeds the bounded text limit`);
  const bytes = await readRangeBytes(file, range);
  try { return new TextDecoder("utf-8", { fatal: true }).decode(bytes); }
  catch { throw new Error(`${label} is not valid UTF-8`); }
}

async function parseCoreMlNeuralNetworkRange(file, range, label) {
  const cursor = new BlobProtoCursor(file, range.start, range.end);
  const state = { layers: [], preprocessing: [], array_input_shape_mapping: 0, image_input_shape_mapping: 0 };
  const singular = new Set();
  let fields = 0;
  while (cursor.position < cursor.end) {
    if (++fields > MAX_PROTO_FIELDS) throw new Error("Core ML neural network contains too many protobuf fields");
    const key = await cursor.varint();
    const field = Math.floor(key / 8);
    const wire = key % 8;
    if (field <= 0) throw new Error("Core ML neural network contains an invalid protobuf field number");
    if (field === 1 || field === 2) {
      const recordLabel = field === 1 ? "CoreML.NeuralNetworkLayer" : "CoreML.NeuralNetworkPreprocessing";
      const record = await readBlobMessageRange(cursor, wire, recordLabel);
      const reader = new ProtoReader(await readRangeBytes(file, record), recordLabel);
      if (field === 1) state.layers.push(parseCoreMlNeuralNetworkLayer(reader, state.layers.length));
      else state.preprocessing.push(parseCoreMlNeuralNetworkPreprocessing(reader));
    } else if (field === 5 || field === 6) {
      if (wire !== 0) throw new Error(`${label} shape mapping has an invalid wire type`);
      if (singular.has(field)) throw new Error(`${label} repeats shape mapping field ${field}`);
      singular.add(field);
      const value = await cursor.varint();
      if (field === 5) state.array_input_shape_mapping = value;
      else state.image_input_shape_mapping = value;
    } else await cursor.skip(wire);
  }
  return finalizeCoreMlNeuralNetwork(state);
}

async function parseCoreMlMilProgramRange(file, range) {
  const cursor = new BlobProtoCursor(file, range.start, range.end);
  const result = { version: null, functions: {}, attributes: {}, source_basis: COREML_MIL_SOURCE };
  const singular = new Set();
  let fields = 0;
  while (cursor.position < cursor.end) {
    if (++fields > MAX_PROTO_FIELDS) throw new Error("Core ML MIL Program contains too many protobuf fields");
    const key = await cursor.varint();
    const field = Math.floor(key / 8);
    const wire = key % 8;
    if (field <= 0) throw new Error("Core ML MIL Program contains an invalid protobuf field number");
    if (field === 1) {
      if (wire !== 0 || singular.has(1)) throw new Error("Core ML MIL Program version is repeated or has an invalid wire type");
      singular.add(1);
      result.version = await cursor.varint();
    } else if (field === 2 || field === 4) {
      const label = field === 2 ? "MIL.Program.functions" : "MIL.Program.attributes";
      const entryRange = await readBlobMessageRange(cursor, wire, label);
      const [name, value] = field === 2
        ? parseCoreMlMilFunctionEntry(new ProtoReader(await readRangeBytes(file, entryRange), label))
        : parseCoreMlMilAttributeEntry(new ProtoReader(await readRangeBytes(file, entryRange), label));
      const target = field === 2 ? result.functions : result.attributes;
      if (Object.hasOwn(target, name)) throw new Error(`Core ML MIL program repeats ${field === 2 ? "function" : "attribute"} ${name}`);
      target[name] = value;
    } else if (field === 3) {
      if (singular.has(3)) throw new Error("Core ML MIL Program repeats docString");
      singular.add(3);
      await validateUtf8Range(file, await readBlobMessageRange(cursor, wire, "MIL.Program.docString"), "MIL.Program.docString");
    } else await cursor.skip(wire);
  }
  return finalizeCoreMlMilProgram(result);
}

async function parseCoreMlPipelineBodyRange(file, range, depth) {
  const cursor = new BlobProtoCursor(file, range.start, range.end);
  const models = [];
  const names = [];
  let fields = 0;
  while (cursor.position < cursor.end) {
    if (++fields > MAX_PROTO_FIELDS) throw new Error("Core ML Pipeline contains too many protobuf fields");
    const key = await cursor.varint();
    const field = Math.floor(key / 8);
    const wire = key % 8;
    if (field <= 0) throw new Error("Core ML Pipeline contains an invalid protobuf field number");
    if (field === 1) {
      if (models.length >= MAX_PIPELINE_MODELS) throw new Error(`Core ML pipeline exceeds ${MAX_PIPELINE_MODELS} nested models`);
      const modelRange = await readBlobMessageRange(cursor, wire, "CoreML.Pipeline.models");
      models.push(await parseCoreMlHeaderRange(file, modelRange, depth + 1));
    } else if (field === 2) {
      if (names.length >= MAX_PIPELINE_MODELS) throw new Error(`Core ML pipeline names exceed ${MAX_PIPELINE_MODELS} entries`);
      names.push(await decodeUtf8Range(file, await readBlobMessageRange(cursor, wire, "CoreML.Pipeline.names"), "CoreML.Pipeline.names"));
    } else await cursor.skip(wire);
  }
  if (!models.length) throw new Error("Core ML pipeline must contain at least one model");
  if (names.length && names.length !== models.length) throw new Error("Core ML pipeline model-name count does not match model count");
  if (new Set(names).size !== names.length) throw new Error("Core ML pipeline model names are not unique");
  return { schema: "deepbom.coreml.pipeline.v1", models, names, source_validation: "pinned_PipelineValidator" };
}

async function parseCoreMlPipelineRange(file, range, field, depth) {
  if (field === 202) return parseCoreMlPipelineBodyRange(file, range, depth);
  const cursor = new BlobProtoCursor(file, range.start, range.end);
  let pipeline = null;
  while (cursor.position < cursor.end) {
    const key = await cursor.varint();
    const nestedField = Math.floor(key / 8);
    const wire = key % 8;
    if (nestedField === 1) {
      if (pipeline) throw new Error("Core ML pipeline classifier/regressor repeats Pipeline");
      pipeline = await parseCoreMlPipelineBodyRange(file, await readBlobMessageRange(cursor, wire, "CoreML.Pipeline"), depth);
    } else await cursor.skip(wire);
  }
  if (!pipeline) throw new Error("Core ML pipeline classifier/regressor is missing Pipeline");
  return pipeline;
}

async function parseCoreMlHeaderRange(file, range, depth = 0) {
  if (depth > MAX_PIPELINE_DEPTH) throw new Error(`Core ML pipeline nesting exceeds ${MAX_PIPELINE_DEPTH} levels`);
  const cursor = new BlobProtoCursor(file, range.start, range.end);
  const header = { specification_version: null, is_updatable: false, model_type: null, model_type_field: null, description: null, neural_network: null, ml_program: null, classical_model: null, pipeline: null };
  const seen = new Set();
  let fields = 0;
  while (cursor.position < cursor.end) {
    if (++fields > MAX_PROTO_FIELDS) throw new Error("Core ML model contains too many protobuf fields");
    const key = await cursor.varint();
    const field = Math.floor(key / 8);
    const wire = key % 8;
    if (field <= 0) throw new Error("Core ML model contains an invalid protobuf field number");
    if ([1, 2, 10].includes(field)) {
      if (seen.has(field)) throw new Error(`Core ML model repeats singular field ${field}`);
      seen.add(field);
    }
    if (field === 1) {
      if (wire !== 0) throw new Error("Core ML specificationVersion has an invalid wire type");
      header.specification_version = await cursor.varint();
    } else if (field === 2) {
      const descriptionRange = await readBlobMessageRange(cursor, wire, "Core ML description");
      if (descriptionRange.length > MAX_MODEL_DESCRIPTION_BYTES) throw new Error("Core ML model description exceeds the bounded parser limit");
      header.description = parseDescription(await readRangeBytes(file, descriptionRange));
    } else if (field === 10) {
      if (wire !== 0) throw new Error("Core ML isUpdatable has an invalid wire type");
      header.is_updatable = (await cursor.varint()) !== 0;
    } else if (MODEL_TYPES[field]) {
      if (wire !== 2) throw new Error("Core ML model type has an invalid wire type");
      if (header.model_type_field != null) throw new Error("Core ML model contains multiple model types");
      header.model_type = MODEL_TYPES[field];
      header.model_type_field = field;
      const payloadRange = await readBlobMessageRange(cursor, wire, `CoreML.${header.model_type}`);
      if (LEGACY_NEURAL_NETWORK_FIELDS.has(field)) {
        header.neural_network = await parseCoreMlNeuralNetworkRange(file, payloadRange, `CoreML.${header.model_type}`);
      } else if (field === 502) {
        header.ml_program = await parseCoreMlMilProgramRange(file, payloadRange);
      } else if (COREML_CLASSICAL_FIELDS.has(field)) {
        header.classical_model = parseCoreMlClassicalModel(field, new ProtoReader(await readRangeBytes(file, payloadRange), `CoreML.${header.model_type}`));
      } else if (COREML_PIPELINE_FIELDS.has(field)) {
        header.pipeline = await parseCoreMlPipelineRange(file, payloadRange, field, depth);
      }
    } else await cursor.skip(wire);
  }
  if (header.specification_version == null || !header.description || !header.model_type) throw new Error("Core ML model is missing required identity, description, or type fields");
  return header;
}

export function parseCoreMlModel(bytes, filename = "model.mlmodel", fileSize = bytes.length) {
  const header = parseTopLevelBytes(bytes);
  header.payload_read_strategy = "in_memory_source_bytes";
  return analysisFromHeader(header, filename, fileSize);
}

export async function readCoreMlModelFile(file) {
  if (!file?.size) throw new Error("Core ML model file is empty");
  const header = await parseCoreMlHeaderRange(file, { start: 0, end: file.size, length: file.size });
  header.payload_read_strategy = "range_streamed_top_level_records";
  return { analysis: analysisFromHeader(header, file.name, file.size), retainedBytes: new Uint8Array(), payloadLoaded: false, payloadScanned: true };
}
