import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const target = join(root, "node_modules", "@litertjs", "core", "dist", "index.js");
let source = readFileSync(target, "utf8");

const int8Needle = `  {
    dtype: "uint8",
    typedArrayConstructor: Uint8Array,
    elementType: ElementType.UINT8
  }
]);`;

const int8Replacement = `  {
    dtype: "uint8",
    typedArrayConstructor: Uint8Array,
    elementType: ElementType.UINT8
  },
  {
    dtype: "int8",
    typedArrayConstructor: Int8Array,
    elementType: ElementType.INT8
  }
]);`;

if (!source.includes('dtype: "int8"')) {
  if (!source.includes(int8Needle)) {
    throw new Error("Could not find LiteRT.js DATATYPES block to patch.");
  }
  source = source.replace(int8Needle, int8Replacement);
  console.log("Patched LiteRT.js int8 type map.");
} else {
  console.log("LiteRT.js int8 type map already patched.");
}

const dynamicHelper = `function rankedTensorTypeDimensions(rankedTensorType) {
  const layout = rankedTensorType.layout();
  const dimensions = layout.dimensions();
  layout.delete();
  return emscriptenVectorToArray(dimensions);
}
function tensorBufferCompatibleWithDynamicDims(tensorBuffer, expectedRankedTensorType) {
  const actualRankedTensorType = tensorBuffer.tensorType();
  try {
    if (actualRankedTensorType.elementType().value !== expectedRankedTensorType.elementType().value) {
      return false;
    }
    const expectedDims = rankedTensorTypeDimensions(expectedRankedTensorType);
    const actualDims = rankedTensorTypeDimensions(actualRankedTensorType);
    return expectedDims.length === actualDims.length && expectedDims.every((dim, index) => dim < 0 || dim === actualDims[index]);
  } finally {
    actualRankedTensorType.delete();
  }
}
`;

const helperNeedle = `function webGpuBufferToLiteRtTensorBuffer(gpuBuffer, shape, dtype, environment) {`;

if (!source.includes("function tensorBufferCompatibleWithDynamicDims(")) {
  if (!source.includes(helperNeedle)) {
    throw new Error("Could not find LiteRT.js tensor helper insertion point.");
  }
  source = source.replace(helperNeedle, `${dynamicHelper}${helperNeedle}`);
  console.log("Patched LiteRT.js dynamic input shape helper.");
} else {
  console.log("LiteRT.js dynamic input shape helper already patched.");
}

const checkNeedle = `      getGlobalLiteRt().liteRtWasm.checkTensorBufferCompatible(
        inputTensor.liteRtTensorBuffer,
        expectedRankedTensorType,
        inputRequirements
      );`;

const checkReplacement = `      try {
        getGlobalLiteRt().liteRtWasm.checkTensorBufferCompatible(
          inputTensor.liteRtTensorBuffer,
          expectedRankedTensorType,
          inputRequirements
        );
      } catch (error) {
        if (!tensorBufferCompatibleWithDynamicDims(inputTensor.liteRtTensorBuffer, expectedRankedTensorType)) {
          throw error;
        }
      }`;

if (!source.includes("tensorBufferCompatibleWithDynamicDims(inputTensor.liteRtTensorBuffer")) {
  if (!source.includes(checkNeedle)) {
    throw new Error("Could not find LiteRT.js input compatibility check to patch.");
  }
  source = source.replace(checkNeedle, checkReplacement);
  console.log("Patched LiteRT.js dynamic input compatibility check.");
} else {
  console.log("LiteRT.js dynamic input compatibility check already patched.");
}

writeFileSync(target, source, "utf8");
