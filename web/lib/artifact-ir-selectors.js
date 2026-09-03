// Compatibility selectors are the only permitted native-shaped read surface.
// Orchestrated UI/report/export paths receive Artifact IR primary views; parser
// and isolated unit-test callers may still supply a native analysis object.
export function artifactIrOperators(analysis) {
  return Array.isArray(analysis?.ops) ? analysis.ops : [];
}

export function artifactIrValues(analysis) {
  return Array.isArray(analysis?.tensors) ? analysis.tensors : [];
}

export function artifactIrOperatorByNativeIndex(analysis, index) {
  const expected = Number(index);
  return artifactIrOperators(analysis).find((row, fallback) => nativeIndex(row, fallback) === expected) || null;
}

export function artifactIrValueByNativeIndex(analysis, index) {
  const expected = Number(index);
  return artifactIrValues(analysis).find((row, fallback) => nativeIndex(row, fallback) === expected) || null;
}

function nativeIndex(row, fallback) {
  const value = Number(row?.index);
  return Number.isSafeInteger(value) && value >= 0 ? value : fallback;
}
