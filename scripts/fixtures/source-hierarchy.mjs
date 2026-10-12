// Small synthetic ONNX oracle. No private model or framework dependency.
const join = rows => Uint8Array.from(rows.flatMap(row => [...row]));
const vint = input => { let n = BigInt(input); const out = []; do { let b = Number(n & 127n); n >>= 7n; if (n) b |= 128; out.push(b); } while (n); return Uint8Array.from(out); };
const field = (id, value) => join([vint(id * 8 + 2), vint(value.length), value]);
const str = (id, text) => field(id, new TextEncoder().encode(text));
const int = (id, value) => join([vint(id * 8), vint(value)]);
const info = name => join([str(1, name), field(2, field(1, join([int(1, 1), field(2, join([field(1, int(1, 1)), field(1, int(1, 2))]))])))]);
export function hierarchyFixture({ duplicate = false, malformed = false, name = "enc.0", annotate = true, unknown = false, weights = false, distinctWeights = false, modelMetadata = [] } = {}) {
  const op = (kind, id, inputs, output, scope, call) => {
    const entries = annotate ? [
      ["pkg.torch.onnx.name_scopes", malformed ? "[__import__('os')]" : JSON.stringify(["", scope, call])],
      ["pkg.torch.onnx.class_hierarchy", JSON.stringify(["test.Model", "test.Shared", `aten.${kind.toLowerCase()}.default`])],
      ...(duplicate ? [["pkg.torch.onnx.name_scopes", "['contradiction']"]] : []),
    ] : [];
    return join([...inputs.map(v => str(1, v)), str(2, output), str(3, id), str(4, kind),
      ...entries.map(([k, v]) => field(9, join([str(1, k), str(2, v)])))]);
  };
  const graph = join([
    field(1, op(weights ? "Mul" : "Relu", "first", weights ? ["x", "w"] : ["x"], "a", name, "relu_0")),
    field(1, op(unknown ? "UnrecognizedProbe" : "Add", "sum", ["a", weights ? distinctWeights ? "w2" : "w" : "a"], "b", "dec.0", "add_0")),
    field(1, op("Identity", "second_call", ["a"], "c", name, "alias_1")),
    str(2, "source_hierarchy_oracle"), field(11, info("x")),
    ...["a", "b", "c"].map(v => field(12, info(v))),
    ...(weights ? ["w", ...(distinctWeights ? ["w2"] : [])].map(name => field(5, join([int(1, 2), int(2, 1), str(8, name), field(9, new Uint8Array(new Float32Array([1, 1]).buffer))]))) : []),
  ]);
  return join([int(1, 10), str(2, "deepbom_synthetic_source_oracle"), field(7, graph), field(8, join([str(1, ""), int(2, 18)])),
    ...modelMetadata.map(([key, value]) => field(14, join([str(1, key), str(2, value)])))]);
}
