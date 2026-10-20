import { portableContract } from "./portable-contracts.js";

// Host text/number coercion preserves the existing public call contract. The
// grammar, rank expansion and label relationships are owned by common WASM.
export function parseOnnxEinsumEquation(equation, ranks) {
  return portableContract("einsum_equation", {
    source: String(equation || "").replace(/\s+/g, ""),
    ranks: ranks.map(Number),
  });
}
