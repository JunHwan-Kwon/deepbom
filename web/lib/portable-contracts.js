// JSON transport only; calculation and structural verdicts live in WASM.
import { numericalWasm } from "./numerical-wasm.js";
export function portableContract(operation, input) {
  try { return JSON.parse(numericalWasm().numerical_contract(operation, JSON.stringify(input))); }
  catch (error) { throw typeof error === "string" ? new Error(error) : error; }
}
export function aggregateExactCounts(values, assessedEmpty = true) {
  return portableContract("sum", { values, assessed_empty: assessedEmpty });
}
