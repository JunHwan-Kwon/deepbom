import { validateStatistics } from "./statistics.js";
import { numericalWasm } from "../numerical-wasm.js";

// IR validation remains shared with the readers. All projection/ratio/quantile
// arithmetic uses the common WASM owner, including arbitrary-precision counts.
export function describeDistribution(statistics) {
  return numericalWasm().numerical_distribution(validateStatistics(statistics));
}
export function compareDistributions(baseline, candidate) {
  return numericalWasm().numerical_distribution_comparison(validateStatistics(baseline), validateStatistics(candidate));
}
