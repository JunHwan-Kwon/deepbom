import { createCheck } from "./check-assert.mjs";
import {
  overridesFromForm,
  validateCustomTargetSpec,
} from "../web/lib/custom-targets.js";

const { done, expectDeepEqual, expectEqual, expectThrows } = createCheck("Custom target contract check");

const base = {
  id: "android_mid_a55",
  chain_break_overhead_us_low: 18,
  chain_break_overhead_us_high: 55,
  effective_peak_gops: 300,
  compute_utilization_factor: 1,
};

const valid = validateCustomTargetSpec({
  base: base.id,
  id: "custom:a55-lab",
  label: "A55 lab profile",
  evidence_class: "MEASURED",
  evidence_note: "benchmark_model p50, one thread",
  overrides: { compute_utilization_factor: 0.125 },
}, base);
expectEqual(valid.overrides.compute_utilization_factor, 0.125, "Valid utilization override");

expectThrows(() => validateCustomTargetSpec({
  ...valid,
  id: "custom:측정",
}, base), "printable ASCII", "Unicode custom target identifiers must fail before audit");

expectThrows(() => validateCustomTargetSpec({
  ...valid,
  overrides: { chain_break_overhead_us_high: 17 },
}, base), "high must not be below low", "High override must be compared with inherited low");

expectThrows(() => validateCustomTargetSpec({
  ...valid,
  overrides: { chain_break_overhead_us_low: 56 },
}, base), "high must not be below low", "Low override must be compared with inherited high");

expectDeepEqual(overridesFromForm(base, {
  effective_peak_gops: "300",
  compute_utilization_factor: "0.25",
}), { compute_utilization_factor: 0.25 }, "Form conversion must retain only changed numeric values");

done("Custom target contract passed (ASCII identity, inherited bounds, and override normalization).");
