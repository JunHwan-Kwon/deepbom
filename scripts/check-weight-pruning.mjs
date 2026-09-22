import assert from 'node:assert/strict';
import { prepareMagnitudePruning, simulateMagnitudePruning } from '../web/lib/weight-pruning.js';
import { matrixView } from '../web/lib/numerical-ir/weight-math.js';
import { checkDigest } from '../web/lib/numerical-ir/common.js';
import { sha256BytesHex } from '../web/lib/sha256-sync.js';

const source = { artifact_sha256: 'a'.repeat(64), model_ir_sha256: 'b'.repeat(64), artifact_set_sha256: null };
const make = (values, shape = [2, values.length / 2], axis = 0, order = 'last_axis_fastest', representation = 'stored_scalar', dtype = 'FLOAT64') => prepareMagnitudePruning({
  matrix: matrixView(values, shape, axis, order), source, weight_ir_sha256: 'd'.repeat(64),
  row: { weight_ref: 'weight:0', status: 'assessed', shape, dtype, representation, axes: { channel_axis: axis, storage_order: order } },
});
const near = (a, b) => assert(Math.abs(a - b) < Math.max(1e-14, Math.abs(b) * 1e-12), `${a} != ${b}`);
const fixture = [0, -0, 1, -1, 2, -3, 4, -5], prepared = make(fixture);
const mid = simulateMagnitudePruning(prepared, { target_percent: 50 });
assert.equal(mid.original_zero_count, 2); assert.equal(mid.newly_zeroed_count, 2); assert.equal(mid.candidate_zero_count, 4);
near(mid.retained_energy_fraction, 54 / 56); near(mid.metrics.rmse, .5);
assert.equal(mid.mask_sha256, sha256BytesHex(Uint8Array.from([0, 0, 1, 1, 0, 0, 0, 0])));
assert.deepEqual([...prepared.data], fixture, 'preview does not mutate its source');
assert.deepEqual(simulateMagnitudePruning(prepared, { target_percent: 50 }), mid, 'deterministic preview');
checkDigest(mid, 'pruning_preview_sha256');
const channel = simulateMagnitudePruning(prepared, { target_percent: 50, channel: 1 });
assert.equal(channel.mask_sha256, mid.mask_sha256, 'drill-down cannot change the experiment');
assert.equal(channel.projections.original.cells.reduce((n, c) => n + c.count, 0), 4);
assert.equal(simulateMagnitudePruning(prepared, { target_percent: 0 }).candidate_zero_count, 2, 'existing zeros are never filled');
assert.equal(simulateMagnitudePruning(prepared, { target_percent: 100 }).metrics.relative_l2, 1);
assert.equal(simulateMagnitudePruning(make([0, -0, 0, 0])).retained_energy_fraction, null);
const tie = simulateMagnitudePruning(make([1, -1, 1, -1]), { target_percent: 25 });
assert.equal(tie.mask_sha256, sha256BytesHex(Uint8Array.from([1, 0, 0, 0])), 'ties use explicit unfolded positions');
assert.throws(() => simulateMagnitudePruning(prepared, { target_percent: 1.5 }), /integer percentage/);
assert.throws(() => simulateMagnitudePruning(prepared, { channel: 2 }), /outside/);
assert.throws(() => make([NaN, 0]), /finite/);
assert.throws(() => make([1, 2], [1, 2], 0, 'last_axis_fastest', 'stored_scalar', 'INT8'), /integer codes/);
assert(simulateMagnitudePruning(make([1, 2], [1, 2], 0, 'last_axis_fastest', 'dequantized_real', 'INT8')));
assert.throws(() => make(Array(4097).fill(1), [4097, 1]), /4,096 channels/);

// Independent exhaustive mask and energy oracle for every slider position.
let seed = 1234567;
for (let trial = 0; trial < 12; trial++) {
  const values = Array.from({ length: 240 }, () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return (seed % 31) - 15; });
  const input = make(values, [3, 4, 20], trial % 3, trial % 2 ? 'first_axis_fastest' : 'last_axis_fastest');
  const truth = Array.from(input.data), originalZeros = truth.filter(x => x === 0).length;
  const ranked = truth.map((v, i) => ({ v, i })).filter(x => x.v !== 0).sort((a, b) => Math.abs(a.v) - Math.abs(b.v) || a.i - b.i);
  for (let target = 0; target <= 100; target++) {
    const n = Math.max(0, Math.floor(truth.length * target / 100) - originalZeros), removed = new Set(ranked.slice(0, n).map(x => x.i));
    const expected = truth.map((v, i) => removed.has(i) ? 0 : v), actual = simulateMagnitudePruning(input, { target_percent: target });
    const originalEnergy = truth.reduce((sum, v) => sum + v * v, 0), energy = expected.reduce((sum, v) => sum + v * v, 0);
    near(actual.retained_energy_fraction, energy / originalEnergy); near(actual.metrics.relative_l2, Math.sqrt((originalEnergy - energy) / originalEnergy));
    assert.equal(actual.candidate_zero_count, expected.filter(x => x === 0).length);
    assert.equal(actual.channels.reduce((n, c) => n + c.newly_zeroed_count, 0), n);
    assert.equal(actual.projections.removed.cells.reduce((n, c) => n + c.count - c.zero_count, 0), n);
    assert.equal(actual.mask_sha256, sha256BytesHex(Uint8Array.from(truth, (_, i) => Number(removed.has(i)))));
  }
}
const extreme = simulateMagnitudePruning(make([1e308, -1e308, 1e-200, 0]), { target_percent: 75 });
assert.equal(extreme.newly_zeroed_count, 2); near(extreme.retained_energy_fraction, .5); near(extreme.metrics.relative_l2, Math.SQRT1_2);
console.log('PASS: pruning preview, 1,212 independent slider/mask oracles, counts, axes, zero/quantization boundaries, deterministic identity and extreme values.');
