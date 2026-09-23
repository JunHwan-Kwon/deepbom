import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { latencyStats, benchmarkNoise, movingAverage } from "../web/lib/format.js";
import { ExactComparison } from "../web/lib/numerical-ir/exact-moments.js";
import { evaluateGuardedIntegerFormula, integerSymbol, serializeGuardedIntegerFormula } from "../web/lib/guarded-integer-expression.js";
import { summarizeOnnxAssessedMacs, projectOnnxCompleteMacTotals } from "../web/onnx.js";
import { staticTensorPayloadBytes } from "../web/lib/tensor-inventory.js";

assert.equal(staticTensorPayloadBytes({ dtype: "FLOAT32", shape: [] }), 4);
assert.equal(staticTensorPayloadBytes({ dtype: "FLOAT32", shape: [0, 10] }), 0);
assert.equal(staticTensorPayloadBytes({ dtype: "INT4", shape: [3] }), 2);
assert.equal(staticTensorPayloadBytes({ dtype: "INT4", shape: [0] }), 0);
assert.equal(staticTensorPayloadBytes({ dtype: "FLOAT32", shape: [1, 4], shape_signature: [-1, 4] }), null);
assert.equal(staticTensorPayloadBytes({ dtype: "STRING", shape: [4] }), null);
assert.equal(staticTensorPayloadBytes({ dtype: "FLOAT64", shape: [2 ** 52] }), null);
for (const dimension of [null, false, true, "1", {}, -1, Number.MAX_SAFE_INTEGER + 1]) {
  assert.equal(staticTensorPayloadBytes({ dtype: "FLOAT32", shape: [dimension] }), null);
}

assert.equal(latencyStats([1e308, 1e308]).mean, 1e308);
assert.equal(latencyStats([1, 1 + Number.EPSILON]).stddev, Number.EPSILON / 2);
assert.equal(latencyStats([Number.MIN_VALUE, 0]).cv, 1);
assert.equal(latencyStats([]).mean, null);
assert.equal(latencyStats([0, 0]).cv, null);
assert.equal(benchmarkNoise([9, 1]).trimmedMean, 5);
assert.equal(benchmarkNoise([9, 1]).trimmedP50, 1);
assert.equal(benchmarkNoise([1, 2, 3, 4]).trimmedP50, 2);
assert.equal(benchmarkNoise([]).trimmedMean, null);
assert.deepEqual(movingAverage([1e308, 1e308]), [1e308, 1e308]);
assert.throws(() => movingAverage(Array(2)), /finite numbers/);
assert.deepEqual(movingAverage([1, 2, 3, 4], 2), [1.5, 2.5, 3.5, 4]);
for (const invalid of [null, false, "1", -1, Infinity, NaN]) assert.throws(() => latencyStats([invalid]), /finite nonnegative/);

// An independent rational/decimal oracle: do not derive expected answers from
// DEEPBOM's arithmetic, normalizers, or serializers.
const oracle = spawnSync("python3", ["-c", String.raw`
import decimal, fractions, json, math, random
decimal.getcontext().prec=1800
D=decimal.Decimal
F=fractions.Fraction
def dec(x): return D(x.numerator)/D(x.denominator)
def number(x):
 try: y=float(x)
 except OverflowError: return None
 return y if math.isfinite(y) else None
pairs=[([1,0],[1,1e-8]),([1,0],[1,1e-150]),([1e308,0],[-1e308,0]),([1e-308,0],[0,1e-308]),([0,0],[1,0]),([1e16,1,-1e16],[1,1,1])]
r=random.Random(230926)
for _ in range(120):
 a=[math.ldexp(r.randint(-8,8),r.randint(-1020,1000)) for i in range(r.randint(1,16))]
 b=[math.ldexp(r.randint(-8,8),r.randint(-1020,1000)) for i in a]
 pairs.append((a,b))
rows=[]
for a,b in pairs:
 x=list(map(F,a));y=list(map(F,b));n=len(x)
 delta=[r-l for l,r in zip(x,y)];ab=sum(map(abs,delta));sq=sum(v*v for v in delta)
 l=sum(v*v for v in x);r=sum(v*v for v in y);dot=sum(v*w for v,w in zip(x,y))
 rows.append(dict(a=a,b=b,expected=dict(total_absolute_difference=number(ab),mean_absolute_difference=number(ab/n),root_mean_square_difference=number(dec(sq/n).sqrt()),maximum_absolute_difference=number(max(map(abs,delta))),relative_l2_difference=number(dec(sq/l).sqrt()) if l else None,cosine_distance=number(1-dec(dot)/dec(l*r).sqrt()) if l and r else None)))
print(json.dumps(rows,allow_nan=False))
`], { encoding: "utf8", maxBuffer: 8 * 1024 * 1024 });
assert.equal(oracle.status, 0, oracle.stderr);
const cases = JSON.parse(oracle.stdout);
let comparisons = 0;
for (const row of cases) {
  const total = new ExactComparison(), left = new ExactComparison(), right = new ExactComparison();
  row.a.forEach((value, index) => {
    total.add(value, row.b[index]);
    (index % 2 ? left : right).add(value, row.b[index]);
  });
  const actual = total.finish(), merged = left.merge(right).finish();
  for (const [name, expected] of Object.entries(row.expected)) {
    assert(actual[name] === expected, `${name}: ${actual[name]} != ${expected}; ${JSON.stringify(row)}`);
    comparisons++;
  }
  assert.deepEqual(merged, actual, "Tensor/sample partitioning must not change numeric metrics.");
}

const guarded = serializeGuardedIntegerFormula(integerSymbol("N"), "bytes", "Test binding identity", []);
for (const value of [null, true, false, "", " ", [], {}, 9007199254740992]) {
  assert.equal(evaluateGuardedIntegerFormula(guarded, { N: value }), null);
}
assert.equal(evaluateGuardedIntegerFormula(guarded, { N: "9007199254740993" }), 9007199254740993n);
assert.throws(() => summarizeOnnxAssessedMacs([9007199254740992]), /invalid nonnegative integer/);
for (const value of [null, true, false, "", "0"]) {
  assert.throws(() => projectOnnxCompleteMacTotals(summarizeOnnxAssessedMacs([4]), value), /nonnegative safe integer/);
}
console.log(`Calculation boundaries passed: ${comparisons} independent exact metrics, ${cases.length} aggregation invariants, strict symbolic and MAC bindings.`);
