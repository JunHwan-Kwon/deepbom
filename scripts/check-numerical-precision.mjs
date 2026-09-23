import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { statisticsOf, validateStatistics } from '../web/lib/numerical-ir/statistics.js';
import { tileProjection, spectrumAnalysis, matrixView, sparsityAnalysis, cosine } from '../web/lib/numerical-ir/weight-math.js';
import { dequantize } from '../web/lib/numerical-ir/weight-contracts.js';
import { roundTiesAway } from '../web/lib/quantization-math.js';
import { formatExactInteger, formatPercent, formatPercentRange, formatNumber, formatBytes, formatDrift, formatPercent1, formatScientific, score100 } from '../web/lib/format.js';
import { summarizeProjectedHessians, computeHessian2D, subtractCenter, aggregateGrids, computeRadialProfileSEM, requantRatio, computePatternInputStats } from '../web/lib/research.js';
import { macDistributionData } from '../web/lib/analysis.js';
import { countRatio, zeroLabel } from '../web/lib/weight-visuals.js';

// Independent standard-library oracle: exact rational moments, then Decimal
// square roots at 1800 digits. This does not reuse the JS accumulation method.
const oracle = spawnSync('python3', ['-c', String.raw`
import decimal, fractions, json, math, random, sys
decimal.getcontext().prec = 1800
r = random.Random(9023)
tiny = math.ulp(0.0)
cases = [[1e16, 1, -1e16], [1e308, 1, -1e308], [1e308, 1e-308, -1e308],
         [1e16, 1e16+2], [1, math.nextafter(1, math.inf)], [tiny, 0],
         [tiny, tiny*2], [tiny]*3, [sys.float_info.max]*2, [-1e308, 1e308]]
for i in range(160):
    n = r.randrange(1, 80)
    if i % 3 == 0:
        center = math.ldexp(r.uniform(-1, 1), r.randrange(-1000, 1000))
        values = [center + r.randrange(-3, 4)*math.ulp(center) for _ in range(n)]
    else:
        values = [math.ldexp(r.uniform(-1, 1), r.randrange(-1074, 1024)) for _ in range(n)]
    cases.append(values)
def finite(x):
    return x if math.isfinite(x) else None
def sqrt(q):
    return finite(float((decimal.Decimal(q.numerator)/decimal.Decimal(q.denominator)).sqrt()))
out = []
for values in cases:
    n = len(values)
    xs = list(map(fractions.Fraction, values))
    mean = sum(xs)/n
    square = sum(x*x for x in xs)
    ys = list(reversed(xs))
    dot = sum(x*y for x,y in zip(xs,ys))
    cos = math.copysign(sqrt(dot*dot/(square*square)), -1 if dot<0 else 1) if square else None
    out.append(dict(values=values, mean=float(mean), population_stddev=sqrt(square/n-mean*mean),
                    rms=sqrt(square/n), l2_norm=sqrt(square), cosine=cos))
print(json.dumps(out, allow_nan=False))
`], { encoding: 'utf8', maxBuffer: 8*1024*1024 });
for(const absent of [null,undefined,'',false]) {
  assert.equal(formatExactInteger(null,absent),'Not assessed');
  assert.equal(formatPercentRange(absent,0.5),'Not assessed');
  for(const format of [formatNumber,formatBytes,formatDrift,formatPercent,formatPercent1,formatScientific,score100]) assert.equal(format(absent),'Not assessed');
}
assert.equal(formatNumber('9007199254740993'),'9,007,199,254,740,993');assert.equal(formatNumber(0),'0');
assert.equal(formatPercentRange(0,1e-13),'0% to 1.00e-11%');
assert.equal(formatExactInteger(null,0),'0');assert.equal(formatExactInteger('9007199254740993'),'9,007,199,254,740,993');
assert.equal(formatExactInteger(null,null,'N/A'),'N/A');
assert.equal(formatPercent(-1e-8),'-1.00e-6%');assert.equal(formatPercent(1e-8),'1.00e-6%');
assert.equal(macDistributionData({total_macs:1,mac_assessment:{compute_ops:null,assessed_compute_ops:null},ops:[]}).coverageComplete,false);
assert.deepEqual(summarizeProjectedHessians([{lambdaMax:1e16},{lambdaMax:1e16+2}]),{assessedCount:2,lambdaMean:1e16,lambdaStd:1,directionalLambdaMaxCv:1e-16});
assert(summarizeProjectedHessians([{lambdaMax:1e-14},{lambdaMax:2e-14}]).directionalLambdaMaxCv>0);
const h=1e-14,axes=[-h,0,h],quadratic=axes.map(y=>axes.map(x=>x*x+y*y));
const curvature=computeHessian2D(quadratic,axes,3);assert(curvature);assert.equal(curvature.Haa,2);assert.equal(curvature.Hbb,2);
const missingCenter=[[1,1,1],[1,null,1],[1,1,1]];
assert.equal(computeHessian2D(missingCenter,[-1,0,1],3),null);
assert(subtractCenter(missingCenter,3).flat().every(Number.isNaN));
const single=aggregateGrids([[[5]]],1);assert.equal(single.mean[0][0],5);assert(Number.isNaN(single.sem[0][0]));
const missing=aggregateGrids([[[null]]],1);assert(Number.isNaN(missing.centerLoss));assert(Number.isNaN(missing.maxDmean));
const radial=computeRadialProfileSEM([[[5]]],[0],1,4);assert.equal(radial.mu[0],5);assert(Number.isNaN(radial.mu[1]));assert.deepEqual(radial.sample_counts,[1,0,0,0]);assert(radial.sem.every(Number.isNaN));
assert.equal(requantRatio([[2e-300]],[[1e-300]],1),2);assert.equal(requantRatio([[1e308]],[[1e308]],1),1);assert.equal(requantRatio([[1]],[[0]],1),null);
assert.equal(computePatternInputStats([1e16,1,-1e16],'float64').mean,1/3);
assert.equal(countRatio('1','8000000'),1/8000000);
assert.equal(countRatio('1'+'0'.repeat(400),'2'+'0'.repeat(400)),0.5);
assert.equal(zeroLabel(statisticsOf([0n,9007199254740993n])),'50.00%');
assert.equal(zeroLabel({zero_count:'1',finite_count:'8000000'}),'1.25e-5%');
assert.equal(oracle.status, 0, oracle.stderr);
const cases = JSON.parse(oracle.stdout);
for(const row of cases) assert(cosine(row.values,[...row.values].reverse())===row.cosine,'exact dot/energy cosine oracle');
assert.equal(cosine([1e16,1,-1e16],[1,1,1]),4.0824829046386305e-17);
assert.equal(cosine([0,0],[1,2]),null);
let momentChecks = 0;
for (const row of cases) for (const values of [row.values, [...row.values].reverse(), [...row.values].sort((a,b)=>a-b)]) {
  const actual = statisticsOf(values); validateStatistics(actual);
  for (const key of ['mean','population_stddev','rms','l2_norm']) {
    assert(actual[key] === row[key], `${key}: ${actual[key]} != ${row[key]} (${JSON.stringify(values)})`);
    momentChecks++;
  }
  const merged = tileProjection({rows:1, columns:values.length, data:values}, 1);
  assert.equal(merged.cells[0].mean, row.mean);
}
const exactCells = [1e308, 1e-308, -1e308, Number.MIN_VALUE, -0];
const projection = tileProjection({rows:1, columns:exactCells.length, data:exactCells});
projection.cells.forEach((cell,i)=>assert(cell.mean === exactCells[i]));
// Dependent columns above the numerical-rank threshold must not be skipped
// merely because the matrix has many other unit-scale singular values.
const n=100, dependent=new Float64Array(n*n);
for(let i=0;i<n-2;i++) dependent[i*n+i]=1;
dependent[(n-2)*n+n-2]=dependent[(n-2)*n+n-1]=3e-14;
const spectrum=spectrumAnalysis({rows:n,columns:n,data:dependent});
assert.equal(spectrum.status,'assessed'); assert.equal(spectrum.numerical_rank,99);
assert.equal(spectrum.condition_status,'numerically_rank_deficient');
assert(Math.abs(spectrum.singular_values.at(-2)-Math.SQRT2*3e-14)<1e-28);
for(const tiny of [1e-160,1e-170,1e-300,Number.MIN_VALUE]) {
  const diagonal=spectrumAnalysis({rows:2,columns:2,data:[1,0,0,tiny]});
  assert.equal(diagonal.status,'assessed');assert.equal(diagonal.singular_values[1],tiny,'small singular value must not become zero through squaring');
  assert.equal(diagonal.relative_frobenius_error_by_rank[1],tiny,'small approximation error must not become zero through squaring');
  if(tiny>Number.MIN_VALUE) {
    const duplicate=spectrumAnalysis({rows:3,columns:3,data:[1,0,0,0,tiny,tiny,0,0,0]});
    assert.equal(duplicate.status,'assessed');assert.equal(duplicate.singular_values[2],0);
    assert(Math.abs(duplicate.singular_values[1]/tiny-Math.SQRT2)<1e-14);
  }
}
const extremeDiagonal=spectrumAnalysis({rows:2,columns:2,data:[1e308,0,0,1e-308]});
assert.equal(extremeDiagonal.method,'diagonal_closed_form_binary64');assert.deepEqual(extremeDiagonal.singular_values,[1e308,1e-308]);
assert.equal(spectrumAnalysis({rows:2,columns:2,data:[1e308,1e-308,0,1e-308]}).reason,'svd_normalization_underflow');
for(const tiny of [1e-17,1e-100,1e-300,Number.MIN_VALUE]) {
  const triangular=spectrumAnalysis({rows:2,columns:2,data:[1,tiny,0,tiny]});
  assert.equal(triangular.status,'assessed');
  assert(Math.abs(triangular.singular_values[1]/tiny-1)<1e-14,'unequally scaled correlated columns still require orthogonalization');
}
for(const tiny of [1e-40,1e-200,1e-300]) {
  const almostDependent=spectrumAnalysis({rows:2,columns:2,data:[1,1,0,tiny]});
  assert.equal(almostDependent.status,'assessed');
  assert(Math.abs(almostDependent.singular_values[1]/tiny-Math.SQRT1_2)<1e-14,'input-relative deflation must preserve a tiny independent direction');
}
const wideN=257,wide=new Float64Array(wideN*wideN);for(let i=0;i<wideN;i++)wide[i*wideN+i]=1;
const wideSpectrum=spectrumAnalysis({rows:wideN,columns:wideN,data:wide});
assert.equal(wideSpectrum.status,'assessed','a matrix within the work budget must not hit a hidden dimension cap');
assert.equal(wideSpectrum.numerical_rank,wideN);
const diagonalN=1000,largeDiagonal=new Float64Array(diagonalN*diagonalN);for(let i=0;i<diagonalN;i++)largeDiagonal[i*diagonalN+i]=1;
const closedForm=spectrumAnalysis({rows:diagonalN,columns:diagonalN,data:largeDiagonal});
assert.equal(closedForm.status,'assessed');assert.equal(closedForm.method,'diagonal_closed_form_binary64');
assert.equal(closedForm.numerical_rank,diagonalN);assert.equal(closedForm.work_count,largeDiagonal.length);
assert.equal(spectrumAnalysis({rows:2,columns:2,data:[1,1,1,-1]},2).status,'assessed','one-sweep orthogonal matrices must use actual work, not a cubic estimate');
for(const [value,expected] of [[0.49999999999999994,0],[0.5,1],[0.5000000000000001,1],[4503599627370497,4503599627370497],[Number.MAX_VALUE,Number.MAX_VALUE]]) {
  assert.equal(roundTiesAway(value),expected);assert.equal(roundTiesAway(-value),-expected);
}

// Coordinate oracle for every axis and both native storage orders.
let axisChecks=0;
for(const shape of [[2,3,4],[4,2,3],[2,4,4]]) for(const order of ['first_axis_fastest','last_axis_fastest']) {
  const length=shape.reduce((a,b)=>a*b,1), values=Array.from({length},(_,i)=>i%7===0?0:i%127);
  const coordinates=index=>{const result=Array(shape.length);for(const dim of order==='first_axis_fastest'?[0,1,2]:[2,1,0]){result[dim]=index%shape[dim];index=Math.floor(index/shape[dim]);}return result;};
  for(let axis=0;axis<shape.length;axis++) {
    const scales=Array.from({length:shape[axis]},(_,i)=>(i+1)/8),zeros=scales.map((_,i)=>i);
    const result=dequantize(values,{dtype:'INT8',shape,representation:'stored_scalar'},{scales,zero_points:zeros,axis,parameterization:'per_axis'},order);
    assert.equal(result.report.status,'assessed');
    result.values.forEach((v,i)=>assert.equal(v,(values[i]-zeros[coordinates(i)[axis]])*scales[coordinates(i)[axis]]));
    const view=matrixView(values,shape,axis,order);
    for(let c=0;c<shape[axis];c++)assert.deepEqual([...view.data.subarray(c*view.columns,(c+1)*view.columns)],values.filter((_,i)=>coordinates(i)[axis]===c));
    assert.equal(sparsityAnalysis(values,shape,order,axis).zero_count,String(values.filter(v=>v===0).length));
    axisChecks++;
  }
}
console.log(`Numerical precision: ${momentChecks} exact rational moment comparisons, 510 aggregate means, exact-cell preservation, dependent-column SVD rank, ${axisChecks} axis/affine/sparsity cases passed.`);
