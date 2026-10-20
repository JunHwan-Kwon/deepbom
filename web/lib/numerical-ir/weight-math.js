import { numericalWasm } from "../numerical-wasm.js";
import { statisticsOf } from "./statistics.js";
import { exactCosine } from "./exact-moments.js";

export const notAssessed = reason => ({ status: "not_assessed", reason });
export const assessed = data => ({ status: "assessed", ...data });

// Host adapters preserve synchronous public returns. Arithmetic, layout and
// complete aggregation have one owner in Rust/WASM.
export function magnitude(values) { return numericalWasm().numerical_weight_magnitude(values); }
export function cosine(a, b) { return exactCosine(a, b); }
export function matrixView(values, shape, axis, order) {
  const data=numericalWasm().numerical_weight_matrix(values,shape,axis,order);
  const rows=shape.length?shape[axis]:1,columns=rows?data.length/rows:0;
  return {data,rows,columns};
}
export function channelStatistics(matrix) {
  const {data,rows,columns}=matrix;
  return assessed({count:rows, values_per_channel:columns, channels:Array.from({length:rows},(_,i)=>({index:i,statistics:statisticsOf(data.subarray(i*columns,(i+1)*columns))}))});
}
export function similarityAnalysis(matrix, limit, threshold) {
  return numericalWasm().numerical_weight_similarity(matrix.data,matrix.rows,matrix.columns,limit,threshold);
}

// Calculation and import consistency validation share the WASM owner.
export function spectrumAnalysis(matrix, maxWork=30_000_000) {
  return JSON.parse(numericalWasm().numerical_spectrum(matrix.data,matrix.rows,matrix.columns,maxWork));
}
export function spectrumSummary(sigma, scale, rows, columns) {
  return JSON.parse(numericalWasm().numerical_spectrum_summary(Float64Array.from(sigma),scale,rows,columns));
}

export function tileProjection(matrix, maximum=48) {
  return numericalWasm().numerical_weight_tiles(matrix.data,matrix.rows,matrix.columns,maximum);
}
export function kernelProjection(matrix,shape,axis,order,index=0) {
  if(index>=matrix.rows)return notAssessed("kernel_channel_outside_inventory");
  const otherAxes=shape.map((_,i)=>i).filter(i=>i!==axis);
  const fastest=order==="first_axis_fastest"?otherAxes[0]:otherAxes.at(-1);
  const width=fastest===undefined?matrix.columns:shape[fastest];
  if(!width)return notAssessed("empty_kernel_slice");
  return assessed({channel_index:index,channel_count:matrix.rows,horizontal_native_axis:fastest??null,vertical_basis:"remaining_axes_flattened_in_storage_order",image:tileProjection({data:matrix.data.subarray(index*matrix.columns,(index+1)*matrix.columns),rows:matrix.columns/width,columns:width},48)});
}
export function sparsityAnalysis(values, shape, order, axis, blockSize=16) {
  return numericalWasm().numerical_weight_sparsity(values,shape,order,axis,blockSize);
}
export function compareValues(baseline,candidate) {
  if(baseline.length!==candidate.length)return notAssessed("value_count_mismatch");
  const wasm=numericalWasm();
  return {metrics:wasm.numerical_weight_comparison(baseline,candidate),differences:wasm.numerical_weight_differences(baseline,candidate)};
}
