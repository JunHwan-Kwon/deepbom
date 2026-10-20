// Host batching only. Exact arithmetic and rounding are owned by Rust/WASM.
import { numericalWasm } from "../numerical-wasm.js";
const CHUNK = 1024;
export function exactCountRatio(numerator, denominator) { return numericalWasm().numerical_count_ratio(String(BigInt(numerator)), String(BigInt(denominator))) ?? null; }
function vectors(a,b,method) {
  if(a.length!==b.length)throw new Error("Vector products require equal value counts.");
  const kernel=new (numericalWasm().NumericalComparison)();
  try { for(let i=0;i<a.length;i+=CHUNK)kernel.add_chunk(Float64Array.from(a.slice(i,i+CHUNK)),Float64Array.from(b.slice(i,i+CHUNK)));return kernel[method]() ?? null; }
  finally {kernel.free();}
}
export function exactRegressionGain(a,b) {return vectors(a,b,"regression_gain");}
export function exactCosine(a,b) {return vectors(a,b,"cosine");}
export class ExactMoments {
  constructor({squares=true}={}) {this.kernel=new (numericalWasm().NumericalMoments)(squares);this.pending=new Float64Array(CHUNK);this.length=0;}
  add(value) {if(!Number.isFinite(value))throw new Error("Exact moments require finite values.");this.pending[this.length++]=value;if(this.length===CHUNK)this.flush();}
  flush(){if(this.length){this.kernel.add_chunk(this.pending.subarray(0,this.length));this.length=0;}}
  totals(){this.flush();const [sum,squares]=this.kernel.totals();return {sum:BigInt(sum),squares:BigInt(squares)};}
  mean(count){this.flush();return this.kernel.mean(count) ?? null;}
  dividedBy(value,factor=1){this.flush();return this.kernel.divided_by(value,String(BigInt(factor)));}
  coefficientOfVariation(count){this.flush();return this.kernel.coefficient_of_variation(count) ?? null;}
  finish(count){this.flush();return this.kernel.finish(count);}
  free(){this.kernel.free();this.length=0;}
}
export class ExactComparison {
  constructor(){this.kernel=new (numericalWasm().NumericalComparison)();this.left=new Float64Array(CHUNK);this.right=new Float64Array(CHUNK);this.length=0;this.count=0;}
  add(left,right){if(!Number.isFinite(left)||!Number.isFinite(right))throw new Error("Exact moments require finite values.");this.left[this.length]=left;this.right[this.length++]=right;this.count++;if(this.length===CHUNK)this.flush();}
  flush(){if(this.length){this.kernel.add_chunk(this.left.subarray(0,this.length),this.right.subarray(0,this.length));this.length=0;}}
  merge(other){this.flush();other.flush();if(other===this)this.kernel.duplicate();else this.kernel.merge(other.kernel);this.count+=other.count;return this;}
  finish(){this.flush();return this.kernel.finish();}
  free(){this.kernel.free();this.length=0;}
}
