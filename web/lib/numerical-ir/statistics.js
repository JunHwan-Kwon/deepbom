import { canonicalJson } from "../report-utils.js";
import { portableContract } from "../portable-contracts.js";
import { parseNumeric } from "./common.js";
import { numericalWasm } from "../numerical-wasm.js";

// Fixed log2 bins permit exact-count streaming and merging without a second
// payload read. The bin definition is identical for every format and tensor.
const EXPONENTS = [-1074, -32, -24, -16, -12, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024];
export const HISTOGRAM_EDGES = Object.freeze([-Number.MAX_VALUE, ...EXPONENTS.slice(0, -1).map(e => -(2 ** e)).reverse(), 0, ...EXPONENTS.slice(0, -1).map(e => 2 ** e), Number.MAX_VALUE]);
export class TensorStatistics {
  constructor() {this.kernel=new (numericalWasm().NumericalStatistics)();this.pending=new Float64Array(1024);this.length=0;this.count=0n;}
  add(raw) {
    const value=parseNumeric(raw);this.count++;
    if(typeof value === "bigint"){this.flush();this.kernel.add_integer(value.toString());return;}
    this.pending[this.length++]=value;if(this.length===this.pending.length)this.flush();
  }
  flush(){if(this.length){this.kernel.add_chunk(this.pending.subarray(0,this.length));this.length=0;}}
  finish(){this.flush();return this.kernel.finish();}
  free(){this.kernel.free();this.length=0;}
}
export function statisticsOf(values) {const s=new TensorStatistics();try{for(const v of values)s.add(v);return s.finish();}finally{s.free();}}
export function validateStatistics(s) {
  canonicalJson(s);
  portableContract("semantic_validation", {operation:"statistics",value:s});
  return s;
}
