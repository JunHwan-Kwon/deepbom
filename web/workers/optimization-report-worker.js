import { initializeNumericalWasm } from "../lib/numerical-wasm.js";
import {openOptimizationReport} from '../lib/optimization-report-access.js';
self.onmessage=async event=>{
  try{await initializeNumericalWasm();self.postMessage({ok:true,data:openOptimizationReport(event.data.text)});}
  catch(error){self.postMessage({ok:false,error:String(error.message||error)});}
};
