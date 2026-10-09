import {openOptimizationReport} from '../lib/optimization-report-access.js';
self.onmessage=event=>{
  try{self.postMessage({ok:true,data:openOptimizationReport(event.data.text)});}
  catch(error){self.postMessage({ok:false,error:String(error.message||error)});}
};
