import { buildActivationIr, buildNumericalEvidenceBundle } from "./activation-ir.js";
import { parseStrictJson } from "./strict-json.js";

export function installNumericalPanel(host, getContext) {
  if (!host || host.querySelector("[data-numerical-panel]")) return;
  const panel=document.createElement("details");panel.dataset.numericalPanel="";panel.className="model-summary-panel";
  panel.innerHTML=`<summary>Optional weight and activation evidence</summary><p>Inspect numeric distributions locally. Weight analysis reads stored values without running the model. Activation import requires a recorded execution capture.</p><div class="graph-map-toolbar"><button type="button" data-action="weights">Analyze weights</button><button type="button" data-action="cancel" disabled>Cancel</button><button type="button" data-action="model">Save Model IR</button><button type="button" data-action="import">Import activation capture</button><input type="file" accept="application/json,.json" hidden><button type="button" data-action="save" disabled>Save numerical evidence</button></div><p role="status"></p><label>Find a tensor, storage ID or operation <input type="search"></label><div data-results></div>`;
  const dossier=host.querySelector("#artifactDossier");if(dossier)dossier.after(panel);else host.append(panel);const status=panel.querySelector('[role="status"]'),results=panel.querySelector('[data-results]'),filter=panel.querySelector('[type="search"]'),fileInput=panel.querySelector('[type="file"]');
  let worker=null, weight=null,activation=null,model=null;
  const button=action=>panel.querySelector(`[data-action="${action}"]`);
  function resetContext() { const current=getContext();if(!current?.model)throw new Error("Analyze a model first.");if(model?.model_ir_sha256!==current.model.model_ir_sha256){weight=null;activation=null;results.replaceChildren();button("save").disabled=true;}model=current.model;return current; }
  function save(name,value) { const blob=new Blob([JSON.stringify(value,null,2)+"\n"],{type:"application/json"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60_000); }
  function render() {
    results.replaceChildren();const rows=[...(weight?.tensors||[]),...(activation?.tensors||[])],query=filter.value.toLowerCase();
    for(const row of rows.filter(r=>JSON.stringify(r).toLowerCase().includes(query)).slice(0,200)) {
      const item=document.createElement("details"),summary=document.createElement("summary"),text=document.createElement("pre");
      summary.textContent=`${row.name||row.native_locator||row.value_ref} · ${row.status||"captured"} · ${row.statistics?.value_count??"unavailable"} values`;
      const bindings=model.weight_bindings.bindings.filter(b=>row.binding_refs?.includes(b.id)).map(b=>({operation_ref:b.operation_ref,port_ref:b.port_ref,value_ref:b.value_ref}));
      text.textContent=JSON.stringify({...row,model_bindings:bindings},null,2);item.append(summary);
      if(row.statistics?.histogram){const canvas=document.createElement("canvas");canvas.width=600;canvas.height=140;canvas.style.maxWidth="100%";canvas.setAttribute("role","img");canvas.setAttribute("aria-label","Logarithmic-bin histogram; exact counts and bin edges appear below.");const ctx=canvas.getContext("2d"),counts=row.statistics.histogram.counts.map(Number),max=Math.max(...counts,1);ctx.fillStyle="#1e8876";counts.forEach((n,i)=>{const h=n/max*120;ctx.fillRect(i*600/counts.length,130-h,Math.max(1,600/counts.length-1),h);});item.append(canvas);}
      item.append(text);results.append(item);
    }
    const note=document.createElement("p");note.textContent="Showing up to 200 matching tensors. Saved JSON includes the complete inventory. Histograms use fixed logarithmic value bins; bar height is linear count.";results.append(note);
    button("save").disabled=!weight&&!activation;
  }
  filter.addEventListener("input",render);
  button("cancel").onclick=()=>{worker?.terminate();worker=null;button("weights").disabled=false;button("cancel").disabled=true;status.textContent="Weight analysis cancelled.";};
  button("weights").onclick=()=>{
    try {const context=resetContext(),digest=model.model_ir_sha256;if(!context.source)throw new Error("Select one serialized artifact file for numeric inspection.");button("weights").disabled=true;button("cancel").disabled=false;status.textContent="Checking artifact identity…";
      worker=new Worker(new URL("../workers/numerical-ir-worker.js",import.meta.url),{type:"module"});
      const finish=()=>{worker?.terminate();worker=null;button("weights").disabled=false;button("cancel").disabled=true;};
      worker.onerror=event=>{status.textContent=event.message||"Numeric analysis worker failed.";finish();};
      worker.onmessage=event=>{if(event.data.progress){const p=event.data.progress;status.textContent=p.phase==="weights"?`Weights: ${p.completed}/${p.total}`:"Checking artifact identity…";return;}finish();if(getContext()?.model?.model_ir_sha256!==digest){status.textContent="Model changed; discarded the previous result.";return;}if(event.data.error){status.textContent=event.data.error;return;}weight=event.data.result;status.textContent=`${weight.coverage.assessed_count}/${weight.coverage.inventory_count} payloads assessed. ${weight.coverage.not_assessed_count} not assessed. Stored integer codes are not automatically dequantized.`;render();};
      worker.postMessage({model,analysis:context.analysis,source:context.source});
    }catch(error){status.textContent=error.message;button("weights").disabled=false;button("cancel").disabled=true;}
  };
  button("model").onclick=()=>{try{resetContext();save("deepbom-model-ir.json",model);}catch(error){status.textContent=error.message;}};
  button("import").onclick=()=>fileInput.click();
  fileInput.onchange=async()=>{try{resetContext();const selected=fileInput.files[0];if(!selected)return;if(selected.size>16*1024*1024)throw new Error("Activation capture exceeds 16 MiB.");activation=buildActivationIr(model,parseStrictJson(await selected.text(),"activation capture"));status.textContent=`${activation.coverage.captured_count}/${activation.coverage.requested_count} requested values captured; ${activation.coverage.missing_count} missing. Imported evidence is not independently attested.`;render();}catch(error){status.textContent=error.message;}finally{fileInput.value="";}};
  button("save").onclick=()=>{try{resetContext();save("deepbom-numerical-evidence.json",{bundle:buildNumericalEvidenceBundle(model,weight,activation),model_ir:model,weight_ir:weight,activation_ir:activation});}catch(error){status.textContent=error.message;}};
}
