if (typeof document !== "undefined") {
const $=id=>document.getElementById(id);let state=null,offset=0,worker=null;const limit=25;
$('run').onclick=()=>{
  const request=$('request-file').files[0];if(!request){$('status').textContent='Select a request or result JSON first.';return;}
  worker?.terminate();state=null;$('result').hidden=true;$('status').textContent='Validating records and selected bytes…';$('run').disabled=true;
  worker=new Worker(new URL('./evidence-workflow-worker.js',import.meta.url),{type:'module'});
  worker.onerror=e=>{worker.terminate();$('run').disabled=false;$('status').textContent=e.message||'Worker failed';};
  worker.onmessage=({data})=>{worker.terminate();$('run').disabled=false;if(data.error){$('status').textContent=data.error;return;}state=data;offset=0;$('result').hidden=false;$('status').textContent=data.acquisition==='saved_result'?data.view.validation_scope:`Checked ${data.selected_files} selected files in this browser. Missing references remain unresolved. External measurements are supplied records.`;$('verdict').textContent=data.view.status;$('counts').textContent=data.view.counts?`${data.view.counts.total} checks · ${data.view.counts.pass} met · ${data.view.counts.fail} not met · ${data.view.counts.not_assessed} not assessed`:`${data.view.rows.length} evidence checks`;$('identity').textContent=`Before: ${data.view.before?.sha256??'not applicable'}\nAfter: ${data.view.after?.sha256??'not applicable'}\nResult: ${data.result.result_sha256}`;$('record').textContent=JSON.stringify(data.result.document,null,2);draw();};
  worker.postMessage({request,selected:[...$('evidence-files').files]});
};
function draw(){if(!state)return;const term=$('search').value.toLowerCase();const rows=state.view.rows.filter(r=>JSON.stringify(r).toLowerCase().includes(term));const page=rows.slice(offset,offset+limit);$('rows').replaceChildren();$('detail').textContent='Select a check to inspect its source references and result.';for(const r of page){const b=document.createElement('button');b.type='button';b.textContent=`${r.population??r.id??r.ref?.kind??'Evidence'} · ${r.metric??''} · ${r.status}`;b.onclick=()=>{$('detail').textContent=JSON.stringify(r,null,2);};$('rows').append(b);}$('page').textContent=`${page.length?offset+1:0}–${offset+page.length} of ${rows.length} matching (${state.view.rows.length} total)`;$('prev').disabled=offset===0;$('next').disabled=offset+limit>=rows.length;}
$('search').oninput=()=>{offset=0;draw();};$('prev').onclick=()=>{offset=Math.max(0,offset-limit);draw();};$('next').onclick=()=>{offset+=limit;draw();};
function save(data,type,name){const u=URL.createObjectURL(new Blob([data],{type}));const a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),30000);}
$('save-json').onclick=()=>state&&save(JSON.stringify(state.result,null,2)+'\n','application/json','evidence-result.json');$('save-html').onclick=()=>state&&save(state.html,'text/html','evidence-report.html');
$('print').onclick=()=>{if(!state)return;const w=window.open('','_blank');if(!w){$('status').textContent='Allow the report window or download HTML and print it.';return;}w.opener=null;w.document.write(state.html);w.document.close();w.focus();w.print();};

}
