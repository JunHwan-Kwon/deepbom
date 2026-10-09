import {queryOptimizationReport,REPORT_MAX_BYTES} from './optimization-report-access.js';
import {optimizationReportHtml} from './optimization-report-html.js';
export function initializeOptimizationReportPage() {
const $=id=>document.getElementById(id),node=(tag,text)=>{const el=document.createElement(tag);if(text!==undefined)el.textContent=String(text);return el;};
const json=x=>JSON.stringify(x,null,2);
let data=null,offset=0,treePage=0,selected=null,worker=null,loadId=0;
function table(heads,rows){const t=node('table'),h=node('tr');for(const text of heads)h.append(node('th',text));const head=node('thead');head.append(h);t.append(head);const body=node('tbody');for(const row of rows){const tr=node('tr');for(const text of row){const td=node('td');td.append(node('pre',text));tr.append(td);}body.append(tr);}t.append(body);return t;}
function download(bytes,name,mime){const url=URL.createObjectURL(new Blob([bytes],{type:mime})),a=node('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
function detail(entry){selected=entry.id;const root=$('report-detail');root.replaceChildren(node('h2',entry.id),node('p',[entry.status,...entry.tags].join(' · ')));
  for(const rule of data.diff.rules.filter(r=>entry.rules.includes(r.index))){const box=node('section');box.className='reason';box.append(node('strong',`${rule.index}. ${rule.title}`),node('p',rule.reason),node('p',rule.description),node('p',`${rule.state} · ${rule.tradeoff}`),node('small',`${rule.rule}@${rule.rule_version}`));root.append(box);}
  if(entry.fields.length)root.append(table(['Changed field','Before (−)','After (+)'],entry.fields.map(f=>[f.path||'(whole subject)',f.before_present?json(f.before):'Absent',f.after_present?json(f.after):'Absent'])));
  else root.append(node('p','No recorded difference for this subject.'));
  for(const [label,record] of [['Before',entry.before],['After',entry.after]]){const d=node('details');d.append(node('summary',`${label}: complete recorded evidence`),node('pre',record===null?'Absent':json(record)));root.append(d);}
  for(const el of $('report-entries').children)el.setAttribute('aria-current',String(el.dataset.subject===selected));
  root.scrollTop=0;
}
function list(){const q=queryOptimizationReport(data,{section:$('report-filter').value,search:$('report-search').value,offset,limit:50});const root=$('report-entries');root.replaceChildren();
  for(const e of q.rows){const b=node('button',e.id);b.dataset.subject=e.id;b.append(node('small',[e.status,...e.tags].join(' · ')));b.onclick=()=>detail(e);root.append(b);}
  $('report-page').textContent=`${q.returned?offset+1:0}–${offset+q.returned} of ${q.matched} entries`;$('report-prev').disabled=offset===0;$('report-next').disabled=q.next_offset===null;
  if(q.rows.length)detail(q.rows.find(e=>e.id===selected)??q.rows.find(e=>e.rules.some(i=>data.diff.rules.find(r=>r.index===i)?.subject===e.id))??q.rows[0]);else $('report-detail').replaceChildren(node('p','No matching subjects.'));
}
function trees(){const before=data.report.baseline,after=data.report.candidate,pageSize=35,max=Math.max(before.nodes.length,after.nodes.length),pages=Math.max(1,Math.ceil(max/pageSize));
  $('tree-page').textContent=`${treePage+1} / ${pages} · ${before.nodes.length} Before / ${after.nodes.length} After modules`;$('tree-prev').disabled=treePage===0;$('tree-next').disabled=treePage+1>=pages;
  for(const [id,side] of [['before-tree',before],['after-tree',after]]){const root=$(id);root.replaceChildren();const parents=new Map();for(const r of side.relationships)if(r.kind==='contains'){const p=parents.get(r.to)||[];p.push(r.from);parents.set(r.to,p);}
    for(const n of side.nodes.slice(treePage*pageSize,(treePage+1)*pageSize)){const e=data.diff.entries.find(e=>e.status!=='context'&&e.id===n.id),b=node('button',n.id);b.className=`tree-row ${e?.status||''}`;let cur=n.id,depth=0,seen=new Set([cur]);while(parents.get(cur)?.length===1){cur=parents.get(cur)[0];if(seen.has(cur))break;seen.add(cur);depth++;}b.style.paddingInlineStart=`${12+Math.min(depth,6)*12}px`;b.append(node('small',n.kind));if(parents.has(n.id))b.append(node('small',`Contained in: ${parents.get(n.id).join(', ')}`));b.onclick=()=>{if(e)detail(e);$('report-detail').scrollIntoView({block:'nearest'});};root.append(b);}}
}
function ready(){const {report:r,diff:d}=data;$('report-content').hidden=false;$('report-summary').replaceChildren(node('h2','Before / After evidence'),node('p',`Report SHA-256: ${r.report_sha256}`),table(['Metric','Before','After','Delta'],Object.keys(r.static_deltas).map(k=>[k,r.comparison.totals.baseline[k],r.comparison.totals.candidate[k],r.static_deltas[k]]).concat([['Mapped MAC subtotal',r.baseline.cost.macs??'Not assessed',r.candidate.cost.macs??'Not assessed',r.cost_delta??'Not assessed']])));
  $('report-summary').append(node('p',`${d.counts.changed} changed · ${d.counts.added} added · ${d.counts.removed} removed · ${d.counts.unchanged} unchanged modules`),node('p',`MAC coverage: Before ${r.baseline.cost.unmapped_invocations} / After ${r.candidate.cost.unmapped_invocations} unmapped calls.`));
  $('report-boundary').textContent=d.boundary;offset=0;treePage=0;selected=null;$('report-search').value='';$('report-filter').value='changes';list();trees();
}
$('report-file').onchange=async()=>{const id=++loadId,file=$('report-file').files[0];worker?.terminate();data=null;$('report-content').hidden=true;const status=$('report-status');status.dataset.error='false';if(!file)return;status.textContent='Checking report schema, identities and common diff…';
  try{if(file.size>REPORT_MAX_BYTES)throw Error('Report exceeds the 16 MiB import limit');const text=await file.text();if(id!==loadId)return;
    const w=worker=new Worker(new URL('../workers/optimization-report-worker.js',import.meta.url),{type:'module'});
    const result=await new Promise((resolve,reject)=>{const t=setTimeout(()=>{w.terminate();reject(Error('Report validation timed out'));},60000);w.onmessage=e=>{clearTimeout(t);resolve(e.data);};w.onerror=()=>{clearTimeout(t);reject(Error('Report worker could not complete validation'));};w.postMessage({text});});w.terminate();if(id!==loadId)return;
    if(!result.ok)throw Error(result.error);data=result.data;ready();status.textContent='Report ready. Schema, report digest and internal bindings checked. Original model bytes and producer authenticity are not verified.';
  }catch(error){if(id!==loadId)return;status.dataset.error='true';status.textContent=error.message;data=null;$('report-content').hidden=true;}
};
for(const id of ['report-search','report-filter'])$(id).addEventListener(id==='report-search'?'input':'change',()=>{offset=0;list();});
$('report-prev').onclick=()=>{offset=Math.max(0,offset-50);list();};$('report-next').onclick=()=>{offset+=50;list();};$('tree-prev').onclick=()=>{treePage--;trees();};$('tree-next').onclick=()=>{treePage++;trees();};
$('export-json').onclick=()=>download(json(data.report)+'\n','report.json','application/json');$('export-diff').onclick=()=>download(json(data.diff)+'\n','report.diff.json','application/json');$('export-html').onclick=()=>download(optimizationReportHtml(data),'report.html','text/html');
$('export-pdf').onclick=()=>{const tab=window.open('','_blank');if(!tab){$('report-status').textContent='Allow the print window to open, or download HTML and print it locally.';return;}tab.opener=null;tab.document.write(optimizationReportHtml(data));tab.document.close();tab.focus();tab.print();};

}
if(typeof document!=="undefined" && document.getElementById("report-file"))initializeOptimizationReportPage();
