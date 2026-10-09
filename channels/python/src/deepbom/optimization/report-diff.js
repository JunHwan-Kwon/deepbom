(() => {
  'use strict';
  const data=JSON.parse(document.getElementById('model-diff-data').textContent);
  const $=id=>document.getElementById(id), entries=data.entries;
  const ruleMap=new Map(data.rules.map(r=>[r.index,r]));
  const stringify=value=>JSON.stringify(value,null,2);
  const searchable=entries.map(e=>[e.id,e.before?.module?.kind,e.after?.module?.kind,...e.rules.map(id=>{const r=ruleMap.get(id);return `${r.rule} ${r.title}`;})].join(' ').toLowerCase());
  const mark=e=>({added:'+',removed:'−',changed:'~',context:'~'})[e.status]||(e.tags.length?'~':'=');
  const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;return n;};
  let visible=[], selected=null, page=0;const pageSize=50;
  function renderDetail(){
    const root=$('diff-detail');root.replaceChildren();
    if(selected===null){root.append(node('p','No matching changes. Choose All modules to inspect unchanged evidence.'));return;}
    const e=entries[selected];root.append(node('h3',`${mark(e)} ${e.id}`),node('p',e.tags.length?e.tags.join(' · '):'Unchanged recorded evidence','diff-tags'));
    if(e.status!=='context')root.append(node('p',`Before: ${e.before?.module?.kind??'Absent'} → After: ${e.after?.module?.kind??'Absent'}`));
    for(const id of e.rules){const r=ruleMap.get(id),reason=node('details',null,'diff-reason');reason.append(node('summary',`${r.rule}@${r.rule_version} · Why this change?`),node('p',r.reason));if(r.description)reason.append(node('p',r.description));if(r.tradeoff)reason.append(node('p',r.tradeoff));root.append(reason);}
    if($('diff-mode').value==='split'){
      const split=node('div',null,'diff-split');
      for(const [title,value] of [['Before',e.before],['After',e.after]]){const panel=node('section');panel.append(node('strong',title),node('pre',value===null?'Absent':stringify(value)));split.append(panel);}root.append(split);
    }else{
      const code=node('div',null,'diff-code');
      code.setAttribute('aria-label','Unified evidence diff');
      for(const line of (e.patch||'No recorded difference.').trimEnd().split('\n')){const cls=line.startsWith('@@')?'hunk':/^---|^\+\+\+/.test(line)?'header':line.startsWith('+')?'add':line.startsWith('-')?'remove':'context';code.append(node('span',line,`diff-line ${cls}`));}root.append(code);
    }
    root.scrollTop=0;
  }
  function renderList(){
    const list=$('diff-list');list.replaceChildren();
    for(const index of visible.slice(page*pageSize,(page+1)*pageSize)){
      const e=entries[index],button=node('button',`${mark(e)} ${e.id}`,'diff-subject');button.type='button';button.dataset.entry=String(index);button.setAttribute('aria-current',String(index===selected));button.append(node('small',[e.status,...e.tags.filter(t=>t!=='structure')].join(' · ')));
      button.addEventListener('click',()=>{selected=index;renderList();renderDetail();});list.append(button);
    }
    const pages=Math.max(1,Math.ceil(visible.length/pageSize));
    $('diff-page-label').textContent=`${page+1} / ${pages}`;
    $('diff-page-prev').disabled=page===0;$('diff-page-next').disabled=page+1>=pages;
    const changed=visible.filter(i=>entries[i].tags.length);
    $('diff-prev').disabled=$('diff-next').disabled=changed.length===0;
    $('diff-status').textContent=`${visible.length} matching / ${entries.length} entries; ${data.changed_entries} entries with recorded changes. ${entries.filter(e=>!e.tags.length).length} unchanged entries are available under All modules.`;
  }
  function filter(){
    const query=$('diff-search').value.trim().toLowerCase(),mode=$('diff-filter').value;
    visible=entries.flatMap((e,i)=>{
      const matches=mode==='all'||(mode==='changes'?e.tags.length:mode==='state'||mode==='connections'?e.tags.includes(mode):e.status===mode);
      return matches&&searchable[i].includes(query)?[i]:[];
    });
    page=0;if(!visible.includes(selected))selected=visible.find(i=>entries[i].status==='changed'&&entries[i].rules.some(r=>ruleMap.get(r).subject===entries[i].id))??visible.find(i=>entries[i].status==='changed')??visible[0]??null;
    if(selected!==null)page=Math.floor(visible.indexOf(selected)/pageSize);
    renderList();renderDetail();
  }
  function jump(direction){
    const candidates=visible.filter(i=>entries[i].tags.length);if(!candidates.length)return;
    const next=direction>0?candidates.find(i=>i>selected):[...candidates].reverse().find(i=>i<selected);
    selected=next??(direction>0?candidates[0]:candidates.at(-1));page=Math.floor(visible.indexOf(selected)/pageSize);renderList();renderDetail();$('diff-list').querySelector('[aria-current=true]')?.scrollIntoView({block:'nearest'});
  }
  $('diff-search').addEventListener('input',filter);$('diff-filter').addEventListener('change',filter);$('diff-mode').addEventListener('change',renderDetail);
  $('diff-prev').addEventListener('click',()=>jump(-1));$('diff-next').addEventListener('click',()=>jump(1));
  for(const [id,delta] of [['diff-page-prev',-1],['diff-page-next',1]])$(id).addEventListener('click',()=>{page+=delta;selected=visible[page*pageSize]??null;renderList();renderDetail();$('diff-list').scrollTop=0;});
  $('diff-download').addEventListener('click',()=>{
    const preface=`# DEEPBOM read-only evidence diff (not an executable model patch)\n# Report: ${data.report_sha256}\n# Before snapshot: ${data.before_sha256}\n# After snapshot: ${data.after_sha256}\n# Native-ID alignment; module containment is not an execution graph.\n\n`;
    const bytes=preface+entries.filter(e=>e.tags.length).map(e=>e.patch).join('\n');
    const url=URL.createObjectURL(new Blob([bytes],{type:'text/plain;charset=utf-8'})),link=node('a');link.href=url;link.download='model-evidence.diff';link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);
  });
  // The static appendices remain complete in browser printing, then return to
  // the user's previous expanded/collapsed state afterward.
  let printState=[];window.addEventListener('beforeprint',()=>{printState=[...document.querySelectorAll('.complete-evidence')].map(el=>[el,el.open]);for(const [el] of printState)el.open=true;});window.addEventListener('afterprint',()=>{for(const [el,open] of printState)el.open=open;});
  filter();
})();
