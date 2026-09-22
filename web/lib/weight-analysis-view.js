import { escapeXml as esc, numberLabel as num, countLabel, histogramWindow, intervalLabel } from "./weight-visuals.js";

export const WEIGHT_VIEWS=Object.freeze({distribution:"Distribution",channels:"Channels",kernel:"Kernel slice",similarity:"Similarity",spectrum:"Singular spectrum",sparsity:"Sparsity",quantization:"Quantization",comparison:"Original / candidate"});
const phrase=x=>String(x||"").replaceAll("_"," ");
const note=x=>'<p class="weight-chart-note">'+esc(x)+'</p>';
const unavailable=result=>'<div class="weight-empty"><h4>Not assessed</h4><p>'+esc(phrase(result?.reason || "run_weight_analysis_first"))+'</p></div>';
function svg(title,p,body,description="") {return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 340" role="img" aria-label="'+esc(title)+'"><title>'+esc(title)+'</title><desc>'+esc(description)+'</desc><rect width="760" height="340" fill="'+p.surface+'"/><g font-family="ui-monospace, monospace" font-size="11" fill="'+p.ink+'"><text x="18" y="24" font-size="14">'+esc(title)+'</text>'+body+'</g></svg>';}
function tileSvg(title,projection,p,{zero=false}={}) {
  if(projection?.status!=="assessed")return null;
  const {rows,columns,cells}=projection,w=688/columns,h=264/rows;
  const max=Math.max(...cells.map(c=>Math.abs(c.mean)),Number.MIN_VALUE);
  let body='';
  cells.forEach((cell,i)=>{const value=zero?cell.zero_count/cell.count:cell.mean/max,label=zero?countLabel(cell.zero_count)+' / '+countLabel(cell.count)+' zeros':'Mean '+String(cell.mean)+'; min '+String(cell.minimum)+'; max '+String(cell.maximum)+'; '+cell.count+' values';body+='<rect x="'+(52+(i%columns)*w)+'" y="'+(45+Math.floor(i/columns)*h)+'" width="'+Math.max(0.2,w-0.4)+'" height="'+Math.max(0.2,h-0.4)+'" fill="'+(value<0?p.blue:p.accent)+'" fill-opacity="'+(0.05+0.95*Math.abs(value))+'"><title>'+esc(label)+'</title></rect>';});
  body+='<text x="52" y="329" fill="'+p.muted+'">'+esc(projection.source_rows+' × '+projection.source_columns+' values; '+phrase(projection.method))+'</text>';
  return svg(title,p,body,'Each tile contains all values in its area; mean color can conceal opposing signs. Hover exposes exact counts and extrema.');
}
function lineSvg(title,series,p,{log=false,xlabel="Index",ylabel="Value"}={}) {
  const defined=series.flatMap(s=>s.values.filter(x=>Number.isFinite(x)&&(!log||x>0)).map(x=>log?Math.log10(x):x));
  if(!defined.length)return null;
  const low=Math.min(0,...defined),high=Math.max(...defined),range=high-low || 1,x=76,y=46,w=654,h=238;
  let body='';for(let tick=0;tick<=4;tick++){const fraction=tick/4;body+='<path d="M'+x+' '+(y+h*(1-fraction))+'H'+(x+w)+'" stroke="'+p.line+'"/><text x="65" y="'+(y+h*(1-fraction)+4)+'" text-anchor="end">'+esc(num(log?10**(low+fraction*range):low+fraction*range))+'</text>';}
  for(const [seriesIndex,s] of series.entries()) {
    let path='',connected=false;s.values.forEach((v,i)=>{if(v===null||!Number.isFinite(v)||log&&v<=0){connected=false;return;}const px=x+(s.values.length===1?0.5:i/(s.values.length-1))*w,py=y+h*(1-((log?Math.log10(v):v)-low)/range);path+=(connected?'L':'M')+px+' '+py+' ';connected=true;body+='<circle cx="'+px+'" cy="'+py+'" r="2.5" fill="'+(seriesIndex?p.blue:p.accent)+'"><title>'+esc(s.name+' ['+i+']: '+String(v))+'</title></circle>';});
    body+='<path d="'+path+'" fill="none" stroke="'+(seriesIndex?p.blue:p.accent)+'" stroke-width="1.5"/>';
    body+='<text x="'+(x+seriesIndex*290)+'" y="311" fill="'+(seriesIndex?p.blue:p.accent)+'">'+esc(s.name)+'</text>';
  }
  body+='<text x="'+x+'" y="334" fill="'+p.muted+'">'+esc(xlabel+' · '+ylabel+(log?' (log10)':''))+'</text>';
  return svg(title,p,body);
}
function distributionComparison(pair,p) {
  const rows=[{statistics:pair.distributions.baseline},{statistics:pair.distributions.candidate}],window=histogramWindow(rows);
  if(!window)return null;
  const a=rows[0].statistics.histogram,b=rows[1].statistics.histogram,n=window.last-window.first+1;
  const maximum=Math.max(1,...a.counts.slice(window.first,window.last+1).map(Number),...b.counts.slice(window.first,window.last+1).map(Number));
  const w=654/n;let body='';
  for(let i=window.first;i<=window.last;i++)for(const [j,h] of [a,b].entries()) {
    const count=Number(h.counts[i]),height=count/maximum*235;
    body+='<rect x="'+(76+(i-window.first)*w+j*w/2)+'" y="'+(284-height)+'" width="'+Math.max(0.2,w/2-0.3)+'" height="'+height+'" fill="'+(j?p.blue:p.accent)+'"><title>'+esc((j?'Candidate':'Original')+' '+intervalLabel(h,i)+': '+h.counts[i])+'</title></rect>';
  }
  body+='<text x="76" y="311" fill="'+p.accent+'">Original</text><text x="366" y="311" fill="'+p.blue+'">Candidate</text><text x="76" y="332" fill="'+p.muted+'">Shared fixed value intervals; unequal intervals drawn at equal width. Count, not density.</text>';
  return svg('Original and candidate distributions',p,body);
}
const metrics=rows=>'<dl class="weight-stat-grid">'+rows.map(([k,v])=>'<div><dt>'+esc(k)+'</dt><dd>'+esc(v)+'</dd></div>').join('')+'</dl>';
export function renderWeightAnalysisView(view,row,comparison,p) {
  if(!row)return {html:unavailable(),svg:null};
  if(row.status!=="assessed")return {html:unavailable(row),svg:null};
  let figure=null,html='',result=row[view];
  const basis=note('Analysis representation: '+phrase(row.representation)+'. Axis '+row.axes.channel_axis+' · '+phrase(row.axes.channel_meaning)+'. '+phrase(row.axes.storage_order)+'.');
  if(view==="channels" && result.status==="assessed") {
    const channels=result.channels,shown=channels.slice(0,24),keys=["mean","population_stddev","rms","l2_norm","zero_count"],labels=["Mean","Std dev","RMS","L2","Zeros"];
    const maxima=keys.map(k=>Math.max(...channels.map(c=>Math.abs(Number.isFinite(Number(c.statistics[k])) ? Math.abs(Number(c.statistics[k])) : 0)),Number.MIN_VALUE));let body='';
    labels.forEach((label,j)=>body+='<text x="'+(126+j*125)+'" y="45">'+label+'</text>');
    shown.forEach((channel,i)=>{body+='<text x="18" y="'+(64+i*10.7)+'">#'+channel.index+'</text>';keys.forEach((key,j)=>{const raw=channel.statistics[key],value=raw===null?null:Number(raw),color=value===null?p.line:value<0?p.blue:p.accent;body+='<rect x="'+(126+j*125)+'" y="'+(55+i*10.7)+'" width="115" height="9" fill="'+color+'" fill-opacity="'+(value===null?1:0.07+0.93*Math.abs(value)/maxima[j])+'"><title>'+esc('Channel '+channel.index+' '+labels[j]+': '+(value===null?'Not assessed':channel.statistics[key]))+'</title></rect>';});});
    figure=svg('Channel statistics · each column normalized independently',p,body);
    html=note('Showing '+shown.length+' / '+channels.length+' channels in the map. The table includes every assessed channel; no value sampling.')+'<details><summary>Complete channel statistics</summary><div class="weight-table-scroll"><table><thead><tr><th>Channel</th>'+labels.map(x=>'<th>'+x+'</th>').join('')+'<th>Min</th><th>Max</th><th>Variance</th><th>Zero fraction</th></tr></thead><tbody>'+channels.map(c=>'<tr><td>'+c.index+'</td>'+[...keys,'minimum','maximum'].map(k=>'<td title="'+esc(c.statistics[k])+'">'+esc(k==='zero_count'?countLabel(c.statistics[k]):num(c.statistics[k]))+'</td>').join('')+'<td>'+esc(num(c.statistics.population_stddev===null||!Number.isFinite(c.statistics.population_stddev**2)?null:c.statistics.population_stddev**2))+'</td><td>'+esc(num(Number(c.statistics.zero_count)/Number(c.statistics.finite_count)))+'</td></tr>').join('')+'</tbody></table></div></details>';
  } else if(view==="kernel" && result.status==="assessed") {
    figure=tileSvg('Channel '+result.channel_index+' · native tensor slice',result.image,p);
    html='<label class="weight-kernel-choice">Channel <input type="number" data-kernel-channel min="0" max="'+(result.channel_count-1)+'" value="'+result.channel_index+'"><button type="button" data-action="kernel-apply">Analyze this channel</button></label>'+note('Horizontal native axis: '+result.horizontal_native_axis+'. Other axes are stacked vertically. Aggregated cells report their mean and extrema; this is not an activation image.');
  } else if(view==="similarity" && result.status==="assessed") {
    const projection={status:"assessed",rows:result.size,columns:result.size,source_rows:result.size,source_columns:result.size,method:"complete_signed_cosine",cells:result.values.map(v=>({mean:v??0,minimum:v,maximum:v,count:1,zero_count:0}))};
    figure=tileSvg('Signed cosine similarity · −1 to +1',projection,p);
    // A zero-norm comparison is undefined, not a measured zero.
    figure=figure.replace(/<rect x="([^"]+)" y="([^"]+)" width="([^"]+)" height="([^"]+)" fill="[^"]+" fill-opacity="[^"]+"><title>Mean 0; min null; max null; 1 values<\/title><\/rect>/g,'<rect x="$1" y="$2" width="$3" height="$4" fill="'+p.line+'"><title>Undefined: zero-norm channel</title></rect>');
    html=metrics([["Threshold",String(result.threshold)],["Matching pairs",String(result.pairs.length)],["Groups",String(result.clusters.length)]])+note('Signed cosine; negative alignment is distinct. Gray cells have zero norm. Groups connect threshold edges; group members need not all exceed the threshold. Similarity does not establish redundant computation.');
  } else if(view==="spectrum" && result.status==="assessed") {
    figure=lineSvg('Singular value spectrum',[{name:'Singular values',values:result.singular_values}],p,{log:true,xlabel:'Singular-value index',ylabel:'Magnitude'});
    const energy=lineSvg('Energy retained and low-rank residual',[{name:'Cumulative energy',values:result.cumulative_energy},{name:'Relative Frobenius residual after rank k+1',values:result.relative_frobenius_error_by_rank.slice(1)}],p,{ylabel:'Fraction'});
    html=metrics([["Numerical rank",String(result.numerical_rank)],["Condition number",result.condition_number===null?phrase(result.condition_status):num(result.condition_number)],["Rank tolerance",num(result.rank_tolerance)],["Energy residual",num(result.energy_relative_residual)]])+(energy||'')+note('Full singular spectrum of the declared axis unfolding, computed with scaled Jacobi rotations. Rank depends on the reported tolerance; this is not the full convolution operator. Zero singular values are omitted from the logarithmic plot.');
  } else if(view==="sparsity" && result.status==="assessed") {
    figure=tileSvg('Zero fraction · complete tensor areas',row.projection,p,{zero:true});const pattern=result.structured_2_4;
    html=metrics([["Exact zeros",countLabel(result.zero_count)],["Zero fraction",num(result.zero_fraction===null?null:result.zero_fraction*100)+'%'],["All-zero blocks",result.block.all_zero_blocks+' / '+result.block.complete_blocks],["2:4 groups",pattern.compliant_groups+' / '+pattern.groups],["2:4 result",pattern.pattern_satisfied===null?'Not assessable':pattern.pattern_satisfied?'Pattern satisfied':'Pattern violated']])+note('Blocks contain '+result.block.size+' contiguous stored positions; '+result.block.trailing_values+' tail values. 2:4 is checked along native axis '+pattern.axis+'; '+pattern.trailing_values+' ungrouped values. This pattern test does not establish TensorRT eligibility or performance.');
  } else if(view==="quantization" && result.status==="assessed") {
    figure=lineSvg('Serialized quantization scales',[{name:'Scale',values:result.scales}],p,{log:true,xlabel:'Scale index',ylabel:'Scale'});
    html=metrics([["Parameterization",phrase(result.parameterization)],["Axis",String(result.axis??'Per tensor')],["Range used",num(result.range_utilization===null?null:100*result.range_utilization)+'%'],["Lower endpoint",countLabel(result.lower_endpoint_count)],["Upper endpoint",countLabel(result.upper_endpoint_count)]])+note('Endpoint occupancy does not prove clipping. Values are reconstructed with the serialized affine parameters. Original-reference error is available in Original / candidate after selecting a compatible baseline.')+'<details><summary>All scales and zero-points</summary><pre>'+esc(JSON.stringify({scales:result.scales,zero_points:result.zero_points},null,2))+'</pre></details>';
  } else if(view==="comparison") {
    const pair=comparison?.tensors.find(x=>x.candidate_ref===row.weight_ref);
    if(pair?.status!=="assessed")return {html:unavailable(pair || {reason:'select_a_baseline_model_and_analyze_weights'}),svg:null};
    result=pair;
    figure=tileSvg('Candidate − original · aligned values',pair.delta,p);
    const m=pair.metrics;html=metrics([["Changed values",countLabel(m.changed_count)+' / '+countLabel(m.count)],["RMSE",num(m.rmse)],["Relative L2",m.relative_l2===null?phrase(m.relative_l2_status):num(m.relative_l2)],["Cosine",num(m.cosine)],["SQNR dB",m.sqnr_db===null?phrase(m.sqnr_status):num(m.sqnr_db)]])+note('Matching: '+phrase(pair.matching_basis)+'. Compared '+comparison.coverage.compared_count+' / '+comparison.coverage.candidate_count+' candidate tensors. Baseline SHA-256: '+comparison.baseline.source.artifact_sha256);
    html+=distributionComparison(pair,p)||'';
    if(pair.spectrum_change.status==="assessed")html+=lineSvg('Original and candidate spectra',[{name:'Original',values:pair.spectrum_change.baseline_singular_values},{name:'Candidate',values:pair.spectrum_change.candidate_singular_values}],p,{log:true})||'';
    const comparedRows=comparison.tensors.slice(0,64);
    html+=(lineSvg('Aligned tensor RMSE · first '+comparedRows.length+' / '+comparison.tensors.length+' candidates',[{name:'RMSE',values:comparedRows.map(t=>t.metrics?.rmse??null)}],p,{xlabel:'Candidate inventory index',ylabel:'RMSE'})||'')+note('Unassessed tensor errors are omitted from the plot. The complete comparison table includes every candidate and reason.');
    html+='<details><summary>All tensor comparisons and gaps</summary><div class="weight-table-scroll"><table><thead><tr><th>Tensor</th><th>Status</th><th>RMSE</th><th>Reason</th></tr></thead><tbody>'+comparison.tensors.map(x=>'<tr><td>'+esc(x.name)+'</td><td>'+esc(x.status)+'</td><td>'+esc(num(x.metrics?.rmse))+'</td><td>'+esc(phrase(x.reason))+'</td></tr>').join('')+'</tbody></table></div></details>';
  }
  if(result?.status!=="assessed")return {html:basis+unavailable(result),svg:null};
  return {html:basis+(figure||note('No finite plottable values for this chart.'))+html,svg:figure};
}
