import { statisticsOf } from "./statistics.js";

export const notAssessed = reason => ({ status: "not_assessed", reason });
export const assessed = data => ({ status: "assessed", ...data });
const safe = value => Number.isFinite(value) ? value : null;

// Rescale before products. Original values are never rounded to float32.
export function magnitude(values) {
  let scale = 0;
  for (const value of values) scale = Math.max(scale, Math.abs(value));
  return scale;
}
export function cosine(a, b) {
  const sa = magnitude(a), sb = magnitude(b);
  if (!sa || !sb) return null;
  let dot = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) { const x = a[i] / sa, y = b[i] / sb; dot += x*y; aa += x*x; bb += y*y; }
  return Math.max(-1, Math.min(1, dot / Math.sqrt(aa * bb)));
}
export function matrixView(values, shape, axis, order) {
  const rows = shape.length ? shape[axis] : 1, columns = rows ? values.length / rows : 0;
  const stride = (order === "first_axis_fastest" ? shape.slice(0,axis) : shape.slice(axis+1)).reduce((a,b)=>a*b,1);
  const data = new Float64Array(values.length), offsets = new Uint32Array(rows);
  for (let i=0;i<values.length;i++) {const row = Math.floor(i/stride)%rows; data[row*columns+offsets[row]++] = values[i];}
  return { data, rows, columns };
}
export function channelStatistics(matrix) {
  const {data,rows,columns}=matrix;
  return assessed({count:rows, values_per_channel:columns, channels:Array.from({length:rows},(_,i)=>({index:i,statistics:statisticsOf(data.subarray(i*columns,(i+1)*columns))}))});
}
export function similarityAnalysis(matrix, limit, threshold) {
  const {data,rows,columns}=matrix;
  if (rows>limit || rows*rows*columns>20_000_000) return notAssessed("complete_pairwise_matrix_exceeds_budget");
  const cells=Array(rows*rows).fill(null), pairs=[], parent=Array.from({length:rows},(_,i)=>i);
  const root=i=>{while(parent[i]!==i)i=parent[i];return i;};
  for(let a=0;a<rows;a++) for(let b=a;b<rows;b++) {
    const value=cosine(data.subarray(a*columns,(a+1)*columns),data.subarray(b*columns,(b+1)*columns));
    cells[a*rows+b]=cells[b*rows+a]=value;
    if(a!==b && value!==null && value>=threshold) {pairs.push({a,b,cosine:value});parent[root(b)]=root(a);}
  }
  const groups=new Map();for(let i=0;i<rows;i++){const r=root(i);if(!groups.has(r))groups.set(r,[]);groups.get(r).push(i);}
  return assessed({method:"complete_signed_cosine_no_sampling",size:rows,values:cells,threshold,pairs,clusters:[...groups.values()].filter(g=>g.length>1),cluster_method:"connected_components_of_threshold_edges_not_all_pairs_equivalence",undefined_reason:"zero_norm_channels"});
}

// One-sided cyclic Jacobi on a scaled matrix, transposed when this reduces the
// number of columns. No Gram matrix is formed (it would square conditioning).
export function spectrumAnalysis(matrix, maxWork=30_000_000) {
  const {rows,columns,data}=matrix, n=Math.min(rows,columns), m=Math.max(rows,columns);
  if(!n || !m) return notAssessed("empty_matrix");
  if(n>256 || m*n*n>maxWork) return notAssessed("full_svd_exceeds_budget");
  const scale=magnitude(data), a=new Float64Array(data.length);
  for(let i=0;i<m;i++) for(let j=0;j<n;j++) a[i*n+j]=scale ? (rows>=columns ? data[i*columns+j] : data[j*columns+i])/scale : 0;
  let initialEnergy=0;for(const x of a)initialEnergy+=x*x;
  const tolerance=32*Number.EPSILON, nullEnergy=initialEnergy*(Math.max(m,n)*Number.EPSILON)**2;
  let work=0,sweeps=0,converged=false,residual=0;
  for(;sweeps<64;sweeps++) {
    let changed=false; residual=0;
    for(let p=0;p<n;p++) for(let q=p+1;q<n;q++) {
      work+=m;
      if(work>maxWork) return notAssessed("svd_iteration_budget_exceeded");
      let aa=0,bb=0,ab=0;
      for(let i=0;i<m;i++){const x=a[i*n+p],y=a[i*n+q];aa+=x*x;bb+=y*y;ab+=x*y;}
      if(aa<=nullEnergy || bb<=nullEnergy) continue;
      const correlation=Math.abs(ab)/Math.sqrt(aa*bb);residual=Math.max(residual,correlation);
      if(correlation<=tolerance)continue;
      const z=(bb-aa)/(2*ab),t=(z<0?-1:1)/(Math.abs(z)+Math.hypot(1,z)),c=1/Math.hypot(1,t),s=c*t;
      for(let i=0;i<m;i++){const x=a[i*n+p],y=a[i*n+q];a[i*n+p]=c*x-s*y;a[i*n+q]=s*x+c*y;}
      changed=true;
    }
    if(!changed){converged=true;break;}
  }
  if(!converged)return notAssessed("svd_not_converged");
  const sigma=Array.from({length:n},(_,j)=>{let norm=0;for(let i=0;i<m;i++)norm+=a[i*n+j]**2;return Math.sqrt(norm);}).sort((x,y)=>y-x);
  const total=sigma.reduce((sum,x)=>sum+x*x,0),energyError=initialEnergy?Math.abs(total-initialEnergy)/initialEnergy:0;
  if(energyError>1e-10)return notAssessed("svd_energy_residual_exceeded");
  const cutoff=Math.max(m,n)*Number.EPSILON*sigma[0], rank=sigma.filter(x=>x>cutoff).length;
  let cumulative=0;
  const retained=sigma.map(x=>{cumulative+=x*x;return total?Math.min(1,cumulative/total):null;});
  // Tail accumulation avoids subtracting nearly equal energies.
  const errors=Array(n+1).fill(0);let tail=0;for(let k=n-1;k>=0;k--){tail+=sigma[k]**2;errors[k]=total?Math.sqrt(tail/total):null;}if(!total)errors[n]=null;
  return assessed({method:"scaled_one_sided_jacobi_binary64",rows,columns,singular_values:sigma.map(x=>safe(x*scale)),normalized_singular_values:sigma,overflowed_singular_value_indices:sigma.flatMap((x,i)=>Number.isFinite(x*scale)?[]:[i]),scale,numerical_rank:rank,rank_tolerance:safe(cutoff*scale),rank_tolerance_normalized:cutoff,condition_number:rank===n && sigma[n-1]>0?safe(sigma[0]/sigma[n-1]):null,condition_status:rank<n?"numerically_rank_deficient":Number.isFinite(sigma[0]/sigma[n-1])?"finite":"overflow",cumulative_energy:retained,relative_frobenius_error_by_rank:errors,sweeps,orthogonality_residual:residual,energy_relative_residual:energyError,work_count:work,scope:"declared_axis_unfolding_not_full_convolution_operator"});
}

export function tileProjection(matrix, maximum=48) {
  const {rows,columns,data}=matrix, height=Math.min(rows,maximum),width=Math.min(columns,maximum);
  if(!height||!width)return notAssessed("empty_matrix");
  const cells=Array.from({length:height*width},()=>({count:0,zero_count:0,mean:0,minimum:null,maximum:null}));
  const scale=magnitude(data);
  for(let r=0;r<rows;r++)for(let c=0;c<columns;c++) {
    const cell=cells[Math.floor(r*height/rows)*width+Math.floor(c*width/columns)],v=data[r*columns+c];
    cell.count++;cell.zero_count+=Number(v===0);cell.mean+=(scale?v/scale:0)/cell.count-cell.mean/cell.count;
    cell.minimum=cell.minimum===null?v:Math.min(cell.minimum,v);cell.maximum=cell.maximum===null?v:Math.max(cell.maximum,v);
  }
  for(const cell of cells)cell.mean*=scale;
  return assessed({method:height===rows&&width===columns?"exact_cells":"complete_area_aggregation_no_sampling",rows:height,columns:width,source_rows:rows,source_columns:columns,cells});
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
  let zeros=0;for(const x of values)zeros+=Number(x===0);
  let fullBlocks=0,zeroBlocks=0,tail=values.length%blockSize;
  for(let start=0;start+blockSize<=values.length;start+=blockSize){fullBlocks++;let allZero=true;for(let i=start;i<start+blockSize;i++)if(values[i]!==0)allZero=false;zeroBlocks+=Number(allZero);}
  const matrix=matrixView(values,shape,axis,order), {rows,columns,data}=matrix;
  // Groups are along the selected native axis, with all other coordinates fixed.
  let groups=0,compliant=0;
  for(let c=0;c<columns;c++)for(let r=0;r+4<=rows;r+=4){let nonzero=0;for(let k=0;k<4;k++)nonzero+=Number(data[(r+k)*columns+c]!==0);groups++;compliant+=Number(nonzero<=2);}
  return assessed({zero_count:String(zeros),value_count:String(values.length),zero_fraction:values.length?zeros/values.length:null,block:{basis:"contiguous_serialized_values",size:blockSize,complete_blocks:fullBlocks,all_zero_blocks:zeroBlocks,trailing_values:tail},structured_2_4:{axis,groups,compliant_groups:compliant,trailing_values:(rows%4)*columns,pattern_satisfied:groups>0&&rows%4===0?compliant===groups:null,scope:"selected_axis_pattern_only_not_backend_eligibility_or_speedup"}});
}
export function compareValues(baseline,candidate) {
  if(baseline.length!==candidate.length)return notAssessed("value_count_mismatch");
  const scale=Math.max(magnitude(baseline),magnitude(candidate)),referenceScale=magnitude(baseline);
  let square=0,reference=0,max=0,absolute=0,changed=0;
  const differences=Float64Array.from(baseline,(v,i)=>candidate[i]-v);
  const differenceScale=differences.every(Number.isFinite)?magnitude(differences):scale;
  // Subtract before scaling to preserve differences between adjacent numbers.
  // For an overflowing difference, subtract the individually scaled operands.
  for(let i=0;i<baseline.length;i++) {
    const difference=candidate[i]-baseline[i];differences[i]=difference;
    const d=differenceScale?(Number.isFinite(difference)?difference/differenceScale:candidate[i]/differenceScale-baseline[i]/differenceScale):0;
    max=Math.max(max,Math.abs(d));changed+=Number(baseline[i]!==candidate[i]);
    if(referenceScale)reference+=(baseline[i]/referenceScale)**2;
  }
  // Normalize a second time by the maximum difference: tiny errors must not
  // disappear when their squares would underflow relative to the model values.
  for(let i=0;i<baseline.length;i++) {
    const d=differenceScale?(Number.isFinite(differences[i])?differences[i]/differenceScale:candidate[i]/differenceScale-baseline[i]/differenceScale):0;
    const unit=max?d/max:0;square+=unit*unit;absolute+=Math.abs(unit);
  }
  const n=baseline.length,errorScale=max*differenceScale;
  const rmse=n?Math.sqrt(square/n)*max*differenceScale:null;
  const referenceLog=referenceScale?Math.log(referenceScale)+Math.log(reference)/2:null;
  const errorLog=max?Math.log(max)+Math.log(differenceScale)+Math.log(square)/2:null;
  const relative=referenceLog===null?null:errorLog===null?0:Math.exp(errorLog-referenceLog);
  return {metrics:assessed({count:String(n),changed_count:String(changed),mean_absolute_error:n?safe(absolute/n*max*differenceScale):null,maximum_absolute_error:safe(errorScale),rmse:safe(rmse),mse:rmse===null?null:safe(rmse*rmse),relative_l2:safe(relative),relative_l2_status:referenceLog===null?"zero_reference_norm":Number.isFinite(relative)?"assessed":"overflow",cosine:cosine(baseline,candidate),sqnr_db:referenceLog!==null&&errorLog!==null?safe(20/Math.LN10*(referenceLog-errorLog)):null,sqnr_status:referenceLog===null?"zero_reference_energy":errorLog===null?"zero_error":"finite"}),differences};
}
