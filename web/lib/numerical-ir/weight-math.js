import { statisticsOf } from "./statistics.js";
import { ExactMoments, exactCosine } from "./exact-moments.js";

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
  return exactCosine(a, b);
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
function splitMagnitude(value) {
  if(!value)return [0,0];
  const exponent=Math.min(1023,Math.floor(Math.log2(value)));
  return [value/2**exponent,exponent];
}
function scaledMagnitude(value, exponent) {
  if(!value)return 0;
  const [mantissa,shift]=splitMagnitude(value);exponent+=shift;
  if(exponent>1023)return Infinity;
  return exponent < -1074 ? mantissa*2**(exponent+1074)*Number.MIN_VALUE : mantissa*2**exponent;
}
export function spectrumAnalysis(matrix, maxWork=30_000_000) {
  const {rows,columns,data}=matrix, n=Math.min(rows,columns), m=Math.max(rows,columns);
  if(!n || !m) return notAssessed("empty_matrix");
  const diagonal=data.every((value,i)=>value===0||Math.floor(i/columns)===i%columns)
    ? Array.from({length:n},(_,i)=>Math.abs(data[i*columns+i])).sort((a,b)=>b-a) : null;
  if((diagonal?data.length:m*n*(n-1)/2)>maxWork) return notAssessed("full_svd_exceeds_budget");
  const scale=magnitude(data), a=new Float64Array(data.length);
  for(let i=0;i<m;i++) for(let j=0;j<n;j++) a[i*n+j]=scale ? (rows>=columns ? data[i*columns+j] : data[j*columns+i])/scale : 0;
  const normalizationLoss=data.some(value=>value!==0&&value/scale===0);
  if(normalizationLoss&&!diagonal)return notAssessed("svd_normalization_underflow");
  let initialEnergy=0;for(const x of a)initialEnergy+=x*x;
  let smallestExponent=Infinity;for(const x of a)if(x)smallestExponent=Math.min(smallestExponent,splitMagnitude(Math.abs(x))[1]);
  // Deflate only far below the precision of even the smallest nonzero input,
  // not below the largest column. The per-column discarded norm is bounded by
  // eps^2 times that input magnitude, also below the reported rank tolerance.
  const deflationExponent=smallestExponent-104-Math.ceil(Math.log2(m)/2);
  // Keep column magnitudes separately from unit-scale entries. Otherwise
  // rank-deficient columns can stagnate in subnormal rounding noise instead
  // of converging, and rescaling can erase representable small singular values.
  const magnitudes=new Float64Array(n),exponents=new Int32Array(n);
  if(!diagonal)for(let j=0;j<n;j++) {
    let maximum=0;for(let i=0;i<m;i++)maximum=Math.max(maximum,Math.abs(a[i*n+j]));
    [magnitudes[j],exponents[j]]=splitMagnitude(maximum);
    if(maximum)for(let i=0;i<m;i++)a[i*n+j]/=maximum;
  }
  const tolerance=32*Number.EPSILON;
  let work=diagonal?data.length:0,sweeps=0,converged=!!diagonal,residual=0;
  for(;!diagonal&&sweeps<64;sweeps++) {
    let changed=false; residual=0;
    for(let p=0;p<n;p++) for(let q=p+1;q<n;q++) {
      work+=m;
      if(work>maxWork) return notAssessed("svd_iteration_budget_exceeded");
      const sa=magnitudes[p],sb=magnitudes[q];
      // Small correlated columns still need orthogonalization. A Frobenius-
      // scaled cutoff here can falsely count dependent columns as full rank.
      if(!sa || !sb) continue;
      let aa=0,bb=0,ab=0;
      for(let i=0;i<m;i++){const x=a[i*n+p],y=a[i*n+q];aa+=x*x;bb+=y*y;ab+=x*y;}
      const correlation=Math.abs(ab)/Math.sqrt(aa)/Math.sqrt(bb);residual=Math.max(residual,correlation);
      if(correlation<=tolerance)continue;
      const pairExponent=Math.max(exponents[p],exponents[q]),ra=sa*2**(exponents[p]-pairExponent),rb=sb*2**(exponents[q]-pairExponent);
      const delta=(bb*rb*rb-aa*ra*ra)/2,product=ab*ra*rb;
      const denominator=delta+(delta<0?-1:1)*Math.hypot(delta,product);
      if(!denominator)return notAssessed("svd_rotation_underflow");
      const t=product/denominator,c=1/Math.hypot(1,t);
      // Form the scaled rotation coefficients before multiplying tiny column
      // magnitudes. A vanishing t can still have a finite effect on the small
      // column; skipping that pair corrupts its singular value.
      const pMix=ab*rb*rb/denominator,qMix=ab*ra*ra/denominator;
      let maxP=0,maxQ=0;
      for(let i=0;i<m;i++){const x=a[i*n+p],y=a[i*n+q];a[i*n+p]=c*(x-pMix*y);a[i*n+q]=c*(qMix*x+y);maxP=Math.max(maxP,Math.abs(a[i*n+p]));maxQ=Math.max(maxQ,Math.abs(a[i*n+q]));}
      for(const [j,maximum] of [[p,maxP],[q,maxQ]]) {
        const [mantissa,shift]=splitMagnitude(magnitudes[j]*maximum);magnitudes[j]=mantissa;exponents[j]+=shift;
        if(exponents[j]<deflationExponent)magnitudes[j]=0;
        if(maximum)for(let i=0;i<m;i++)a[i*n+j]/=maximum;
      }
      changed=true;
    }
    if(!changed){converged=true;break;}
  }
  if(!converged)return notAssessed("svd_not_converged");
  const factors=diagonal?null:Array.from({length:n},(_,j)=>{let norm=0;for(let i=0;i<m;i++)norm=Math.hypot(norm,a[i*n+j]);const [mantissa,shift]=splitMagnitude(norm*magnitudes[j]);return {mantissa,exponent:exponents[j]+shift};}).sort((a,b)=>Number(b.mantissa>0)-Number(a.mantissa>0)||b.exponent-a.exponent||b.mantissa-a.mantissa);
  const sigma=diagonal?diagonal.map(x=>scale?x/scale:0):factors.map(f=>scaledMagnitude(f.mantissa,f.exponent));
  const [scaleMantissa,scaleExponent]=splitMagnitude(scale);
  const singularValues=diagonal||factors.map(f=>scaledMagnitude(f.mantissa*scaleMantissa,f.exponent+scaleExponent));
  const total=sigma.reduce((sum,x)=>sum+x*x,0),energyError=initialEnergy?Math.abs(total-initialEnergy)/initialEnergy:0;
  if(energyError>1e-10)return notAssessed("svd_energy_residual_exceeded");
  const cutoff=Math.max(m,n)*Number.EPSILON*sigma[0], rank=sigma.filter(x=>x>cutoff).length;
  let cumulative=0;
  const retained=sigma.map(x=>{cumulative+=x*x;return total?Math.min(1,cumulative/total):null;});
  // Tail accumulation avoids subtracting nearly equal energies.
  const errors=Array(n+1).fill(0);let tail=0;for(let k=n-1;k>=0;k--){tail=Math.hypot(tail,sigma[k]);errors[k]=total?tail/Math.sqrt(total):null;}if(!total)errors[n]=null;
  return assessed({method:diagonal?"diagonal_closed_form_binary64":"scaled_one_sided_jacobi_binary64",rows,columns,singular_values:singularValues.map(safe),normalized_singular_values:sigma,overflowed_singular_value_indices:singularValues.flatMap((x,i)=>Number.isFinite(x)?[]:[i]),scale,numerical_rank:rank,rank_tolerance:safe(cutoff*scale),rank_tolerance_normalized:cutoff,condition_number:rank===n && sigma[n-1]>0?safe(sigma[0]/sigma[n-1]):null,condition_status:rank<n?"numerically_rank_deficient":Number.isFinite(sigma[0]/sigma[n-1])?"finite":"overflow",cumulative_energy:retained,relative_frobenius_error_by_rank:errors,sweeps,orthogonality_residual:residual,energy_relative_residual:energyError,work_count:work,scope:"declared_axis_unfolding_not_full_convolution_operator"});
}

export function tileProjection(matrix, maximum=48) {
  const {rows,columns,data}=matrix, height=Math.min(rows,maximum),width=Math.min(columns,maximum);
  if(!height||!width)return notAssessed("empty_matrix");
  const cells=Array.from({length:height*width},()=>({count:0,zero_count:0,mean:0,minimum:null,maximum:null}));
  const sums=cells.map(()=>new ExactMoments({squares:false}));
  for(let r=0;r<rows;r++)for(let c=0;c<columns;c++) {
    const index=Math.floor(r*height/rows)*width+Math.floor(c*width/columns),cell=cells[index],v=data[r*columns+c];
    cell.count++;cell.zero_count+=Number(v===0);sums[index].add(v);
    cell.minimum=cell.minimum===null?v:Math.min(cell.minimum,v);cell.maximum=cell.maximum===null?v:Math.max(cell.maximum,v);
  }
  for(const [index,cell] of cells.entries())cell.mean=sums[index].mean(cell.count);
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
