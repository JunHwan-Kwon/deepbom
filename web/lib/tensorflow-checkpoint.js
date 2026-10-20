// TensorFlow v2 TensorBundle: bounded LevelDB table metadata + selected shards.
// No tf.load, Python object construction, graph execution, or file discovery.
import { protobufFields, parseTensorShape, tensorFlowDtype } from './safe-source-artifacts.js';
import { sha256BytesHex } from './sha256-sync.js';
const MAX_INDEX = 64 * 1024 * 1024, MAX_ENTRIES = 100_000, MAX_TENSOR = 64 * 1024 * 1024;
const need = (ok,message) => { if (!ok) throw new Error(`TensorFlow checkpoint: ${message}`); };
const utf8 = new TextDecoder('utf-8',{fatal:true});
const crcTable = Uint32Array.from({length:256},(_,i) => {for(let j=0;j<8;j++)i=(i>>>1)^((i&1)?0x82f63b78:0);return i>>>0;});
function maskedCrc(bytes) { let c=0xffffffff;for(const byte of bytes)c=crcTable[(c^byte)&255]^(c>>>8);c=(c^0xffffffff)>>>0;return (((c>>>15)|(c<<17))+0xa282ead8)>>>0; }
function u32(bytes, offset) { need(Number.isSafeInteger(offset)&&offset>=0&&offset+4<=bytes.length,'truncated fixed32');return new DataView(bytes.buffer,bytes.byteOffset+offset,4).getUint32(0,true); }
function vint(bytes,start,end=bytes.length) {
  let value=0n,offset=start;
  for(let shift=0n;shift<=63n;shift+=7n){need(offset<end,'truncated varint');const b=bytes[offset++];need(shift!==63n||b<=1,'varint overflow');value|=BigInt(b&127)<<shift;if(!(b&128)){need(value<=BigInt(Number.MAX_SAFE_INTEGER),'offset exceeds safe indexing');return {value:Number(value),next:offset};}}
  throw new Error('TensorFlow checkpoint: overlong varint');
}
function handle(bytes,start,end=bytes.length){const a=vint(bytes,start,end),b=vint(bytes,a.next,end);return {offset:a.value,length:b.value,next:b.next};}
function block(index,h){
  need(h.offset>=0&&h.length>=0&&h.offset+h.length+5<=index.length-48,'block outside index');
  const payload=index.subarray(h.offset,h.offset+h.length),type=index[h.offset+h.length];
  need(maskedCrc(index.subarray(h.offset,h.offset+h.length+1))===u32(index,h.offset+h.length+1),'table block CRC32C mismatch');
  need(type===0,'compressed metadata blocks are not qualified');return payload;
}
function entries(bytes){
  need(bytes.length>=4,'truncated restart table');const count=u32(bytes,bytes.length-4),limit=bytes.length-4-count*4;
  need(count>=1&&limit>=0,'invalid restart count');const restart=new Set();
  for(let i=0;i<count;i++){const offset=u32(bytes,limit+i*4);need(offset<=limit&&(!i?offset===0:offset>u32(bytes,limit+(i-1)*4)),'invalid restart offset');restart.add(offset);}
  const result=[];let offset=0,previous=new Uint8Array();
  while(offset<limit){const start=offset,a=vint(bytes,offset,limit),b=vint(bytes,a.next,limit),c=vint(bytes,b.next,limit);offset=c.next;
    need(a.value<=previous.length&&offset+b.value+c.value<=limit,'invalid prefix-compressed entry');need(!restart.has(start)||a.value===0,'restart has a shared prefix');restart.delete(start);
    const key=new Uint8Array(a.value+b.value);key.set(previous.subarray(0,a.value));key.set(bytes.subarray(offset,offset+b.value),a.value);offset+=b.value;
    if(result.length) need(compareBytes(previous,key)<0,'table keys are not strictly ordered');
    result.push({key,value:bytes.subarray(offset,offset+c.value)});offset+=c.value;previous=key;need(result.length<=MAX_ENTRIES,'entry count exceeds budget');
  }
  need(offset===limit&&([...restart].every(n=>n===0&&limit===0)),'unreached restart boundary');return result;
}
function compareBytes(a,b){for(let i=0;i<Math.min(a.length,b.length);i++)if(a[i]!==b[i])return a[i]-b[i];return a.length-b.length;}
function scalar(fields,n,wire=0, fallback=0){const rows=fields.filter(f=>f.number===n);need(rows.length<=1,`duplicate field ${n}`);if(!rows.length)return fallback;need(rows[0].wire===wire,'wrong field wire type');return wire===0?Number(rows[0].value):rows[0].bytes;}
function exactScalar(fields,n){const rows=fields.filter(f=>f.number===n);need(rows.length<=1&&(!rows.length||rows[0].wire===0),'invalid scalar field');const value=rows[0]?.value??0n;need(value<=BigInt(Number.MAX_SAFE_INTEGER),'value exceeds bounded file indexing');return Number(value);}

export async function readTensorBundleIndex(file, indexPath) {
  need(file&&Number.isSafeInteger(file.size)&&file.size>=48&&file.size<=MAX_INDEX,'index must be at most 64 MiB');
  const bytes=new Uint8Array(await file.arrayBuffer());need(bytes.length===file.size,'index changed while reading');
  const footer=bytes.length-48;
  need(u32(bytes,bytes.length-8)===0x8b80fb57&&u32(bytes,bytes.length-4)===0xdb477524,'incorrect table magic');
  const meta=handle(bytes,footer,footer+40),idx=handle(bytes,meta.next,footer+40);
  entries(block(bytes,meta)); // Validate selected metadata too; never follow external filters.
  const all=[];let previous=null;const seenBlocks=new Set();
  for(const row of entries(block(bytes,idx))){const h=handle(row.value,0);need(h.next===row.value.length,'trailing block handle');need(!seenBlocks.has(h.offset),'repeated data block');seenBlocks.add(h.offset);
    for(const pair of entries(block(bytes,h))){need(!previous||compareBytes(previous,pair.key)<0,'data keys overlap or are unordered');previous=pair.key;all.push(pair);need(all.length<=MAX_ENTRIES,'global entry budget exceeded');}}
  need(all.length&&all[0].key.length===0,'bundle header missing');
  const header=protobufFields(all[0].value,'BundleHeaderProto',0),shards=exactScalar(header,1),endianness=exactScalar(header,2);
  need(shards>=1&&shards<=20_000&&[0,1].includes(endianness),'unsupported shard count or byte order');
  const version=scalar(header,3,2,null),versionFields=version?protobufFields(version,'BundleHeaderProto.version',1):[];
  const minimum=exactScalar(versionFields,2);need(minimum<=1,'minimum reader version is newer than supported TensorBundle v1');
  for(const field of versionFields.filter(f=>f.number===3)) {
    need(field.wire===0||field.wire===2,'invalid excluded-reader version wire type');
    if(field.wire===0)need(field.value!==1n,'reader version explicitly excluded');
    else for(let offset=0;offset<field.bytes.length;){const item=vint(field.bytes,offset);need(item.value!==1,'reader version explicitly excluded');offset=item.next;}
  }
  const tensors=all.slice(1).map((row,index)=>{
    const fields=protobufFields(row.value,'BundleEntryProto',0),name=utf8.decode(row.key),shapeBytes=scalar(fields,2,2,null);
    const dtype=tensorFlowDtype(exactScalar(fields,1)),shape=shapeBytes?parseTensorShape(shapeBytes,'BundleEntryProto.shape'):null;
    const shard=exactScalar(fields,3),offset=exactScalar(fields,4),length=exactScalar(fields,5),crc=scalar(fields,6,5,null);
    const sliced=fields.some(f=>f.number===7);need(sliced||shard<shards,'shard outside header range');
    const dataPath=indexPath.replace(/\.index$/,`.data-${String(shard).padStart(5,'0')}-of-${String(shards).padStart(5,'0')}`);
    return {index,name,dtype,shape,shard_id:shard,file_path:dataPath,byte_offset:offset,byte_length:length,
      crc32c_masked:crc?u32(crc,0):0,status:sliced?'not_assessed_sliced_tensor':'serialized_dense_entry',byte_order:endianness===0?'little':'big'};
  });
  return {schema:'deepbom.tensorflow_tensor_bundle.v1',index_path:indexPath,index_sha256:sha256BytesHex(bytes),shard_count:shards,byte_order:endianness===0?'little':'big',tensors,
    boundary:'Dense TensorBundle entries are serialized variables/storage, not graph-call or trainability evidence. Sliced, string, variant and resource payloads are not numerically decoded. Metadata compression other than none is not supported.'};
}

export async function visitTensorBundleEntry(source,row,visitor,{maxValues=100_000_000,signal}={}){
  if(row.status!=='serialized_dense_entry')return {status:'not_assessed',reason:row.status};
  if(!Array.isArray(row.shape)||row.shape.some(d=>!Number.isSafeInteger(d)||d<0))return {status:'not_assessed',reason:'checkpoint_shape_not_assessable'};
  if(!/^(?:FLOAT(?:16|32|64)|BFLOAT16|INT(?:8|16|32|64)|UINT(?:8|16|32|64)|BOOL|COMPLEX(?:64|128))$/.test(row.dtype))return {status:'not_assessed',reason:'checkpoint_dtype_not_qualified'};
  const count=row.shape.reduce((n,d)=>n*BigInt(d),1n);
  if(count>BigInt(maxValues)||row.byte_length>MAX_TENSOR)return {status:'not_assessed',reason:'checkpoint_payload_budget_exceeded'};
  if(!source)return {status:'not_assessed',reason:'checkpoint_data_member_not_selected'};
  need(row.byte_offset+row.byte_length<=source.size,'tensor range outside selected shard');signal?.throwIfAborted();
  const bytes=new Uint8Array(await source.slice(row.byte_offset,row.byte_offset+row.byte_length).arrayBuffer());need(bytes.length===row.byte_length,'truncated shard payload');
  need(maskedCrc(bytes)===row.crc32c_masked,'tensor payload CRC32C mismatch');
  const hash=sha256BytesHex(bytes);
  if(row.byte_order==='big'){
    const width=row.dtype==='BOOL'?1:Number(row.dtype.match(/\d+$/)?.[0])/8/(row.dtype.startsWith('COMPLEX')?2:1);
    for(let i=0;i<bytes.length;i+=width)bytes.subarray(i,i+width).reverse();
  }
  const {visitDeclaredNumericBytes}=await import('../onnx.js');const result=visitDeclaredNumericBytes(bytes,row.dtype,row.shape,visitor);
  return result.ok?{status:'assessed',representation:row.dtype.startsWith('COMPLEX')?'complex_magnitude':'stored_scalar',payload_sha256:hash,invalid_encoding_count:0}:{status:'not_assessed',reason:result.reason};
}
