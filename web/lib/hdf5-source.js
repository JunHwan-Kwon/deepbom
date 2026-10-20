// Selected bytes only. Never import h5wasm/node (which enables host filesystem access).
const MAX_FILE_BYTES = 128 * 1024 * 1024;
const MAX_OBJECTS = 20_000;
const MAX_VALUE_BYTES = 64 * 1024 * 1024;
let mutex = Promise.resolve();

async function withHdf5(source, action) {
  const size = source instanceof Uint8Array ? source.length : source?.size;
  if (!Number.isSafeInteger(size) || size < 8 || size > MAX_FILE_BYTES) throw new Error('HDF5 bounded object reader requires a file of at most 128 MiB');
  const previous = mutex;
  let release; mutex = new Promise(resolve => { release = resolve; });
  await previous;
  let h, file;
  const filename = '/deepbom-selected.h5';
  try {
    h = await import('../../node_modules/h5wasm/dist/esm/hdf5_hl.js');
    const module = await h.ready;
    const bytes = source instanceof Uint8Array ? source : new Uint8Array(await source.arrayBuffer());
    if (bytes.length !== size) throw new Error('HDF5 source changed during read');
    // A single selected file exists at a time; external files cannot be resolved.
    module.FS.writeFile(filename, bytes);
    file = new h.File(filename, 'r');
    return await action(file, module);
  } finally {
    try { file?.close(); } finally {
      try { if (h?.FS?.analyzePath(filename).exists) h.FS.unlink(filename); } finally { release(); }
    }
  }
}

function token(object) {
  const data = object.create_reference().ref_data;
  return Array.from(data, value => value.toString(16).padStart(2,'0')).join('');
}
function dtype(meta) {
  if (meta.vlen || ![1,2,4,8].includes(meta.size)) return 'UNKNOWN';
  if (meta.type === 0) return `${meta.signed ? 'INT' : 'UINT'}${meta.size * 8}`;
  if (meta.type === 1 && [2,4,8].includes(meta.size)) return `FLOAT${meta.size * 8}`;
  return 'UNKNOWN';
}
function descriptor(dataset) {
  const meta = dataset.metadata;
  const shape = Array.isArray(meta.shape) && meta.shape.every(n => Number.isSafeInteger(n) && n >= 0) ? meta.shape : null;
  const count = shape ? shape.reduce((n,d) => n * BigInt(d), 1n) : null;
  return {path:dataset.path, object_token:token(dataset), shape, dtype:dtype(meta),
    element_count_decimal:count?.toString() ?? null,
    logical_bytes_decimal:count != null && dtype(meta) !== 'UNKNOWN' ? (count * BigInt(meta.size)).toString() : null,
    serialized_bytes:null, serialized_bytes_status:'not_exposed_by_container_decoder',
    byte_order:meta.littleEndian ? 'little' : 'big', filters:dataset.filters,
    virtual_sources:meta.virtual_sources || [],
    numeric_status:meta.virtual_sources?.length ? 'not_assessed_virtual_sources' : dtype(meta) === 'UNKNOWN' ? 'not_assessed_datatype' : 'available_within_budget'};
}

export async function inspectHdf5Objects(source) {
  return withHdf5(source, (file) => {
    const objects = [], aliases = [], excluded = [], visited = new Map([[token(file), '/']]);
    const walk = (group, depth) => {
      if (depth > 64) throw new Error('HDF5 group depth exceeds 64');
      for (const name of group.keys().sort()) {
        if (objects.length + aliases.length + excluded.length >= MAX_OBJECTS) throw new Error('HDF5 object/link count exceeds budget');
        const path = `${group.path === '/' ? '' : group.path}/${name}`;
        const symbolic = file.get_link(path);
        if (symbolic != null) { excluded.push({path, kind:'soft_link', target:symbolic, status:'not_followed'}); continue; }
        const object = file.get(path);
        if (object?.type === 'ExternalLink') { excluded.push({path, kind:'external_link', status:'not_followed'}); continue; }
        if (!['Group','Dataset'].includes(object?.type)) { excluded.push({path, kind:object?.type || 'unknown', status:'not_assessed'}); continue; }
        const id = token(object);
        if (visited.has(id)) { aliases.push({path, canonical_path:visited.get(id), basis:'hdf5_object_reference'}); continue; }
        visited.set(id,path);
        if (object.type === 'Group') { objects.push({path, kind:'group', object_token:id}); walk(object,depth+1); }
        else objects.push({kind:'dataset', ...descriptor(object)});
      }
    };
    walk(file,0);
    return {schema:'deepbom.hdf5_object_inventory.v1', method:'h5wasm-0.10.3-memfs', objects, aliases, excluded,
      boundary:'Serialized dataset objects and links, not training parameters. Hard-link aliases share storage; soft/external links are not followed. Variable-length/compound types and virtual data are not numerically decoded. Physical storage size is not substituted with logical element bytes.'};
  });
}

export async function visitHdf5Dataset(source, expected, visitor, {maxValues=100_000_000, signal} = {}) {
  signal?.throwIfAborted();
  return withHdf5(source, async (file, module) => {
    const dataset = file.get(expected.path);
    if (dataset?.type !== 'Dataset') throw new Error('Selected HDF5 dataset is no longer present');
    const current = descriptor(dataset);
    if (JSON.stringify(current) !== JSON.stringify(expected)) throw new Error('HDF5 dataset contract changed');
    if (current.numeric_status !== 'available_within_budget') return {status:'not_assessed', reason:current.numeric_status};
    const count = BigInt(current.element_count_decimal), length = BigInt(current.logical_bytes_decimal);
    if (count > BigInt(maxValues) || length > BigInt(MAX_VALUE_BYTES)) return {status:'not_assessed',reason:'hdf5_numeric_budget_exceeded'};
    const pointer = module._malloc(Math.max(1,Number(length)));
    if (!pointer) throw new Error('HDF5 numeric allocation failed');
    let bytes;
    try {
      // The high-level getter discards H5Dread's return code. Never accept an
      // uninitialized/fill-value result after missing external payload or filter.
      const status = module.get_dataset_data(file.file_id, current.path, null, null, null, BigInt(pointer));
      if (status < 0) return {status:'not_assessed',reason:'hdf5_dataset_read_failed_or_external_payload_unavailable'};
      bytes = module.HEAPU8.slice(pointer,pointer+Number(length));
    } finally { module._free(pointer); }
    signal?.throwIfAborted();
    // H5Dread uses the native memory type (little-endian in WASM), even for
    // big-endian source datasets. Its decompressed/native bytes are not the
    // original serialized payload, so no serialized-payload digest is claimed.
    const { visitDeclaredNumericBytes } = await import('../onnx.js');
    const result = visitDeclaredNumericBytes(bytes,current.dtype,current.shape,visitor);
    return result.ok ? {status:'assessed', representation:'stored_scalar', payload_sha256:null, invalid_encoding_count:0}
      : {status:'not_assessed',reason:result.reason};
  });
}

export function applyHdf5Inventory(analysis, inventory, {memberPath=null} = {}) {
  const rows = inventory.objects.filter(row => row.kind === 'dataset');
  const first = analysis.tensors?.length || 0;
  analysis.tensors = [...(analysis.tensors || []), ...rows.map((row,index) => ({index:first+index, name:row.path,
    dtype:row.dtype, shape:row.shape || [], shape_declared:Array.isArray(row.shape), value_kind:'serialized_hdf5_dataset',
    constant_buffer:true, hdf5_path:row.path, hdf5_member_path:memberPath,
    contract_status:'serialized_dataset_contract_physical_storage_bytes_not_assessed'}))];
  analysis.tensor_count = analysis.tensors.length;
  analysis.format_extensions ||= {};
  analysis.format_extensions.hdf5_objects = {...inventory, member_path:memberPath};
  if (analysis.safe_source_artifact) analysis.safe_source_artifact.object_inventory_status = 'assessed_bounded_dataset_and_link_inventory';
}
