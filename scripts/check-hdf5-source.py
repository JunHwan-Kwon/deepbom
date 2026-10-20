"""Real HDF5/Keras files, with values independently produced by h5py/Keras."""
from pathlib import Path
import os
import tempfile
import numpy as np
import h5py
import deepbom
import subprocess

with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp); path=tmp/'weights.h5'
    with h5py.File(path,'w') as f:
        f['float']=np.array([1.,-2.,3.],dtype='>f4')
        d=f.create_dataset('compressed',data=np.arange(6,dtype='float32').reshape(2,3),compression='gzip')
        f['alias']=d; f['cycle']=f; f['soft']=h5py.SoftLink('/float')
        f['external']=h5py.ExternalLink('/tmp/not-selected-by-deepbom.h5','/data')
        f['integer']=np.array([-(2**60),2**60],dtype='int64')
        f['text']=np.array(['abc'],dtype=h5py.string_dtype())
    result=deepbom.numerical_evidence(path,weights=True)['sections']
    rows={r['name']:r for r in result['weight_ir']['tensors']}
    assert rows['/float']['statistics']['minimum']==-2,rows['/float']
    assert rows['/float']['statistics']['maximum']==3,rows['/float']
    assert rows['/alias']['statistics']['value_count']=='6'
    assert rows['/integer']['statistics']['unsafe_integer_count']=='2'
    assert rows['/text']['status']=='not_assessed'
    assert not any(n in rows for n in ['/compressed','/external','/soft'])
    assert result['model_ir']['tensors_and_storage']['totals']['storage_object_count']==0
    import keras
    x=keras.Input((3,)); y=keras.layers.Dense(2,kernel_initializer='ones',bias_initializer='zeros')(x)
    model=keras.Model(x,y); kpath=tmp/'actual.keras';model.save(kpath)
    result=deepbom.numerical_evidence(kpath,weights=True)['sections']
    rows=result['weight_ir']['tensors']
    assert len(rows)==2 and all(r['status']=='assessed' for r in rows),rows
    assert sorted(int(r['statistics']['value_count']) for r in rows)==[2,6]
    browser_check=r'''
    import assert from 'node:assert/strict';import {createServer} from 'node:http';import fs from 'node:fs/promises';import path from 'node:path';
    import {chromium} from 'playwright';import {launchChromium} from './scripts/browser-launch.mjs';
    const root=process.cwd();const server=createServer(async(req,res)=>{try{if(req.url==='/'){res.end('<!doctype html><title>HDF5 fixture</title>');return;}const f=path.resolve(root,'.'+req.url);if(!f.startsWith(root+path.sep))throw Error('outside');res.setHeader('Content-Type',f.endsWith('.js')?'text/javascript':'application/octet-stream');res.end(await fs.readFile(f));}catch{res.statusCode=404;res.end();}});
    await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
    try{browser=await launchChromium(chromium);const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`);
      for(const [file,format] of [[process.argv[1],'hdf5'],[process.argv[2],'keras']]){
        const rows=await page.evaluate(async({bytes,name,format})=>{
          const {initializeNumericalWasm}=await import('/web/lib/numerical-wasm.js');await initializeNumericalWasm();
          const {readSafeSourceArtifactFile}=await import('/web/lib/safe-source-artifacts.js');
          const {getArtifactIrContext}=await import('/web/lib/artifact-ir-context.js');const {sha256BytesHex}=await import('/web/lib/sha256-sync.js');
          const {buildWeightIr}=await import('/web/lib/weight-ir.js');const raw=Uint8Array.from(bytes),file=new File([raw],name);
          const {analysis}=await readSafeSourceArtifactFile(file,format,name);analysis.model_sha256=sha256BytesHex(raw);
          const model=getArtifactIrContext(analysis,{sha256:analysis.model_sha256,size:raw.length,filename:name,format:analysis.format}).model_ir;
          return (await buildWeightIr(model,analysis,file)).tensors;
        },{bytes:[...await fs.readFile(file)],name:path.basename(file),format});
        if(format==='hdf5'){assert.equal(rows.find(r=>r.name==='/float').statistics.minimum,-2);assert.equal(rows.find(r=>r.name==='/text').status,'not_assessed');}
        else{assert.equal(rows.length,2);assert(rows.every(r=>r.status==='assessed'));}
      }
      console.log('Actual browser MEMFS HDF5 and Keras payloads use the same Model/Weight IR and decoder results');
    }finally{await browser?.close();await new Promise(r=>server.close(r));}
    '''
    subprocess.run(['node','--input-type=module','-e',browser_check,str(path),str(kpath)],check=True)
    if os.environ.get('DEEPBOM_HDF5_DIST_ROOT'):
        subprocess.run(['node','scripts/check-hdf5-browser.mjs',str(path),str(kpath)],check=True)
    print('HDF5 BE floats/compressed data/int64/aliases/cycles/external exclusions and actual .keras weights passed')
