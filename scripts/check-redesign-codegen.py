"""Execute both generated framework scaffolds with identical bound weights."""
import importlib.util, json, os, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch
import tensorflow as tf
torch.set_num_threads(1)
source = '''
import fs from 'node:fs';import path from 'node:path';import {createHash} from 'node:crypto';
import {initSync,analyze_tflite_for_target,project_tflite_redesign,explore_tflite_redesign_pareto} from './pkg/tflite_wasm_audit.js';
import {buildRedesignImplementationFiles} from './web/lib/redesign-codegen.js';
initSync({module:fs.readFileSync('pkg/tflite_wasm_audit_bg.wasm')});
const bytes=fs.readFileSync('web/samples/mobilenet_v2_1.0_224_quant.tflite'),sha=createHash('sha256').update(bytes).digest('hex');
const a=analyze_tflite_for_target(bytes,'model.tflite','android_mid_a55');
const r={schema:'deepbom.redesign_request.v1',source_sha256:sha,input_height:224,input_width:224,width_multiplier:1,activation_dtype:'source',block_edits:[]};
const p=project_tflite_redesign(bytes,'model.tflite','android_mid_a55',r);
for(const f of buildRedesignImplementationFiles({analysis:a,projection:p,request:r})){const out=path.join(process.argv[1],f.name);fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,f.data);}
'''
def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path); result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result);return result
with tempfile.TemporaryDirectory(prefix='deepbom-codegen-') as tmp:
    root=Path(tmp)
    subprocess.run(['node','--input-type=module','-e',source,str(root)],check=True,capture_output=True)
    plan=json.loads((root/'implementation_plan.json').read_text())
    a=module(root/'pytorch/model.py','generated_torch').ReconstructedModel().eval()
    b=module(root/'keras/model.py','generated_keras').build_model()
    rng=np.random.default_rng(13); captures={}; handles=[]; compared=[]
    with torch.no_grad():
        for row in plan['nodes']:
            name=row['generated_symbol'];kind=row['module_kind']
            if kind not in ('conv2d','depthwise_conv2d','linear'): continue
            layer=b.get_layer(name);arrays=layer.get_weights()
            weights=[rng.normal(0,.04,size=w.shape).astype('float32') for w in arrays]
            layer.set_weights(weights)
            dest=getattr(a,name);dest=dest.conv if hasattr(dest,'conv') else dest
            w=weights[0]
            if kind=='depthwise_conv2d':w=w.transpose(2,3,0,1).reshape(dest.weight.shape)
            elif kind=='conv2d':w=w.transpose(3,2,0,1)
            else:w=w.T
            dest.weight.copy_(torch.from_numpy(w.copy()));dest.bias.copy_(torch.from_numpy(weights[1]))
            handles.append(getattr(a,name).register_forward_hook(lambda m,args,y,name=name:captures.__setitem__(name,y.detach().numpy())))
            compared.append(name)
        shape=plan['model_inputs'][0]['projected_shape'];x=rng.normal(size=shape).astype('float32')
        actual=a(torch.from_numpy(x.copy()))  # Generated public boundary preserves NHWC.
        observed=tf.keras.Model(b.inputs,[b.get_layer(n).output for n in compared])(x,training=False)
        for name,wanted in zip(compared,observed):
            got=captures[name];got=got.transpose(0,2,3,1) if got.ndim==4 else got
            np.testing.assert_allclose(got,wanted.numpy(),rtol=2e-4,atol=2e-5,err_msg=name)
        wanted=b(x,training=False)
        got=actual.detach().numpy() if torch.is_tensor(actual) else actual[0].detach().numpy()
        np.testing.assert_allclose(got,wanted.numpy(),rtol=2e-4,atol=2e-5)
        assert list(wanted.shape)==plan['model_outputs'][0]['projected_shape']
    for handle in handles:handle.remove()
    print(f'Generated Torch/Keras execution passed: {len(compared)} shared-weight intermediate layers and final output contract')

# Boundary cases the main CNN does not contain: output rank one and a negative
# NHWC concat axis, verified by running generated code in both frameworks.
edge_source = r'''
import fs from 'node:fs';import path from 'node:path';
import {buildRedesignImplementationFiles} from './web/lib/redesign-codegen.js';
const sha='a'.repeat(64),request={source_sha256:sha};
for(const kind of ['reshape','concatenate']){
 const inputShape=kind==='reshape'?[2,1]:[1,2,3,1];const outputShape=kind==='reshape'?[2]:[1,2,3,2];
 const ins=kind==='reshape'?[0]:[0,1];const out=ins.length;
 const plan={schema:'deepbom.redesign_implementation_plan.v1',exportable:true,model_inputs:ins.map(tensor_index=>({tensor_index,projected_shape:inputShape})),model_outputs:[{tensor_index:out,projected_shape:outputShape}],nodes:[{op_index:0,op_name:kind,generated_symbol:'op_0',module_kind:kind,activation_inputs:ins,input_shapes:ins.map(()=>inputShape),outputs:[out],output_shapes:[outputShape],concatenation_axis_nhwc:-1,codegen_status:'exact_structure'}]};
 const projection={projection_status:'PROJECTED_UNTRAINED',source:{sha256_before:sha,sha256_after:sha,loaded_source_bytes_unchanged:true},implementation_plan:plan};
 for(const f of buildRedesignImplementationFiles({analysis:{model_sha256:sha},projection,request})){const dest=path.join(process.argv[1],kind,f.name);fs.mkdirSync(path.dirname(dest),{recursive:true});fs.writeFileSync(dest,f.data);}
}
'''
with tempfile.TemporaryDirectory(prefix='deepbom-codegen-edges-') as tmp:
    root=Path(tmp)
    subprocess.run(['node','--input-type=module','-e',edge_source,str(root)],check=True,capture_output=True)
    for kind in ['reshape','concatenate']:
        p=root/kind
        a=module(p/'pytorch/model.py','edge_torch_'+kind).ReconstructedModel().eval()
        b=module(p/'keras/model.py','edge_keras_'+kind).build_model()
        if kind=='reshape':
            arrays=[np.array([[1.],[2.]],dtype=np.float32)];wanted=np.array([1.,2.],dtype=np.float32)
        else:
            arrays=[np.arange(6,dtype=np.float32).reshape(1,2,3,1),np.arange(6,12,dtype=np.float32).reshape(1,2,3,1)];wanted=np.concatenate(arrays,axis=-1)
        with torch.no_grad(): actual=a(*(torch.from_numpy(x.copy()) for x in arrays)).numpy()
        reference=b([tf.convert_to_tensor(x) for x in arrays],training=False).numpy()
        np.testing.assert_array_equal(actual,wanted);np.testing.assert_array_equal(reference,wanted)
    print('Generated Torch/Keras rank-one reshape and negative NHWC concatenation axis passed')
