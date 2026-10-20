"""TensorBundle values are compared with TensorFlow's own checkpoint reader."""
from pathlib import Path
import tempfile
import numpy as np
import tensorflow as tf
import deepbom

class Module(tf.Module):
    def __init__(self):
        self.w=tf.Variable([[1.,2.],[-3.,0.]],name='w')
        self.b=tf.Variable([-(2**60),2**60],dtype=tf.int64,name='large')
    @tf.function(input_signature=[tf.TensorSpec([None,2],tf.float32)])
    def __call__(self,x): return x @ self.w

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)/'model';tf.saved_model.save(Module(),str(root))
    reader=tf.train.load_checkpoint(str(root/'variables/variables'))
    result=deepbom.numerical_evidence(root,weights=True)['sections']
    rows={r['name']:r for r in result['weight_ir']['tensors'] if r['native_locator'].startswith('tensorflow:checkpoint:')}
    for key,shape in reader.get_variable_to_shape_map().items():
        if key=='_CHECKPOINTABLE_OBJECT_GRAPH':
            assert rows[key]['status']=='not_assessed'
            continue
        arr=reader.get_tensor(key)
        assert rows[key]['status']=='assessed',rows[key]
        assert rows[key]['statistics']['value_count']==str(arr.size)
        if arr.dtype==np.float32:
            assert rows[key]['statistics']['minimum']==float(arr.min())
            assert rows[key]['statistics']['maximum']==float(arr.max())
        else: assert rows[key]['statistics']['unsafe_integer_count']=='2'
    # Index damage must be rejected instead of silently returning zero variables.
    index=root/'variables/variables.index';raw=bytearray(index.read_bytes());raw[8]^=1;index.write_bytes(raw)
    try: deepbom.numerical_evidence(root,weights=True)
    except deepbom.DeepBomError: pass
    else: raise AssertionError('Corrupt TensorBundle table was accepted')
    print('Actual SavedModel TensorBundle variable values, exact int64, unsupported string and index corruption checks passed')
