"""Safe inspection of real torch.save and torch.export archives; no unpickling."""
import tempfile
from pathlib import Path
import torch
import deepbom

with tempfile.TemporaryDirectory(prefix="deepbom-torch-source-") as temporary:
    root=Path(temporary)
    checkpoint=root/'state.pt'
    # Enough unique memo entries to exercise LONG_BINPUT / LONG_BINGET.
    weights={f'layer_{i}.weight':torch.ones(1) for i in range(300)}
    weights['shared_alias']=weights['layer_299.weight']
    torch.save(weights,checkpoint)
    result=deepbom.audit(checkpoint,output="analysis")
    assert result['format']=='pytorch_checkpoint',result.get('format')
    model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.ReLU()).eval()
    exported=torch.export.export(model,(torch.ones(1,2),))
    path=root/'exported.pt2';torch.export.save(exported,path)
    result=deepbom.audit(path,output="analysis")
    assert result['format']=='pt2',result.get('format')
    assert result['operator_count']>=2,result
    print('Actual torch.save memo/shared storage and torch.export PT2 safe inspection passed')
