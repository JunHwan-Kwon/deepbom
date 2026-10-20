"""Actual optional frameworks; local-only files and deterministic small models."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import deepbom

with tempfile.TemporaryDirectory(prefix="deepbom-integration-") as temporary:
    root = Path(temporary)
    handle = deepbom.autolog(logdir=root / "disabled")
    handle.disable()
    import keras
    import numpy as np
    import torch
    import wrapt
    assert handle.handle is None, "disabled post-import hook reinstalled itself"
    torch.set_num_threads(1)
    original_fit = keras.Model.fit
    a = keras.Sequential([keras.Input((2,)), keras.layers.Dense(1)])
    b = keras.models.model_from_json(a.to_json()); b.set_weights(a.get_weights())
    for model in (a,b): model.compile(optimizer=keras.optimizers.SGD(.01), loss="mse")
    x, y = np.ones((4,2),np.float32), np.zeros((4,1),np.float32)
    a.fit(x,y,batch_size=2,epochs=1,shuffle=False,verbose=0)
    class UserCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs=None): self.began = True
    callback = UserCallback(); callbacks = [callback]
    with deepbom.autolog(logdir=root / "autolog", every=1, preset="weights", errors="strict") as handle:
        try: deepbom.autolog(logdir=root / "duplicate")
        except ValueError: pass
        else: raise AssertionError("duplicate autolog accepted")
        history = b.fit(x,y,2,1,0,callbacks=callbacks,shuffle=False)
    assert isinstance(history, keras.callbacks.History) and callback.began and callbacks == [callback]
    assert all(np.array_equal(u,v) for u,v in zip(a.get_weights(),b.get_weights()))
    assert handle.receipts[0]["training_lifecycle"] == "ended"
    assert keras.Model.fit is original_fit
    active = deepbom.autolog(logdir=root / "wrappers")
    outer = wrapt.wrap_function_wrapper(keras.Model,"fit",lambda wrapped,instance,args,kwargs: wrapped(*args,**kwargs))
    active.disable()
    assert wrapt.is_wrapped_by(wrapt.resolve_path(keras.Model,"fit")[2],outer)
    wrapt.unwrap_object(keras.Model,"fit",outer)
    assert keras.Model.fit is original_fit
    expected = RuntimeError("user callback")
    class Broken(keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs=None): raise expected
    with deepbom.autolog(logdir=root / "failure", errors="strict") as handle:
        try: b.fit(x,y,callbacks=[Broken()],verbose=0)
        except RuntimeError as error: assert error is expected
        else: raise AssertionError("user exception suppressed")
    assert handle.receipts[0]["training_lifecycle"] == "failed"
    assert not list(root.rglob(".writer.lock"))
    script = root / "exit.py"
    script.write_text("import sys\nassert sys.argv[1:] == ['space argument']\nraise SystemExit(7)\n")
    completed = subprocess.run([sys.executable,"-m","deepbom","instrument","--framework","keras","--logdir",str(root / "launcher"),"--",sys.executable,str(script),"space argument"])
    assert completed.returncode == 7
    print("PASS autolog: before/after import, duplicate, callbacks, original exception, wrapper restoration, exact weights, launcher argv/exit")

    from lightning.pytorch import LightningModule, Trainer
    from torch.utils.data import DataLoader, TensorDataset
    from deepbom.training import lightning_callback
    from deepbom.training.integrations import read_run
    class Network(LightningModule):
        def __init__(self, net): super().__init__(); self.net = net
        def training_step(self, batch, batch_idx):
            loss = self.net(batch[0]).square().mean(); self.log("loss",loss); return loss
        def configure_optimizers(self): return torch.optim.SGD(self.parameters(),lr=.01)
        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            self.log("validation_loss", self.net(batch[0]).square().mean(), on_step=True, on_epoch=False)
    loader = DataLoader(TensorDataset(torch.ones(6,2)), batch_size=2, shuffle=False)
    for precision in ("32-true","bf16-mixed"):
        base = torch.nn.Sequential(torch.nn.Linear(2,1))
        left, right = Network(copy.deepcopy(base)), Network(copy.deepcopy(base))
        options = dict(max_epochs=1,logger=False,enable_checkpointing=False,enable_progress_bar=False,
                       enable_model_summary=False,accelerator="cpu",devices=1,precision=precision,accumulate_grad_batches=2)
        Trainer(**options).fit(left,loader,[loader,loader])
        callback = lightning_callback(right.net,logdir=root / precision,preset="gradients",every=1,errors="strict")
        Trainer(callbacks=[callback],**options).fit(right,loader,[loader,loader])
        assert all(torch.equal(u,v) for u,v in zip(left.parameters(),right.parameters()))
        assert callback.session.receipt["training_lifecycle"] == "ended"
        assert callback.session.receipt["collection_status"] == "completed"
        events = read_run(root / precision)["events"]
        validation = [e for e in events if e["kind"] == "metrics" and e["at"].get("phase") == "validation"]
        assert {e["at"]["dataloader_idx"] for e in validation} == {0, 1}
    print("PASS Lightning: one optimizer, CPU fp32/bf16, accumulated updates, exact state and lifecycle")

    from deepbom.metrics import evaluate as evaluate_builtin
    def evaluate(*args, **kwargs): return evaluate_builtin(*args, **kwargs, backend="torchmetrics")
    image = torch.arange(256,dtype=torch.float32).reshape(1,1,16,16)/255
    options = dict(data_range=1.,gaussian_kernel=False,sigma=1.5,kernel_size=3,k1=.01,k2=.03)
    for _ in range(2):
        observed = evaluate(image,image,metric="ssim",sample_ids=["sample-a"],options=options)
        assert abs(observed["value"]-1)<1e-6 and observed["denominator"] == "1"
    labels = torch.tensor([[[0,0],[1,1]]])
    dice = dict(num_classes=2,include_background=False,weight_type="square",input_format="index",empty_class_policy="torchmetrics_1.9.0")
    observed = evaluate(labels,labels,metric="generalized_dice",sample_ids=["sample-a"],options=dice)
    assert observed["value"] == 1
    bad = image.clone(); bad[0,0,0,0] = torch.nan
    absent = evaluate(bad,image,metric="ssim",sample_ids=["sample-a"],options=options,missing_policy="omit_sample")
    assert absent["status"] == "not_assessed" and absent["value"] is None and absent["denominator"] == "0"
    print("PASS metrics: exact configuration, fresh metric state, SSIM/Dice and missing-value boundary")

    from deepbom.metrics import evaluation_record
    metric_name = "generalized_dice"
    protocol_input = {"schema":"deepbom.evaluation_protocol.v1","id":"synthetic-dice","comparison":"descriptive","data_policy":"same_release",
        "metrics":[{"id":metric_name,"unit":"1","analysis_unit":"sample","higher_is_better":True,"max_regression":0.0,"range":[0,1],"aggregation":"equal_weight_sample_mean","missing_policy":"reject","repeated_observations":"one row per supplied synthetic sample"}],
        "required_populations":["synthetic"],"boundary":"Synthetic adapter check, no clinical assessment"}
    protocol_request = {"schema":"deepbom.evidence_workflow_request.v1","operation":"protocol","input":protocol_input,"documents":[]}
    protocol = deepbom.evidence_workflow(protocol_request)["document"]
    ref = lambda schema,sha: {"kind":"evidence_document","schema":schema,"sha256":sha,"digest_scope":"canonical_document","subject_ref":None}
    file_ref = {"kind":"artifact_file","schema":None,"sha256":"1"*64,"digest_scope":"file_bytes","subject_ref":None}
    template = {key:dict(file_ref) for key in ("subject","dataset","cohort","run","code","environment")}
    template.update(protocol=ref(protocol["schema"],protocol["protocol_sha256"]),configuration=ref("deepbom.metric_configuration.v1",observed["configuration_sha256"]),population="synthetic",comparison="descriptive",pair_set=None,boundary="Synthetic externally measured estimate")
    result=evaluation_record(observed,template,metric_id=metric_name,documents=[protocol])
    assert result["document"]["metrics"][0]["value"]==1 and result["document"]["origin"]=="externally_reported"
    print("PASS measured metric configuration and result use the common Evaluation workflow")

    from deepbom import analyzers
    loaded = []
    descriptor = {"name":"example", "version":"1.0.0", "operation":"protocol", "result_schema":"deepbom.evaluation_protocol.v1", "required_inputs":[], "executes_model":False, "cost":"bounded metadata only", "dependencies":[]}
    class Distribution:
        metadata = {"Name":"fixture"}; version = "1.0.0"
        def read_text(self, filename): return json.dumps({"schema":analyzers.CONTRACT,"analyzers":[descriptor]})
    class Entry:
        name = "example"; dist = Distribution()
        def load(self): loaded.append(self.name); return lambda context: {"operation":"unknown"}
    with patch.object(analyzers,"_entries",return_value=[Entry()]):
        assert analyzers.discover()[0]["available"] and not loaded
        try: analyzers.run("example",{})
        except ValueError as error: assert "undeclared" in str(error)
        else: raise AssertionError("unknown output accepted")
        assert loaded == ["example"]
    class GoodEntry(Entry):
        def load(self): loaded.append(self.name); return lambda context: protocol_request
    with patch.object(analyzers,"_entries",return_value=[GoodEntry()]):
        assert analyzers.run("example",{})["document"]["schema"] == "deepbom.evaluation_protocol.v1"
    with patch.object(analyzers,"_entries",return_value=[Entry(),Entry()]):
        try: analyzers.discover()
        except ValueError as error: assert "collision" in str(error)
        else: raise AssertionError("duplicate name accepted")
    print("PASS plugins: no import during discovery, selected-only execution, unknown contract/collision rejection")
