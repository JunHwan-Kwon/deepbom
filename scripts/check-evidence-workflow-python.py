#!/usr/bin/env python3
"""Optional installed/source Python scenario; requires report, MLflow, TensorBoard and W&B extras."""
import os,sys,json,hashlib,tempfile,subprocess,io
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,os.environ.get('DEEPBOM_TEST_PACKAGE_ROOT',str(root/'channels/python/src')))
engine=os.environ.get('DEEPBOM_ENGINE', 'deepbom')
from deepbom import evidence_workflow
from deepbom.evidence import export
source=Path(tempfile.mkdtemp(prefix='deepbom-workflow-python-'))/'inputs'
subprocess.run(['node',str(root/'examples/evidence-workflow/write-synthetic.mjs'),str(source)],check=True,cwd=root)
output=source.parent/'output';output.mkdir()
expected=subprocess.check_output(['node','--input-type=module','-e',"import {syntheticScenario} from './examples/evidence-workflow/synthetic.mjs';import {runEvidenceWorkflow} from './web/lib/evidence-workflow.js';const s=syntheticScenario();console.log(JSON.stringify(runEvidenceWorkflow(s.request,{files:s.files})));"],cwd=root,text=True)
request=json.loads((source/'request.json').read_text());files={hashlib.sha256(f.read_bytes()).hexdigest():str(f) for f in source.glob('source-*.bin')}
result=evidence_workflow(request,files=files)
assert result==json.loads(expected)
resultfile=output/'result.json';resultfile.write_text(json.dumps(result,indent=2)+'\n')
for format in ('html','pdf','cyclonedx','spdx'):
 target=output/('report.'+format)
 if target.exists():target.unlink()
 env={**os.environ,'DEEPBOM_REPORT_PYTHON':sys.executable,'PYTHONPATH':os.environ.get('DEEPBOM_TEST_PACKAGE_ROOT',str(root/'channels/python/src'))}
 subprocess.run([str(engine),'evidence-workflow',str(resultfile),'--format',format,'--output',str(target)],env=env,check=True)
assert (output/'report.pdf').read_bytes().startswith(b'%PDF-')
from pypdf import PdfReader
text='\n'.join(page.extract_text() for page in PdfReader(str(output/'report.pdf')).pages)
assert 'denominator' in text and 'missing_count' in text
assert result['result_sha256'] in text.replace('\n','')
from deepbom.evidence_render import render
view=json.loads(subprocess.check_output(['node','--input-type=module','-e',"import {readFileSync} from 'node:fs';import {evidenceWorkflowReport} from './web/lib/evidence-workflow-report.js';console.log(JSON.stringify(evidenceWorkflowReport(JSON.parse(readFileSync(process.argv[1],'utf8')))));",str(resultfile)],cwd=root,text=True))
view['rows'][0]['unit']='환자/검사'
unicode_text='\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(render(view))).pages)
assert '\\ud658' in unicode_text and 'explicit Unicode escapes' in unicode_text
tb=export(result,destination='tensorboard',logdir=output/'tb')
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util.tensor_util import make_ndarray
acc=EventAccumulator(tb['path']).Reload();assert make_ndarray(acc.Tensors('original/absolute_error/delta')[0].tensor_proto).item()==1
import mlflow
uri='sqlite:///'+str(output/'mlflow.db');mlflow.set_tracking_uri(uri)
client=mlflow.tracking.MlflowClient();experiment=client.get_experiment_by_name('synthetic-evidence')
experiment_id=experiment.experiment_id if experiment else client.create_experiment('synthetic-evidence',artifact_location=(output/'artifacts').as_uri())
run=client.create_run(experiment_id)
receipt=export(result,destination='mlflow',run_id=run.info.run_id)
path=client.download_artifacts(run.info.run_id,'deepbom/evidence/evidence-result.json')
assert json.loads(Path(path).read_text())==result
client.set_terminated(run.info.run_id)
import wandb
with wandb.init(project='deepbom-evidence-synthetic',mode='offline',dir=str(output)):
 export(result,destination='wandb')
(output/'validation.json').write_text(json.dumps({'python_sdk_parity':True,'html_pdf':True,'tensorboard_roundtrip':True,'mlflow_roundtrip':True,'wandb':'offline_artifact','result_sha256':result['result_sha256']},indent=2)+'\n')
print('Python SDK, PDF/HTML, standard projections, MLflow/TensorBoard round trips and W&B offline passed:',output)
