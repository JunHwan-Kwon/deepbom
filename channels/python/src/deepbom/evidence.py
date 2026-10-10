"""Shared evidence workflows and explicit logger exports; no model execution."""
import json
import tempfile
from pathlib import Path
from .api import evidence_workflow


def export(result, *, destination, run_id=None, logdir=None):
    """Preserve the full result and project already-computed scalars.

    No computation is repeated in Python. Logger values are secondary views;
    exact denominators and source identities remain in the JSON artifact.
    """
    result = evidence_workflow(result)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    rows = result['document'].get('checks', [])
    with tempfile.TemporaryDirectory(prefix='deepbom-export-') as directory:
        filename = Path(directory) / 'evidence-result.json'
        filename.write_text(payload, encoding='utf-8')
        if destination == 'mlflow':
            if not run_id:
                raise ValueError('An explicit MLflow run_id is required')
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.log_artifact(run_id, str(filename), artifact_path='deepbom/evidence')
            client.set_tag(run_id, 'deepbom.evidence_result_sha256', result['result_sha256'])
        elif destination == 'wandb':
            import wandb
            if wandb.run is None:
                raise ValueError('Start an explicit W&B run first (offline is supported)')
            artifact = wandb.Artifact('deepbom-evidence-'+result['result_sha256'][:16],type='evidence')
            artifact.add_file(str(filename)); wandb.run.log_artifact(artifact)
        elif destination == 'tensorboard':
            if logdir is None:
                raise ValueError('An explicit TensorBoard logdir is required')
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
            from tensorboard.compat.proto.tensor_pb2 import TensorProto
            from tensorboard.compat.proto.types_pb2 import DT_DOUBLE
            from urllib.parse import quote
            output=Path(logdir)/result['result_sha256']; output.mkdir(parents=True,exist_ok=True)
            (output/'evidence-result.json').write_text(payload,encoding='utf-8')
            writer = EventFileWriter(str(output))
            try:
                for row in rows:
                    if 'population' not in row or 'metric' not in row:
                        continue
                    prefix=quote(row['population'],safe='')+'/'+quote(row['metric'],safe='')
                    for key in ('before','after','delta'):
                        value=row.get(key)
                        if isinstance(value,(int,float)) and not isinstance(value,bool):
                            writer.add_event(Event(step=0,summary=Summary(value=[Summary.Value(tag=prefix+'/'+key,tensor=TensorProto(dtype=DT_DOUBLE,double_val=[value]))])))
                writer.flush()
            finally:
                writer.close()
            return {'destination':destination,'result_sha256':result['result_sha256'],'path':str(output),'axis':'single review; step=0, not training time','exact_source':'evidence-result.json'}
        else:
            raise ValueError('Destinations: mlflow, wandb, tensorboard')
    return {'destination':destination,'result_sha256':result['result_sha256'],'projection':'complete JSON artifact; no metrics recalculated'}
