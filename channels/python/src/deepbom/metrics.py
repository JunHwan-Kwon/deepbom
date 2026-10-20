"""Optional task measurements with explicit inputs, independent metric state and scope."""
from importlib.metadata import version
import math


def evaluate(prediction, target, *, metric, sample_ids, options, missing_policy="reject",
             max_input_bytes=64*1024*1024, backend="wasm"):
    """Explicit task observations. WASM is built in; TorchMetrics is an opt-in oracle.

    NumPy, eager TensorFlow and PyTorch tensors use one common CPU calculator.
    No gradients, patient labels or task semantics are inferred.
    """
    if backend=="torchmetrics":
        return _evaluate_torchmetrics(prediction,target,metric=metric,sample_ids=sample_ids,
                                     options=options,missing_policy=missing_policy,max_input_bytes=max_input_bytes)
    if backend!="wasm": raise ValueError("backend must be wasm or torchmetrics")
    from ._native.core import tensor_row, engine
    import numpy as np
    def descriptor(value):
        if hasattr(value,"numel"):
            return tuple(value.shape),value.numel()*value.element_size(),str(value.dtype)
        shape=tuple(value.shape)
        dtype = getattr(value.dtype, "as_numpy_dtype", value.dtype)
        return shape,int(np.prod(shape,dtype=object))*np.dtype(dtype).itemsize,str(value.dtype)
    ps,pb,pd=descriptor(prediction);ts,tb,td=descriptor(target)
    if ps!=ts or len(ps)<3 or not ps[0]: raise ValueError("Explicit nonempty prediction/target sample shapes must match")
    if type(max_input_bytes) is not int or max_input_bytes<0 or pb+tb>max_input_bytes:
        raise ValueError("Metric input copy budget exceeded")
    if not isinstance(sample_ids,(list,tuple)) or len(sample_ids)!=ps[0] or any(not isinstance(s,str) or not s or len(s)>256 for s in sample_ids) or len(set(sample_ids))!=len(sample_ids):
        raise ValueError("Provide unique bounded sample_ids")
    rows=[{"sample_id":identifier,"prediction":tensor_row(prediction[i:i+1],"prediction"),
           "target":tensor_row(target[i:i+1],"target")} for i,identifier in enumerate(sample_ids)]
    return engine("metric",{"metric":metric,"samples":rows,"options":options,"missing_policy":missing_policy,
                            "shape":list(ps),"prediction_dtype":pd,"target_dtype":td})


def _evaluate_torchmetrics(prediction, target, *, metric, sample_ids, options,
             missing_policy="reject", max_input_bytes=64 * 1024 * 1024):
    """Measure SSIM or Generalized Dice; never infer patient aggregation or labels.

    `sample_ids` must identify distinct samples, not repeated patient visits.
    Only unmasked, equally weighted sample means are qualified. Prediction and
    target are detached copies; no model is executed and no gradient is changed.
    This supporting observation becomes an Evaluation Record only with explicit
    protocol, model, dataset/cohort, run, code and environment references.
    """
    if version("torchmetrics") != "1.9.0":
        raise RuntimeError("Metric adapter is qualified for torchmetrics==1.9.0")
    import torch
    from ._native.core import content_hash, tensor_row
    if missing_policy not in ("reject", "omit_sample"):
        raise ValueError("missing_policy must be reject or omit_sample")
    if not torch.is_tensor(prediction) or not torch.is_tensor(target) or prediction.shape != target.shape:
        raise ValueError("Prediction and target must be explicit tensors with identical shapes")
    if prediction.ndim < 3 or prediction.shape[0] == 0:
        raise ValueError("A nonempty sample axis and explicit spatial axes are required")
    if not isinstance(sample_ids, (list, tuple)) or len(sample_ids) != prediction.shape[0] or any(not isinstance(x, str) or not x or len(x)>256 for x in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Provide unique bounded sample_ids; repeated-patient aggregation is not inferred")
    if type(max_input_bytes) is not int or max_input_bytes < 0 or sum(t.numel()*t.element_size() for t in (prediction,target)) > max_input_bytes:
        raise ValueError("Metric input copy budget exceeded")
    if not isinstance(options, dict):
        raise TypeError("Metric options must be explicit")
    if metric == "ssim":
        required = {"data_range", "gaussian_kernel", "sigma", "kernel_size", "k1", "k2"}
        if set(options) != required or prediction.ndim not in (4,5) or not prediction.is_floating_point() or not target.is_floating_point():
            raise ValueError("SSIM requires NCHW/NCDHW float tensors and explicit data_range/kernel/sigma/k1/k2")
        if any(type(options[k]) not in (int,float) or not math.isfinite(options[k]) or options[k] <= 0 for k in ("data_range","sigma","k1","k2")) or type(options["gaussian_kernel"]) is not bool or type(options["kernel_size"]) is not int or options["kernel_size"] <= 0 or options["kernel_size"] % 2 != 1:
            raise ValueError("Invalid SSIM numerical settings")
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        create = lambda: StructuralSimilarityIndexMeasure(**options)
    elif metric == "generalized_dice":
        if set(options) != {"num_classes","include_background","weight_type","input_format","empty_class_policy"}:
            raise ValueError("Generalized Dice requires explicit class, background, weight, input and empty-class policies")
        if options["input_format"] != "index" or prediction.is_floating_point() or target.is_floating_point() or options["empty_class_policy"] != "torchmetrics_1.9.0":
            raise ValueError("Qualified Dice inputs are discrete class indices, with the pinned library's empty-class policy")
        if type(options["num_classes"]) is not int or options["num_classes"] < 2 or type(options["include_background"]) is not bool or options["weight_type"] not in ("square","simple","linear"):
            raise ValueError("Invalid Generalized Dice class settings")
        from torchmetrics.segmentation import GeneralizedDiceScore
        kwargs = {key:value for key,value in options.items() if key != "empty_class_policy"}
        create = lambda: GeneralizedDiceScore(**kwargs, per_class=False)
    else:
        raise ValueError("Metrics: ssim, generalized_dice; binary Dice is not an alias")
    # One fresh object per sample prevents prior-run/phase state accumulation.
    configuration = {"schema":"deepbom.metric_configuration.v1", "adapter_method":"1.0.0",
        "producer":{"name":"torchmetrics", "version":version("torchmetrics"), "torch_version":torch.__version__},
        "metric":metric, "options":dict(options), "aggregation":"equal_weight_sample_mean",
        "analysis_unit":"sample", "missing_policy":missing_policy, "mask_policy":"none",
        "device":"cpu", "prediction_dtype":str(prediction.dtype), "target_dtype":str(target.dtype),
        "shape":list(prediction.shape)}
    values, missing, input_refs = [], [], []
    with torch.no_grad():
        for index, identifier in enumerate(sample_ids):
            p = prediction[index:index+1].detach().cpu().clone()
            t = target[index:index+1].detach().cpu().clone()
            input_refs.append({"sample_id":identifier,
                "prediction_sha256":content_hash(tensor_row(p, "prediction")),
                "target_sha256":content_hash(tensor_row(t, "target")),
                "digest_scope":"canonical_native_tensor_transport_row"})
            if not torch.isfinite(p).all() or not torch.isfinite(t).all():
                if missing_policy == "reject":
                    raise ValueError("Metric inputs contain nonfinite values")
                missing.append({"sample_id":identifier,"reason":"nonfinite_input"})
                continue
            if metric == "generalized_dice" and (p.min()<0 or t.min()<0 or p.max()>=options["num_classes"] or t.max()>=options["num_classes"]):
                raise ValueError("Class index is outside the declared class vocabulary")
            calculator = create()
            try:
                calculator.update(p,t)
                value = float(calculator.compute())
            finally:
                calculator.reset()
            if not math.isfinite(value):
                if missing_policy == "reject":
                    raise ValueError("Metric is undefined for an included sample")
                missing.append({"sample_id":identifier,"reason":"undefined_metric"})
            else:
                values.append({"sample_id":identifier,"value":value})
    observation = {"schema":"deepbom.metric_observation.v1", "configuration":configuration,
        "configuration_sha256":content_hash(configuration), "input_refs":input_refs, "status":"assessed" if values else "not_assessed",
        "reason":None if values else "no_assessed_samples", "value":math.fsum(row["value"] / len(values) for row in values) if values else None,
        "denominator":str(len(values)), "missing_count":str(len(missing)), "samples":values, "missing":missing,
        "boundary":"Task measurement on explicitly supplied predictions and targets; no patient-level aggregation, clinical verdict, confidence interval or model execution is inferred."}
    return {**observation,"observation_sha256":content_hash(observation)}


def configuration_document(observation):
    """Return the digest-bearing configuration for the common evidence context."""
    from ._native.core import content_hash
    configuration = observation["configuration"]
    if content_hash(configuration) != observation["configuration_sha256"]:
        raise ValueError("Metric configuration identity mismatch")
    return {**configuration, "configuration_sha256": observation["configuration_sha256"]}


def evaluation_record(observation, template, *, metric_id, documents):
    """Import a measured observation via the existing Evaluation workflow.

    The caller supplies all subject/data/run/protocol references and a matching
    configuration reference. Undefined observations cannot be made into v1 records.
    """
    from ._native.core import content_hash
    from .api import evidence_workflow
    body = {k:v for k,v in observation.items() if k != "observation_sha256"}
    if content_hash(body) != observation.get("observation_sha256") or content_hash(observation["configuration"]) != observation["configuration_sha256"]:
        raise ValueError("Metric observation identity mismatch")
    if observation["status"] != "assessed" or int(observation["denominator"]) <= 0:
        raise ValueError("Evaluation Record v1 requires a finite estimate and positive denominator")
    ref = template.get("configuration", {})
    if ref.get("sha256") != observation["configuration_sha256"] or ref.get("schema") != observation["configuration"]["schema"] or ref.get("kind") != "evidence_document" or ref.get("subject_ref") is not None:
        raise ValueError("Evaluation configuration must identify this metric's exact configuration")
    if "metrics" in template or template.get("origin") not in (None,"externally_reported"):
        raise ValueError("Template must not substitute metric values or a locally-recomputed origin")
    record = {**template, "schema":"deepbom.evaluation_record.v1", "origin":"externally_reported",
        "metrics":[{"id":metric_id,"value":observation["value"],"unit":"1","analysis_unit":"sample",
                    "denominator":observation["denominator"],"missing_count":observation["missing_count"],"resolution":None,"ci":None}]}
    configuration = configuration_document(observation)
    bound_documents = list(documents)
    if not any(d == configuration for d in bound_documents):
        bound_documents.append(configuration)
    return evidence_workflow({"schema":"deepbom.evidence_workflow_request.v1","operation":"evaluation","input":record,"documents":bound_documents})
