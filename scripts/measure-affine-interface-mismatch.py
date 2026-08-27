import argparse
import csv
import hashlib
import json
import math
import os
import tarfile
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image, ImageOps
from scipy.stats import binomtest


IMAGENETTE_URL = (
    "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
)
IMAGENETTE_SHA256 = (
    "569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c"
)
IMAGENETV2_REVISION = "d626240be2538720e83103a0e1178d24aca8b12c"
IMAGENETV2_URL = (
    "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/"
    f"{IMAGENETV2_REVISION}/imagenetv2-matched-frequency.tar.gz"
)
IMAGENETV2_SHA256 = (
    "f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c"
)
IMAGENET_CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "imagenet_class_index.json"
)
MOBILENET_SHA256 = (
    "f08d447cde49b4e0446428aa921aff0a14ea589fa9c5817b31f83128e9a43c1d"
)
EFFICIENTNET_SHA256 = (
    "bc2ffe19c1118de0c0c2a9088992da5589722656e0fba81421385300a4a34b16"
)
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
PREDICTION_WORKER = None


def sha256_file(filename):
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        url, headers={"User-Agent": "DeepBOM-affine-contract-study/1"}
    )
    with urllib.request.urlopen(request) as response, temporary.open("wb") as out:
        expected = int(response.headers.get("Content-Length", "0"))
        copied = 0
        next_notice = 64 * 1024 * 1024
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            if copied >= next_notice:
                total = f"/{expected}" if expected else ""
                print(f"downloaded {copied}{total} bytes", flush=True)
                next_notice += 64 * 1024 * 1024
    temporary.replace(destination)


def safe_extract_tar(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"Archive link is not allowed: {member.name}")
            target = (root / member.name).resolve()
            if os.path.commonpath([root, target]) != str(root):
                raise ValueError(f"Archive path escapes destination: {member.name}")
        bundle.extractall(root, filter="data")


def prepare_dataset(cache_root, dataset):
    dataset_dir = cache_root / "datasets"
    if dataset == "imagenette":
        archive = dataset_dir / "imagenette2-320.tgz"
        extracted = dataset_dir / "imagenette2-320"
        validation = extracted / "val"
        labels = dataset_dir / "imagenet_class_index.json"
        upstream_url = IMAGENETTE_URL
        expected_sha256 = IMAGENETTE_SHA256
        name = "Imagenette 320"
        revision = None
    elif dataset == "imagenetv2":
        archive = dataset_dir / "imagenetv2-matched-frequency.tar.gz"
        extracted = (
            dataset_dir / "imagenetv2-matched-frequency-format-val"
        )
        validation = extracted
        labels = None
        upstream_url = IMAGENETV2_URL
        expected_sha256 = IMAGENETV2_SHA256
        name = "ImageNetV2 MatchedFrequency"
        revision = IMAGENETV2_REVISION
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not archive.exists():
        download_file(upstream_url, archive)
    archive_sha256 = sha256_file(archive)
    if archive_sha256 != expected_sha256:
        raise ValueError(
            f"{name} archive SHA-256 mismatch: "
            f"{archive_sha256} != {expected_sha256}"
        )
    if not validation.is_dir():
        safe_extract_tar(archive, dataset_dir)
    if labels is not None and not labels.exists():
        download_file(IMAGENET_CLASS_INDEX_URL, labels)
    return {
        "id": dataset,
        "name": name,
        "revision": revision,
        "upstream_url": upstream_url,
        "archive": archive,
        "archive_sha256": archive_sha256,
        "root": extracted,
        "validation": validation,
        "labels": labels,
    }


def load_label_map(filename):
    source = json.loads(filename.read_text(encoding="utf-8"))
    result = {}
    for index, value in source.items():
        wnid = value[0]
        result[wnid] = int(index)
    return result


def selection_key(seed, value):
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def list_validation_images(
    dataset_root,
    validation,
    label_map,
    max_images,
    seed,
):
    rows_by_class = {}
    for class_dir in sorted(path for path in validation.iterdir() if path.is_dir()):
        if label_map is None:
            if not class_dir.name.isdigit():
                raise ValueError(
                    f"Expected numeric ImageNetV2 class: {class_dir.name}"
                )
            label = int(class_dir.name)
        else:
            if class_dir.name not in label_map:
                raise ValueError(f"Unknown ImageNet WNID: {class_dir.name}")
            label = label_map[class_dir.name]
        rows = []
        for filename in class_dir.rglob("*"):
            if filename.is_file() and filename.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({
                    "path": filename,
                    "relative_path": filename.relative_to(dataset_root).as_posix(),
                    "wnid": class_dir.name,
                    "label": label,
                })
        rows_by_class[class_dir.name] = sorted(
            rows, key=lambda row: row["relative_path"]
        )
    if not max_images or max_images >= sum(map(len, rows_by_class.values())):
        return [
            row
            for wnid in sorted(rows_by_class)
            for row in rows_by_class[wnid]
        ]
    class_count = len(rows_by_class)
    base = max_images // class_count
    remainder = max_images % class_count
    selected = []
    for class_ordinal, wnid in enumerate(sorted(rows_by_class)):
        count = base + (1 if class_ordinal < remainder else 0)
        ordered = sorted(
            rows_by_class[wnid],
            key=lambda row: selection_key(seed, row["relative_path"]),
        )
        selected.extend(ordered[:count])
    return sorted(selected, key=lambda row: row["relative_path"])


def endpoint_sample(images, per_class, seed):
    grouped = defaultdict(list)
    for row in images:
        grouped[row["wnid"]].append(row)
    selected = []
    for wnid in sorted(grouped):
        ordered = sorted(
            grouped[wnid],
            key=lambda row: selection_key(seed + 1, row["relative_path"]),
        )
        selected.extend(ordered[:per_class])
    return sorted(selected, key=lambda row: row["relative_path"])


def resize_center_crop_rgb(filename, resize_short=256, crop_size=224):
    with Image.open(filename) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        scale = resize_short / min(width, height)
        resized_width = max(crop_size, int(round(width * scale)))
        resized_height = max(crop_size, int(round(height * scale)))
        image = image.resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        )
        left = (resized_width - crop_size) // 2
        top = (resized_height - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))
        return np.asarray(image, dtype=np.uint8)[np.newaxis, ...]


def scalar_affine(detail):
    quant = detail.get("quantization_parameters") or {}
    scales = np.asarray(quant.get("scales", []), dtype=np.float64)
    zero_points = np.asarray(quant.get("zero_points", []), dtype=np.int64)
    if scales.size != 1 or zero_points.size != 1:
        raise ValueError(
            f"Expected scalar affine contract for {detail.get('name')}: "
            f"{scales.size} scales, {zero_points.size} zero-points"
        )
    dtype = np.dtype(detail["dtype"])
    if dtype.kind not in ("i", "u"):
        raise ValueError(f"Expected integer tensor, found {dtype}")
    limits = np.iinfo(dtype)
    return {
        "dtype": dtype.name.upper(),
        "scale": float(scales[0]),
        "zero_point": int(zero_points[0]),
        "qmin": int(limits.min),
        "qmax": int(limits.max),
        "quantized_dimension": int(quant.get("quantized_dimension", 0)),
    }


def reencode_with_contract(codes, source_contract, encoder_contract):
    real = (
        codes.astype(np.float64) - source_contract["zero_point"]
    ) * source_contract["scale"]
    unsaturated = np.rint(
        real / encoder_contract["scale"] + encoder_contract["zero_point"]
    ).astype(np.int64)
    low = unsaturated < encoder_contract["qmin"]
    high = unsaturated > encoder_contract["qmax"]
    clipped = np.clip(
        unsaturated, encoder_contract["qmin"], encoder_contract["qmax"]
    )
    dtype = np.dtype(encoder_contract["dtype"].lower())
    return clipped.astype(dtype), int(low.sum()), int(high.sum())


def dequantize(codes, contract):
    return (
        codes.astype(np.float64) - contract["zero_point"]
    ) * contract["scale"]


def stable_softmax(values):
    shifted = values - np.max(values)
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def output_probabilities(values, mode):
    if mode == "softmax_logits":
        return stable_softmax(values)
    if mode == "serialized_probabilities":
        clipped = np.clip(values, 0.0, None)
        total = clipped.sum()
        return clipped / total if total > 0 else stable_softmax(values)
    raise ValueError(f"Unknown output interpretation: {mode}")


def js_divergence(left, right):
    midpoint = (left + right) / 2.0
    left_term = np.zeros_like(left)
    right_term = np.zeros_like(right)
    left_mask = left > 0
    right_mask = right > 0
    left_term[left_mask] = (
        left[left_mask] * np.log(left[left_mask] / midpoint[left_mask])
    )
    right_term[right_mask] = (
        right[right_mask] * np.log(right[right_mask] / midpoint[right_mask])
    )
    return float((left_term.sum() + right_term.sum()) / 2.0)


def top_k(values, count):
    count = min(count, values.size)
    return np.argpartition(values, -count)[-count:]


def ece(confidences, correctness, bins=15):
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for ordinal in range(bins):
        lower = edges[ordinal]
        upper = edges[ordinal + 1]
        if ordinal == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if mask.any():
            result += float(mask.mean()) * abs(
                float(correctness[mask].mean())
                - float(confidences[mask].mean())
            )
    return result


def bootstrap_paired_delta(correct, wrong, iterations, seed):
    rng = np.random.default_rng(seed)
    differences = wrong.astype(np.float64) - correct.astype(np.float64)
    estimates = np.empty(iterations, dtype=np.float64)
    for ordinal in range(iterations):
        sample = rng.integers(0, differences.size, size=differences.size)
        estimates[ordinal] = differences[sample].mean()
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "point": float(differences.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "iterations": iterations,
        "seed": seed,
    }


def paired_metrics(rows, bootstrap_iterations, seed):
    correct_top1 = np.asarray(
        [row["correct_top1_correct"] for row in rows], dtype=bool
    )
    wrong_top1 = np.asarray(
        [row["wrong_top1_correct"] for row in rows], dtype=bool
    )
    correct_top5 = np.asarray(
        [row["correct_top5_correct"] for row in rows], dtype=bool
    )
    wrong_top5 = np.asarray(
        [row["wrong_top5_correct"] for row in rows], dtype=bool
    )
    correct_confidence = np.asarray(
        [row["correct_confidence"] for row in rows], dtype=np.float64
    )
    wrong_confidence = np.asarray(
        [row["wrong_confidence"] for row in rows], dtype=np.float64
    )
    lost = int(np.sum(correct_top1 & ~wrong_top1))
    gained = int(np.sum(~correct_top1 & wrong_top1))
    discordant = lost + gained
    mcnemar_p = (
        float(binomtest(min(lost, gained), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    total_input = sum(row["input_element_count"] for row in rows)
    low_clipped = sum(row["wrong_input_low_clip_count"] for row in rows)
    high_clipped = sum(row["wrong_input_high_clip_count"] for row in rows)
    return {
        "image_count": len(rows),
        "correct_top1_accuracy": float(correct_top1.mean()),
        "wrong_top1_accuracy": float(wrong_top1.mean()),
        "paired_top1_accuracy_delta": bootstrap_paired_delta(
            correct_top1, wrong_top1, bootstrap_iterations, seed
        ),
        "correct_top5_accuracy": float(correct_top5.mean()),
        "wrong_top5_accuracy": float(wrong_top5.mean()),
        "paired_top5_accuracy_delta": bootstrap_paired_delta(
            correct_top5, wrong_top5, bootstrap_iterations, seed + 1
        ),
        "mcnemar_exact": {
            "correct_to_incorrect": lost,
            "incorrect_to_correct": gained,
            "discordant_pair_count": discordant,
            "two_sided_p_value": mcnemar_p,
        },
        "top1_prediction_agreement": float(np.mean([
            row["correct_prediction"] == row["wrong_prediction"]
            for row in rows
        ])),
        "mean_js_divergence": float(np.mean([
            row["js_divergence"] for row in rows
        ])),
        "median_js_divergence": float(np.median([
            row["js_divergence"] for row in rows
        ])),
        "p95_js_divergence": float(np.quantile([
            row["js_divergence"] for row in rows
        ], 0.95)),
        "correct_mean_confidence": float(correct_confidence.mean()),
        "wrong_mean_confidence": float(wrong_confidence.mean()),
        "correct_ece_15_bin": ece(correct_confidence, correct_top1),
        "wrong_ece_15_bin": ece(wrong_confidence, wrong_top1),
        "identity_reencode_mismatch_count": int(sum(
            not row["identity_reencode_exact"] for row in rows
        )),
        "wrong_input_low_clip_count": low_clipped,
        "wrong_input_high_clip_count": high_clipped,
        "wrong_input_clip_fraction": (
            (low_clipped + high_clipped) / total_input if total_input else 0.0
        ),
    }


def class_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["wnid"]].append(row)
    result = []
    for wnid in sorted(grouped):
        members = grouped[wnid]
        result.append({
            "wnid": wnid,
            "imagenet_index": members[0]["label"],
            "image_count": len(members),
            "correct_top1_accuracy": float(np.mean([
                row["correct_top1_correct"] for row in members
            ])),
            "wrong_top1_accuracy": float(np.mean([
                row["wrong_top1_correct"] for row in members
            ])),
            "top1_prediction_agreement": float(np.mean([
                row["correct_prediction"] == row["wrong_prediction"]
                for row in members
            ])),
        })
    return result


def invoke(interpreter, input_index, output_index, values):
    interpreter.set_tensor(input_index, values)
    interpreter.invoke()
    return interpreter.get_tensor(output_index).reshape(-1).copy()


def prediction_worker_initializer(config, wrong_contract):
    global PREDICTION_WORKER
    descriptor = model_descriptor(
        config["id"],
        Path(config["filename"]),
        config["artifact_sha256"],
        config["output_label_offset"],
        config["output_interpretation"],
        1,
    )
    PREDICTION_WORKER = (descriptor, wrong_contract)


def measure_one_prediction(descriptor, wrong_contract, image_row):
    interpreter = descriptor["interpreter"]
    input_detail = descriptor["input_detail"]
    output_detail = descriptor["output_detail"]
    target_contract = descriptor["input_contract"]
    output_contract = descriptor["output_contract"]
    output_offset = descriptor["output_label_offset"]
    correct_input = resize_center_crop_rgb(image_row["path"])
    identity_input, identity_low, identity_high = reencode_with_contract(
        correct_input, target_contract, target_contract
    )
    wrong_input, wrong_low, wrong_high = reencode_with_contract(
        correct_input, target_contract, wrong_contract
    )
    correct_output = invoke(
        interpreter,
        int(input_detail["index"]),
        int(output_detail["index"]),
        correct_input,
    )
    wrong_output = invoke(
        interpreter,
        int(input_detail["index"]),
        int(output_detail["index"]),
        wrong_input,
    )
    correct_real = dequantize(correct_output, output_contract)
    wrong_real = dequantize(wrong_output, output_contract)
    correct_probabilities = output_probabilities(
        correct_real, descriptor["output_interpretation"]
    )
    wrong_probabilities = output_probabilities(
        wrong_real, descriptor["output_interpretation"]
    )
    correct_prediction = int(np.argmax(correct_output)) - output_offset
    wrong_prediction = int(np.argmax(wrong_output)) - output_offset
    correct_top5 = set(
        (top_k(correct_output, 5) - output_offset).tolist()
    )
    wrong_top5 = set((top_k(wrong_output, 5) - output_offset).tolist())
    label = image_row["label"]
    return {
        "model_id": descriptor["id"],
        "relative_path": image_row["relative_path"],
        "wnid": image_row["wnid"],
        "label": label,
        "correct_prediction": correct_prediction,
        "wrong_prediction": wrong_prediction,
        "correct_top1_correct": correct_prediction == label,
        "wrong_top1_correct": wrong_prediction == label,
        "correct_top5_correct": label in correct_top5,
        "wrong_top5_correct": label in wrong_top5,
        "correct_confidence": float(
            correct_probabilities[int(np.argmax(correct_output))]
        ),
        "wrong_confidence": float(
            wrong_probabilities[int(np.argmax(wrong_output))]
        ),
        "js_divergence": js_divergence(
            correct_probabilities, wrong_probabilities
        ),
        "identity_reencode_exact": bool(
            identity_low == 0
            and identity_high == 0
            and np.array_equal(identity_input, correct_input)
        ),
        "input_element_count": int(correct_input.size),
        "wrong_input_low_clip_count": wrong_low,
        "wrong_input_high_clip_count": wrong_high,
        "_correct_histogram": np.bincount(
            correct_input.reshape(-1), minlength=256
        ).astype(np.int32).tobytes(),
        "_wrong_histogram": np.bincount(
            wrong_input.reshape(-1), minlength=256
        ).astype(np.int32).tobytes(),
    }


def prediction_worker(image_row):
    if PREDICTION_WORKER is None:
        raise RuntimeError("Prediction worker is not initialized")
    descriptor, wrong_contract = PREDICTION_WORKER
    return measure_one_prediction(descriptor, wrong_contract, image_row)


def model_descriptor(
    model_id,
    filename,
    expected_sha256,
    output_offset,
    mode,
    num_threads,
):
    actual_sha256 = sha256_file(filename)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{model_id} SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    interpreter = Interpreter(
        model_path=str(filename), num_threads=num_threads
    )
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"{model_id} must have one input and one output")
    input_detail = inputs[0]
    output_detail = outputs[0]
    if list(input_detail["shape"]) != [1, 224, 224, 3]:
        raise ValueError(f"Unexpected {model_id} input shape")
    return {
        "id": model_id,
        "filename": filename,
        "artifact_sha256": actual_sha256,
        "artifact_size_bytes": filename.stat().st_size,
        "output_label_offset": output_offset,
        "output_interpretation": mode,
        "interpreter": interpreter,
        "input_detail": input_detail,
        "output_detail": output_detail,
        "input_contract": scalar_affine(input_detail),
        "output_contract": scalar_affine(output_detail),
    }


def measure_predictions(
    descriptor,
    wrong_contract,
    images,
    bootstrap_iterations,
    seed,
    workers,
):
    rows = []
    correct_histogram = np.zeros(256, dtype=np.int64)
    wrong_histogram = np.zeros(256, dtype=np.int64)
    started = time.perf_counter()
    config = {
        "id": descriptor["id"],
        "filename": str(descriptor["filename"]),
        "artifact_sha256": descriptor["artifact_sha256"],
        "output_label_offset": descriptor["output_label_offset"],
        "output_interpretation": descriptor["output_interpretation"],
    }
    if workers == 1:
        iterator = (
            measure_one_prediction(descriptor, wrong_contract, image_row)
            for image_row in images
        )
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=prediction_worker_initializer,
            initargs=(config, wrong_contract),
        )
        iterator = executor.map(prediction_worker, images, chunksize=8)
    try:
        for ordinal, row in enumerate(iterator):
            correct_histogram += np.frombuffer(
                row.pop("_correct_histogram"), dtype=np.int32
            )
            wrong_histogram += np.frombuffer(
                row.pop("_wrong_histogram"), dtype=np.int32
            )
            rows.append(row)
            if (ordinal + 1) % 250 == 0 or ordinal + 1 == len(images):
                print(
                    f"{descriptor['id']}: {ordinal + 1}/{len(images)} images",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    metrics = paired_metrics(rows, bootstrap_iterations, seed)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics["correct_input_distinct_code_count"] = int(
        np.count_nonzero(correct_histogram)
    )
    metrics["wrong_input_distinct_code_count"] = int(
        np.count_nonzero(wrong_histogram)
    )
    metrics["correct_input_histogram"] = correct_histogram.tolist()
    metrics["wrong_input_histogram"] = wrong_histogram.tolist()
    metrics["per_class"] = class_metrics(rows)
    return rows, metrics


def endpoint_tensor_descriptors(interpreter):
    details = {
        int(detail["index"]): detail
        for detail in interpreter.get_tensor_details()
    }
    input_indices = {
        int(detail["index"]) for detail in interpreter.get_input_details()
    }
    output_indices = {
        int(detail["index"]) for detail in interpreter.get_output_details()
    }
    operation_outputs = {
        int(index)
        for operation in interpreter._get_ops_details()
        for index in operation["outputs"]
        if int(index) >= 0
    }
    selected = sorted(input_indices | output_indices | operation_outputs)
    rows = []
    for index in selected:
        detail = details.get(index)
        if detail is None:
            continue
        dtype = np.dtype(detail["dtype"])
        quant = detail.get("quantization_parameters") or {}
        scales = np.asarray(quant.get("scales", []))
        zero_points = np.asarray(quant.get("zero_points", []))
        if dtype.kind not in ("i", "u") or scales.size != 1:
            continue
        limits = np.iinfo(dtype)
        role = (
            "input" if index in input_indices
            else "output" if index in output_indices
            else "intermediate"
        )
        rows.append({
            "tensor_index": index,
            "tensor_name": str(detail["name"]),
            "shape": [int(value) for value in detail["shape"].tolist()],
            "dtype": dtype.name.upper(),
            "qmin": int(limits.min),
            "qmax": int(limits.max),
            "zero_point": int(zero_points[0]) if zero_points.size == 1 else None,
            "scale": float(scales[0]),
            "role": role,
        })
    return rows


def measure_endpoint_occupancy(
    descriptor,
    wrong_contract,
    images,
    num_threads,
):
    interpreter = Interpreter(
        model_path=str(descriptor["filename"]),
        num_threads=num_threads,
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    tensors = endpoint_tensor_descriptors(interpreter)
    accumulators = {
        tensor["tensor_index"]: {
            **tensor,
            "correct_elements": 0,
            "correct_lower": 0,
            "correct_upper": 0,
            "correct_zero": 0,
            "wrong_elements": 0,
            "wrong_lower": 0,
            "wrong_upper": 0,
            "wrong_zero": 0,
            "unreadable_count": 0,
        }
        for tensor in tensors
    }
    for image_row in images:
        correct_input = resize_center_crop_rgb(image_row["path"])
        wrong_input, _, _ = reencode_with_contract(
            correct_input, descriptor["input_contract"], wrong_contract
        )
        for prefix, values in (
            ("correct", correct_input),
            ("wrong", wrong_input),
        ):
            interpreter.set_tensor(int(input_detail["index"]), values)
            interpreter.invoke()
            for tensor in tensors:
                row = accumulators[tensor["tensor_index"]]
                try:
                    observed = interpreter.get_tensor(
                        tensor["tensor_index"]
                    ).reshape(-1)
                except (ValueError, RuntimeError):
                    row["unreadable_count"] += 1
                    continue
                row[f"{prefix}_elements"] += int(observed.size)
                row[f"{prefix}_lower"] += int(
                    np.count_nonzero(observed == tensor["qmin"])
                )
                row[f"{prefix}_upper"] += int(
                    np.count_nonzero(observed == tensor["qmax"])
                )
                if tensor["zero_point"] is not None:
                    row[f"{prefix}_zero"] += int(
                        np.count_nonzero(observed == tensor["zero_point"])
                    )
    rows = []
    for row in accumulators.values():
        correct_elements = row["correct_elements"]
        wrong_elements = row["wrong_elements"]
        if not correct_elements or not wrong_elements:
            continue
        for side in ("lower", "upper", "zero"):
            row[f"correct_{side}_fraction"] = (
                row[f"correct_{side}"] / correct_elements
            )
            row[f"wrong_{side}_fraction"] = (
                row[f"wrong_{side}"] / wrong_elements
            )
            row[f"{side}_delta_fraction"] = (
                row[f"wrong_{side}_fraction"]
                - row[f"correct_{side}_fraction"]
            )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -abs(row["upper_delta_fraction"]),
            -abs(row["lower_delta_fraction"]),
            row["tensor_index"],
        )
    )
    aggregate = {}
    for prefix in ("correct", "wrong"):
        elements = sum(row[f"{prefix}_elements"] for row in rows)
        aggregate[f"{prefix}_elements"] = elements
        for side in ("lower", "upper", "zero"):
            count = sum(row[f"{prefix}_{side}"] for row in rows)
            aggregate[f"{prefix}_{side}_count"] = count
            aggregate[f"{prefix}_{side}_fraction"] = (
                count / elements if elements else None
            )
    aggregate["upper_delta_fraction"] = (
        aggregate["wrong_upper_fraction"]
        - aggregate["correct_upper_fraction"]
    )
    aggregate["lower_delta_fraction"] = (
        aggregate["wrong_lower_fraction"]
        - aggregate["correct_lower_fraction"]
    )
    return {
        "sample_image_count": len(images),
        "assessed_tensor_count": len(rows),
        "measurement_name": "integer_endpoint_occupancy",
        "interpretation_boundary": (
            "Endpoint occupancy is observed exactly, but it is not an exact "
            "clamp-event count. Natural zeros or rounded endpoint values can "
            "occupy qmin/qmax without a pre-clamp overflow."
        ),
        "aggregate": aggregate,
        "top_tensors_by_endpoint_delta": rows[:20],
    }, rows


def verify_repeat(descriptor, wrong_contract, images):
    interpreter = descriptor["interpreter"]
    input_index = int(descriptor["input_detail"]["index"])
    output_index = int(descriptor["output_detail"]["index"])
    checked = []
    for image_row in images:
        correct_input = resize_center_crop_rgb(image_row["path"])
        wrong_input, _, _ = reencode_with_contract(
            correct_input, descriptor["input_contract"], wrong_contract
        )
        for mode, values in (("correct", correct_input), ("wrong", wrong_input)):
            first = invoke(interpreter, input_index, output_index, values)
            second = invoke(interpreter, input_index, output_index, values)
            if not np.array_equal(first, second):
                raise ValueError(
                    f"Non-deterministic output for {descriptor['id']} "
                    f"{image_row['relative_path']} {mode}"
                )
            checked.append({
                "relative_path": image_row["relative_path"],
                "mode": mode,
                "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            })
    return {
        "image_count": len(images),
        "invocation_pair_count": len(checked),
        "exact_repeat_match": True,
        "ledger_sha256": sha256_json(checked),
    }


def write_csv(filename, rows, fieldnames):
    with filename.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row[field], separators=(",", ":"))
                if isinstance(row.get(field), (list, dict))
                else row.get(field)
                for field in fieldnames
            })


def percentage(value):
    return f"{100.0 * value:.2f}%"


def write_summary(filename, result):
    lines = [
        "# Affine Interface Mismatch Measurement",
        "",
        (
            f"Dataset: {result['dataset']['name']}, "
            f"{result['dataset']['image_count']} images, "
            f"{result['dataset']['class_count']} ImageNet classes."
        ),
        "",
        "| Target model | Correct top-1 | Wrong top-1 | Paired delta | "
        "95% CI | Prediction agreement | Input clip | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in result["models"]:
        metrics = model["metrics"]
        delta = metrics["paired_top1_accuracy_delta"]
        lines.append(
            f"| {model['id']} | "
            f"{percentage(metrics['correct_top1_accuracy'])} | "
            f"{percentage(metrics['wrong_top1_accuracy'])} | "
            f"{percentage(delta['point'])} | "
            f"[{percentage(delta['lower_95'])}, "
            f"{percentage(delta['upper_95'])}] | "
            f"{percentage(metrics['top1_prediction_agreement'])} | "
            f"{percentage(metrics['wrong_input_clip_fraction'])} | "
            f"{metrics['mcnemar_exact']['two_sided_p_value']:.4g} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        (
            "This is a controlled contract-fault injection. For each target "
            "model, the decoded RGB byte tensor is treated as the correct "
            "integer input. The same target-real tensor is then re-encoded "
            "with the other model's affine contract. Architecture, weights, "
            "images, crop, and runtime remain fixed."
        ),
        "",
        (
            "Input clipping is counted exactly before invocation. Internal "
            "qmin/qmax occupancy is an observed proxy and is not reported as "
            "an exact clamp-event count."
        ),
        "",
        "## Limits",
        "",
        (
            f"{result['dataset']['scope_limit']} This experiment estimates "
            "the effect of the injected contract mismatch on these two "
            "hash-pinned artifacts; it does not estimate mismatch prevalence "
            "or establish a universal accuracy effect."
        ),
        "",
        f"Result ledger SHA-256: `{result['ledger_sha256']}`",
        "",
    ])
    filename.write_text("\n".join(lines), encoding="utf-8")


def find_default_models(cache_root):
    return {
        "mobilenet": (
            cache_root
            / "google-legacy-corpus-v1"
            / "models"
            / "mobilenet-v2-1.0-224-quant"
            / "mobilenet_v2_1.0_224_quant.tflite"
        ),
        "efficientnet": (
            cache_root
            / "public-model-corpus-v1"
            / "image_classifier__efficientnet_lite0__int8__1__"
            "efficientnet_lite0.tflite"
        ),
    }


def main():
    default_cache = Path(
        os.environ.get("LOCALAPPDATA", str(Path.home()))
    ) / "DeepBOM"
    parser = argparse.ArgumentParser(
        description=(
            "Measure a paired affine input-contract mismatch on two "
            "hash-pinned public TFLite classifiers."
        )
    )
    parser.add_argument("--cache-root", default=str(default_cache))
    parser.add_argument("--mobilenet-model")
    parser.add_argument("--efficientnet-model")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset",
        choices=("imagenette", "imagenetv2"),
        default="imagenette",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--endpoint-images-per-class", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    cache_root = Path(args.cache_root).resolve()
    defaults = find_default_models(cache_root)
    mobilenet_path = Path(
        args.mobilenet_model or defaults["mobilenet"]
    ).resolve()
    efficientnet_path = Path(
        args.efficientnet_model or defaults["efficientnet"]
    ).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_dataset(cache_root, args.dataset)
    label_file = dataset["labels"]
    label_map = load_label_map(label_file) if label_file else None
    images = list_validation_images(
        dataset["root"],
        dataset["validation"],
        label_map,
        args.max_images,
        args.seed,
    )
    endpoint_images = endpoint_sample(
        images, args.endpoint_images_per_class, args.seed
    )
    if not images:
        raise ValueError("No validation images were selected")

    mobilenet = model_descriptor(
        "google-mobilenet-v2-1.0-224-quant",
        mobilenet_path,
        MOBILENET_SHA256,
        1,
        "softmax_logits",
        1,
    )
    efficientnet = model_descriptor(
        "mediapipe-efficientnet-lite0-int8",
        efficientnet_path,
        EFFICIENTNET_SHA256,
        0,
        "serialized_probabilities",
        1,
    )
    pairs = [
        (mobilenet, efficientnet["input_contract"]),
        (efficientnet, mobilenet["input_contract"]),
    ]

    all_prediction_rows = []
    all_endpoint_rows = []
    model_results = []
    repeat_source = endpoint_images if endpoint_images else images
    repeat_images = repeat_source[: min(5, len(repeat_source))]
    for model_ordinal, (descriptor, wrong_contract) in enumerate(pairs):
        print(f"measuring {descriptor['id']}", flush=True)
        prediction_rows, metrics = measure_predictions(
            descriptor,
            wrong_contract,
            images,
            args.bootstrap_iterations,
            args.seed + model_ordinal * 100,
            args.workers,
        )
        if endpoint_images:
            endpoint_result, endpoint_rows = measure_endpoint_occupancy(
                descriptor,
                wrong_contract,
                endpoint_images,
                1,
            )
        else:
            endpoint_result = {
                "status": "not_run",
                "reason": "--endpoint-images-per-class was 0",
            }
            endpoint_rows = []
        repeat = verify_repeat(descriptor, wrong_contract, repeat_images)
        all_prediction_rows.extend(prediction_rows)
        for row in endpoint_rows:
            all_endpoint_rows.append({
                "model_id": descriptor["id"],
                **row,
            })
        model_results.append({
            "id": descriptor["id"],
            "artifact_sha256": descriptor["artifact_sha256"],
            "artifact_size_bytes": descriptor["artifact_size_bytes"],
            "input_contract": descriptor["input_contract"],
            "wrong_encoder_contract": wrong_contract,
            "output_contract": descriptor["output_contract"],
            "output_label_offset": descriptor["output_label_offset"],
            "output_interpretation": descriptor["output_interpretation"],
            "fault_injection": (
                "target-real input reconstructed from correct integer codes, "
                "then re-encoded with the other model's affine contract using "
                "nearest-even rounding and integer saturation"
            ),
            "metrics": metrics,
            "endpoint_occupancy": endpoint_result,
            "determinism_repeat": repeat,
        })

    predictions_csv = output_dir / "predictions.csv"
    endpoint_csv = output_dir / "endpoint-occupancy.csv"
    prediction_fields = list(all_prediction_rows[0].keys())
    write_csv(predictions_csv, all_prediction_rows, prediction_fields)
    if all_endpoint_rows:
        endpoint_fields = list(all_endpoint_rows[0].keys())
        write_csv(endpoint_csv, all_endpoint_rows, endpoint_fields)

    image_ledger = [{
        "relative_path": row["relative_path"],
        "wnid": row["wnid"],
        "label": row["label"],
    } for row in images]
    endpoint_ledger = [row["relative_path"] for row in endpoint_images]
    result = {
        "schema": "deepbom.affine_interface_mismatch_measurement.v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "script": Path(__file__).name,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": os.sys.version,
            "ai_edge_litert": version("ai-edge-litert"),
            "numpy": version("numpy"),
            "pillow": version("Pillow"),
            "scipy": version("scipy"),
            "prediction_workers": args.workers,
            "interpreter_threads_per_worker": 1,
        },
        "dataset": {
            "id": dataset["id"],
            "name": dataset["name"],
            "upstream_url": dataset["upstream_url"],
            "source_revision": dataset["revision"],
            "archive_sha256": dataset["archive_sha256"],
            "label_index_url": (
                IMAGENET_CLASS_INDEX_URL if label_file else None
            ),
            "label_index_sha256": (
                sha256_file(label_file) if label_file else None
            ),
            "split": (
                "val" if args.dataset == "imagenette"
                else "matched-frequency"
            ),
            "image_count": len(images),
            "class_count": len(set(row["wnid"] for row in images)),
            "selection_seed": args.seed,
            "max_images": args.max_images or None,
            "image_ledger_sha256": sha256_json(image_ledger),
            "endpoint_sample_image_count": len(endpoint_images),
            "endpoint_sample_per_class": args.endpoint_images_per_class,
            "endpoint_sample_ledger_sha256": sha256_json(endpoint_ledger),
            "scope_limit": (
                "Imagenette is a curated 10-class ImageNet subset and may "
                "overlap the source images used to train ImageNet-pretrained "
                "models; its paired result is not a generalization benchmark."
                if args.dataset == "imagenette"
                else
                "ImageNetV2 is an independently collected test set, but this "
                "run may use a deterministic subset rather than all 10,000 "
                "MatchedFrequency images."
            ),
        },
        "preprocessing": {
            "decode": "Pillow RGB with EXIF transpose",
            "resize": (
                "bilinear; shortest side to 256; dimensions rounded to nearest "
                "integer"
            ),
            "crop": "center 224x224 using integer floor offset",
            "correct_integer_input": "decoded/cropped RGB uint8 byte tensor",
        },
        "protocol": {
            "paired": True,
            "fixed_between_conditions": [
                "target artifact",
                "runtime",
                "image",
                "decode",
                "resize",
                "crop",
                "target-real tensor before affine encoding",
            ],
            "changed_variable": "affine encoder scale and zero-point",
            "rounding": "IEEE nearest-even via numpy.rint",
            "integer_saturation": True,
            "accuracy_test": (
                "paired top-1/top-5 with bootstrap confidence intervals and "
                "two-sided exact McNemar test"
            ),
            "internal_tensor_measurement": (
                "integer endpoint occupancy; explicitly not interpreted as "
                "exact clamp-event count"
            ),
        },
        "models": model_results,
        "outputs": {
            "predictions_csv": predictions_csv.name,
            "predictions_csv_sha256": sha256_file(predictions_csv),
            "endpoint_occupancy_csv": endpoint_csv.name,
            "endpoint_occupancy_csv_sha256": (
                sha256_file(endpoint_csv) if endpoint_csv.exists() else None
            ),
        },
        "interpretation_boundary": [
            (
                "The experiment identifies the paired effect of an injected "
                "affine encoder mismatch on the selected artifacts and data."
            ),
            (
                "It does not prove that a deployed harness is currently "
                "mismatched or estimate the prevalence of such mismatches."
            ),
            (
                "Dataset scope and sampling are recorded explicitly; subset "
                "results must not be presented as full-dataset accuracy."
            ),
            (
                "Internal endpoint occupancy is not an exact clamp-event "
                "counter because natural or rounded endpoint values are "
                "observationally indistinguishable from clipped values."
            ),
        ],
    }
    result["ledger_sha256"] = sha256_json(result)
    result_path = output_dir / "measurement.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir / "summary.md", result)
    print(json.dumps({
        "measurement": str(result_path),
        "ledger_sha256": result["ledger_sha256"],
        "models": [{
            "id": model["id"],
            "correct_top1_accuracy":
                model["metrics"]["correct_top1_accuracy"],
            "wrong_top1_accuracy": model["metrics"]["wrong_top1_accuracy"],
            "paired_delta":
                model["metrics"]["paired_top1_accuracy_delta"],
            "mcnemar_p":
                model["metrics"]["mcnemar_exact"]["two_sided_p_value"],
        } for model in model_results],
    }, indent=2))


if __name__ == "__main__":
    main()
