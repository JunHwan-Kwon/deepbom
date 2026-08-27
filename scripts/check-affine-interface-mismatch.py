import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCHEMA = "deepbom.affine_interface_mismatch_measurement.v1"
FLOAT_TOLERANCE = 1e-12


def fail(message):
    raise ValueError(message)


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


def parse_bool(value):
    if value == "True":
        return True
    if value == "False":
        return False
    fail(f"Invalid boolean value: {value!r}")


def close(actual, expected, label, tolerance=FLOAT_TOLERANCE):
    if not math.isclose(
        float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance
    ):
        fail(f"{label}: {actual!r} != {expected!r}")


def quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        fail("Cannot compute a quantile for an empty collection")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exact_binomial_two_sided(successes, trials):
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, value) for value in range(successes + 1))
    return min(1.0, 2.0 * tail / (2**trials))


def expected_calibration_error(confidences, correctness, bins=15):
    total = len(confidences)
    result = 0.0
    for ordinal in range(bins):
        lower = ordinal / bins
        upper = (ordinal + 1) / bins
        members = [
            index
            for index, confidence in enumerate(confidences)
            if confidence >= lower
            and (confidence <= upper if ordinal == bins - 1 else confidence < upper)
        ]
        if not members:
            continue
        accuracy = sum(correctness[index] for index in members) / len(members)
        confidence = sum(confidences[index] for index in members) / len(members)
        result += len(members) / total * abs(accuracy - confidence)
    return result


def bootstrap_paired_delta(correct, wrong, iterations, seed):
    differences = np.asarray(wrong, dtype=np.float64) - np.asarray(
        correct, dtype=np.float64
    )
    estimates = np.empty(iterations, dtype=np.float64)
    random = np.random.default_rng(seed)
    for ordinal in range(iterations):
        sample = random.integers(0, differences.size, size=differences.size)
        estimates[ordinal] = differences[sample].mean()
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(differences.mean()), float(lower), float(upper)


def verify_bootstrap(correct, wrong, recorded, label):
    point, lower, upper = bootstrap_paired_delta(
        correct,
        wrong,
        recorded["iterations"],
        recorded["seed"],
    )
    close(point, recorded["point"], f"{label} point")
    close(lower, recorded["lower_95"], f"{label} lower 95%")
    close(upper, recorded["upper_95"], f"{label} upper 95%")


def read_prediction_rows(filename):
    rows = []
    with filename.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            rows.append({
                "model_id": raw["model_id"],
                "relative_path": raw["relative_path"],
                "wnid": raw["wnid"],
                "label": int(raw["label"]),
                "correct_prediction": int(raw["correct_prediction"]),
                "wrong_prediction": int(raw["wrong_prediction"]),
                "correct_top1_correct": parse_bool(
                    raw["correct_top1_correct"]
                ),
                "wrong_top1_correct": parse_bool(raw["wrong_top1_correct"]),
                "correct_top5_correct": parse_bool(
                    raw["correct_top5_correct"]
                ),
                "wrong_top5_correct": parse_bool(raw["wrong_top5_correct"]),
                "correct_confidence": float(raw["correct_confidence"]),
                "wrong_confidence": float(raw["wrong_confidence"]),
                "js_divergence": float(raw["js_divergence"]),
                "identity_reencode_exact": parse_bool(
                    raw["identity_reencode_exact"]
                ),
                "input_element_count": int(raw["input_element_count"]),
                "wrong_input_low_clip_count": int(
                    raw["wrong_input_low_clip_count"]
                ),
                "wrong_input_high_clip_count": int(
                    raw["wrong_input_high_clip_count"]
                ),
            })
    return rows


def verify_histograms(metrics, model_id):
    for mode in ("correct", "wrong"):
        histogram = metrics[f"{mode}_input_histogram"]
        if len(histogram) != 256 or any(value < 0 for value in histogram):
            fail(f"{model_id}: invalid {mode} input histogram")
        expected_elements = (
            metrics["image_count"] * 224 * 224 * 3
        )
        if sum(histogram) != expected_elements:
            fail(f"{model_id}: {mode} histogram element sum mismatch")
        distinct = sum(value > 0 for value in histogram)
        if distinct != metrics[f"{mode}_input_distinct_code_count"]:
            fail(f"{model_id}: {mode} distinct code count mismatch")


def verify_class_metrics(rows, expected, model_id):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["wnid"]].append(row)
    expected_by_wnid = {row["wnid"]: row for row in expected}
    if set(grouped) != set(expected_by_wnid):
        fail(f"{model_id}: per-class key mismatch")
    for wnid, members in grouped.items():
        recorded = expected_by_wnid[wnid]
        if recorded["image_count"] != len(members):
            fail(f"{model_id}/{wnid}: per-class count mismatch")
        if recorded["imagenet_index"] != members[0]["label"]:
            fail(f"{model_id}/{wnid}: per-class label mismatch")
        close(
            sum(row["correct_top1_correct"] for row in members) / len(members),
            recorded["correct_top1_accuracy"],
            f"{model_id}/{wnid}: correct top-1",
        )
        close(
            sum(row["wrong_top1_correct"] for row in members) / len(members),
            recorded["wrong_top1_accuracy"],
            f"{model_id}/{wnid}: wrong top-1",
        )
        close(
            sum(
                row["correct_prediction"] == row["wrong_prediction"]
                for row in members
            )
            / len(members),
            recorded["top1_prediction_agreement"],
            f"{model_id}/{wnid}: prediction agreement",
        )


def verify_model(rows, model):
    model_id = model["id"]
    metrics = model["metrics"]
    count = len(rows)
    if count != metrics["image_count"]:
        fail(f"{model_id}: prediction row count mismatch")

    correct_top1 = [row["correct_top1_correct"] for row in rows]
    wrong_top1 = [row["wrong_top1_correct"] for row in rows]
    correct_top5 = [row["correct_top5_correct"] for row in rows]
    wrong_top5 = [row["wrong_top5_correct"] for row in rows]
    correct_confidence = [row["correct_confidence"] for row in rows]
    wrong_confidence = [row["wrong_confidence"] for row in rows]
    divergences = [row["js_divergence"] for row in rows]

    correct_top1_accuracy = sum(correct_top1) / count
    wrong_top1_accuracy = sum(wrong_top1) / count
    correct_top5_accuracy = sum(correct_top5) / count
    wrong_top5_accuracy = sum(wrong_top5) / count
    close(
        correct_top1_accuracy,
        metrics["correct_top1_accuracy"],
        f"{model_id}: correct top-1",
    )
    close(
        wrong_top1_accuracy,
        metrics["wrong_top1_accuracy"],
        f"{model_id}: wrong top-1",
    )
    close(
        correct_top5_accuracy,
        metrics["correct_top5_accuracy"],
        f"{model_id}: correct top-5",
    )
    close(
        wrong_top5_accuracy,
        metrics["wrong_top5_accuracy"],
        f"{model_id}: wrong top-5",
    )
    verify_bootstrap(
        correct_top1,
        wrong_top1,
        metrics["paired_top1_accuracy_delta"],
        f"{model_id}: paired top-1 bootstrap",
    )
    verify_bootstrap(
        correct_top5,
        wrong_top5,
        metrics["paired_top5_accuracy_delta"],
        f"{model_id}: paired top-5 bootstrap",
    )

    losses = sum(
        correct and not wrong
        for correct, wrong in zip(correct_top1, wrong_top1)
    )
    gains = sum(
        not correct and wrong
        for correct, wrong in zip(correct_top1, wrong_top1)
    )
    mcnemar = metrics["mcnemar_exact"]
    if losses != mcnemar["correct_to_incorrect"]:
        fail(f"{model_id}: McNemar loss count mismatch")
    if gains != mcnemar["incorrect_to_correct"]:
        fail(f"{model_id}: McNemar gain count mismatch")
    if losses + gains != mcnemar["discordant_pair_count"]:
        fail(f"{model_id}: McNemar discordant count mismatch")
    close(
        exact_binomial_two_sided(min(losses, gains), losses + gains),
        mcnemar["two_sided_p_value"],
        f"{model_id}: exact McNemar p-value",
    )

    close(
        sum(
            row["correct_prediction"] == row["wrong_prediction"]
            for row in rows
        )
        / count,
        metrics["top1_prediction_agreement"],
        f"{model_id}: prediction agreement",
    )
    close(
        sum(divergences) / count,
        metrics["mean_js_divergence"],
        f"{model_id}: mean JS divergence",
    )
    close(
        quantile(divergences, 0.5),
        metrics["median_js_divergence"],
        f"{model_id}: median JS divergence",
    )
    close(
        quantile(divergences, 0.95),
        metrics["p95_js_divergence"],
        f"{model_id}: p95 JS divergence",
    )
    close(
        sum(correct_confidence) / count,
        metrics["correct_mean_confidence"],
        f"{model_id}: correct mean confidence",
    )
    close(
        sum(wrong_confidence) / count,
        metrics["wrong_mean_confidence"],
        f"{model_id}: wrong mean confidence",
    )
    close(
        expected_calibration_error(correct_confidence, correct_top1),
        metrics["correct_ece_15_bin"],
        f"{model_id}: correct ECE",
    )
    close(
        expected_calibration_error(wrong_confidence, wrong_top1),
        metrics["wrong_ece_15_bin"],
        f"{model_id}: wrong ECE",
    )

    identity_mismatches = sum(
        not row["identity_reencode_exact"] for row in rows
    )
    if identity_mismatches != metrics["identity_reencode_mismatch_count"]:
        fail(f"{model_id}: identity re-encode mismatch count differs")
    low_clip = sum(row["wrong_input_low_clip_count"] for row in rows)
    high_clip = sum(row["wrong_input_high_clip_count"] for row in rows)
    elements = sum(row["input_element_count"] for row in rows)
    if low_clip != metrics["wrong_input_low_clip_count"]:
        fail(f"{model_id}: low input clip count mismatch")
    if high_clip != metrics["wrong_input_high_clip_count"]:
        fail(f"{model_id}: high input clip count mismatch")
    close(
        (low_clip + high_clip) / elements,
        metrics["wrong_input_clip_fraction"],
        f"{model_id}: input clip fraction",
    )
    verify_histograms(metrics, model_id)
    verify_class_metrics(rows, metrics["per_class"], model_id)


def verify_endpoint_csv(root, result):
    outputs = result["outputs"]
    expected_hash = outputs["endpoint_occupancy_csv_sha256"]
    filename = outputs["endpoint_occupancy_csv"]
    if expected_hash is None:
        if filename and (root / filename).exists():
            fail("Endpoint CSV exists but no SHA-256 is recorded")
        return
    if not filename:
        fail("Endpoint CSV SHA-256 exists without a filename")
    path = root / filename
    if not path.is_file():
        fail(f"Missing endpoint CSV: {path}")
    if sha256_file(path) != expected_hash:
        fail("Endpoint CSV SHA-256 mismatch")

    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[row["model_id"]].append(row)
    for model in result["models"]:
        endpoint = model["endpoint_occupancy"]
        rows = grouped[model["id"]]
        if len(rows) != endpoint["assessed_tensor_count"]:
            fail(f"{model['id']}: endpoint tensor count mismatch")
        aggregate = endpoint["aggregate"]
        for mode in ("correct", "wrong"):
            elements = sum(int(row[f"{mode}_elements"]) for row in rows)
            if elements != aggregate[f"{mode}_elements"]:
                fail(f"{model['id']}: endpoint element count mismatch")
            for side in ("lower", "upper", "zero"):
                count = sum(int(row[f"{mode}_{side}"]) for row in rows)
                if count != aggregate[f"{mode}_{side}_count"]:
                    fail(
                        f"{model['id']}: endpoint {mode} {side} count mismatch"
                    )
                close(
                    count / elements,
                    aggregate[f"{mode}_{side}_fraction"],
                    f"{model['id']}: endpoint {mode} {side} fraction",
                )


def verify_result(root):
    result_path = root / "measurement.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != SCHEMA:
        fail(f"{root}: unsupported schema {result.get('schema')!r}")

    producer = Path(__file__).resolve().parent / result["provenance"]["script"]
    if not producer.is_file():
        fail(f"{root}: missing measurement producer {producer}")
    if sha256_file(producer) != result["provenance"]["script_sha256"]:
        fail(f"{root}: measurement producer SHA-256 mismatch")

    recorded_ledger = result.get("ledger_sha256")
    unsigned = dict(result)
    unsigned.pop("ledger_sha256", None)
    if sha256_json(unsigned) != recorded_ledger:
        fail(f"{root}: result ledger SHA-256 mismatch")

    outputs = result["outputs"]
    predictions = root / outputs["predictions_csv"]
    if not predictions.is_file():
        fail(f"Missing predictions CSV: {predictions}")
    if sha256_file(predictions) != outputs["predictions_csv_sha256"]:
        fail(f"{root}: predictions CSV SHA-256 mismatch")

    rows = read_prediction_rows(predictions)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model_id"]].append(row)
    model_ids = {model["id"] for model in result["models"]}
    if set(grouped) != model_ids:
        fail(f"{root}: model IDs differ between JSON and CSV")

    expected_paths = None
    image_ledger = None
    for model in result["models"]:
        model_rows = grouped[model["id"]]
        paths = [row["relative_path"] for row in model_rows]
        if len(paths) != len(set(paths)):
            fail(f"{model['id']}: duplicate image rows")
        if expected_paths is None:
            expected_paths = set(paths)
            image_ledger = [{
                "relative_path": row["relative_path"],
                "wnid": row["wnid"],
                "label": row["label"],
            } for row in model_rows]
        elif paths != [row["relative_path"] for row in image_ledger]:
            fail("Models were not evaluated on the same image set")
        verify_model(model_rows, model)

    if len(expected_paths) != result["dataset"]["image_count"]:
        fail(f"{root}: dataset image count mismatch")
    if sha256_json(image_ledger) != result["dataset"]["image_ledger_sha256"]:
        fail(f"{root}: selected image ledger SHA-256 mismatch")
    class_counts = Counter(row["wnid"] for row in grouped[next(iter(model_ids))])
    if len(class_counts) != result["dataset"]["class_count"]:
        fail(f"{root}: dataset class count mismatch")
    if max(class_counts.values()) - min(class_counts.values()) > 1:
        fail(f"{root}: selected image set is not class-stratified")

    verify_endpoint_csv(root, result)
    return {
        "directory": str(root),
        "ledger_sha256": recorded_ledger,
        "image_count": result["dataset"]["image_count"],
        "class_count": result["dataset"]["class_count"],
        "models": [{
            "id": model["id"],
            "correct_top1": model["metrics"]["correct_top1_accuracy"],
            "wrong_top1": model["metrics"]["wrong_top1_accuracy"],
            "paired_delta": model["metrics"][
                "paired_top1_accuracy_delta"
            ]["point"],
            "mcnemar_p": model["metrics"]["mcnemar_exact"][
                "two_sided_p_value"
            ],
            "input_clip_fraction": model["metrics"][
                "wrong_input_clip_fraction"
            ],
        } for model in result["models"]],
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Independently recompute affine-interface mismatch result "
            "invariants from the emitted CSV ledgers."
        )
    )
    parser.add_argument("result_directories", nargs="+", type=Path)
    args = parser.parse_args()
    summaries = [verify_result(path.resolve()) for path in args.result_directories]
    print(json.dumps({
        "status": "pass",
        "verified_result_count": len(summaries),
        "results": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
