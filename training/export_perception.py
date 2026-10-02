"""Export a trained model and verify CPU ONNX Runtime against PyTorch on validation RGB."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from training.perception_data import PerceptionDataset
from training.perception_model import SegmentationMetrics
from training.perception_runtime import OnnxPerception, prepare_rgb
from training.train_perception import load_checkpoint, write_json

ROOT = Path(__file__).resolve().parents[1]


LOGIT_ATOL = 5e-3
PROBABILITY_ATOL = 2e-3
MIN_LABEL_AGREEMENT = 1.0 - 1e-5
MAX_CHANGED_MARGIN = 2e-3
MAX_CLASS_IOU_DIFFERENCE = 1e-4


def probabilities(logits):
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def compare_outputs(actual, reference):
    # CPU kernels can accumulate/reduce FP32 in different orders. Keep explicit
    # bounds on scores AND probabilities; label/IoU gates below reject material drift.
    logit_error = float(np.max(np.abs(actual - reference)))
    probability_error = float(np.max(np.abs(probabilities(actual) - probabilities(reference))))
    if not np.isfinite(actual).all() or not np.isfinite(reference).all() or logit_error > LOGIT_ATOL or probability_error > PROBABILITY_ATOL:
        raise AssertionError(f"Export numerical error: logits={logit_error}, probabilities={probability_error}")
    return logit_error, probability_error


def label_changes(actual, reference, valid):
    """Changed decisions must be nearly tied in the reference probabilities.

    Aggregate agreement and class IoU must ALSO pass their gates; a near tie alone
    does not permit an arbitrarily large number of changed labels.
    """
    changed = (actual.argmax(axis=1) != reference.argmax(axis=1)) & valid
    if not changed.any():
        return 0, 0.0
    top = np.partition(probabilities(reference), -2, axis=1)[:, -2:]
    margin = top[:, 1] - top[:, 0]
    max_margin = float(margin[changed].max())
    if max_margin > MAX_CHANGED_MARGIN:
        raise AssertionError(f"Export changed a confident prediction; probability margin={max_margin}")
    return int(changed.sum()), max_margin


def class_iou_difference(reference, actual, classes):
    differences = []
    for name in classes:
        a, b = reference["class_iou"][name], actual["class_iou"][name]
        differences.append(0.0 if a is None and b is None else 1.0 if a is None or b is None else abs(a - b))
    return max(differences)


def export(checkpoint: Path, dataset: Path, out: Path, threads=2, verify_only=False):
    if threads < 1:
        raise ValueError("threads must be positive")
    if not verify_only and (out.exists() or out.with_suffix(".verification.json").exists()):
        raise FileExistsError(f"Choose a fresh ONNX path: {out}")
    if verify_only and not out.is_file():
        raise FileNotFoundError(out)
    torch.set_num_threads(threads)
    model, meta = load_checkpoint(checkpoint, torch.device("cpu"))
    validation = PerceptionDataset(dataset, "val", label_scheme=model.label_scheme)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Explicitly use the classic exporter to target opset 17, supported by ORT 1.23.2.
    # No training state, optimizer, dataset labels or world state is exported.
    if not verify_only:
        torch.onnx.export(model, torch.zeros(1, 3, 200, 200), str(out),
                          input_names=["rgb"], output_names=["logits"],
                          dynamic_axes={"rgb": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)
    graph = onnx.load(out)
    if not verify_only:
        onnx.helper.set_model_props(graph, {
            "classes": json.dumps(list(model.classes)), "label_scheme": model.label_scheme,
            "input": "RGB NCHW float32 [0,1]",
            "condition": "1-2", "m_per_px": "0.5", "frame": "bow_up",
            "ignore_ego_label": "255", "checkpoint_epoch": str(meta["epoch"]),
        })
    onnx.checker.check_model(graph)
    if not verify_only:
        onnx.save(graph, out)
    runtime = OnnxPerception(out, threads=threads)
    if runtime.classes != model.classes:
        raise ValueError("ONNX class names differ from the checkpoint")
    reference_metrics, runtime_metrics = SegmentationMetrics(model.classes), SegmentationMetrics(model.classes)
    max_error, max_probability_error, agreed, valid_pixels = 0.0, 0.0, 0, 0
    max_changed_margin = 0.0
    for i in range(len(validation)):
        rgb, target, buoys = validation.rgb[i], validation.labels[i], validation.buoys[i]
        x = prepare_rgb(rgb)
        with torch.inference_mode():
            reference = model(torch.from_numpy(x)).numpy()
        actual = runtime.session.run(["logits"], {"rgb": x})[0]
        error, probability_error = compare_outputs(actual, reference)
        max_error = max(max_error, error)
        max_probability_error = max(max_probability_error, probability_error)
        prediction = runtime.predict(rgb)
        ref_labels = reference[0].argmax(axis=0)
        valid = target != 255
        changed, margin = label_changes(actual, reference, valid[None])
        max_changed_margin = max(max_changed_margin, margin)
        if not np.array_equal(prediction["valid"], valid):
            raise RuntimeError("Public ego mask does not match official raster geometry")
        if not np.allclose(prediction["class_probabilities"].sum(axis=2), 1.0, atol=1e-6):
            raise RuntimeError("Class probabilities do not sum to one")
        if np.any(prediction["occupancy_probability"][~valid] != 0.0):
            raise RuntimeError("Ego hull leaked into occupancy")
        if not np.allclose(prediction["occupancy_probability"],
                           prediction["static_occupancy_probability"] + prediction["vessel_probability"], atol=1e-7):
            raise RuntimeError("Occupancy must combine fixed surfaces and vessels")
        if not np.array_equal(prediction["labels"][valid], actual[0].argmax(axis=0)[valid]):
            raise RuntimeError("Repeated ONNX inference changed its pixel labels")
        agreed += int(valid.sum()) - changed
        valid_pixels += int(valid.sum())
        reference_metrics.update(ref_labels, target, buoys)
        runtime_metrics.update(prediction["labels"], target, buoys)
    if agreed / valid_pixels < MIN_LABEL_AGREEMENT:
        raise AssertionError(f"Export changed too many labels: {valid_pixels - agreed}/{valid_pixels}")
    reference_report, runtime_report = reference_metrics.report(), runtime_metrics.report()
    iou_difference = class_iou_difference(reference_report, runtime_report, model.classes)
    if iou_difference > MAX_CLASS_IOU_DIFFERENCE:
        raise AssertionError(f"Export class IoU changed by {iou_difference}")
    # Also verify the advertised dynamic batch axis with two real frames.
    batch = np.concatenate([prepare_rgb(validation.rgb[i]) for i in range(min(2, len(validation)))])
    with torch.inference_mode():
        reference_batch = model(torch.from_numpy(batch)).numpy()
    actual_batch = runtime.session.run(["logits"], {"rgb": batch})[0]
    batch_logit_error, batch_probability_error = compare_outputs(actual_batch, reference_batch)
    batch_valid = validation.labels[:len(batch)] != 255
    batch_changes, batch_margin = label_changes(actual_batch, reference_batch, batch_valid)
    if 1.0 - batch_changes / int(batch_valid.sum()) < MIN_LABEL_AGREEMENT:
        raise AssertionError("Dynamic batch changed too many valid pixel predictions")
    timings = []
    for i in range(13):
        start = time.perf_counter()
        runtime.predict(validation.rgb[i % len(validation)])
        ms = 1000.0 * (time.perf_counter() - start)
        if i >= 3:
            timings.append(ms)
    result = {
        "format": "usvnav-perception-onnx-verification/1", "status": "passed",
        "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
        "providers": runtime.session.get_providers(), "opset": 17, "ir_version": graph.ir_version,
        "checkpoint_epoch": meta["epoch"], "model_bytes": out.stat().st_size,
        "classes": list(model.classes), "label_scheme": model.label_scheme,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "validation_samples": len(validation), "dynamic_batch_verified": len(batch),
        "max_absolute_logit_error": max_error,
        "max_absolute_probability_error": max_probability_error,
        "dynamic_batch_max_logit_error": batch_logit_error,
        "dynamic_batch_max_probability_error": batch_probability_error,
        "dynamic_batch_changed_pixels": batch_changes,
        "dynamic_batch_max_changed_probability_margin": batch_margin,
        "verification_tolerances": {"absolute_logit_error": LOGIT_ATOL, "absolute_probability_error": PROBABILITY_ATOL,
                                    "minimum_valid_pixel_prediction_agreement": MIN_LABEL_AGREEMENT,
                                    "maximum_changed_probability_margin": MAX_CHANGED_MARGIN,
                                    "maximum_class_iou_difference": MAX_CLASS_IOU_DIFFERENCE},
        "changed_pixels": valid_pixels - agreed,
        "max_changed_probability_margin": max_changed_margin,
        "maximum_class_iou_difference": iou_difference,
        "valid_pixel_prediction_agreement": agreed / valid_pixels,
        "torch_validation": reference_report, "onnx_validation": runtime_report,
        "full_perception_latency_ms": {"median": float(np.median(timings)), "p95": float(np.percentile(timings, 95)),
                                       "threads": threads, "measured_calls": len(timings),
                                       "scope": "RGB preparation, ORT CPU forward, probabilities and ego masking; excludes tracking, control and runner"},
        "runtime_limitations": "Local Python 3.11 and NumPy 2.1; official Python 3.13, NumPy 2.5.3 and CUDA provider not verified here",
    }
    write_json(out.with_suffix(".verification.json"), result)
    latency = result["full_perception_latency_ms"]
    lines = ["# ONNX 인식 추론 검증", "",
             f"ONNX Runtime {ort.__version__} CPU에서 검증 이미지 {len(validation)}장을 비교했다.",
             f"자기 선박을 제외한 {valid_pixels:,}픽셀 중 {valid_pixels - agreed}픽셀의 분류가 달랐다. 일치율 {100 * agreed / valid_pixels:.6f}%.",
             f"최대 내부 점수 오차: {max_error:.7f}. 최대 클래스 확률 오차: {max_probability_error:.7f}.",
             "허용 기준: 점수 오차 0.005, 확률 오차 0.002 이하; 픽셀 일치율 99.999% 이상; 종류별 IoU 차이 0.0001 이하.",
             f"달라진 픽셀의 최대 정답 모델 확률 차이(1·2위): {max_changed_margin:.7f}. 허용 한도 0.002.",
             f"최대 종류별 IoU 차이: {iou_difference:.7f}. 수치 오차가 확실한 분류를 바꾸거나 작은 종류의 지표를 훼손하면 검증은 실패한다.",
             "배치 1과 2를 검증했고, 공개된 자기 선체 형상으로 만든 제외 마스크도 정답과 일치했다.", "",
             f"전처리·ORT 추론·확률 계산·자기 선박 제외를 포함한 시간: 중앙값 {latency['median']:.1f}ms, p95 {latency['p95']:.1f}ms.",
             f"ONNX 파일 크기: {out.stat().st_size:,}바이트. 입력 RGB NCHW float32 [0,1], 출력 {len(model.classes)}종류의 픽셀 점수.",
             f"클래스 순서: {', '.join(model.classes)}.", "",
             "로컬 Python 3.11/NumPy 2.1/CPU에서 확인한 결과다. 공식 Python 3.13/NumPy 2.5.3/GPU 환경은 별도 검증이 필요하다.",
             "추적·주행 정책·프로세스 통신은 포함하지 않았다. 공식 에이전트의 시간 제한 통과나 완주 성능을 뜻하지 않는다.", ""]
    out.with_suffix(".verification.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "validation_samples", "model_bytes", "max_absolute_logit_error",
                                          "max_absolute_probability_error", "valid_pixel_prediction_agreement", "full_perception_latency_ms"]}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "training/runs/pilot-unet8-01/best.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "training/data/perception-pilot")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true", help="Validate an existing ONNX file without exporting or replacing it")
    args = parser.parse_args()
    export(args.checkpoint, args.dataset, args.out or args.checkpoint.parent / "perception.onnx", args.threads, args.verify_only)


if __name__ == "__main__":
    main()
