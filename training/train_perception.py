"""Train a small RGB segmentation model; select on validation, test once at the end."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from usvnav.png import write_png
from training.perception_data import PerceptionDataset, read_manifest
from training.perception_model import SegmentationLoss, SegmentationMetrics, TinyUNet
from training.perception_schema import CLASS_SCHEMES, colour_labels, scheme_for

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate(model, loader, criterion, device):
    model.eval()
    metrics, total_loss, samples = SegmentationMetrics(model.classes), 0.0, 0
    with torch.inference_mode():
        for x, y, buoys in loader:
            x, y, buoys = x.to(device), y.to(device), buoys.to(device)
            logits = model(x)
            loss = criterion(logits, y, buoys)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite evaluation loss")
            total_loss += loss.item() * len(x)
            samples += len(x)
            metrics.update(logits.argmax(1).cpu().numpy(), y.cpu().numpy(), buoys.cpu().numpy())
    result = metrics.report()
    result.update({"loss": total_loss / samples, "samples": samples})
    return result


def water_baseline(dataset):
    metrics = SegmentationMetrics(dataset.classes)
    metrics.update(np.zeros_like(dataset.labels), dataset.labels, dataset.buoys)
    return metrics.report()


def save_checkpoint(path, model, epoch, score):
    torch.save({
        "format": "usvnav-perception-model/1", "architecture": "tiny_unet",
        "base_channels": model.base_channels,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "classes": list(model.classes), "label_scheme": model.label_scheme,
        "ignore_index": 255, "input": "RGB NCHW float32 [0,1]",
        "epoch": epoch, "validation_miou": score,
    }, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format"] != "usvnav-perception-model/1":
        raise ValueError("Unexpected checkpoint format")
    label_scheme = scheme_for(checkpoint["classes"])
    if checkpoint.get("label_scheme", label_scheme) != label_scheme:
        raise ValueError("Checkpoint class names disagree with its label scheme")
    model = TinyUNet(checkpoint["base_channels"], classes=checkpoint["classes"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def predict(model, dataset, i, device):
    x, _, _ = dataset[i]
    with torch.inference_mode():
        return model(x[None].to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)


def preview(model, dataset, device, path):
    # First three validation samples, fixed in dataset order; no best-example selection.
    rows = []
    for i in range(min(3, len(dataset))):
        prediction = predict(model, dataset, i, device)
        prediction[dataset.labels[i] == 255] = 255
        rows.append(np.concatenate([dataset.rgb[i], colour_labels(dataset.labels[i], dataset.classes),
                                    colour_labels(prediction, dataset.classes)], axis=1))
    write_png(path, np.concatenate(rows, axis=0))


def model_latency(model, dataset, device):
    x = dataset[0][0][None].to(device)
    times = []
    with torch.inference_mode():
        for i in range(13):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = 1000.0 * (time.perf_counter() - start)
            if i >= 3:
                times.append(elapsed)
    return {"median": float(np.median(times)), "p95": float(np.percentile(times, 95)),
            "measured_forwards": len(times), "warmup_forwards": 3,
            "scope": "PyTorch model only, batch 1; excludes preprocessing, simulator, policy and runner"}


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def write_report(out, results, config):
    val, test = results["validation"], results["test"]
    classes = config["classes"]
    detail = config["label_scheme"] == "detail"
    buoy_definition = "부표로" if detail else "고정 장애물로"
    legend = ("파랑=물, 황갈색=강둑, 노랑=부표, 보라=구조물/암반, 주황갈색=접안시설, 청록=선박, 회색=자기 선박 무시."
              if detail else "파랑=물, 갈색=고정 장애물·육지, 청록=다른 선박, 회색=자기 선박 무시.")
    lines = [
        "# 1-2 인식 모델 초기 실험", "",
        f"탑뷰 RGB 한 장에서 {', '.join(classes)}를 분류한다. 자기 선박은 손실과 평가에서 제외한다.", "",
        f"학습 {config['split_samples']['train']}장, 검증 {config['split_samples']['val']}장, 테스트 {config['split_samples']['test']}장이다.",
        "학습/검증/테스트는 코스 단위로 분리했다. 검증 mIoU로 체크포인트를 선택했고 테스트는 선택 후 한 번 평가했다.", "",
        f"모델: TinyUNet, 시작 채널 {config['base_channels']}, 파라미터 {results['parameters']:,}개.",
        f"선택된 에포크: {results['best_epoch']} / {config['epochs']}. 장치: {config['device']}.", "",
        "| 지표 | 검증 | 테스트 |", "|---|---:|---:|",
        f"| mIoU | {percent(val['miou'])} | {percent(test['miou'])} |",
    ]
    for name in classes:
        lines.append(f"| {name} IoU | {percent(val['class_iou'][name])} | {percent(test['class_iou'][name])} |")
    lines += [
        f"| 부표 픽셀 재현율 | {percent(val['buoy_pixel_recall'])} | {percent(test['buoy_pixel_recall'])} |",
        f"| 전부 물이라고 예측하는 기준 mIoU | {percent(results['water_baseline_validation']['miou'])} | {percent(results['water_baseline_test']['miou'])} |", "",
        f"IoU는 예측 영역과 정답 영역의 교집합/합집합이다. mIoU는 {len(classes)}종 중 정답 또는 예측에 나타난 종류의 IoU 평균이다.",
        "정답과 예측 모두 없는 종류는 IoU가 정의되지 않아 평균에서 제외한다. 종류별 정답 픽셀 수는 metrics.json에 기록한다.",
        f"부표 픽셀 재현율은 정답 부표 픽셀 중 {buoy_definition} 예측한 비율이며 부표 객체 검출률과는 다르다.", "",
        f"현재 PC의 모델 단독 추론 시간: 중앙값 {results['model_only_latency_ms']['median']:.1f}ms, p95 {results['model_only_latency_ms']['p95']:.1f}ms.",
        "영상 전처리·추적·행동 결정·프로세스 통신 시간을 포함하지 않으므로 공식 시간 제한 통과를 뜻하지 않는다.", "",
        "`validation.png`는 검증 데이터의 첫 세 장이다. 각 행은 원본 / 정답 / 예측 순서다.",
        legend, "",
        "## 한계와 다음 단계", "",
        "제공된 연습 코스와 자체 작성한 보완 코스의 소량 데이터다. 숨겨진 코스의 일반화 성능이나 주행 성공률을 검증하지 않았다.",
        "부표는 매우 작으므로 전체 IoU가 높아도 놓칠 수 있다. 부표 재현율과 미리보기를 함께 확인한다.",
        "그림자·갑판·얕은 물색은 별도 장애물 라벨이 아니다. 정답은 공식 class_map의 픽셀 중심 형상이다.",
        "암반은 공식 pier 클래스에 포함된다. 계류·이동 선박은 vessel로 합치며 움직임은 연속 관측에서 추정한다.",
        "다양한 구조·물체 배치·교통의 코스를 더 만들고 데이터 분할을 고정한 뒤 재학습한다.",
        "현재 모델은 한 프레임만 처리하며 속도·추적·행동 출력은 없다. 이후 시간축 인식과 PPO 환경을 연결한다.",
        "`best.pt`는 PyTorch 학습 가중치다. 제출용 에이전트에 연결하기 전 ONNX로 내보내고 런타임과 지연을 검증해야 한다.", "",
        "설계 참고: [U-Net 원 논문](https://arxiv.org/abs/1505.04597), [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).", "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.lr <= 0 or args.dice_weight < 0 or args.buoy_weight < 1:
        raise ValueError("Invalid training hyperparameters")
    if args.out.exists():
        raise FileExistsError(f"Choose a fresh run directory: {args.out}")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but it is unavailable")
    manifest = read_manifest(args.dataset)
    datasets = {split: PerceptionDataset(args.dataset, split, manifest, label_scheme=args.label_scheme) for split in ("train", "val")}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(data, batch_size=args.batch_size, shuffle=split == "train", num_workers=0,
                                generator=generator if split == "train" else None)
               for split, data in datasets.items()}
    weights, counts = datasets["train"].class_weights()
    model = TinyUNet(args.base_channels, classes=datasets["train"].classes).to(device)
    criterion = SegmentationLoss(weights, args.dice_weight, args.buoy_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"device": str(device), "classes": list(model.classes), "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                   "dataset_manifest_sha256": hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest(),
                   "split_samples": {s: sum(c['samples'] for c in manifest['courses'] if c['split'] == s) for s in ("train", "val", "test")},
                   "train_pixel_counts": counts, "class_weights_from_train": weights.tolist(), "augmentation": "none",
                   "selection": "highest validation mIoU; earliest checkpoint wins ties",
                   "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / 'training/perception_model.py', ROOT / 'training/perception_data.py', ROOT / 'training/perception_schema.py']}})
    args.out.mkdir(parents=True)
    write_json(args.out / "config.json", config)
    history, best, best_epoch = [], -1.0, 0
    started = time.perf_counter()
    parameters = sum(p.numel() for p in model.parameters())
    print(f"device={device}, parameters={parameters:,}, splits={config['split_samples']}, class_weights={weights.tolist()}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for x, y, buoy_pixels in loaders["train"]:
            x, y, buoy_pixels = x.to(device), y.to(device), buoy_pixels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y, buoy_pixels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(x)
            n += len(x)
        val = evaluate(model, loaders["val"], criterion, device)
        history.append({"epoch": epoch, "train_loss": total_loss / n, "validation": val,
                        "elapsed_s": round(time.perf_counter() - started, 3)})
        if val["miou"] > best:
            best, best_epoch = val["miou"], epoch
            save_checkpoint(args.out / "best.pt", model, epoch, best)
        write_json(args.out / "history.json", history)
        print(f"epoch={epoch:02d}/{args.epochs} loss={total_loss/n:.4f} val_mIoU={percent(val['miou'])} "
              f"buoy_recall={percent(val['buoy_pixel_recall'])} best_epoch={best_epoch} elapsed={history[-1]['elapsed_s']:.1f}s", flush=True)
    model, checkpoint = load_checkpoint(args.out / "best.pt", device)
    val = evaluate(model, loaders["val"], criterion, device)
    if abs(val["miou"] - best) > 1e-8:
        raise RuntimeError("Reloaded checkpoint validation differs from selected result")
    test_data = PerceptionDataset(args.dataset, "test", manifest, label_scheme=args.label_scheme)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=0)
    results = {"format": "usvnav-perception-results/1", "best_epoch": checkpoint["epoch"], "parameters": parameters,
               "validation": val, "test": evaluate(model, test_loader, criterion, device),
               "water_baseline_validation": water_baseline(datasets["val"]), "water_baseline_test": water_baseline(test_data),
               "model_only_latency_ms": model_latency(model, datasets["val"], device),
               "checkpoint_bytes": (args.out / "best.pt").stat().st_size,
               "elapsed_s": round(time.perf_counter() - started, 3)}
    preview(model, datasets["val"], device, args.out / "validation.png")
    write_json(args.out / "metrics.json", results)
    write_report(args.out, results, config)
    print(f"Finished: validation mIoU={percent(val['miou'])}, test mIoU={percent(results['test']['miou'])}; {args.out.resolve()}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "training/data/perception-pilot")
    parser.add_argument("--out", type=Path, default=ROOT / "training/runs/pilot-unet8-01")
    parser.add_argument("--label-scheme", choices=CLASS_SCHEMES, default="coarse")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--buoy-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
