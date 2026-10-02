"""Verify PyTorch with one real segmentation batch and an optimizer update."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def check(dataset: Path) -> dict:
    torch.manual_seed(0)
    torch.set_num_threads(2)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    course = next(c for c in manifest["courses"] if c["split"] == "train")
    with np.load(dataset / course["shard"], allow_pickle=False) as data:
        rgb = data["rgb"][:2].copy()
        labels = data["semantic"][:2].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255.0
    y = torch.from_numpy(labels).to(device=device, dtype=torch.long)
    # Disposable smoke model; this is NOT the eventual perception architecture.
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 3, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model[0].weight.detach().clone()
    started = time.perf_counter()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y, ignore_index=255)
    assert tuple(logits.shape) == (len(rgb), 3, 200, 200)
    assert torch.isfinite(loss).item(), "Non-finite loss"
    optimizer.zero_grad()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    optimizer.step()
    delta = (model[0].weight.detach() - before).abs().max().item()
    assert delta > 0.0, "Optimizer did not update weights"
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = {
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
        "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda,
        "dataset": str(dataset.resolve()), "batch_shape": list(x.shape),
        "loss": loss.item(), "max_weight_change": delta,
        "forward_backward_update_s": round(time.perf_counter() - started, 3),
        "status": "passed", "saved_model": False,
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "training/data/perception-pilot")
    args = parser.parse_args()
    print(json.dumps(check(args.dataset), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
