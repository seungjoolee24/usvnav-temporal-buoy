"""Build a compact interactive replay from recorded inference; no model rerun."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from usvnav.png import encode_png
from training.collect_perception import ROOT
from training.tracking import world_to_body


def build_preview(report_dir, output, clip_id=None):
    report_dir, output = Path(report_dir), Path(output)
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    records = json.loads((report_dir / "tracks.json").read_text(encoding="utf-8"))
    if clip_id is None:
        record = next(r for r in records if r["clip"]["profile"] == "gentle_turn" and r["clip"]["visible_moving_frames"] > 0)
    else:
        record = next(r for r in records if r["clip"]["clip_id"] == clip_id)
    clip, frames = record["clip"], []
    with np.load(Path(metrics["dataset"]) / clip["path"], allow_pickle=False) as data:
        for j, saved in enumerate(record["frames"]):
            pose = data["pose"][j].astype(float)
            truth = []
            for i in np.flatnonzero(data["gt_visible"][j]):
                state = data["gt_state"][j, i]
                truth.append(dict(id=str(data["gt_ids"][i]), kind=int(data["gt_kind"][i]),
                                  body=world_to_body(state[:2][None], pose)[0].tolist(),
                                  velocity=world_to_body((pose[:2] + state[5:7])[None], pose)[0].tolist(),
                                  speed=float(np.linalg.norm(state[5:7]))))
            frames.append(dict(png="data:image/png;base64," + base64.b64encode(encode_png(data["rgb"][j])).decode("ascii"),
                               seconds=(saved["tick"] - record["frames"][0]["tick"]) * .1,
                               tracks=saved["tracks"], associations=saved["offline_associations"], truth=truth))
    payload = dict(clip_id=clip["clip_id"], label=f"{clip['split']} · {clip['course_id']} · 우리 배 이동·회전", frames=frames)
    template = (ROOT / "training/tracking_view.html").read_text(encoding="utf-8")
    fragment = template.replace("__TRACKING_PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c"))
    if len(fragment.encode("utf-8")) >= 1000000:
        raise ValueError("Preview too large")
    if output.exists():
        raise FileExistsError(f"Choose a new preview path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment, encoding="utf-8")
    return dict(path=str(output.resolve()), bytes=output.stat().st_size, clip=clip["clip_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "training/runs/tracking-baseline-01")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clip")
    args = parser.parse_args()
    print(json.dumps(build_preview(args.report, args.out, args.clip), ensure_ascii=False))


if __name__ == "__main__":
    main()
