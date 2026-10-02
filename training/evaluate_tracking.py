"""Evaluate RGB-only perception + public-pose tracking against offline truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from usvnav.coursefile import load
from usvnav.plant import DT, Vessel
from training.collect_perception import ROOT, course_identity
from training.collect_tracking import continuous_states, fully_visible, instance_map, ship_states
from training.perception_runtime import OnnxPerception
from training.perception_schema import DETAIL_CLASSES
from training.tracking import VesselTracker, minimum_assignment


def audit_dataset(root):
    """Check every saved clip, including held-out clips without model predictions."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "usvnav-tracking/1" or manifest["dt_s"] != DT:
        raise ValueError("Unsupported tracking dataset")
    courses, splits, hashes = {}, {}, set()
    for item in manifest["courses"]:
        course = load(root / item["snapshot"])
        identity = course_identity(course)
        if identity != item["sha256"] or identity in hashes:
            raise ValueError("Invalid or duplicated course identity")
        if item["split"] not in ("train", "val", "test"):
            raise ValueError("Invalid course split")
        hashes.add(identity)
        courses[item["course_id"]], splits[item["course_id"]] = course, item["split"]
    frames, minimum, visible = 0, 10., 0
    clip_ids = set()
    for item in manifest["clips"]:
        if item["clip_id"] in clip_ids or item["split"] != splits[item["course_id"]]:
            raise ValueError("Duplicate clip or inconsistent split")
        clip_ids.add(item["clip_id"])
        course = courses[item["course_id"]]
        with np.load(root / item["path"], allow_pickle=False) as data:
            f = item["frames"]
            if (data["rgb"].shape != (f, 200, 200, 3) or data["rgb"].dtype != np.uint8
                    or data["semantic"].shape != (f, 200, 200) or data["tick"].shape != (f,)
                    or not np.all(np.diff(data["tick"]) == 1)
                    or int(data["tick"][0]) != item["start_tick"]):
                raise ValueError("Malformed sensor sequence")
            if data["actions"].shape != (f - 1, 2):
                raise ValueError("Actions must describe frame-to-next-frame transitions")
            np.testing.assert_array_equal(data["pose"], data["pose_exact"].astype(np.float32))
            expected = continuous_states(course, data["pose_exact"][0], item["start_tick"], f,
                                         item["action"], manifest["min_clearance_m"])
            if expected is None:
                raise ValueError("Unsafe clip")
            np.testing.assert_allclose(data["actions"], np.tile(item["action"], (f - 1, 1)), atol=1e-7)
            np.testing.assert_allclose(data["pose_exact"], expected[0], atol=1e-10, rtol=0)
            np.testing.assert_allclose(data["vel"], expected[1], atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(data["clearance_m"], expected[2], atol=1e-6, rtol=0)
            ids = list(data["gt_ids"])
            if len(ids) != len(set(ids)):
                raise ValueError("Repeated truth ID")
            lookup = {identity: i for i, identity in enumerate(ids)}
            for j, tick in enumerate(data["tick"]):
                truth = ship_states(course, int(tick) * DT)
                np.testing.assert_array_equal(data["gt_alive"][j], [identity in truth for identity in ids])
                np.testing.assert_array_equal(data["gt_instance"][j],
                                              instance_map(truth, lookup, data["pose_exact"][j], data["semantic"][j]))
                for identity, (kind, rect, velocity) in truth.items():
                    i = lookup[identity]
                    np.testing.assert_allclose(data["gt_state"][j, i],
                                               [rect.x, rect.y, rect.heading, rect.length, rect.width, *velocity],
                                               atol=1e-9, rtol=0)
                    assert data["gt_kind"][i] == kind
                    is_visible = np.count_nonzero(data["gt_instance"][j] == i) >= 6
                    assert data["gt_visible"][j, i] == is_visible
                    assert data["gt_fully_visible"][j, i] == (is_visible and fully_visible(rect, data["pose_exact"][j]))
            minimum = min(minimum, float(data["clearance_m"].min()))
            visible += int(data["gt_visible"].sum())
            frames += f
    return manifest, dict(status="passed", courses=len(courses), clips=len(clip_ids), frames=frames,
                          min_clearance_m=minimum, visible_object_frames=visible,
                          checks=["canonical course identity and whole-course splits", "contiguous 10 Hz ticks",
                                  "official dynamics for every transition", "full-hull safety at every tick",
                                  "stable instance IDs, geometry and ground velocity", "NumPy archive without pickle"])


def match_instances(detections, truth_map, visible, min_iou=.2):
    gt = np.flatnonzero(visible)
    areas = np.bincount(truth_map[truth_map >= 0], minlength=len(visible))
    cost = np.full((len(gt), len(detections) + len(gt)), 1 - min_iou + 1e-6)
    ious = np.zeros((len(gt), len(detections)))
    for j, detection in enumerate(detections):
        inside = truth_map[detection.rows, detection.cols]
        overlap = np.bincount(inside[inside >= 0], minlength=len(visible))[gt]
        ious[:, j] = overlap / np.maximum(1, areas[gt] + len(inside) - overlap)
        cost[:, j] = np.where(ious[:, j] >= min_iou, 1 - ious[:, j], 1e6)
    return [(int(gt[i]), j, float(ious[i, j])) for i, j in minimum_assignment(cost)
            if j < len(detections) and ious[i, j] >= min_iou]


class Metrics:
    def __init__(self):
        self.frames = self.gt = self.detected = self.matched = self.switches = 0
        self.position_errors, self.velocity_errors = [], []
        self.confusion = np.zeros((2, 4), int)  # truth static/moving; missed/unknown/stationary/moving
        self.previous = {}

    def start_clip(self):
        self.previous.clear()

    def add(self, frame, data, tracker, tracks):
        detections = tracker.last_detections
        matches = match_instances(detections, data["gt_instance"][frame], data["gt_visible"][frame])
        self.frames += 1
        self.gt += int(data["gt_visible"][frame].sum())
        self.detected += len(detections)
        self.matched += len(matches)
        by_id = {t["track_id"]: t for t in tracks}
        associations, matched_by_gt = [], {}
        tick = int(data["tick"][frame])
        for i, j, iou in matches:
            detection = detections[j]
            track = by_id[detection.track_id]
            matched_by_gt[i] = track
            prev = self.previous.get(i)
            if prev and tick - prev[0] <= 5 and track["track_id"] != prev[1]:
                self.switches += 1
            self.previous[i] = (tick, track["track_id"])
            full = bool(data["gt_fully_visible"][frame, i]) and not detection.truncated
            if full:
                self.position_errors.append(float(np.linalg.norm(
                    np.asarray(track["position_world_m"]) - data["gt_state"][frame, i, :2])))
            associations.append(dict(gt_id=str(data["gt_ids"][i]), track_id=track["track_id"], iou=iou))
        # Require a full 2 s GT visibility window. Include missed and abstained tracks in the denominator.
        eligible = np.all(data["gt_fully_visible"][max(0, frame - 20):frame + 1], axis=0) if frame >= 20 else np.zeros(len(data["gt_ids"]), bool)
        for i in np.flatnonzero(eligible):
            kind = int(data["gt_kind"][i])
            track = matched_by_gt.get(int(i))
            state = track["motion_state"] if track else "missed"
            self.confusion[kind, ["missed", "unknown", "stationary", "moving"].index(state)] += 1
            if track and track["velocity_valid"] and not track["truncated"]:
                self.velocity_errors.append(float(np.linalg.norm(
                    np.asarray(track["velocity_world_mps"]) - data["gt_state"][frame, i, 5:7])))
        return associations

    def summary(self):
        precision = self.matched / self.detected if self.detected else None
        recall = self.matched / self.gt if self.gt else None
        known = int(self.confusion[:, 2:].sum())
        eligible = int(self.confusion.sum())
        correct = int(self.confusion[0, 2] + self.confusion[1, 3])
        def errors(values):
            return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                        p95=float(np.percentile(values, 95)) if values else None)
        return dict(frames=self.frames, visible_object_frames=self.gt, detections=self.detected, matches=self.matched,
                    instance_precision=precision, instance_recall=recall,
                    instance_f1=(2 * self.matched / (self.detected + self.gt) if self.detected + self.gt else None),
                    identity_switches=self.switches, position_error_m=errors(self.position_errors),
                    velocity_error_mps=errors(self.velocity_errors),
                    motion_confusion=dict(rows=["stationary", "moving"], columns=["missed", "unknown", "stationary", "moving"], counts=self.confusion.tolist()),
                    eligible_motion_object_frames=eligible, known_motion_object_frames=known,
                    motion_known_coverage=known / eligible if eligible else None,
                    motion_accuracy_when_known=correct / known if known else None,
                    moving_recall_including_unknown=(self.confusion[1, 3] / self.confusion[1].sum() if self.confusion[1].sum() else None))


def evaluate(root, model, output, *, splits=("val",), threads=2):
    root, model, output = Path(root), Path(model), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new report directory: {output}")
    manifest, audit = audit_dataset(root)
    perception = OnnxPerception(model, threads=threads)
    if perception.classes != DETAIL_CLASSES:
        raise ValueError("Use the current 6-class perception model")
    output.mkdir(parents=True)
    model_metrics, oracle_metrics, profile_metrics, timings = {}, {}, {}, []
    records, per_clip = [], []
    for clip in manifest["clips"]:
        split = clip["split"]
        if split not in splits:
            continue
        learned = model_metrics.setdefault(split, Metrics())
        oracle = oracle_metrics.setdefault(split, Metrics())
        profile = profile_metrics.setdefault(f"{split}/{clip['profile']}", Metrics())
        learned.start_clip(); oracle.start_clip(); profile.start_clip()
        tracker, oracle_tracker = VesselTracker(perception.classes), VesselTracker(perception.classes)
        clip_metrics = Metrics()
        frames = []
        with np.load(root / clip["path"], allow_pickle=False) as data:
            for j in range(clip["frames"]):
                started = time.perf_counter()
                maps = perception.predict(data["rgb"][j])
                cnn_end = time.perf_counter()
                tracks = tracker.update(maps["labels"], data["pose"][j], int(data["tick"][j]),
                                        ego_velocity=data["vel"][j], probabilities=maps["vessel_probability"])
                tracking_end = time.perf_counter()
                if j > 0:
                    timings.append([(cnn_end - started) * 1000, (tracking_end - cnn_end) * 1000])
                association = learned.add(j, data, tracker, tracks)
                profile.add(j, data, tracker, tracks)
                clip_metrics.add(j, data, tracker, tracks)
                oracle_tracks = oracle_tracker.update(data["semantic"][j], data["pose"][j], int(data["tick"][j]), ego_velocity=data["vel"][j])
                oracle.add(j, data, oracle_tracker, oracle_tracks)
                frames.append(dict(frame=j, tick=int(data["tick"][j]), pose=data["pose"][j].tolist(),
                                   tracks=tracks, offline_associations=association))
        records.append(dict(clip=clip, frames=frames))
        per_clip.append(dict(clip_id=clip["clip_id"], profile=clip["profile"], split=split, **clip_metrics.summary()))
        print(f"{clip['clip_id']}: recall={clip_metrics.summary()['instance_recall']:.3f}", flush=True)
    if not timings:
        raise ValueError("No matching clips")
    timings = np.asarray(timings)
    report = dict(format="usvnav-tracking-evaluation/1", audit=audit, dataset=str(root.resolve()),
                  model=str(model.resolve()), model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  tracking_sha256=hashlib.sha256((ROOT / "training/tracking.py").read_bytes()).hexdigest(),
                  split_selection=list(splits), classes=list(perception.classes), threads=threads,
                  metrics={s: m.summary() for s, m in model_metrics.items()},
                  oracle_semantic_diagnostic={s: m.summary() for s, m in oracle_metrics.items()},
                  profiles={s: m.summary() for s, m in profile_metrics.items()}, clips=per_clip,
                  timing_ms=dict(n=len(timings), perception_median=float(np.median(timings[:, 0])),
                                 tracking_median=float(np.median(timings[:, 1])),
                                 total_median=float(np.median(timings.sum(axis=1))),
                                 total_p95=float(np.percentile(timings.sum(axis=1), 95))),
                  protocol=dict(instance_iou_threshold=.2, minimum_ship_pixels=6, velocity_fit_window_s=2.,
                                min_velocity_history_s=1.2, motion_speed_threshold_mps=.25,
                                motion_rule="speed +/- 2 regression noise scales; this is a heuristic, not calibrated confidence",
                                motion_evaluation="GT fully visible continuously for 2 seconds; misses and unknowns included",
                                velocity_evaluation="Eligible full matched objects with valid velocity; coverage reported separately",
                                identity_switch="same GT linked to a new track within 0.5 seconds of the previous match"))
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "tracks.json").write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")
    write_report(report, output)
    return report


def write_report(report, output):
    def percent(value):
        return "해당 없음" if value is None else f"{100 * value:.1f}%"
    def error(value, unit):
        return "해당 없음" if value["mean"] is None else f"{value['mean']:.3f} {unit} (n={value['n']})"
    lines = ["# 탑뷰 선박 추적 — 연속 장면 초기 평가", "",
             "기존 6종 ONNX 인식 모델에 좌표 보정·선박 연결·속도 회귀를 연결했다. 추적 모델을 별도로 학습하지 않았다.", "",
             "입력: RGB, 공개 자기 좌표·방향·지상 속도, tick. 실제 객체 ID·정답 속도·코스의 숨겨진 물체는 평가에만 사용했다.", "",
             f"전체 데이터: {report['audit']['courses']}개 코스, {report['audit']['clips']}개 클립, {report['audit']['frames']}프레임.",
             "클립당 4초 / 41프레임(기본 설정). 정지 관찰과 천천히 이동·회전하는 관찰을 자동 수집했다.",
             f"저장 데이터 검사: {report['audit']['status']}, 최소 선체 여유 {report['audit']['min_clearance_m']:.2f} m.", "",
             "## 실제 인식 모델을 사용한 결과", "",
             "| 지표 | " + " | ".join(report["metrics"]) + " |",
             "|---|" + "---|" * len(report["metrics"])]
    specs = [("선박 개체 정밀도", lambda m: percent(m["instance_precision"])),
             ("선박 개체 재현율", lambda m: percent(m["instance_recall"])),
             ("정답 연결이 바뀐 횟수", lambda m: str(m["identity_switches"])),
             ("위치 평균 오차", lambda m: error(m["position_error_m"], "m")),
             ("속도 벡터 평균 오차", lambda m: error(m["velocity_error_mps"], "m/s")),
             ("움직임 판단 가능 비율", lambda m: percent(m["motion_known_coverage"])),
             ("판단한 움직임의 정확도", lambda m: percent(m["motion_accuracy_when_known"])),
             ("모름·누락 포함 이동 선박 재현율", lambda m: percent(m["moving_recall_including_unknown"]))]
    for label, function in specs:
        lines.append("| " + label + " | " + " | ".join(function(m) for m in report["metrics"].values()) + " |")
    t = report["timing_ms"]
    lines += ["", "움직임 평가는 정답 선체 전체가 2초 연속 보인 객체를 대상으로 한다. 새 객체·화면에 일부만 보이는 객체·관측이 끊긴 객체의 움직임은 모름이다.",
              "속도 오차는 속도를 추정할 수 있었던 연결된 객체에서만 계산했다. 따라서 판단 가능 비율·누락을 함께 봐야 한다.",
              "정지/이동은 선박의 정답 종류를 맞히는 평가다. 속도 크기 0.25 m/s와 회귀 잡음으로 판단하며 확률을 보정한 통계적 신뢰구간은 아니다.",
              "위치 오차는 화면에 전체가 보이는 연결된 선박 중심 기준이다. 화면 가장자리의 잘린 선체 중심에는 원래 중심 편향이 생긴다.", "",
              f"CPU 인식+추적 중앙값 {t['total_median']:.1f} ms, 95백분위 {t['total_p95']:.1f} ms; 추적만 중앙값 {t['tracking_median']:.1f} ms.",
              "렌더링·주행 정책·통신은 포함하지 않아 공식 전체 에이전트 시간 제한의 검증을 대신하지 않는다.", "",
              "## 진단과 한계", "",
              "정답 영역을 넣은 추적 진단은 metrics.json의 oracle_semantic_diagnostic에 분리했다. 실제 모델 성능과 혼동하면 안 된다.",
              "현재 서로 붙은 선박 영역은 하나의 연결 영역으로 합쳐질 수 있다. 재등장한 선박의 장기 ID 보존, 급한 방향 변화, 긴 에피소드는 아직 검증하지 않았다.",
              "약 2초의 과거 관측으로 회귀하므로 곡선 항로·가속에서는 순간 속도보다 지연될 수 있다. 선박의 장축은 알 수 있지만 선수/선미 방향은 180도 모호하다.",
              "두 동작은 관찰을 위한 안전한 짧은 기동이다. 전문가 운전 데이터·목적지 완주·주행 점수의 검증이 아니다. 1-4 외란은 수집하지 않았다.", "",
              "## 다음 단계", "",
              "공식 시뮬레이터의 물리·관측·종료·경유지 판정을 그대로 사용하는 reset/step 강화학습 환경을 만든다.",
              "정책에는 인식 지도·추적 목록 외에 자기 상태·다음 경유지 좌표를 직접 전달한다. 인식 가중치는 우선 고정하고 주행 정책을 학습한다.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "training/data/tracking-pilot")
    parser.add_argument("--model", type=Path, default=ROOT / "training/runs/detail-unet8-01/perception.onnx")
    parser.add_argument("--out", type=Path, default=ROOT / "training/runs/tracking-pilot-01")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["val"])
    args = parser.parse_args()
    evaluate(args.data, args.model, args.out, splits=args.splits)


if __name__ == "__main__":
    main()
