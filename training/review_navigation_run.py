"""Plot saved PPO validation trajectories and write a Korean experiment summary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as PlotCircle, Polygon
import numpy as np

from usvnav.coursefile import load
from usvnav.geometry import Circle


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    metrics, config = read(run / "metrics.json"), read(run / "config.json")
    # Preserve the exact recorded training sources before future iterations.
    snapshot = run / "source"
    snapshot.mkdir(exist_ok=True)
    for name, expected in config["source_sha256"].items():
        source = Path(__file__).resolve().parent / name
        saved = snapshot / name
        if saved.exists():
            candidate = saved
        else:
            candidate = source
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Recorded source no longer available: {name}")
        if not saved.exists():
            shutil.copy2(source, saved)
    completed_names = {r["course"] for r in metrics["pre"]} & {r["course"] for r in metrics["post"]}
    courses = [r for r in config["curriculum"]["splits"]["val"] if r["file"] in completed_names]
    fig, axes = plt.subplots(1, len(courses), figsize=(6.2 * len(courses), 5.5), layout="constrained", squeeze=False)
    for ax, item, before, after in zip(axes[0], courses, metrics["pre"], metrics["post"]):
        course = load(run / "courses" / item["file"])
        ax.set_facecolor("#e9f5f9")
        ax.add_patch(Polygon(course.boundary, closed=True, fill=False, edgecolor="#5e7251", linewidth=2))
        for body in course.bodies:
            shape = body.shape
            color = {"buoy": "#d9a900", "moored_vessel": "#536b87", "pier": "#7c7c7c", "dock": "#8b6550"}.get(body.cls, "#888888")
            if isinstance(shape, Circle):
                ax.add_patch(PlotCircle((shape.x, shape.y), shape.r, color=color))
            else:
                ax.add_patch(Polygon(shape.corners(), color=color, alpha=.8))
        chain = np.vstack([course.start[:2], course.waypoints])
        ax.plot(chain[:, 0], chain[:, 1], "--", color="#579e66", linewidth=1.5, label="Target route")
        ax.scatter(*course.start[:2], s=45, marker="s", color="#242e39", zorder=5, label="Start")
        for i, (point, radius) in enumerate(zip(course.waypoints, course.arrival_radii)):
            ax.add_patch(PlotCircle(point, radius, fill=False, edgecolor="#579e66", linestyle=":"))
            ax.text(point[0] + 1, point[1] + radius + 1, f"WP{i + 1}", fontsize=9, color="#367945")
        for phase, row, color in (("pre", before, "#4477cc"), ("post", after, "#dc692f")):
            trace = read(run / f"{phase}-{Path(item['file']).stem}-trace.json")
            positions = np.array([course.start[:2]] + [r["pose"][:2] for r in trace])
            ax.plot(positions[:, 0], positions[:, 1], color=color, linewidth=2,
                    label=f"{'Before' if phase == 'pre' else 'After'}: {row['waypoints_reached']}/2 WPs")
            ax.scatter(*positions[-1], s=35, color=color, marker="x", zorder=6)
        ax.set_title(f"{Path(item['file']).stem}\nBefore: {before['outcome']} | After: {after['outcome']}", fontsize=10)
        # Include the full travelled path even when the agent wanders beyond
        # the nominal route. Otherwise a failed run could be hidden by cropping.
        all_x = [course.start[0] - 8, course.waypoints[-1, 0] + 14]
        for phase in ("pre", "post"):
            saved = read(run / f"{phase}-{Path(item['file']).stem}-trace.json")
            all_x.extend(r["pose"][0] for r in saved)
        ax.set_xlim(min(all_x) - 3, max(all_x) + 3)
        ax.set_ylim(32, 118)
        ax.set_aspect("equal")
        ax.set_xlabel("World x (m)")
        ax.set_ylabel("World y (m)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Frozen RGB perception + PPO driving policy | {metrics['actual_steps']:,} real training steps", fontsize=12)
    fig.savefig(run / "validation-trajectories.png", dpi=160)
    plt.close(fig)
    outcomes = {"goal": "완주", "training_cutoff": "80초 수집 중단", "out_of_bounds": "강둑 이탈",
                "static_collision": "정적 물체 충돌", "dynamic_collision": "이동 선박 충돌", "timeout": "공식 시간 초과"}
    lines = ["# 첫 PPO 주행 실험 결과", "",
             "탑뷰 인식 모델은 고정한 채, 인식 지도·추적 선박·자기 상태·경유지 정보를 받는 주행 신경망을 학습했다.",
             "자기 상태와 경유지 좌표는 인식 신경망을 거치지 않고 직접 전달한다. actor는 전진·회전 행동을 정하고, critic은 앞으로 받을 누적 보상을 추정한다.", "",
             f"학습용 코스 4개 중 무작위로 선택되는 환경에서 실제 행동 **{metrics['actual_steps']:,}회**, PPO 갱신 **{len(metrics['updates'])}묶음**을 실행했다. 키보드 조작이나 기본 에이전트 시범 데이터는 사용하지 않았다.",
             f"학습하지 않은 쉬운 검증 코스 {len(courses)}개에서 같은 시드로 학습 전후를 비교했다. 확률적 탐색을 끄고 주행망이 선택한 행동을 사용했다.", "",
             "| 검증 코스 | 구분 | 결과 | 경유지 | 경로 거리 감소 | 누적 보상 | 진행 시간 |",
             "|---|---|---|---|---:|---:|---:|"]
    for before, after in zip(metrics["pre"], metrics["post"]):
        for row, phase in ((before, "학습 전"), (after, "학습 후")):
            outcome = outcomes.get(row["outcome"], row["outcome"])
            if row["outcome"] == "training_cutoff":
                outcome = f"{config['cutoff'] / 10:g}초 실험 중단"
            lines.append(f"| {row['course']} | {phase} | {outcome} | {row['waypoints_reached']}/2 | "
                         f"{row['route_progress_m']:.2f}m | {row['episode_return']:.2f} | {row['elapsed_s']:.1f}초 |")
    lines += ["", "경로 거리 감소는 ‘현재 위치→다음 경유지→나머지 경유지’의 남은 거리 감소량이다. 실제 이동 거리와 다르며, 경유지를 놓치고 지나가면 다시 줄어들 수 있다.",
              f"실험은 {config['cutoff'] / 10:g}초에서 중단했다. 공식 시간 제한 600초와 다르다. 이 실험의 미완주를 공식 시간 초과로 해석하면 안 된다.", "",
              "![학습 전후 실제 주행 궤적](validation-trajectories.png)", "",
              f"주행망 파라미터 **{metrics['policy_parameters']:,}개**, 실제 변경된 가중치 텐서 **{metrics['changed_parameter_tensors']}개**. "
              f"고정 인식 모델 유지 **{metrics['perception_unchanged']}**, 저장 후 재로딩 행동 일치 **{metrics['save_load_match']}**.", "",
              "이번 코스는 직진과 작은 각도 변화, 경로 밖 정적 물체로 구성되어 있다. 장애물 우회·이동 선박 조우·외란 대응 성능은 이 실험으로 확인하지 않았다.",
              f"단일 학습 시드와 검증 {len(courses)}개만 사용한 짧은 실험이므로, 성공률의 일반화나 최종 제출 성능을 뜻하지 않는다.",
              "전체 탑뷰 데이터셋을 미리 만들어 학습한 방식이 아니다. 현재 정책이 시뮬레이터에서 행동→관측→보상을 반복하며 PPO 경험을 수집했다.", "",
              "- [실험 설정](config.json)", "- [수치와 학습 지표](metrics.json)", "- [학습 에피소드](train-episodes.json)",
              "- [최종 학습 체크포인트](final-policy.zip)", "- [세부 보고서](report.md)", ""]
    (run / "summary-ko.md").write_text("\n".join(lines), encoding="utf-8")
    print(run / "summary-ko.md")


if __name__ == "__main__":
    main()
