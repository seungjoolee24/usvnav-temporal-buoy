"""Review a completed raw-RGB PPO run; geometry is used only for this plot.

python -m training.review_rgb_run training/runs/rgb-ppo-stage1-02
Existing review files are preserved unless --overwrite is explicitly supplied.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as PlotCircle, Polygon, Rectangle
import numpy as np

from usvnav.coursefile import load
from usvnav.geometry import Circle, Rect
from usvnav.plant import HULL_LENGTH, HULL_WIDTH


PRE_COLOR, POST_COLOR = "#3975b4", "#d2642e"
OUTCOMES = {"goal": "완주", "training_cutoff": "학습용 시간 제한",
            "static_collision": "정적 장애물 충돌", "dynamic_collision": "이동 선박 충돌",
            "out_of_bounds": "강둑 이탈", "timeout": "공식 시간 초과"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def positions(course, trace):
    return np.asarray([course.start[:2]] + [row["pose"][:2] for row in trace], dtype=float)


def draw_geometry(axis, course, *, labels=True):
    axis.set_facecolor("#eee9de")
    axis.add_patch(Polygon(course.boundary, closed=True, facecolor="#e8f4f8",
                           edgecolor="#697e59", linewidth=1.5, zorder=0))
    for body in course.bodies:
        shape = body.shape
        color = {"buoy": "#bf9300", "pier": "#697078", "moored_vessel": "#536b87",
                 "dock": "#8b6550"}.get(body.cls, "#888888")
        if isinstance(shape, Circle):
            patch = PlotCircle((shape.x, shape.y), shape.r, facecolor=color,
                               edgecolor="#544400", linewidth=.7, zorder=6)
        else:
            patch = Polygon(shape.corners(), facecolor=color, edgecolor="#424951", zorder=3)
        axis.add_patch(patch)
        if labels and len(course.bodies) <= 8:
            axis.annotate(body.cls.replace("_", " "), (shape.x, shape.y),
                          xytext=(10, 17), textcoords="offset points", fontsize=9,
                          color=color, arrowprops=dict(arrowstyle="-", color=color, lw=.7))
    for moving in course.traffic_rects(0.):
        axis.add_patch(Polygon(moving.corners(), facecolor="#9a84b6", alpha=.7, zorder=3))
    chain = np.vstack([course.start[:2], course.waypoints])
    axis.plot(chain[:, 0], chain[:, 1], "--", color="#558c55", linewidth=1.1,
              label="Ordered target route", zorder=1)
    for index, (point, radius) in enumerate(zip(course.waypoints, course.arrival_radii)):
        axis.add_patch(PlotCircle(point, radius, fill=False, edgecolor="#558c55",
                                 linestyle=":", linewidth=1.2, zorder=1))
        if labels:
            name = "Goal" if index == course.n_waypoints - 1 else f"WP {index + 1}"
            axis.text(point[0], point[1] + radius + 1., name, fontsize=10,
                      color="#356535", horizontalalignment="center")
    axis.set_aspect("equal")


def draw_navigation(axis, course, before, after, traces):
    draw_geometry(axis, course)
    all_positions = []
    for phase, summary, color, style in (("pre", before, PRE_COLOR, "--"),
                                         ("post", after, POST_COLOR, "-")):
        trace = traces[phase]
        path = positions(course, trace)
        all_positions.append(path)
        axis.plot(path[:, 0], path[:, 1], color=color, linestyle=style, linewidth=2.,
                  label=f"{'Before' if phase == 'pre' else 'After'}: {summary['outcome']}", zorder=4)
        axis.scatter(*path[-1], color=color, marker="x", s=45, linewidths=1.6, zorder=7)
        if trace:
            x, y, heading = trace[-1]["pose"]
            hull = Rect(x, y, HULL_LENGTH, HULL_WIDTH, heading)
            axis.add_patch(Polygon(hull.corners(), facecolor=color, edgecolor=color,
                                   linewidth=1., alpha=.2, zorder=5))
    axis.scatter(*course.start[:2], color="#252d38", marker="s", s=35, label="Start", zorder=5)
    joined = np.vstack(all_positions + [course.waypoints])
    low, high = joined.min(axis=0), joined.max(axis=0)
    # Keep every recorded position visible, including a failed detour.
    span = high - low
    pad = np.maximum(span * .12, [8., 10.])
    axis.set_xlim(low[0] - pad[0], high[0] + pad[0])
    axis.set_ylim(low[1] - pad[1], high[1] + pad[1])
    axis.set_title("Held-out route: before / after (zoom)", fontsize=12)
    axis.set_xlabel("World x (m)")
    axis.set_ylabel("World y (m)")
    axis.grid(alpha=.18)
    axis.legend(fontsize=9, loc="lower left")

    inset = axis.inset_axes([.025, .68, .31, .26])
    draw_geometry(inset, course, labels=False)
    for path, color in zip(all_positions, (PRE_COLOR, POST_COLOR)):
        inset.plot(path[:, 0], path[:, 1], color=color, linewidth=.8)
    bounds = course.boundary
    inset.set_xlim(bounds[:, 0].min() - 5, bounds[:, 0].max() + 5)
    inset.set_ylim(bounds[:, 1].min() - 5, bounds[:, 1].max() + 5)
    inset.add_patch(Rectangle((low - pad), *(high - low + 2 * pad),
                              fill=False, edgecolor="#6f7781", linewidth=.7))
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("Full course", fontsize=8, pad=2)


def series(updates, key):
    return np.asarray([row.get(key, np.nan) for row in updates], dtype=float)


def make_figure(course, metrics, config, traces):
    fig = plt.figure(figsize=(15.5, 10.3), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.])
    draw_navigation(fig.add_subplot(grid[0, :2]), course, metrics["pre"], metrics["post"], traces)
    summary_axis = fig.add_subplot(grid[0, 2])
    summary_axis.axis("off")
    before, after = metrics["pre"], metrics["post"]
    rows = [["Outcome", before["outcome"], after["outcome"]],
            ["Waypoints", f"{before['waypoints_reached']}/{before['waypoints_total']}",
             f"{after['waypoints_reached']}/{after['waypoints_total']}"],
            ["Elapsed (s)", f"{before['elapsed_s']:.1f}", f"{after['elapsed_s']:.1f}"],
            ["Return", f"{before['reward']:.2f}", f"{after['reward']:.2f}"],
            ["Distance (m)", f"{before['distance_m']:.2f}", f"{after['distance_m']:.2f}"],
            ["Min recorded gap (m)", f"{before['min_clearance_m']:.2f}", f"{after['min_clearance_m']:.2f}"],
            ["Remaining route (m)", f"{before['remaining_route_m']:.2f}", f"{after['remaining_route_m']:.2f}"]]
    table = summary_axis.table(cellText=rows, colLabels=["Metric", "Before", "After"],
                                cellLoc="left", colWidths=[.53, .24, .23], bbox=[0., .27, 1., .64])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dae0e6")
        if row == 0:
            cell.set_facecolor("#eef2f5")
            cell.set_text_props(weight="bold")
        elif col in (1, 2):
            cell.get_text().set_color(PRE_COLOR if col == 1 else POST_COLOR)
    summary_axis.set_title("Same course and evaluation seed", fontsize=12, pad=8)
    goals, episodes = metrics["training_goals"], metrics["completed_training_episodes"]
    text = (f"Completed training episodes: {goals}/{episodes} goals\n"
            f"RGB weight change (L2): {metrics['rgb_weight_delta_l2']:.4g}\n"
            f"RGB tensors changed: {metrics['rgb_changed_tensors']}\n"
            f"Checkpoint reload matches: {metrics['save_load_match']}")
    summary_axis.text(0., .21, text, fontsize=10, va="top", linespacing=1.6)

    updates = metrics.get("updates", [])
    steps = np.arange(1, len(updates) + 1)
    loss_axis, rgb_axis, timing_axis = [fig.add_subplot(grid[1, index]) for index in range(3)]
    if updates:
        loss_axis.plot(steps, series(updates, "train/value_loss"), "o-", color="#3975b4",
                        label="Value loss")
        kl_axis = loss_axis.twinx()
        kl_axis.plot(steps, series(updates, "train/approx_kl"), "s--", color="#bb6b32",
                      label="Policy approx. KL")
        loss_axis.set_ylabel("Value loss", color="#3975b4")
        kl_axis.set_ylabel("Approx. KL", color="#bb6b32")
        loss_axis.set_title("PPO value error and policy change")

        rgb_axis.plot(steps, series(updates, "rgb_gradient_abs_sum"), "o-", color="#47905e")
        rgb_axis.set_title("RGB CNN gradient (last minibatch)")
        rgb_axis.set_ylabel("Sum of absolute gradients")
        rgb_axis.set_ylim(bottom=0.)

        rollout, update_time = series(updates, "rollout_wall_s"), series(updates, "update_wall_s")
        timing_axis.bar(steps, rollout, color="#8cb8d2", label="Experience collection")
        timing_axis.bar(steps, update_time, bottom=rollout, color="#dc9b63", label="PPO update + save")
        timing_axis.set_title("Recorded wall time per PPO block")
        timing_axis.set_ylabel("Seconds")
        timing_axis.legend(fontsize=8)
        for axis in (loss_axis, rgb_axis, timing_axis):
            axis.set_xlabel("PPO update block in this run")
            axis.grid(axis="y", alpha=.18)
            if len(updates) <= 8:
                axis.set_xticks(steps)
    else:
        for axis in (loss_axis, rgb_axis, timing_axis):
            axis.text(.5, .5, "No PPO updates recorded", ha="center", transform=axis.transAxes)
            axis.axis("off")
    fig.suptitle(f"Joint RGB PPO + coordinate steering prior | Stage {config['stage']} | "
                 f"{metrics['actual_additional_decisions']:,} additional decisions, "
                 f"{metrics['physics_ticks']:,} physics ticks", fontsize=15, weight="bold")
    fig.supxlabel("Offline review: geometry is plotted only. Goal rate is the navigation measure; "
                  "CNN gradients and PPO losses do not establish obstacle avoidance.", fontsize=10)
    return fig


def summary_markdown(run, metrics, config):
    before, after = metrics["pre"], metrics["post"]
    rows = ["# 로컬 RGB PPO 실험 결과", "",
            "좌표 추종 기본 명령에 신경망 보정을 더하며, 최근 4장 탑뷰의 CNN 표현과 PPO를 함께 학습했다.",
            f"이번 실행에서 **{metrics['actual_additional_decisions']:,}회 추가 의사결정**, "
            f"**{metrics['physics_ticks']:,}회 물리 틱**을 수집했다. "
            f"끝난 학습 에피소드 {metrics['completed_training_episodes']}개 중 "
            f"{metrics['training_goals']}개가 완주했다. 진행 중인 마지막 에피소드는 이 비율에 포함하지 않는다.", "",
            f"{config['stage']}단계 검증 코스 `{before['course']}`에서 같은 시드와 결정적 행동으로 비교했다.", "",
            "| 항목 | 학습 전 | 학습 후 |", "|---|---:|---:|"]
    pairs = [("결과", OUTCOMES.get(before["outcome"], before["outcome"]),
              OUTCOMES.get(after["outcome"], after["outcome"])),
             ("도달한 경유지", f"{before['waypoints_reached']}/{before['waypoints_total']}",
              f"{after['waypoints_reached']}/{after['waypoints_total']}"),
             ("진행 시간", f"{before['elapsed_s']:.1f}초", f"{after['elapsed_s']:.1f}초"),
             ("누적 보상", f"{before['reward']:.2f}", f"{after['reward']:.2f}"),
             ("실제 이동 거리", f"{before['distance_m']:.2f}m", f"{after['distance_m']:.2f}m"),
             ("남은 경로 거리", f"{before['remaining_route_m']:.2f}m", f"{after['remaining_route_m']:.2f}m"),
             ("기록된 최소 간격", f"{before['min_clearance_m']:.2f}m", f"{after['min_clearance_m']:.2f}m")]
    rows.extend(f"| {label} | {pre} | {post} |" for label, pre, post in pairs)
    if before["outcome"] == after["outcome"] == "goal":
        difference = after["elapsed_s"] - before["elapsed_s"]
        judgement = (f"두 정책 모두 완주했다. 학습 후 도달 시간은 {abs(difference):.1f}초 "
                     f"{'늘었다' if difference > 0 else '줄었다' if difference < 0 else '차이가 없었다'}.")
    elif after["outcome"] == "goal":
        judgement = "학습 전에는 미완주했고, 학습 후에는 이 검증 코스에서 완주했다."
    elif before["outcome"] == "goal":
        judgement = "학습 전에는 완주했지만 학습 후에는 미완주했다. 이전 성능이 유지되지 않았다."
    else:
        judgement = "학습 전후 모두 미완주하여 이 검증 코스의 완주 능력은 확보하지 못했다."
    rows.extend(["", judgement,
                 "진행 시간은 종료 또는 학습용 제한까지의 시간이다. 미완주 기록을 완주 시간과 비교하지 않는다. "
                 "최소 간격은 상한값이 적용된 기록이며, 남은 경로 거리와 실제 이동 거리는 서로 다른 지표다.", "",
                 "![주행 궤적과 학습 지표](review.png)", "",
                 f"RGB 가중치 텐서 {metrics['rgb_changed_tensors']}개가 바뀌었으며, "
                 f"변화 L2는 {metrics['rgb_weight_delta_l2']:.6g}이다. "
                 f"유한한 비영 기울기 확인: {metrics['rgb_nonzero_finite_gradients']}. "
                 f"저장 후 행동 일치: {metrics['save_load_match']}.",
                 "그래프의 CNN 기울기는 마지막 미니배치의 기록이다. PPO 손실 감소나 CNN 가중치 변경만으로 "
                 "장애물 인식·회피를 배웠다고 판단하지 않는다. 그림의 실제 객체 형상은 사후 분석에만 사용했다.",
                 "검증은 한 코스·한 시드의 비교이며, 일반적인 완주율이나 최종 제출 성능을 뜻하지 않는다.", ""])
    retention_file = run / "retention-stage0-evaluation.json"
    if retention_file.exists():
        retained = read_json(retention_file)
        rows.extend([f"이전 0단계 별도 검증: {OUTCOMES.get(retained['outcome'], retained['outcome'])}, "
                     f"경유지 {retained['waypoints_reached']}/{retained['waypoints_total']}, "
                     f"진행 시간 {retained['elapsed_s']:.1f}초.", ""])
    rows.extend(["[원본 수치](metrics.json) · [실험 설정](config.json) · "
                 "[최종 정책](final-policy.zip) · [학습 보고서](report.md)", ""])
    return "\n".join(rows)


def review(run, *, overwrite=False):
    run = Path(run).resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"Select an existing run directory: {run}")
    outputs = [run / "review.png", run / "review.md"]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Review already exists; use --overwrite to replace it")
    metrics, config = read_json(run / "metrics.json"), read_json(run / "config.json")
    before, after = metrics["pre"], metrics["post"]
    if before["course"] != after["course"] or before["seed"] != after["seed"]:
        raise ValueError("Before/after comparison requires the same course and evaluation seed")
    course_path = (run / "courses" / before["course"]).resolve()
    if not course_path.is_relative_to(run / "courses"):
        raise ValueError("Course must be inside the run's courses directory")
    course = load(course_path)
    traces = {phase: read_json(run / f"{phase}-trace.json") for phase in ("pre", "post")}
    for phase, trace in traces.items():
        points = positions(course, trace)
        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite trajectory: {phase}")
        if trace and trace[-1]["tick"] != metrics[phase]["tick"]:
            raise ValueError(f"Trace does not match finalized metrics: {phase}")
    temporary = [run / f"review.pending-{uuid4().hex}.png", run / f"review.pending-{uuid4().hex}.md"]
    figure = None
    try:
        figure = make_figure(course, metrics, config, traces)
        figure.savefig(temporary[0], dpi=160)
        temporary[1].write_text(summary_markdown(run, metrics, config), encoding="utf-8")
        for source, destination in zip(temporary, outputs):
            os.replace(source, destination)
    finally:
        if figure is not None:
            plt.close(figure)
        for path in temporary:
            if path.exists():
                path.unlink()
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        outputs = review(args.run, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
