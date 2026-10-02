"""Show the latest trained policy's saved successes and its buoy failure.

Only recorded traces are animated; no policy, sensor, or simulator is run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
import numpy as np

from usvnav.coursefile import load
from usvnav.geometry import Rect
from usvnav.plant import HULL_LENGTH, HULL_WIDTH
from training.review_rgb_run import draw_geometry, positions, read_json


ROOT = Path(__file__).resolve().parents[1]
SUCCESS_COLOR, FAILURE_COLOR, BEFORE_COLOR = "#286a9b", "#c64635", "#71808c"


def read_case(run, phase, course_run):
    summary = read_json(run / f"{phase}-evaluation.json")
    trace = read_json(run / f"{phase}-trace.json")
    course = load(course_run / "courses" / summary["course"])
    if not trace or trace[-1]["tick"] != summary["tick"]:
        raise ValueError(f"Trace mismatch: {phase}")
    if not np.allclose(trace[-1]["pose"], summary["final_pose"]):
        raise ValueError(f"Final pose mismatch: {phase}")
    return dict(course=course, summary=summary, trace=trace,
                positions=positions(course, trace), phase=phase, run=run)


def setup_axis(axis, case, title):
    course = case["course"]
    draw_geometry(axis, course, labels=False)
    points = np.vstack([case["positions"], course.waypoints])
    low, high = points.min(axis=0), points.max(axis=0)
    axis.set_xlim(low[0] - 8., high[0] + 8.)
    axis.set_ylim(low[1] - 18., high[1] + 18.)
    if high[1] - low[1] < 5.:
        axis.set_ylim(low[1] - 23., high[1] + 23.)
    axis.scatter(*course.start[:2], marker="s", s=40, color="#19232c", zorder=7, label="Start")
    axis.scatter(*course.waypoints[-1], marker="*", s=140, color="#39864a", zorder=7, label="Goal")
    for index, (point, radius) in enumerate(zip(course.waypoints, course.arrival_radii)):
        label = "Goal" if index == course.n_waypoints - 1 else f"WP {index + 1}"
        axis.text(point[0], point[1] + radius + 1.4, label, color="#367047", fontsize=11, ha="center")
    axis.set_title(title, fontsize=13, weight="bold", pad=12)
    axis.set_xlabel("World x (m)", fontsize=10)
    axis.set_ylabel("World y (m)", fontsize=10)
    axis.tick_params(labelsize=9)
    axis.grid(alpha=.17)


def end_hull(axis, case, color, *, alpha=.2):
    x, y, heading = case["trace"][-1]["pose"]
    axis.add_patch(Polygon(Rect(x, y, HULL_LENGTH, HULL_WIDTH, heading).corners(),
                           facecolor=color, edgecolor=color, alpha=alpha, zorder=8))
    axis.scatter(x, y, marker="x", color=color, s=40, zorder=9)


def trajectory_figure(straight, turn, buoy_before, buoy_after, output):
    figure, axes = plt.subplots(1, 3, figsize=(18., 6.3), layout="constrained")
    for axis, case, name in zip(axes[:2], (straight, turn), ("Straight", "Turn")):
        summary = case["summary"]
        setup_axis(axis, case, f"{name}: GOAL\n{summary['elapsed_s']:.1f} s | "
                              f"{summary['waypoints_reached']}/{summary['waypoints_total']} waypoints")
        path = case["positions"]
        axis.plot(path[:, 0], path[:, 1], color=SUCCESS_COLOR, lw=2.3, label="Latest policy", zorder=5)
        end_hull(axis, case, SUCCESS_COLOR)
        axis.legend(fontsize=9, loc="lower left")
    axis = axes[2]
    summary = buoy_after["summary"]
    setup_axis(axis, buoy_after, f"Buoy: COLLISION\n{summary['elapsed_s']:.1f} s | "
                                f"{summary['waypoints_reached']}/{summary['waypoints_total']} waypoints")
    for case, color, label, style in ((buoy_before, BEFORE_COLOR, "Before buoy training", "--"),
                                      (buoy_after, FAILURE_COLOR, "Latest policy", "-")):
        path = case["positions"]
        axis.plot(path[:, 0], path[:, 1], color=color, lw=2., linestyle=style, label=label, zorder=5)
        end_hull(axis, case, color)
    buoy = buoy_after["course"].bodies[0].shape
    axis.annotate("Buoy (diameter 0.6 m)", (buoy.x, buoy.y), xytext=(-48, -30),
                  textcoords="offset points", fontsize=9, color="#846509",
                  arrowprops=dict(arrowstyle="-", color="#846509"))
    axis.legend(fontsize=9, loc="lower left")
    inset = axis.inset_axes([.55, .62, .43, .32])
    draw_geometry(inset, buoy_after["course"], labels=False)
    for case, color, style in ((buoy_before, BEFORE_COLOR, "--"),
                               (buoy_after, FAILURE_COLOR, "-")):
        path = case["positions"][-16:]
        inset.plot(path[:, 0], path[:, 1], color=color, linestyle=style, linewidth=1.)
        end_hull(inset, case, color, alpha=.3)
    inset.set_xlim(buoy.x - 4.5, buoy.x + 4.5)
    inset.set_ylim(buoy.y - 3.5, buoy.y + 3.5)
    inset.set_title("Actual collision geometry", fontsize=8, pad=3)
    inset.tick_params(labelsize=6)
    inset.grid(alpha=.2)
    figure.suptitle("Latest trained policy | True saved trajectories", fontsize=19, weight="bold")
    figure.supxlabel("Straight and turn reached the goal. Buoy avoidance is not learned yet. "
                     "Hull and obstacle shapes are plotted only for offline review.", fontsize=12)
    figure.savefig(output, dpi=100)
    plt.close(figure)


def animation_figure(straight, turn, output):
    figure, axes = plt.subplots(1, 2, figsize=(12., 5.1), layout="constrained")
    tracks, hulls, counters, records = [], [], [], []
    for axis, case, name in zip(axes, (straight, turn), ("Straight", "Turn")):
        setup_axis(axis, case, name)
        track, = axis.plot([], [], color=SUCCESS_COLOR, linewidth=2., zorder=5)
        start_hull = Rect(*case["course"].start[:2], HULL_LENGTH, HULL_WIDTH, case["course"].start[2])
        hull = Polygon(start_hull.corners(), facecolor=SUCCESS_COLOR, edgecolor="#194263", zorder=8)
        axis.add_patch(hull)
        counter = axis.text(.03, .96, "", transform=axis.transAxes, fontsize=11, va="top",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        rows = [dict(tick=0, pose=list(case["course"].start), waypoints=0)] + case["trace"]
        tracks.append(track)
        hulls.append(hull)
        counters.append(counter)
        records.append(rows)
        axis.legend(fontsize=8, loc="lower left")
    figure.suptitle("Latest trained policy | Two recorded successful runs", fontsize=15, weight="bold")
    figure.supxlabel("Playback about 5x | Actual saved positions, headings, time and waypoint counts", fontsize=10)

    def update(frame):
        for case, rows, track, hull, counter in zip((straight, turn), records, tracks, hulls, counters):
            index = min(frame, len(rows) - 1)
            row = rows[index]
            x, y, heading = row["pose"]
            path = case["positions"][:index + 1]
            track.set_data(path[:, 0], path[:, 1])
            hull.set_xy(Rect(x, y, HULL_LENGTH, HULL_WIDTH, heading).corners())
            state = " | GOAL" if index == len(rows) - 1 else ""
            counter.set_text(f"t = {row['tick'] * .1:.1f} s | WP {row['waypoints']}/"
                             f"{case['course'].n_waypoints}{state}")
        return tracks + hulls + counters

    frames = max(len(rows) for rows in records)
    animation = FuncAnimation(figure, update, frames=frames, interval=100, blit=False, repeat=True)
    animation.save(output, writer=PillowWriter(fps=10), dpi=100)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-run", type=Path, default=ROOT / "training/runs/rgb-ppo-stage2-01")
    parser.add_argument("--straight-course-run", type=Path, default=ROOT / "training/runs/rgb-ppo-02")
    parser.add_argument("--turn-course-run", type=Path, default=ROOT / "training/runs/rgb-ppo-stage1-03")
    parser.add_argument("--out", type=Path, default=ROOT / "training/runs/local-progress-01")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    straight = read_case(args.latest_run, "retention-stage0", args.straight_course_run)
    turn = read_case(args.latest_run, "retention-stage1", args.turn_course_run)
    if any(case["summary"]["outcome"] != "goal" for case in (straight, turn)):
        raise ValueError("Successful trajectory examples must actually end at goal")
    buoy_before, buoy_after = (read_case(args.latest_run, phase, args.latest_run) for phase in ("pre", "post"))
    args.out.mkdir(parents=True, exist_ok=True)
    image_path, gif_path = args.out / "trajectories.png", args.out / "successful-runs.gif"
    if image_path.exists() or (args.gif and gif_path.exists()):
        raise FileExistsError("Visualization already exists; choose another --out directory")
    trajectory_figure(straight, turn, buoy_before, buoy_after, image_path)
    print(image_path.resolve(), flush=True)
    if args.gif:
        print("Rendering recorded success animation...", flush=True)
        animation_figure(straight, turn, gif_path)
        print(gif_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
