"""Real RGB/ONNX rollout and replay against the official in-process simulator.

The temporary controller follows waypoints only. No policy optimizer or learned
driving agent is involved. The easy completion validates wiring, not obstacle avoidance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from usvnav import sim
from usvnav.coursefile import default_terrain_extent, load, to_dict, validate
from usvnav.geometry import Circle, Rect
from usvnav.png import write_png
from usvnav.world import BUOY, DOCK, MOORED, PIER, Course, StaticBody
from training.collect_perception import ROOT, course_identity
from training.navigation_env import NavigationEnv, NavigationEpisode, RewardConfig
from training.perception_runtime import OnnxPerception
from training.policy_env import PolicyEnv
from training.policy_observation import SHIP_FIELDS, STATE_FIELDS, WAYPOINT_FIELDS


def easy_course():
    boundary = np.array([[0., 35.], [400., 35.], [400., 115.], [0., 115.]])
    return Course(boundary, [StaticBody(MOORED, Rect(88, 93, 12, 4, 0)),
                             StaticBody(BUOY, Circle(105, 63, .3)),
                             StaticBody(PIER, Rect(150, 95, 12, 4, .2)),
                             StaticBody(DOCK, Rect(110, 100, 15, 6, 0))],
                  np.array([[90., 75.], [130., 75.]]), np.array([5., 3.]), (50., 75., 0.),
                  terrain_extent=default_terrain_extent(boundary))


class ReplayAgent:
    def __init__(self, actions):
        self.actions = actions

    def reset(self, meta):
        self.i = 0

    def act(self, public):
        if self.i >= len(self.actions):
            raise RuntimeError("Reference asked for an unexpected extra action")
        action = self.actions[self.i]
        self.i += 1
        return action


def rollout(name, course, perception, output, *, cutoff, follow_waypoint):
    env = PolicyEnv(NavigationEnv([course], training_tick_limit=cutoff), perception)
    features, reset_info = env.reset(seed=47)
    snapshots, actions, measurements, trace, rewards = [], [], [], [], []
    images = {}
    feature_shapes = {k: list(v.shape) for k, v in features.items()}
    last_info = reset_info
    for j in range(cutoff):
        raw = env.env._observation
        snapshots.append({k: np.asarray(v).copy() for k, v in raw.items() if k != "perception"})
        if j % 20 == 0:
            images[j] = raw["perception"].copy()
        if not env.observation_space.contains(features):
            raise AssertionError(f"Policy observation outside declared space at tick {j}")
        bearing = math.atan2(float(features["state"][10]), float(features["state"][11]))
        normalized = np.array([.6, np.clip(bearing / .6, -1., 1.)]) if follow_waypoint else np.array([-.12, .2])
        started = time.perf_counter()
        features, reward, terminated, truncated, info = env.step(normalized)
        measurements.append((time.perf_counter() - started) * 1000)
        actions.append(info["applied_action"].copy())
        trace.append(dict(tick=info["tick"], pose=env.env._observation["pose"].tolist(),
                          action=info["applied_action"].tolist(), reward=reward, terms=info["reward_terms"],
                          waypoints=info["waypoints_reached"], clearance=info["clearance_m"],
                          tracked_ships=info["tracked_ships"], dropped_tracks=info["dropped_tracks"]))
        rewards.append(reward)
        last_info = info
        if j % 100 == 0:
            print(f"{name}: tick={info['tick']}, wp={info['waypoints_reached']}/{info['waypoints_total']}", flush=True)
        if terminated or truncated:
            break
    if follow_waypoint and last_info["outcome"] != "goal":
        raise AssertionError(f"Easy wiring check did not complete: {last_info['outcome']}")
    tested = {"common_observation_frames": 0, "rgb_sample_frames": 0}
    def compare_observation(tick, meta, public):
        saved = snapshots[tick]
        for k, value in saved.items():
            np.testing.assert_array_equal(public[k], value, err_msg=f"{name}: tick={tick}, field={k}")
        tested["common_observation_frames"] += 1
        if tick in images:
            np.testing.assert_array_equal(public["perception"], images[tick])
            tested["rgb_sample_frames"] += 1
    reference_limit = sim.TICK_LIMIT if last_info["outcome"] == "goal" else len(actions)
    reference = sim.run_episode(course, ReplayAgent(actions), condition="1-2", seed=47, tick_limit=reference_limit,
                                record_trace=True, on_observation=compare_observation)
    actual = env.env.episode
    np.testing.assert_array_equal(reference.pose, [actual.vessel.x, actual.vessel.y, actual.vessel.psi])
    np.testing.assert_array_equal(reference.per_tick_clearance, actual.per_tick_clearance)
    np.testing.assert_array_equal(reference.trace, actual.trace)
    assert reference.distance == actual.distance and reference.fuel == actual.fuel
    assert reference.waypoints_reached == actual.wp_index
    if actual.outcome is not None:
        assert reference.outcome == actual.outcome
    if follow_waypoint:
        save_frames = [images[min(images)], images[sorted(images)[len(images) // 2]], env.render()]
        write_png(output / "easy-frames.png", np.concatenate(save_frames, axis=1))
    (output / f"{name}-trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    np.savez_compressed(output / f"{name}-public-replay.npz", actions=np.asarray(actions),
                        **{k: np.array([s[k] for s in snapshots]) for k in snapshots[0]})
    timings = np.asarray(measurements)
    result = dict(name=name, outcome=last_info["outcome"], official_outcome=actual.outcome,
                  reference_outcome=reference.outcome, ticks=len(actions), simulated_seconds=len(actions) * .1,
                  waypoints_reached=actual.wp_index, waypoints_total=course.n_waypoints,
                  return_sum=float(sum(rewards)), official_metrics=dict(distance=reference.distance, fuel=reference.fuel,
                                                                     clearance=reference.clearance, elapsed=reference.elapsed),
                  feature_shapes=feature_shapes, final_tracked_ships=last_info["tracked_ships"],
                  max_dropped_tracks=max(t["dropped_tracks"] for t in trace),
                  step_time_ms=dict(median=float(np.median(timings)), p95=float(np.percentile(timings, 95)),
                                    mean=float(np.mean(timings))), measured_steps_per_second=1000. / float(np.mean(timings)),
                  replay=dict(status="passed", **tested, exact_terminal_pose=True, exact_clearance_trace=True,
                              exact_control_effort=True, exact_applied_command_trace=True))
    env.close()
    print(f"{name}: {result['outcome']}, {result['ticks']} steps, replay passed", flush=True)
    return result


def check(output, model):
    output, model = Path(output), Path(model)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    easy = easy_course()
    messages = validate(easy)
    if any(m["level"] == "error" for m in messages):
        raise ValueError(f"Invalid easy course: {messages}")
    output.mkdir(parents=True)
    snapshot = to_dict(easy)
    (output / "easy-course.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    # Replay the exact serialized course, including course-file rounding.
    easy = load(output / "easy-course.json")
    practice_path = ROOT / "sets/practice/courses/practice-01.json"  # Training split only.
    practice = load(practice_path)
    before = hashlib.sha256(model.read_bytes()).hexdigest()
    perception = OnnxPerception(model)
    easy_result = rollout("easy", easy, perception, output, cutoff=800, follow_waypoint=True)
    practice_result = rollout("practice-train", practice, perception, output, cutoff=30, follow_waypoint=False)
    if hashlib.sha256(model.read_bytes()).hexdigest() != before:
        raise AssertionError("Perception weights changed")
    report = dict(status="passed", experiment="RL environment wiring and official replay, no policy learning",
                  gymnasium=__import__("gymnasium").__version__, condition="1-2", model=str(model.resolve()),
                  model_sha256=before, perception_frozen=True, policy_updates=0,
                  code_sha256={name: hashlib.sha256((ROOT / f"training/{name}.py").read_bytes()).hexdigest()
                               for name in ("navigation_env", "policy_env", "policy_observation")},
                  easy_course_validation=messages, easy_course_sha256=course_identity(easy),
                  practice_course_sha256=course_identity(practice), practice_course_split="train",
                  task_tick_limit=sim.TICK_LIMIT, action_physical_bounds=[[-.5, 2.], [-.6, .6]],
                  reward_config=asdict(RewardConfig()), state_fields=list(STATE_FIELDS), ship_fields=list(SHIP_FIELDS),
                  waypoint_fields=list(WAYPOINT_FIELDS), rollouts=[easy_result, practice_result])
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(report, output)
    return report


def write_report(report, output):
    lines = ["# 강화학습 주행 환경 — 연결 및 공식 실행 비교", "",
             "실험 대상은 reset/step 환경, 고정 인식 모델, 선박 추적, 정책 입력, 보상·종료 판정의 연결이다.",
             "주행 정책은 학습하지 않았다. 쉬운 코스에서는 경유지만 따라가는 임시 명령을 사용했다.",
             "기존 어려운 코스의 완주·장애물 회피·경쟁 점수를 의미하지 않는다.", "",
             "## 실행 결과", "", "| 장면 | 결과 | 물리 틱 | 경유지 | 실제 환경 step 중앙값 |",
             "|---|---|---:|---:|---:|"]
    for r in report["rollouts"]:
        lines.append(f"| {r['name']} | {r['outcome']} | {r['ticks']} | {r['waypoints_reached']}/{r['waypoints_total']} | {r['step_time_ms']['median']:.1f} ms |")
    lines += ["", "위 step 시간에는 시뮬레이터 렌더링·기하 검사·인식·추적·입력 변환이 포함된다. 공식 에이전트의 행동 응답 시간 제한과 직접 비교할 수 없다.",
              "practice-train은 기존 학습 코스에서 30틱(3초)만 실행한 관측·물리 연결 검사다.",
              "training_cutoff는 학습용 중단이며 공식 6000틱 시간 초과·완주 실패 판정과 구분한다.",
              "같은 실제 명령을 공식 run_episode에 넣어 관측 공통 필드, 일부 RGB 프레임, 모든 위치·명령·여유 거리와 최종 물리 상태를 비교했고 일치했다.", "",
              "## 정책 입력", "", "| 입력 | 모양 | 내용 |", "|---|---|---|",
              "| map | 7×50×50 | 6종 인식 확률 특징 + 유효 픽셀 비율. 고정·선박 전경은 4×4 최댓값, 물은 평균 |",
              "| state | 19 | 공개 위치·방향·속도·이전 명령·다음 경유지·남은 시간·선체 크기 |",
              "| ships | 32×16 | 가까운 추적 선박의 위치·속도·크기·정지/이동/모름·현재 관측 여부. 첫 열은 존재 마스크 |",
              "| waypoints | 32×4 | 남은 경유지의 몸체 좌표·도달 반경·존재 마스크. 인식 CNN을 거치지 않음 |", "",
              "지도 각 칸은 2m다. 전경 최댓값은 인식된 작은 부표를 축소할 때 평균으로 지우지 않기 위한 것이다.",
              "서로 다른 전경 채널의 합이 1일 필요는 없다. 정확한 충돌 검사는 원래 전체 선체·원·사각형·강둑으로 진행한다.",
              "실제 객체 ID·위치·속도·정답 분류·여유 거리는 정책 관측에 포함하지 않는다. info의 여유 거리·종료 지표·보상 항은 오프라인 진단용이다.",
              "선박 움직임 모름을 정지 속도 0으로 해석하면 안 된다. 속도 사용 가능 여부와 모름 마스크를 함께 전달한다.", "",
              "## 행동과 보상", "",
              "정책 행동은 [-1,1] 두 값이다. 실제 명령은 v=0.75+1.25a₀ m/s, w=0.6a₁ rad/s다. 0 행동의 전진 명령은 0.75m/s이며 정지 명령은 a₀=-0.6이다.",
              "실제 명령은 공식 sanitize와 slew·관성·횡미끄러짐·추력 분배가 있는 Vessel.step을 그대로 거친다.", "",
              "초안 보상: 남은 경유지 경로 거리 감소 1점/m, 경유지 +5, 완주 +50, 충돌·강둑 이탈·잘못된 명령 -100, 공식 시간 초과 -10.",
              "시간 비용 0.1점/s, 실제 명령 변화의 공식 제어 비용 0.01배, 2m 이내 선체 여유의 근접 비용 최대 0.5점/s를 뺀다.",
              "거리 변화는 음수도 허용해 반복 왕복으로 양의 전진 보상만 모을 수 없게 했다. 보상은 학습용 초안이며 다른 팀과 비교하는 공식 점수 자체가 아니다.",
              "공식 6000틱 제한은 유한한 과제의 종료이고 남은 시간을 상태에 포함한다. 짧은 수집 중단은 truncation으로 표시해 이후 PPO의 가치 추정을 이어갈 수 있다.", "",
              "## 다음 진행", "",
              "지도용 작은 CNN + 마스크를 사용하는 선박 특징 처리 + 자기 상태·경유지 직접 입력으로 PPO actor/critic을 만든다.",
              "직선 도달 → 회전·경유지 연결 → 고정 장애물 → 이동 선박 순으로 코스를 늘리며 완주율·충돌률·시간·제어 비용을 별도로 측정한다.",
              "학습은 학습 코스만 사용하고 검증 코스로 체크포인트를 선택한다. 기존 작은 선박 인식 누락은 정책 성공률 검사와 함께 보강할 항목이다.", "",
              f"![쉬운 코스의 초기·중간·완료 탑뷰]({(output / 'easy-frames.png').resolve().as_posix()})", "",
              "API 및 유한 시간 과제의 종료/수집 중단 구분은 [Gymnasium 환경 API](https://gymnasium.farama.org/api/env/)와 [시간 제한 처리 안내](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)를 따랐다.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def verify_saved_replays(output):
    """Verify stored physical/common traces in a fresh process using final sources."""
    output = Path(output)
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    for name, digest in report["code_sha256"].items():
        if hashlib.sha256((ROOT / f"training/{name}.py").read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed since the report: {name}")
    if hashlib.sha256(Path(report["model"]).read_bytes()).hexdigest() != report["model_sha256"]:
        raise ValueError("Perception model changed")
    checked = []
    for summary in report["rollouts"]:
        name = summary["name"]
        source = output / "easy-course.json" if name == "easy" else ROOT / "sets/practice/courses/practice-01.json"
        course = load(source)
        expected_hash = report["easy_course_sha256"] if name == "easy" else report["practice_course_sha256"]
        if course_identity(course) != expected_hash:
            raise ValueError("Course contents changed")
        ep = NavigationEpisode(course, condition=report["condition"], seed=47, tick_limit=report["task_tick_limit"])
        trace = json.loads((output / f"{name}-trace.json").read_text(encoding="utf-8"))
        with np.load(output / f"{name}-public-replay.npz", allow_pickle=False) as data:
            for j, action in enumerate(data["actions"]):
                public = sim.observation(ep.vessel, ep.course, ep.tick, ep.wp_index, ep.prev_applied)
                for key, value in public.items():
                    np.testing.assert_array_equal(value, data[key][j])
                ep.advance(action)
                np.testing.assert_array_equal(np.asarray([ep.vessel.x, ep.vessel.y, ep.vessel.psi], np.float32), trace[j]["pose"])
                assert ep.clearance == trace[j]["clearance"] and ep.wp_index == trace[j]["waypoints"]
                assert ep.remaining_route_m() >= 0
        assert ep.outcome == summary["official_outcome"]
        assert ep.fuel == summary["official_metrics"]["fuel"] and ep.distance == summary["official_metrics"]["distance"]
        checked.append(dict(name=name, ticks=ep.tick, status="passed"))
    result = dict(status="passed", code_sha256=report["code_sha256"], checks=checked)
    (output / "final-source-replay.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "training/runs/navigation-env-01")
    parser.add_argument("--model", type=Path, default=ROOT / "training/runs/detail-unet8-01/perception.onnx")
    parser.add_argument("--verify-replays", action="store_true", help="Check saved traces and final source hashes without neural reruns")
    args = parser.parse_args()
    if args.verify_replays:
        print(json.dumps(verify_saved_replays(args.out), indent=2))
    else:
        check(args.out, args.model)


if __name__ == "__main__":
    main()
