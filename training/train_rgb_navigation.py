"""Trainable raw-RGB + explicit-waypoint residual PPO, CPU/CUDA and resumable."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from training.collect_perception import ROOT
from training.navigation_env import RewardConfig
from training.rgb_curriculum import write_rgb_curriculum
from training.rgb_navigation_env import RgbNavigationEnv
from training.rgb_navigation_policy import rgb_policy_kwargs


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_latest(model, output):
    temporary = output / "latest-policy.pending.zip"
    model.save(temporary)
    os.replace(temporary, output / "latest-policy.zip")


class Progress(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output, self.updates, self.episodes = output, [], []
        self.physical_ticks = 0
        self.end_time = None

    def capture_update(self):
        if self.end_time is None:
            return
        row = {k: float(v) for k, v in self.model.logger.name_to_value.items()
               if k.startswith("train/") and np.isscalar(v)}
        if not all(np.isfinite(v) for v in row.values()):
            raise RuntimeError("Non-finite PPO update")
        encoder = self.model.policy.features_extractor.rgb_encoder
        gradients = [p.grad for p in encoder.parameters() if p.grad is not None]
        row.update(decisions=self.num_timesteps, physics_ticks=self.physical_ticks,
                   rollout_wall_s=self.rollout_seconds, update_wall_s=time.perf_counter() - self.end_time,
                   rgb_gradient_abs_sum=float(sum(g.abs().sum().item() for g in gradients)),
                   rgb_gradients_finite=all(torch.isfinite(g).all().item() for g in gradients))
        self.updates.append(row)
        write_json(self.output / "update-metrics.json", self.updates)
        save_latest(self.model, self.output)
        self.end_time = None

    def _on_rollout_start(self):
        self.capture_update()
        self.rollout_start = time.perf_counter()

    def _on_rollout_end(self):
        self.rollout_seconds = time.perf_counter() - self.rollout_start
        self.end_time = time.perf_counter()

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            self.physical_ticks += info["physics_ticks_advanced"]
            if done:
                row = {k: info[k] for k in ("course_index", "seed", "tick", "outcome", "official_outcome",
                                            "waypoints_reached", "waypoints_total", "elapsed_s", "distance_m", "fuel")}
                row.update(decisions=self.num_timesteps, reward=info["episode"]["r"])
                self.episodes.append(row)
                write_json(self.output / "train-episodes.json", self.episodes)
                print(f"episode: outcome={row['outcome']} wp={row['waypoints_reached']}/{row['waypoints_total']} "
                      f"time={row['elapsed_s']:.1f}s decisions={self.num_timesteps}", flush=True)
        if self.n_calls % 128 == 0:
            print(f"train: decisions={self.num_timesteps}, physics_ticks={self.physical_ticks}", flush=True)
        return True

    def _on_training_end(self):
        self.capture_update()


def evaluate(model, course_path, cutoff, output, phase, *, seed=101, history_frames=4):
    env = RgbNavigationEnv([course_path], training_tick_limit=cutoff, history_frames=history_frames)
    obs, _ = env.reset(seed=seed)
    initial_remaining = env.episode.remaining_route_m()
    trace, reward_sum, decisions, timings = [], 0., 0, []
    started = time.perf_counter()
    while True:
        t = time.perf_counter()
        residual = np.zeros(2, np.float32) if model is None else model.predict(obs, deterministic=True)[0]
        timings.append(time.perf_counter() - t)
        obs, reward, terminated, truncated, info = env.step(residual)
        decisions += 1
        reward_sum += reward
        trace.append(dict(tick=info["tick"], pose=env._last_public["pose"].tolist(),
                          residual=residual.tolist(), actions=info["applied_actions"],
                          reward=reward, waypoints=info["waypoints_reached"], clearance_m=info["clearance_m"]))
        if terminated or truncated:
            break
    ep = env.episode
    row = {k: info[k] for k in ("tick", "outcome", "official_outcome", "waypoints_reached", "waypoints_total",
                                 "elapsed_s", "distance_m", "fuel")}
    row.update(seed=seed, course=course_path.name, policy_decisions=decisions, reward=reward_sum,
               route_progress_m=initial_remaining - ep.remaining_route_m(), remaining_route_m=ep.remaining_route_m(),
               min_clearance_m=float(min(ep.per_tick_clearance)), wall_s=time.perf_counter() - started,
               policy_median_ms=float(np.median(timings)) * 1000,
               final_pose=env._last_public["pose"].tolist())
    write_json(output / f"{phase}-evaluation.json", row)
    write_json(output / f"{phase}-trace.json", trace)
    print(f"{phase}: {row}", flush=True)
    env.close()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "training/runs/rgb-ppo-01")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--cutoff", type=int, default=1200)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.steps < 128 or not 1 <= args.n_envs <= 16 or not 0 < args.cutoff <= 6000 or args.cutoff % 5:
        parser.error("Need steps>=128, 1..16 environments, cutoff<=6000 divisible by5")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use cpu/auto or GPU Colab runtime")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.perf_counter()
    plan = write_rgb_curriculum(output / "courses", args.stage)
    train_paths = [output / "courses" / row["file"] for row in plan["train"]]
    val_path = output / "courses" / plan["val"][0]["file"]
    factories = [lambda: Monitor(RgbNavigationEnv(train_paths, training_tick_limit=args.cutoff)) for _ in range(args.n_envs)]
    vector = DummyVecEnv(factories) if args.n_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    if args.resume:
        model = PPO.load(args.resume, env=vector, device=args.device)
        model.set_random_seed(args.seed)
    else:
        model = PPO("MultiInputPolicy", vector, policy_kwargs=rgb_policy_kwargs(), n_steps=128, batch_size=32,
                    n_epochs=4, learning_rate=3e-4, gamma=.995, gae_lambda=.95,
                    ent_coef=.0005, target_kl=.03, seed=args.seed, device=args.device, verbose=0)
        # Coordinate reference carries basic navigation; zero residual mean
        # starts at that baseline. CNN/route/critic remain trainable from scratch.
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
    model.set_logger(configure(str(output / "logs"), ["csv", "json"]))
    encoder = model.policy.features_extractor.rgb_encoder
    before_weights = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
    config = dict(stage=args.stage, seed=args.seed, requested_additional_decisions=args.steps, n_envs=args.n_envs,
                  device=str(model.device), resume=str(args.resume.resolve()) if args.resume else None,
                  physics_dt=.1, action_repeat=5, decision_dt=.5, cutoff_ticks=args.cutoff,
                  history_frames=4, observation_shapes={k: list(v.shape) for k, v in vector.observation_space.spaces.items()},
                  discount_per_decision=model.gamma, inner_rewards="sum; gamma=.995 per .5s decision (~100s horizon)",
                  reward=asdict(RewardConfig()), curriculum=plan, policy=str(model.policy),
                  versions={p: importlib.metadata.version(p) for p in ("torch", "numpy", "gymnasium", "stable-baselines3")},
                  source_sha256={name: hashlib.sha256((ROOT / "training" / name).read_bytes()).hexdigest()
                                 for name in ("rgb_navigation_env.py", "rgb_navigation_policy.py", "rgb_curriculum.py", "train_rgb_navigation.py")})
    write_json(output / "config.json", config)
    model.save(output / "initial-policy")
    initial_steps = model.num_timesteps
    before = evaluate(model, val_path, args.cutoff, output, "pre")
    callback = Progress(output)
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False, callback=callback)
    model.save(output / "final-policy")
    after_weights = encoder.state_dict()
    changed = sum(not torch.equal(v, after_weights[k].cpu()) for k, v in before_weights.items())
    delta = float(sum((v - after_weights[k].cpu()).square().sum().item() for k, v in before_weights.items()) ** .5)
    if not changed or not callback.updates or not any(row["rgb_gradient_abs_sum"] > 0 for row in callback.updates):
        raise RuntimeError("Joint RGB training was not observed")
    loaded = PPO.load(output / "final-policy", device=args.device)
    probe = RgbNavigationEnv([val_path], training_tick_limit=args.cutoff)
    observation, _ = probe.reset(seed=101)
    np.testing.assert_array_equal(model.predict(observation, deterministic=True)[0], loaded.predict(observation, deterministic=True)[0])
    probe.close()
    after = evaluate(loaded, val_path, args.cutoff, output, "post")
    metrics = dict(actual_additional_decisions=model.num_timesteps - initial_steps, physics_ticks=callback.physical_ticks,
                   completed_training_episodes=len(callback.episodes), training_goals=sum(r["outcome"] == "goal" for r in callback.episodes),
                   policy_parameters=sum(p.numel() for p in model.policy.parameters()), rgb_changed_tensors=changed,
                   rgb_weight_delta_l2=delta, rgb_nonzero_finite_gradients=all(r["rgb_gradients_finite"] for r in callback.updates),
                   save_load_match=True, pre=before, post=after, updates=callback.updates, wall_s=time.perf_counter() - started)
    write_json(output / "metrics.json", metrics)
    lines = ["# RGB 표현과 경유지 주행의 공동 PPO 학습", "",
             "입력은 최근 탑뷰 4장(원본 200×200), 공개 자기 상태·남은 경유지·최종 목적지·관측 이력이다.",
             "탑뷰 CNN과 주행 actor/critic을 함께 학습했다. 기존 인식 ONNX·정답 객체·정답 속도는 정책에 넣지 않았다.",
             "공개 좌표 기반 추종 명령에 신경망의 전진·회전 보정을 더한다. 초기 보정 평균은0이다.",
             "따라서 초기 완주는 기본 추종 제어가 제공하며, 이번 결과를 신경망이 독립적으로 조종을 배운 증거로 해석하면 안 된다.", "",
             f"실제 PPO 결정 {metrics['actual_additional_decisions']:,}회 / 물리 {metrics['physics_ticks']:,}틱 / "
             f"끝난 학습 에피소드 {metrics['completed_training_episodes']}개 중 완주 {metrics['training_goals']}개.",
             f"RGB CNN 변경 텐서 {changed}개, 가중치 L2 변화 {delta:.6f}; 유한한 학습 기울기와 저장/재로딩 행동 일치 확인.", "",
             "| 검증 | 결과 | 경유지 | 시뮬레이션 시간 | 보상 |",
             "|---|---|---|---:|---:|"]
    for phase, row in (("학습 전", before), ("학습 후", after)):
        lines.append(f"| {phase} | {row['outcome']} | {row['waypoints_reached']}/{row['waypoints_total']} | {row['elapsed_s']:.1f}초 | {row['reward']:.2f} |")
    lines += ["", "같은 검증 코스·시드에서 탐색을 끈 행동을 비교했다. 검증 코스는 학습 코스와 해시가 다르다.",
              "물리·충돌·경유지 판정은0.1초마다 수행하며 신경망 보정은0.5초마다 갱신한다. RGB는 결정마다 한 번만 만든다.",
              "할인율 .995와 GAE .95는0.5초 결정 단위이다. 이전0.1초 정책과 할인 시간 범위가 다르다.",
              "CNN 갱신은 공동 학습의 증거이며, 장애물 인식·회피 또는 선박 속도 예측 성능을 보장하지 않는다.",
              "0단계는 장애물 없는 직선이다.1단계는 방향 전환,2단계는 부표 한 개이다. 자동 승급은 없으며 각 단계 결과를 확인하고 --stage/--resume을 지정한다.",
              "움직이는 선박과1-4 외란을 포함하는 확장 커리큘럼은 아직 구현하지 않았다. 공식 제출용 ONNX 변환·시간 제한 검증도 이후 작업이다.", "",
              "[수치](metrics.json) · [설정](config.json) · [최종 체크포인트](final-policy.zip) · [Colab 실행 안내](../../cloud_training.md)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    vector.close()
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
