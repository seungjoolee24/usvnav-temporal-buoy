"""Real RGB/ONNX/tracking PPO pilot with fixed held-out pre/post evaluation.

No expert actions, GT object features, or CNN updates. Run from the repo root.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from training.collect_perception import ROOT
from training.navigation_curriculum import write_curriculum
from training.navigation_env import NavigationEnv, RewardConfig
from training.navigation_policy import policy_kwargs
from training.perception_runtime import OnnxPerception
from training.policy_env import PolicyEnv


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_env(courses, perception, cutoff):
    return PolicyEnv(NavigationEnv(courses, training_tick_limit=cutoff), perception)


class PilotCallback(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output, self.episodes = Path(output), []
        self.metrics = []
        self.started = time.perf_counter()

    def _on_step(self):
        info = self.locals["infos"][0]
        if self.num_timesteps % 128 == 0:
            print(f"train: step={self.num_timesteps}, tick={info['tick']}, "
                  f"wp={info['waypoints_reached']}/{info['waypoints_total']}, "
                  f"wall={time.perf_counter() - self.started:.1f}s", flush=True)
        if self.locals["dones"][0]:
            row = {k: info[k] for k in ("course_index", "seed", "tick", "outcome", "official_outcome",
                                         "waypoints_reached", "waypoints_total", "distance_m", "fuel")}
            row.update(step=self.num_timesteps, episode_return=info["episode"]["r"])
            self.episodes.append(row)
            write_json(self.output / "train-episodes.json", self.episodes)
            print(f"train episode: {row}", flush=True)
        return True

    def _on_rollout_start(self):
        # SB3 trains after on_rollout_end; previous update metrics are available
        # at the NEXT rollout start. Capture the final update on training end.
        self.capture_update()

    def capture_update(self):
        values = self.model.logger.name_to_value
        if "train/n_updates" not in values:
            return
        row = {k: float(v) for k, v in values.items() if k.startswith("train/") and np.isscalar(v)}
        if not all(np.isfinite(v) for v in row.values()):
            raise RuntimeError("Non-finite PPO update metric")
        row["step"] = self.num_timesteps
        self.metrics.append(row)
        write_json(self.output / "update-metrics.json", self.metrics)

    def _on_training_end(self):
        self.capture_update()


def evaluate(model, paths, perception, cutoff, output, label):
    summaries = []
    for i, path in enumerate(paths):
        env = build_env([path], perception, cutoff)
        features, _ = env.reset(seed=1007 + i)
        initial_remaining = env.env.episode.remaining_route_m()
        trace, total_reward, inference_ms = [], 0., []
        started = time.perf_counter()
        for j in range(cutoff):
            t = time.perf_counter()
            action, _ = model.predict(features, deterministic=True)
            inference_ms.append((time.perf_counter() - t) * 1000)
            features, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            trace.append(dict(tick=info["tick"], pose=env.env._observation["pose"].tolist(),
                              action=info["applied_action"].tolist(), normalized_action=action.tolist(),
                              reward=reward, reward_terms=info["reward_terms"],
                              waypoints=info["waypoints_reached"], clearance_m=info["clearance_m"]))
            if j % 128 == 0:
                print(f"{label}/{path.stem}: tick={info['tick']}, wp={info['waypoints_reached']}/2", flush=True)
            if terminated or truncated:
                break
        ep = env.env.episode
        summary = {k: info[k] for k in ("tick", "seed", "outcome", "official_outcome", "waypoints_reached",
                                        "waypoints_total", "elapsed_s", "distance_m", "fuel")}
        summary.update(course=path.name, completed=info["official_outcome"] == "goal", episode_return=total_reward,
                       remaining_route_m=ep.remaining_route_m(), route_progress_m=initial_remaining - ep.remaining_route_m(),
                       final_pose=env.env._observation["pose"].tolist(),
                       min_clearance_m=float(min(ep.per_tick_clearance)),
                       actor_inference_median_ms=float(np.median(inference_ms)), wall_s=time.perf_counter() - started)
        summaries.append(summary)
        write_json(output / f"{label}-{path.stem}-trace.json", trace)
        write_json(output / f"{label}-evaluation.json", summaries)
        print(f"{label} result: {summary}", flush=True)
        env.close()
    return summaries


def report(output, config, before, after, metrics):
    lines = ["# Easy-course PPO pilot", "", "Frozen RGB perception → tracking/features → trainable shared features → actor/critic.",
             "Raw vessel state and ordered waypoint coordinates bypass perception. Actor chooses surge/yaw; critic estimates future return.",
             "", f"Actual training transitions: **{metrics['actual_steps']}**; PPO update rounds: **{len(metrics['updates'])}**.",
             f"Policy parameters: **{metrics['policy_parameters']:,}**; changed tensors: **{metrics['changed_parameter_tensors']}**.",
             "", "| Validation course | Phase | Outcome | Waypoints | Progress (m) | Return | Simulated time (s) |",
             "|---|---|---|---|---:|---:|---:|"]
    for rows, phase in ((before, "Before"), (after, "After")):
        for r in rows:
            lines.append(f"| {r['course']} | {phase} | {r['outcome']} | {r['waypoints_reached']}/{r['waypoints_total']} | "
                         f"{r['route_progress_m']:.2f} | {r['episode_return']:.2f} | {r['elapsed_s']:.1f} |")
    lines += ["", "Both evaluations use the same two validation courses and seeds, with deterministic actions. Their layouts/routes are excluded from training.",
              f"An unfinished episode is truncated at {config['cutoff']} ticks ({config['cutoff'] / 10:g}s); this is not the official 6000-tick timeout.",
              "Training uses stochastic actions sampled from PPO, without keyboard driving or expert demonstrations. All observations use actual simulator RGB, frozen ONNX and tracking.",
              "Four easy training courses and two validation courses are saved under courses/; all pass the official course validator and have distinct hashes.",
              "The official practice validation/test courses are not used in this pilot. No moving traffic or required obstacle avoidance is tested.",
              "", f"Frozen perception unchanged: **{metrics['perception_unchanged']}**. Save/load action check passed: **{metrics['save_load_match']}**.",
              "Policy timings include only the actor forward pass, not rendering, perception, tracking, process transport or the official response limit.",
              "This is a short training/pipeline experiment, not a submission-ready agent or a statistically reliable navigation benchmark.",
              "", "Files: initial-policy.zip, final-policy.zip, config.json, metrics.json, update-metrics.json, train-episodes.json, pre/post-evaluation.json, per-course trace JSON and SB3 CSV/JSON logs.",
              "", "PPO implementation and custom extractor follow [SB3 PPO](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html) and [custom policies](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/custom_policy.html).", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "training/runs/navigation-ppo-01")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--cutoff", type=int, default=800)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--perception", type=Path, default=ROOT / "training/runs/detail-unet8-01/perception.onnx")
    args = parser.parse_args()
    if args.steps < 256 or not 0 < args.cutoff < 6000:
        parser.error("Need at least 256 training steps and a cutoff within the official horizon")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    manifest = write_curriculum(output / "courses")
    paths = {s: [output / "courses" / r["file"] for r in rows] for s, rows in manifest["splits"].items()}
    perception = OnnxPerception(args.perception)
    perception_hash = digest(args.perception)
    env = Monitor(build_env(paths["train"], perception, args.cutoff))
    parameters = dict(learning_rate=5e-4, n_steps=256, batch_size=64, n_epochs=10,
                      gamma=.995, gae_lambda=.95, clip_range=.2, ent_coef=.001,
                      vf_coef=.5, max_grad_norm=.5, target_kl=.03)
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs(), seed=args.seed,
                device="cpu", verbose=0, **parameters)
    model.set_logger(configure(str(output / "logs"), ["csv", "json"]))
    initial_weights = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
    model.save(output / "initial-policy")
    config = dict(seed=args.seed, requested_steps=args.steps, cutoff=args.cutoff, condition="1-2",
                  ppo=parameters, reward=asdict(RewardConfig()), perception=str(args.perception.resolve()),
                  perception_sha256=perception_hash, curriculum=manifest, architecture=str(model.policy),
                  versions={p: importlib.metadata.version(p) for p in ("torch", "numpy", "gymnasium", "stable-baselines3", "onnxruntime")},
                  source_sha256={name: digest(ROOT / "training" / name) for name in (
                      "navigation_env.py", "policy_env.py", "policy_observation.py", "navigation_policy.py", "navigation_curriculum.py", "train_navigation.py")})
    write_json(output / "config.json", config)
    before = evaluate(model, paths["val"], perception, args.cutoff, output, "pre")
    callback = PilotCallback(output)
    model.learn(total_timesteps=args.steps, callback=callback)
    model.save(output / "final-policy")
    updated = model.policy.state_dict()
    changed = sum(not torch.equal(v, updated[k].cpu()) for k, v in initial_weights.items())
    if not changed or not callback.metrics or not all(torch.isfinite(v).all() for v in updated.values()):
        raise RuntimeError("No finite policy optimization was observed")
    loaded = PPO.load(output / "final-policy", device="cpu")
    probe_env = build_env(paths["val"][:1], perception, args.cutoff)
    probe, _ = probe_env.reset(seed=1007)
    expected, _ = model.predict(probe, deterministic=True)
    actual, _ = loaded.predict(probe, deterministic=True)
    np.testing.assert_array_equal(actual, expected)
    probe_env.close()
    after = evaluate(loaded, paths["val"], perception, args.cutoff, output, "post")
    unchanged = digest(args.perception) == perception_hash
    if not unchanged:
        raise AssertionError("Frozen perception model changed")
    metrics = dict(actual_steps=model.num_timesteps, policy_parameters=sum(p.numel() for p in model.policy.parameters()),
                   changed_parameter_tensors=changed, updates=callback.metrics, training_episodes=callback.episodes,
                   pre=before, post=after, perception_unchanged=unchanged, save_load_match=True,
                   wall_s=time.perf_counter() - started)
    write_json(output / "metrics.json", metrics)
    report(output, config, before, after, metrics)
    env.close()
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
