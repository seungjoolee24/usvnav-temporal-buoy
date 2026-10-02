"""Local collision-rich PPO with train-only buoy localization supervision."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from training.buoy_curriculum import write_buoy_curriculum
from training.buoy_policy import buoy_policy_kwargs
from training.temporal_buoy_policy import temporal_buoy_policy_kwargs
from training.buoy_supervision import generate_buoy_dataset, save_buoy_dataset
from training.navigation_env import RewardConfig
from training.rgb_navigation_env import RgbNavigationEnv
from training.train_rgb_navigation import Progress, evaluate, write_json
from training.collect_perception import course_identity
from usvnav.coursefile import from_dict, validate


def append_mined_courses(plan, course_dir, manifests):
    """Copy verified replay failures into training only, preserving provenance."""
    for group, manifest_path in enumerate(manifests):
        manifest_path = manifest_path.resolve()
        if manifest_path.is_dir():
            manifest_path = manifest_path / 'manifest.json'
        mined = json.loads(manifest_path.read_text(encoding='utf-8'))
        for index, row in enumerate(mined['train']):
            original = manifest_path.parent / row['file']
            content = original.read_bytes()
            course = from_dict(json.loads(content))
            digest = course_identity(course)
            if digest != row['sha256']:
                raise ValueError(f'Mined course hash mismatch: {original}')
            errors = validate(course)
            if errors:
                raise ValueError(f'Invalid mined course: {original}: {errors}')
            name = f'mined-{group:02d}-{index:03d}.json'
            (course_dir / name).write_bytes(content)
            copied = dict(row, file=name, kind='mined', source_manifest=str(manifest_path))
            plan['train'].append(copied)
    if manifests:
        write_json(course_dir / 'manifest.json', plan)
    return plan


def load_reviewed_curriculum(manifest_path, course_dir, stage, history_frames):
    """Use approved draft courses and exact proposed stage mixture.

    Repeated entries implement course sampling weights with the original env's
    uniform index sampler. They are course repeats, never old-action PPO replay.
    The independent final test split is kept out of the learning loop.
    """
    manifest_path = manifest_path.resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / 'manifest.json'
    proposal = json.loads(manifest_path.read_text(encoding='utf-8'))
    metadata = proposal['metadata']
    if metadata.get('approval_status') != 'approved_by_user':
        raise ValueError('This curriculum is pending user confirmation; do not start training yet')
    if metadata['history_frames'] != history_frames:
        raise ValueError('History length must match the reviewed curriculum')
    selected = [row for row in proposal['train'] if row['level'] <= stage]
    validation = [row for row in proposal['val'] if row['level'] == stage]
    course_dir.mkdir(parents=True, exist_ok=False)
    for row in selected + validation:
        content = (manifest_path.parent / row['file']).read_bytes()
        course = from_dict(json.loads(content))
        if course_identity(course) != row['sha256'] or validate(course):
            raise ValueError(f'Changed/invalid reviewed course: {row["file"]}')
        (course_dir / row['file']).write_bytes(content)
    sampling = metadata['sampling_percent'][str(stage)]
    weighted = []
    for key, quota in sampling.items():
        level = 0 if key == 'clean' else int(key)
        pool = [row for row in selected if row['level'] == level]
        if not pool:
            raise ValueError(f'Missing reviewed curriculum level {level}')
        weighted.extend(dict(pool[index % len(pool)], sampling_group=key) for index in range(quota))
    plan = dict(train=weighted, val=validation, metadata=dict(metadata, active_stage=stage,
                source_manifest=str(manifest_path), unique_train_courses=len(selected),
                learning_sampling_entries=len(weighted)))
    write_json(course_dir / 'manifest.json', plan)
    return plan


class BuoyTrainingEnv(RgbNavigationEnv):
    """Same public observations and true physics, plus public route cost."""

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.route_chain = np.vstack([self._last_public['pose'][:2], self.meta['waypoints']])
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        point = np.asarray(self._last_public['pose'][:2], float)
        start, delta = self.route_chain[:-1], np.diff(self.route_chain, axis=0)
        fractions = np.clip(np.einsum('ij,ij->i', point - start, delta) /
                            np.einsum('ij,ij->i', delta, delta), 0., 1.)
        deviation = float(np.linalg.norm(point - start - fractions[:, None] * delta, axis=1).min())
        cost = -.04 * min(deviation, 10.) ** 2 * info['physics_ticks_advanced'] * .1
        info['reward_terms']['route_deviation'] = cost
        info['route_deviation_m'] = deviation
        return observation, reward + cost, terminated, truncated, info


def tensor_batch(data, indices, device):
    rgb = torch.as_tensor(data['rgb'][indices], device=device).float() / 255.
    labels = torch.as_tensor(data['buoy'][indices], device=device).float()
    valid = torch.as_tensor(data['valid'][indices], device=device).float()
    return rgb, labels, valid


def localization_scores(extractor, data, device, batch_size=16):
    tp = fp = fn = valid_count = 0
    with torch.no_grad():
        for start in range(0, len(data['rgb']), batch_size):
            rgb, label, valid = tensor_batch(data, slice(start, start + batch_size), device)
            predicted = extractor.buoy_logits(rgb).sigmoid()[:, 0] >= .5
            target, mask = label > .5, valid > .5
            tp += int((predicted & target & mask).sum())
            fp += int((predicted & ~target & mask).sum())
            fn += int((~predicted & target & mask).sum())
            valid_count += int(mask.sum())
    return dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                f1=2 * tp / max(2 * tp + fp + fn, 1), false_positive_fraction=fp / max(valid_count, 1),
                true_positive_pixels=tp, false_positive_pixels=fp, missed_pixels=fn)


def auxiliary_step(model, data, indices):
    extractor = model.policy.features_extractor
    rgb, labels, valid = tensor_batch(data, indices, model.device)
    model.policy.optimizer.zero_grad()
    loss = extractor.segmentation_loss(labels, valid, rgb=rgb)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite localization loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(extractor.parameters(), 1.)
    model.policy.optimizer.step()
    return float(loss.detach())


class BuoyProgress(Progress):
    def __init__(self, output, data, plan, aux_steps, seed):
        super().__init__(output)
        self.data, self.plan, self.aux_steps = data, plan, aux_steps
        self.rng = np.random.default_rng(seed)
        self.auxiliary = []

    def _on_rollout_start(self):
        # Log PPO gradients before auxiliary zero_grad. Then supervise vision
        # BEFORE collecting fresh on-policy experience, never mid-rollout.
        self.capture_update()
        losses = [auxiliary_step(self.model, self.data,
                                self.rng.integers(len(self.data['rgb']), size=16))
                  for _ in range(self.aux_steps)]
        self.auxiliary.append(dict(decisions=self.num_timesteps, losses=losses))
        write_json(self.output / 'auxiliary-updates.json', self.auxiliary)
        self.rollout_start = time.perf_counter()

    def _on_step(self):
        result = super()._on_step()
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if done:
                task = self.plan['train'][info['course_index']]
                self.episodes[-1].update(buoy_count=task['count'], kind=task['kind'])
                write_json(self.output / 'train-episodes.json', self.episodes)
        return result


def evaluation_suite(model, paths, cutoff, output, phase, *, history_frames=4):
    rows = [evaluate(model, path, cutoff, output, f'{phase}-{index:02d}', history_frames=history_frames)
            for index, path in enumerate(paths)]
    summary = dict(episodes=len(rows), goals=sum(row['outcome'] == 'goal' for row in rows),
                   collisions=sum('collision' in row['outcome'] for row in rows),
                   cutoffs=sum(row['outcome'] == 'training_cutoff' for row in rows), rows=rows)
    write_json(output / f'{phase}-suite.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--stage', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--seed', type=int, default=84)
    parser.add_argument('--warmup-epochs', type=int, default=8)
    parser.add_argument('--dataset-size', type=int, default=512)
    parser.add_argument('--aux-steps', type=int, default=4)
    parser.add_argument('--val-layouts', type=int, default=4)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--policy', choices=('temporal', 'latest'), default='temporal')
    parser.add_argument('--history-frames', type=int, choices=(4, 8), default=4)
    parser.add_argument('--course-manifest', type=Path,
                        help='User-approved temporal curriculum manifest/directory')
    parser.add_argument('--warm-start-vision', type=Path,
                        help='Transfer only shared RGB stem/buoy head; actor and GRU remain new')
    parser.add_argument('--mined-courses', type=Path, action='append', default=[],
                        help='Replay-verified mined manifest/directory; repeat for multiple sets')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.steps < 256 or args.warmup_epochs < 0 or args.dataset_size < 64 or args.aux_steps < 0 or args.val_layouts < 1:
        parser.error('Need steps>=256, dataset>=64 and nonnegative warmup/aux steps')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if args.resume and args.warm_start_vision:
        parser.error('Choose resume or vision warm start, not both')
    torch.set_num_threads(2)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    plan = (load_reviewed_curriculum(args.course_manifest, output / 'courses', args.stage, args.history_frames)
            if args.course_manifest else write_buoy_curriculum(output / 'courses', args.stage,
                                                              seed=args.seed, val_layouts=args.val_layouts))
    append_mined_courses(plan, output / 'courses', args.mined_courses)
    train_paths = [output / 'courses' / row['file'] for row in plan['train']]
    val_paths = [output / 'courses' / row['file'] for row in plan['val']]
    reward = RewardConfig(goal=100., danger_distance_m=6., danger_per_s=2., time_per_s=.2)
    cutoff = plan.get('metadata', {}).get('cutoff_ticks', 1200)
    vector = DummyVecEnv([lambda: Monitor(BuoyTrainingEnv(train_paths, reward_config=reward,
                    history_frames=args.history_frames, training_tick_limit=cutoff))])
    if args.resume:
        model = PPO.load(args.resume, env=vector, device=args.device)
        model.set_random_seed(args.seed)
        expected = 'TemporalBuoyNavigationFeatures' if args.policy == 'temporal' else 'BuoyNavigationFeatures'
        if model.policy.features_extractor.__class__.__name__ != expected:
            raise ValueError(f'Resume requires {expected}; use vision warm start for the new architecture')
    else:
        kwargs = temporal_buoy_policy_kwargs() if args.policy == 'temporal' else buoy_policy_kwargs()
        kwargs['log_std_init'] = -3.
        model = PPO('MultiInputPolicy', vector, policy_kwargs=kwargs, n_steps=256, batch_size=64,
                    n_epochs=4, gamma=.995, gae_lambda=.95, learning_rate=3e-4,
                    ent_coef=.001, target_kl=.04, use_sde=True, sde_sample_freq=16,
                    seed=args.seed, device=args.device, verbose=0)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
        if args.warm_start_vision:
            previous = PPO.load(args.warm_start_vision, device=args.device).policy.features_extractor
            for name in ('rgb_encoder', 'buoy_head'):
                getattr(model.policy.features_extractor, name).load_state_dict(getattr(previous, name).state_dict())
    model.set_logger(configure(str(output / 'logs'), ['csv', 'json']))
    source_names = ('train_buoy_navigation.py', 'buoy_curriculum.py', 'buoy_policy.py', 'buoy_loss_candidate.py',
                    'temporal_buoy_policy.py', 'temporal_buoy_curriculum.py',
                    'buoy_supervision.py', 'rgb_navigation_env.py', 'train_rgb_navigation.py')
    source = output / 'source'
    source.mkdir()
    hashes = {}
    for name in source_names:
        path = Path(__file__).parent / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, source / name)
    config = dict(family='temporal-buoy-ppo-v2' if args.policy == 'temporal' else 'buoy-localization-assisted-ppo-v1',
                  policy=args.policy, history_frames=args.history_frames,
                  history_span_s=(args.history_frames - 1) * .5,
                  warm_start_vision=str(args.warm_start_vision) if args.warm_start_vision else None,
                  stage=args.stage, seed=args.seed,
                  requested_additional_decisions=args.steps, resume=str(args.resume) if args.resume else None,
                  physics_dt=.1, decision_dt=.5, cutoff_ticks=cutoff, device=str(model.device),
                  warmup_epochs=args.warmup_epochs, dataset_size=args.dataset_size,
                  aux_steps_per_rollout=args.aux_steps, reward=asdict(reward),
                  evaluation_reward=asdict(RewardConfig()),
                  mined_course_manifests=[str(path.resolve()) for path in args.mined_courses],
                  public_route_deviation_cost_per_s='.04*min(distance_to_route,10)^2',
                  observation_shapes={key: list(space.shape) for key, space in vector.observation_space.spaces.items()},
                  gSDE_refresh_decisions=16, curriculum=plan, source_sha256=hashes,
                  privileged_data_scope='Buoy pixel labels only in train-only auxiliary loss; no GT policy inputs')
    write_json(output / 'config.json', config)
    print('Generating train-only localization frames...', flush=True)
    data = generate_buoy_dataset(train_paths, count=args.dataset_size, seed=args.seed + 200)
    heldout = generate_buoy_dataset(val_paths, count=128, seed=args.seed + 1200)
    save_buoy_dataset(data, output / 'localization-train.npz')
    save_buoy_dataset(heldout, output / 'localization-validation.npz')
    write_json(output / 'localization-data.json', dict(train=data['metadata'], validation=heldout['metadata']))
    warmup = []
    rng = np.random.default_rng(args.seed + 100)
    for epoch in range(args.warmup_epochs):
        order = rng.permutation(len(data['rgb']))
        losses = [auxiliary_step(model, data, order[start:start + 16]) for start in range(0, len(order), 16)]
        scores = localization_scores(model.policy.features_extractor, heldout, model.device)
        row = dict(epoch=epoch + 1, loss=float(np.mean(losses)), **scores)
        warmup.append(row)
        write_json(output / 'localization-warmup.json', warmup)
        print(f'vision epoch={epoch+1}: loss={row["loss"]:.4f}, precision={scores["precision"]:.3f}, recall={scores["recall"]:.3f}', flush=True)
    model.save(output / 'initial-policy.zip')
    before = evaluation_suite(model, val_paths, cutoff, output, 'pre', history_frames=args.history_frames)
    extractor = model.policy.features_extractor
    weights = {key: value.detach().cpu().clone() for key, value in extractor.rgb_encoder.state_dict().items()}
    initial_steps = model.num_timesteps
    callback = BuoyProgress(output, data, plan, args.aux_steps, args.seed + 300)
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False, callback=callback)
    model.save(output / 'final-policy.zip')
    loaded = PPO.load(output / 'final-policy.zip', device=args.device)
    probe = RgbNavigationEnv([val_paths[0]], history_frames=args.history_frames)
    obs, _ = probe.reset(seed=101)
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
    probe.close()
    after = evaluation_suite(loaded, val_paths, cutoff, output, 'post', history_frames=args.history_frames)
    delta = float(sum((value - extractor.rgb_encoder.state_dict()[key].cpu()).square().sum()
                      for key, value in weights.items()) ** .5)
    metrics = dict(actual_additional_decisions=model.num_timesteps - initial_steps,
                   physics_ticks=callback.physical_ticks, completed_training_episodes=len(callback.episodes),
                   training_goals=sum(row['outcome'] == 'goal' for row in callback.episodes),
                   policy_parameters=sum(parameter.numel() for parameter in model.policy.parameters()),
                   rgb_weight_delta_l2=delta, save_load_match=True, pre=before, post=after,
                   localization_validation=localization_scores(extractor, heldout, model.device),
                   updates=callback.updates, wall_s=time.perf_counter() - started)
    write_json(output / 'metrics.json', metrics)
    vector.close()
    print(f'Completed: before={before["goals"]}/{before["episodes"]}, after={after["goals"]}/{after["episodes"]}; {output}', flush=True)


if __name__ == '__main__':
    main()
