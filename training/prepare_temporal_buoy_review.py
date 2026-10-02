"""Build model/environment review artifacts; never collect PPO training steps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as PlotCircle, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image
import torch
from stable_baselines3 import PPO

from usvnav.coursefile import load
from usvnav.plant import Vessel
from usvnav.render import top_view
from training.buoy_teacher import TrainingBuoyTeacher, teacher_rollout
from training.rgb_navigation_env import RgbNavigationEnv
from training.temporal_buoy_curriculum import write_temporal_buoy_draft, LEVELS
from training.temporal_buoy_policy import temporal_buoy_policy_kwargs


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def model_figure(output, parameters):
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set(xlim=(0, 14), ylim=(0, 8))
    ax.axis('off')

    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.12',
                                    edgecolor='#697586', facecolor=color, linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=10)

    def arrow(a, b):
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', color='#475467', linewidth=1.3))

    box(.25, 6.15, 2.5, 1., '최근 탑뷰 4장\n200×200 RGB ×4\n0.5초 간격 / 1.5초 범위', '#e5efff')
    box(3.3, 6.15, 2.7, 1., '프레임별 공유 CNN\n장면 특징 64차원\n부표 지도 100×100', '#e5efff')
    box(6.6, 6.15, 2.9, 1., '자기 이동·회전으로 정렬\n근거리 지도 20×20\n프레임별 공간 특징 400', '#e5efff')
    box(10.1, 6.15, 3.4, 1., '시간 순서대로 GRU(64)\n장면64 + 공간64 + 자기동작8\n매 관측 윈도우에서 상태 초기화', '#e5efff')
    for a, b in (((2.75, 6.65), (3.3, 6.65)), ((6., 6.65), (6.6, 6.65)), ((9.5, 6.65), (10.1, 6.65))):
        arrow(a, b)
    box(.25, 3.95, 2.5, 1.25, '자기 상태 이력\n위치·방향·속도·시점\n과거 이미지와 1:1 대응', '#e7f5ed')
    arrow((2.75, 4.6), (7.3, 6.15))
    box(.25, 1.8, 2.5, 1.25, '공개 경유지·목적지 좌표\n기하적으로 상대 좌표 계산\n다음 경유지 / 최종 목표', '#fff1d8')
    box(3.3, 1.8, 2.7, 1.25, '직접 좌표 분기 + 경로 MLP\n현재 상태도 직접 연결\nCNN을 거치지 않는 입력', '#fff1d8')
    arrow((2.75, 2.45), (3.3, 2.45))
    box(6.6, 3.65, 2.9, 1.3, '특징 결합: 621차원\n현재 공간·장면 + 시간 특징\n좌표·자기 상태·기본 명령', '#edf0f5')
    arrow((11.8, 6.15), (8.05, 4.95))
    arrow((8.05, 6.15), (8.05, 4.95))
    arrow((6., 2.45), (6.6, 3.85))
    arrow((2.75, 4.55), (6.6, 4.35))
    box(10.1, 3.65, 3.4, 1.3, 'Actor: 64→64→속도·회전 보정\nCritic: 64→64→가치 예측\nPPO + 학습 전용 부표 인식 손실', '#f3e8ff')
    arrow((9.5, 4.3), (10.1, 4.3))
    box(6.6, 1.45, 2.9, 1.25, '목표 좌표 기반 추종 명령\n0.1초마다 계산\n신경망 보정은 0.5초 유지', '#fff1d8')
    box(10.1, 1.45, 3.4, 1.25, '최종 명령 = 추종 + 신경망 보정\n공식 속도·회전 범위로 제한\n선체 충돌 판정은 매 0.1초', '#e7f5ed')
    arrow((6., 2.4), (6.6, 2.15))
    arrow((9.5, 2.1), (10.1, 2.1))
    arrow((11.8, 3.65), (11.8, 2.7))
    fig.suptitle(f'시계열 부표 회피 모델 초안 · {parameters:,} 파라미터 · 새 Actor/GRU는 미학습', fontsize=16)
    fig.tight_layout()
    fig.savefig(output / 'model-architecture.png', dpi=140)
    plt.close(fig)


def layout_figure(output, examples):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for ax, (row, course, _) in zip(axes, examples[::2]):
        boundary = course.boundary
        ax.fill(boundary[:, 0], boundary[:, 1], color='#e3f2fa')
        low, high = boundary[:, 1].min(), boundary[:, 1].max()
        ax.axhspan(low - 5, low, color='#c9bb9c')
        ax.axhspan(high, high + 5, color='#c9bb9c')
        route = np.vstack([course.start[:2], course.waypoints])
        ax.plot(route[:, 0], route[:, 1], '--', color='#3568bb', linewidth=1.4, label='공개 경로')
        ax.scatter(*course.start[:2], marker='>', s=70, color='#e06f72', label='자기 선박')
        for index, (point, radius) in enumerate(zip(course.waypoints, course.arrival_radii)):
            ax.add_patch(PlotCircle(point, radius, fill=False, color='#1d8a64', linewidth=1.1))
            ax.text(point[0], point[1] + radius + 1, '경유지' if index == 0 else '목적지', ha='center', fontsize=9)
        for index, body in enumerate(course.bodies):
            shape = body.shape
            ax.add_patch(PlotCircle((shape.x, shape.y), shape.r, color='#a58600'))
            ax.annotate(f'B{index+1}', (shape.x, shape.y), xytext=(4, 3), textcoords='offset points', fontsize=8)
        ax.set(xlim=(45, 148), ylim=(low - 4, high + 4), aspect='equal', xlabel='세계 x (m)', ylabel='세계 y (m)')
        ax.set_title(f'{row["level"]}단계: 부표 {row["count"]}개 / 수로 {high-low:.0f}m')
        ax.grid(alpha=.2)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('학습 환경 초안: 실제 부표 지름 0.6m · 공개 경로 45m +45m · 좌우 반전 배치도 포함', fontsize=13)
    fig.tight_layout()
    fig.savefig(output / 'environment-layouts.png', dpi=150)
    plt.close(fig)


def input_figure(output, examples):
    fig, axes = plt.subplots(3, 2, figsize=(9, 11))
    for index, (row, course, rgb) in enumerate(examples[::2]):
        for col in range(2):
            if col == 0:
                view, title = rgb, f'{row["count"]}개 부표: 원본 탑뷰 200×200'
            else:
                view, title = rgb[24:112, 72:128], '전방 부표 주변 확대 (설명용 / 입력 아님)'
            axes[index, col].imshow(view, interpolation='nearest')
            axes[index, col].set_title(title, fontsize=11)
            axes[index, col].set_xticks([])
            axes[index, col].set_yticks([])
    fig.suptitle('실제 공식 렌더러 출력 · 화면 위 = 선박 전방 · 큰 분홍색 물체 = 자기 선박', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / 'topview-examples.png', dpi=130)
    plt.close(fig)


def paired_figure(output, examples):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.7))
    for ax, (row, _, rgb) in zip(axes, examples[4:]):
        ax.imshow(rgb[24:112, 64:136], interpolation='nearest')
        ax.set_title('6개 부표 배치 A' if not row['mirror'] else '6개 부표 배치 A의 좌우 반전')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('자기 위치·방향·목표는 동일 / 부표 배치만 변경 (전방 확대)', fontsize=13)
    fig.tight_layout()
    fig.savefig(output / 'paired-layouts.png', dpi=130)
    plt.close(fig)


def temporal_figure(output, course):
    teacher = TrainingBuoyTeacher(course)
    env = RgbNavigationEnv([course], training_tick_limit=1800)
    obs, _ = env.reset(seed=109)
    # Real controls and 10Hz dynamics; no learner inference and no optimizer.
    for _ in range(20):
        obs, _, done, truncated, _ = env.step(teacher.residual(env._last_public))
        if done or truncated:
            raise RuntimeError('Review trajectory ended before the example window')
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.7))
    frames = []
    for index, (ax, (rgb, public)) in enumerate(zip(axes, env.frames)):
        pixels = rgb.transpose(1, 2, 0)
        tick = int(public['t'])
        Image.fromarray(pixels).save(output / f'temporal-{index}.png')
        ax.imshow(pixels, interpolation='nearest')
        ax.set_title(f't={tick*.1:.1f}s\nx={public["pose"][0]:.2f}, y={public["pose"][1]:.2f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        frames.append(dict(tick=tick, time_s=tick*.1, pose=public['pose'].tolist()))
    fig.suptitle('입력되는 실제 연속 탑뷰 4장 · 교사로 예시만 생성한 주행이며 새 모델의 성공 궤적이 아님', fontsize=12)
    fig.tight_layout()
    fig.savefig(output / 'temporal-sequence.png', dpi=140)
    plt.close(fig)
    env.close()
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False})
    torch.set_num_threads(2)
    plan = write_temporal_buoy_draft(output / 'courses')
    examples = []
    proofs = []
    for row in plan['review']:
        course = load(output / 'courses' / row['file'])
        rgb = top_view(course, Vessel(*course.start), 0.)
        Image.fromarray(rgb).save(output / f'raw-l{row["level"]}-mirror{int(row["mirror"])}.png')
        probe = teacher_rollout(course, seed=109, cutoff=1800)['result']
        proofs.append(dict(file=row['file'], **probe))
        examples.append((row, course, rgb))
        print(f'Review solvability {row["file"]}: {probe["outcome"]}, clearance={probe["min_clearance_m"]:.3f}', flush=True)
    # Review cases must be demonstrated feasible, including real hull dynamics.
    if any(row['outcome'] != 'goal' for row in proofs):
        write_json(output / 'solvability-failed.json', proofs)
        raise RuntimeError('Revise draft geometry; a review case failed its real-dynamics feasibility check')
    env = RgbNavigationEnv([examples[0][1]])
    obs, _ = env.reset(seed=109)
    model = PPO('MultiInputPolicy', env, policy_kwargs=temporal_buoy_policy_kwargs(),
                n_steps=256, batch_size=64, n_epochs=4, use_sde=True, seed=109, device='cpu', verbose=0)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    timings = []
    for _ in range(8):
        started = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        timings.append(time.perf_counter() - started)
    model.save(output / 'untrained-temporal-policy.zip')
    restored = PPO.load(output / 'untrained-temporal-policy.zip', device='cpu')
    np.testing.assert_array_equal(action, restored.predict(obs, deterministic=True)[0])
    parameters = sum(parameter.numel() for parameter in model.policy.parameters())
    env.close()
    model_figure(output, parameters)
    layout_figure(output, examples)
    input_figure(output, examples)
    paired_figure(output, examples)
    frames = temporal_figure(output, examples[4][1])
    metadata = dict(training_performed=False, approval_status='pending_user_confirmation',
                    new_policy_parameters=parameters, feature_dim=621, cnn_is_shared=True,
                    history_frames=4, history_span_s=1.5, gru_hidden=64,
                    gru_state_scope='Reset for every observed window; chronological frames retained',
                    motion_alignment='Only public vessel pose; no object GT or future frames',
                    inference_median_ms=float(np.median(timings[1:]))*1000,
                    checkpoint='untrained-temporal-policy.zip', save_load_match=True,
                    split_counts={key: len(plan[key]) for key in ('train', 'val', 'test', 'review')},
                    representative_solvability=proofs, temporal_frames=frames,
                    example_trajectory='Training-only geometric teacher; not learned policy performance',
                    source_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                        for name in ('temporal_buoy_policy.py', 'temporal_buoy_curriculum.py', 'prepare_temporal_buoy_review.py')})
    write_json(output / 'review.json', metadata)
    print(json.dumps({key: value for key, value in metadata.items() if key not in ('source_sha256', 'representative_solvability', 'temporal_frames')}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
