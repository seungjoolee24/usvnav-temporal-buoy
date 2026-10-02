"""Prepare and report checkpoint continuation with varied targets and buoys.

Call from a connected GPU Colab notebook. Preparation preserves the earlier
run and checkpoint, creates fresh directories, and never starts PPO itself.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from training.colab_runtime import approve_curriculum_copy, sync_run
from training.varied_buoy_curriculum import preview_curriculum, write_varied_buoy_curriculum


def prepare_varied_run(project_root, drive_root, previous_checkpoint, *, confirmed=False,
                       steps=8192, seed=209):
    if confirmed is not True:
        raise ValueError('Explicit approval of varied waypoints/buoys and continuation is required')
    if steps < 256 or steps % 256:
        raise ValueError('Choose a positive multiple of 256 PPO decisions')
    project_root, drive_root, previous_checkpoint = map(Path,
        (project_root, drive_root, previous_checkpoint))
    if not previous_checkpoint.is_file():
        raise FileNotFoundError(previous_checkpoint)
    run_id='stage2-varied-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    draft=project_root/'runtime-courses'/run_id
    plan=write_varied_buoy_curriculum(draft, project_root/'training/courses/temporal-buoy-v1')
    preview=preview_curriculum(draft)
    manifest=approve_curriculum_copy(draft,project_root/'approved-courses'/run_id,confirmed=confirmed)
    local_run=Path('/content/usvnav-runs')/run_id
    drive_run=drive_root/run_id
    command=[sys.executable,'-B','-u','-m','training.train_buoy_navigation',
             '--policy','temporal','--history-frames','4','--stage','2',
             '--steps',str(steps),'--seed',str(seed),'--device','cuda',
             '--warmup-epochs','2','--dataset-size','768','--aux-steps','4',
             '--course-manifest',str(manifest),'--resume',str(previous_checkpoint),
             '--out',str(local_run)]
    return dict(run_id=run_id,project_root=project_root,local_run=local_run,drive_run=drive_run,
                command=command,preview=preview,manifest=manifest,plan=plan,
                previous_checkpoint=previous_checkpoint,steps=steps)


def summarize_varied_run(prepared):
    root=prepared['local_run']
    metrics=json.loads((root/'metrics.json').read_text())
    group_of={row['file']:row['group'] for row in prepared['plan']['val']}
    groups={}
    for phase in ('pre','post'):
        rows=metrics[phase]['rows']
        grouped={}
        for group in sorted(set(group_of.values())):
            samples=[r for r in rows if group_of[r['course']]==group]
            grouped[group]=dict(episodes=len(samples),goals=sum(r['outcome']=='goal' for r in samples),
                collisions=sum('collision' in r['outcome'] for r in samples),
                worst_clearance_m=min(r['min_clearance_m'] for r in samples),
                mean_min_clearance_m=sum(r['min_clearance_m'] for r in samples)/len(samples))
        groups[phase]=grouped
    paired={}
    for phase in ('pre','post'):
        rows={r['course']:r for r in metrics[phase]['rows']}
        pairs={}
        for row in prepared['plan']['val']:
            if row.get('pair_id') and row['kind']!='clean':
                pairs.setdefault(row['pair_id'],[]).append(rows[row['file']])
        complete=[members for members in pairs.values() if len(members)==2]
        paired[phase]=dict(pairs=len(complete),both_goal=sum(all(r['outcome']=='goal' for r in pair) for pair in complete))
    total=sum(r['rollout_wall_s']+r['update_wall_s'] for r in metrics['updates'])
    summary=dict(run_id=prepared['run_id'],additional_decisions=metrics['actual_additional_decisions'],
                 minutes=metrics['wall_s']/60,ppo_decisions_per_s=metrics['actual_additional_decisions']/max(total,1e-6),
                 groups=groups,paired=paired,localization=metrics['localization_validation'],
                 checkpoint=str(prepared['drive_run']/'final-policy.zip'),
                 final_test_used=False)
    (root/'continuation-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    sync_run(root,prepared['drive_run'])
    print(f"추가 PPO {summary['additional_decisions']:,}스텝 / {summary['minutes']:.2f}분",flush=True)
    for group in sorted(set(group_of.values())):
        before,after=groups['pre'][group],groups['post'][group]
        print(f"{group}: 완주 {before['goals']}/{before['episodes']} → {after['goals']}/{after['episodes']}; "
              f"학습 후 충돌 {after['collisions']}, 최소 여유 {after['worst_clearance_m']:.2f}m",flush=True)
    print('좌우 반전 쌍 모두 완주:',paired,flush=True)
    print('Final checkpoint:',summary['checkpoint'],flush=True)
    return summary
