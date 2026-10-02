"""Prepare the reviewed courses and the temporal-policy GPU notebook locally."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from training.package_colab import ROOT, source_files


def cell(kind, identifier, source):
    result = dict(cell_type=kind, id=identifier, metadata={}, source=source.strip().splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def notebook():
    # Bootstrap source loading is self-contained: project helpers aren't loaded yet.
    return dict(nbformat=4, nbformat_minor=5,
        metadata=dict(colab=dict(name='train_temporal_buoy_colab.ipynb', provenance=[]),
                      accelerator='GPU', kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                      language_info=dict(name='python')),
        cells=[
cell('markdown', 'intro', '''
# 시계열 탑뷰 + 직접 목표 좌표: 다중 부표 PPO
최근 4장의 200×200 탑뷰 → 공유 CNN → 자기 이동 정렬 → GRU와 직접 목표 좌표 분기입니다.
부표 2→4→6개 환경을 사용합니다. **런타임 → 런타임 유형 변경 → GPU**를 선택하세요.
GitHub는 코드·코스, Colab은 연산, Google Drive는 체크포인트를 보관합니다.
환경 예시를 확인하고 승인 셀을 설정해야 학습합니다. 단계는 결과를 보고 수동으로 올립니다.
'''),
cell('code', 'runtime', '''
import os, sys, subprocess, json, time, hashlib, io, stat, zipfile, tempfile
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import torch
print('Python:', sys.version.split()[0], 'Torch:', torch.__version__)
if not torch.cuda.is_available():
    raise RuntimeError('GPU 런타임을 선택한 뒤 이 셀부터 다시 실행하세요.')
DEVICE = 'cuda'
print('GPU:', torch.cuda.get_device_name(0), 'CPU cores:', os.cpu_count())
'''),
cell('markdown', 'source-note', '''
## GitHub 소스 가져오기
아래 공개 GitHub 저장소에서 코드를 가져옵니다. 별도 GitHub 토큰은 필요 없습니다.
GITHUB_URL과 GITHUB_REF가 맞는지 확인하고 셀을 실행하세요.
필요하면 SOURCE_MODE='zip'으로 바꾸어 제공한 소스 ZIP을 업로드할 수도 있습니다.
'''),
cell('code', 'source', r'''
SOURCE_MODE = 'github'  # 'github' 또는 'zip'
GITHUB_URL = 'https://github.com/seungjoolee24/usvnav-temporal-buoy.git'
GITHUB_REF = 'main'
GITHUB_PRIVATE = False
PROJECT_ROOT = Path('/content/usvnav-temporal-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f'))
if SOURCE_MODE == 'github':
    from urllib.parse import urlparse
    import re
    parsed = urlparse(GITHUB_URL)
    if (parsed.scheme != 'https' or parsed.hostname != 'github.com' or parsed.username
        or parsed.password or parsed.query or parsed.fragment
        or not re.fullmatch(r'/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?/?', parsed.path)
        or 'YOUR_ACCOUNT' in GITHUB_URL):
        raise ValueError('GITHUB_URL에 실제 GitHub 저장소 HTTPS URL을 입력하세요.')
    command = ['git', 'clone', '--depth', '1']
    if GITHUB_REF:
        if GITHUB_REF.startswith('-'):
            raise ValueError('올바른 브랜치 또는 태그를 입력하세요.')
        command += ['--branch', GITHUB_REF]
    command += [GITHUB_URL, str(PROJECT_ROOT)]
    clone_environment = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    if GITHUB_PRIVATE:
        from google.colab import userdata
        from getpass import getpass
        try:
            token = userdata.get('GITHUB_TOKEN')
        except Exception:
            token = getpass('GitHub 저장소 읽기 토큰 (숨김 입력): ')
        if not token:
            raise ValueError('비공개 저장소를 읽을 인증 정보가 필요합니다.')
        try:
            with tempfile.TemporaryDirectory() as temporary:
                askpass = Path(temporary) / 'askpass.py'
                askpass.write_text('#!/usr/bin/env python3\nimport os,sys\nprint("x-access-token" if "username" in sys.argv[1].lower() else os.environ["USVNAV_GITHUB_TOKEN"])\n')
                askpass.chmod(0o700)
                clone_environment.update(GIT_ASKPASS=str(askpass), USVNAV_GITHUB_TOKEN=token)
                subprocess.run(command, env=clone_environment, check=True)
        finally:
            token = None
            clone_environment.pop('USVNAV_GITHUB_TOKEN', None)
    else:
        subprocess.run(command, env=clone_environment, check=True)
    print('Git commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip())
elif SOURCE_MODE == 'zip':
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError('소스 ZIP 하나만 선택하세요.')
    name, payload = next(iter(uploaded.items()))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        manifest = json.loads(archive.read('_source_manifest.json'))
        expected = set(manifest['files']) | {'_source_manifest.json'}
        if manifest.get('format') != 'usvnav-colab-source/1' or set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError('제공한 소스 ZIP을 사용하세요.')
        if sum(item.file_size for item in archive.infolist()) > 100_000_000:
            raise ValueError('소스 ZIP이 너무 큽니다.')
        for item in archive.infolist():
            if ('\\' in item.filename or ':' in item.filename or PurePosixPath(item.filename).is_absolute()
                or '..' in PurePosixPath(item.filename).parts or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('올바르지 않은 ZIP 경로')
            if item.filename != '_source_manifest.json':
                record, content = manifest['files'][item.filename], archive.read(item.filename)
                if len(content) != record['bytes'] or hashlib.sha256(content).hexdigest() != record['sha256']:
                    raise ValueError('소스 검증 실패: ' + item.filename)
        PROJECT_ROOT.mkdir(parents=True)
        archive.extractall(PROJECT_ROOT)
    print('Source ZIP SHA-256:', hashlib.sha256(payload).hexdigest())
    del uploaded, payload
else:
    raise ValueError('SOURCE_MODE는 github 또는 zip입니다.')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
print('Project:', PROJECT_ROOT)
'''),
cell('markdown', 'dependencies-note', '''
## 학습 패키지
현재 GPU용 Torch를 유지합니다. 로컬 CPU Torch 설치 파일은 사용하지 않습니다.
설치 후 아래 버전과 CUDA 확인이 실패하면 새 GPU 런타임에서 다시 실행합니다.
'''),
cell('code', 'dependencies', r'''
import importlib.metadata as metadata
torch_pin = Path('/content/usvnav-torch-constraint.txt')
torch_pin.write_text('torch==' + metadata.version('torch') + '\n')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-c', str(torch_pin),
               'numpy>=2,<3', 'gymnasium==1.2.3', 'stable-baselines3==2.9.0',
               'matplotlib>=3.9,<4', 'Pillow>=10'], check=True)
subprocess.run([sys.executable, '-c', "import torch; assert torch.cuda.is_available(); import gymnasium,stable_baselines3; print('GPU dependencies ready:',torch.__version__,stable_baselines3.__version__)"], check=True)
subprocess.run([sys.executable, '-B', '-m', 'training.train_buoy_navigation', '--help'], check=True)
'''),
cell('code', 'drive', '''
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive/usvnav-temporal-buoy-ppo')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print('Checkpoint backup:', DRIVE_ROOT)
'''),
cell('markdown', 'review-note', '''
## 실제 모델·환경 예시와 승인
모델과 코스의 이미지를 확인합니다. 1=부표2개/40m, 2=4개/30m, 3=6개/24m이며,
90m 경로에서 경유지와 목적지를 달성합니다. 학습60·검증30·최종시험60·검토6개를 분리했습니다.
원하는 구성이라면 다음 설정 셀의 CONFIRM_ENVIRONMENT를 True로 바꿉니다.
이는 아직 학습되지 않은 모델이며 예시 주행은 기하적 교사로 만든 것입니다.
'''),
cell('code', 'review', '''
from IPython.display import display, Image
COURSE_SOURCE = PROJECT_ROOT / 'training/courses/temporal-buoy-v1'
for name in ('model-architecture.png', 'environment-layouts.png', 'topview-examples.png', 'temporal-sequence.png'):
    display(Image(filename=str(COURSE_SOURCE / name)))
plan = json.loads((COURSE_SOURCE / 'manifest.json').read_text(encoding='utf-8'))
print('Course splits:', {k:len(plan[k]) for k in ('train','val','test','review')})
'''),
cell('code', 'settings', '''
CONFIRM_ENVIRONMENT = False  # 위 모델·환경을 확인한 뒤 True
STAGE = 1                  # 1=2개, 2=4개, 3=6개; 검증 후 수동 승급
STEPS = 4096               # 첫 실행. 이후 10_000~50_000 등으로 늘릴 수 있음
SEED = 109
WARMUP_EPOCHS = 8
RESUME_CHECKPOINT = ''     # Drive에 저장된 temporal 모델 latest/final-policy.zip
# 옛 single-frame Actor는 resume 불가. 새 temporal 체크포인트만 지정.
from training.colab_runtime import approve_curriculum_copy
RUN_ID = 'stage' + str(STAGE) + '-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
APPROVED_MANIFEST = approve_curriculum_copy(COURSE_SOURCE, PROJECT_ROOT / 'approved-courses' / RUN_ID, confirmed=CONFIRM_ENVIRONMENT)
if STAGE not in (1,2,3) or STEPS < 256:
    raise ValueError('STAGE=1/2/3, STEPS>=256으로 설정하세요.')
if STAGE > 1 and not RESUME_CHECKPOINT:
    raise ValueError('이전 단계 temporal 체크포인트와 검증 결과를 확인하고 재개 경로를 지정하세요.')
if RESUME_CHECKPOINT and not Path(RESUME_CHECKPOINT).is_file():
    raise FileNotFoundError(RESUME_CHECKPOINT)
'''),
cell('markdown', 'training-note', '''
## GPU 학습 실행
소스·이미지·PPO 버퍼는 /content에서 처리하고, 체크포인트·설정·수치·주행 기록을
30초마다 Drive에 복사합니다. 최신 정책은 PPO 갱신 완료마다 저장됩니다.
중단 시 마지막으로 복사된 latest-policy.zip에서 재개합니다.
신경망 연산은 GPU, 물리·충돌·이미지 렌더링은 CPU이므로 실제 속도는 측정해야 합니다.
'''),
cell('code', 'train', '''
from training.colab_runtime import run_with_backup
LOCAL_RUN = Path('/content/usvnav-runs') / RUN_ID
DRIVE_RUN = DRIVE_ROOT / RUN_ID
command = [sys.executable, '-B', '-u', '-m', 'training.train_buoy_navigation',
           '--policy', 'temporal', '--history-frames', '4', '--stage', str(STAGE),
           '--steps', str(STEPS), '--seed', str(SEED), '--device', 'cuda',
           '--warmup-epochs', str(WARMUP_EPOCHS), '--course-manifest', str(APPROVED_MANIFEST),
           '--out', str(LOCAL_RUN)]
if RESUME_CHECKPOINT:
    command += ['--resume', RESUME_CHECKPOINT]
print('Run:', RUN_ID)
execution = run_with_backup(command, project_root=PROJECT_ROOT, run_dir=LOCAL_RUN, backup_dir=DRIVE_RUN)
print('Saved to Drive:', execution)
'''),
cell('code', 'results', '''
metrics = json.loads((LOCAL_RUN / 'metrics.json').read_text())
updates = metrics['updates']
learning_s = sum(r['rollout_wall_s'] + r['update_wall_s'] for r in updates)
decisions = metrics['actual_additional_decisions']
print('This run total minutes:', round(metrics['wall_s']/60,2))
print('PPO decisions/sec:', round(decisions/max(learning_s,1e-6),2))
print('Estimated 50k PPO-only hours:', round(50_000*learning_s/max(decisions,1)/3600,2))
for phase in ('pre','post'):
    result = metrics[phase]
    print(phase, 'goals:', result['goals'], '/', result['episodes'], 'collisions:', result['collisions'])
    for row in result['rows']:
        print(' ',row['course'],row['outcome'],'clearance:',round(row['min_clearance_m'],3))
print('Latest:', DRIVE_RUN / 'latest-policy.zip')
print('Final:', DRIVE_RUN / 'final-policy.zip')
print('단계 승급은 완주율·충돌·여유 거리·이전 단계 검증을 확인한 후 결정합니다.')
''')])


def main():
    destination = ROOT / 'training/courses/temporal-buoy-v1'
    original = ROOT / 'training/runs/temporal-buoy-review-01'
    if not destination.exists():
        shutil.copytree(original / 'courses', destination)
        for name in ('model-architecture.png', 'environment-layouts.png', 'topview-examples.png',
                     'temporal-sequence.png', 'paired-layouts.png'):
            shutil.copy2(original / name, destination / name)
        shutil.copy2(original / 'report.md', destination / 'review.md')
    output = ROOT / 'training/notebooks/train_temporal_buoy_colab.ipynb'
    if output.exists():
        raise FileExistsError('Preserve existing notebook; edit it directly if needed')
    output.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print(output)
    print(destination)


def export_github_ready(destination):
    """Create a standalone source checkout; do not initialize the original folder."""
    destination = Path(destination).resolve()
    if not destination.is_relative_to(ROOT / 'training/cloud'):
        raise ValueError('Export must stay in training/cloud')
    destination.mkdir(parents=True, exist_ok=False)
    selected = source_files(ROOT)
    for source in selected:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    readme = '''# Temporal top-view buoy navigation

Training source for a shared CNN, ego-motion alignment, 64-unit GRU,
direct waypoint/goal input and residual PPO control. Four 200×200 RGB frames
span 1.5 seconds. Static buoy curriculum: 2 / 4 / 6 buoys, 40 / 30 / 24 m channels.

[Open the GPU notebook in Colab](https://colab.research.google.com/github/seungjoolee24/usvnav-temporal-buoy/blob/main/training/notebooks/train_temporal_buoy_colab.ipynb)

Open the public GitHub notebook in Colab and sign in to your Google account.
No GitHub token is needed. Select a GPU runtime, run the setup cells, review the examples and set
`CONFIRM_ENVIRONMENT=True` before the first 4,096-decision stage-1 experiment.

Checkpoints and logs are copied every 30 seconds to
`MyDrive/usvnav-temporal-buoy-ppo/`. Resume from a temporal-policy
`latest-policy.zip` or `final-policy.zip`. Stage promotion is manual and should
follow validation success, collision and hull-clearance checks.

This repository includes simulator source, training code and reviewed course
settings. Local environments, collected data, existing checkpoints and credentials
are excluded. No newly trained policy is claimed by this initial export.

[Korean setup instructions](training/cloud_training.md) ·
[Model and course review](training/courses/temporal-buoy-v1/review.md)

![Model](training/courses/temporal-buoy-v1/model-architecture.png)
![Courses](training/courses/temporal-buoy-v1/environment-layouts.png)
'''
    (destination / 'README.md').write_text(readme, encoding='utf-8')
    return dict(path=str(destination), files=len(selected)+1)


def write_upload_notebook():
    """Use the same training pipeline when private GitHub OAuth cannot open."""
    book = notebook()
    output = ROOT / 'training/notebooks/train_temporal_buoy_colab_upload.ipynb'
    book['metadata']['colab']['name'] = output.name
    for item in book['cells']:
        if item['id'] == 'source':
            item['source'] = [line.replace("SOURCE_MODE = 'github'", "SOURCE_MODE = 'zip'")
                              for line in item['source']]
        if item['cell_type'] == 'code':
            compile(''.join(item['source']), item['id'], 'exec')
    output.write_text(json.dumps(book, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    main()
