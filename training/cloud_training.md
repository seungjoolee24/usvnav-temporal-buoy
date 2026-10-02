# 시계열 부표 모델: GitHub + Colab + Drive

현재 작업에는 **`training/notebooks/train_temporal_buoy_colab.ipynb`**를 사용합니다.
최근 탑뷰 4장·자기 이동 정렬·GRU·직접 목표 좌표 분기와 부표 2→4→6개 환경을
포함합니다. 아래의 기존 RGB 노트북 안내는 이전 모델용입니다.

1. 준비한 코드와 코스 설정은 [GitHub 저장소](https://github.com/seungjoolee24/usvnav-temporal-buoy)에 있습니다.
2. 저장소가 공개 상태라면 [새 Colab 노트북](https://colab.research.google.com/github/seungjoolee24/usvnav-temporal-buoy/blob/main/training/notebooks/train_temporal_buoy_colab.ipynb)을
   바로 엽니다. Google 계정으로 로그인하며 별도 GitHub 승인은 필요 없습니다.
3. **런타임 → 런타임 유형 변경 → GPU**를 선택합니다. 소스 셀의 `GITHUB_URL`을
   실제 저장소 URL로 지정하고 `GITHUB_PRIVATE=False`를 유지합니다.
4. 소스 셀을 실행하면 공개 코드를 clone합니다. 제공한 소스 ZIP을 사용하는 경우에는
   `SOURCE_MODE='zip'`으로 바꾸어 업로드합니다.
5. Drive 저장 셀을 본인 계정으로 승인한 뒤 모델·환경 이미지를 확인합니다.
   원하는 구성이라면 `CONFIRM_ENVIRONMENT=True`로 설정합니다. 기본 첫 실험은
   1단계 4,096결정이며, 사용자 확인 없이 학습 셀이 진행되지 않습니다.
6. 결과에서 실제 GPU 학습 속도·완주·충돌·선체 여유 거리를 확인합니다. 같은 단계에서
   더 학습하거나, 검증 후 `STAGE`와 `RESUME_CHECKPOINT`를 변경해 다음 단계를 진행합니다.

결과는 `/content`에서 생성하고 Drive의 `MyDrive/usvnav-temporal-buoy-ppo/`에 30초마다
복사합니다. `latest-policy.zip`는 완료한 PPO 갱신마다 생성됩니다. 마지막 갱신 후
Drive 복사가 완료된 체크포인트에서 재개할 수 있습니다. 런타임이 갑자기 종료되면
직전 복사 이후 경험은 유실될 수 있습니다. 대용량 보조 데이터 NPZ는 주기 복사에서
제외하고 코드·코스·설정·체크포인트·수치·주행 기록을 보관합니다.

기존 단일 이미지 정책은 새 Actor/GRU와 호환되지 않으므로 새 temporal 모델로 시작합니다.
GPU는 신경망을 가속하지만 시뮬레이터 물리·충돌·이미지 생성은 CPU에서 실행합니다.
첫 실행의 실측 처리량으로 긴 학습의 시간을 계산하며 GPU 종류와 속도를 가정하지 않습니다.
[Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html)

## 이전 RGB + 경유지 PPO 노트북

탑뷰 CNN과 주행 정책을 함께 학습하는 `training.train_rgb_navigation`용 실행 노트북입니다. 최근 4장의 RGB 탑뷰와 공개 자기 상태·경유지 좌표를 입력하며, 각 이미지의 시점·자기 위치·방향·속도 이력도 전달합니다. 이를 통해 화면의 변화와 자기 움직임을 함께 해석할 수 있게 합니다. 좌표 기반 추종 제어기의 기본 행동에 학습된 보정을 더하므로, 조종을 무작위 초기 신경망만으로 처음부터 배우는 구조는 아닙니다. 탑뷰 표현은 PPO와 함께 갱신되지만, 장애물 없는 0단계 학습만으로 장애물 인식·이동 선박 예측을 배웠다고 판단하지 않습니다. 기존 ONNX 인식 모델이나 수집 데이터셋은 필요하지 않습니다. 환경이 실행될 때 RGB와 행동 경험을 생성합니다.

## 가장 빠른 시작: ZIP 업로드

1. 준비된 `training/cloud/usvnav-colab-source-v2.zip`를 사용합니다. 이후 코드가 바뀌면 프로젝트 폴더에서 새 이름으로 소스 ZIP을 다시 만듭니다.

   ```powershell
   .\.venv\Scripts\python.exe -B -m training.package_colab --out training/cloud/usvnav-colab-source-v3.zip
   ```

2. [Colab](https://colab.research.google.com/)의 **파일 → 노트북 업로드**로 `training/notebooks/train_rgb_navigation_colab.ipynb`를 엽니다.
3. **런타임 → 런타임 유형 변경 → GPU**를 선택하고 셀을 순서대로 실행합니다. 소스 로드 셀에서 `training/cloud/usvnav-colab-source-v2.zip`를 업로드합니다.
4. Google Drive 저장 셀을 직접 실행·승인합니다. 최신 로컬 정책을 이어 가려면 선택 셀의 `UPLOAD_LOCAL_CHECKPOINT=True`로 바꾸고 `training/runs/rgb-ppo-stage1-02/final-policy.zip`를 별도로 업로드합니다. 이 정책은 1단계까지 학습했으므로 `LOCAL_CHECKPOINT_STAGE=1`로 지정합니다. 소스 ZIP과 정책 ZIP은 다른 파일입니다.
5. 짧은 실행 검사가 통과하면 학습 셀을 실행합니다. 기본값은 **추가 10,000회 의사결정**이며, 업로드한 로컬 정책이 있으면 그 정책과 지정한 단계(`STAGE=1`)를 사용합니다. 없으면 실행 검사 정책으로 0단계(`STAGE=0`)를 이어 학습합니다. 0단계 로컬 정책을 업로드할 때는 `LOCAL_CHECKPOINT_STAGE=0`으로 바꿉니다. 실행 속도와 결과를 확인한 뒤 추가 학습량을 늘리세요. 2단계는 1단계와 이전 단계의 검증을 더 확인한 뒤 선택합니다.

ZIP에는 공식 `usvnav` 소스와 `training` 코드·코스 설정·노트북만 들어갑니다. 가상환경, 기존 데이터, 실험 결과, 체크포인트, Git 정보는 제외됩니다. 각 파일의 SHA-256을 ZIP 내부와 옆의 manifest JSON에 기록합니다. 이 명령은 파일을 외부로 전송하지 않습니다. 준비된 최신 패키지는 `training/cloud/usvnav-colab-source-v2.zip`입니다.

## GitHub로 연결하기

현재 폴더에는 `.git`이 없으므로 연결된 저장소를 전제로 하지 않습니다. 저장소를 직접 만든 뒤 공식 소스와 `training` 코드를 올렸다면, 노트북의 `SOURCE_MODE="github"`와 `GITHUB_URL`을 설정합니다. 소스는 `git clone`으로 내려받고, 결과는 Google Drive에 저장합니다. GitHub는 코드 보관, Colab은 실행, Drive는 체크포인트 보관 역할입니다.

공개 저장소라면 다음 주소로 노트북을 열 수도 있습니다. `<소유자>`, `<저장소>`, `<브랜치>`는 실제 값으로 바꿉니다.

```text
https://colab.research.google.com/github/<소유자>/<저장소>/blob/<브랜치>/training/notebooks/train_rgb_navigation_colab.ipynb
```

비공개 저장소는 본인 계정의 인증 절차가 필요합니다. 토큰을 노트북 코드나 URL에 붙여 넣지 마세요. ZIP 업로드 방식은 GitHub 인증 없이 쓸 수 있습니다. 여기서는 저장소 생성이나 업로드를 실행하지 않았습니다.

## 저장과 재개

노트북은 Google Drive의 `MyDrive/usvnav-rgb-ppo/` 아래에 실행별 폴더를 만듭니다. `latest-policy.zip`는 학습 중 저장되는 체크포인트, `final-policy.zip`는 정상 종료 시 저장되는 정책입니다. 로컬 업로드 정책은 `uploaded-local-policy-.../local-final-policy.zip`로 보관합니다. 별도 재개 셀의 `RESUME_CHECKPOINT`에 실제 파일을 지정하면 됩니다.

`--steps`는 **이번 실행에서 추가 수집할 PPO 의사결정 수**입니다. 물리 시뮬레이션의 0.1초 tick 수와 다를 수 있으며, 실제 수집량은 PPO rollout 단위 때문에 요청값보다 늘어날 수 있습니다. 재개는 정책과 최적화 상태를 이어 받되, 환경 상태·난수·미완료 rollout까지 완전히 동일하게 복원하는 방식은 아닙니다.

단계는 `--stage 0`의 장애물 없는 직선 추종, `1`의 방향 전환, `2`의 부표 하나 회피로 선택합니다. 높은 단계의 학습 풀에는 이전 단계 코스도 포함합니다. 1단계는 5개 중 1개, 2단계는 6개 중 2개가 이전 단계 코스입니다. 현재는 이동 선박 예측·회피를 완성한 단계까지 포함하지 않습니다. 각 단계의 코스 구성은 코드와 결과 설정에 기록됩니다. 선택한 단계의 추가 학습을 먼저 확인하고, 이전 단계도 포함한 검증 완주율과 충돌률을 보고 다음 단계로 넘어갑니다. 단계를 올리는 명령만으로 이전 능력이 유지된다고 보장되지는 않습니다.

## GPU가 빨라지는 부분과 남는 병목

GPU는 CNN 순전파·역전파와 PPO 가중치 갱신을 가속합니다. 공식 시뮬레이터의 물리 계산, 충돌 검사, NumPy RGB 렌더링은 CPU에서 실행됩니다. GPU가 있어도 경험 생성이 느리면 전체 속도 향상은 작을 수 있습니다. 노트북은 CUDA 가용성과 실제 장치 이름을 출력하고, CLI에 `--device auto` 또는 `cuda`를 전달합니다.

`--n-envs`로 환경을 병렬 실행할 수 있습니다. 먼저 1~2개로 속도와 메모리를 확인하고 늘리세요. Colab의 CPU 할당이 적은데 환경을 많이 만들면 오히려 느려질 수 있습니다. 기존 CPU 전용 `requirements-cpu.lock.txt`를 Colab에서 설치하면 CUDA Torch를 바꿀 수 있으므로, 노트북은 Colab의 Torch를 유지하고 필요한 패키지만 설치합니다.

현재 노트북 GPU가 NVIDIA CUDA 장치라면 로컬에서도 같은 CLI의 `--device cuda`를 사용할 수 있습니다. NVIDIA 드라이버와 CUDA 지원 Torch가 별도로 필요합니다. CPU 전용 Torch는 NVIDIA GPU가 있어도 `torch.cuda.is_available()`이 `False`입니다. AMD 내장 GPU는 CUDA 장치로 사용할 수 없습니다.

Colab의 GPU 종류·접속 가능 여부·사용 한도는 변동되며 보장되지 않습니다. 런타임 파일은 세션이 끝나면 사라질 수 있어 체크포인트는 Drive에 저장합니다. 소스와 학습 중 데이터를 `/content`에서 사용하고 Drive에는 주기적인 저장만 하는 편이 I/O에 유리합니다. [공식 Colab FAQ](https://research.google.com/colaboratory/faq.html), [공식 파일 입출력 예제](https://colab.research.google.com/notebooks/io.ipynb)

## 이 폴더에서 검증한 범위

노트북 JSON과 모든 Python 셀의 문법, ZIP의 포함·제외 파일과 SHA-256 검증을 수행합니다. Google 계정 접속, Drive mount, Colab GPU 실제 학습은 사용자가 노트북을 실행해야 확인됩니다. GPU 속도 향상을 실측한 결과로 해석하지 마세요.
