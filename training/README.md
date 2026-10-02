# 1-2 학습의 첫 단계

먼저 **탑뷰 → 장애물과 물의 영역** 인식에 쓸 데이터를 준비한다.
첫 실험은 물/고정 장애물/선박 3종이고, 두 번째 실험은 물/강둑/부표/구조물·암반/접안시설/선박 6종이다.
주행 에이전트 없이, 제공된 코스 안에서 선체 전체가 안전한 위치·방향과 시각을
샘플링한 뒤 공식 렌더러로 이미지와 정답을 함께 만든다. 시뮬레이터 코드는 수정하지 않는다.

## 프로젝트 전용 학습 환경

저장소 루트의 `.venv`를 사용한다. 아래 명령은 PowerShell에서 실행하며 환경 활성화는
필요하지 않다. 학습용 패키지는 이 환경에만 설치한다.

현재 로컬 환경은 Python 3.11.9, PyTorch 2.14.1+cpu, NumPy 2.1.2다.
이 PC에서 확인된 그래픽 장치는 AMD Vega 8이며 CPU용 PyTorch를 설치했다.

```powershell
.\.venv\Scripts\python.exe -B -m training.check_environment
.\.venv\Scripts\python.exe -B tests/run.py perception_dataset
```

`check_environment`는 생성한 학습 탑뷰 두 장을 불러와 작은 임시 신경망의 순전파,
자기 선박을 제외한 cross-entropy 손실, 역전파, AdamW 업데이트를 검증한다.
최종 인식 모델의 학습이나 성능 평가를 대신하지 않으며 모델 파일은 저장하지 않는다.

새 환경에서 재설치하려면 다음 명령을 사용한다. 잠금 파일은 CPU 환경용이다.
GPU 환경은 해당 장치에 맞는 PyTorch 설치 구성을 따로 선택해야 한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r training/requirements-cpu.lock.txt
```

PyTorch의 CPU/CUDA 설치 기준은 [공식 안내](https://pytorch.org/get-started/locally/)를 따른다.

## 데이터 생성

```powershell
.\.venv\Scripts\python.exe -B -m training.collect_perception
```

기본 출력은 `training/data/perception-pilot/`이며 코스당 16장, 총 64장이다.
이는 데이터 형식과 정답을 확인하는 소량 샘플이지 학습 규모의 데이터셋이 아니다.
출력 폴더가 이미 있으면 덮어쓰지 않는다. 다른 실행에는 새 경로를 지정한다.

```powershell
.\.venv\Scripts\python.exe -B -m training.collect_perception --out training/data/perception-v2 --samples-per-course 128
```

## 객체·환경 정리를 반영한 6종 실험

`training/perception_schema.py`에 RGB에서 관측 가능한 종류를 정의한다.
계류 선박과 이동 선박은 같은 `vessel`로 인식하고, 움직임은 이후 연속 관측에서 추정한다.
암반은 공식 표현대로 `pier`에 포함한다. 그림자·밝은 갑판·윤곽선·얕은 물색을 별도 장애물로 만들지 않는다.
자기 선체는 255로 제외하고, 실제 경계와 경유지 좌표는 이후 주행 모듈에 직접 전달한다.

| ID | 종류 | 미리보기 색 |
|---|---|---|
| 0 | `water`: 물 | 파랑 |
| 1 | `bank`: 강둑·육지 | 황갈색 |
| 2 | `buoy`: 부표 | 노랑 |
| 3 | `pier`: 구조물·암반 | 보라 |
| 4 | `dock`: 접안시설 | 주황갈색 |
| 5 | `vessel`: 다른 선박 | 청록 |
| 255 | 자기 선체 무시 | 회색 |

첫 데이터셋의 학습 코스에는 접안시설 픽셀이 없었다. `training/courses/dock-*.json`에
서로 다른 강둑·물체 배치의 작은 보완 코스를 작성했다. 세 코스 모두 공식 `coursefile.validate`를
통과했다. 보완 코스에는 교통 항로가 없고, 기존 연습 코스가 이동 선박 장면을 제공한다.
이 보완은 종류별 인식을 위한 것이며 좁은 통로·대기·후진·외란을 포괄하는 주행 훈련 세트가 아니다.

`training/courses/perception-splits.json`이 코스별 분할을 고정한다. 기존 practice-01/02는 학습,
practice-03은 검증, practice-04는 테스트를 유지하고, 보완 코스도 각각 한 분할에만 배정한다.
같은 코스의 위치·방향·시각을 바꾼 이미지가 다른 분할로 넘어가지 않게 한다.

```powershell
.\.venv\Scripts\python.exe -B -u -m training.collect_perception --split-file training/courses/perception-splits.json --out training/data/perception-detail-pilot --samples-per-course 16
.\.venv\Scripts\python.exe -B -u -m training.train_perception --dataset training/data/perception-detail-pilot --label-scheme detail --out training/runs/detail-unet8-01
.\.venv\Scripts\python.exe -B -u -m training.export_perception --checkpoint training/runs/detail-unet8-01/best.pt --dataset training/data/perception-detail-pilot
```

두 번째 데이터셋은 7코스·112장(학습 48, 검증 32, 테스트 32)의 소량 실험이다.
NPZ의 원래 `semantic` 3종 정답을 보존하고, 학습 로더가 `raster_class`에서 6종 정답을 만든다.
학습 설정·체크포인트·ONNX 메타데이터에 실제 클래스 순서를 기록한다. 모든 종류가 학습 데이터에
있어야 학습할 수 있으며, 빠진 종류가 있으면 종류 이름을 표시하고 중단한다.
6종 mIoU와 이전 3종 mIoU는 평가 종류가 달라 직접 비교할 수 없다.

## 입력과 정답

코스별 압축 NPZ 파일은 `np.load(path, allow_pickle=False)`로 읽는다.

| 필드 | 모양 / 타입 | 의미 |
|---|---|---|
| `rgb` | N×200×200×3 / uint8 | 공식 1-2 탑뷰. 정규화 전 원본 |
| `semantic` | N×200×200 / uint8 | 주 인식 정답. 0=물, 1=고정 장애물·육지, 2=다른 선박, 255=자기 선박 무시 |
| `physical_semantic` | N×200×200 / uint8 | 최소 표시 크기를 제거한 픽셀 중심의 실제 형상 분류 |
| `raster_class` | N×200×200 / uint8 | 병합 전 공식 클래스 ID. 이름은 manifest의 `raster_classes` |
| `pose` | N×3 / float64 | 세계 좌표 x,y,방향(rad). 정답 재생성을 위해 정밀도 유지 |
| `t` | N / float64 | 교통 선박 상태를 정하는 시뮬레이션 시각(초) |
| `clearance_m` | N / float32 | 선체 표면과 장애물·경계의 거리. 최대 10m로 제한 |

좌표계는 선수 위쪽, 좌현 왼쪽, 0.5m/픽셀이다. 계류 선박과 이동 선박은 이미지에서
같은 종류이므로 둘 다 2로 분류한다. 움직이는지 여부는 이후 시계열 추적으로 학습한다.
자기 선박 픽셀은 loss에서 제외해야 한다(`ignore_index=255`).

`semantic`은 공식 클래스 맵의 픽셀 중심 정답이다. 그림자·갑판·윤곽선은 별도
장애물 정답으로 만들지 않는다. RGB 가장자리는 안티앨리어싱되어 있지만 정답은 단일 클래스다.
작은 부표는 화면 표시를 위해 실제보다 크게 그려진다. 따라서 `physical_semantic`도 따로
저장한다. 이 역시 픽셀 중심 분류이므로 작은 실제 형상은 픽셀 사이에서 사라질 수 있다.
정확한 충돌 판단에는 이 마스크가 아니라 `courses/`의 연속 좌표 원·사각형을 사용한다.

미리보기는 **왼쪽 원본 / 가운데 인식 정답 / 오른쪽 형상 정답** 순서다.
정답 색은 파랑=물, 갈색=고정 장애물·육지, 청록=다른 선박, 회색=자기 선박 무시다.

## 데이터 분할과 한계

기본 네 코스는 정렬된 순서로 1·2 학습, 3 검증, 4 테스트에 배정한다.
같은 코스의 모든 샘플은 같은 분할에 속하며 내용이 같은 코스 복사본은 거부한다.
`manifest.json`에 원본 경로, 코스 해시, 클래스 비율 계산용 픽셀 수, 샘플 수와 실행 설정을 기록한다.

시드가 바꾸는 것은 수집기의 위치·방향·시각 샘플이다. 1-2 시뮬레이션의 교통 스케줄은
코스에 고정되어 있다. 같은 네 코스에서 이미지 수만 늘려도 코스 구조의 다양성이 늘지는 않는다.
실제 학습 전에는 서로 다른 구조·배치·교통의 코스를 더 준비하고 분할을 고정해야 한다.
이 작은 분할의 결과로 숨겨진 코스에 대한 일반화를 주장할 수 없다.

이 데이터의 각 프레임은 **독립적이며 자기 선박은 정지 상태**다. 인접 배열 요소를 연결해
추적 시퀀스로 사용하면 안 된다. 행동 정답, 움직임 정답, PPO 경험은 아직 포함하지 않는다.

## 인식 모델 학습

학습 코드는 `training/train_perception.py`, 모델은 `training/perception_model.py`다.
현재 모델은 U-Net 구조를 작게 만든 것으로, 입력은 탑뷰 RGB 한 장이고 출력은
각 픽셀의 3종 또는 6종에 대한 점수(logit)다. 확률이 필요하면 softmax를 적용한다.

```text
RGB 200×200×3 → [0,1] 정규화 → 3×200×200
    → 인코더: 8×200×200 → 16×100×100 → 32×50×50
    → 디코더: 16×100×100 → 8×200×200
    → 클래스 수×200×200 분류 점수
```

디코더의 각 단계에 같은 해상도의 인코더 출력을 연결한다(skip connection).
픽셀 위치와 작은 물체의 정보를 전달하기 위한 연결이다. `encode()`를 분리해 두었으므로
이후 시간축 인식 모듈을 붙일 때 공간 특징을 재사용할 수 있다.

훈련 입력에는 RGB만 들어간다. 경유지·자기 상태는 이후 주행 정책에 직접 전달할
정보이며, 현재 인식 모델에 섞지 않는다. 시뮬레이터 정답은 손실 계산에만 사용한다.

손실은 **가중 cross-entropy + 0.5×전경 Dice**다. 클래스 가중치는 학습 분할의
픽셀 빈도 역제곱근으로 계산한다. 부표 정답 픽셀에는 기본 5배 가중치를 더 주고,
자기 선박 픽셀은 두 손실 모두에서 제외한다. 평가에서도 자기 선박은 제외한다.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s training/tests -v
.\.venv\Scripts\python.exe -B -u -m training.train_perception
```

기본 실험은 30에포크, 배치 4, 초기 채널 8, 학습률 0.003, CPU 스레드 2다.
매 에포크 검증 mIoU를 계산하고 가장 높은 체크포인트를 저장한다. 테스트 코스는
체크포인트 선택이 끝난 후 한 번 평가한다. 변환·증강은 이 초기 실험에 적용하지 않는다.

기본 출력은 `training/runs/pilot-unet8-01/`이다. 출력 폴더가 있으면 덮어쓰지 않는다.
새 실험은 새 폴더를 지정한다.

```powershell
.\.venv\Scripts\python.exe -B -u -m training.train_perception --out training/runs/experiment-02
```

| 파일 | 내용 |
|---|---|
| `best.pt` | 검증 mIoU가 가장 높은 모델 가중치, 구조 설정, 클래스 정의 |
| `config.json` | 학습 설정, 학습 데이터 통계, 환경 버전, 데이터·코드 해시 |
| `history.json` | 에포크별 손실과 검증 지표 |
| `metrics.json` | 최종 검증·테스트 지표와 현재 PC의 모델 단독 추론 시간 |
| `validation.png` | 검증 데이터의 첫 세 장. 각 행은 원본 / 정답 / 예측 |
| `report.md` | 지표의 의미, 결과와 한계 |

mIoU는 각 종류의 영역 겹침 비율(IoU)을 평균한 지표다. 정답·예측 모두 없는 종류는 제외한다. 물 픽셀이 많으므로 전체
정확도만 보지 않고 종류별 IoU도 기록한다. 부표 픽셀 재현율은 정답 부표 픽셀 중
3종 모델에서는 고정 장애물, 6종 모델에서는 부표로 예측한 비율이다. 부표 개체를 몇 개 찾았는지와는 다른 지표다.

`best.pt`는 학습 가중치다. 제출 환경에는 PyTorch가 제공되지 않으므로, 제출용 추론에는
ONNX 내보내기와 ONNX Runtime 검증이 추가로 필요하다. 모델 단독 시간은 공식 에이전트의
관측 처리·추적·행동 결정·통신을 모두 포함한 시간 제한의 검증을 대신하지 않는다.

## ONNX 내보내기와 추론

```powershell
.\.venv\Scripts\python.exe -B -u -m training.export_perception
```

선택된 체크포인트에서 `perception.onnx`를 만든다. ONNX Runtime CPU 1.23.2로 검증 코스의
모든 이미지에서 PyTorch 출력과 수치 오차를 비교하며 배치 1과 2도 검증한다.
검증 결과는 `perception.verification.json`에 저장한다. 이미 출력 파일이 있으면 덮어쓰지 않는다.
새 내보내기는 `--out`으로 새 경로를 지정한다. `--verify-only`는 기존 ONNX를 변경하지 않고
검증을 다시 실행하며 JSON·Markdown 검증 보고서를 갱신한다. 6종 실험에서 서로 다른 CPU 연산의
반올림 때문에 약 128만 픽셀 중 3픽셀의 분류가 달라져, 확률과 지표를 함께 검사하도록 보완했다.
점수 오차 0.005, 확률 오차 0.002 이하, 픽셀 일치율 99.999% 이상, 종류별 IoU 차이 0.0001 이하를
모두 요구한다. 분류가 바뀐 픽셀의 PyTorch 1·2위 확률 차이도 0.002 이하여야 하며, 확실한 분류가
바뀌면 실패한다. 이전 3종 모델은 픽셀 분류가 완전히 일치했다. 공식 런타임의 ORT 버전을 맞췄지만, 로컬 Python과
NumPy 버전·CPU 장치는 공식 Python 3.13/NumPy 2.5.3/GPU 환경의 최종 검증을 대신하지 않는다.

`training/perception_runtime.py`와 `training/perception_schema.py`는 NumPy와 ONNX Runtime만 사용하는 추론 모듈이다.
RGB를 정규화하고 모델 출력에서 클래스 확률과 장애물 점유 확률을 만든다.
알려진 자기 선체 크기로 자기 선박 영역을 제외하며 숨겨진 물체 정보는 읽지 않는다.

```python
from training.perception_runtime import OnnxPerception

perception = OnnxPerception("training/runs/pilot-unet8-01/perception.onnx")
maps = perception.predict(rgb)  # 공식 200×200×3 uint8 탑뷰
```

| 출력 | 의미 |
|---|---|
| `labels` | 200×200 분류. ID 순서는 `perception.classes`; 255는 자기 선박 제외 |
| `class_probabilities` | 200×200×클래스 수의 확률. 자기 선박 부분은 `valid`로 제외해야 함 |
| `occupancy_probability` | 물을 제외한 모든 종류의 확률 합. 자기 선박 영역은 0 |
| `static_occupancy_probability` | 고정 객체와 강둑의 확률 합. 계류 선박은 여기서 제외되고 선박 영역에 들어감 |
| `vessel_probability` | 계류·이동 선박을 합친 선박 확률. 움직임 자체의 확률은 아님 |
| `valid` | 200×200 bool. 알려진 자기 선박 영역에서 False |

이 모듈은 인식 지도만 출력한다. 물체 속도는 추적 단계에서, 속도·회전 명령은 주행 정책에서
결정한다. 제출할 때는 두 추론 모듈과 ONNX 파일을 제출 디렉터리에 복사하고,
제출 모듈 배치에 맞게 `perception_schema` import를 바꾼 뒤 전체 에이전트로 검증한다.
두 클래스 정의를 모두 지원하므로 기존 3종 ONNX도 계속 읽을 수 있다.
수집기의 `previews/`는 저장 형식인 3종 정답을 표시한다. 6종 모델의 `validation.png`는
6종 정답과 예측을 위 표의 색으로 표시한다.

학습 시드를 기록하지만 다른 PyTorch 버전·장치에서 수치까지 같은 결과를 보장하지 않는다.
설계 참고는 [U-Net 원 논문](https://arxiv.org/abs/1505.04597)과
[PyTorch 재현성 안내](https://docs.pytorch.org/docs/2.14/notes/randomness.html)다.

## 연속 탑뷰와 선박 추적

6종 인식 뒤에 `tracking.py`를 연결했다. 추적은 지금 별도 신경망을 학습하지 않고
연결 영역·세계 좌표 변환·선박 연결·최근 관측의 속도 회귀로 구현했다.

```text
RGB → 6종 영역 지도 → 선박 연결 영역 → 공개 자기 pose로 세계 좌표 변환
    → 이전 선박과 1:1 연결 → 최근 2초 위치로 지상 속도 추정
    → 선박별 위치 / 크기 / 속도 / 이동·정지·모름
```

1프레임의 선박 영역에서 계류/이동 선박을 구분하지 않는다. 공개 자기 좌표·방향으로
우리 배의 이동·회전을 보정한 후 다른 선박의 세계 좌표가 변하는지 본다.
처음 발견한 선박, 가장자리에서 잘린 선박, 관측이 끊긴 선박은 움직임을 **모름**으로 둔다.
최소 1.2초의 정상 관측 뒤 최대 2초 창으로 속도를 회귀한다. 속도 0.25 m/s와
회귀 잡음 여유로 정지/이동을 판단하며, 확률적으로 보정된 신뢰도는 아니다.

```python
from training.perception_runtime import OnnxPerception
from training.tracking import VesselTracker

perception = OnnxPerception("training/runs/detail-unet8-01/perception.onnx")
tracker = VesselTracker(perception.classes)
tracker.reset()  # 매 새 에피소드에서 초기화
maps = perception.predict(rgb)
tracks = tracker.update(maps["labels"], pose, tick,
                        ego_velocity=vel, probabilities=maps["vessel_probability"])
```

| 주요 선박 출력 | 의미 |
|---|---|
| `track_id` | 이 에피소드 안에서 추적기가 만든 ID. 실제 시뮬레이터 ID와 무관 |
| `position_world_m`, `position_body_m` | 세계 좌표 / 우리 배 중심 좌표의 선박 중심 |
| `length_m`, `width_m` | 인식 영역의 장축·단축 크기 추정. 충돌 정답 치수와 다를 수 있음 |
| `axis_heading_world_rad` | 장축 방향. 선수·선미 구분은 180도 모호함 |
| `velocity_world_mps`, `velocity_body_mps` | 다른 선박의 지상 속도 벡터를 세계 축 / 우리 배 축으로 표현 |
| `relative_velocity_body_mps` | 다른 선박 지상 속도에서 우리 배 병진 속도를 뺀 벡터 |
| `position_rate_body_mps` | 우리 배의 회전 효과까지 포함한 선박 상대 좌표의 변화율 |
| `motion_state`, `velocity_valid` | `stationary` / `moving` / `unknown`, 속도 사용 가능 여부 |
| `observed`, `missing_s`, `truncated` | 현재 관측 여부 / 마지막 관측 이후 시간 / 영상 가장자리 잘림 |

물체가 잠시 안 보여도 0.5초 동안 위치 예측을 유지하지만 움직임은 모름이고 공개 속도는
`None`이다. 더 오래 사라지면 트랙을 지운다. 장기 재식별을 구현한 것은 아니다.
모름 선박도 장애물이다. 이후 정책에서 물로 취급하면 안 된다. 계류 선박은 선박 클래스에
속하므로 고정 지도만으로 회피하지 않고 전체 점유 지도와 추적 선박 정보를 함께 사용한다.

### 연속 데이터 수집과 평가

```powershell
.\.venv\Scripts\python.exe -B -u -m training.collect_tracking
.\.venv\Scripts\python.exe -B -u -m training.evaluate_tracking --out training/runs/tracking-baseline-01 --splits val test
.\.venv\Scripts\python.exe -B -m unittest discover -s training/tests -v
```

기존 출력 폴더가 있으면 덮어쓰지 않는다. 수집·평가를 다시 실행하려면 새 `--out`을 지정한다.
인식 데이터와 같은 코스 분할을 사용하며 코스 중복·정답 누출을 검사한다.
수집의 기본 설정은 7개 코스 × 2개 클립 × 41프레임 = 574프레임이다.
각 클립은 4초이며 정지 관찰과 `[0.6 m/s, ±0.12 rad/s]` 기동을 번갈아 생성한다.
처음 관측 위치·시각은 자동으로 고르고 그 뒤는 공식 `Vessel.step`으로만 진행한다.
모든 틱에서 실제 선체의 충돌·강둑 이탈·여유 거리를 확인해 안전한 전체 클립만 저장한다.
키보드 운전은 필요 없다. 짧은 관찰 기동이므로 전문가 운전 데이터나 완주 데이터는 아니다.

`training/data/tracking-pilot/`의 NPZ에는 모델 입력인 RGB·공개 pose·vel·tick과
오프라인 평가용 정답을 별도 필드로 저장했다. `actions[f]`는 f프레임에서 f+1프레임으로
진행하는 명령이며 마지막 프레임에는 다음 액션이 없다. `pose_exact`는 정확한 재현용이다.
`gt_ids`, `gt_instance`, `gt_kind`, `gt_state`의 정답 ID·픽셀·종류·중심/방향/치수/세계 속도는
추론 모듈에 전달하지 않는다. 열린 항로에서 새로 출현하는 선박도 항로 내 출현 순번으로 ID를 유지한다.

현재 결과는 `training/runs/tracking-baseline-01/report.md`에 있다. 검증/테스트 각 4클립,
164프레임에서 개체 재현율은 98.8%/90.2%, 속도를 추정할 수 있었던 성숙한 선박의 속도
벡터 평균 오차는 0.047/0.056 m/s였다. 테스트에서는 움직임 판단 가능 비율 85.7%,
누락·모름을 포함한 이동 선박 재현율은 75.0%였다. 판단한 객체의 정확도만으로 평가하면 안 된다.
CPU 인식+추적 중앙값 약 49 ms지만 주행 정책·통신·렌더링은 포함하지 않았다.

`metrics.json`은 분할·기동별 지표와 정답 영역을 넣은 별도 추적 진단,
`tracks.json`은 프레임별 추론 출력과 별도로 표기한 오프라인 정답 연결을 담는다.
서로 붙은 선박을 분리하는 인스턴스 모델은 아직 없고, 작은 선박 누락·곡선 항로에서의
속도 지연·긴 에피소드·1-4 외란은 추가 검증이 필요하다.
처음 검증만 한 `tracking-pilot-01`은 개발 기록이고, 현재 출력 필드를 포함한 결과는
`tracking-baseline-01`을 사용한다. 테스트를 본 뒤 추적 파라미터나 인식 가중치는 바꾸지 않았다.

테스트 누락을 사후 확인한 결과, 4.93×1.26 m 이동 선박 하나가 보이는 41프레임 중
32프레임에서 발견되지 않았다. 예시 2초 프레임에서는 선박 정답 24픽셀을 모두 물로
분류했다. 상세 수치와 원본/정답/예측 그림은 같은 결과 폴더의 `miss-diagnostic.json`,
`miss-example-1.png`에 있다. 작은 선박에 대한 인식 데이터 보강이 필요한 근거다.

## 강화학습 주행 환경

`navigation_env.py`는 공식 시뮬레이터를 Gymnasium 1.2.3의 `reset/step` 환경으로
연결한다. 물리·센서·전체 선체 충돌·제어 비용을 재사용하며, 틱 판정 순서는 공식 실행과
같이 충돌 → 강둑 이탈 → 경유지 하나 도달 → 완주 → 공식 시간 초과다.
`NavigationEpisode`의 루프를 공식 `run_episode`와 명령별로 비교하는 검사를 추가했다.
1-2가 현재 학습 대상이며 1-4의 동일 시드 외란도 단위 검사에서 비교했다.
정책의 실행 속도 제한·프로세스 통신은 이 학습 환경의 검사에 포함되지 않는다.

```python
from training.navigation_env import NavigationEnv
from training.perception_runtime import OnnxPerception
from training.policy_env import PolicyEnv

env = PolicyEnv(
    NavigationEnv(["sets/practice/courses/practice-01.json"],
                  training_tick_limit=256),
    OnnxPerception("training/runs/detail-unet8-01/perception.onnx"),
)
features, info = env.reset(seed=47)
features, reward, terminated, truncated, info = env.step([0.0, 0.0])
```

`PolicyEnv`의 행동은 [-1,1] 범위의 전진·회전 두 값이다. 실제 명령은
`v=0.75+1.25*a[0]`, `w=0.6*a[1]`로 변환한다. **[0,0]은 정지가 아니라
0.75m/s 전진 명령**이고, 정지는 [-0.6,0]이다. 명령 속도는 실제 순간 속도가 아니며
공식 관성·slew·추력 분배와 횡미끄러짐을 거친다. `NavigationEnv`를 직접 사용할 때는
행동에 실제 단위인 m/s와 rad/s를 넣는다.

| 정책 입력 | 크기 | 의미 |
|---|---|---|
| `map` | 7×50×50 | 6종 인식 특징 + 유효 픽셀 비율. 2m 칸, 전경 4×4 최대·물 평균 |
| `state` | 19 | 공개 자기 상태·이전 명령·다음 경유지 상대 좌표·남은 시간·선체 치수 |
| `ships` | 32×16 | 가까운 추적 선박. 존재·속도 유효·모름·관측 여부를 별도 마스크로 전달 |
| `waypoints` | 32×4 | 남은 경유지의 몸체 좌표와 도달 반경. 인식 CNN을 거치지 않음 |

`policy_observation.py`는 공개 관측·메타 정보만 받으며 시뮬레이터/코스를 import하지 않는다.
경유지·자기 상태는 인식 신경망과 별도로 정책에 전달한다. 인식된 작은 부표가 2m 칸으로
축소할 때 사라지지 않도록 전경에는 최대 풀링을 사용한다. 채널 합이 1인 확률 분포는 아니다.
지상 속도가 모름인 선박은 0 속도 값과 **속도 유효=False, 모름=True**를 함께 전달한다.
인식·추적 상태는 새 에피소드마다 초기화하며 패딩은 존재 마스크로 구분한다.
표의 크기는 초기 정책용 설정으로, 경유지·선박 상한은 각각 32다.

초안 학습 보상은 남은 경유지 경로 거리 감소 1점/m, 경유지 +5, 완주 +50,
충돌·강둑 이탈·잘못된 명령 -100, 공식 시간 초과 -10이다. 시간 비용 0.1점/s,
공식 명령 변화 비용의 0.01배, 선체 여유 2m 이내 근접 비용 최대 0.5점/s를 뺀다.
거리 변화에는 후진·우회로 인한 음수도 포함한다. 각 보상 항을 info에 별도로 기록한다.
이는 학습용 초안이고 다른 팀과 비교해 정해지는 공식 경쟁 점수를 대체하지 않는다.
실제 객체 정보·여유 거리는 정책 특징에 넣지 않으며 오프라인 보상·진단에만 사용한다.

공식 6000틱은 과제 자체의 유한한 종료(`terminated`)이며 남은 시간을 정책 상태에 넣었다.
`training_tick_limit`으로 더 일찍 끊는 학습 수집은 `truncated`다. 그 중단을 충돌·공식 시간
초과로 표시하지 않고 실패 보너스도 주지 않는다. PPO를 연결할 때 최종 관측을 이용해
수집 중단 뒤 가치 추정을 이어가야 한다. [Gymnasium 시간 제한 안내](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

```powershell
.\.venv\Scripts\python.exe -B -m unittest training.tests.test_navigation -v
.\.venv\Scripts\python.exe -B -u -m training.check_navigation_env
```

두 번째 명령은 실제 렌더링·ONNX·추적을 모두 사용해 쉬운 코스를 경유지 추종 명령으로
실행하고, 학습 코스 practice-01에서 30틱 연결 검사를 한다. 모든 명령을 공식 실행에서
다시 적용해 관측·물리 결과를 비교한다. 기존 출력이 있으면 새 `--out`을 지정한다.
기본 결과는 `training/runs/navigation-env-01/report.md`다. 쉬운 코스는 40m씩 두 구간과
주행선 밖의 물체들로 구성하며 공식 코스 검사기를 통과한 파일을 사용한다.
완주는 환경 연결 검사이고, PPO 가중치 학습이나 장애물 회피 성능의 증거는 아니다.

첫 연결 검사에서는 쉬운 코스를 521틱(52.1초)에 완주했고, 실제 학습 코스 30틱도
공식 재실행의 공통 관측·RGB 샘플·물리 상태·명령·여유 거리·제어 비용과 일치했다.
환경 관련 10개 검사와 기존 인식·추적 검사를 합쳐 28개가 통과했다.
최종 코드로 저장된 명령과 공통 관측을 다시 검사할 때는 다음 명령을 사용한다.

```powershell
.\.venv\Scripts\python.exe -B -u -m training.check_navigation_env --verify-replays
```

CPU에서 렌더링까지 포함한 환경 처리량은 쉬운 코스 약 7.6틱/s, 실제 코스의 짧은
검사는 약 2.4틱/s였다. 첫 PPO 실험은 쉬운 코스의 적은 학습량으로 연결·학습 여부를
확인한 뒤 늘린다. 이 시간은 공식 에이전트 행동 시간 제한의 검증이 아니다.

## PPO 주행 정책 첫 실험

`navigation_policy.py`는 탑뷰 인식 결과·추적 선박·공개 상태·경유지를 받아
전진 및 회전 명령을 정하는 정책이다. 지도 CNN 64개 특징, 선박 공통 MLP의 마스크
평균/최대 64개 특징, 순서를 유지하는 경유지 특징 32개를 결합한다. 상태 19개와
다음 경유지 4개 값은 직접 전달한다. 이 183개 입력 뒤에서 행동을 정하는 actor와
남은 누적 보상을 추정하는 critic이 각각 64→64 신경망을 사용한다.
인식 ONNX는 고정하고, 주행 특징 추출기와 actor/critic만 함께 학습한다.
빈 선박 패딩은 제외하고, 선박의 나열 순서가 결과에 영향을 주지 않도록 구성했다.

```powershell
.\.venv\Scripts\python.exe -B -m unittest training.tests.test_navigation_policy -v
.\.venv\Scripts\python.exe -B -u -m training.train_navigation --steps 2048 --cutoff 800 --out training/runs/navigation-ppo-02
```

기존 출력 폴더는 덮어쓰지 않는다. `--steps`는 실제 시뮬레이터 행동 횟수이며
PPO가 256개씩 수집하므로 256의 배수로 올림될 수 있다. 같은 256개의 경험을
작은 묶음으로 나누어 최대 10회 학습한 뒤 새 행동을 수집한다. 키보드나 기본
에이전트 시범 없이 현재 주행망의 확률적 행동으로 경험을 만든다.
`--cutoff 800`은 80초에서 끝내는 **실험용 수집 중단**이다. 공식 제한 600초와
구분하며, PPO는 수집 중단의 마지막 관측에서 critic 값을 이어서 사용한다.
충돌·완주·공식 시간 초과는 실제 종료다.

첫 단계 코스는 학습 4개와 검증 2개로 분리한다. 직진·작은 각도 변화와
경로 밖 정적 물체만 포함하고 움직이는 선박은 없다. 코스를 파일로 저장한 뒤
공식 로더로 읽어 사용하며 공식 검사기를 통과해야 한다. 기존 공식 검증·시험
코스는 이 실험에 사용하지 않는다. 같은 검증 코스와 시드에서 학습 전후의
결정적 행동을 비교한다. 장애물 회피나 이동 선박 회피 검증은 이후 단계다.

결과 폴더에는 초기/최종 PPO 체크포인트, 코스와 해시, 설정·버전·소스 해시,
PPO 갱신 지표, 학습 에피소드 결과, 검증 전후 명령·위치 궤적 및 보고서를 저장한다.
최종 체크포인트를 다시 읽어 행동이 일치하는지, 가중치가 실제로 갱신되었는지,
고정 인식 모델의 해시가 유지되는지 검사한다. 전체 탑뷰와 PPO 임시 버퍼를
영구 학습 데이터셋으로 저장하는 수집기는 아직 포함하지 않는다.

```python
from stable_baselines3 import PPO

model = PPO.load("training/runs/navigation-ppo-01/final-policy.zip", device="cpu")
# features는 위 PolicyEnv.reset()/step()이 돌려주는 공개 입력 네 가지다.
action, _ = model.predict(features, deterministic=True)
features, reward, terminated, truncated, info = env.step(action)
```

이 체크포인트는 PyTorch/SB3 학습용이다. 공식 제출 환경에서는 actor를 ONNX로
내보내고 인식·추적·행동 전체 실행 시간과 주행 성능을 검증하는 과정이 추가로 필요하다.
[SB3 PPO 안내](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html),
[공통 표현과 actor/critic 구성](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/custom_policy.html).

저장된 실험의 한국어 요약과 궤적 비교 그림은 다음 명령으로 만든다.

```powershell
.\.venv\Scripts\python.exe -B -m training.review_navigation_run training/runs/navigation-ppo-01
```

코스의 실제 형상을 그림에만 표시하며, 정책에 실제 객체 정보를 추가하지 않는다.

## RGB 표현과 경유지 주행의 공동 학습

현재 새 학습 경로는 `train_rgb_navigation.py`다. 입력은 원본 200×200 탑뷰
4장(0.5초 간격), 각 관측의 자기 상태 이력, 현재 상태, 남은 경유지와 최종 목적지다.
RGB CNN, 경유지 특징 추출기, actor/critic을 PPO로 함께 갱신한다. 고정 인식 ONNX나
추적기는 이 경로에서 사용하지 않는다. 이미지와 경험은 시뮬레이터가 즉석에서 만든다.
과거 탑뷰의 화면 변화에는 자기 움직임도 포함되므로, 각 이미지의 위치·방향·속도·시점도 전달한다.

좌표 추종 기본 명령에 신경망 보정을 더하는 구조이며, 새 정책의 보정 평균은 0이다.
기본 추종 명령은 0.1초마다 다시 계산하고 신경망 보정은 0.5초간 유지한다.
잔차는 전진 ±2m/s·회전 ±1.2rad/s 범위이며 합산한 명령을 공식 범위로 자른다.
기본 행동을 취소해 정지·후진·반대 방향 선회할 권한이 있다. 기본 추종기는 장애물 회피를 보장하지 않는다.
따라서 초기 완주는 신경망이 처음부터 조종을 배운 증거가 아니다.

물리 적분·충돌·강둑·경유지 판정은 원래 0.1초 단위를 유지한다. 매 결정에서 최대 5틱을
진행하되 종료 순간에 멈추며, RGB는 마지막 틱에서 한 번만 생성한다. 보상은 내부 틱 보상의
합이고, PPO 할인율 .995와 GAE .95는 0.5초 결정 단위이다. 이전 실험과 할인 시간 범위가 다르다.

```powershell
.\.venv\Scripts\python.exe -B -m unittest training.tests.test_rgb_navigation -v
.\.venv\Scripts\python.exe -B -u -m training.train_rgb_navigation --steps 1024 --stage 0 --device cpu --out training/runs/rgb-new-01
.\.venv\Scripts\python.exe -B -u -m training.train_rgb_navigation --steps 1024 --stage 1 --resume training/runs/rgb-new-01/final-policy.zip --out training/runs/rgb-new-02
```

구현 단계는 0=장애물 없는 직선, 1=방향 전환, 2=부표 하나다. 2단계 부표는 두 번째
경유지 구간의 주행선 위에 배치해 회피가 필요한 상황을 만든다. 높은 단계에서는 이전 단계
코스를 학습 풀에 함께 넣는다(1단계는 5개 중 1개, 2단계는 6개 중 2개가 이전 단계).
매 에피소드 현재 정책으로 다시 주행하며, 과거 행동 버퍼를 재사용하는 방식은 아니다.
자동 승급은 하지 않는다. 아직 이동 선박·외란 학습 커리큘럼은 포함하지 않았다.
이전 단계의 검증 성능도 확인하고 복잡도를 올려야 한다.

`--steps`는 PPO 결정 수이고 실제 물리 틱 수는 최대 5배다. 출력은 둘 다 기록한다.
`latest-policy.zip`는 매 갱신 후 저장하고, `--resume`은 정책·최적화 상태를 이어 받는다.
환경 위치·난수까지 같은 궤적을 복원하지는 않는다. 기존 출력 폴더는 덮어쓰지 않는다.
GPU에서는 `--device cuda`, 환경 병렬 실행에는 `--n-envs 2`를 사용한다.
Colab은 CPU lock파일을 설치하지 말고 [준비한 노트북과 안내](cloud_training.md)를 사용한다.

로컬 첫 완료 실험 `runs/rgb-ppo-02/report.md`에서 1,024결정·5,109물리틱을 학습했고,
끝난 학습 에피소드 8개가 모두 완주했다. 검증 코스는 전후 모두 2/2도달, 60.0초→63.3초여서
속도 개선은 없었다. RGB CNN의 8개 가중치 텐서 변경·유한한 기울기·저장 후 행동 일치를 확인했다.
그 모델에서 1단계 256결정을 이어 학습한 `runs/rgb-ppo-stage1-02/report.md`에서는
방향 전환 검증 코스가 전후 모두 2/2도달, 64.1초→63.3초였다. 이어 학습한 최종 모델로
기존 직선 검증 코스도 다시 완주했다(2/2, 62.6초). 각 단계 검증은 별도 코스 하나와
고정 시드 하나로 수행한 짧은 확인이므로 여러 코스의 성공률이나 일반화 성능을 의미하지 않는다.
실행 당시 네 학습 소스의 원본은 각 결과 폴더의 `source/`에 보관했다.
이것은 공동 학습 연결의 검증이다. 장애물 없는 학습만으로 장애물 인식·회피·선박 속도 예측을
배웠다고 해석하면 안 된다. 정책은 학습용 Torch체크포인트이고 제출용 ONNX 변환은 이후 작업이다.

## 충돌 경험을 이용한 부표 전용 학습

`train_buoy_navigation.py`는 작은 부표의 표현과 회피 탐색을 보완한 새 학습 경로다.
최신 탑뷰를 원본 해상도에서 처리해 학습된 100×100 부표 확률 지도를 만들고,
최대 풀링으로 보존한 공간 특징을 좌표·경유지 분기와 함께 정책에 전달한다.
기존 작은 RGB CNN과 구조가 달라 이전 RGB 체크포인트를 그대로 읽지 않는다.
새 체크포인트끼리는 정책·최적화 상태를 이어 학습할 수 있다.

부표 정답 마스크는 공식 렌더러에서 생성해 보조 인식 손실에만 사용한다.
실제 정책 입력에는 탑뷰와 공개 좌표·자기 상태만 들어간다. 기존 부표 전용 이미지
인코더는 최근 4장 중 최신 한 장을 사용하고, 공개 자기 상태 이력은 유지한다.
이동 선박 단계에서 시간에 따른 영상 특징 결합은 추가로 설계해야 한다.

같은 공개 경로에서 부표의 위치를 독립적으로 바꿔 경유지 좌표로 배치를 암기하기
어렵게 만든다. 기본 학습 풀은 부표 배치 32개와 빈 코스 8개이고, 검증은 별도
배치다. 첫 부표는 출발 후 12~25m에 배치한다. 부표 전용 `--stage`의 의미는
1=부표 한 개, 2=두 개, 3=세 개 또는 네 개이며, 앞의 RGB 주행 단계와 구분한다.
모든 부표는 공식 지름 0.6m이고 코스는 저장 후 공식 검사기를 통과한다.

gSDE 탐색 잡음을 최대 8초 간격으로 바꿔 방향이 일정 시간 유지되는 회피 경험을
만든다. 부표를 찾는 보조 갱신은 새 PPO 주행 경험을 모으기 전에 수행한다.
PPO는 실제 충돌·완주 결과로 정책과 공유 CNN을 함께 갱신한다.
보상에는 6m 이내 근접 비용과 공개 경로에서의 이탈 비용도 포함하며,
충돌 없는 정지만으로 승급하지 않고 목적지 완주를 확인한다.

```powershell
.\.venv\Scripts\python.exe -B -u -m training.train_buoy_navigation --policy latest --stage 1 --steps 4096 --out training/runs/buoy-new-01
.\.venv\Scripts\python.exe -B -u -m training.train_buoy_navigation --policy latest --stage 2 --steps 4096 --warmup-epochs 12 --resume training/runs/buoy-new-01/final-policy.zip --out training/runs/buoy-new-02
```

완료한 `runs/buoy-ppo-02`에서는 4,096회 결정 학습 후 검증이 0/4완주에서
4/4완주로 바뀌었다. 학습 중 부표 코스는 완주 16회·충돌 17회, 빈 코스는
8회 완주했다. 검증 완주 시간은 68.5~69.1초였다. 이는 한 개 부표의 별도 배치
네 개에서 확인한 결과이며 복잡한 환경의 일반적인 회피 성능을 뜻하지 않는다.
부표 영상에 대한 의존성과 여러 부표를 연속 회피하는 능력은 추가 검증한다.

## 시계열 다중 부표 모델: 환경 승인 대기

새 기본 정책 `temporal`은 최근 탑뷰 4장을 같은 CNN으로 읽는다. 간격은 0.5초,
관측 범위는 1.5초다. 공개 자기 위치·방향 이력으로 과거 부표 지도를 현재 선박
좌표계에 정렬하고, GRU 64차원으로 시간 순서대로 결합한다. GRU 상태는 관측
윈도우마다 초기화하므로 일반 PPO 미니배치를 사용할 수 있다. 현재 장면·부표
공간 특징과 목표 좌표·자기 상태를 직접 결합한 특징은 621차원이다.
총 정책 파라미터는 186,804개다. 경유지 추종 기본 명령에 신경망 보정을 더한다.

[환경 및 실제 탑뷰 검토 자료](runs/temporal-buoy-review-01/report.md)를 먼저 확인한다.
부표 2→4→6개, 수로 폭 40→30→24m, 경로 약 90m의 초안이며,
학습 60·검증 30·최종 시험 60·사용자 검토 6개 코스를 분리했다.
학습은 아직 시작하지 않았다. 승인되지 않은 환경 manifest는 학습기가 거부한다.
`--course-manifest`는 승인된 코스 풀에만 사용하고, `--stage`는 이 초안에서
1=2개, 2=4개, 3=6개다. 기존 자동 생성 코스의 단계 의미와 구분한다.
기존 정책의 재개에는 `--policy latest`가 필요하다. 선택적으로
`--warm-start-vision`으로 기존 부표 CNN·인식 head만 옮길 수 있고, 새 Actor·GRU는
별도로 학습한다. 제출용 이동 선박 예측·외란 대응은 이후 검증 과제다.

## 이후 단계

1. Colab에서 현재 체크포인트의 직선·방향 전환 학습량을 늘리고 이전 단계 성능도 확인한다.
2. 부표 하나 회피를 학습하고, 배치·크기·모양이 다른 정적 장애물로 확장한다.
3. 다수 장애물·좁은 통로·여러 경유지를 포함하되 쉬운 코스를 일정 비율 유지한다.
4. 이동 선박의 횡단·마주침·추월을 추가해 탑뷰 이력의 예측·회피 성능을 확인한다.
5. 외란이 있는 1-4로 확장하고, 공식 검증 코스와 제출용 실행 환경에서 평가한다.

부표 회피를 시작할 때는 작은 물체가 입력 표현에 남는지도 확인해야 한다. 필요한 경우
시뮬레이터 정답으로 보조 인식 손실을 더할 수 있지만, 현재 PPO에는 별도 인식 손실이 없다.

데이터 수집기 자체는 NumPy와 기본 라이브러리만 사용한다. PyTorch는 학습과 환경
검증에 사용한다. Gymnasium은 환경 연결, Stable-Baselines3는 PPO 학습에 사용한다.
수집 속도나 임시 신경망 연산 시간은 최종 모델 및 강화학습의 처리량을 뜻하지 않는다.
