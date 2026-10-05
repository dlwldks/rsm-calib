# rsm_calib — 학습 없는 채널 캘리브레이션 재현 코드

Moon & Cho (2026), *Remote Sensing* 18:205 재현용.
모델 가중치는 고정하고, 입력 정규화 값 6개(RGB mean/std)를 반응표면법(RSM)으로 튜닝한다.

| 문서 | 내용 |
|---|---|
| [`docs/paper_vs_code.md`](docs/paper_vs_code.md) | 논문과 코드의 항목별 대조, 기준 IoU 재현 기록 |
| [`docs/experiments.md`](docs/experiments.md) | 지금까지 한 실험 목록 (폴더·조건·결과·판단) |
| `reports/` | 미팅별 보고서 (날짜별 폴더) |

## 파일

| 파일 | 역할 |
|---|---|
| `rsm.py` | 2차 회귀 피팅(식 1·2), R²/Adj.R²/F/p, 정상점(식 3), 비례 전이(식 4) |
| `search.py` | 탐색. `rsm_search_paper`(논문 Figure 1) / `rsm_search_ours`(우리 변형) / `broad_sweep`(1단계 폭 비교) |
| `data.py` | TIF 로딩, 라벨 매핑, smp U-Net 로딩, 추론/IoU, decision fusion |
| `run.py` | CLI: `synthetic` / `inspect` / `calibrate` / `broad` / `apply` / `transfer` |
| `labelmap_19.json` | FLAIR 마스크 값(1~19) → 모델 출력 인덱스(19채널) |
| `analysis/order_compare.py` | trials csv로 1~4차 반응면 모델 비교 (추론 불필요) |
| `analysis/order_by_n.py` | 관측 수(35~125)별 1차·2차 LOO 비교 (추론 불필요) |
| `analysis/solver_compare.py` | 계수 추정 방법 5가지 비교 (추론 불필요) |
| `analysis/reinfer_candidates.py` | 후보 θ(정상점·상자 내 최대점)를 실제로 추론해 IoU 확인 |
| `analysis/plot_results.py` | `analysis/results/`의 그래프(`fig_*.png`) 생성 |
| `analysis/results/` | 분석 결과 csv·txt·그래프 |
| `out_*/` | 탐색 결과. 폴더별 설명은 `docs/experiments.md` |

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
GPU 컴퓨터에서는 `requirements.txt`의 `torch`, `torchvision` 두 줄을 빼고, CUDA 버전에 맞는 torch를 따로 설치한 뒤 `--device cuda`.

## 데이터·모델 준비

용량 때문에 레포에 포함하지 않는다 (`.gitignore`: `flair_1_toy_dataset/`, `models/`, `*.tif`, `*.pth`).
레포 폴더 안에 아래 구조로 둔다. 아래 명령 예시는 모두 이 위치 기준.

```
rsm_calib/
├── flair_1_toy_dataset/
│   └── flair_1_toy_dataset/
│       ├── flair_1_toy_aerial_train/   <도메인>/<구역>/img/IMG_*.tif
│       ├── flair_1_toy_aerial_test/    <도메인>/<구역>/img/IMG_*.tif   (50타일)
│       ├── flair_1_toy_labels_train/   <도메인>/<구역>/msk/MSK_*.tif
│       └── flair_1_toy_labels_test/    <도메인>/<구역>/msk/MSK_*.tif
└── models/
    ├── rgb15/FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth
    └── mitb5/FLAIR-INC_rgb_15cl_mitb5-unet_weights.pth   (모델 간 적용 실험에만 사용)
```

| 항목 | 출처 |
|---|---|
| 데이터 | FLAIR #1 toy dataset (IGNF, https://github.com/IGNF/FLAIR-1 안내에 따라 다운로드 후 압축 해제) |
| ResNet34 U-Net | Hugging Face `IGNF/FLAIR-INC_rgb_15cl_resnet34-unet` |
| MiT-B5 U-Net | Hugging Face `IGNF/FLAIR-INC_rgb_15cl_mitb5-unet` |

`--img-dir`와 `--msk-dir`는 둘 다 `flair_1_toy_dataset/flair_1_toy_dataset`로 준다. 그 아래에서 `IMG_*.tif`, `MSK_*.tif`를 찾고, `--include test`를 주면 경로에 `test`가 들어간 타일(test 50타일)만 쓴다.

## 탐색 모드

| | `--paper-mode` (논문 Figure 1) | 기본 (우리 변형) |
|---|---|---|
| 1단계 | θ0 포함 30개, θ0 × (1 ± 0.5) | θ0 포함 30개, θ0 × (1 ± 0.10) |
| 이후 회차 | 정상점 θ̂ 포함 5개, θ̂ × (1 ± 0.15) | 상자 안 곡면 최댓값 1개 + 주변 6개, ±0.03 |
| 탐색 경계 | 없음 (정상점 따라 이동) | θ0 ± 10% 상자 고정 |
| 정상점이 saddle/min | 그대로 사용 (`--move boxmax`로 변경 가능) | 상자 안 곡면 최댓값 사용 |
| ε1 | 이번 회차 best − 지난 회차 best | 누적 best 개선폭 |
| ε2 | \|이번 회차 best − 예측 IoU\| | \|정상점 실제 IoU − 예측 IoU\| |
| 종료 | ε1 ≤ ε_stop **OR** ε2 ≤ ε_stop (ε_stop=0) | ε1 ≤ 0.1 **AND** ε2 ≤ 1.0 |
| calib/test 분리 | 없음 | 70/30 |

`--max-evals`(논문 모드 80, 우리 모드 60)에서 종료. 모든 폭은 `--adj0 / --adj-t / --n0 / --n-t / --eps-stop`으로 바꿀 수 있다.

### 추가 옵션 (논문에 없는 변형)

| 옵션 | 기본 | 내용 |
|---|---|---|
| `--fit ols\|ridge` | `ols` | 회차마다 2차식 계수 추정 방법. `ridge`는 λ를 LOO로 선택 (`--ridge-lambda`로 고정 가능) |
| `--move stationary\|boxmax` | `stationary` | 논문 모드 다음 회차 중심. `boxmax`는 정상점이 극대가 아니면 1단계 범위 안 곡면 최대점으로 이동 |
| `--reuse <trials csv>` | 없음 | 같은 조건의 이전 기록에 같은 θ가 있으면 추론 없이 기록값 사용 (끊긴 탐색 재개용) |
| `--base-mode pooled\|per-tile\|given` | `pooled` | 기준 θ0 계산 방식. `pooled` = 보정 타일 전체 채널 통계 한 세트 (논문과 일치, `docs/paper_vs_code.md` 8절) |

## 순서

아래 예시는 Windows cmd 기준 (줄 이어쓰기 `^`). 공통 인자는 매번 같다.

```bat
set DATA=flair_1_toy_dataset\flair_1_toy_dataset
set CKPT=models\rgb15\FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth
```

**0. 로직 검증 (모델·데이터 불필요)**
```bat
python run.py synthetic --paper-mode
```

**1. 데이터/모델 점검**
```bat
python run.py inspect --img-dir %DATA% --msk-dir %DATA% --checkpoint %CKPT% --n-classes 19
```

**2. 1단계 탐색 폭 비교 (adj0 결정용)**
```bat
python run.py broad --img-dir %DATA% --msk-dir %DATA% --checkpoint %CKPT% --n-classes 19 ^
  --label-map labelmap_19.json --targets 5 --adjs 0.1 0.25 0.5 --out out_broad
```
adj0별 IoU 분포(최저·중앙·최고, 5 미만 개수)와 1차/2차 피팅 LOOCV를 `broad_summary.csv`로 저장.

**3. 논문 E2 조건 캘리브레이션 (toy test 50타일 전부로 보정, 기준 = 타일 전체 채널 통계 한 세트)**

이 조건에서 기준 IoU가 논문 Table 6 (b)와 14개 클래스 모두 ±1.2 이내로 재현됨 (`docs/paper_vs_code.md` 8절).
```bat
python run.py calibrate --img-dir %DATA% --msk-dir %DATA% --checkpoint %CKPT% --n-classes 19 ^
  --label-map labelmap_19.json --targets 5 --paper-mode ^
  --include test --calib-tiles all --out out_e2_test50
```
출력: `result.json`(θ, 비율, RSM 통계, 회차별 ε1·ε2, 종료 사유), `trials_c*.csv`(모든 추론 기록 + 회차 번호), `iou_table.csv`.

폭·추정 방법·이동 규칙을 바꾼 실행 예 (`out_e2_test50_adj0.25_ridge_boxmax`):
```bat
python run.py calibrate --img-dir %DATA% --msk-dir %DATA% --checkpoint %CKPT% --n-classes 19 ^
  --label-map labelmap_19.json --targets 5 --paper-mode --include test --calib-tiles all ^
  --adj0 0.25 --fit ridge --move boxmax --out out_e2_test50_adj0.25_ridge_boxmax
```

**4. 차수 비교 (추론 불필요)**
```bat
python analysis/order_compare.py out_e2_test50_adj10/trials_c5.csv out_e2_test50_adj0.25/trials_c5.csv out_e2_test50/trials_c5.csv ^
  --out analysis/results/order_summary_e2.csv
python analysis/order_by_n.py out_e2_test50_n125/trials_c5.csv --out analysis/results/order_by_n.csv
```

**5. 계수 추정 방법 비교 (최소제곱 / Ridge / 가중 최소제곱 / Huber / 붕괴 표본 제외, 추론 불필요)**
```bat
python analysis/solver_compare.py out_e2_test50_adj10/trials_c5.csv out_e2_test50_adj0.25/trials_c5.csv out_e2_test50/trials_c5.csv ^
  --labels 10% 25% 50% --out analysis/results/solver_summary.csv --theta-out analysis/results/solver_theta.csv
```
`solver_theta.csv`의 정상점·상자 내 최대점은 `analysis/reinfer_candidates.py` 또는 `run.py apply --theta`로 실제 IoU를 확인한다 (결과: `analysis/results/reinfer_iou.csv`).

**6. 그래프**
```bat
python analysis/plot_results.py
```

**7. 비례 전이 (E4·E5) / 모델 간 적용**
```bat
python run.py transfer --img-dir %DATA% --msk-dir %DATA% --checkpoint %CKPT% --n-classes 19 ^
  --label-map labelmap_19.json --source out_d004/result.json --targets 5 --out out_d067_from_d004
python run.py apply --img-dir %DATA% --msk-dir %DATA% --checkpoint models\mitb5\FLAIR-INC_rgb_15cl_mitb5-unet_weights.pth ^
  --encoder mit_b5 --n-classes 19 --label-map labelmap_19.json ^
  --theta-from out_e2_test50/result.json --targets 5 --out out_apply_mitb5
```

## 논문에 없어서 우리가 정한 값 (`--paper-mode`에서도 적용)

- adj는 **상대 비율** (θ × (1 ± adj)). 근거는 `docs/paper_vs_code.md`.
- Neighborhood 샘플은 **LHS**, seed 0.
- 정상점과 샘플은 **물리적 범위**(mean 0~255, std 1~255)로 자른다. 경계 없는 탐색에서 std ≤ 0을 막기 위한 장치.
- adj_t는 0.15 고정 (논문은 0.1–0.2 범위라고만 씀).
- `--max-evals` 80 상한 (Figure 1 (h)가 약 70회까지 진행). 본문의 R² 조건은 `--r2-min 0.9`로 켤 수 있음 (기본 꺼짐).
