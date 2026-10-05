# rsm_calib — 학습 없는 채널 캘리브레이션 재현 코드

Moon & Cho (2026), "Training-Free Lightweight Transfer Learning for Land Cover Segmentation Using Multispectral Calibration", *Remote Sensing* 18(2):205 재현용 (논문 PDF: `remotesensing-18-00205-v2.pdf`).
모델 가중치는 고정하고, 입력 정규화 값 6개(R·G·B 평균, R·G·B 표준편차 = θ)를 반응표면법(RSM)으로 찾는다.

## 현재 상태

IoU는 대상 클래스(E1·E2·E4·E5 침엽수, E3 가로수) IoU(%). 직접 탐색은 모두 seed 0 한 번.

| 실험 | 데이터 | 논문 (기준 → 최적) | 재현 결과 | 남은 것 |
|---|---|---|---|---|
| E1 | AI-HUB 71361 AP12 32장 (12cm UAV) | 13.57 → 81.43 | 기준: 32장 통계 θ 8.01, 역추정 θ 11.82. 탐색 125회: 8.01 → 71.26, 11.82 → 61.92 | seed 반복. 기준 θ 평균·영상 전처리는 논문으로 확인 불가 |
| E2 | FLAIR toy test 50장 (논문 47장) | 13.36 → 75.24 | 기준 13.34 (14개 클래스 ±1.2 이내). 탐색 최고 45.61, θ 하나 경사하강 상한 37.70 | 47장 재실행, 타일별 선택 가설 확인 |
| E3a·E3b | AI-HUB AP12·AP25, AI-HUB U-Net | 80.86 → 82.65, 64.27 → 66.46 | 시작 전 | 모델 정규화 값·클래스 순서 확인, 타일 선택 |
| E4 | FLAIR D004 50장 | 25.77 → 42.80 (Table 8 b1·c1 열이 바뀌어 기재된 것으로 판단, θbest 42.61) | 논문 θ 적용: θbest 42.59, 비율 전이 42.85. MiT-B5 a2·b2·c2 논문과 같음. 직접 탐색 41.62~43.68 | ResNet34 기준(a1) 침엽수·활엽수 차이 원인. seed 반복 배치(`scripts/laptop_e4_seeds.bat`) 미실행 |
| E5 | FLAIR D067 150장 | 75.93 → 80.35 | Table 9 침엽수 6개 열(ResNet34·MiT-B5 × 기준·최적·비율 전이) 0.01 이내 재현. 직접 탐색 75.93 → 81.85 | seed 반복, 수역 차이(+0.4~1.7) 원인 |

- 직접 탐색의 최고값은 E1을 빼면 대부분 1단계 무작위 표본에서 나왔고, 2차식 정상점은 거의 모두 안장점이었다.
- 논문 표·본문 사이 오기재 13곳과 Table 8 열 바뀜은 보고서 12.1절과 `docs/paper_vs_code.md`에 정리.
- 보고서(계속 업데이트): https://claude.ai/code/artifact/c3d07ecf-8258-4775-9d7c-041bb37ed40b

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/experiments.md`](docs/experiments.md) | 실험 목록 (번호·폴더·조건·결과·판단) |
| [`docs/paper_vs_code.md`](docs/paper_vs_code.md) | 논문과 코드의 항목별 대조, 기준 IoU 재현, 논문 표 확인 기록 |
| [`docs/paper_tiles/`](docs/paper_tiles/README.md) | 저자 시범 서비스(EcoVision)에서 찾은 E1·E2·E4·E5 타일 목록, E2 저자 지도 IoU |
| [`docs/aihub_data.md`](docs/aihub_data.md) | AI-HUB 71361 데이터 확인 내용, E1 라벨 매핑 |
| [`docs/flair1_spec/`](docs/flair1_spec/summary.md) | FLAIR #1 전체 데이터셋 명세 통계 |
| [`reports/`](reports/README.md) | 미팅별 보고서 (날짜별 폴더) |

## 파일

| 파일 | 역할 |
|---|---|
| `rsm.py` | 2차 회귀 피팅(식 1·2), R²/Adj.R²/F/p, 정상점(식 3), 비례 전이(식 4) |
| `search.py` | 탐색. `rsm_search_paper`(논문 Figure 1) / `rsm_search_ours`(우리 변형) / `broad_sweep`(1단계 폭 비교) |
| `data.py` | TIF 로딩, 타일 번호 매칭(`tile_key`), 라벨 매핑, smp U-Net 로딩, 추론/IoU, decision fusion |
| `run.py` | CLI: `synthetic` / `inspect` / `calibrate` / `broad` / `apply` / `transfer` |
| `labelmap_19.json` | FLAIR 마스크 값(1~19) → 모델 출력 인덱스(19채널) |
| `labelmap_aihub_e1.json` | AI-HUB 코드 → FLAIR 인덱스 (논문 Table 4의 E1 5개 클래스) |
| `scripts/laptop_e5_transfer_mitb5.bat` | E5 논문 θ 적용, E4·E5 비율 전이, MiT-B5 적용 (탐색 없음) |
| `scripts/laptop_e4_seeds.bat` | E4 seed 반복 A/R/B × seed 0·1·2, 각 125회 |
| `analysis/order_compare.py` | trials csv로 1~4차 반응면 모델 비교 (추론 불필요) |
| `analysis/order_by_n.py` | 관측 수(35~125)별 1차·2차 LOO 비교 (추론 불필요) |
| `analysis/solver_compare.py` | 계수 추정 방법 5가지 비교 (추론 불필요) |
| `analysis/reinfer_candidates.py` | 후보 θ(정상점·상자 내 최대점)를 실제로 추론해 IoU 확인 |
| `analysis/theta_upper.py` | 진단용: θ 6개를 soft IoU 경사하강으로 직접 올려 θ 하나로 도달 가능한 IoU 확인 (논문 방법 아님) |
| `analysis/flair_spec.py` | FLAIR #1 전체 데이터셋 명세 통계 추출 (모델 불필요) |
| `analysis/full_eval.py` | 전체 test 15,700타일 기준 IoU (pooled·타일별 정규화) |
| `analysis/full_candidates.py` | 후보 θ를 전체 test 표본(도메인 10개 × 150장)에 적용 |
| `analysis/plot_results.py` | `analysis/results/`의 그래프(`fig_*.png`) 생성 |
| `analysis/results/` | 분석·실험 결과 csv·json (`e1/`, `e4/`, `e5/`, `e45_paper_theta/`, `mitb5/`, 전체 test 결과 등) |
| `out_*/` | 탐색 결과 폴더. 기본은 `.gitignore`라 올릴 때 `git add -f`. 폴더별 설명은 `docs/experiments.md` |

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
GPU 컴퓨터에서는 `requirements.txt`의 `torch`, `torchvision` 두 줄을 빼고, CUDA 버전에 맞는 torch를 따로 설치한 뒤 `--device cuda`.

## 데이터·모델 준비

용량 때문에 레포에 포함하지 않는다 (`.gitignore`: `flair_1_toy_dataset/`, `models/`, `*.tif`, `*.pth`, `out*/`).

```
rsm_calib/
├── flair_1_toy_dataset/flair_1_toy_dataset/   E2 (toy test 50타일)
│   ├── flair_1_toy_aerial_test/    <도메인>/<구역>/img/IMG_*.tif
│   └── flair_1_toy_labels_test/    <도메인>/<구역>/msk/MSK_*.tif
└── models/
    ├── rgb15/FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth
    └── mitb5/FLAIR-INC_rgb_15cl_mitb5-unet_weights.pth   (MiT-B5 적용 실험)
```

| 항목 | 쓰는 실험 | 출처 · 위치 |
|---|---|---|
| FLAIR #1 toy dataset | E2 | IGNF (https://github.com/IGNF/FLAIR-1 안내에 따라 다운로드) |
| FLAIR #1 전체 | E4 (D004_2021), E5 (D067_2021), 전체 test 평가 | 노트북 예: `D:\flair1_hf\extracted\train\<도메인>\aerial`, `...\labels` |
| AI-HUB 71361 AP12 | E1 (32장) | AI-HUB "토지피복지도 항공위성 이미지". 파일명 `LC_GG_AP12_<번호>_<연도>.tif` (`docs/aihub_data.md`) |
| ResNet34 U-Net | E1·E2·E4·E5 | Hugging Face `IGNF/FLAIR-INC_rgb_15cl_resnet34-unet` |
| MiT-B5 U-Net | E4·E5 | Hugging Face `IGNF/FLAIR-INC_rgb_15cl_mitb5-unet` |
| AI-HUB U-Net | E3 | AI-HUB 71361 모델 (받음, 실험 전) |

- toy: `--img-dir`와 `--msk-dir`를 둘 다 `flair_1_toy_dataset/flair_1_toy_dataset`로 주고 `--include test`로 test 50타일만 쓴다.
- 논문 타일만 쓸 때는 `--tiles docs/paper_tiles/<목록>.txt` (E1 `e1_ap12_tiles32.txt`, E2 `e2_tiles47.txt`, E4 `e4_d004_tiles50.txt`, E5 `e5_d067_tiles150.txt`).

## 탐색 모드

| | `--paper-mode` (논문 Figure 1) | 기본 (우리 변형) |
|---|---|---|
| 1단계 | θ0 포함 30개, θ0 × (1 ± adj0) (기본 0.5) | θ0 포함 30개, θ0 × (1 ± 0.10) |
| 이후 회차 | 정상점 θ̂ 포함 5개, θ̂ × (1 ± 0.15) | 상자 안 곡면 최댓값 1개 + 주변 6개, ±0.03 |
| 탐색 경계 | 없음 (정상점 따라 이동) | θ0 ± 10% 상자 고정 |
| 정상점이 saddle/min | 그대로 사용 (`--move boxmax`로 변경 가능) | 상자 안 곡면 최댓값 사용 |
| ε1 | 이번 회차 best − 지난 회차 best | 누적 best 개선폭 |
| ε2 | \|이번 회차 best − 예측 IoU\| | \|정상점 실제 IoU − 예측 IoU\| |
| 종료 | ε1 ≤ ε_stop **OR** ε2 ≤ ε_stop (ε_stop=0) | ε1 ≤ 0.1 **AND** ε2 ≤ 1.0 |
| calib/test 분리 | 없음 | 70/30 |

`--max-evals`(기본: 논문 모드 80, 우리 모드 60)에서 종료. 현재 탐색 실험은 `--adj0 0.1 --eps-stop -1000 --max-evals 125`(종료 조건 끄고 125회 고정)로 돌린다.

### 추가 옵션

| 옵션 | 기본 | 내용 |
|---|---|---|
| `--tiles <목록 파일>` | 없음 | 목록에 있는 타일 번호만 사용 (`docs/paper_tiles/*.txt`) |
| `--calib-tiles target\|all` | `target` | `all`이면 대상 클래스 유무와 상관없이 모든 타일로 보정 (논문 재현은 `all`) |
| `--base-mode pooled\|per-tile\|given` | `pooled` | 기준 θ0 계산 방식. `pooled` = 보정 타일 전체 채널 통계 한 세트, `given` = `--theta0`로 직접 지정 (예: 논문 θbase) |
| `--theta0 <6개>` | 없음 | `--base-mode given`일 때 R·G·B 평균, R·G·B 표준편차 |
| `--fit ols\|ridge` | `ols` | 회차마다 2차식 계수 추정 방법. `ridge`는 λ를 LOO로 선택 (`--ridge-lambda`로 고정 가능) |
| `--move stationary\|boxmax` | `stationary` | 논문 모드 다음 회차 중심. `boxmax`는 정상점이 극대가 아니면 1단계 범위 안 곡면 최대점으로 이동 (논문에 없는 변형) |
| `--r2-min 0.9` | 꺼짐 | 본문 3.3절 "R² ≥ 0.9까지 반복" 적용 |
| `--steepest` | 꺼짐 | 1차식 최급상승 후 2차식 (고전 RSM 순서, 논문에 없는 변형) |
| `--resume` | 꺼짐 | `--out` 폴더의 `progress_c*.csv`로 끊긴 탐색 이어가기 (같은 seed·조건) |
| `--reuse <trials csv>` | 없음 | 이전 기록에 같은 θ가 있으면 추론 없이 기록값 사용 |
| `--seed` | 0 | 1단계 표본(LHS) seed |

## 실행 순서

Windows cmd 기준 (줄 이어쓰기 `^`).

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

**2. E2: toy test 50타일 (기준 = 타일 전체 채널 통계 한 세트)**

기준 IoU가 논문 Table 6 (b)와 14개 클래스 모두 ±1.2 이내로 재현됨 (`docs/paper_vs_code.md` 8절).
```bat
python run.py calibrate --img-dir %DATA% --msk-dir %DATA% --checkpoint %CKPT% --n-classes 19 ^
  --label-map labelmap_19.json --targets 5 --paper-mode ^
  --include test --calib-tiles all --out out_e2_test50
```
출력: `result.json`(θ, 비율, RSM 통계, 회차별 ε1·ε2, 종료 사유), `trials_c*.csv`(모든 추론 기록 + 회차 번호), `progress_c*.csv`, `iou_table.csv`.

1단계 폭 비교는 `run.py broad ... --adjs 0.1 0.25 0.5 --out out_broad`.

**3. 차수·계수 추정 방법 비교 (추론 불필요)**
```bat
python analysis/order_compare.py out_e2_test50_adj10/trials_c5.csv out_e2_test50_adj0.25/trials_c5.csv out_e2_test50/trials_c5.csv ^
  --out analysis/results/order_summary_e2.csv
python analysis/order_by_n.py out_e2_test50_n125/trials_c5.csv --out analysis/results/order_by_n.csv
python analysis/solver_compare.py out_e2_test50_adj10/trials_c5.csv out_e2_test50_adj0.25/trials_c5.csv out_e2_test50/trials_c5.csv ^
  --labels 10% 25% 50% --out analysis/results/solver_summary.csv --theta-out analysis/results/solver_theta.csv
python analysis/plot_results.py
```
`solver_theta.csv`의 후보 θ는 `analysis/reinfer_candidates.py` 또는 `run.py apply --theta`로 실제 IoU를 확인한다 (결과: `analysis/results/reinfer_iou.csv`).

**4. E4·E5: 논문 θ 적용, 비율 전이, MiT-B5 (탐색 없음)**
```bat
scripts\laptop_e5_transfer_mitb5.bat
```
한 줄 예 (E5 논문 θbest):
```bat
set F=D:\flair1_hf\extracted\train
python run.py apply --img-dir %F%\D067_2021\aerial --msk-dir %F%\D067_2021\labels ^
  --tiles docs\paper_tiles\e5_d067_tiles150.txt --checkpoint %CKPT% --n-classes 19 ^
  --label-map labelmap_19.json --targets 5 --theta 98.05 114.46 103.70 62.32 56.19 53.94 --out out_e5_paper_best
```
MiT-B5는 `--checkpoint models\mitb5\FLAIR-INC_rgb_15cl_mitb5-unet_weights.pth --encoder mit_b5`.

**5. E4·E5: 직접 탐색 (논문 θbase 출발, 125회)**
```bat
python run.py calibrate --img-dir %F%\D067_2021\aerial --msk-dir %F%\D067_2021\labels ^
  --tiles docs\paper_tiles\e5_d067_tiles150.txt --checkpoint %CKPT% --n-classes 19 ^
  --label-map labelmap_19.json --targets 5 --base-mode given ^
  --theta0 97.28 108.99 109.48 62.47 56.64 54.35 ^
  --paper-mode --calib-tiles all --adj0 0.1 --eps-stop -1000 --max-evals 125 --seed 0 ^
  --out out_e5_A_seed0_n125
```
E4 seed 반복(A 논문 절차 / R 무작위 대조군 / B 상자 내 최대점 이동, seed 0·1·2)은 `scripts\laptop_e4_seeds.bat`.

**6. E1: AI-HUB AP12 32장**
```bat
python run.py calibrate --img-dir <AP12 영상 폴더> --msk-dir <AP12 라벨 Tif 폴더> ^
  --img-glob LC_*.tif --msk-glob LC_*.tif --tiles docs\paper_tiles\e1_ap12_tiles32.txt ^
  --checkpoint %CKPT% --n-classes 19 --label-map labelmap_aihub_e1.json --targets 5 ^
  --paper-mode --calib-tiles all --adj0 0.1 --eps-stop -1000 --max-evals 125 --out out_e1_n125
```

**7. 전체 test 평가 (FLAIR #1 전체)**
```bat
python analysis/full_eval.py --img-dir <영상 루트> --msk-dir <라벨 루트> --include test ^
  --checkpoint %CKPT% --n-classes 19 --label-map labelmap_19.json --out out_full_test --resume
python analysis/full_candidates.py --spec out_full_test/spec_tiles.csv --checkpoint %CKPT% ^
  --label-map labelmap_19.json --per-domain 150 --out out_full_candidates --resume
```
요약 결과: `analysis/results/full_test_base_iou.csv`, `full_candidates_iou.csv`, `full_candidates_seed12_iou.csv`.

## 논문에 없어서 우리가 정한 값 (`--paper-mode`에서도 적용)

- adj는 **상대 비율** (θ × (1 ± adj)). 근거는 `docs/paper_vs_code.md`.
- Neighborhood 샘플은 **LHS**, seed 0 기본.
- 정상점과 샘플은 **물리적 범위**(mean 0~255, std 1~255)로 자른다. 경계 없는 탐색에서 std ≤ 0을 막기 위한 장치.
- adj_t는 0.15 고정 (논문은 0.1–0.2 범위라고만 씀).
- 추론 횟수: 논문 본문은 40~50회, Table 10 통계 역산은 125회. 현재 탐색 실험은 125회 고정.
