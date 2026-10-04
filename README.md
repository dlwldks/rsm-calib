# rsm_calib — 학습 없는 채널 캘리브레이션 재현 코드

Moon & Cho (2026), *Remote Sensing* 18:205 재현용.
모델 가중치는 고정하고, 입력 정규화 값 6개(RGB mean/std)를 반응표면법(RSM)으로 튜닝한다.
논문과 코드의 항목별 대조는 [`docs/paper_vs_code.md`](docs/paper_vs_code.md).

## 파일

| 파일 | 역할 |
|---|---|
| `rsm.py` | 2차 회귀 피팅(식 1·2), R²/Adj.R²/F/p, 정상점(식 3), 비례 전이(식 4) |
| `search.py` | 탐색. `rsm_search_paper`(논문 Figure 1) / `rsm_search_ours`(우리 변형) / `broad_sweep`(1단계 폭 비교) |
| `data.py` | TIF 로딩, 라벨 매핑, smp U-Net 로딩, 추론/IoU, decision fusion |
| `run.py` | CLI: `synthetic` / `inspect` / `calibrate` / `broad` / `apply` / `transfer` |
| `analysis/order_compare.py` | trials csv로 1~4차 반응면 모델 비교 (추론 불필요) |
| `labelmap_19.json` | FLAIR 마스크 값(1~19) → 모델 출력 인덱스(19채널) |

## 설치

```bash
pip install -r requirements.txt
```
GPU 컴퓨터에서는 `torch`, `torchvision` 두 줄을 빼고, CUDA 버전에 맞는 torch를 따로 설치한 뒤 `--device cuda`.

## 탐색 모드

| | `--paper-mode` (논문 Figure 1) | 기본 (우리 변형) |
|---|---|---|
| 1단계 | θ0 포함 30개, θ0 × (1 ± 0.5) | θ0 포함 30개, θ0 × (1 ± 0.10) |
| 이후 회차 | 정상점 θ̂ 포함 5개, θ̂ × (1 ± 0.15) | 상자 안 곡면 최댓값 1개 + 주변 6개, ±0.03 |
| 탐색 경계 | 없음 (정상점 따라 이동) | θ0 ± 10% 상자 고정 |
| 정상점이 saddle/min | 그대로 사용 | 상자 안 곡면 최댓값 사용 |
| ε1 | 이번 회차 best − 지난 회차 best | 누적 best 개선폭 |
| ε2 | \|이번 회차 best − 예측 IoU\| | \|정상점 실제 IoU − 예측 IoU\| |
| 종료 | ε1 ≤ ε_stop **OR** ε2 ≤ ε_stop (ε_stop=0) | ε1 ≤ 0.1 **AND** ε2 ≤ 1.0 |
| calib/test 분리 | 없음 | 70/30 |

`--max-evals`(논문 모드 80, 우리 모드 60)에서 종료. 모든 폭은 `--adj0 / --adj-t / --n0 / --n-t / --eps-stop`으로 바꿀 수 있다.

## 순서

**0. 로직 검증 (모델·데이터 불필요)**
```bash
python run.py synthetic --paper-mode
```

**1. 데이터/모델 점검**
```bash
python run.py inspect --img-dir <DIR> --msk-dir <DIR> --checkpoint <CKPT> --n-classes 19
```

**2. 1단계 탐색 폭 비교 (adj0 결정용)**
```bash
python run.py broad --img-dir <DIR> --msk-dir <DIR> --checkpoint <CKPT> --n-classes 19 ^
  --label-map labelmap_19.json --targets 5 --adjs 0.1 0.25 0.5 --out out_broad
```
adj0별 IoU 분포(최저·중앙·최고, 5 미만 개수)와 1차/2차 피팅 LOOCV를 `broad_summary.csv`로 저장.

**3. 캘리브레이션 (E2 재현)**
```bash
python run.py calibrate --img-dir <DIR> --msk-dir <DIR> --checkpoint <CKPT> --n-classes 19 ^
  --label-map labelmap_19.json --targets 5 --paper-mode --out out_e2_fig1
```
출력: `result.json`(θ, 비율, RSM 통계, 회차별 ε1·ε2, 종료 사유), `trials_c*.csv`(모든 추론 기록 + 회차 번호), `iou_table.csv`.

**3-1. 논문 E2 조건 (toy test 50타일 전부로 보정, 기준 = 타일 전체 채널 통계 한 세트)**

이 조건에서 기준 IoU가 논문 Table 6 (b)와 14개 클래스 모두 ±1.2 이내로 재현됨 (`docs/paper_vs_code.md` 8절).
```bash
python run.py calibrate --img-dir <DIR> --msk-dir <DIR> --checkpoint <CKPT> --n-classes 19 ^
  --label-map labelmap_19.json --targets 5 --paper-mode ^
  --include test --calib-tiles all --out out_e2_test50
```

**4. 차수 비교**
```bash
python analysis/order_compare.py out_e2_fig1/trials_c5.csv
```

**4-1. 계수 추정 방법 비교 (최소제곱 / Ridge / 가중 최소제곱 / Huber / 붕괴 표본 제외)**
```bash
python analysis/solver_compare.py out_e2_test50_adj10/trials_c5.csv out_e2_test50_adj0.25/trials_c5.csv out_e2_test50/trials_c5.csv ^
  --labels 10% 25% 50% --out analysis/results/solver_summary.csv --theta-out analysis/results/solver_theta.csv
```
결과는 `analysis/results/`. `solver_theta.csv`의 정상점·상자 내 최대점은 `run.py apply --theta`로 실제 IoU를 확인한다.

**5. 비례 전이 (E4·E5) / 모델 간 적용**
```bash
python run.py transfer ... --source out_d004/result.json --targets 5 --out out_d067_from_d004
python run.py apply ... --encoder mit_b5 --theta-from out_e2_fig1/result.json --targets 5 --out out_apply_mitb5
```

## 논문에 없어서 우리가 정한 값 (`--paper-mode`에서도 적용)

- adj는 **상대 비율** (θ × (1 ± adj)). 근거는 `docs/paper_vs_code.md`.
- Neighborhood 샘플은 **LHS**, seed 0.
- 정상점과 샘플은 **물리적 범위**(mean 0~255, std 1~255)로 자른다. 경계 없는 탐색에서 std ≤ 0을 막기 위한 장치.
- adj_t는 0.15 고정 (논문은 0.1–0.2 범위라고만 씀).
- `--max-evals` 80 상한 (Figure 1 (h)가 약 70회까지 진행). 본문의 R² 조건은 `--r2-min 0.9`로 켤 수 있음 (기본 꺼짐).
