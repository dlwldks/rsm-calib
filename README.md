# rsm_calib — 학습 없는 채널 캘리브레이션 재현 코드

Moon & Cho (2026), *Remote Sensing* 18:205 재현용.
모델 가중치는 고정하고, 입력 정규화 값 6개(RGB mean/std)를 반응표면법(RSM)으로 튜닝한다.

## 파일

| 파일 | 역할 |
|---|---|
| `rsm.py` | 2차 회귀 피팅(식 1·2), R²/Adj.R²/F/p, 정상점(식 3), 비례 전이(식 4) |
| `search.py` | 2단계 순차 탐색 (넓게 LHS 30회 → 최적점 주변 좁혀서 반복) |
| `data.py` | TIF 로딩, 라벨 매핑, smp U-Net 로딩, 추론/IoU, decision fusion |
| `run.py` | CLI: `synthetic` / `inspect` / `calibrate` / `transfer` |

## 설치

```bash
pip install numpy scipy torch segmentation-models-pytorch rasterio safetensors
```

## 순서

**0. 로직 검증 (모델·데이터 불필요)**
```bash
python run.py synthetic
```
정답 최적점을 아는 가짜 IoU 함수로 탐색이 수렴하는지 확인.

**1. 데이터/모델 점검**
```bash
python run.py inspect --img-dir flair_toy --msk-dir flair_toy \
  --checkpoint FLAIR_rgb_resnet34.safetensors
```
- 이미지 밴드 수, 마스크 원본 값 분포, 모델 출력 클래스 수를 확인한다.
- **꼭 확인할 것**: 마스크 값(FLAIR는 1~19)을 모델 출력 인덱스로 옮기는 매핑.
  기본값은 `1..n → 0..n-1`인데, 모델 카드와 다르면 `labelmap.json`을 만들어 `--label-map`으로 넘긴다.
  ```json
  {"1": 0, "2": 1, "13": 12, "18": 13, "19": 14, "14": null}
  ```
- 모델이 `(x-mean)/std`를 0~255 스케일에서 쓰는지도 모델 카드에서 확인 (코드는 0~255 기준).

**2. 캘리브레이션 (E2 재현)**
```bash
python run.py calibrate --img-dir flair_toy --msk-dir flair_toy \
  --checkpoint FLAIR_rgb_resnet34.safetensors --target 5 --out out_e2 \
  --class-names building,pervious,impervious,bare_soil,water,coniferous,deciduous,brushwood,vineyard,herbaceous,agricultural,plowed,swimming_pool,greenhouse,other
```
- `--target`: 모델 클래스 인덱스 (위 순서라면 coniferous=5)
- 타깃 클래스가 있는 타일을 **calib 70% / test 30%로 분리**한다. 논문은 분리하지 않았으니, test 열에서 개선이 유지되는지가 핵심 확인 포인트.
- MiT-B5는 `--encoder mit_b5`
- 출력: `result.json`(theta, 비율, RSM 통계), `trials.csv`(모든 추론 기록), `iou_table.csv`(base / best / fusion)

**3. 비례 전이 (E4·E5 재현)**
```bash
python run.py transfer --img-dir D067 --msk-dir D067 --checkpoint ... \
  --source out_d004/result.json --target 5 --out out_d067_from_d004
```
D004에서 찾은 `best/base` 비율을 D067 기본 통계에 곱해서 재탐색 없이 평가한다.

## 논문과 다른 점

- 종료 조건: 논문은 (ε1 또는 ε2), 여기선 **둘 다** 만족해야 멈춤 (조기 종료 방지). `--max-evals`로 상한.
- 탐색 범위: 논문의 `adj`는 단위가 모호해서 base 대비 **상대 비율**(±10% → ±3%)로 둠.
  논문 최적값이 base 대비 대략 ±7% 안이라 이 범위면 충분.
- 정상점이 극대가 아니거나 범위 밖이면, 범위 내 L-BFGS-B로 반응면 최댓값을 찾는다.
- calib/test 분리 추가.

## CPU 시간 감각

512×512 타일, ResNet34 기준 타일당 약 0.3~0.5초 → 47타일 × 60회 추론이면 15~25분 정도.
처음엔 `--limit 30`, `--max-evals 40`으로 짧게 돌려보는 것 추천.
