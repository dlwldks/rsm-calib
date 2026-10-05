"""관측 수 n에 따라 1차·2차 중 어느 쪽 LOO 오차가 작은지 (trials csv만 사용).

사용: python analysis/order_by_n.py out_e2_test50_n125/trials_c5.csv --out analysis/results/order_by_n.csv
- 앞에서부터 n개: 탐색이 n회에서 멈췄다면 얻었을 기록
- 무작위 n개: 125개 중 n개를 200번 뽑은 평균

보고서 6.1 (관측 수에 따른 1차·2차 경계)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from order_compare import PARAMS, features  # noqa: E402

KINDS = {"linear": "1차(7)", "quad_diag": "2차 교차항 없음(13)", "quad": "2차 논문 식 1(28)"}


def loo(Z, y, kind):
    """표준화 좌표 Z와 IoU y에 kind 식을 맞췄을 때의 LOO RMSE. 관측이 계수 수 + 1 이하이면 NaN."""
    A = features(Z, kind)
    n, p = A.shape
    if n <= p + 1:
        return np.nan
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ b
    h = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
    return float(np.sqrt(np.mean((r / (1 - h)) ** 2))) if np.all(h < 1 - 1e-6) else np.nan


def scores(X, y):
    """1차·2차(교차항 없음)·논문 식 1의 LOO RMSE와, 식 없이 평균값만으로 예측했을 때의 LOO RMSE."""
    sd = X.std(0)
    sd[sd == 0] = 1
    Z = (X - X.mean(0)) / sd
    out = {k: loo(Z, y, k) for k in KINDS}
    n = len(y)
    out["mean"] = float(np.sqrt(np.mean(((y - y.mean()) * n / (n - 1)) ** 2)))
    return out


def main():
    """n마다 앞에서부터 n개, 무작위 n개(reps번 평균) 두 방식으로 LOO를 비교해 표로 저장한다 (보고서 6.1)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--ns", type=int, nargs="+", default=[35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 125])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    X, y = df[PARAMS].to_numpy(float), df["iou"].to_numpy(float)
    rng = np.random.default_rng(0)
    rows = []
    for n in a.ns:
        s = scores(X[:n], y[:n])
        rows.append(dict(n=n, sample="앞에서부터", **s))
        reps = [scores(X[i], y[i]) for i in (rng.choice(len(y), n, replace=False) for _ in range(a.reps))]
        rows.append(dict(n=n, sample="무작위 평균", **{k: float(np.nanmedian([r[k] for r in reps])) for k in reps[0]}))
    t = pd.DataFrame(rows)
    t["best"] = t[list(KINDS)].idxmin(axis=1).map(KINDS)
    print(t.round(2).to_string(index=False))
    if a.out:
        t.round(4).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
