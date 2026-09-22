"""반응면 모델 차수 비교 (trials_c*.csv만 사용, 추론 불필요).

사용:
    python analysis/order_compare.py out_e2_paper_full/trials_c5.csv
    python analysis/order_compare.py out_*/trials_c5.csv --boot 2000 --n-broad 30

출력:
    1) 모델별 계수 수, R2, Adj.R2, AIC, BIC, LOOCV RMSE, holdout RMSE
       - LOOCV: OLS hat 행렬 공식 e_i / (1 - h_ii)
       - holdout: 앞 n_broad개(LHS broad 단계)로 피팅 -> 나머지(local 단계) 예측
    2) 1차 모델 표준화 계수 (어느 파라미터가 IoU를 움직이는지)
    3) 논문 2차 모델 헤시안 고유값 (부호 섞이면 saddle)
    4) 부트스트랩 95% CI가 0을 포함하지 않는 계수 목록

모델:
    linear      1 + z_i                              (7)
    quad_diag   1 + z_i + z_i^2                      (13)
    diag_cubic  1 + z_i + z_i^2 + z_i^3              (19)
    quad        1 + z_i + z_i z_j (i<=j)  = 논문 식 1 (28)
    cubic_pure  quad + z_i^3                         (34)
입력은 컬럼별 표준화(z = (x - mean) / std) 후 피팅.
"""
import argparse
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]
SHORT = ["Rm", "Gm", "Bm", "Rs", "Gs", "Bs"]
KINDS = ["linear", "quad_diag", "diag_cubic", "quad", "cubic_pure"]


def features(Z, kind):
    n, d = Z.shape
    cols = [np.ones(n)] + [Z[:, i] for i in range(d)]
    if kind in ("quad", "cubic_pure"):
        cols += [Z[:, i] * Z[:, j] for i, j in combinations_with_replacement(range(d), 2)]
    elif kind in ("quad_diag", "diag_cubic"):
        cols += [Z[:, i] ** 2 for i in range(d)]
    if kind in ("diag_cubic", "cubic_pure"):
        cols += [Z[:, i] ** 3 for i in range(d)]
    return np.column_stack(cols)


def names(kind):
    out = ["1"] + SHORT
    if kind in ("quad", "cubic_pure"):
        out += [f"{a}*{b}" for a, b in combinations_with_replacement(SHORT, 2)]
    elif kind in ("quad_diag", "diag_cubic"):
        out += [f"{a}^2" for a in SHORT]
    if kind in ("diag_cubic", "cubic_pure"):
        out += [f"{a}^3" for a in SHORT]
    return out


def ols(A, y):
    return np.linalg.lstsq(A, y, rcond=None)[0]


def fit_stats(A, y, n_broad):
    n, p = A.shape
    b = ols(A, y)
    r = y - A @ b
    rss, tss = float(r @ r), float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss
    adj = 1 - (1 - r2) * (n - 1) / (n - p)
    aic = n * np.log(rss / n) + 2 * p
    bic = n * np.log(rss / n) + p * np.log(n)
    h = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
    loo = float(np.sqrt(np.mean((r / (1 - h)) ** 2))) if np.all(h < 1 - 1e-6) else np.nan
    ho = np.nan
    if n_broad and n_broad < n and p < n_broad:
        bb = ols(A[:n_broad], y[:n_broad])
        ho = float(np.sqrt(np.mean((y[n_broad:] - A[n_broad:] @ bb) ** 2)))
    return dict(p=p, r2=r2, adj_r2=adj, aic=aic, bic=bic, loo_rmse=loo, holdout_rmse=ho)


def hessian(beta, d=6):
    H = np.zeros((d, d))
    for k, (i, j) in enumerate(combinations_with_replacement(range(d), 2)):
        c = beta[1 + d + k]
        H[i, j] = H[j, i] = 2 * c if i == j else c
    return H


def bootstrap_sig(A, y, labels, n_boot, rng):
    n, p = A.shape
    B = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.linalg.matrix_rank(A[idx]) < p:
            continue
        B.append(ols(A[idx], y[idx]))
    if len(B) < 30:
        return len(B), None
    lo, hi = np.percentile(np.array(B), [2.5, 97.5], axis=0)
    return len(B), [labels[i] for i in range(p) if lo[i] > 0 or hi[i] < 0]


def analyze(path, n_boot, n_broad, seed):
    df = pd.read_csv(path)
    X, y = df[PARAMS].to_numpy(float), df["iou"].to_numpy(float)
    n = len(y)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(0)) / sd
    rng = np.random.default_rng(seed)

    print(f"\n## {path}  (n={n}, IoU {y.min():.2f} ~ {y.max():.2f})\n")
    print("| model | p | R2 | Adj.R2 | AIC | BIC | LOOCV RMSE | holdout RMSE |")
    print("|---|---|---|---|---|---|---|---|")
    rows = []
    for k in KINDS:
        A = features(Z, k)
        if A.shape[1] >= n:
            print(f"| {k} | {A.shape[1]} | 관측 부족 | | | | | |")
            continue
        s = fit_stats(A, y, n_broad)
        rows.append(dict(file=str(path), model=k, **s))
        print(f"| {k} | {s['p']} | {s['r2']:.3f} | {s['adj_r2']:.3f} | {s['aic']:.1f} | "
              f"{s['bic']:.1f} | {s['loo_rmse']:.2f} | {s['holdout_rmse']:.2f} |")
    base_loo = float(np.sqrt(np.mean(((y - y.mean()) * n / (n - 1)) ** 2)))
    print(f"| mean only | 1 | | | | | {base_loo:.2f} | |")

    bl = ols(features(Z, "linear"), y)
    print("\n1차 표준화 계수:", ", ".join(f"{s}={v:+.2f}" for s, v in zip(SHORT, bl[1:])))

    Aq = features(Z, "quad")
    if Aq.shape[1] < n:
        ev = np.linalg.eigvalsh(hessian(ols(Aq, y)))
        kind = "max" if np.all(ev < 0) else "min" if np.all(ev > 0) else "saddle"
        print(f"논문 2차 헤시안 고유값: {np.round(ev, 2).tolist()} -> {kind}")

    for k in ("quad", "quad_diag"):
        A = features(Z, k)
        if A.shape[1] >= n:
            continue
        valid, sig = bootstrap_sig(A, y, names(k), n_boot, rng)
        msg = "유효 표본 부족" if sig is None else f"{len(sig)}/{A.shape[1]} {sig}"
        print(f"부트스트랩 [{k}] (유효 {valid}/{n_boot}) 95% CI가 0 미포함: {msg}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="trials_c*.csv 경로 (여러 개 가능)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--n-broad", type=int, default=30, help="broad 단계 시행 수 (holdout 분리 기준)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="요약 CSV 저장 경로 (선택)")
    a = ap.parse_args()

    paths = []
    for c in a.csv:  # Windows cmd는 와일드카드를 확장하지 않으므로 직접 처리
        paths += sorted(Path().glob(c)) if any(ch in c for ch in "*?[") else [Path(c)]
    rows = []
    for p in paths:
        rows += analyze(p, a.boot, a.n_broad, a.seed)
    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
