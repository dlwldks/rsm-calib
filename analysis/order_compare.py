"""반응면 모델 차수 비교 (trials csv만 사용, 추론 불필요).

사용:
    python analysis/order_compare.py out_e2_fig1/trials_c5.csv
    python analysis/order_compare.py "out_broad/trials_c5_adj*.csv" --pool --out order_summary.csv

같은 (theta, IoU) 데이터에 차수만 바꾼 식을 각각 최소제곱으로 맞추고 비교한다.

모델 (입력은 컨럼별 표준화 z = (x - mean) / std):
    linear        1 + z_i                                  계수 7   (1차)
    quad_diag     linear + z_i^2                           계수 13  (2차, 교차항 없음)
    cubic_diag    quad_diag + z_i^3                        계수 19  (3차, 교차항 없음)
    quartic_diag  cubic_diag + z_i^4                       계수 25  (4차, 교차항 없음)
    quad          1 + z_i + z_i z_j (i<=j) = 논문 식 1     계수 28  (2차, 교차항 포함)
교차항까지 넣으면 3차 84개, 4차 210개라 관측 수로 풀 수 없어서, 1~4차 비교는 교차항 없는 형태로 맞춘다.

지표:
    R2, Adj.R2, AIC, BIC  : 학습 데이터 적합도 (계수가 많을수록 유리)
    LOOCV RMSE            : 점 하나를 빼고 맞춘 식으로 그 점을 예측한 오차 (낮을수록 좋음, 비교 기준)
    holdout RMSE          : 앞 n_broad개로 맞추고 나머지를 예측 (파일에 2단계 점이 있을 때만)

보고서 2.6 (1~4차 비교)
"""
import argparse
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]
SHORT = ["Rm", "Gm", "Bm", "Rs", "Gs", "Bs"]
KINDS = ["linear", "quad_diag", "cubic_diag", "quartic_diag", "quad"]
ORDER = {"linear": 1, "quad_diag": 2, "cubic_diag": 3, "quartic_diag": 4, "quad": 2}


def features(Z, kind):
    """차수별 설계행렬. quad는 논문 식 1(교차항 포함, 28열), 나머지는 1차 + 각 변수의 거듭제곱(교차항 없음)."""
    n, d = Z.shape
    cols = [np.ones(n)] + [Z[:, i] for i in range(d)]
    if kind == "quad":
        cols += [Z[:, i] * Z[:, j] for i, j in combinations_with_replacement(range(d), 2)]
    else:
        for power in range(2, ORDER[kind] + 1):
            cols += [Z[:, i] ** power for i in range(d)]
    return np.column_stack(cols)


def names(kind):
    """설계행렬 열 이름 (계수 표 출력용)."""
    out = ["1"] + SHORT
    if kind == "quad":
        out += [f"{a}*{b}" for a, b in combinations_with_replacement(SHORT, 2)]
    else:
        for power in range(2, ORDER[kind] + 1):
            out += [f"{a}^{power}" for a in SHORT]
    return out


def ols(A, y):
    """최소제곱 계수."""
    return np.linalg.lstsq(A, y, rcond=None)[0]


def fit_stats(A, y, n_broad):
    """한 식의 R², 조정 R², AIC, BIC, LOO RMSE, holdout RMSE를 계산한다."""
    n, p = A.shape
    b = ols(A, y)
    r = y - A @ b
    rss, tss = float(r @ r), float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss
    adj = 1 - (1 - r2) * (n - 1) / (n - p)
    aic = n * np.log(max(rss, 1e-12) / n) + 2 * p
    bic = n * np.log(max(rss, 1e-12) / n) + p * np.log(n)
    h = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
    loo = float(np.sqrt(np.mean((r / (1 - h)) ** 2))) if np.all(h < 1 - 1e-6) else np.nan
    ho = np.nan
    if n_broad and n_broad < n and p < n_broad:
        bb = ols(A[:n_broad], y[:n_broad])
        ho = float(np.sqrt(np.mean((y[n_broad:] - A[n_broad:] @ bb) ** 2)))
    return dict(p=p, df_resid=n - p, r2=r2, adj_r2=adj, aic=aic, bic=bic,
                loo_rmse=loo, holdout_rmse=ho)


def hessian(beta, d=6):
    """논문 식 1 계수에서 헤시안(대각 = 제곱 계수 × 2, 비대각 = 교차 계수)을 만든다."""
    H = np.zeros((d, d))
    for k, (i, j) in enumerate(combinations_with_replacement(range(d), 2)):
        c = beta[1 + d + k]
        H[i, j] = H[j, i] = 2 * c if i == j else c
    return H


def bootstrap_sig(A, y, labels, n_boot, rng):
    """부트스트랩(표본 복원 추출)으로 계수별 95% 구간을 구해, 0을 포함하지 않는 계수 수를 센다."""
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


def analyze(label, df, n_boot, n_broad, seed):
    """한 탐색 기록에 차수별 식을 모두 맞추고 표로 출력한다. 반환: 요약 행 목록."""
    X, y = df[PARAMS].to_numpy(float), df["iou"].to_numpy(float)
    n = len(y)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(0)) / sd
    rng = np.random.default_rng(seed)

    print(f"\n## {label}  (n={n}, IoU {y.min():.2f} ~ {y.max():.2f})\n")
    print("| model | 차수 | 계수 | 잔차df | R2 | Adj.R2 | AIC | BIC | LOOCV RMSE | holdout RMSE |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for k in KINDS:
        A = features(Z, k)
        if A.shape[1] >= n:
            print(f"| {k} | {ORDER[k]} | {A.shape[1]} | - | 관측 부족 | | | | | |")
            continue
        s = fit_stats(A, y, n_broad)
        rows.append(dict(data=label, model=k, order=ORDER[k], n=n, **s))
        print(f"| {k} | {ORDER[k]} | {s['p']} | {s['df_resid']} | {s['r2']:.3f} | {s['adj_r2']:.3f} | "
              f"{s['aic']:.1f} | {s['bic']:.1f} | {s['loo_rmse']:.2f} | {s['holdout_rmse']:.2f} |")
    base_loo = float(np.sqrt(np.mean(((y - y.mean()) * n / (n - 1)) ** 2)))
    print(f"| mean only | 0 | 1 | {n - 1} | | | | | {base_loo:.2f} | |")
    valid = [r for r in rows if not np.isnan(r["loo_rmse"])]
    if valid:
        best = min(valid, key=lambda r: r["loo_rmse"])
        print(f"\nLOOCV 최소: {best['model']} ({best['loo_rmse']:.2f})")

    bl = ols(features(Z, "linear"), y)
    print("1차 표준화 계수:", ", ".join(f"{s}={v:+.2f}" for s, v in zip(SHORT, bl[1:])))
    bd = ols(features(Z, "quad_diag"), y)
    print("2차(대각) 제곱항 계수:", ", ".join(f"{s}^2={v:+.2f}" for s, v in zip(SHORT, bd[7:13])))

    Aq = features(Z, "quad")
    if Aq.shape[1] < n:
        ev = np.linalg.eigvalsh(hessian(ols(Aq, y)))
        kind = "max" if np.all(ev < 0) else "min" if np.all(ev > 0) else "saddle"
        print(f"논문 2차 헤시안 고유값: {np.round(ev, 2).tolist()} -> {kind}")

    if n_boot:
        for k in ("quad", "quad_diag"):
            A = features(Z, k)
            if A.shape[1] >= n:
                continue
            nv, sig = bootstrap_sig(A, y, names(k), n_boot, rng)
            msg = "유효 표본 부족" if sig is None else f"{len(sig)}/{A.shape[1]} {sig}"
            print(f"부트스트랩 [{k}] (유효 {nv}/{n_boot}) 95% CI가 0 미포함: {msg}")
    return rows


def main():
    """trials csv 여러 개를 각각(그리고 --pool이면 합쳐서) 분석하고 요약 CSV를 저장한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="trials csv 경로 (여러 개, 와일드카드 가능)")
    ap.add_argument("--pool", action="store_true", help="파일들을 합쳠서 한 번 더 분석 (중복 theta 제거)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--n-broad", type=int, default=30, help="1단계 시행 수 (holdout 분리 기준)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="요약 CSV 저장 경로 (선택)")
    a = ap.parse_args()

    paths = []
    for c in a.csv:  # Windows cmd는 와일드카드를 확장하지 않으므로 직접 처리
        paths += sorted(Path().glob(c)) if any(ch in c for ch in "*?[") else [Path(c)]
    rows, frames = [], []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
        rows += analyze(str(p), df, a.boot, a.n_broad, a.seed)
    if a.pool and len(frames) > 1:
        pooled = pd.concat(frames, ignore_index=True)
        pooled = pooled.loc[~pooled[PARAMS].round(3).duplicated()].reset_index(drop=True)
        rows += analyze(f"pooled({len(frames)} files)", pooled, a.boot, 0, a.seed)
    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
