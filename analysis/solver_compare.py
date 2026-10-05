"""2차 반응면(논문 식 1) 계수 추정 방법 비교 (trials csv만 사용, 추론 불필요).

사용:
    python analysis/solver_compare.py out_e2_test50_adj10/trials_c5.csv out_e2_test50_adj0.25/trials_c5.csv ^
        out_e2_test50/trials_c5.csv --labels 10% 25% 50% --out solver_summary.csv --theta-out solver_theta.csv

같은 (theta, IoU) 기록에 계수 28개짜리 2차식을 방법만 바꿔서 맞추고 비교한다.
입력은 rsm.py와 같게 컬럼별 표준화 z = (x - mean) / std.

방법:
    ols      최소제곱 (현재 rsm.py 방식, 기준)
    ridge    Ridge. 상수항 빼고 계수 크기에 벌점 lambda*|b|^2. lambda는 LOO 오차가 가장 작은 값을 격자에서 고름
    wls      가중 최소제곱. 가중치 w = IoU / max(IoU), 최소 0.05 -> IoU 높은 표본(최적점 근처)을 더 정확히 맞춤
    huber    Huber 회귀 (epsilon 1.35). 잔차가 큰 표본은 제곱 대신 절댓값으로 덜 반영
    exclude  IoU < 5 표본(무너진 표본)을 빼고 최소제곱. 남은 표본이 29개 미만이면 2차식을 풀 수 없음

지표:
    R2, Adj.R2          학습 데이터 적합도 (계수가 많을수록 유리하므로 참고만)
    LOO RMSE (전체)     표본 하나를 빼고 맞춘 식으로 그 표본을 예측한 오차, 전체 표본 기준
    LOO RMSE (IoU>=5)   같은 오차를 무너지지 않은 표본에서만 계산 (최적점 근처 예측력)
    정상점              기울기 = 0 인 점 (논문 식 3). 헤시안 고유값으로 max/min/saddle
    상자 내 최대점      탐색한 범위(각 변수의 최소~최대) 안에서 식이 가장 높은 점
    ratio               점 / theta0 (theta0 = 탐색 시작점)
정상점·상자 내 최대점의 실제 IoU는 추론이 필요하므로 --theta-out 파일의 값을 run.py apply --theta로 돌려 확인한다.

보고서 3.2·3.4·3.5 (계수 추정 방법 비교)
"""
import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.linear_model import HuberRegressor

PARAMS = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]
METHODS = ["ols", "ridge", "wls", "huber", "exclude"]
COLLAPSE = 5.0                      # 이 값 미만 IoU = 무너진 표본
RIDGE_GRID = np.logspace(-3, 2, 26)  # 0.001 ~ 100


def quad_features(Z):
    """[1, z_i, z_i*z_j (i<=j)] — rsm.py와 같은 순서, 계수 28개."""
    n, d = Z.shape
    cols = [np.ones(n)] + [Z[:, i] for i in range(d)]
    cols += [Z[:, i] * Z[:, j] for i, j in combinations_with_replacement(range(d), 2)]
    return np.column_stack(cols)


def grad_hess(beta, d=6):
    """f = b0 + b^T z + 1/2 z^T H z 형태의 b, H (rsm.py와 같음)."""
    b = beta[1:1 + d]
    H = np.zeros((d, d))
    for k, (i, j) in enumerate(combinations_with_replacement(range(d), 2)):
        c = beta[1 + d + k]
        H[i, j] = H[j, i] = 2 * c if i == j else c
    return b, H


# ------------------------------------------------------------------ 추정 방법
def fit_linear(A, y, w=None, lam=0.0):
    """(가중) 최소제곱 / Ridge 닫힌 해. 반환: 계수, hat 행렬 대각(LOO 계산용)."""
    w = np.ones(len(y)) if w is None else w
    D = np.eye(A.shape[1])
    D[0, 0] = 0.0                                   # 상수항은 벌점 없음
    M = A.T @ (A * w[:, None]) + lam * D
    Minv = np.linalg.pinv(M)
    beta = Minv @ (A.T @ (w * y))
    h = np.einsum("ij,jk,ik->i", A, Minv, A * w[:, None])
    return beta, h


def loo_linear(A, y, w=None, lam=0.0):
    """선형 평활기의 LOO 예측: y_i - r_i / (1 - h_i)."""
    beta, h = fit_linear(A, y, w, lam)
    r = y - A @ beta
    ok = h < 1 - 1e-8
    pred = np.full(len(y), np.nan)
    pred[ok] = y[ok] - r[ok] / (1 - h[ok])
    return beta, pred


def fit_huber(A, y):
    """Huber 회귀 계수 (ε 1.35, 정규화 없음). 설계행렬에 상수열이 있으므로 절편은 따로 맞추지 않는다."""
    m = HuberRegressor(epsilon=1.35, alpha=0.0, fit_intercept=False, max_iter=5000)
    m.fit(A, y)
    return m.coef_


def loo_explicit(A, y, fit, keep=None):
    """한 표본씩 빼고 다시 맞춰서 예측. keep: 학습에 쓸 수 있는 표본 마스크."""
    n = len(y)
    keep = np.ones(n, bool) if keep is None else keep
    pred = np.full(n, np.nan)
    for i in range(n):
        m = keep.copy()
        m[i] = False
        if m.sum() <= A.shape[1]:
            continue
        pred[i] = A[i] @ fit(A[m], y[m])
    return pred


def estimate(method, A, y):
    """반환: (계수 또는 None, LOO 예측, 메모)."""
    n, p = A.shape
    if method == "ols":
        beta, pred = loo_linear(A, y)
        return beta, pred, ""
    if method == "wls":
        w = np.maximum(y / y.max(), 0.05)
        beta, pred = loo_linear(A, y, w)
        return beta, pred, "w = IoU/max, 최소 0.05"
    if method == "ridge":
        best = None
        for lam in RIDGE_GRID:
            beta, pred = loo_linear(A, y, lam=lam)
            e = np.sqrt(np.nanmean((y - pred) ** 2))
            if best is None or e < best[0]:
                best = (e, lam, beta, pred)
        return best[2], best[3], f"lambda = {best[1]:.3g}"
    if method == "huber":
        beta = fit_huber(A, y)
        pred = loo_explicit(A, y, fit_huber)
        return beta, pred, "epsilon 1.35"
    if method == "exclude":
        keep = y >= COLLAPSE
        if keep.sum() <= p:
            return None, np.full(n, np.nan), f"남은 표본 {keep.sum()}개 <= 계수 {p}개, 풀 수 없음"
        ols_fit = lambda Ak, yk: np.linalg.lstsq(Ak, yk, rcond=None)[0]
        beta = ols_fit(A[keep], y[keep])
        pred = loo_explicit(A, y, ols_fit, keep)
        pred[~keep] = A[~keep] @ beta               # 학습에 안 쓴 표본은 전체 식으로 예측
        return beta, pred, f"표본 {keep.sum()}개 사용"
    raise ValueError(method)


# ------------------------------------------------------------------ 분석
def box_max(beta, zlo, zhi):
    """탐색 범위(z 상자) 안에서 2차식 최대점."""
    b, H = grad_hess(beta)
    f = lambda z: -(beta[0] + b @ z + 0.5 * z @ H @ z)
    g = lambda z: -(b + H @ z)
    starts = [np.zeros(len(b)), zlo, zhi, np.clip(np.linalg.lstsq(H, -b, rcond=None)[0], zlo, zhi)]
    best = min((optimize.minimize(f, s, jac=g, bounds=list(zip(zlo, zhi)), method="L-BFGS-B")
                for s in starts), key=lambda r: r.fun)
    return best.x, -best.fun


def theta0_of(csv_path, X):
    """탐색 시작점 θ₀: 같은 폴더 result.json의 theta_base, 없으면 기록 첫 행."""
    rj = Path(csv_path).parent / "result.json"
    if rj.exists():
        r = json.load(open(rj, encoding="utf-8"))
        for c in r.get("classes", {}).values():
            return np.array(c["theta_base"], float)
    return X[0]


def analyze(label, path):
    """한 탐색 기록에 5가지 방법으로 계수를 맞추고, 방법별 적합도·LOO·정상점·상자 내 최대점을 계산한다."""
    df = pd.read_csv(path)
    X, y = df[PARAMS].to_numpy(float), df["iou"].to_numpy(float)
    n = len(y)
    center, scale = X.mean(0), X.std(0)
    scale[scale == 0] = 1.0
    Z = (X - center) / scale
    A = quad_features(Z)
    th0 = theta0_of(path, X)
    zlo, zhi = Z.min(0), Z.max(0)
    hi_mask = y >= COLLAPSE
    tss = float(((y - y.mean()) ** 2).sum())

    print(f"\n## {label}  ({path}, 표본 {n}개, IoU {y.min():.2f}~{y.max():.2f}, "
          f"IoU<{COLLAPSE:g} {int((~hi_mask).sum())}개)\n")
    print("| 방법 | R2 | Adj.R2 | LOO RMSE 전체 | LOO RMSE IoU>=5 | 정상점 | 정상점 예측 IoU | "
          "정상점 ratio 범위 | 상자 내 최대 예측 IoU | 메모 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows, thetas = [], []
    for m in METHODS:
        beta, pred, note = estimate(m, A, y)
        if beta is None:
            print(f"| {m} | | | | | | | | | {note} |")
            rows.append(dict(data=label, method=m, note=note))
            continue
        fitted = A @ beta
        if m == "exclude":
            yk, fk = y[hi_mask], fitted[hi_mask]
            r2 = 1 - float(((yk - fk) ** 2).sum()) / float(((yk - yk.mean()) ** 2).sum())
            nn = int(hi_mask.sum())
        else:
            r2, nn = 1 - float(((y - fitted) ** 2).sum()) / tss, n
        adj = 1 - (1 - r2) * (nn - 1) / (nn - A.shape[1]) if nn > A.shape[1] else np.nan
        loo_all = float(np.sqrt(np.nanmean((y - pred) ** 2)))
        loo_hi = float(np.sqrt(np.nanmean((y[hi_mask] - pred[hi_mask]) ** 2)))

        b, H = grad_hess(beta)
        zs = np.linalg.lstsq(H, -b, rcond=None)[0]
        ev = np.linalg.eigvalsh(H)
        kind = "max" if np.all(ev < 0) else "min" if np.all(ev > 0) else "saddle"
        th_s = zs * scale + center
        pred_s = float(beta[0] + b @ zs + 0.5 * zs @ H @ zs)
        rs = th_s / th0
        zb, pred_b = box_max(beta, zlo, zhi)
        th_b = zb * scale + center

        print(f"| {m} | {r2:.3f} | {adj:.3f} | {loo_all:.2f} | {loo_hi:.2f} | {kind} | {pred_s:.1f} | "
              f"{rs.min():.2f}~{rs.max():.2f} | {pred_b:.1f} | {note} |")
        rows.append(dict(data=label, method=m, n_fit=nn, r2=r2, adj_r2=adj, loo_rmse_all=loo_all,
                         loo_rmse_hi=loo_hi, stationary=kind, pred_iou_stationary=pred_s,
                         ratio_min=rs.min(), ratio_max=rs.max(), pred_iou_boxmax=pred_b, note=note))
        for kind_pt, th, pr in (("stationary", th_s, pred_s), ("boxmax", th_b, pred_b)):
            thetas.append(dict(data=label, method=m, point=kind_pt, pred_iou=pr,
                               **{p: v for p, v in zip(PARAMS, th)}))
    base_loo = float(np.sqrt(np.mean(((y - y.mean()) * n / (n - 1)) ** 2)))
    print(f"| 평균값만 | | | {base_loo:.2f} | | | | | | 비교 기준 |")
    print(f"\n탐색 최고 IoU {y.max():.2f} (표본 {int(np.argmax(y))}번, 회차 {int(df['iter'].iloc[np.argmax(y)])})")
    return rows, thetas


def main():
    """trials csv 여러 개를 분석해 요약(--out)과 후보 θ 목록(--theta-out)을 저장한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--theta-out")
    a = ap.parse_args()
    labels = a.labels or a.csv
    rows, thetas = [], []
    for lab, p in zip(labels, a.csv):
        r, t = analyze(lab, p)
        rows += r
        thetas += t
    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"\n저장: {a.out}")
    if a.theta_out:
        pd.DataFrame(thetas).round(4).to_csv(a.theta_out, index=False)
        print(f"저장: {a.theta_out}")


if __name__ == "__main__":
    main()
