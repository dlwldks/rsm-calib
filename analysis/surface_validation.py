"""반응표면 2차식 타당성 점검 (10/12 README 4장 P1, 5장 4번). 추론 없음.

같은 (θ, IoU) 기록에 식 4종을 맞추고, 관측 수(앞에서부터 n개)별로 비교한다.
    linear      1 + z_i                       계수 7
    quad_diag   linear + z_i^2                계수 13
    quad        논문 식 1 (교차항 포함)        계수 28
    ridge_quad  quad + Ridge (λ는 LOO 최소)    계수 28
지표
    loo        표본 하나를 빼고 맞춘 식으로 그 표본을 예측한 RMSE (평균값 예측 = mean_only)
    future     앞 n개로 맞춘 식으로 n+1번째 이후 표본을 예측한 RMSE (탐색에서 "다음 후보"를 맞히는가)
    kind       quad·ridge_quad 정상점 종류 (헤시안 고유값: max / min / saddle)

사용:
    python analysis/surface_validation.py --out analysis/results/efficiency_20261012/surface_validation.csv
"""
import argparse
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]
LAMS = np.logspace(-3, 2, 26)

# (이름, 파일, IoU 열, 설명)
SOURCES = [
    ("E1 RSM seed0 (침엽수)", "analysis/results/e1/search_adj0.1_n125/trials_c5.csv", "iou", "논문 절차 125회"),
    ("E1 무작위 ±10% (침엽수)", "analysis/results/per_class/e1_box/samples_all_classes.csv", "iou_5", "LHS 75개"),
    ("E1 무작위 ±10% (활엽수)", "analysis/results/per_class/e1_box/samples_all_classes.csv", "iou_6", "LHS 75개"),
    ("E1 무작위 ±10% (건물)", "analysis/results/per_class/e1_box/samples_all_classes.csv", "iou_0", "LHS 75개"),
    ("E1 무작위 ±10% (비닐하우스)", "analysis/results/per_class/e1_box/samples_all_classes.csv", "iou_17", "LHS 75개"),
    ("E4 RSM A seed0 (침엽수)", "out_e4_A_seed0/trials_c5.csv", "iou", "논문 절차 65회"),
    ("E4 RSM B seed0 (침엽수)", "out_e4_B_seed0/trials_c5.csv", "iou", "boxmax 65회"),
    ("E5 RSM seed0 (침엽수)", "analysis/results/e5/search_A_seed0_n125/trials_c5.csv", "iou", "논문 절차 125회"),
    ("E2 RSM ±50% (침엽수)", "out_e2_test50_n125/trials_c5.csv", "iou", "논문 절차 125회"),
]


def feats(Z, kind):
    n, d = Z.shape
    cols = [np.ones(n)] + [Z[:, i] for i in range(d)]
    if kind == "quad_diag":
        cols += [Z[:, i] ** 2 for i in range(d)]
    elif kind in ("quad", "ridge_quad"):
        cols += [Z[:, i] * Z[:, j] for i, j in combinations_with_replacement(range(d), 2)]
    return np.column_stack(cols)


def solve(A, y, lam):
    D = np.eye(A.shape[1]); D[0, 0] = 0
    M = np.linalg.pinv(A.T @ A + lam * D)
    return M @ A.T @ y, np.einsum("ij,jk,ik->i", A, M, A)


def fit(X, y, kind):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    A = feats((X - mu) / sd, kind)
    if A.shape[1] >= len(y):
        return None
    lams = LAMS if kind == "ridge_quad" else [0.0]
    best = None
    for lam in lams:
        b, h = solve(A, y, lam)
        r = y - A @ b
        if np.any(h >= 1 - 1e-8):
            continue
        loo = float(np.sqrt(np.mean((r / (1 - h)) ** 2)))
        if best is None or loo < best[0]:
            best = (loo, b, lam)
    if best is None:
        return None
    loo, b, lam = best
    return dict(loo=loo, beta=b, lam=lam, mu=mu, sd=sd)


def predict(m, X, kind):
    return feats((X - m["mu"]) / m["sd"], kind) @ m["beta"]


def kind_of(m, d=6):
    b = m["beta"]; H = np.zeros((d, d))
    for k, (i, j) in enumerate(combinations_with_replacement(range(d), 2)):
        c = b[1 + d + k]
        if i == j:
            H[i, i] = 2 * c
        else:
            H[i, j] = H[j, i] = c
    e = np.linalg.eigvalsh(H)
    return "max" if np.all(e < 0) else "min" if np.all(e > 0) else "saddle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[35, 50, 65, 75, 100, 125])
    ap.add_argument("--out", default="analysis/results/efficiency_20261012/surface_validation.csv")
    a = ap.parse_args()
    rows = []
    for name, path, col, desc in SOURCES:
        f = ROOT / path
        if not f.exists():
            print("없음:", path); continue
        d = pd.read_csv(f)
        X, y = d[P].to_numpy(float), d[col].to_numpy(float)
        for n in a.ns:
            if n > len(y):
                continue
            Xn, yn = X[:n], y[:n]
            mo = float(np.sqrt(np.mean(((yn - yn.mean()) * n / (n - 1)) ** 2)))
            row = dict(source=name, desc=desc, n=n, n_total=len(y), mean_only_loo=round(mo, 2))
            for kind in ["linear", "quad_diag", "quad", "ridge_quad"]:
                m = fit(Xn, yn, kind)
                if m is None:
                    row[f"{kind}_loo"] = np.nan
                    continue
                row[f"{kind}_loo"] = round(m["loo"], 2)
                if n < len(y):
                    pr = predict(m, X[n:], kind)
                    row[f"{kind}_future"] = round(float(np.sqrt(np.mean((pr - y[n:]) ** 2))), 2)
                if kind in ("quad", "ridge_quad"):
                    row[f"{kind}_kind"] = kind_of(m)
                if kind == "ridge_quad":
                    row["ridge_lambda"] = m["lam"]
            if n < len(y):
                row["mean_only_future"] = round(float(np.sqrt(np.mean((yn.mean() - y[n:]) ** 2))), 2)
            loos = {k: row.get(f"{k}_loo", np.nan) for k in ["linear", "quad_diag", "quad", "ridge_quad"]}
            loos["mean_only"] = mo
            row["best_by_loo"] = min((v, k) for k, v in loos.items() if v == v)[1]
            rows.append(row)
    df = pd.DataFrame(rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    cols = ["source", "n", "mean_only_loo", "linear_loo", "quad_diag_loo", "quad_loo", "ridge_quad_loo",
            "best_by_loo", "quad_kind", "ridge_quad_kind", "mean_only_future", "quad_future", "ridge_quad_future"]
    print(df[[c for c in cols if c in df]].to_string(index=False))
    print("저장:", out)


if __name__ == "__main__":
    main()
