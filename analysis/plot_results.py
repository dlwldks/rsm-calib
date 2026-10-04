"""계수 추정 방법 비교 그래프 3장 (analysis/results/*.csv -> analysis/results/*.png).

사용: python analysis/plot_results.py
    fig_loo_by_method.png      방법별 LOO 오차 (기록 4개), 평균값 예측 기준선
    fig_pred_vs_actual.png     후보 theta의 식 예측 IoU vs 실제 IoU
    fig_candidate_iou.png      후보 theta의 실제 IoU, 탐색 최고값 범위 표시
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # 범주색 고정 순서
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RECORDS = ["±10%", "±25%", "±50%", "±50% 125회"]
METHODS = ["ols", "ridge", "wls", "huber", "exclude"]
MNAME = {"ols": "최소제곱", "ridge": "Ridge", "wls": "가중 최소제곱", "huber": "Huber", "exclude": "IoU<5 제외"}
SEARCH_BEST = {"±10%": 29.34, "±25%": 29.10, "±50%": 29.50, "±50% 125회": 29.50}


def setup():
    for name in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic"):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "font.size": 10})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_loo():
    s = pd.read_csv(RES / "solver_summary.csv")
    base = {"±10%": 6.68, "±25%": 10.10, "±50%": 8.60, "±50% 125회": 7.91}   # 평균값만으로 예측한 LOO
    cap = 25
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w, gap = 0.15, 0.02
    for j, m in enumerate(METHODS):
        for i, r in enumerate(RECORDS):
            row = s[(s.data == r) & (s.method == m)]
            v = row.loo_rmse_all.iloc[0] if len(row) and pd.notna(row.loo_rmse_all.iloc[0]) else np.nan
            x = i + (j - 2) * (w + gap)
            if np.isnan(v):
                ax.text(x, 0.6, "풀 수\n없음", ha="center", va="bottom", fontsize=7.5, color=INK2)
                continue
            ax.bar(x, min(v, cap), w, color=SERIES[j], label=MNAME[m] if i == 0 else None, zorder=2)
            if v > cap:
                ax.text(x, cap + 0.3, f"{v:.0f}↑", ha="center", va="bottom", fontsize=8, color=INK2)
    for i, r in enumerate(RECORDS):
        ax.hlines(base[r], i - 0.45, i + 0.45, colors=INK, lw=1.2, ls=(0, (4, 3)), zorder=3,
                  label="평균값만으로 예측" if i == 0 else None)
    ax.set_xticks(range(len(RECORDS)), [f"{r}\n(표본 {125 if '125' in r else 35}개)" for r in RECORDS])
    ax.set_ylim(0, cap + 3)
    ax.set_ylabel("LOO RMSE (IoU %p, 낮을수록 좋음)")
    ax.set_title("방법별 LOO 오차: 35회 기록에서 평균값 예측보다 낮은 방법은 Ridge뿐", loc="left", fontsize=11.5)
    style(ax)
    ax.legend(ncol=6, fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.tight_layout()
    fig.savefig(RES / "fig_loo_by_method.png", dpi=160)


def fig_pred_actual():
    d = pd.read_csv(RES / "reinfer_iou.csv")
    d = d[~d.method.isin(["theta0", "check"]) & d.iou.notna()]
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, (pt, lab) in enumerate((("stationary", "정상점"), ("boxmax", "탐색 범위 내 최대점"))):
        q = d[d.point == pt]
        ax.scatter(q.pred_iou, q.iou, s=46, color=SERIES[k], edgecolor=SURF, lw=1.5, label=lab, zorder=3)
    diag = np.linspace(0, 40, 400)
    ax.plot(diag, diag, color=INK2, lw=1, ls=(0, (4, 3)), label="예측 = 실제")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(-1, 1500)
    ax.set_ylim(-1, 40)
    ax.set_xlabel("2차식이 예측한 IoU (%, 10 이상은 로그 눈금)")
    ax.set_ylabel("실제 IoU (%)")
    ax.set_title("탐색 범위 내 최대점 19개 중 17개는 예측 60 이상, 13개는 실제 3 이하", loc="left", fontsize=11.5)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(RES / "fig_pred_vs_actual.png", dpi=160)


def fig_candidates():
    d = pd.read_csv(RES / "reinfer_iou.csv")
    d = d[~d.method.isin(["theta0", "check"])]
    rows = [(m, p) for m in METHODS for p in ("stationary", "boxmax")]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.axvspan(29.10, 29.50, color="#d9d8d3", zorder=1)
    ax.text(29.3, len(rows) - 0.35, "탐색 최고\n29.1~29.5", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.axvline(13.34, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(13.34, len(rows) - 0.35, "기준 13.34", ha="center", va="bottom", fontsize=8, color=INK2)
    for k, r in enumerate(RECORDS):
        for i, (m, p) in enumerate(rows):
            q = d[(d.data == r) & (d.method == m) & (d.point == p)]
            if not len(q) or pd.isna(q.iou.iloc[0]):
                continue
            v = q.iou.iloc[0]
            y = len(rows) - 1 - i + (k - 1.5) * 0.17
            ax.scatter(v, y, s=40, color=SERIES[k], edgecolor=SURF, lw=1.2, zorder=3,
                       label=r if i == 0 or (m, p) == ("ols", "boxmax") and k == 1 else None)
            if v > 29.5:
                ax.annotate(f"{v:.2f}", (v, y), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=INK)
    labels = [f"{MNAME[m]} · {'정상점' if p == 'stationary' else '범위 내 최대'}" for m, p in rows]
    ax.set_yticks(range(len(rows)), labels[::-1])
    ax.set_xlim(-1, 35)
    ax.set_ylim(-0.7, len(rows) + 0.6)
    ax.set_xlabel("후보 θ의 실제 침엽수 IoU (%)")
    ax.set_title("탐색 최고값을 넘은 후보는 Ridge 2개 (30.80, 30.73)", loc="left", fontsize=11.5)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(r) for r in RECORDS if r in l]
    ax.legend([h[i] for i in order], [l[i] for i in order], title="탐색 기록", frameon=False, fontsize=8.5,
              title_fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "fig_candidate_iou.png", dpi=160)


def fig_order_by_n():
    """관측 수 n에 따른 1차·2차 LOO 오차 (order_by_n.csv, 탐색 순서대로 앞에서부터 n개)."""
    t = pd.read_csv(RES / "order_by_n.csv")
    t = t[t["sample"] == "앞에서부터"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for k, (col, lab) in enumerate((("linear", "1차 (계수 7)"), ("quad_diag", "2차 교차항 없음 (13)"),
                                    ("quad", "2차 논문 식 1 (28)"))):
        ax.plot(t.n, t[col], color=SERIES[k], lw=2, marker="o", ms=5, label=lab, zorder=3)
    ax.plot(t.n, t["mean"], color=INK2, lw=1.2, ls=(0, (4, 3)), label="평균값만으로 예측")
    ax.axvspan(40, 50, color="#e4e3df", zorder=1)
    ax.text(45, 17.2, "논문 본문\n40~50회", ha="center", va="top", fontsize=8, color=INK2)
    ax.set_ylim(5, 18.5)
    ax.set_xlabel("회귀에 쓴 관측 수 n (±50% 탐색 기록, 탐색 순서대로 앞에서부터)")
    ax.set_ylabel("LOO RMSE (IoU %p)")
    ax.set_title("논문 식 1은 관측 70개부터 오차가 가장 작다", loc="left", fontsize=11.5)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(RES / "fig_order_by_n.png", dpi=160)


if __name__ == "__main__":
    setup()
    fig_loo()
    fig_pred_actual()
    fig_candidates()
    fig_order_by_n()
    print("저장:", *sorted(p.name for p in RES.glob("fig_*.png")))
