"""기존 탐색 기록 정리 (10/12 README 5장 1번: legacy audit).

레포에 있는 탐색 결과(result.json + trials_c*.csv)를 한 표로 모아
실험·타일·θ0·탐색 설정·seed·실제 추론 수·1단계 최고·전체 최고·2차식 회차 기여를 비교한다.
추론 없음. 출력: analysis/results/efficiency_20261012/legacy_audit.csv

사용: python analysis/legacy_audit.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/results/efficiency_20261012/legacy_audit.csv"


def experiment_of(path, args):
    t = str(args.get("tiles") or "")
    s = str(path)
    if "e1" in s or "labelmap_aihub" in str(args.get("label_map")):
        return "E1"
    if "e4" in t or "e4" in s:
        return "E4"
    if "e5" in t or "e5" in s:
        return "E5"
    return "E2"


def method_of(st, args):
    if args.get("steepest"):
        return "steepest"
    m = "RSM(논문 Figure 1)"
    extra = []
    if st.get("fit") == "ridge":
        extra.append("ridge")
    if st.get("move") == "boxmax":
        extra.append("boxmax")
    if st.get("r2_min"):
        extra.append(f"R2>={st['r2_min']}")
    return m + (" + " + ", ".join(extra) if extra else "")


def summarize_trials(df):
    y = df["iou"].to_numpy(float)
    it = df["iter"].to_numpy(int) if "iter" in df else np.zeros(len(y), int)
    s1 = y[it == 0]
    later = y[it > 0]
    i = int(np.argmax(y))
    return dict(n_evals=len(y), base=y[0], stage1_n=len(s1), stage1_best=s1.max() if len(s1) else np.nan,
                best=y.max(), best_index=i + 1, best_iter=int(it[i]),
                later_best=later.max() if len(later) else np.nan,
                quad_gain=(later.max() - s1.max()) if len(later) and len(s1) else np.nan,
                best_at_30=y[:30].max(), best_at_50=y[:50].max() if len(y) >= 50 else np.nan,
                best_at_75=y[:75].max() if len(y) >= 75 else np.nan)


def main():
    rows = []
    for rj in sorted(ROOT.rglob("result.json")):
        if ".git" in rj.parts:
            continue
        r = json.load(open(rj))
        if "classes" not in r:
            continue
        a = r.get("args", {})
        for c, cls in r["classes"].items():
            tr = rj.parent / f"trials_c{c}.csv"
            if not tr.exists():
                continue
            st = cls.get("settings", {})
            hist = cls.get("history", [])
            kinds = [h.get("stationary_kind") for h in hist]
            note = "" if st else "이전 코드(θ0 ±10% 상자 고정, ±3%)로 실행, 현재 논문 모드와 비교 불가"
            row = dict(note=note, run=str(rj.parent.relative_to(ROOT)).replace("\\", "/"),
                       experiment=experiment_of(rj.parent, a), target_class=c,
                       tiles=a.get("tiles") or a.get("include") or "",
                       theta0=a.get("base_mode"), method=method_of(st, a),
                       adj0=st.get("adj0"), adj_t=st.get("adj_t"), n_t=st.get("n_t"),
                       max_evals=st.get("max_evals"), eps_stop=st.get("eps_stop"), seed=st.get("seed"),
                       stop_reason=cls.get("stop_reason"),
                       n_rounds=len(hist), n_saddle=sum("saddle" in str(k) for k in kinds),
                       n_max=sum(str(k) in ("max", "stationary") for k in kinds),
                       final_r2=(cls.get("rsm") or {}).get("r2"))
            row.update(summarize_trials(pd.read_csv(tr)))
            rows.append(row)

    # 무작위(LHS) 기록: per_class_rsm 결과 (E1 ±10% 75개)
    for samp in sorted(ROOT.glob("analysis/results/per_class/*/samples_all_classes.csv")):
        if "box" not in samp.parent.name:
            continue
        d = pd.read_csv(samp).rename(columns={"iou_5": "iou"})
        d["iter"] = [0] * min(30, len(d)) + [1] * max(0, len(d) - 30)   # 31번째부터 = 추가 무작위 표본
        row = dict(note="quad_gain 자리 = 추가 무작위 45개가 처음 30개를 넘은 폭", run=str(samp.parent.relative_to(ROOT)).replace("\\", "/"), experiment="E1",
                   target_class="5", tiles="e1 32장", theta0="pooled",
                   method="무작위 LHS (처음 30개 = RSM seed0 1단계와 같음, 이후 seed1)",
                   adj0=0.1, max_evals=len(d), seed="0+1")
        row.update(summarize_trials(d))
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(2)
    df.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)
    print(df[["run", "experiment", "method", "adj0", "seed", "n_evals", "stage1_best", "best",
              "best_iter", "quad_gain", "n_rounds", "n_saddle"]].to_string(index=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
