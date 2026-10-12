"""클래스별 2차식 (교수님 10/6 피드백: 침엽수·활엽수·가로수 2차식을 따로 구하기).

한 번 추론하면 모든 클래스의 IoU가 함께 나온다. 그래서
  1) θ 표본마다 한 번 추론해서 전 클래스 IoU를 기록하고
  2) 대상 클래스마다 같은 표본으로 2차식(논문 식 1, 계수 28개)을 따로 맞추고
  3) 클래스마다 식이 가리킨 θ(정상점 / 표본 범위 안 최대점)를 실제로 추론해 확인하고
  4) 클래스별 최적 θ 예측을 겹쳐 놓았을 때 같은 픽셀을 두 클래스가 동시에 차지하는 수(겹침)와
     논문 방식(개선폭 큰 클래스 우선) 융합 결과를 계산한다.

θ 표본
  --trials  기존 탐색 기록(trials_c*.csv) 재사용. 여러 개 가능, 중복 θ는 한 번만 추론
  --n-new   θ0 ±adj 상자 안 LHS 표본 새로 추가 (첫 점은 θ0)

예 (E1: FLAIR 모델 + AI-HUB AP12 32장, 침엽수 5·활엽수 6):
  python analysis/per_class_rsm.py --img-dir <img> --msk-dir <msk> --img-glob "LC_*.tif" --msk-glob "LC_*.tif" ^
      --tiles docs/paper_tiles/e1_ap12_tiles32.txt --checkpoint models/rgb15/FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth ^
      --n-classes 19 --label-map labelmap_aihub_e1.json --targets 5 6 --names coniferous deciduous ^
      --trials analysis/results/e1/search_adj0.1_n125/trials_c5.csv --out analysis/results/per_class/e1

가로수는 FLAIR 모델에 클래스가 없어서 AI-HUB 모델(E3)로 돌려야 한다. 이때 --label-map 에
AI-HUB 코드 -> AI-HUB 모델 출력 인덱스 매핑을 넣고 --targets 에 가로수 인덱스를 준다.
"""
import argparse
import csv
import json
import sys
import time
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rsm import QuadraticRSM, n_min, quad_features  # noqa: E402
from search import NAMES, lhs, physical_clip  # noqa: E402


# ------------------------------------------------------------------ 식 적합도
def loo_rmse(X, y):
    """2차식(28계수) 최소제곱의 LOO RMSE (hat 행렬)."""
    sd = X.std(0)
    sd[sd == 0] = 1.0
    A = quad_features((X - X.mean(0)) / sd)
    if len(y) <= A.shape[1]:
        return np.nan
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ b
    h = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
    return float(np.sqrt(np.mean((r / (1 - h)) ** 2))) if np.all(h < 1 - 1e-8) else np.nan


def mean_only_loo(y):
    n = len(y)
    return float(np.sqrt(np.mean(((y - y.mean()) * n / (n - 1)) ** 2)))


def raw_equation(X, y):
    """원 단위(0~255) 변수로 맞춘 같은 2차식의 계수. 예측값은 표준화 좌표 피팅과 같다(발표용 식)."""
    A = quad_features(X)
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    names = ["1"] + NAMES + [f"{NAMES[i]}*{NAMES[j]}" if i != j else f"{NAMES[i]}^2"
                             for i, j in combinations_with_replacement(range(len(NAMES)), 2)]
    return dict(zip(names, b.tolist()))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--msk-dir", required=True)
    ap.add_argument("--img-glob", default="IMG_*.tif")
    ap.add_argument("--msk-glob", default="MSK_*.tif")
    ap.add_argument("--tiles")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--n-classes", type=int, default=19)
    ap.add_argument("--label-map")
    ap.add_argument("--targets", type=int, nargs="+", required=True)
    ap.add_argument("--names", nargs="+", help="--targets 순서대로 클래스 이름")
    ap.add_argument("--theta0", type=float, nargs=6, help="없으면 타일 전체 채널 통계(pooled)")
    ap.add_argument("--trials", nargs="*", default=[])
    ap.add_argument("--n-new", type=int, default=0)
    ap.add_argument("--adj", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--device", default="auto", help="auto = GPU 있으면 cuda, 없으면 cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from data import (Evaluator, build_lut, channel_stats, class_iou, load_model, load_tiles,
                      pair_files, tile_key)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.names or [str(c) for c in a.targets]
    cname = dict(zip(a.targets, names))

    pairs = pair_files(a.img_dir, a.msk_dir, a.img_glob, a.msk_glob)
    if a.tiles:
        keep = {ln.strip() for ln in open(a.tiles) if ln.strip()}
        pairs = [p for p in pairs if tile_key(p[1]) in keep]
    imgs, msks = load_tiles(pairs, build_lut(a.label_map, a.n_classes))
    print(f"[data] tiles={len(imgs)}")
    theta0 = np.array(a.theta0, float) if a.theta0 else channel_stats(imgs)
    print("[theta0]", np.round(theta0, 2).tolist())

    # θ 목록
    thetas = [theta0]
    for t in a.trials:
        thetas += [r for r in pd.read_csv(t)[NAMES].to_numpy(float)]
    rng = np.random.default_rng(a.seed)
    for u in lhs(a.n_new, 6, rng) if a.n_new else []:
        thetas.append(physical_clip(theta0 * (1 + a.adj * (2 * u - 1))))
    uniq = []
    for t in thetas:
        if not any(np.abs(t - s).max() <= 1e-4 for s in uniq):
            uniq.append(np.asarray(t, float))
    print(f"[theta] {len(uniq)}개 (중복 제거 전 {len(thetas)})")

    if a.device == "auto":
        import torch
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", a.device)
    ev = Evaluator(load_model(a.checkpoint, a.encoder, a.n_classes, a.arch), imgs, a.n_classes,
                   a.batch, a.device, a.threads)

    # 1) 표본마다 전 클래스 IoU (끊겨도 이어가기)
    prog = out / "samples_all_classes.csv"
    done = pd.read_csv(prog) if prog.exists() else None
    cols = NAMES + [f"iou_{c}" for c in range(a.n_classes)]
    if done is None:
        with open(prog, "w", newline="") as f:
            csv.writer(f).writerow(cols)
    for k, th in enumerate(uniq):
        if done is not None and len(done) and np.abs(done[NAMES].to_numpy() - th).max(1).min() <= 1e-4:
            continue
        t = time.time()
        iou = class_iou(ev.predict(th), msks, a.n_classes)
        with open(prog, "a", newline="") as f:
            csv.writer(f).writerow([repr(float(v)) for v in th] + [repr(float(v)) for v in iou])
        print(f"[{k + 1:3d}/{len(uniq)}] " + " ".join(f"{cname[c]}={iou[c]:.2f}" for c in a.targets)
              + f" ({time.time() - t:.0f}s)", flush=True)
    S = pd.read_csv(prog)
    X = S[NAMES].to_numpy(float)
    lo, hi = X.min(0), X.max(0)

    # 2) 클래스별 2차식
    summary, eqs, best_theta = [], {}, {}
    for c in a.targets:
        y = S[f"iou_{c}"].to_numpy(float)
        ok = ~np.isnan(y)
        Xc, yc = X[ok], y[ok]
        m = QuadraticRSM().fit(Xc, yc)
        xs, kind = m.stationary_point()
        xb, how = m.maximize(lo, hi)
        i = int(np.argmax(yc))
        best_theta[c] = Xc[i]
        eqs[cname[c]] = raw_equation(Xc, yc)
        summary.append(dict(cls=c, name=cname[c], n=len(yc), base=float(yc[0]), best_obs=float(yc[i]),
                            best_obs_idx=i, r2=m.r2, adj_r2=m.adj_r2, p_value=m.p_value,
                            loo=loo_rmse(Xc, yc), loo_mean_only=mean_only_loo(yc),
                            stationary=kind, stationary_theta=np.round(xs, 2).tolist(),
                            boxmax_theta=np.round(xb, 2).tolist(), boxmax_how=how,
                            boxmax_pred=float(m.predict(xb)[0]),
                            best_obs_theta=np.round(Xc[i], 2).tolist()))

    # 3) 식이 가리킨 θ(상자 안 최대점) 실제 추론
    for r in summary:
        th = np.array(r["boxmax_theta"], float)
        iou = class_iou(ev.predict(th), msks, a.n_classes)
        r["boxmax_actual"] = float(iou[r["cls"]])
        r["boxmax_actual_all"] = {cname[c]: float(iou[c]) for c in a.targets}
        print(f"[{r['name']}] 식 최대점 예측 {r['boxmax_pred']:.2f} / 실제 {r['boxmax_actual']:.2f}")

    # 4) 클래스별 최적 θ(표본 최고) 예측끼리 겹침 + 융합
    base = ev.predict(theta0)
    base_iou = class_iou(base, msks, a.n_classes)
    preds = {c: ev.predict(best_theta[c]) for c in a.targets}
    overlap = []
    for c1, c2 in combinations(a.targets, 2):
        m1, m2 = preds[c1] == c1, preds[c2] == c2
        both = int((m1 & m2).sum())
        gt = {cname[c]: int(((m1 & m2) & (msks == c)).sum()) for c in (c1, c2)}
        overlap.append(dict(pair=f"{cname[c1]}-{cname[c2]}", px_c1=int(m1.sum()), px_c2=int(m2.sum()),
                            overlap_px=both, overlap_pct_of_c1=100 * both / max(1, int(m1.sum())),
                            overlap_pct_of_c2=100 * both / max(1, int(m2.sum())), overlap_gt=gt))
    delta = {c: float(class_iou(preds[c], msks, a.n_classes)[c] - base_iou[c]) for c in a.targets}
    order = sorted(a.targets, key=lambda c: delta[c])          # 개선폭 작은 것부터 덮어씀 -> 큰 것이 이김
    fused = base.copy()
    for c in order:
        fused[preds[c] == c] = c
    fused_iou = class_iou(fused, msks, a.n_classes)
    fusion = dict(order_low_to_high=[cname[c] for c in order], delta={cname[c]: delta[c] for c in a.targets},
                  base={cname[c]: float(base_iou[c]) for c in a.targets},
                  own_theta={cname[c]: float(class_iou(preds[c], msks, a.n_classes)[c]) for c in a.targets},
                  fused={cname[c]: float(fused_iou[c]) for c in a.targets})

    pd.DataFrame(summary).to_csv(out / "per_class_summary.csv", index=False)
    pd.DataFrame(eqs).to_csv(out / "equations_raw_units.csv")
    json.dump(dict(theta0=theta0.tolist(), n_samples=len(S), summary=summary, overlap=overlap,
                   fusion=fusion), open(out / "per_class.json", "w"), indent=2, ensure_ascii=False,
              default=float)
    print(json.dumps(dict(overlap=overlap, fusion=fusion), indent=2, ensure_ascii=False, default=float))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
