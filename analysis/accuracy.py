"""분류정확도 (10/13 교수님 피드백: "중요한 건 분류정확도가 어떻게 되는가").

IoU 외에 혼동행렬 기반 지표를 낸다. 비교 대상:
  base        기준 θ0 (타일 묶음 통계 또는 --theta0)
  best_<cls>  per_class_rsm 표본 중 그 클래스 IoU가 가장 높았던 θ 하나로 전체 추론
  fusion      클래스별 최고 θ 예측을 논문 방식(IoU 개선폭 큰 클래스가 나중에 덮어씀)으로 합친 결과
지표 (모두 %, 정답이 있는 픽셀만):
  OA     전체 정확도 = 맞힌 픽셀 / 전체 픽셀
  kappa  Cohen's kappa (우연 일치 보정)
  mIoU   정답에 있는 클래스 IoU 평균
  클래스별 PA(생산자 정확도 = 재현율), UA(사용자 정확도 = 정밀도), F1, IoU

사용 (E1):
  python analysis/accuracy.py --img-dir <img> --msk-dir <msk> --img-glob "LC_*.tif" --msk-glob "LC_*.tif" \
      --checkpoint models/rgb15/FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth --n-classes 19 \
      --label-map labelmap_aihub_e1.json --samples analysis/results/per_class/e1_box/samples_all_classes.csv \
      --targets 5 6 0 4 17 --names coniferous deciduous building water greenhouse --out analysis/results/accuracy/e1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from search import NAMES  # noqa: E402


def metrics(cm):
    cm = cm.astype(float)
    n = cm.sum()
    tp = np.diag(cm)
    gt = cm.sum(1)          # 행 = 정답
    pr = cm.sum(0)          # 열 = 예측
    oa = tp.sum() / n * 100
    pe = (gt * pr).sum() / n ** 2
    kappa = ((tp.sum() / n) - pe) / (1 - pe) * 100
    with np.errstate(invalid="ignore", divide="ignore"):
        pa = np.where(gt > 0, tp / gt, np.nan) * 100
        ua = np.where(pr > 0, tp / pr, np.nan) * 100
        f1 = 2 * pa * ua / (pa + ua)
        iou = np.where(gt + pr - tp > 0, tp / (gt + pr - tp), np.nan) * 100
    miou = np.nanmean(np.where(gt > 0, iou, np.nan))
    return dict(OA=oa, kappa=kappa, mIoU=miou), dict(PA=pa, UA=ua, F1=f1, IoU=iou, gt_px=gt)


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
    ap.add_argument("--class-names", nargs="*", help="출력 인덱스 순 클래스 이름 (표 표시용)")
    ap.add_argument("--samples", required=True, help="per_class_rsm 의 samples_all_classes.csv")
    ap.add_argument("--targets", type=int, nargs="+", required=True)
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--theta0", type=float, nargs=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from data import (Evaluator, build_lut, channel_stats, confusion, decision_fusion, load_model, load_tiles,
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
    theta0 = np.array(a.theta0, float) if a.theta0 else channel_stats(imgs)
    if a.device == "auto":
        import torch
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    ev = Evaluator(load_model(a.checkpoint, a.encoder, a.n_classes, a.arch), imgs, a.n_classes,
                   a.batch, a.device, a.threads)
    print(f"[data] tiles={len(imgs)} device={a.device}")

    S = pd.read_csv(a.samples)
    X = S[NAMES].to_numpy(float)
    best = {c: X[int(np.nanargmax(S[f"iou_{c}"].to_numpy(float)))] for c in a.targets}

    preds = {"base": ev.predict(theta0)}
    for c in a.targets:
        preds[f"best_{cname[c]}"] = ev.predict(best[c])
    cm_base = confusion(preds["base"], msks, a.n_classes)
    _, pc0 = metrics(cm_base)
    delta = {c: pc0["IoU"][c] for c in a.targets}
    gain = {c: metrics(confusion(preds[f"best_{cname[c]}"], msks, a.n_classes))[1]["IoU"][c] - delta[c]
            for c in a.targets}
    order = sorted(a.targets, key=lambda c: gain[c])
    preds["fusion"] = decision_fusion(preds["base"], [(c, preds[f"best_{cname[c]}"]) for c in order])

    lab = a.class_names or [str(i) for i in range(a.n_classes)]
    rows, per = [], []
    for k, p in preds.items():
        cm = confusion(p, msks, a.n_classes)
        np.savetxt(out / f"confusion_{k}.csv", cm, fmt="%d", delimiter=",")
        g, pc = metrics(cm)
        rows.append(dict(setting=k, **{m: round(v, 2) for m, v in g.items()}))
        for i in range(a.n_classes):
            if pc["gt_px"][i] == 0 and np.isnan(pc["UA"][i]):
                continue
            per.append(dict(setting=k, cls=i, name=lab[i], gt_px=int(pc["gt_px"][i]),
                            **{m: round(float(pc[m][i]), 2) for m in ("PA", "UA", "F1", "IoU")}))
    pd.DataFrame(rows).to_csv(out / "overall.csv", index=False)
    pd.DataFrame(per).to_csv(out / "per_class.csv", index=False)
    json.dump(dict(theta0=theta0.tolist(), best_theta={cname[c]: best[c].tolist() for c in a.targets},
                   fusion_order_low_to_high=[cname[c] for c in order]),
              open(out / "settings.json", "w"), indent=2, ensure_ascii=False)
    print(pd.DataFrame(rows).to_string(index=False))
    t = pd.DataFrame(per)
    print(t[t.cls.isin(a.targets)].to_string(index=False))


if __name__ == "__main__":
    main()
