"""solver_compare.py가 낸 후보 theta(정상점·탐색 범위 내 최대점)를 실제로 추론해 IoU 확인.

사용:
    python analysis/reinfer_candidates.py --img-dir <DIR> --msk-dir <DIR> --checkpoint <CKPT> ^
        --theta-csv analysis/results/solver_theta.csv --out analysis/results/reinfer_iou.csv

조건은 E2와 같음: test 50타일(--include test), 19채널 라벨맵, 대상 클래스 5(침엽수), 타일 픽셀 합산 IoU.
먼저 기준(theta0)과 확인용 theta를 돌려 기존 결과와 같은지 본 뒤 후보를 돌린다.
물리적 범위(평균 0~255, 표준편차 1~255)를 벗어난 후보는 추론하지 않는다.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import Evaluator, build_lut, channel_stats, class_iou, load_model, load_tiles, pair_files  # noqa: E402

PARAMS = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]


def physical(th):
    return np.all(th[:3] >= 0) and np.all(th[:3] <= 255) and np.all(th[3:] >= 1) and np.all(th[3:] <= 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--msk-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label-map", default="labelmap_19.json")
    ap.add_argument("--n-classes", type=int, default=19)
    ap.add_argument("--target", type=int, default=5)
    ap.add_argument("--include", default="test")
    ap.add_argument("--theta-csv", default="analysis/results/solver_theta.csv")
    ap.add_argument("--check", nargs="*", default=[],
                    help="확인용 trials csv:행번호 (예: out_e2_test50_adj10/trials_c5.csv:9)")
    ap.add_argument("--out", default="analysis/results/reinfer_iou.csv")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()

    pairs = [p for p in pair_files(a.img_dir, a.msk_dir) if a.include in str(p[1].parent)]
    imgs, msks = load_tiles(pairs, build_lut(a.label_map, a.n_classes), (0, 1, 2))
    print(f"[data] tiles={len(imgs)}")
    ev = Evaluator(load_model(a.checkpoint, "resnet34", a.n_classes, "Unet"), imgs, a.n_classes,
                   threads=a.threads)

    def iou(th):
        t = time.time()
        v = float(class_iou(ev.predict(np.asarray(th, float)), msks, a.n_classes)[a.target])
        return v, time.time() - t

    rows = []
    th0 = channel_stats(imgs)
    v, s = iou(th0)
    print(f"[base] theta0={np.round(th0, 2).tolist()} IoU={v:.2f} ({s:.0f}s)")
    rows.append(dict(data="-", method="theta0", point="base", pred_iou=np.nan, iou=v, sec=s,
                     **dict(zip(PARAMS, th0))))
    for spec in a.check:
        path, i = spec.rsplit(":", 1)
        r = pd.read_csv(path).iloc[int(i)]
        v, s = iou(r[PARAMS].to_numpy(float))
        print(f"[check] {spec}: 기록 {r['iou']:.2f} / 재추론 {v:.2f}")
        rows.append(dict(data=path, method="check", point=f"row{i}", pred_iou=r["iou"], iou=v, sec=s,
                         **{p: r[p] for p in PARAMS}))

    cand = pd.read_csv(a.theta_csv)
    for _, r in cand.iterrows():
        th = r[PARAMS].to_numpy(float)
        if not physical(th):
            print(f"[skip] {r['data']} {r['method']} {r['point']}: 물리적 범위 밖")
            rows.append(dict(r, iou=np.nan, sec=np.nan, note="물리적 범위 밖"))
            continue
        v, s = iou(th)
        print(f"[cand] {r['data']:>10} {r['method']:>8} {r['point']:>10}: 예측 {r['pred_iou']:8.1f} / 실제 {v:6.2f} ({s:.0f}s)")
        rows.append(dict(r, iou=v, sec=s))
    pd.DataFrame(rows).round(4).to_csv(a.out, index=False)
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
