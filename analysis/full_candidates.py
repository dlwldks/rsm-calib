"""toy 데이터에서 찾은 후보 theta를 전체 테스트셋(표본)에 적용해 IoU 비교 (이슈 #10).

사용 (데스크톱, C:\\rsm_calib 에서, PowerShell 한 줄):
    python analysis/full_candidates.py --spec out_full_test/spec_tiles.csv --checkpoint models/rgb15/FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth --label-map labelmap_19.json --per-domain 150 --out out_full_candidates --resume

입력
  --spec        full_eval.py가 만든 spec_tiles.csv (타일 경로·도메인·채널 합계). 여기서 test 타일만 사용
  --candidates  후보 theta 목록 csv (name, source, toy_iou, R_mean..B_std). 기본 analysis/results/full_candidates.csv
  --per-domain  도메인마다 뽑을 타일 수 (seed 고정). 0이면 test 전체 (15,700타일, 후보 1개당 약 2.5시간)

비교 기준으로 full_theta0(전체 test 통계 한 세트, 10/1 결과의 pooled와 같은 값)를 후보 맨 앞에 넣는다.
모든 후보를 같은 타일에서 계산한다. 타일을 한 번 읽고 후보마다 U-Net을 한 번씩 돌린다.

출력 (--out 폴더)
  sample_tiles.csv     사용한 타일 목록
  progress.npz         후보별 혼동행렬 누적 (중간 저장, --resume으로 이어서)
  candidates_iou.csv   후보별 클래스 IoU, 침엽수 IoU, mIoU, 걸린 시간
  domain_iou.csv       후보 × 도메인별 대상 클래스 IoU, 대상 픽셀 수, mIoU

보고서 4.2 (toy에서 찾은 θ를 전체 테스트셋에 적용)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import _read_tif, build_lut, confusion, load_model  # noqa: E402

PARAMS = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]
NAMES = ["building", "pervious surface", "impervious surface", "bare soil", "water", "coniferous",
         "deciduous", "brushwood", "vineyard", "herbaceous vegetation", "agricultural land",
         "plowed land", "swimming pool", "snow", "clear cut", "mixed", "ligneous", "greenhouse",
         "other"]


def iou_from_cm(cm):
    """혼동행렬 -> 클래스별 IoU(%) = TP / (TP + FP + FN) × 100."""
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1) - tp
    return np.where(den > 0, tp / np.maximum(den, 1), np.nan) * 100


def pooled_theta(df):
    """spec_tiles.csv의 채널 합계로 타일 전체 픽셀 기준 평균·표준편차(θ₀)를 계산한다."""
    n = df["n_px"].astype(float).sum()
    s = np.array([df[f"sum_{c}"].astype(float).sum() for c in "RGB"])
    ss = np.array([df[f"sq_{c}"].astype(float).sum() for c in "RGB"])
    mean = s / n
    return np.concatenate([mean, np.sqrt(ss / n - mean ** 2)])


def main():
    """test 표본 타일을 한 장씩 읽어 후보 θ마다 추론하고, 후보별·도메인별 혼동행렬을 누적해 IoU를 저장한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="out_full_test/spec_tiles.csv")
    ap.add_argument("--candidates", default="analysis/results/full_candidates.csv")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label-map", default="labelmap_19.json")
    ap.add_argument("--n-classes", type=int, default=19)
    ap.add_argument("--target", type=int, default=5, help="대상 클래스 (모델 인덱스, 5 = 침엽수)")
    ap.add_argument("--bands", type=int, nargs=3, default=[0, 1, 2])
    ap.add_argument("--per-domain", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--out", default="out_full_candidates")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = pd.read_csv(a.spec)
    test = spec[spec["split"] == "test"].reset_index(drop=True)
    th_full = pooled_theta(test)
    if a.per_domain > 0:
        parts = [g.sample(min(a.per_domain, len(g)), random_state=a.seed)
                 for _, g in test.groupby("domain", sort=True)]
        tiles = pd.concat(parts).reset_index(drop=True)
    else:
        tiles = test
    tiles[["img", "msk", "domain"]].to_csv(out / "sample_tiles.csv", index=False)

    cand = pd.read_csv(a.candidates)
    base = dict(name="full_theta0", source="전체 test 통계 (10/1 pooled와 같은 값)", toy_iou=np.nan,
                **dict(zip(PARAMS, th_full)))
    cand = pd.concat([pd.DataFrame([base]), cand], ignore_index=True)
    thetas = cand[PARAMS].to_numpy(float)
    K, n = len(cand), a.n_classes
    print(f"[data] test {len(test)}타일 중 {len(tiles)}타일 사용 (도메인 {tiles['domain'].nunique()}개), 후보 {K}개")
    print(f"[data] full_theta0 = {np.round(th_full, 2).tolist()}")
    tcol = f"cls_{a.target + 1}"  # spec의 원본 마스크 값 = 모델 인덱스 + 1 (labelmap_19 기준)
    if tcol in tiles:
        print(f"[data] 대상 클래스 픽셀: 표본 {int(tiles[tcol].sum())} / test 전체 {int(test[tcol].sum())}, "
              f"등장 타일 {int((tiles[tcol] > 0).sum())}개")

    lut = build_lut(a.label_map, n)
    model = load_model(a.checkpoint, "resnet34", n, "Unet")
    means = [torch.tensor(t[:3], dtype=torch.float32).view(1, 3, 1, 1) for t in thetas]
    stds = [torch.tensor(t[3:], dtype=torch.float32).view(1, 3, 1, 1) for t in thetas]

    domains = sorted(tiles["domain"].unique())
    dix = {d: i for i, d in enumerate(domains)}
    ck = out / "progress.npz"
    cms, done, sec = np.zeros((K, n, n), np.int64), 0, np.zeros(K)
    dcms = np.zeros((K, len(domains), n, n), np.int64)  # 도메인별 혼동행렬
    if a.resume and ck.exists():
        z = np.load(ck)
        if z["cms"].shape == cms.shape and "dcms" in z.files and z["dcms"].shape == dcms.shape:
            cms, dcms, done, sec = z["cms"], z["dcms"], int(z["done"]), z["sec"]
            print(f"[iou] 이어서 시작: {done}/{len(tiles)}")
        else:
            print("[iou] 이전 progress.npz와 후보 수 또는 형식이 달라 처음부터 시작")

    t0 = time.time()
    rows = tiles.to_dict("records")
    for s in range(done, len(rows), a.batch):
        chunk = rows[s:s + a.batch]
        x = torch.from_numpy(np.stack([_read_tif(r["img"])[list(a.bands)] for r in chunk]).astype(np.float32))
        gts = [lut[_read_tif(r["msk"])[0].astype(np.int64) % 256] for r in chunk]
        with torch.inference_mode():
            for k in range(K):
                tk = time.time()
                pred = model((x - means[k]) / stds[k]).argmax(1).numpy()
                for p, gt, r in zip(pred, gts, chunk):
                    cm = confusion(p, gt, n)
                    cms[k] += cm
                    dcms[k, dix[r["domain"]]] += cm
                sec[k] += time.time() - tk
        done = s + len(chunk)
        if done % (a.batch * 25) < a.batch or done == len(rows):
            np.savez(ck, cms=cms, dcms=dcms, done=done, sec=sec)
            el = time.time() - t0
            print(f"[iou] {done}/{len(rows)}  {el / 60:.1f}분, 남은 예상 "
                  f"{el / max(done - 0, 1) * (len(rows) - done) / 60:.0f}분  "
                  f"침엽수 " + " ".join(f"{iou_from_cm(cms[k])[a.target]:.1f}" for k in range(K)),
                  flush=True)

    res = []
    for k in range(K):
        iou = iou_from_cm(cms[k])
        res.append(dict(name=cand.loc[k, "name"], source=cand.loc[k, "source"],
                        toy_iou=cand.loc[k, "toy_iou"], target_iou=iou[a.target],
                        miou=np.nanmean(iou), sec=sec[k], n_tiles=len(tiles),
                        **dict(zip(PARAMS, thetas[k])),
                        **{f"iou_{c}_{NAMES[c]}": iou[c] for c in range(n)}))
    df = pd.DataFrame(res).round(4)
    df.to_csv(out / "candidates_iou.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(df[["name", "toy_iou", "target_iou", "miou"]].to_string(index=False))
    drows = []
    for k in range(K):
        for d, j in dix.items():
            iou = iou_from_cm(dcms[k, j])
            drows.append(dict(name=cand.loc[k, "name"], domain=d, target_iou=iou[a.target],
                              target_pixels=int(dcms[k, j][a.target].sum()), miou=np.nanmean(iou)))
    dd = pd.DataFrame(drows).round(4)
    dd.to_csv(out / "domain_iou.csv", index=False)
    print("\n[도메인별 대상 클래스 IoU]")
    print(dd.pivot(index="domain", columns="name", values="target_iou")[list(cand["name"])].round(1).to_string())
    print(f"저장: {out / 'candidates_iou.csv'}, {out / 'domain_iou.csv'}")


if __name__ == "__main__":
    main()
