"""FLAIR 전체 데이터셋: 데이터 명세 집계 + U-Net 기준 IoU (타일을 하나씩 읽어서 메모리 안 터지게).

사용법
  # 1) 데이터 명세만 (모델 불필요, 빠름)
  python analysis/full_eval.py --img-dir <IMG_ROOT> --msk-dir <MSK_ROOT> --out out_full_spec

  # 2) 명세 + U-Net 기준 IoU (오래 걸림, 중간에 끊겨도 --resume 으로 이어서)
  python analysis/full_eval.py --img-dir <IMG_ROOT> --msk-dir <MSK_ROOT> --out out_full_test ^
      --include test --checkpoint models/rgb15/FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth ^
      --n-classes 19 --label-map labelmap_19.json --threads 4 --resume

출력 (--out 폴더)
  spec_tiles.csv     타일별: split, domain, 채널 합계, 원본 마스크 값(1~19)별 픽셀 수
  spec_summary.md    split·도메인별 타일 수, 클래스별 픽셀 비율·등장 타일 수 (명세서용)
  base_iou.csv       기준 IoU (pooled / per-tile 두 방식), 클래스별
  tile_iou.csv       타일별·클래스별 IoU (per-tile 기준)
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import IGNORE, _read_tif, build_lut, confusion, load_model, pair_files  # noqa: E402

FLAIR_NAMES = {1: "building", 2: "pervious surface", 3: "impervious surface", 4: "bare soil",
               5: "water", 6: "coniferous", 7: "deciduous", 8: "brushwood", 9: "vineyard",
               10: "herbaceous vegetation", 11: "agricultural land", 12: "plowed land",
               13: "swimming pool", 14: "snow", 15: "clear cut", 16: "mixed", 17: "ligneous",
               18: "greenhouse", 19: "other"}


def split_of(path):
    s = str(path).lower()
    return "test" if "test" in s else "train"


def domain_of(path):
    m = re.search(r"D\d{3}_\d{4}", str(path))
    return m.group(0) if m else "unknown"


# ------------------------------------------------------------ 1) 데이터 명세
def build_spec(pairs, bands, out):
    """타일을 하나씩 읽어 채널 합계·마스크 값 분포를 모은다. 결과는 spec_tiles.csv에 캐시."""
    cache = out / "spec_tiles.csv"
    if cache.exists():
        rows = list(csv.DictReader(open(cache, encoding="utf-8")))
        if len(rows) == len(pairs):
            print(f"[spec] 캐시 사용: {cache}")
            return rows
    rows, t0 = [], time.time()
    for i, (ip, mp) in enumerate(pairs):
        img = _read_tif(ip)[list(bands)].astype(np.float64)
        msk = _read_tif(mp)[0].astype(np.int64) % 256
        x = img.reshape(3, -1)
        cnt = np.bincount(msk.ravel(), minlength=256)
        r = {"img": str(ip), "msk": str(mp), "split": split_of(mp), "domain": domain_of(mp),
             "n_px": x.shape[1],
             **{f"sum_{c}": x[k].sum() for k, c in enumerate("RGB")},
             **{f"sq_{c}": (x[k] ** 2).sum() for k, c in enumerate("RGB")},
             **{f"cls_{v}": int(cnt[v]) for v in range(1, 20)}}
        rows.append(r)
        if (i + 1) % 500 == 0:
            print(f"[spec] {i + 1}/{len(pairs)}  {time.time() - t0:.0f}s")
    with open(cache, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def pooled_theta(rows):
    n = sum(float(r["n_px"]) for r in rows)
    s = np.array([sum(float(r[f"sum_{c}"]) for r in rows) for c in "RGB"])
    ss = np.array([sum(float(r[f"sq_{c}"]) for r in rows) for c in "RGB"])
    mean = s / n
    return np.concatenate([mean, np.sqrt(ss / n - mean ** 2)])


def write_summary(rows, out):
    L = ["# FLAIR 데이터 명세 (자동 집계)", ""]
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        doms = sorted({r["domain"] for r in rs})
        th = pooled_theta(rs)
        L += [f"## {split}", "",
              f"- 타일 수: {len(rs)}",
              f"- 도메인 수: {len(doms)} ({', '.join(doms[:10])}{' …' if len(doms) > 10 else ''})",
              f"- 전체 채널 평균 (R,G,B): {th[0]:.2f}, {th[1]:.2f}, {th[2]:.2f}",
              f"- 전체 채널 표준편차 (R,G,B): {th[3]:.2f}, {th[4]:.2f}, {th[5]:.2f}", "",
              "| 값 | 클래스 | 픽셀 비율(%) | 등장 타일 수 |", "|---|---|---|---|"]
        tot = sum(sum(int(r[f"cls_{v}"]) for v in range(1, 20)) for r in rs)
        for v in range(1, 20):
            px = sum(int(r[f"cls_{v}"]) for r in rs)
            nt = sum(1 for r in rs if int(r[f"cls_{v}"]) > 0)
            L.append(f"| {v} | {FLAIR_NAMES[v]} | {100 * px / max(tot, 1):.2f} | {nt} |")
        L += ["", "| 도메인 | 타일 수 |", "|---|---|"]
        for d in doms:
            L.append(f"| {d} | {sum(1 for r in rs if r['domain'] == d)} |")
        L.append("")
    (out / "spec_summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[spec] 저장: {out / 'spec_summary.md'}")


# ------------------------------------------------------------ 2) U-Net 기준 IoU
def iou_from_cm(cm):
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1) - tp
    return np.where(den > 0, tp / np.maximum(den, 1), np.nan) * 100


def run_iou(rows, a, out):
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    n = a.n_classes
    lut = build_lut(a.label_map, n)
    model = load_model(a.checkpoint, a.encoder, n, a.arch).to(a.device)
    theta = pooled_theta(rows)          # 평가 대상 타일 전체로 계산한 통계 한 세트
    print("[iou] pooled θ:", np.round(theta, 2))
    mean = torch.tensor(theta[:3], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(theta[3:], dtype=torch.float32).view(1, 3, 1, 1)

    ck = out / "iou_progress.npz"
    cm_pool = np.zeros((n, n), np.int64)
    cm_tile = np.zeros((n, n), np.int64)
    done = 0
    tile_f = out / "tile_iou.csv"
    if a.resume and ck.exists():        # 끊긴 지점부터 이어서
        z = np.load(ck)
        cm_pool, cm_tile, done = z["cm_pool"], z["cm_tile"], int(z["done"])
        print(f"[iou] 이어서 시작: {done}/{len(rows)}")
    else:
        with open(tile_f, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["img", "domain"] + [f"iou_{c}" for c in range(n)])

    t0 = time.time()
    for s in range(done, len(rows), a.batch):
        chunk = rows[s:s + a.batch]
        x = torch.from_numpy(np.stack([_read_tif(r["img"])[list(a.bands)] for r in chunk]).astype(np.float32))
        gts = [lut[_read_tif(r["msk"])[0].astype(np.int64) % 256] for r in chunk]
        m = x.mean(dim=(2, 3), keepdim=True)
        sd = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
        with torch.inference_mode():
            p_pool = model(((x - mean) / std).to(a.device)).argmax(1).cpu().numpy()   # 통계 한 세트
            p_tile = model(((x - m) / sd).to(a.device)).argmax(1).cpu().numpy()       # 타일별 통계
        with open(tile_f, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r, gt, pp, pt in zip(chunk, gts, p_pool, p_tile):
                cm_pool += confusion(pp, gt, n)
                ct = confusion(pt, gt, n)
                cm_tile += ct
                w.writerow([r["img"], r["domain"]] + [f"{v:.2f}" for v in iou_from_cm(ct)])
        done = s + len(chunk)
        if done % (a.batch * 25) < a.batch or done == len(rows):
            np.savez(ck, cm_pool=cm_pool, cm_tile=cm_tile, done=done)
            el = time.time() - t0
            print(f"[iou] {done}/{len(rows)}  {el / 60:.1f}분, 남은 예상 {el / max(done - 0, 1) * (len(rows) - done) / 60:.0f}분")

    ip, it = iou_from_cm(cm_pool), iou_from_cm(cm_tile)
    names = {int(k): int(v) for k, v in json.load(open(a.label_map)).items()
             if not k.startswith("_") and v is not None} if a.label_map else {}
    inv = {v: k for k, v in names.items()}
    with open(out / "base_iou.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_idx", "mask_value", "class", "iou_pooled", "iou_per_tile"])
        for c in range(n):
            mv = inv.get(c, c + 1)
            w.writerow([c, mv, FLAIR_NAMES.get(mv, ""), f"{ip[c]:.2f}", f"{it[c]:.2f}"])
    print(f"[iou] 저장: {out / 'base_iou.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--msk-dir", required=True)
    p.add_argument("--img-glob", default="IMG_*.tif")
    p.add_argument("--msk-glob", default="MSK_*.tif")
    p.add_argument("--bands", type=int, nargs=3, default=[0, 1, 2])
    p.add_argument("--include", help="마스크 경로에 이 문자열이 있는 타일만 (예: test)")
    p.add_argument("--limit", type=int, help="앞에서 N개만 (시험용)")
    p.add_argument("--out", default="out_full")
    p.add_argument("--checkpoint", help="주면 U-Net 기준 IoU까지 계산")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--arch", default="Unet")
    p.add_argument("--n-classes", type=int, default=19)
    p.add_argument("--label-map")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pair_files(a.img_dir, a.msk_dir, a.img_glob, a.msk_glob)
    if a.include:
        pairs = [pp for pp in pairs if a.include in str(pp[1])]
    if a.limit:
        pairs = pairs[: a.limit]
    print(f"타일 쌍 {len(pairs)}개")
    if not pairs:
        sys.exit("타일을 못 찾음. --img-dir/--msk-dir 경로와 파일 이름(IMG_*.tif, MSK_*.tif) 확인")

    rows = build_spec(pairs, a.bands, out)
    write_summary(rows, out)
    if a.checkpoint:
        run_iou(rows, a, out)


if __name__ == "__main__":
    main()
