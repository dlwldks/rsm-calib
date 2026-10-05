"""FLAIR #1 전체 데이터셋 명세 통계 추출.

압축 푼 폴더를 훑어서 데이터 명세서에 들어갈 수치를 CSV/JSON/MD로 저장한다.
모델·torch 불필요 (numpy, rasterio만 사용).

사용:
    python analysis/flair_spec.py --root data/flair1 --out docs/flair1_spec
    python analysis/flair_spec.py --root data/flair1 --out docs/flair1_spec --img-sample 50   # 영상 통계는 도메인당 50타일만

출력 (--out):
    tiles.csv        타일별 split·domain·zone·id·영상/마스크 존재·메타데이터
    format.json      영상/마스크 규격(크기·밴드·dtype·CRS·해상도) 집계, 짝 불일치, 용량
    bands.csv        split별 밴드 평균·표준편차·최소·최대
    domains.csv      도메인별 타일 수·zone 수·RGB 평균/표준편차
    class_freq.csv   클래스별 픽셀 수·비율 (train / test)
    metadata.json    카메라·촬영 연도·월 분포 (메타데이터 json이 있을 때)
    summary.md       위 내용을 표로 정리
"""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CLASSES = {
    1: "building", 2: "pervious surface", 3: "impervious surface", 4: "bare soil",
    5: "water", 6: "coniferous", 7: "deciduous", 8: "brushwood", 9: "vineyard",
    10: "herbaceous vegetation", 11: "agricultural land", 12: "plowed land",
    13: "swimming pool", 14: "snow", 15: "clear cut", 16: "mixed", 17: "ligneous",
    18: "greenhouse", 19: "other",
}
BAND_NAMES = ["R", "G", "B", "NIR", "Elevation"]
RE_DOMAIN = re.compile(r"^D\d{3}_\d{4}$")
RE_ZONE = re.compile(r"^Z\d+_[A-Za-z]+$")


# ------------------------------------------------------------------ discovery
def tile_id(p):
    nums = re.findall(r"\d+", p.stem)
    return nums[-1] if nums else p.stem


def split_of(p):
    parts = [s.lower() for s in p.parts]
    return "test" if any("test" in s for s in parts) else "train"


def domain_zone(p):
    dom = zone = ""
    for s in p.parts:
        if RE_DOMAIN.match(s):
            dom = s
        elif RE_ZONE.match(s):
            zone = s
    return dom, zone


def discover(root):
    imgs, msks = {}, {}
    for p in root.rglob("IMG_*.tif"):
        imgs[(split_of(p), tile_id(p))] = p
    for p in root.rglob("MSK_*.tif"):
        msks[(split_of(p), tile_id(p))] = p
    return imgs, msks


# ------------------------------------------------------------------ workers
def read_img(path):
    import rasterio
    with rasterio.open(path) as src:
        a = src.read()
        info = dict(
            shape=f"{src.height}x{src.width}", count=src.count,
            dtype=str(a.dtype), crs=str(src.crs) if src.crs else "",
            res=f"{abs(src.res[0]):.3f}",
        )
    x = a.reshape(a.shape[0], -1).astype(np.float64)
    info["n"] = x.shape[1]
    info["sum"] = x.sum(1)
    info["sumsq"] = (x ** 2).sum(1)
    info["min"] = x.min(1)
    info["max"] = x.max(1)
    return info


def read_msk(path):
    import rasterio
    with rasterio.open(path) as src:
        a = src.read(1)
        info = dict(shape=f"{src.height}x{src.width}", count=src.count, dtype=str(a.dtype))
    info["hist"] = np.bincount(a.ravel().astype(np.int64) % 256, minlength=256)
    return info


def work(job):
    kind, key, path = job
    try:
        return kind, key, (read_img if kind == "img" else read_msk)(path), None
    except Exception as e:  # 깨진 파일은 기록만 하고 계속
        return kind, key, None, f"{type(e).__name__}: {e}"


# ------------------------------------------------------------------ helpers
class Acc:
    """밴드별 합·제곱합·최솟값·최댓값 누적."""
    def __init__(self):
        self.n = 0
        self.s = self.ss = self.mn = self.mx = None

    def add(self, r):
        if self.s is None:
            self.s, self.ss = r["sum"].copy(), r["sumsq"].copy()
            self.mn, self.mx = r["min"].copy(), r["max"].copy()
        elif len(self.s) == len(r["sum"]):
            self.s += r["sum"]; self.ss += r["sumsq"]
            self.mn = np.minimum(self.mn, r["min"]); self.mx = np.maximum(self.mx, r["max"])
        else:
            return
        self.n += r["n"]

    def stats(self):
        mean = self.s / self.n
        std = np.sqrt(np.maximum(self.ss / self.n - mean ** 2, 0))
        return mean, std, self.mn, self.mx


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def load_metadata(root):
    for p in root.rglob("*.json"):
        if "metadata" in p.name.lower():
            try:
                return p, json.load(open(p, encoding="utf-8"))
            except Exception:
                pass
    return None, None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="압축 푼 FLAIR #1 폴더")
    ap.add_argument("--out", default="docs/flair1_spec")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--img-sample", type=int, default=0,
                    help="영상 밴드 통계를 도메인당 N타일만 계산 (0=전체)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[1/4] 파일 탐색: {root}")
    imgs, msks = discover(root)
    keys = sorted(set(imgs) | set(msks))
    print(f"      영상 {len(imgs)}개, 마스크 {len(msks)}개")
    if not keys:
        sys.exit("IMG_*.tif / MSK_*.tif 를 찾지 못함. --root 확인.")

    meta_path, meta = load_metadata(root)

    # 도메인당 샘플링
    tile_dom = {}
    for k in keys:
        p = imgs.get(k) or msks.get(k)
        tile_dom[k] = domain_zone(p)
    img_keys = sorted(imgs)
    if a.img_sample > 0:
        rng = np.random.default_rng(a.seed)
        by_dom = defaultdict(list)
        for k in img_keys:
            by_dom[(k[0], tile_dom[k][0])].append(k)
        img_keys = []
        for ks in by_dom.values():
            n = min(a.img_sample, len(ks))
            img_keys += [ks[i] for i in rng.choice(len(ks), n, replace=False)]

    jobs = [("img", k, imgs[k]) for k in img_keys] + [("msk", k, msks[k]) for k in sorted(msks)]
    print(f"[2/4] 읽기: 영상 {len(img_keys)}개 + 마스크 {len(msks)}개 (workers={a.workers})")

    fmt = defaultdict(Counter)
    errors = []
    band_acc = defaultdict(Acc)          # split
    dom_acc = defaultdict(Acc)           # (split, domain)
    hist = defaultdict(lambda: np.zeros(256, np.int64))  # split
    done = 0
    with Pool(a.workers) as pool:
        for kind, key, r, err in pool.imap_unordered(work, jobs, chunksize=16):
            done += 1
            if done % 2000 == 0:
                el = time.time() - t0
                print(f"      {done}/{len(jobs)}  ({el/60:.1f}분 경과)", flush=True)
            if err:
                errors.append((kind, key[0], key[1], err))
                continue
            split = key[0]
            if kind == "img":
                for f in ("shape", "count", "dtype", "crs", "res"):
                    fmt[f"img_{f}"][str(r[f])] += 1
                band_acc[split].add(r)
                dom_acc[(split, tile_dom[key][0])].add(r)
            else:
                for f in ("shape", "count", "dtype"):
                    fmt[f"msk_{f}"][str(r[f])] += 1
                hist[split] += r["hist"]

    print("[3/4] 집계")
    # ---- tiles.csv
    meta_cols = []
    if meta:
        first = next(iter(meta.values()))
        meta_cols = [c for c in first if c not in ("domain", "zone")]
    rows = []
    for k in keys:
        dom, zone = tile_dom[k]
        m = (meta or {}).get(f"IMG_{k[1]}", {})
        rows.append([k[0], dom, zone, k[1], int(k in imgs), int(k in msks)]
                    + [m.get(c, "") for c in meta_cols])
    write_csv(out / "tiles.csv", ["split", "domain", "zone", "id", "has_img", "has_msk"] + meta_cols, rows)

    # ---- format.json
    def size_of(ps):
        return sum(p.stat().st_size for p in ps)
    splits = sorted({k[0] for k in keys})
    fmt_out = {
        "root": str(root),
        "n_img": {s: sum(1 for k in imgs if k[0] == s) for s in splits},
        "n_msk": {s: sum(1 for k in msks if k[0] == s) for s in splits},
        "img_only": [f"{s}/{i}" for s, i in keys if (s, i) in imgs and (s, i) not in msks][:50],
        "msk_only": [f"{s}/{i}" for s, i in keys if (s, i) in msks and (s, i) not in imgs][:50],
        "bytes_img": {s: size_of(p for k, p in imgs.items() if k[0] == s) for s in splits},
        "bytes_msk": {s: size_of(p for k, p in msks.items() if k[0] == s) for s in splits},
        "img_stats_tiles": len(img_keys),
        "img_sample_per_domain": a.img_sample,
        "format_counts": {k: dict(v) for k, v in fmt.items()},
        "read_errors": errors[:100],
        "n_read_errors": len(errors),
        "metadata_file": str(meta_path) if meta_path else None,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    json.dump(fmt_out, open(out / "format.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- bands.csv
    nb = max((len(acc.s) for acc in band_acc.values() if acc.s is not None), default=0)
    bnames = BAND_NAMES[:nb] + [f"B{i+1}" for i in range(len(BAND_NAMES), nb)]
    band_rows = []
    for s in splits:
        if band_acc[s].s is None:
            continue
        mean, std, mn, mx = band_acc[s].stats()
        for i, b in enumerate(bnames):
            band_rows.append([s, b, f"{mean[i]:.3f}", f"{std[i]:.3f}", f"{mn[i]:.0f}", f"{mx[i]:.0f}"])
    write_csv(out / "bands.csv", ["split", "band", "mean", "std", "min", "max"], band_rows)

    # ---- domains.csv
    dom_tiles = Counter((k[0], tile_dom[k][0]) for k in keys)
    dom_zones = defaultdict(set)
    for k in keys:
        dom_zones[(k[0], tile_dom[k][0])].add(tile_dom[k][1])
    dom_rows = []
    for sd in sorted(dom_tiles):
        r = [sd[0], sd[1], dom_tiles[sd], len(dom_zones[sd])]
        acc = dom_acc.get(sd)
        if acc is not None and acc.s is not None:
            mean, std, _, _ = acc.stats()
            r += [f"{v:.2f}" for v in mean[:3]] + [f"{v:.2f}" for v in std[:3]]
        else:
            r += [""] * 6
        dom_rows.append(r)
    write_csv(out / "domains.csv",
              ["split", "domain", "n_tiles", "n_zones", "R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"],
              dom_rows)

    # ---- class_freq.csv
    vals = sorted(set(np.nonzero(sum(hist.values()))[0].tolist()) | set(CLASSES))
    tot = {s: hist[s].sum() for s in splits}
    cls_rows = []
    for v in vals:
        r = [v, CLASSES.get(v, "(undefined)")]
        for s in splits:
            c = int(hist[s][v])
            r += [c, f"{100 * c / tot[s]:.3f}" if tot[s] else ""]
        cls_rows.append(r)
    write_csv(out / "class_freq.csv",
              ["value", "class"] + sum([[f"{s}_pixels", f"{s}_pct"] for s in splits], []), cls_rows)

    # ---- metadata.json
    meta_out = {}
    if meta:
        tiles_by_split = {s: {f"IMG_{k[1]}" for k in keys if k[0] == s} for s in splits}
        for s in splits:
            ms = [meta[t] for t in tiles_by_split[s] if t in meta]
            d = {"n_with_meta": len(ms)}
            for c in ("camera",):
                d[c] = dict(Counter(m.get(c, "") for m in ms).most_common())
            dates = [str(m.get("date", "")) for m in ms if m.get("date")]
            d["year"] = dict(sorted(Counter(x[:4] for x in dates).items()))
            d["month"] = dict(sorted(Counter(x[5:7] for x in dates).items()))
            meta_out[s] = d
        json.dump(meta_out, open(out / "metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- summary.md
    print("[4/4] summary.md 작성")
    gb = lambda b: f"{b / 1024**3:.2f} GB"
    L = ["# FLAIR #1 데이터 통계 (flair_spec.py 자동 생성)", ""]
    L += ["## 구성", md_table(
        ["split", "영상 수", "마스크 수", "도메인 수", "영상 용량", "마스크 용량"],
        [[s, fmt_out["n_img"][s], fmt_out["n_msk"][s],
          len({d for (sp, d) in dom_tiles if sp == s}),
          gb(fmt_out["bytes_img"][s]), gb(fmt_out["bytes_msk"][s])] for s in splits]), ""]
    L += [f"- 영상만 있고 마스크 없음: {len(fmt_out['img_only'])}개, 마스크만 있음: {len(fmt_out['msk_only'])}개",
          f"- 읽기 오류: {len(errors)}개",
          f"- 밴드 통계 계산 타일 수: {len(img_keys)}"
          + (f" (도메인당 최대 {a.img_sample}개 샘플)" if a.img_sample else " (전체)"), ""]
    L += ["## 파일 규격", md_table(["항목", "값 (타일 수)"],
          [[k, ", ".join(f"{v} ({n})" for v, n in c.items())] for k, c in fmt_out["format_counts"].items()]), ""]
    L += ["## 밴드 통계", md_table(["split", "band", "mean", "std", "min", "max"], band_rows), ""]
    L += ["## 클래스 픽셀 비율 (%)", md_table(
        ["값", "클래스"] + [f"{s} %" for s in splits],
        [[r[0], r[1]] + [r[3 + 2 * i] for i in range(len(splits))] for r in cls_rows]), ""]
    L += ["## 도메인", md_table(
        ["split", "domain", "타일", "zone", "R mean", "G mean", "B mean"],
        [r[:7] for r in dom_rows]), ""]
    if meta_out:
        L += ["## 메타데이터"]
        for s, d in meta_out.items():
            L += [f"### {s}", f"- 메타데이터 있는 타일: {d['n_with_meta']}",
                  f"- 카메라: {d['camera']}", f"- 연도: {d['year']}", f"- 월: {d['month']}", ""]
    L += [f"소요 시간: {fmt_out['elapsed_min']}분"]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"완료 → {out}  ({fmt_out['elapsed_min']}분)")


if __name__ == "__main__":
    main()
