"""
사용법
  python run.py synthetic                       # 모델 없이 RSM 탐색 로직만 검증
  python run.py inspect   --img-dir .. --msk-dir .. [--checkpoint ..]
  python run.py calibrate --img-dir .. --msk-dir .. --checkpoint .. --target 5 --out out_d004
  python run.py transfer  --img-dir .. --msk-dir .. --checkpoint .. --source out_d004/result.json --out out_d067
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rsm import proportional_transfer
from search import NAMES, rsm_search


def _json(o):
    if isinstance(o, dict):
        return {k: _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _fmt(theta):
    return ", ".join(f"{n}={v:.2f}" for n, v in zip(NAMES, theta))


# ------------------------------------------------------------------ synthetic
def cmd_synthetic(a):
    rng = np.random.default_rng(a.seed)
    theta0 = np.array([97.28, 108.99, 109.48, 62.47, 56.64, 54.35])       # D067 base
    true = theta0 * np.array([0.98, 1.05, 0.95, 1.00, 0.99, 0.99])         # R↓ G↑ B↓
    width = theta0 * 0.06

    def f(t):
        z = (t - true) / width
        return 80 - 2.0 * np.sum(z ** 2) + 0.5 * z[0] * z[1] + rng.normal(0, a.noise)

    res = rsm_search(f, theta0, max_evals=a.max_evals, seed=a.seed)
    print("\n정답 최적점 :", _fmt(true))
    print("탐색 best   :", _fmt(res["theta_best"]))
    print("RSM 정상점  :", _fmt(res["theta_rsm"]))
    print(f"IoU {res['iou_base']:.2f} -> {res['iou_best']:.2f}  (evals={len(res['Y'])})")
    print("RSM 통계    :", {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in res["rsm"].items()})


# ------------------------------------------------------------------ shared
def _load(a):
    from data import build_lut, load_tiles, pair_files
    pairs = pair_files(a.img_dir, a.msk_dir, a.img_glob, a.msk_glob)
    if not pairs:
        raise SystemExit("IMG/MSK 쌍을 못 찾음. --img-glob/--msk-glob 확인")
    if a.limit:
        pairs = pairs[: a.limit]
    imgs, msks = load_tiles(pairs, build_lut(a.label_map, a.n_classes), a.bands)
    print(f"[data] tiles={len(imgs)} img={imgs.shape[1:]} {imgs.dtype}")
    return imgs, msks


def _split(msks, target, test_frac, seed):
    """타깃 클래스가 있는 타일만 골라 calib/test로 분리 (논문은 분리 안 함)."""
    has = np.array([(m == target).any() for m in msks])
    idx = np.where(has)[0]
    np.random.default_rng(seed).shuffle(idx)
    n_test = int(round(len(idx) * test_frac))
    return idx[n_test:], idx[:n_test]


def _report(ev_all, msks, splits, thetas, n, names, target, out):
    """splits: {name: idx}, thetas: {label: theta}. 클래스별 IoU 표 + fusion."""
    from data import class_iou, decision_fusion
    preds = {lab: ev_all.predict(th) for lab, th in thetas.items()}
    base_lab = next(iter(thetas))
    rows, table = [], {}
    for sname, idx in splits.items():
        if len(idx) == 0:
            continue
        ious = {lab: class_iou(p[idx], msks[idx], n) for lab, p in preds.items()}
        for lab in list(thetas)[1:]:
            fused = decision_fusion(preds[base_lab][idx], [(target, preds[lab][idx])])
            ious[f"fusion({lab})"] = class_iou(fused, msks[idx], n)
        table[sname] = ious
        print(f"\n=== {sname} ({len(idx)} tiles) — IoU % ===")
        cols = list(ious)
        print(f"{'class':<22}" + "".join(f"{c:>18}" for c in cols))
        for c in range(n):
            if all(np.isnan(ious[k][c]) for k in cols):
                continue
            mark = " *" if c == target else ""
            name = (names[c] if names and c < len(names) else str(c)) + mark
            print(f"{name:<22}" + "".join(f"{ious[k][c]:>18.2f}" for k in cols))
            rows.append([sname, name] + [ious[k][c] for k in cols])
        mi = {k: np.nanmean(v) for k, v in ious.items()}
        print(f"{'mIoU':<22}" + "".join(f"{mi[k]:>18.2f}" for k in cols))
    with open(out / "iou_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        for sname, ious in table.items():
            w.writerow(["split", "class"] + list(ious))
            break
        w.writerows(rows)
    return table


def _setup_model(a):
    from data import load_model
    return load_model(a.checkpoint, a.encoder, a.n_classes, a.arch)


# ------------------------------------------------------------------ inspect
def cmd_inspect(a):
    a.label_map, a.limit = None, a.limit or 20
    from data import _read_tif, pair_files
    pairs = pair_files(a.img_dir, a.msk_dir, a.img_glob, a.msk_glob)[: a.limit]
    print(f"쌍 {len(pairs)}개 (최대 {a.limit}개 확인)")
    if not pairs:
        return
    img = _read_tif(pairs[0][0])
    print(f"이미지 {pairs[0][0].name}: shape={img.shape} dtype={img.dtype} "
          f"min={img.min()} max={img.max()}")
    vals = {}
    for _, mp in pairs:
        u, c = np.unique(_read_tif(mp)[0], return_counts=True)
        for x, y in zip(u, c):
            vals[int(x)] = vals.get(int(x), 0) + int(y)
    tot = sum(vals.values())
    print("마스크 원본 값 분포:")
    for k in sorted(vals):
        print(f"  {k:>3}: {vals[k] / tot:6.2%}")
    if a.checkpoint:
        import torch
        m = _setup_model(a)
        x = torch.from_numpy(img[list(a.bands)][None].astype(np.float32))
        with torch.inference_mode():
            print("모델 출력 shape:", tuple(m(x).shape))


# ------------------------------------------------------------------ calibrate
def cmd_calibrate(a):
    from data import Evaluator, channel_stats, class_iou
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.class_names.split(",") if a.class_names else None
    imgs, msks = _load(a)
    calib, test = _split(msks, a.target, a.test_frac, a.seed)
    print(f"[split] target={a.target} calib={len(calib)} test={len(test)} all={len(imgs)}")
    if len(calib) < 5:
        raise SystemExit("타깃 클래스가 있는 타일이 너무 적음")

    model = _setup_model(a)
    ev_cal = Evaluator(model, imgs[calib], a.n_classes, a.batch, a.device, a.threads)
    theta0 = channel_stats(imgs[calib])
    print("[theta_base]", _fmt(theta0))

    def objective(theta):
        return class_iou(ev_cal.predict(theta), msks[calib], a.n_classes)[a.target]

    res = rsm_search(objective, theta0, a.rel_broad, a.n_broad, a.rel_local, a.n_local,
                     a.max_evals, a.eps_best, a.eps_pred, a.seed)
    print("\n[theta_best]", _fmt(res["theta_best"]))
    print("[ratio best/base]", np.round(res["theta_best"] / theta0, 4))
    print("[RSM]", res["rsm"])

    with open(out / "trials.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(NAMES + ["iou"])
        w.writerows(np.column_stack([res["X"], res["Y"]]).round(4).tolist())

    ev_all = Evaluator(model, imgs, a.n_classes, a.batch, a.device, a.threads)
    # 전체/테스트 타일의 base는 각 집합 통계가 맞지만, 비교 단순화를 위해 calib 통계 사용
    table = _report(ev_all, msks, {"calib": calib, "test": test, "all": np.arange(len(imgs))},
                    {"base": theta0, "best": res["theta_best"]},
                    a.n_classes, names, a.target, out)
    json.dump(_json(dict(args=vars(a), theta_base=theta0, theta_best=res["theta_best"],
                         theta_rsm=res["theta_rsm"], ratio=res["theta_best"] / theta0,
                         iou_calib_base=res["iou_base"], iou_calib_best=res["iou_best"],
                         rsm=res["rsm"], history=res["history"],
                         iou={s: {k: v for k, v in t.items()} for s, t in table.items()})),
              open(out / "result.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n저장: {out}/result.json, trials.csv, iou_table.csv")


# ------------------------------------------------------------------ transfer
def cmd_transfer(a):
    """다른 도메인(B)의 result.json 비율을 이 도메인(A)에 적용 (재탐색 없음)."""
    from data import Evaluator, channel_stats
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = json.load(open(a.source))
    imgs, msks = _load(a)
    idx = np.where([(m == a.target).any() for m in msks])[0]
    theta_a = channel_stats(imgs[idx])
    theta_t = proportional_transfer(theta_a, src["theta_base"], src["theta_best"])
    print("[A base]    ", _fmt(theta_a))
    print("[A pred|B]  ", _fmt(theta_t))
    ev = Evaluator(_setup_model(a), imgs, a.n_classes, a.batch, a.device, a.threads)
    names = a.class_names.split(",") if a.class_names else None
    table = _report(ev, msks, {"target_tiles": idx, "all": np.arange(len(imgs))},
                    {"base": theta_a, "transfer": theta_t}, a.n_classes, names, a.target, out)
    json.dump(_json(dict(source=a.source, theta_base=theta_a, theta_transfer=theta_t,
                         iou=table)), open(out / "transfer.json", "w"), indent=2)


# ------------------------------------------------------------------ cli
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synthetic")
    s.add_argument("--noise", type=float, default=0.3)
    s.add_argument("--max-evals", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--img-dir", required=True)
    common.add_argument("--msk-dir", required=True)
    common.add_argument("--img-glob", default="IMG_*.tif")
    common.add_argument("--msk-glob", default="MSK_*.tif")
    common.add_argument("--bands", type=int, nargs=3, default=[0, 1, 2])
    common.add_argument("--checkpoint")
    common.add_argument("--encoder", default="resnet34")      # MiT-B5: mit_b5
    common.add_argument("--arch", default="Unet")
    common.add_argument("--n-classes", type=int, default=15)
    common.add_argument("--label-map")
    common.add_argument("--class-names")
    common.add_argument("--limit", type=int)
    common.add_argument("--batch", type=int, default=8)
    common.add_argument("--device", default="cpu")
    common.add_argument("--threads", type=int)
    common.add_argument("--seed", type=int, default=0)

    sub.add_parser("inspect", parents=[common])

    c = sub.add_parser("calibrate", parents=[common])
    c.add_argument("--target", type=int, required=True, help="모델 클래스 인덱스")
    c.add_argument("--out", default="out")
    c.add_argument("--test-frac", type=float, default=0.3)
    c.add_argument("--rel-broad", type=float, default=0.10)
    c.add_argument("--n-broad", type=int, default=30)
    c.add_argument("--rel-local", type=float, default=0.03)
    c.add_argument("--n-local", type=int, default=6)
    c.add_argument("--max-evals", type=int, default=60)
    c.add_argument("--eps-best", type=float, default=0.1)
    c.add_argument("--eps-pred", type=float, default=1.0)

    t = sub.add_parser("transfer", parents=[common])
    t.add_argument("--source", required=True)
    t.add_argument("--target", type=int, required=True)
    t.add_argument("--out", default="out_transfer")

    a = p.parse_args()
    {"synthetic": cmd_synthetic, "inspect": cmd_inspect,
     "calibrate": cmd_calibrate, "transfer": cmd_transfer}[a.cmd](a)


if __name__ == "__main__":
    main()
