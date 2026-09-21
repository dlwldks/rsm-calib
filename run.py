"""
사용법
  python run.py synthetic                       # 모델 없이 RSM 탐색 로직만 검증
  python run.py inspect   --img-dir .. --msk-dir .. [--checkpoint ..]
  python run.py calibrate --img-dir .. --msk-dir .. --checkpoint .. --targets 5 [6 9 ..] [--paper-mode] --out out_x
  python run.py apply     --img-dir .. --msk-dir .. --checkpoint .. [--encoder mit_b5] --theta-from out_x/result.json --targets 5 --out out_apply
  python run.py transfer  --img-dir .. --msk-dir .. --checkpoint .. --source out_d004/result.json --targets 5 --out out_d067

기준선(--base-mode)
  pooled   : 대상 타일 전체를 합쳐 계산한 채널 평균·표준편차 하나로 정규화 (기본)
  per-tile : 타일마다 자기 통계로 정규화 (논문 '영상별' 기준선의 다른 해석, 보고용 기준선에만 적용)
  given    : --theta0 로 준 6개 값 (예: 모델 카드 권장값)
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
        return {str(k): _json(v) for k, v in o.items()}
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

    res = rsm_search(f, theta0, max_evals=a.max_evals, seed=a.seed, paper_mode=a.paper_mode)
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


def _setup_model(a):
    from data import load_model
    return load_model(a.checkpoint, a.encoder, a.n_classes, a.arch)


def _names(a):
    return a.class_names.split(",") if a.class_names else None


def _cname(names, c):
    return names[c] if names and c < len(names) else str(c)


def _tiles_with(msks, c):
    return np.where([(m == c).any() for m in msks])[0]


def _split(msks, target, test_frac, seed):
    """타깃 클래스가 있는 타일만 골라 calib/test로 분리. test_frac=0이면 전부 calib (논문 방식)."""
    idx = _tiles_with(msks, target)
    np.random.default_rng(seed).shuffle(idx)
    n_test = int(round(len(idx) * test_frac))
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])


def _base_theta(a, imgs):
    """탐색 시작점이자 pooled/given 기준선의 정규화 값."""
    from data import channel_stats
    if a.base_mode == "given":
        if not a.theta0:
            raise SystemExit("--base-mode given 에는 --theta0 값 6개가 필요함 (R G B mean, R G B std)")
        return np.array(a.theta0, float)
    return channel_stats(imgs)


def _base_pred(a, ev, theta0):
    return ev.predict_per_tile() if a.base_mode == "per-tile" else ev.predict(theta0)


def _src_class(src, c):
    """result.json에서 클래스 c의 (theta_base, theta_best, delta). 구버전 형식도 지원."""
    if "classes" in src:
        s = src["classes"].get(str(c))
        if s is None:
            raise SystemExit(f"소스 결과에 클래스 {c}가 없음. 있는 클래스: {list(src['classes'])}")
        return np.array(s["theta_base"]), np.array(s["theta_best"]), s.get("delta")
    return np.array(src["theta_base"]), np.array(src["theta_best"]), None


def _report(msks, splits, cols, n, names, targets, out, fname):
    """splits: {이름: 타일 인덱스}, cols: {열 이름: 전체 타일 예측}. 클래스별 IoU 표 출력·저장."""
    from data import class_iou
    rows, table = [], {}
    labels = list(cols)
    for sname, idx in splits.items():
        if len(idx) == 0:
            continue
        ious = {lab: class_iou(p[idx], msks[idx], n) for lab, p in cols.items()}
        table[sname] = ious
        w = max(16, max(len(l) for l in labels) + 2)
        print(f"\n=== {sname} ({len(idx)} tiles) — IoU % ===")
        print(f"{'class':<22}" + "".join(f"{l:>{w}}" for l in labels))
        for c in range(n):
            if all(np.isnan(ious[k][c]) for k in labels):
                continue
            name = _cname(names, c) + (" *" if c in targets else "")
            print(f"{name:<22}" + "".join(f"{ious[k][c]:>{w}.2f}" for k in labels))
            rows.append([sname, name] + [ious[k][c] for k in labels])
        mi = {k: np.nanmean(v) for k, v in ious.items()}
        print(f"{'mIoU':<22}" + "".join(f"{mi[k]:>{w}.2f}" for k in labels))
    with open(out / fname, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "class"] + labels)
        wr.writerows(rows)
    return table


def _fuse(base_pred, class_preds):
    """class_preds: [(클래스, 예측, 개선폭)]. 개선폭이 큰 클래스가 나중에 덮어써서 우선권을 가짐."""
    from data import decision_fusion
    order = sorted(class_preds, key=lambda t: (t[2] if t[2] is not None else 0.0))
    return decision_fusion(base_pred, [(c, p) for c, p, _ in order]), [c for c, _, _ in order]


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
    from data import Evaluator, class_iou
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = _names(a)
    imgs, msks = _load(a)
    model = _setup_model(a)
    test_frac = 0.0 if a.paper_mode else a.test_frac
    print(f"[mode] {'논문 방식 (분리 없음, 정상점 그대로, OR 종료)' if a.paper_mode else '우리 방식 (calib/test 분리)'}"
          f" / 기준선={a.base_mode}")

    per_class, test_union = {}, set()
    for c in a.targets:
        calib, test = _split(msks, c, test_frac, a.seed)
        print(f"\n##### 클래스 {c} ({_cname(names, c)}): calib={len(calib)} test={len(test)}")
        if len(calib) < 5:
            print("  -> 대상 타일이 5개 미만이라 건너뜀")
            continue
        ev_cal = Evaluator(model, imgs[calib], a.n_classes, a.batch, a.device, a.threads)
        theta0 = _base_theta(a, imgs[calib])
        print("[theta_base]", _fmt(theta0))
        base_iou = class_iou(_base_pred(a, ev_cal, theta0), msks[calib], a.n_classes)[c]

        def objective(theta, ev=ev_cal, idx=calib, cc=c):
            return class_iou(ev.predict(theta), msks[idx], a.n_classes)[cc]

        res = rsm_search(objective, theta0, a.rel_broad, a.n_broad, a.rel_local, a.n_local,
                         a.max_evals, a.eps_best, a.eps_pred, a.seed, paper_mode=a.paper_mode)
        delta = res["iou_best"] - base_iou
        print(f"\n[class {c}] base={base_iou:.2f} best={res['iou_best']:.2f} delta={delta:+.2f}")
        print("[theta_best]", _fmt(res["theta_best"]))
        print("[ratio best/base]", np.round(res["theta_best"] / theta0, 4))
        print("[RSM]", res["rsm"])

        with open(out / f"trials_c{c}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(NAMES + ["iou"])
            w.writerows(np.column_stack([res["X"], res["Y"]]).round(4).tolist())
        per_class[c] = dict(calib=calib, test=test, theta_base=theta0,
                            theta_best=res["theta_best"], theta_rsm=res["theta_rsm"],
                            ratio=res["theta_best"] / theta0, iou_base_calib=base_iou,
                            iou_best_calib=res["iou_best"], delta=delta,
                            rsm=res["rsm"], history=res["history"])
        test_union |= set(test.tolist())

    if not per_class:
        raise SystemExit("최적화한 클래스가 없음")

    # 전체 타일 보고: 기준선 / 클래스별 최적값 / 개선폭 우선 융합
    ev_all = Evaluator(model, imgs, a.n_classes, a.batch, a.device, a.threads)
    theta_all = _base_theta(a, imgs)
    base_pred = _base_pred(a, ev_all, theta_all)
    cols = {"base": base_pred}
    class_preds = []
    for c, r in per_class.items():
        p = ev_all.predict(r["theta_best"])
        cols[f"best(c{c})"] = p
        class_preds.append((c, p, r["delta"]))
    fused, order = _fuse(base_pred, class_preds)
    cols["fusion"] = fused
    print(f"\n[fusion] 우선순위(뒤가 우선): {order}")

    splits = {"all": np.arange(len(imgs))}
    if test_union:
        splits["test(union)"] = np.array(sorted(test_union))
    for c, r in per_class.items():
        splits[f"calib(c{c})"] = r["calib"]
        if len(r["test"]):
            splits[f"test(c{c})"] = r["test"]
    table = _report(msks, splits, cols, a.n_classes, names, list(per_class), out, "iou_table.csv")

    json.dump(_json(dict(args=vars(a), paper_mode=a.paper_mode, base_mode=a.base_mode,
                         fusion_order=order, classes=per_class,
                         iou={s: dict(t) for s, t in table.items()})),
              open(out / "result.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n저장: {out}/result.json, trials_c*.csv, iou_table.csv")


# ------------------------------------------------------------------ apply
def cmd_apply(a):
    """이미 찾은 최적값(절대값)을 그대로 다른 모델/데이터에 적용. 논문의 ResNet34 -> MiT-B5 전이."""
    from data import Evaluator
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = _names(a)
    src = json.load(open(a.theta_from)) if a.theta_from else None
    if not src and not a.theta:
        raise SystemExit("--theta-from 또는 --theta 중 하나가 필요함")
    imgs, msks = _load(a)
    ev = Evaluator(_setup_model(a), imgs, a.n_classes, a.batch, a.device, a.threads)
    theta0 = _base_theta(a, imgs)
    base_pred = _base_pred(a, ev, theta0)
    cols, class_preds, used = {"base": base_pred}, [], {}
    for c in a.targets:
        if a.theta:
            th, delta = np.array(a.theta, float), None
        else:
            _, th, delta = _src_class(src, c)
        print(f"[apply c{c}]", _fmt(th))
        p = ev.predict(th)
        cols[f"apply(c{c})"] = p
        class_preds.append((c, p, delta))
        used[c] = th
    fused, order = _fuse(base_pred, class_preds)
    cols["fusion"] = fused
    splits = {"all": np.arange(len(imgs))}
    for c in a.targets:
        splits[f"tiles(c{c})"] = _tiles_with(msks, c)
    table = _report(msks, splits, cols, a.n_classes, names, a.targets, out, "iou_table.csv")
    json.dump(_json(dict(args=vars(a), theta_base=theta0, theta_applied=used, fusion_order=order,
                         iou={s: dict(t) for s, t in table.items()})),
              open(out / "apply.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n저장: {out}/apply.json, iou_table.csv")


# ------------------------------------------------------------------ transfer
def cmd_transfer(a):
    """다른 도메인(B)의 result.json 비율을 이 도메인(A)에 적용 (재탐색 없음, 논문 식 4)."""
    from data import Evaluator
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = _names(a)
    src = json.load(open(a.source))
    imgs, msks = _load(a)
    ev = Evaluator(_setup_model(a), imgs, a.n_classes, a.batch, a.device, a.threads)
    theta_all = _base_theta(a, imgs)
    base_pred = _base_pred(a, ev, theta_all)
    cols, class_preds, used = {"base": base_pred}, [], {}
    for c in a.targets:
        idx = _tiles_with(msks, c)
        theta_a = _base_theta(a, imgs[idx])
        b_base, b_best, delta = _src_class(src, c)
        theta_t = proportional_transfer(theta_a, b_base, b_best)
        print(f"[c{c}] A base    ", _fmt(theta_a))
        print(f"[c{c}] A pred|B  ", _fmt(theta_t))
        p = ev.predict(theta_t)
        cols[f"transfer(c{c})"] = p
        class_preds.append((c, p, delta))
        used[c] = dict(theta_base=theta_a, theta_transfer=theta_t)
    fused, order = _fuse(base_pred, class_preds)
    cols["fusion"] = fused
    splits = {"all": np.arange(len(imgs))}
    for c in a.targets:
        splits[f"tiles(c{c})"] = _tiles_with(msks, c)
    table = _report(msks, splits, cols, a.n_classes, names, a.targets, out, "iou_table.csv")
    json.dump(_json(dict(source=a.source, classes=used, fusion_order=order,
                         iou={s: dict(t) for s, t in table.items()})),
              open(out / "transfer.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n저장: {out}/transfer.json, iou_table.csv")


# ------------------------------------------------------------------ cli
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synthetic")
    s.add_argument("--noise", type=float, default=0.3)
    s.add_argument("--max-evals", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--paper-mode", action="store_true")

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
    common.add_argument("--base-mode", choices=["pooled", "per-tile", "given"], default="pooled")
    common.add_argument("--theta0", type=float, nargs=6, help="R G B mean, R G B std (--base-mode given)")

    sub.add_parser("inspect", parents=[common])

    c = sub.add_parser("calibrate", parents=[common])
    c.add_argument("--targets", "--target", type=int, nargs="+", required=True, help="모델 클래스 인덱스 (여러 개 가능)")
    c.add_argument("--out", default="out")
    c.add_argument("--paper-mode", action="store_true", help="논문 Figure 1 방식 그대로")
    c.add_argument("--test-frac", type=float, default=0.3)
    c.add_argument("--rel-broad", type=float, default=0.10)
    c.add_argument("--n-broad", type=int, default=30)
    c.add_argument("--rel-local", type=float, default=0.03)
    c.add_argument("--n-local", type=int, default=6)
    c.add_argument("--max-evals", type=int, default=60)
    c.add_argument("--eps-best", type=float, default=0.1)
    c.add_argument("--eps-pred", type=float, default=1.0)

    ap = sub.add_parser("apply", parents=[common])
    ap.add_argument("--targets", "--target", type=int, nargs="+", required=True)
    ap.add_argument("--theta-from", help="calibrate가 만든 result.json")
    ap.add_argument("--theta", type=float, nargs=6, help="직접 지정할 6개 값")
    ap.add_argument("--out", default="out_apply")

    t = sub.add_parser("transfer", parents=[common])
    t.add_argument("--source", required=True)
    t.add_argument("--targets", "--target", type=int, nargs="+", required=True)
    t.add_argument("--out", default="out_transfer")

    a = p.parse_args()
    {"synthetic": cmd_synthetic, "inspect": cmd_inspect, "calibrate": cmd_calibrate,
     "apply": cmd_apply, "transfer": cmd_transfer}[a.cmd](a)


if __name__ == "__main__":
    main()