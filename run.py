"""
사용법
  python run.py synthetic [--paper-mode]         # 모델 없이 RSM 탐색 로직만 검증
  python run.py inspect   --img-dir .. --msk-dir .. [--checkpoint ..]
  python run.py calibrate --img-dir .. --msk-dir .. --checkpoint .. --targets 5 [6 9 ..] [--paper-mode] --out out_x
  python run.py broad     --img-dir .. --msk-dir .. --checkpoint .. --targets 5 --adjs 0.1 0.25 0.5 --out out_broad
  python run.py apply     --img-dir .. --msk-dir .. --checkpoint .. [--encoder mit_b5] --theta-from out_x/result.json --targets 5 --out out_apply
  python run.py transfer  --img-dir .. --msk-dir .. --checkpoint .. --source out_d004/result.json --targets 5 --out out_d067

탐색 모드
  --paper-mode : 논문 Figure 1 그대로. 기본 adj0=0.5(n0=30) -> adj_t=0.15(n_t=5), 경계 없음,
                 eps1(회차 best 개선폭) <= eps_stop OR eps2(|회차 best - 예측|) <= eps_stop 이면 종료
  (기본)       : 우리 변형. theta0 ±10% 상자 고정 -> ±3% 7개씩, AND 종료

기준선(--base-mode)
  pooled   : 대상 타일 전체를 합쳐 계산한 채널 평균·표준편차 하나로 정규화 (기본)
  per-tile : 타일마다 자기 통계로 정규화 (논문 '영상별' 기준선의 한 해석, 보고용 기준선에만 적용)
  given    : --theta0 로 준 6개 값 (예: 모델 카드 권장값)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rsm import n_min, proportional_transfer, quad_features
from search import NAMES, broad_sweep, rsm_search


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
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _fmt(theta):
    return ", ".join(f"{n}={v:.2f}" for n, v in zip(NAMES, theta))


def _search_kwargs(a):
    return dict(paper_mode=a.paper_mode, rel_broad=a.rel_broad, n_broad=a.n_broad,
                rel_local=a.rel_local, n_local=a.n_local, max_evals=a.max_evals,
                eps_best=a.eps_best, eps_pred=a.eps_pred, eps_stop=a.eps_stop,
                r2_min=a.r2_min, seed=a.seed,
                fit=a.fit, ridge_lambda=a.ridge_lambda, move=a.move)


def _mode_str(a):
    if a.paper_mode:
        return (f"논문 Figure 1 (adj0={0.5 if a.rel_broad is None else a.rel_broad}, "
                f"adj_t={0.15 if a.rel_local is None else a.rel_local}, "
                f"n_t={5 if a.n_local is None else a.n_local}, eps_stop={a.eps_stop}, fit={a.fit}, move={a.move}, 경계 없음)")
    return (f"우리 변형 (±{0.10 if a.rel_broad is None else a.rel_broad} 상자, "
            f"±{0.03 if a.rel_local is None else a.rel_local}, AND 종료)")


# ------------------------------------------------------------------ synthetic
def cmd_synthetic(a):
    rng = np.random.default_rng(a.seed)
    theta0 = np.array([97.28, 108.99, 109.48, 62.47, 56.64, 54.35])       # D067 base
    true = theta0 * np.array([0.98, 1.05, 0.95, 1.00, 0.99, 0.99])         # R↓ G↑ B↓
    width = theta0 * 0.06

    def f(t):
        z = (t - true) / width
        return 80 - 2.0 * np.sum(z ** 2) + 0.5 * z[0] * z[1] + rng.normal(0, a.noise)

    print("[mode]", _mode_str(a))
    res = rsm_search(f, theta0, **_search_kwargs(a))
    print("\n정답 최적점 :", _fmt(true))
    print("탐색 best   :", _fmt(res["theta_best"]))
    print("RSM 정상점  :", _fmt(res["theta_rsm"]))
    print(f"IoU {res['iou_base']:.2f} -> {res['iou_best']:.2f}  "
          f"(evals={len(res['Y'])}, stop={res['stop_reason']})")
    print("RSM 통계    :", {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in res["rsm"].items()})


# ------------------------------------------------------------------ shared
def _load(a):
    from data import build_lut, load_tiles, pair_files
    pairs = pair_files(a.img_dir, a.msk_dir, a.img_glob, a.msk_glob)
    if not pairs:
        raise SystemExit("IMG/MSK 쌍을 못 찾음. --img-glob/--msk-glob 확인")
    if a.include:
        pairs = [p for p in pairs if a.include in str(p[1].parent)]
    if a.exclude:
        pairs = [p for p in pairs if not any(x in p[1].stem for x in a.exclude)]
    if not pairs:
        raise SystemExit("--include/--exclude 적용 후 남은 타일이 없음")
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


def _split(msks, target, test_frac, seed, calib_tiles="target"):
    """calib_tiles=target: 타깃 클래스가 있는 타일만 / all: 전체 타일 (논문 E2는 47타일 전부).
    test_frac>0이면 그 비율만큼 test로 분리. test_frac=0이면 전부 calib (논문 방식)."""
    idx = _tiles_with(msks, target) if calib_tiles == "target" else np.arange(len(msks))
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


def _write_trials(path, X, Y, it=None):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(NAMES + ["iou"] + (["iter"] if it is not None else []))
        cols = [X, Y] + ([it] if it is not None else [])
        w.writerows(np.column_stack(cols).round(4).tolist())


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
def _reuse_wrap(objective, path, tol=1e-3):
    """이전 실행 trials csv(소수 4자리 저장)에 같은 theta(차이 tol 이하)가 있으면 그 IoU를 돌려줌.
    재시작 등으로 끊긴 탐색을 다시 돌릴 때 앞부분 추론을 건너뛰는 용도. 조건(데이터·클래스·seed)이 같아야 함."""
    import pandas as pd
    d = pd.read_csv(path)
    X, Y = d[NAMES].to_numpy(float), d["iou"].to_numpy(float)

    def wrapped(theta):
        diff = np.abs(X - np.asarray(theta, float)).max(1)
        i = int(np.argmin(diff))
        if diff[i] <= tol:
            return float(Y[i])
        return objective(theta)
    return wrapped


def cmd_calibrate(a):
    from data import Evaluator, class_iou
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = _names(a)
    imgs, msks = _load(a)
    model = _setup_model(a)
    test_frac = 0.0 if a.paper_mode else a.test_frac
    print(f"[mode] {_mode_str(a)} / 기준선={a.base_mode} / 보정 타일={a.calib_tiles} / "
          f"{'calib/test 분리 없음' if test_frac == 0 else f'test {test_frac:.0%} 분리'}")

    per_class, test_union = {}, set()
    for c in a.targets:
        calib, test = _split(msks, c, test_frac, a.seed, a.calib_tiles)
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

        if a.reuse:  # 같은 조건의 이전 trials csv에서 같은 theta는 추론 없이 기록값 사용
            objective = _reuse_wrap(objective, a.reuse)

        res = rsm_search(objective, theta0, **_search_kwargs(a))
        delta = res["iou_best"] - base_iou
        print(f"\n[class {c}] base={base_iou:.2f} best={res['iou_best']:.2f} delta={delta:+.2f} "
              f"(evals={len(res['Y'])}, stop={res['stop_reason']})")
        print("[theta_best]", _fmt(res["theta_best"]))
        print("[ratio best/base]", np.round(res["theta_best"] / theta0, 4))
        print("[RSM]", res["rsm"])

        _write_trials(out / f"trials_c{c}.csv", res["X"], res["Y"], res["iteration"])
        per_class[c] = dict(calib=calib, test=test, theta_base=theta0,
                            theta_best=res["theta_best"], theta_rsm=res["theta_rsm"],
                            ratio=res["theta_best"] / theta0, iou_base_calib=base_iou,
                            iou_best_calib=res["iou_best"], delta=delta,
                            rsm=res["rsm"], history=res["history"],
                            stop_reason=res["stop_reason"], settings=res["settings"])
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


# ------------------------------------------------------------------ broad
def _fit_stats(X, y, kind):
    """kind: linear(7) / quad_diag(13) / quad(28). R2, Adj.R2, LOOCV RMSE."""
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(0)) / sd
    if kind == "quad":
        A = quad_features(Z)
    else:
        A = np.column_stack([np.ones(len(Z)), Z] + ([Z ** 2] if kind == "quad_diag" else []))
    n, p = A.shape
    if n <= p:
        return dict(p=p, r2=np.nan, adj_r2=np.nan, loo=np.nan)
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ b
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(r @ r) / tss if tss > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / (n - p)
    h = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
    loo = float(np.sqrt(np.mean((r / (1 - h)) ** 2))) if np.all(h < 1 - 1e-6) else np.nan
    return dict(p=p, r2=r2, adj_r2=adj, loo=loo)


def cmd_broad(a):
    """1단계(Iteration_0)만 adj0별로 돌려 IoU 분포와 피팅 적합도를 비교."""
    from data import Evaluator, class_iou
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs, msks = _load(a)
    model = _setup_model(a)
    summary = []
    for c in a.targets:
        idx = _tiles_with(msks, c)
        print(f"\n##### 클래스 {c}: 대상 타일 {len(idx)}개, adj0={a.adjs}, 각 {a.n_broad}회")
        ev = Evaluator(model, imgs[idx], a.n_classes, a.batch, a.device, a.threads)
        theta0 = _base_theta(a, imgs[idx])
        print("[theta_base]", _fmt(theta0))

        def objective(theta, cc=c):
            return class_iou(ev.predict(theta), msks[idx], a.n_classes)[cc]

        runs = broad_sweep(objective, theta0, a.adjs, a.n_broad, a.seed)
        for adj, (X, Y) in runs.items():
            _write_trials(out / f"trials_c{c}_adj{adj:g}.csv", X, Y)
            i = int(np.argmax(Y))
            row = dict(target=c, adj0=adj, n=len(Y), iou_base=Y[0], iou_min=Y.min(),
                       iou_median=float(np.median(Y)), iou_max=Y.max(),
                       n_below5=int((Y < 5).sum()), ratio_best=np.round(X[i] / theta0, 4).tolist())
            for k in ("linear", "quad_diag", "quad"):
                s = _fit_stats(X, Y, k)
                row.update({f"{k}_r2": s["r2"], f"{k}_adj_r2": s["adj_r2"], f"{k}_loo": s["loo"]})
            summary.append(row)

    print(f"\n{'c':>3} {'adj0':>5} {'base':>6} {'min':>6} {'med':>6} {'max':>6} {'<5':>3} "
          f"{'lin R2':>7} {'lin LOO':>8} {'qdiag R2':>9} {'qdiag LOO':>10} {'quad R2':>8} {'quadAdj':>8}")
    for r in summary:
        print(f"{r['target']:>3} {r['adj0']:>5.2f} {r['iou_base']:>6.2f} {r['iou_min']:>6.2f} "
              f"{r['iou_median']:>6.2f} {r['iou_max']:>6.2f} {r['n_below5']:>3} "
              f"{r['linear_r2']:>7.3f} {r['linear_loo']:>8.2f} {r['quad_diag_r2']:>9.3f} "
              f"{r['quad_diag_loo']:>10.2f} {r['quad_r2']:>8.3f} {r['quad_adj_r2']:>8.3f}")
    print(f"\n참고: 2차(quad) 계수 {n_min(6)}개 / 관측 {a.n_broad}개 -> 잔차 자유도 {a.n_broad - n_min(6)}. "
          "quad R2는 거의 1이 나오므로 비교는 LOO와 분포로 할 것.")
    with open(out / "broad_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"저장: {out}/broad_summary.csv, trials_c*_adj*.csv")


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
def _add_search_args(p):
    p.add_argument("--paper-mode", action="store_true", help="논문 Figure 1 방식 그대로")
    p.add_argument("--rel-broad", "--adj0", type=float, default=None,
                   help="1단계 폭 (상대 비율). 기본: 논문 0.5 / 우리 0.10")
    p.add_argument("--n-broad", "--n0", type=int, default=30)
    p.add_argument("--rel-local", "--adj-t", type=float, default=None,
                   help="2단계 폭 (상대 비율). 기본: 논문 0.15 / 우리 0.03")
    p.add_argument("--n-local", "--n-t", type=int, default=None,
                   help="회차당 추론 수. 기본: 논문 5 / 우리 6(+정상점 1)")
    p.add_argument("--max-evals", type=int, default=None, help="추론 상한. 기본: 논문 80 / 우리 60")
    p.add_argument("--r2-min", type=float, default=None,
                   help="논문 모드: (e)를 만족해도 R2가 이 값 미만이면 계속 (본문 3.3절, 기본 꺼짐)")
    p.add_argument("--eps-stop", type=float, default=0.0, help="논문 모드 종료 임계값 (Figure 1: ≈0)")
    p.add_argument("--fit", choices=["ols", "ridge"], default="ols",
                   help="논문 모드 계수 추정: ols(기본) / ridge(회차마다 LOO로 λ 선택)")
    p.add_argument("--ridge-lambda", type=float, default=None, help="ridge λ 고정값 (기본: LOO 선택)")
    p.add_argument("--move", choices=["stationary", "boxmax"], default="stationary",
                   help="논문 모드 이동: stationary(논문) / boxmax(정상점이 극대가 아니면 1단계 범위 안 최대점, 우리 변형)")
    p.add_argument("--reuse", default=None, help="이전 trials csv: 같은 theta는 추론 없이 기록값 사용 (끊긴 탐색 재개용)")
    p.add_argument("--eps-best", type=float, default=0.1, help="우리 모드 종료 임계값 1")
    p.add_argument("--eps-pred", type=float, default=1.0, help="우리 모드 종료 임계값 2")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synthetic")
    s.add_argument("--noise", type=float, default=0.3)
    s.add_argument("--seed", type=int, default=0)
    _add_search_args(s)

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
    common.add_argument("--include", help="마스크 경로(폴더)에 이 문자열이 있는 타일만 사용. 예: test")
    common.add_argument("--exclude", nargs="+", help="파일명에 이 문자열이 있는 타일 제외 (오라벨 타일 등)")
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
    c.add_argument("--test-frac", type=float, default=0.3)
    c.add_argument("--calib-tiles", choices=["target", "all"], default="target",
                   help="target: 대상 클래스가 있는 타일만 (Table 3 정의) / all: 전체 타일 (논문 E2)")
    _add_search_args(c)

    b = sub.add_parser("broad", parents=[common])
    b.add_argument("--targets", "--target", type=int, nargs="+", required=True)
    b.add_argument("--adjs", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    b.add_argument("--n-broad", type=int, default=30)
    b.add_argument("--out", default="out_broad")

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
     "broad": cmd_broad, "apply": cmd_apply, "transfer": cmd_transfer}[a.cmd](a)


if __name__ == "__main__":
    main()
