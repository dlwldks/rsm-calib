"""상한 확인 (진단용): 2차식 없이 theta 6개를 직접 최적화해, 이 모델·이 타일로 도달 가능한 IoU를 본다.

논문 방법이 아니다. U-Net 가중치는 고정하고 theta(R·G·B 평균, R·G·B 표준편차)만 바꾸는 것은 같지만,
2차식 탐색 대신 경사하강(Adam)으로 theta를 직접 올린다.

- IoU는 argmax 때문에 미분이 안 되므로, 학습 방향은 대상 클래스의 soft IoU(softmax 확률로 계산)로 잡는다.
- 매 단계 실제 IoU(argmax, 타일 픽셀 합산)를 따로 계산해 기록하고, 최종 결과도 실제 IoU 최대값이다.
- 1단계 = 기울기 없는 추론 1회(실제 IoU·soft IoU) + 배치별 역전파 1회. 메모리는 배치 크기만큼만 쓴다. --steps 로 정한다.
- 시작점을 여러 개(--restarts) 둘 수 있다. 0번은 theta0, 나머지는 theta0 ±rel 범위 무작위.

조건은 E2와 같음: test 50타일(--include test), 19채널 라벨맵, 대상 클래스 5(침엽수).

사용 (C:\\rsm_calib 에서):
    python analysis/theta_upper.py --img-dir flair_1_toy_dataset\\flair_1_toy_dataset ^
        --msk-dir flair_1_toy_dataset\\flair_1_toy_dataset ^
        --checkpoint models\\rgb15\\FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth ^
        --out out_upper_c5

출력: <out>/trace.csv (단계별 theta, soft IoU, 실제 IoU), <out>/result.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import build_lut, channel_stats, class_iou, load_model, load_tiles, pair_files  # noqa: E402

PARAMS = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]


def load(a):
    pairs = pair_files(a.img_dir, a.msk_dir)
    if a.include:
        pairs = [p for p in pairs if a.include in str(p[1].parent)]
    if a.exclude:
        pairs = [p for p in pairs if not any(x in p[1].stem for x in a.exclude)]
    if a.limit:
        pairs = pairs[: a.limit]
    if not pairs:
        raise SystemExit("타일을 못 찾음. --img-dir/--msk-dir/--include 확인")
    imgs, msks = load_tiles(pairs, build_lut(a.label_map, a.n_classes))
    print(f"[data] tiles={len(imgs)} img={imgs.shape[1:]}")
    return imgs, msks


def run(a, model, imgs, msks, theta_start, tag, log):
    import torch
    t = torch
    target = a.target
    X = t.from_numpy(imgs).float()
    G = t.from_numpy((msks == target).astype(np.float32))
    valid = t.from_numpy((msks != 255).astype(np.float32))  # IGNORE(255) 픽셀 제외

    # 평균은 그대로, 표준편차는 log로 두어 항상 양수
    mean = t.tensor(theta_start[:3], dtype=t.float32, requires_grad=True)
    logstd = t.tensor(np.log(theta_start[3:]), dtype=t.float32, requires_grad=True)
    # 평균(원 단위)과 표준편차(log 단위)는 단위가 달라 학습률을 따로 둔다
    opt = t.optim.Adam([{"params": [mean], "lr": a.lr}, {"params": [logstd], "lr": a.lr_std}])
    th0 = np.asarray(theta_start, float)
    if a.box > 0:  # theta0 ±box 안으로 제한 (논문 Figure 5 탐색 범위 약 ±5~14%)
        lo_m, hi_m = th0[:3] * (1 - a.box), th0[:3] * (1 + a.box)
        lo_s, hi_s = np.log(th0[3:] * (1 - a.box)), np.log(th0[3:] * (1 + a.box))
    else:          # 물리 범위만: 평균 0~255, 표준편차 1~255
        lo_m, hi_m = np.zeros(3), np.full(3, 255.0)
        lo_s, hi_s = np.zeros(3), np.full(3, np.log(255.0))
    lo_m, hi_m, lo_s, hi_s = (t.tensor(v, dtype=t.float32) for v in (lo_m, hi_m, lo_s, hi_s))

    def theta_now():
        return np.concatenate([mean.detach().numpy(), np.exp(logstd.detach().numpy())])

    def soft_terms(p, i):
        g, v = G[i:i + a.batch], valid[i:i + a.batch]
        return (p * g * v).sum(), ((p + g - p * g) * v).sum()

    best = (-1.0, None, -1)
    for step in range(a.steps + 1):
        th = theta_now()
        m = t.tensor(th[:3], dtype=t.float32).view(1, 3, 1, 1)
        s = t.tensor(th[3:], dtype=t.float32).view(1, 3, 1, 1)

        # 1차 통과 (기울기 없음): 실제 IoU(argmax)와 soft IoU 합계(I, U)를 같이 계산
        pred = np.empty(msks.shape, np.uint8)
        I = U = 0.0
        with t.inference_mode():
            for i in range(0, len(X), a.batch):
                out = model((X[i:i + a.batch] - m) / s)
                pred[i:i + a.batch] = out.argmax(1).numpy()
                ii, uu = soft_terms(t.softmax(out, 1)[:, target], i)
                I, U = I + ii.item(), U + uu.item()
        iou = float(class_iou(pred, msks, a.n_classes)[target])
        soft = I / max(U, 1.0)
        if iou > best[0]:
            best = (iou, th.copy(), step)
        log.append(dict(run=tag, step=step, soft_iou=soft * 100, iou=iou, **dict(zip(PARAMS, th))))
        print(f"[{tag}] step {step:3d} soft={soft * 100:6.2f} iou={iou:6.2f} (best {best[0]:.2f} @ {best[2]})",
              flush=True)
        if step == a.steps:
            break

        # 2차 통과 (기울기): 배치마다 역전파해서 메모리를 배치 크기만큼만 사용.
        # soft = I/U 의 기울기 = (dI·U − I·dU)/U² 를 배치별로 나눠 더함 (전체 한 번에 한 것과 같은 값)
        opt.zero_grad()
        for i in range(0, len(X), a.batch):
            std = t.exp(logstd).view(1, 3, 1, 1)
            x = (X[i:i + a.batch] - mean.view(1, 3, 1, 1)) / std
            ii, uu = soft_terms(t.softmax(model(x), 1)[:, target], i)
            loss = -(ii * U - I * uu) / max(U, 1.0) ** 2
            loss.backward()
        opt.step()
        with t.no_grad():
            mean.copy_(t.max(t.min(mean, hi_m), lo_m))
            logstd.copy_(t.max(t.min(logstd, hi_s), lo_s))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--msk-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--label-map", default="labelmap_19.json")
    ap.add_argument("--n-classes", type=int, default=19)
    ap.add_argument("--include", default="test")
    ap.add_argument("--exclude", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--target", type=int, default=5, help="모델 클래스 인덱스 (5 = 침엽수)")
    ap.add_argument("--steps", type=int, default=40, help="시작점 1개당 경사 단계 수")
    ap.add_argument("--lr", type=float, default=1.0, help="평균의 Adam 학습률 (원 단위, 한 단계 최대 약 1)")
    ap.add_argument("--lr-std", type=float, default=0.01, help="표준편차의 Adam 학습률 (log 단위, 한 단계 최대 약 1%%)")
    ap.add_argument("--box", type=float, default=0.10, help="theta0 ±box 안으로 제한. 0이면 물리 범위(0~255)만")
    ap.add_argument("--restarts", type=int, default=1, help="시작점 개수 (0번 = theta0)")
    ap.add_argument("--rel", type=float, default=0.25, help="추가 시작점 범위 (theta0 ±rel)")
    ap.add_argument("--batch", type=int, default=4, help="역전파 배치. 메모리가 부족하면 2로")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out_upper_c5")
    a = ap.parse_args()

    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    imgs, msks = load(a)
    model = load_model(a.checkpoint, a.encoder, a.n_classes)
    for p in model.parameters():  # 가중치 고정
        p.requires_grad_(False)

    theta0 = channel_stats(imgs)
    print("[theta0]", np.round(theta0, 2))
    rng = np.random.default_rng(a.seed)
    starts = [theta0] + [theta0 * (1 + rng.uniform(-a.rel, a.rel, 6)) for _ in range(a.restarts - 1)]

    log, results = [], []
    t0 = time.time()
    for k, th in enumerate(starts):
        iou, th_best, step = run(a, model, imgs, msks, th, f"r{k}", log)
        results.append(dict(run=f"r{k}", theta_start=th, iou_best=iou, theta_best=th_best, step=step))

    import pandas as pd
    pd.DataFrame(log).to_csv(out / "trace.csv", index=False)
    top = max(results, key=lambda r: r["iou_best"])
    summary = dict(
        note="진단용 상한 확인. 논문 방법 아님 (2차식 대신 soft IoU 경사하강)",
        args=vars(a), theta0=theta0.tolist(), iou_best=top["iou_best"], theta_best=top["theta_best"].tolist(),
        ratio_best_over_theta0=(top["theta_best"] / theta0).tolist(),
        runs=[dict(run=r["run"], iou_best=r["iou_best"], step=r["step"],
                   theta_start=r["theta_start"].tolist(), theta_best=r["theta_best"].tolist()) for r in results],
        minutes=round((time.time() - t0) / 60, 1))
    json.dump(summary, open(out / "result.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[best] IoU={top['iou_best']:.2f} ({top['run']}) theta={np.round(top['theta_best'], 2)}")
    print(f"저장: {out}/trace.csv, result.json ({summary['minutes']}분)")


if __name__ == "__main__":
    main()
