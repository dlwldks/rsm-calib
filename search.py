"""Two-stage sequential RSM search (논문 Figure 1)."""
import numpy as np

from rsm import QuadraticRSM, n_min

NAMES = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]


def lhs(n, d, rng):
    """Latin hypercube 샘플 [0,1]^d. 랜덤보다 반응면을 고르게 덮음."""
    pts = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        pts[:, j] = pts[rng.permutation(n), j]
    return pts


def rsm_search(objective, theta0, rel_broad=0.10, n_broad=30, rel_local=0.03,
               n_local=6, max_evals=60, eps_best=0.1, eps_pred=1.0, seed=0,
               paper_mode=False, verbose=True):
    """
    objective(theta) -> IoU(%).  theta = [R_mean, G_mean, B_mean, R_std, G_std, B_std]

    1단계: theta0 기준 ±rel_broad 범위를 LHS로 n_broad번 샘플링
    2단계: 2차 회귀 -> 다음 탐색 중심 결정 -> 그 주변 ±rel_local에서 n_local번 추가 샘플 -> 재피팅

    paper_mode=False (기본, 우리 방식)
      - 다음 중심: 정상점이 탐색 범위 안의 극대점일 때만 사용, 아니면 범위 내 곡면 최댓값
      - 종료: (best 개선폭 <= eps_best) AND (|실제 IoU - 예측 IoU| <= eps_pred)
    paper_mode=True (논문 Figure 1 그대로)
      - 다음 중심: 정상점(grad f = 0)을 종류와 관계없이 그대로 사용 (범위 밖이면 경계로 자름)
      - 종료: 조건 (e1) OR (e2)
    공통: max_evals에 도달하면 종료
    """
    rng = np.random.default_rng(seed)
    theta0 = np.asarray(theta0, float)
    d = len(theta0)
    n_broad = max(n_broad, n_min(d) + 2)
    lo, hi = theta0 * (1 - rel_broad), theta0 * (1 + rel_broad)
    X, Y = [], []

    def ev(t, tag):
        t = np.clip(t, lo, hi)
        y = float(objective(t))
        X.append(t)
        Y.append(y)
        if verbose:
            print(f"[{len(Y):3d}] {tag:<24} IoU={y:6.2f}  best={max(Y):6.2f}", flush=True)
        return y

    ev(theta0, "base")
    for u in lhs(n_broad - 1, d, rng):
        ev(lo + u * (hi - lo), "broad")

    best_prev, history = max(Y), []
    while len(Y) < max_evals:
        model = QuadraticRSM().fit(np.array(X), np.array(Y))
        if paper_mode:
            x_hat, kind = model.stationary_point()
            how = f"stationary-{kind}"
        else:
            x_hat, how = model.maximize(lo, hi)
        x_hat = np.clip(x_hat, lo, hi)
        y_pred = float(model.predict(x_hat)[0])
        y_hat = ev(x_hat, f"rsm({how})")
        for u in lhs(n_local, d, rng):
            if len(Y) >= max_evals:
                break
            ev(x_hat * (1 + rel_local * (2 * u - 1)), "local")
        best = max(Y)
        e1, e2 = best - best_prev, abs(y_hat - y_pred)
        history.append(dict(evals=len(Y), iou_pred=y_pred, iou_at_pred=y_hat,
                            best=best, eps1=e1, eps2=e2, r2=model.r2, how=how))
        if verbose:
            print(f"      -> pred={y_pred:.2f} actual={y_hat:.2f} "
                  f"eps1={e1:.3f} eps2={e2:.3f} R2={model.r2:.3f}")
        if paper_mode:
            stop = e1 <= eps_best or e2 <= eps_pred
        else:
            stop = e1 <= eps_best and e2 <= eps_pred
        if stop:
            break
        best_prev = best

    final = QuadraticRSM().fit(np.array(X), np.array(Y))
    if paper_mode:
        theta_rsm = np.clip(final.stationary_point()[0], lo, hi)
    else:
        theta_rsm = final.maximize(lo, hi)[0]
    i = int(np.argmax(Y))
    return dict(theta_base=theta0, theta_best=X[i], iou_base=Y[0], iou_best=Y[i],
                theta_rsm=theta_rsm, X=np.array(X), Y=np.array(Y),
                rsm=final.summary(), history=history, bounds=(lo, hi),
                paper_mode=paper_mode)