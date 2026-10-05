"""RSM 순차 탐색.

rsm_search_paper : 논문 Figure 1 그대로 (--paper-mode)
rsm_search_ours  : 우리 변형 (탐색 상자 고정, 상자 안 곡면 최댓값, AND 종료)
broad_sweep      : 1단계(Iteration_0)만 탐색 폭별로 돌려 비교
"""
import numpy as np

from rsm import QuadraticRSM, n_min

NAMES = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]

# 물리적 범위. 논문에는 없는 우리 안전장치 (경계 없는 탐색에서 std<=0 같은 값 방지)
MEAN_RANGE = (0.0, 255.0)
STD_RANGE = (1.0, 255.0)


def lhs(n, d, rng):
    """Latin hypercube 샘플 [0,1]^d. 랜덤보다 반응면을 고르게 덮음."""
    pts = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        pts[:, j] = pts[rng.permutation(n), j]
    return pts


def physical_bounds(d):
    b = d // 2
    lo = np.array([MEAN_RANGE[0]] * b + [STD_RANGE[0]] * (d - b))
    hi = np.array([MEAN_RANGE[1]] * b + [STD_RANGE[1]] * (d - b))
    return lo, hi


def physical_clip(theta):
    lo, hi = physical_bounds(len(theta))
    return np.clip(np.asarray(theta, float), lo, hi)


def neighborhood(center, adj, n, rng, include_center=True):
    """Figure 1의 Neighborhood(theta; ±adj, n).

    adj는 center 대비 상대 비율: theta = center * (1 + adj * (2u - 1)), u ~ LHS[0,1]^d.
    include_center=True면 첫 점이 center 자신 (Figure 1의 r=0).
    """
    center = np.asarray(center, float)
    pts = [center.copy()] if include_center else []
    k = n - len(pts)
    if k > 0:
        for u in lhs(k, len(center), rng):
            pts.append(center * (1 + adj * (2 * u - 1)))
    return [physical_clip(p) for p in pts]


# ------------------------------------------------------------------ 논문 방식
def rsm_search_paper(objective, theta0, adj0=0.5, n0=30, adj_t=0.15, n_t=5,
                     eps_stop=0.0, max_evals=80, r2_min=None, seed=0, verbose=True,
                     fit="ols", ridge_lambda=None, move="stationary"):
    """논문 Figure 1.

    move: "stationary"(논문, 기본) / "boxmax"(우리 변형: 정상점이 극대가 아니면
          1단계 범위 theta0*(1±adj0) 안에서 식이 가장 높은 점으로 이동).
    fit: 계수 추정 방법. "ols"(최소제곱, 기본) / "ridge"(회차마다 LOO로 λ 선택, ridge_lambda 주면 고정).

    (b1) Iteration_0 : C_0 = Neighborhood(theta0; ±adj0, n0)  -> 추론, 로그 S에 추가
    (f)(g)           : S 전체로 2차식 피팅 -> grad f = 0 인 정상점 theta_hat, IoU_pred = f(theta_hat)
    (bt) Iteration_t : C_t = Neighborhood(theta_hat; ±adj_t, n_t) -> 추론, S에 추가
    (d1) eps1 = IoU_best^t - IoU_best^(t-1)      (각 회차 안에서의 best)
    (d2) eps2 = |IoU_best^t - IoU_pred^t|
    (e)  eps1 <= eps_stop  OR  eps2 <= eps_stop  이면 종료
    출력  theta_opt = S 전체에서 IoU 최대인 theta

    r2_min (선택, 기본 꺼짐): 본문 3.3절 "R2 >= 0.9가 될 때까지 반복"을 반영하는 옵션.
      켜면 (e) 조건을 만족해도 피팅 R2 < r2_min 이면 종료하지 않고 계속한다.

    논문에 없는 것 (우리가 정한 값):
      - adj는 상대 비율, 샘플은 LHS
      - 정상점과 샘플을 물리적 범위(mean 0~255, std 1~255)로 자름 (잘렸으면 history에 clipped=True)
      - max_evals 상한 (Figure 1 (h)가 약 70회까지 진행되어 기본 80)
    """
    rng = np.random.default_rng(seed)
    theta0 = np.asarray(theta0, float)
    d = len(theta0)
    n0 = max(n0, n_min(d) + 2)
    box_lo = physical_clip(theta0 * (1 - adj0))
    box_hi = physical_clip(theta0 * (1 + adj0))
    X, Y, IT = [], [], []

    def run_batch(cands, t):
        ys = []
        for k, th in enumerate(cands):
            if len(Y) >= max_evals:
                break
            y = float(objective(th))
            X.append(th)
            Y.append(y)
            IT.append(t)
            ys.append(y)
            if verbose:
                tag = "center" if k == 0 else "neighbor"
                print(f"[{len(Y):3d}] iter{t:<2d} {tag:<9} IoU={y:6.2f}  best={max(Y):6.2f}",
                      flush=True)
        return ys

    ys = run_batch(neighborhood(theta0, adj0, n0, rng), 0)
    best_prev = max(ys)
    history, stop_reason, t = [], "max_evals", 0

    while len(Y) < max_evals:
        t += 1
        model = QuadraticRSM().fit(np.array(X), np.array(Y), fit, ridge_lambda)
        theta_raw, kind = model.stationary_point()
        if move == "boxmax" and kind != "max":
            theta_raw, kind = model.maximize(box_lo, box_hi)
        theta_hat = physical_clip(theta_raw)
        clipped = bool(not np.allclose(theta_raw, theta_hat))
        y_pred = float(model.predict(theta_hat)[0])
        ys = run_batch(neighborhood(theta_hat, adj_t, n_t, rng), t)
        if not ys:
            break
        best_t = max(ys)
        e1, e2 = best_t - best_prev, abs(best_t - y_pred)
        history.append(dict(iter=t, evals=len(Y), stationary_kind=kind, theta_hat=theta_hat,
                            theta_hat_raw=theta_raw, clipped=clipped,
                            iou_pred=y_pred, iou_best_iter=best_t, eps1=e1, eps2=e2,
                            r2=model.r2, adj_r2=model.adj_r2,
                            ridge_lambda=getattr(model, 'lam', None)))
        if verbose:
            print(f"      -> [{kind}{', clipped' if clipped else ''}] pred={y_pred:.2f} "
                  f"iter_best={best_t:.2f} eps1={e1:+.3f} eps2={e2:.3f} R2={model.r2:.3f}"
                  + (f" λ={model.lam:.3g}" if fit == "ridge" else ""))
        if e1 <= eps_stop or e2 <= eps_stop:
            if r2_min is not None and model.r2 < r2_min:
                if verbose:
                    print(f"      -> (e) 만족했지만 R2 {model.r2:.3f} < {r2_min} 라서 계속")
            else:
                stop_reason = "e1" if e1 <= eps_stop else "e2"
                break
        best_prev = best_t

    final = QuadraticRSM().fit(np.array(X), np.array(Y), fit, ridge_lambda)
    i = int(np.argmax(Y))
    return dict(theta_base=theta0, theta_best=X[i], iou_base=Y[0], iou_best=Y[i],
                theta_rsm=physical_clip(final.stationary_point()[0]),
                X=np.array(X), Y=np.array(Y), iteration=np.array(IT),
                rsm=final.summary(), history=history, bounds=physical_bounds(d),
                paper_mode=True, stop_reason=stop_reason,
                settings=dict(adj0=adj0, n0=n0, adj_t=adj_t, n_t=n_t, eps_stop=eps_stop,
                              max_evals=max_evals, r2_min=r2_min, seed=seed,
                              fit=fit, ridge_lambda=ridge_lambda, move=move))


# ------------------------------------------------------------------ 우리 방식
def rsm_search_ours(objective, theta0, rel_broad=0.10, n_broad=30, rel_local=0.03,
                    n_local=6, max_evals=60, eps_best=0.1, eps_pred=1.0, seed=0,
                    verbose=True):
    """탐색 상자(theta0 ±rel_broad)를 고정하고, 상자 안 곡면 최댓값으로 이동.
    종료: (누적 best 개선폭 <= eps_best) AND (|정상점 실제 IoU - 예측| <= eps_pred)."""
    rng = np.random.default_rng(seed)
    theta0 = np.asarray(theta0, float)
    d = len(theta0)
    n_broad = max(n_broad, n_min(d) + 2)
    lo = physical_clip(theta0 * (1 - rel_broad))
    hi = physical_clip(theta0 * (1 + rel_broad))
    X, Y, IT = [], [], []

    def ev(t, tag, it):
        t = np.clip(t, lo, hi)
        y = float(objective(t))
        X.append(t)
        Y.append(y)
        IT.append(it)
        if verbose:
            print(f"[{len(Y):3d}] {tag:<24} IoU={y:6.2f}  best={max(Y):6.2f}", flush=True)
        return y

    ev(theta0, "base", 0)
    for u in lhs(n_broad - 1, d, rng):
        ev(lo + u * (hi - lo), "broad", 0)

    best_prev, history, stop_reason, it = max(Y), [], "max_evals", 0
    while len(Y) < max_evals:
        it += 1
        model = QuadraticRSM().fit(np.array(X), np.array(Y))
        x_hat, how = model.maximize(lo, hi)
        x_hat = np.clip(x_hat, lo, hi)
        y_pred = float(model.predict(x_hat)[0])
        y_hat = ev(x_hat, f"rsm({how})", it)
        for u in lhs(n_local, d, rng):
            if len(Y) >= max_evals:
                break
            ev(x_hat * (1 + rel_local * (2 * u - 1)), "local", it)
        best = max(Y)
        e1, e2 = best - best_prev, abs(y_hat - y_pred)
        history.append(dict(iter=it, evals=len(Y), iou_pred=y_pred, iou_at_pred=y_hat,
                            best=best, eps1=e1, eps2=e2, r2=model.r2, how=how))
        if verbose:
            print(f"      -> pred={y_pred:.2f} actual={y_hat:.2f} "
                  f"eps1={e1:.3f} eps2={e2:.3f} R2={model.r2:.3f}")
        if e1 <= eps_best and e2 <= eps_pred:
            stop_reason = "e1&e2"
            break
        best_prev = best

    final = QuadraticRSM().fit(np.array(X), np.array(Y))
    i = int(np.argmax(Y))
    return dict(theta_base=theta0, theta_best=X[i], iou_base=Y[0], iou_best=Y[i],
                theta_rsm=final.maximize(lo, hi)[0], X=np.array(X), Y=np.array(Y),
                iteration=np.array(IT), rsm=final.summary(), history=history,
                bounds=(lo, hi), paper_mode=False, stop_reason=stop_reason,
                settings=dict(rel_broad=rel_broad, n_broad=n_broad, rel_local=rel_local,
                              n_local=n_local, eps_best=eps_best, eps_pred=eps_pred,
                              max_evals=max_evals, seed=seed))


def rsm_search(objective, theta0, paper_mode=False, rel_broad=None, n_broad=30,
               rel_local=None, n_local=None, max_evals=None, eps_best=0.1, eps_pred=1.0,
               eps_stop=0.0, r2_min=None, seed=0, verbose=True, fit="ols", ridge_lambda=None,
               move="stationary"):
    """모드별 기본값: 논문 adj0=0.5, adj_t=0.15, n_t=5, 상한 80 / 우리 ±0.10, ±0.03, 6, 상한 60."""
    if max_evals is None:
        max_evals = 80 if paper_mode else 60
    if paper_mode:
        return rsm_search_paper(
            objective, theta0,
            adj0=0.5 if rel_broad is None else rel_broad, n0=n_broad,
            adj_t=0.15 if rel_local is None else rel_local,
            n_t=5 if n_local is None else n_local,
            eps_stop=eps_stop, max_evals=max_evals, r2_min=r2_min, seed=seed, verbose=verbose,
            fit=fit, ridge_lambda=ridge_lambda, move=move)
    if fit != "ols" or move != "stationary":
        raise ValueError("--fit ridge, --move boxmax는 --paper-mode에서만 지원")
    return rsm_search_ours(
        objective, theta0,
        rel_broad=0.10 if rel_broad is None else rel_broad, n_broad=n_broad,
        rel_local=0.03 if rel_local is None else rel_local,
        n_local=6 if n_local is None else n_local,
        max_evals=max_evals, eps_best=eps_best, eps_pred=eps_pred, seed=seed,
        verbose=verbose)


# ------------------------------------------------------------------ 1단계 폭 비교
def broad_sweep(objective, theta0, adjs=(0.10, 0.25, 0.50), n=30, seed=0, verbose=True):
    """Iteration_0만 탐색 폭(adj0)별로 실행. 모든 폭에서 같은 seed(같은 LHS 패턴)를 씀.
    반환: {adj: (X, Y)}"""
    out = {}
    for adj in adjs:
        rng = np.random.default_rng(seed)
        X, Y = [], []
        for th in neighborhood(theta0, adj, n, rng):
            X.append(th)
            Y.append(float(objective(th)))
            if verbose:
                print(f"[adj0={adj:.2f} {len(Y):3d}] IoU={Y[-1]:6.2f}", flush=True)
        out[adj] = (np.array(X), np.array(Y))
    return out
