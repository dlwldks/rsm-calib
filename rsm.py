"""2차 반응면(Quadratic Response Surface) 모델. 논문 식 1~4를 구현한다.

논문: Moon & Cho (2026), Remote Sensing 18(2):205.
- 식 1: IoU ≈ f(x) = β0 + Σ βi·xi + Σ Σ βij·xi·xj   (교차항 포함 2차식)
- 식 2: 최소 관측 수 N_min = 1 + d + d(d+1)/2       (d = 6이면 28)
- 식 3: 정상점 ∇f(x) = 0
- 식 4: 비율 전이 θA,pred|B = θA,base ⊙ θB,best ⊘ θB,base

여기서 x(= θ)는 U-Net 입력 정규화 값 6개
[R 평균, G 평균, B 평균, R 표준편차, G 표준편차, B 표준편차],
y는 그 θ로 정규화한 영상을 U-Net에 넣어 얻은 대상 클래스 IoU(%)다.
"""
from itertools import combinations_with_replacement

import numpy as np
from scipy import optimize, stats


def n_min(d: int) -> int:
    """2차 반응면 계수 개수 = 최소 관측 수 (식 2).

    상수 1개 + 1차항 d개 + 제곱·교차항 d(d+1)/2개. d=6이면 28.
    관측(θ, IoU) 수가 이보다 적으면 계수를 하나로 정할 수 없다.
    """
    return 1 + d + d * (d + 1) // 2


def quad_features(Z: np.ndarray) -> np.ndarray:
    """식 1의 설계행렬을 만든다.

    입력 Z: (n, d) 표준화된 θ.
    출력: (n, 28) 행렬. 열 순서는
      [1, z1..zd, z1·z1, z1·z2, ..., z1·zd, z2·z2, ..., zd·zd]
    즉 상수, 1차항, 그리고 i <= j 인 모든 zi·zj (제곱항 + 교차항).
    """
    Z = np.atleast_2d(Z)
    n, d = Z.shape
    cols = [np.ones(n)] + [Z[:, i] for i in range(d)]
    cols += [Z[:, i] * Z[:, j] for i, j in combinations_with_replacement(range(d), 2)]
    return np.column_stack(cols)


def _ridge(A, y, lam):
    """Ridge 회귀의 닫힌 해 (A^T A + λD) β = A^T y.

    D는 단위행렬에서 상수항 자리만 0으로 바꾼 것이라 상수항에는 벌점을 주지 않는다.
    반환: (계수 β, hat 행렬 대각 h). h는 LOO 오차를 재추론 없이 계산할 때 쓴다
    (LOO 잔차 = 잔차 / (1 − h)).
    """
    D = np.eye(A.shape[1])
    D[0, 0] = 0.0
    Minv = np.linalg.pinv(A.T @ A + lam * D)
    beta = Minv @ (A.T @ y)
    h = np.einsum("ij,jk,ik->i", A, Minv, A)
    return beta, h


class QuadraticRSM:
    """IoU = f(θ) 2차 회귀 (논문 식 1).

    수치 안정성을 위해 θ를 열마다 z = (θ − 표본 평균) / 표본 표준편차로 바꾼 뒤 피팅한다.
    원 단위(0~255)에서는 제곱항이 10^4 크기라 설계행렬 조건수가 커지기 때문이다.
    표준화는 변수 변환이므로 예측값과 정상점 위치는 원 단위로 맞춘 식과 같다.
    """

    RIDGE_GRID = np.logspace(-3, 2, 26)   # Ridge λ 후보: 0.001 ~ 100 (로그 간격 26개)

    def fit(self, X, y, method="ols", lam=None):
        """계수 28개를 추정한다.

        X: (n, 6) θ 표본, y: (n,) IoU 표본.
        method:
          "ols"   = 최소제곱 (논문 방식, 기본값). 오차제곱합 최소.
          "ridge" = 상수항을 뺀 계수에 λ·Σβ² 벌점. 계수가 커지는 것(과적합)을 억제.
        ridge에서 lam=None이면 RIDGE_GRID 중 LOO 오차가 가장 작은 λ를 고른다.

        피팅 후 R², 조정 R², F, p를 계산해 속성으로 저장한다(논문 Table 10과 같은 지표).
        """
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        p = n_min(d)
        if n < p:
            raise ValueError(f"관측 {n}개 < 최소 {p}개")
        self.d, self.n = d, n

        # 표준화 기준(표본 평균·표준편차). 분산이 0인 열은 1로 나눠 그대로 둔다.
        self.center = X.mean(0)
        self.scale = X.std(0)
        self.scale[self.scale == 0] = 1.0
        A = quad_features((X - self.center) / self.scale)

        self.method, self.lam = method, None
        if method == "ols":
            # 최소제곱: 정규방정식 A^T A β = A^T y 와 같은 해
            self.beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        elif method == "ridge":
            grid = self.RIDGE_GRID if lam is None else [lam]
            best = None
            for g in grid:
                b, h = _ridge(A, y, g)
                r = y - A @ b
                # LOO RMSE: 표본 하나를 빼고 맞춘 식으로 그 표본을 예측했을 때의 오차.
                # 선형 평활기라 잔차/(1−h)로 바로 계산된다. h≈1인 점은 제외.
                ok = h < 1 - 1e-8
                loo = float(np.sqrt(np.mean((r[ok] / (1 - h[ok])) ** 2))) if ok.any() else np.inf
                if best is None or loo < best[0]:
                    best = (loo, g, b)
            self.loo, self.lam, self.beta = best
        else:
            raise ValueError(f"fit method {method}")

        # 적합도 통계 (논문 Table 10의 R², Adj.R², F, Prob)
        resid = y - A @ self.beta
        ss_res, ss_tot = float(resid @ resid), float(((y - y.mean()) ** 2).sum())
        df_m, df_r = p - 1, n - p                   # 모형 자유도, 잔차 자유도
        self.r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        self.adj_r2 = 1 - (1 - self.r2) * (n - 1) / df_r if df_r > 0 else np.nan
        if df_r > 0 and ss_res > 0:
            self.F = ((ss_tot - ss_res) / df_m) / (ss_res / df_r)
            self.p_value = float(stats.f.sf(self.F, df_m, df_r))
        else:
            self.F, self.p_value = np.nan, np.nan
        return self

    def predict(self, X):
        """θ(원 단위)를 넣으면 식이 예측한 IoU를 돌려준다."""
        return quad_features((np.atleast_2d(X) - self.center) / self.scale) @ self.beta

    def _grad_hess(self):
        """식을 z 좌표에서 f = b0 + bᵀz + ½ zᵀHz 형태로 바꿨을 때의 b, H.

        b = 1차 계수 6개.
        H = 헤시안. 대각은 제곱 계수의 2배, 비대각은 교차 계수(대칭으로 채움).
        """
        d = self.d
        b = self.beta[1 : 1 + d]
        H = np.zeros((d, d))
        for k, (i, j) in enumerate(combinations_with_replacement(range(d), 2)):
            c = self.beta[1 + d + k]
            if i == j:
                H[i, i] = 2 * c
            else:
                H[i, j] = H[j, i] = c
        return b, H

    def stationary_point(self):
        """정상점 ∇f = b + Hz = 0 을 풀어 원 단위 θ로 돌려준다 (논문 식 3).

        반환: (θ̂, kind)
          kind = "max"    : H 고유값이 모두 음수 → 극대점 (논문 절차가 가정하는 경우)
                 "min"    : 모두 양수 → 극소점
                 "saddle" : 부호가 섞임 → 안장점. 어떤 방향으로는 오르막, 어떤 방향으로는 내리막
        """
        b, H = self._grad_hess()
        z = np.linalg.lstsq(H, -b, rcond=None)[0]
        eig = np.linalg.eigvalsh(H)
        kind = "max" if np.all(eig < 0) else "min" if np.all(eig > 0) else "saddle"
        return z * self.scale + self.center, kind

    def maximize(self, lo, hi):
        """탐색 범위 [lo, hi] 안에서 식이 가장 높은 θ를 찾는다 ("범위 내 최대점").

        정상점이 극대이고 범위 안이면 그 점을 그대로 쓴다("stationary").
        아니면(안장점·극소점·범위 밖) 범위 안에서 L-BFGS-B로 수치 최대화한다
        ("bounded-<정상점 종류>"). 논문 Figure 1에는 없는 우리 변형(--move boxmax)에서 쓴다.
        """
        x_s, kind = self.stationary_point()
        if kind == "max" and np.all(x_s >= lo) and np.all(x_s <= hi):
            return x_s, "stationary"
        b, H = self._grad_hess()
        zlo, zhi = (lo - self.center) / self.scale, (hi - self.center) / self.scale
        f = lambda z: -(self.beta[0] + b @ z + 0.5 * z @ H @ z)   # 최대화 = 음수 최소화
        g = lambda z: -(b + H @ z)
        # 시작점 2개: 표본 중심(z=0), 정상점을 범위로 자른 점. 둘 중 더 높은 결과를 쓴다.
        starts = [np.clip(np.zeros(self.d), zlo, zhi),
                  np.clip((x_s - self.center) / self.scale, zlo, zhi)]
        best = min((optimize.minimize(f, s, jac=g, bounds=list(zip(zlo, zhi)),
                                      method="L-BFGS-B") for s in starts),
                   key=lambda r: r.fun)
        return best.x * self.scale + self.center, f"bounded-{kind}"

    def summary(self):
        """결과 저장용 요약: 관측 수, R², 조정 R², F, p, 정상점 종류, 추정 방법, Ridge λ."""
        _, kind = self.stationary_point()
        return dict(n=self.n, r2=self.r2, adj_r2=self.adj_r2, F=self.F,
                    p_value=self.p_value, stationary_kind=kind,
                    fit=getattr(self, "method", "ols"), ridge_lambda=getattr(self, "lam", None))


def proportional_transfer(theta_a_base, theta_b_base, theta_b_best):
    """비율 전이 (논문 식 4): θA,pred|B = θA,base × θB,best / θB,base (원소별).

    지역 B에서 찾은 최적/기준 비율을 지역 A의 기준 θ에 곱해 A의 최적 θ를 추정한다.
    E4(D004 ← D067 비율), E5(D067 ← D004 비율)에서 쓴다.
    """
    return np.asarray(theta_a_base) * np.asarray(theta_b_best) / np.asarray(theta_b_base)
