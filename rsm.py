"""Quadratic Response Surface Methodology (논문 식 1~4)."""
from itertools import combinations_with_replacement

import numpy as np
from scipy import optimize, stats


def n_min(d: int) -> int:
    """2차 반응면 계수 개수 = 최소 관측 수 (식 2). d=6이면 28."""
    return 1 + d + d * (d + 1) // 2


def quad_features(Z: np.ndarray) -> np.ndarray:
    """[1, z_i, z_i*z_j (i<=j)] 디자인 행렬 (식 1)."""
    Z = np.atleast_2d(Z)
    n, d = Z.shape
    cols = [np.ones(n)] + [Z[:, i] for i in range(d)]
    cols += [Z[:, i] * Z[:, j] for i, j in combinations_with_replacement(range(d), 2)]
    return np.column_stack(cols)


class QuadraticRSM:
    """IoU = f(theta) 2차 회귀. 수치 안정성을 위해 표준화 좌표 z에서 피팅."""

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        p = n_min(d)
        if n < p:
            raise ValueError(f"관측 {n}개 < 최소 {p}개")
        self.d, self.n = d, n
        self.center = X.mean(0)
        self.scale = X.std(0)
        self.scale[self.scale == 0] = 1.0
        A = quad_features((X - self.center) / self.scale)
        self.beta, *_ = np.linalg.lstsq(A, y, rcond=None)

        resid = y - A @ self.beta
        ss_res, ss_tot = float(resid @ resid), float(((y - y.mean()) ** 2).sum())
        df_m, df_r = p - 1, n - p
        self.r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        self.adj_r2 = 1 - (1 - self.r2) * (n - 1) / df_r if df_r > 0 else np.nan
        if df_r > 0 and ss_res > 0:
            self.F = ((ss_tot - ss_res) / df_m) / (ss_res / df_r)
            self.p_value = float(stats.f.sf(self.F, df_m, df_r))
        else:
            self.F, self.p_value = np.nan, np.nan
        return self

    def predict(self, X):
        return quad_features((np.atleast_2d(X) - self.center) / self.scale) @ self.beta

    def _grad_hess(self):
        """z 좌표에서 f = b0 + b^T z + 1/2 z^T H z 형태의 b, H."""
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
        """grad f = b + H z = 0 (식 3). 헤시안 고유값으로 극대/극소/안장 판정."""
        b, H = self._grad_hess()
        z = np.linalg.lstsq(H, -b, rcond=None)[0]
        eig = np.linalg.eigvalsh(H)
        kind = "max" if np.all(eig < 0) else "min" if np.all(eig > 0) else "saddle"
        return z * self.scale + self.center, kind

    def maximize(self, lo, hi):
        """정상점이 극대이고 탐색 범위 안이면 그대로, 아니면 범위 내 수치 최대화."""
        x_s, kind = self.stationary_point()
        if kind == "max" and np.all(x_s >= lo) and np.all(x_s <= hi):
            return x_s, "stationary"
        b, H = self._grad_hess()
        zlo, zhi = (lo - self.center) / self.scale, (hi - self.center) / self.scale
        f = lambda z: -(self.beta[0] + b @ z + 0.5 * z @ H @ z)
        g = lambda z: -(b + H @ z)
        starts = [np.clip(np.zeros(self.d), zlo, zhi),
                  np.clip((x_s - self.center) / self.scale, zlo, zhi)]
        best = min((optimize.minimize(f, s, jac=g, bounds=list(zip(zlo, zhi)),
                                      method="L-BFGS-B") for s in starts),
                   key=lambda r: r.fun)
        return best.x * self.scale + self.center, f"bounded-{kind}"

    def summary(self):
        _, kind = self.stationary_point()
        return dict(n=self.n, r2=self.r2, adj_r2=self.adj_r2, F=self.F,
                    p_value=self.p_value, stationary_kind=kind)


def proportional_transfer(theta_a_base, theta_b_base, theta_b_best):
    """식 4: theta_A,pred|B = theta_A,base * theta_B,best / theta_B,base."""
    return np.asarray(theta_a_base) * np.asarray(theta_b_best) / np.asarray(theta_b_base)
