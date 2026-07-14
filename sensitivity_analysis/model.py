"""Vectorised reimplementation of the Money-Tensor crisis model
(Pinheiro & Pinheiro, Entropy submission). Exactly follows
Money_Tensor_crisis_model_v2.ipynb, but batched over an ensemble of
parameter draws for Monte Carlo sensitivity analysis.

State: x in R^(B,9), B = batch of parameter draws.
dx/dt = A x + s_shock(t) + s_pol(t, x)
RK4 fixed step.
"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

NS, NA = 3, 3
SECTORS = ['Manufacturing', 'Finance', 'Services']

M_STAR = np.array([[3.00, 2.50, 1.50],
                   [1.50, 4.00, 1.00],
                   [3.50, 2.00, 2.00]])
GAMMA0 = np.array([[0.10, 0.12, 0.06],
                   [0.15, 0.18, 0.08],
                   [0.08, 0.11, 0.05]])

# (i,j,k,l,value) — exactly as in the notebook (11 entries)
K_ENTRIES = [
    (0, 1, 1, 1, -0.08), (0, 0, 1, 1, -0.05), (0, 0, 1, 0, -0.04),
    (0, 2, 1, 2, -0.02), (2, 1, 1, 1, -0.07), (2, 0, 1, 0, -0.04),
    (2, 1, 0, 1, +0.03), (2, 0, 0, 0, +0.02), (0, 2, 2, 2, +0.02),
    (2, 2, 2, 2, +0.02), (1, 0, 0, 0, -0.02),
]
# shocks: (i,j) -> (amplitude, decay)
SHOCKS0 = {(1, 1): (-0.50, 0.12), (0, 1): (-0.20, 0.15),
           (2, 1): (-0.18, 0.15), (0, 0): (-0.10, 0.18)}

U = M_STAR.ravel() / M_STAR.sum()   # uniform-policy direction (fixed)


def build_A(k_scale, gamma_scale):
    """Batched system matrices. k_scale: (B, 11); gamma_scale: (B, 9)."""
    B = k_scale.shape[0]
    A = np.zeros((B, 9, 9))
    for m, (i, j, k, l, v) in enumerate(K_ENTRIES):
        A[:, 3 * i + j, 3 * k + l] += v * k_scale[:, m]
    G = GAMMA0.ravel()[None, :] * gamma_scale
    A[:, np.arange(9), np.arange(9)] -= G
    return A


def build_shock(amp_scale, dec_scale):
    """amp (B,9), dec (B,9) arrays; zero where no shock."""
    B = amp_scale.shape[0]
    amp = np.zeros((B, 9))
    dec = np.ones((B, 9))
    for m, ((i, j), (a, d)) in enumerate(SHOCKS0.items()):
        amp[:, 3 * i + j] = a * amp_scale[:, m]
        dec[:, 3 * i + j] = d * dec_scale[:, m]
    return amp, dec


def simulate(A, amp, dec, policy='none', lam=0.25, t0=8.0,
             mode='baseline', T_pol=8.0, T=80.0, dt=0.02):
    """RK4 integration, batched. Returns t_eval (n,), X (B, 9, n).
    mode='baseline': envelope exp(-0.06 (t-t0)), active for t>=t0.
    mode='exit':     constant rate, active t0 <= t <= t0+T_pol."""
    n = int(round(T / dt)) + 1
    t_eval = np.linspace(0, T, n)
    B = A.shape[0]
    X = np.empty((B, 9, n))
    x = np.zeros((B, 9))
    X[:, :, 0] = x

    def f(t, x):
        dx = np.einsum('bij,bj->bi', A, x)
        dx += amp * np.exp(-dec * t)
        if policy != 'none':
            if mode == 'baseline':
                active = t >= t0
                env = np.exp(-0.06 * (t - t0)) if active else 0.0
            else:
                active = (t >= t0) and (t <= t0 + T_pol)
                env = 1.0
            if active:
                if policy == 'uniform':
                    dx += lam * env * U[None, :]
                else:  # targeted
                    d = np.maximum(-x, 0.0)
                    s = d.sum(axis=1, keepdims=True)
                    w = np.where(s > 1e-10, d / np.maximum(s, 1e-30), 0.0)
                    dx += lam * env * w
        return dx

    for k in range(n - 1):
        t = t_eval[k]
        k1 = f(t, x)
        k2 = f(t + dt / 2, x + dt / 2 * k1)
        k3 = f(t + dt / 2, x + dt / 2 * k2)
        k4 = f(t + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        X[:, :, k + 1] = x
    return t_eval, X


def lyap_P(A):
    """Per-draw Lyapunov matrices; returns P (B,9,9) and alpha (B,)."""
    B = A.shape[0]
    P = np.empty_like(A)
    alpha = np.empty(B)
    for b in range(B):
        alpha[b] = np.linalg.eigvals(A[b]).real.max()
        P[b] = (solve_continuous_lyapunov(A[b].T, -np.eye(9))
                if alpha[b] < 0 else np.nan)
    return P, alpha


def V_of(X, P):
    return np.einsum('bit,bij,bjt->bt', X, P, X)


def recovery_time(t_eval, V, V_ref_peak, frac=0.05):
    """First t after own peak with V < frac * V_ref_peak (per draw).
    Censored at horizon (returns t_eval[-1])."""
    B = V.shape[0]
    out = np.full(B, t_eval[-1])
    for b in range(B):
        pk = np.argmax(V[b])
        idx = np.where((np.arange(V.shape[1]) > pk)
                       & (V[b] < frac * V_ref_peak[b]))[0]
        if len(idx):
            out[b] = t_eval[idx[0]]
    return out


def info_diagnostics(t_eval, X):
    """Aggregate anomaly and D_KL recovery times (5% of own peak),
    per draw. Returns t_agg, t_kl, lag, minM."""
    B = X.shape[0]
    M = M_STAR.ravel()[None, :, None] + X          # (B,9,n)
    minM = M.min(axis=(1, 2))
    S = M.sum(axis=1)                               # (B,n)
    p = M / S[:, None, :]
    pstar = (M_STAR.ravel() / M_STAR.sum())[None, :, None]
    DKL = (p * np.log(p / pstar)).sum(axis=1)       # (B,n)
    agg = np.abs(S - M_STAR.sum()) / M_STAR.sum()

    def rec(Y):
        out = np.full(B, t_eval[-1])
        for b in range(B):
            Yn = Y[b] / Y[b].max()
            pk = np.argmax(Yn)
            idx = np.where((np.arange(len(Yn)) > pk) & (Yn < 0.05))[0]
            if len(idx):
                out[b] = t_eval[idx[0]]
        return out

    t_agg, t_kl = rec(agg), rec(DKL)
    return t_agg, t_kl, t_kl - t_agg, minM


def sector_drops(X):
    """Peak % drops per sector and aggregate. Returns (B,3), (B,)."""
    M = M_STAR.ravel()[None, :, None] + X
    Ms = M.reshape(M.shape[0], 3, 3, -1).sum(axis=2)   # (B,3,n)
    eq = M_STAR.sum(axis=1)
    drops = (eq[None, :, None] - Ms).max(axis=2) / eq[None, :] * 100
    agg = (M_STAR.sum() - M.sum(axis=1).min(axis=1)) / M_STAR.sum() * 100
    return drops, agg
