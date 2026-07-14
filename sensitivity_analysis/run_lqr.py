"""LQR benchmark for the Money-Tensor crisis model.

Solves the true LQR problem stated in Section 6.1 of the paper:
    C = int_0^inf [ x'Px + rho ||u||^2 ] dt,   Q = P (Lyapunov), R = rho I,
    B = I (policy can act on every sector-agent channel),
via the algebraic Riccati equation A'S + SA - S R^{-1} S + Q = 0,
feedback u(t) = -(1/rho) S x(t), active for t >= t0 = 8 quarters
(same deployment time as the heuristic policies).

Outputs: recovery time and gross expenditure int sum_ij |u_ij| dt as a
function of rho (the LQR frontier), plus a budget-matched comparison
against the exit-rule heuristics (gross budget 2.0).
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from model import (build_A, build_shock, lyap_P, simulate, V_of,
                   recovery_time)

T, DT, T0 = 80.0, 0.02, 8.0

# baseline calibration
A = build_A(np.ones((1, 11)), np.ones((1, 9)))[0]
amp, dec = build_shock(np.ones((1, 4)), np.ones((1, 4)))
amp, dec = amp[0], dec[0]
P, alpha = lyap_P(A[None]); P = P[0]


def rk4_lqr(Kfb):
    """Integrate with feedback u = -Kfb x for t >= T0. Returns t, X, Ut."""
    n = int(round(T / DT)) + 1
    t_eval = np.linspace(0, T, n)
    X = np.empty((9, n)); x = np.zeros(9); X[:, 0] = x

    def f(t, x):
        dx = A @ x + amp * np.exp(-dec * t)
        if t >= T0:
            dx += -Kfb @ x
        return dx

    for k in range(n - 1):
        t = t_eval[k]
        k1 = f(t, x); k2 = f(t + DT/2, x + DT/2*k1)
        k3 = f(t + DT/2, x + DT/2*k2); k4 = f(t + DT, x + DT*k3)
        x = x + DT/6*(k1 + 2*k2 + 2*k3 + k4)
        X[:, k+1] = x
    U = -(Kfb @ X)
    U[:, t_eval < T0] = 0.0
    return t_eval, X, U


# reference: no-policy peak of V (same threshold convention as paper)
t_eval, Xn = simulate(A[None], amp[None], dec[None], 'none', T=T, dt=DT)
Vn = V_of(Xn, P[None])
ref = Vn.max(axis=1)
rt_none = recovery_time(t_eval, Vn, ref)[0]

# heuristic policies, exit rule (gross budget = lambda * T_pol = 2.0)
_, Xu = simulate(A[None], amp[None], dec[None], 'uniform', mode='exit', T=T, dt=DT)
_, Xt = simulate(A[None], amp[None], dec[None], 'targeted', mode='exit', T=T, dt=DT)
rt_uni_x = recovery_time(t_eval, V_of(Xu, P[None]), ref)[0]
rt_tgt_x = recovery_time(t_eval, V_of(Xt, P[None]), ref)[0]

# heuristic policies, baseline envelope (gross budget = 0.25*(1-e^-0.06*72)/0.06)
_, Xub = simulate(A[None], amp[None], dec[None], 'uniform', T=T, dt=DT)
_, Xtb = simulate(A[None], amp[None], dec[None], 'targeted', T=T, dt=DT)
rt_uni_b = recovery_time(t_eval, V_of(Xub, P[None]), ref)[0]
rt_tgt_b = recovery_time(t_eval, V_of(Xtb, P[None]), ref)[0]
budget_baseline = 0.25 * (1 - np.exp(-0.06 * (T - T0))) / 0.06

# ------------------- LQR frontier over rho -----------------------------
rhos = np.logspace(1, 5, 41)
rows = []
for rho in rhos:
    S = solve_continuous_are(A, np.eye(9), P, rho * np.eye(9))
    Kfb = S / rho
    te, X, U = rk4_lqr(Kfb)
    V = np.einsum('it,ij,jt->t', X, P, X)[None, :]
    rt = recovery_time(te, V, ref)[0]
    gross = np.trapezoid(np.abs(U).sum(axis=0), te)
    net = np.trapezoid(U.sum(axis=0), te)
    rows.append((rho, rt, gross, net))
front = np.array(rows)

# budget-matched LQR: gross expenditure = 2.0 (same as exit-rule policies)
lg = np.log10(front[:, 0])
target = 2.0
i = np.argmin(np.abs(front[:, 2] - target))
# refine by local log-interp + bisection on gross(rho)
lo, hi = lg[max(i-1, 0)], lg[min(i+1, len(lg)-1)]
for _ in range(40):
    mid = 0.5 * (lo + hi)
    S = solve_continuous_are(A, np.eye(9), P, 10**mid * np.eye(9))
    te, X, U = rk4_lqr(S / 10**mid)
    g = np.trapezoid(np.abs(U).sum(axis=0), te)
    if g > target:
        lo = mid
    else:
        hi = mid
rho_star = 10**(0.5 * (lo + hi))
S = solve_continuous_are(A, np.eye(9), P, rho_star * np.eye(9))
te, Xl, Ul = rk4_lqr(S / rho_star)
Vl = np.einsum('it,ij,jt->t', Xl, P, Xl)
rt_lqr = recovery_time(te, Vl[None], ref)[0]
g_lqr = np.trapezoid(np.abs(Ul).sum(axis=0), te)
net_lqr = np.trapezoid(Ul.sum(axis=0), te)

print("=== LQR BENCHMARK (baseline calibration) ===")
print(f"rho* (budget-matched)      : {rho_star:.1f}")
print(f"gross expenditure          : {g_lqr:.3f}  (target 2.0)")
print(f"net expenditure            : {net_lqr:.3f}")
print(f"recovery time LQR          : {rt_lqr:.1f} q")
print(f"recovery time targeted exit: {rt_tgt_x:.1f} q  (budget 2.0)")
print(f"recovery time uniform exit : {rt_uni_x:.1f} q  (budget 2.0)")
print(f"recovery time no policy    : {rt_none:.1f} q")
print(f"\nimprovement over no policy : LQR {rt_none-rt_lqr:.1f}q, "
      f"targeted {rt_none-rt_tgt_x:.1f}q "
      f"({(rt_none-rt_tgt_x)/(rt_none-rt_lqr)*100:.0f}% of LQR improvement)")
print(f"\nbaseline-envelope policies : uniform {rt_uni_b:.1f}q, "
      f"targeted {rt_tgt_b:.1f}q  (gross budget {budget_baseline:.2f})")

np.savez('lqr_results.npz', front=front, rho_star=rho_star,
         rt_lqr=rt_lqr, g_lqr=g_lqr, net_lqr=net_lqr,
         t=te, V_lqr=Vl, V_none=Vn[0],
         V_uni_x=V_of(Xu, P[None])[0], V_tgt_x=V_of(Xt, P[None])[0],
         rt_none=rt_none, rt_uni_x=rt_uni_x, rt_tgt_x=rt_tgt_x,
         rt_uni_b=rt_uni_b, rt_tgt_b=rt_tgt_b,
         budget_baseline=budget_baseline)
print("\nsaved lqr_results.npz")
