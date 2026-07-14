"""Pareto-plane analysis for the LQR benchmark (Figure 9, panels B/C).

Computes, on the baseline calibration:
  - integrated disequilibrium J_V = int V dt, control effort
    J_u = int ||u||^2 dt, and gross expenditure int sum|u| dt for the
    no-policy, uniform (exit-rule) and targeted (exit-rule) policies;
  - the LQR frontier (rho, tau_r, gross, J_V, J_u) over a log grid;
  - the frontier J_V at control effort exactly matching each heuristic
    (bisection on rho), yielding the 1.18x / 2.34x ratios reported in
    Section "Optimal-Control Benchmark".

Run AFTER run_lqr.py (independent, but make_fig9.py needs both npz files).
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from model import build_A, build_shock, lyap_P, recovery_time

T, DT, T0, TPOL, LAM = 80.0, 0.02, 8.0, 8.0, 0.25

A = build_A(np.ones((1, 11)), np.ones((1, 9)))[0]
amp, dec = build_shock(np.ones((1, 4)), np.ones((1, 4)))
amp, dec = amp[0], dec[0]
P, _ = lyap_P(A[None]); P = P[0]
M_STAR = np.array([[3.0, 2.5, 1.5], [1.5, 4.0, 1.0], [3.5, 2.0, 2.0]])
U_DIR = M_STAR.ravel() / M_STAR.sum()


def rk4(policy_u):
    n = int(round(T / DT)) + 1
    te = np.linspace(0, T, n)
    X = np.empty((9, n)); Uh = np.zeros((9, n)); x = np.zeros(9)
    X[:, 0] = x

    def f(t, x):
        u = policy_u(t, x)
        return A @ x + amp * np.exp(-dec * t) + u, u

    for k in range(n - 1):
        t = te[k]
        k1, _ = f(t, x); k2, _ = f(t + DT/2, x + DT/2*k1)
        k3, _ = f(t + DT/2, x + DT/2*k2); k4, _ = f(t + DT, x + DT*k3)
        x = x + DT/6*(k1 + 2*k2 + 2*k3 + k4)
        X[:, k+1] = x
        Uh[:, k+1] = f(te[k+1], x)[1]
    return te, X, Uh


def u_none(t, x):
    return np.zeros(9)


def u_uni(t, x):
    return LAM * U_DIR if T0 <= t <= T0 + TPOL else np.zeros(9)


def u_tgt(t, x):
    if T0 <= t <= T0 + TPOL:
        d = np.maximum(-x, 0.0); s = d.sum()
        return LAM * d / s if s > 1e-10 else np.zeros(9)
    return np.zeros(9)


def metrics(te, X, Uh):
    V = np.einsum('it,ij,jt->t', X, P, X)
    return (np.trapezoid(V, te), np.trapezoid((Uh**2).sum(0), te),
            np.trapezoid(np.abs(Uh).sum(0), te), V)


te, Xn, Un = rk4(u_none); JVn, JUn, Gn, Vn = metrics(te, Xn, Un)
_, Xu, Uu = rk4(u_uni);   JVu, JUu, Gu, Vu = metrics(te, Xu, Uu)
_, Xt, Ut = rk4(u_tgt);   JVt, JUt, Gt, Vt = metrics(te, Xt, Ut)
ref = np.array([Vn.max()])

print(f"{'policy':10s} {'J_V':>8s} {'J_u':>8s} {'gross':>6s} {'tau_r':>6s}")
for nm, JV, JU, G, V in [('none', JVn, JUn, Gn, Vn),
                         ('uniform', JVu, JUu, Gu, Vu),
                         ('targeted', JVt, JUt, Gt, Vt)]:
    rt = recovery_time(te, V[None], ref)[0]
    print(f"{nm:10s} {JV:8.2f} {JU:8.4f} {G:6.2f} {rt:6.1f}")


def lqr_run(rho):
    S = solve_continuous_are(A, np.eye(9), P, rho * np.eye(9))
    K = S / rho

    def u_lqr(t, x):
        return -K @ x if t >= T0 else np.zeros(9)

    te, X, Uh = rk4(u_lqr)
    JV, JU, G, V = metrics(te, X, Uh)
    rt = recovery_time(te, V[None], ref)[0]
    return rt, G, JV, JU


front = np.array([(rho, *lqr_run(rho)) for rho in np.logspace(0.5, 5, 46)])


def frontier_JV_at_effort(target_Ju):
    """Bisection on log10(rho) so that J_u(rho) = target_Ju."""
    lo, hi = 0.0, 5.5
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        _, _, JV, Ju = lqr_run(10**mid)
        if Ju > target_Ju:
            lo = mid
        else:
            hi = mid
    return JV


print("\nequal-effort frontier points:")
for nm, Ju_h, JV_h in [('targeted', JUt, JVt), ('uniform', JUu, JVu)]:
    JV_f = frontier_JV_at_effort(Ju_h)
    print(f"  {nm}: heuristic J_V = {JV_h:.1f}, frontier J_V = {JV_f:.1f} "
          f"-> {JV_h/JV_f:.2f}x frontier")

np.savez('lqr_pareto.npz', front=front, te=te, Vn=Vn, Vu=Vu, Vt=Vt,
         pts=np.array([[JUn, JVn], [JUu, JVu], [JUt, JVt]]),
         gross=np.array([Gn, Gu, Gt]))
print("\nsaved lqr_pareto.npz")
