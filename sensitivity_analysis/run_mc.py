"""Monte Carlo sensitivity analysis for the Money-Tensor crisis model.

Ensemble: N draws with independent multiplicative perturbations
  - each nonzero coupling entry K_m        x U(0.5, 1.5)   (+/-50%)
  - each friction rate Gamma_ij            x U(0.7, 1.3)   (+/-30%)
  - each shock amplitude a_ij              x U(0.7, 1.3)   (+/-30%)
  - each shock decay delta_ij              x U(0.8, 1.2)   (+/-20%)

For each stable draw we simulate: no policy, uniform & targeted
(baseline envelope spec), uniform & targeted (symmetric exit rule),
and record recovery times, sector drops, and entropic-lag diagnostics.
"""
import numpy as np
import pandas as pd
import time
from model import (build_A, build_shock, simulate, lyap_P, V_of,
                   recovery_time, info_diagnostics, sector_drops, SECTORS)

rng = np.random.default_rng(20260713)
N = 1000
T, DT = 120.0, 0.04

t0 = time.time()
ks = rng.uniform(0.5, 1.5, (N, 11))
gs = rng.uniform(0.7, 1.3, (N, 9))
asc = rng.uniform(0.7, 1.3, (N, 4))
dsc = rng.uniform(0.8, 1.2, (N, 4))

A = build_A(ks, gs)
amp, dec = build_shock(asc, dsc)
P, alpha = lyap_P(A)
stable = alpha < 0
print(f"stable draws: {stable.sum()}/{N}  (alpha range "
      f"[{alpha.min():.4f}, {alpha.max():.4f}])")

# keep only stable draws (report count)
A, amp, dec, P, alpha = A[stable], amp[stable], dec[stable], P[stable], alpha[stable]
ks, gs, asc, dsc = ks[stable], gs[stable], asc[stable], dsc[stable]
Ns = A.shape[0]

runs = {}
for name, pol, mode in [('none', 'none', 'baseline'),
                        ('uni', 'uniform', 'baseline'),
                        ('tgt', 'targeted', 'baseline'),
                        ('uni_x', 'uniform', 'exit'),
                        ('tgt_x', 'targeted', 'exit')]:
    t_eval, X = simulate(A, amp, dec, pol, mode=mode, T=T, dt=DT)
    runs[name] = X
    print(f"  simulated {name:6s}  ({time.time()-t0:.0f}s)")

Vref = V_of(runs['none'], P).max(axis=1)
rt = {k: recovery_time(t_eval, V_of(X, P), Vref) for k, X in runs.items()}

t_agg, t_kl, lag, minM = info_diagnostics(t_eval, runs['none'])
drops, agg_drop = sector_drops(runs['none'])

df = pd.DataFrame({
    'alpha': alpha,
    'rt_none': rt['none'], 'rt_uni': rt['uni'], 'rt_tgt': rt['tgt'],
    'rt_uni_exit': rt['uni_x'], 'rt_tgt_exit': rt['tgt_x'],
    'drop_man': drops[:, 0], 'drop_fin': drops[:, 1], 'drop_ser': drops[:, 2],
    'drop_agg': agg_drop, 'ratio_fin_agg': drops[:, 1] / agg_drop,
    't_rec_agg': t_agg, 't_rec_kl': t_kl, 'lag': lag, 'minM': minM,
})
df.to_csv('mc_results.csv', index=False)

cens = {k: (v >= T - DT).mean() * 100 for k, v in rt.items()}
print("\n=== SUMMARY (median [5th, 95th pct]) ===")
for c in ['rt_none', 'rt_uni', 'rt_tgt', 'rt_uni_exit', 'rt_tgt_exit',
          'drop_fin', 'drop_man', 'drop_ser', 'drop_agg',
          'ratio_fin_agg', 't_rec_agg', 't_rec_kl', 'lag']:
    q = np.percentile(df[c], [5, 50, 95])
    print(f"{c:14s}: {q[1]:7.1f}  [{q[0]:6.1f}, {q[2]:6.1f}]")

print("\n=== ORDINAL / SIGN ROBUSTNESS (% of stable draws) ===")
checks = {
    'targeted faster than no policy (baseline)': df.rt_tgt < df.rt_none,
    'uniform slower than no policy (baseline)':  df.rt_uni > df.rt_none,
    'full ordering tgt < none < uni (baseline)': (df.rt_tgt < df.rt_none)
                                                 & (df.rt_none < df.rt_uni),
    'targeted faster than uniform (exit rule)':  df.rt_tgt_exit < df.rt_uni_exit,
    'targeted faster than no policy (exit)':     df.rt_tgt_exit < df.rt_none,
    'structural lag positive (KL after agg)':    df.lag > 0,
    'lag > 8 quarters (2 years)':                df.lag > 8,
    'Finance drop > aggregate drop':             df.drop_fin > df.drop_agg,
    'Finance/aggregate ratio > 1.5':             df.ratio_fin_agg > 1.5,
    'flows remain positive (min M > 0)':         df.minM > 0,
}
for k, v in checks.items():
    print(f"  {k:45s}: {v.mean()*100:5.1f}%")
print("\ncensoring at 120q:", {k: f"{v:.1f}%" for k, v in cens.items()})
print(f"\nspeedup targeted vs uniform (exit): median "
      f"{np.median(1 - df.rt_tgt_exit/df.rt_uni_exit)*100:.0f}%")

# ---------------- one-at-a-time sweeps on the baseline calibration ----
print("\n=== ONE-AT-A-TIME SWEEPS (baseline calibration) ===")
k1 = np.ones((1, 11)); g1 = np.ones((1, 9))
A1 = build_A(k1, g1)
a1, d1 = build_shock(np.ones((1, 4)), np.ones((1, 4)))
P1, _ = lyap_P(A1)

sweep = {}
lams = np.round(np.arange(0.05, 0.61, 0.05), 2)
res = []
for lam in lams:
    te, Xu = simulate(A1, a1, d1, 'uniform', lam=lam, T=T, dt=DT)
    _, Xt = simulate(A1, a1, d1, 'targeted', lam=lam, T=T, dt=DT)
    _, Xn = simulate(A1, a1, d1, 'none', T=T, dt=DT)
    ref = V_of(Xn, P1).max(axis=1)
    res.append((lam, recovery_time(te, V_of(Xu, P1), ref)[0],
                recovery_time(te, V_of(Xt, P1), ref)[0],
                recovery_time(te, V_of(Xn, P1), ref)[0]))
sweep['lambda'] = np.array(res)
print("lambda sweep done")

t0s = np.arange(2, 21, 2.0)
res = []
for tt in t0s:
    te, Xu = simulate(A1, a1, d1, 'uniform', t0=tt, T=T, dt=DT)
    _, Xt = simulate(A1, a1, d1, 'targeted', t0=tt, T=T, dt=DT)
    _, Xn = simulate(A1, a1, d1, 'none', T=T, dt=DT)
    ref = V_of(Xn, P1).max(axis=1)
    res.append((tt, recovery_time(te, V_of(Xu, P1), ref)[0],
                recovery_time(te, V_of(Xt, P1), ref)[0],
                recovery_time(te, V_of(Xn, P1), ref)[0]))
sweep['t0'] = np.array(res)
print("t0 sweep done")

fracs = np.array([0.02, 0.03, 0.05, 0.08, 0.10])
te, Xn = simulate(A1, a1, d1, 'none', T=T, dt=DT)
_, Xu = simulate(A1, a1, d1, 'uniform', T=T, dt=DT)
_, Xt = simulate(A1, a1, d1, 'targeted', T=T, dt=DT)
Vn, Vu, Vt = V_of(Xn, P1), V_of(Xu, P1), V_of(Xt, P1)
ref = Vn.max(axis=1)
res = [(f, recovery_time(te, Vu, ref, f)[0], recovery_time(te, Vt, ref, f)[0],
        recovery_time(te, Vn, ref, f)[0]) for f in fracs]
sweep['thr'] = np.array(res)
print("threshold sweep done")

np.savez('sweeps.npz', **sweep)
print(f"\ntotal time: {time.time()-t0:.0f}s")
