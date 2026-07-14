import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d1 = np.load('lqr_results.npz')
d2 = np.load('lqr_pareto.npz')
front = d2['front']            # rho, rt, gross, J_V, J_u
te = d2['te']
Vn, Vu, Vt = d2['Vn'], d2['Vu'], d2['Vt']
Vl = d1['V_lqr']

JV_none = 107.27
pts = {'uniform': (0.0648, 122.67, 2.00, 45.1, '#777777'),
       'targeted': (0.3464, 42.21, 1.75, 12.8, '#1A7C4F')}
C = {'none': '#C0392B', 'uni': '#777777', 'tgt': '#1A7C4F',
     'lqr': '#1F5FA8'}
BG, GR = '#F7F9FC', '#DDE2EA'
plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False})

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.5, 4.9),
                                    constrained_layout=True)
fig.patch.set_facecolor(BG)
fig.suptitle('Optimal-Control (LQR) Benchmark: heuristics vs the '
             r'Riccati feedback $u = -\rho^{-1}S\,\mathbf{x}$, deployed at '
             r'$t_0 = 8$q', fontsize=12, fontweight='bold')
for ax in (axA, axB, axC):
    ax.set_facecolor(BG)
    ax.grid(True, color=GR, lw=0.5)
    ax.tick_params(labelsize=8)

# (A) trajectories
axA.semilogy(te, Vn, color=C['none'], lw=1.9, label='No policy')
axA.semilogy(te, Vu, color=C['uni'], lw=1.7, ls='--',
             label='Uniform (exit, gross 2.0)')
axA.semilogy(te, Vt, color=C['tgt'], lw=1.9,
             label='Targeted (exit, gross 1.75)')
axA.semilogy(te, Vl, color=C['lqr'], lw=1.9,
             label='LQR (budget-matched, gross 2.0)')
axA.axvspan(8, 16, color='#E8D9B8', alpha=0.35)
axA.set_ylim(1e-6, 20)
axA.set_title(r'(A) Lyapunov convergence $\mathcal{V}(t)$',
              fontsize=10, fontweight='bold')
axA.set_xlabel('Quarters', fontsize=9)
axA.set_ylabel(r'$\mathcal{V}(t)$ (log)', fontsize=9)
axA.legend(fontsize=7.5, framealpha=0.6, loc='upper right')

# (B) Pareto plane (paper's own cost functional, decomposed)
o = np.argsort(front[:, 4])
axB.plot(front[o, 4], front[o, 3], color=C['lqr'], lw=2.2,
         label='LQR frontier (varying $\\rho$)')
axB.axhline(JV_none, color=C['none'], lw=1.4, ls='--', label='No policy')
for nm, (ju, jv, g, rt, col) in pts.items():
    axB.scatter([ju], [jv], s=70, color=col, zorder=5,
                edgecolors='k', linewidths=0.6)
axB.annotate('uniform\n(2.34$\\times$ frontier,\nworse than no policy)',
             xy=pts['uniform'][:2], xytext=(0.11, 100), fontsize=8,
             color=C['uni'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C['uni'], lw=1))
axB.annotate('targeted\n(1.18$\\times$ frontier)',
             xy=pts['targeted'][:2], xytext=(0.5, 55), fontsize=8,
             color=C['tgt'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C['tgt'], lw=1))
axB.set_xscale('log')
axB.set_xlim(3e-3, 2)
axB.set_ylim(20, 135)
axB.set_title('(B) Cost plane: integrated disequilibrium\nvs control effort',
              fontsize=10, fontweight='bold')
axB.set_xlabel(r'Control effort $\int \|u\|^2\, dt$ (log)', fontsize=9)
axB.set_ylabel(r'$\int \mathcal{V}\, dt$', fontsize=9)
axB.legend(fontsize=7.5, framealpha=0.6, loc='lower left')

# (C) operational metric: recovery time vs gross expenditure
o = np.argsort(front[:, 2])
axC.plot(front[o, 2], front[o, 1], color=C['lqr'], lw=2.2,
         label='LQR (varying $\\rho$)')
axC.axhline(40.7, color=C['none'], lw=1.4, ls='--', label='No policy')
for nm, (ju, jv, g, rt, col) in pts.items():
    axC.scatter([g], [rt], s=70, color=col, zorder=5,
                edgecolors='k', linewidths=0.6)
axC.annotate('uniform', xy=(2.0, 45.1), xytext=(2.15, 52), fontsize=8,
             color=C['uni'], fontweight='bold')
axC.annotate('targeted', xy=(1.75, 12.8), xytext=(1.9, 7), fontsize=8,
             color=C['tgt'], fontweight='bold')
axC.set_title('(C) Operational metric: recovery time\nvs gross expenditure',
              fontsize=10, fontweight='bold')
axC.set_xlabel(r'Gross expenditure $\int \Sigma_{ij}|u_{ij}|\, dt$',
               fontsize=9)
axC.set_ylabel(r'$\tau_r$ (quarters)', fontsize=9)
axC.set_xlim(0, 3.6)
axC.set_ylim(0, 60)
axC.legend(fontsize=7.5, framealpha=0.6, loc='upper right')

plt.savefig('fig9_lqr.png', dpi=200, bbox_inches='tight')
plt.savefig('fig9_lqr.pdf', dpi=200, bbox_inches='tight')
print('saved')
