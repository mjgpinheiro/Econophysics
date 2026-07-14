import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('mc_results.csv')
sw = np.load('sweeps.npz')

C = {'none': '#C0392B', 'uni': '#777777', 'tgt': '#1A7C4F',
     'kl': '#7B2D8B'}
BG, GR = '#F7F9FC', '#DDE2EA'
plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False})

fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
fig.patch.set_facecolor(BG)
fig.suptitle('Monte Carlo Sensitivity Analysis '
             r'($N=1000$ draws: $\mathcal{K}\ \pm50\%$, $\Gamma\ \pm30\%$, '
             r'shock $\pm30\%$)',
             fontsize=13, fontweight='bold')
(axA, axB, axC), (axD, axE, axF) = axes
for ax in axes.ravel():
    ax.set_facecolor(BG)
    ax.grid(True, color=GR, lw=0.5)

# --- (A) recovery-time distributions, baseline spec -------------------
bins = np.linspace(0, 120, 61)
for key, col, lab in [('rt_none', C['none'], 'No policy'),
                      ('rt_uni', C['uni'], 'Uniform'),
                      ('rt_tgt', C['tgt'], 'Targeted')]:
    axA.hist(df[key], bins=bins, alpha=0.55, color=col, label=lab)
    axA.axvline(np.median(df[key]), color=col, lw=1.4, ls='--')
axA.set_title('(A) Recovery time distributions\n(baseline policy spec)',
              fontsize=10, fontweight='bold')
axA.set_xlabel('Recovery time $\\tau_r$ (quarters)', fontsize=9)
axA.set_ylabel('Draws', fontsize=9)
axA.legend(fontsize=8, framealpha=0.6)
axA.text(112, axA.get_ylim()[1]*0.55, 'censored\nat 120q',
         fontsize=7, color=C['uni'], ha='right')

# --- (B) targeted vs uniform under symmetric exit rule ----------------
axB.scatter(df.rt_uni_exit, df.rt_tgt_exit, s=8, alpha=0.35,
            color=C['tgt'], edgecolors='none')
lim = [0, 125]
axB.plot(lim, lim, color='k', lw=1, ls=':')
axB.set_xlim(lim); axB.set_ylim(lim)
axB.set_title('(B) Symmetric exit rule:\ntargeted vs uniform, per draw',
              fontsize=10, fontweight='bold')
axB.set_xlabel(r'$\tau_r$ uniform (quarters)', fontsize=9)
axB.set_ylabel(r'$\tau_r$ targeted (quarters)', fontsize=9)
pct = (df.rt_tgt_exit < df.rt_uni_exit).mean() * 100
med = np.median(1 - df.rt_tgt_exit / df.rt_uni_exit) * 100
axB.text(0.05, 0.9, f'targeted faster in {pct:.0f}% of draws\n'
         f'median speed-up {med:.0f}%',
         transform=axB.transAxes, fontsize=8.5, color=C['tgt'],
         fontweight='bold')

# --- (C) structural lag histogram --------------------------------------
axC.hist(df.lag, bins=np.linspace(-5, 45, 51), color=C['kl'], alpha=0.7)
axC.axvline(0, color='k', lw=1)
axC.axvline(21.0, color=C['kl'], lw=1.4, ls='--')
axC.text(21.5, axC.get_ylim()[1]*0.9, 'baseline\n(21q)', fontsize=7.5,
         color=C['kl'])
axC.set_title('(C) Structural lag $t_{rec}(D_{KL}) - t_{rec}(|\\Delta S|)$',
              fontsize=10, fontweight='bold')
axC.set_xlabel('Lag (quarters)', fontsize=9)
axC.set_ylabel('Draws', fontsize=9)
pl = (df.lag > 0).mean() * 100
axC.text(0.55, 0.55, f'lag > 0 in {pl:.0f}%\nof draws\nmedian '
         f'{np.median(df.lag):.0f}q', transform=axC.transAxes,
         fontsize=8.5, color=C['kl'], fontweight='bold')

# --- (D) budget sweep ---------------------------------------------------
L = sw['lambda']
axD.plot(L[:, 0], L[:, 1], 'o-', color=C['uni'], lw=1.8, ms=4,
         label='Uniform')
axD.plot(L[:, 0], L[:, 2], 'o-', color=C['tgt'], lw=1.8, ms=4,
         label='Targeted')
axD.axhline(L[0, 3], color=C['none'], lw=1.4, ls='--', label='No policy')
axD.axvline(0.25, color='k', lw=0.8, ls=':', alpha=0.6)
axD.text(0.26, 95, 'paper\nvalue', fontsize=7.5)
axD.set_title('(D) Budget sweep (baseline calibration)',
              fontsize=10, fontweight='bold')
axD.set_xlabel(r'Budget $\lambda$ (per quarter)', fontsize=9)
axD.set_ylabel(r'$\tau_r$ (quarters)', fontsize=9)
axD.legend(fontsize=8, framealpha=0.6)

# --- (E) deployment-time sweep ------------------------------------------
Tt = sw['t0']
axE.plot(Tt[:, 0], Tt[:, 1], 'o-', color=C['uni'], lw=1.8, ms=4)
axE.plot(Tt[:, 0], Tt[:, 2], 'o-', color=C['tgt'], lw=1.8, ms=4)
axE.axhline(Tt[0, 3], color=C['none'], lw=1.4, ls='--')
axE.axvline(8, color='k', lw=0.8, ls=':', alpha=0.6)
axE.set_title('(E) Deployment-time sweep', fontsize=10, fontweight='bold')
axE.set_xlabel(r'Deployment time $t_0$ (quarters)', fontsize=9)
axE.set_ylabel(r'$\tau_r$ (quarters)', fontsize=9)

# --- (F) epicentre vs aggregate heterogeneity ---------------------------
axF.hist(df.ratio_fin_agg, bins=np.linspace(1, 3.5, 51),
         color=C['none'], alpha=0.7)
axF.axvline(1, color='k', lw=1)
axF.axvline(np.median(df.ratio_fin_agg), color=C['none'], lw=1.4, ls='--')
axF.set_title('(F) Epicentre / aggregate peak-drop ratio',
              fontsize=10, fontweight='bold')
axF.set_xlabel('Finance drop / aggregate drop', fontsize=9)
axF.set_ylabel('Draws', fontsize=9)
r = (df.ratio_fin_agg > 1).mean() * 100
axF.text(0.55, 0.7, f'ratio > 1 in {r:.0f}%\nmedian '
         f'{np.median(df.ratio_fin_agg):.1f}', transform=axF.transAxes,
         fontsize=8.5, color=C['none'], fontweight='bold')

for ax in axes.ravel():
    ax.tick_params(labelsize=8)

plt.savefig('fig8_robustness.png', dpi=200,
            bbox_inches='tight')
plt.savefig('fig8_robustness.pdf', dpi=200,
            bbox_inches='tight')
print('figure saved')
