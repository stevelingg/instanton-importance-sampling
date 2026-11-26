"""Synthetic illustration of the repository's instanton-guided sampler.

Run from the repository root: python scripts/05_plot_sampling_demo.py
No market data or historical output is required.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from src.fit import SVParams
from src.instanton import select_instanton_tau_star
from src.importance_sampling import cap_control, simulate_is_paths, estimate_probability_is
from src.model_sv import effective_variance


def drawdown_paths(params, control, n_paths, seed, threshold):
    """Record paths using the same Euler update and RNG as simulate_is_paths."""
    rng = np.random.default_rng(seed)
    n_steps = len(control)
    x = np.zeros(n_paths)
    v = np.full(n_paths, params.vbar)
    m = x.copy()
    dd = np.zeros((n_steps + 1, n_paths))
    dt = params.dt_years
    log_weights = np.zeros(n_paths)
    for i, ui in enumerate(control):
        eps = rng.standard_normal((n_paths, 2))
        dB = ui * dt + np.sqrt(dt) * eps
        log_weights += -(dB[:, 0]*ui[0] + dB[:, 1]*ui[1]) + 0.5 * (ui @ ui) * dt
        v_eff = effective_variance(v)
        sqrt_v = np.sqrt(v_eff)
        x += params.mu * dt + sqrt_v * dB[:, 0]
        v += params.kappa * (params.vbar - v_eff) * dt + params.xi * sqrt_v * (
            params.rho * dB[:, 0] + np.sqrt(max(1e-12, 1-params.rho**2)) * dB[:, 1])
        m = np.maximum(m, x)
        dd[i+1] = m - x
    # Cross-check recording code against the existing production simulator.
    hits, weights = simulate_is_paths(params, 0., params.vbar, 0., threshold,
                                     n_steps, control, n_paths, seed=seed)
    np.testing.assert_array_equal((dd >= threshold).any(axis=0), hits.astype(bool))
    np.testing.assert_allclose(np.exp(log_weights), weights, rtol=1e-12)
    return -np.expm1(-dd)


def main():
    params = SVParams(mu=.04, kappa=2., vbar=.04, xi=.35, rho=-.6)
    n_steps, delta = 40, .20
    threshold = -np.log1p(-delta)
    sol = select_instanton_tau_star(params, 0., params.vbar, 0., threshold, n_steps)
    full = np.zeros((n_steps, 2))
    full[:sol.tau] = sol.u_star
    q = cap_control(.5 * full, umax=10.)
    p = np.zeros_like(q)
    # Pair the Gaussian innovations across P/Q for the illustration;
    # report hit rates from a separate, larger paired simulation.
    n_show, n_mc = 32, 50_000
    paths = [drawdown_paths(params, c, n_show, 2026, threshold) for c in (p, q)]
    stats = [estimate_probability_is(*simulate_is_paths(params, 0., params.vbar, 0.,
              threshold, n_steps, c, n_mc, seed=42)) for c in (p, q)]
    dstar = np.maximum.accumulate(sol.x_path) - sol.x_path
    star = -np.expm1(-dstar)
    ymax = max(.31, max(float(a.max()) for a in paths)*1.06)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#334155',
                         'xtick.color':'#475569','ytick.color':'#475569'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    for ax, a, stat, label in zip(axes, paths, stats, ('Original measure $P$', 'Proposal measure $Q$')):
        ax.axhspan(delta, ymax, color='#e9c78f', alpha=.27, zorder=0)
        ax.axhline(delta, color='#b77935', linestyle=(0,(5,4)), linewidth=1.4)
        for path in a.T:
            ax.plot(np.arange(n_steps+1), path, color='#4477a9', alpha=.40, linewidth=1.15)
        ax.set_title(label, loc='left', fontweight='bold', fontsize=13, pad=12)
        ax.set_xlim(0,n_steps); ax.set_ylim(ymax,0)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel('Trading days'); ax.set_xticks([0,10,20,30,40])
        ax.grid(axis='y', alpha=.15)
        ax.text(.03,.045,'Rare-event region: drawdown ≥ 20%',transform=ax.transAxes,
                fontsize=9,color='#80531f',bbox=dict(facecolor='#faf2e4',edgecolor='none',pad=2))
        ax.text(.03,.94, f'Hit rate: {stat.hit_rate_under_Q:.2%}',transform=ax.transAxes,
                va='top',fontweight='bold',color='#243c55',
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
    axes[0].set_ylabel('Drawdown from running peak')
    axes[1].plot(np.arange(sol.tau+1),star,color='#17212c',lw=2.1,ls='--',label='Optimised instanton (untempered)')
    axes[1].legend(loc='upper right',frameon=True,fontsize=8,framealpha=.95)
    fig.suptitle('Instanton-guided importance sampling',x=.09,ha='left',fontweight='bold',fontsize=17,y=.98)
    fig.text(.09,.895,'Same stochastic-volatility model, different sampling measures',fontsize=10,color='#475569')
    fig.text(.09,.065,'Synthetic experiment · 32 trajectories shown per panel; hit rates from 50,000 paths per measure.',fontsize=9,color='#475569')
    fig.text(.09,.035,'Proposal hit rate is not the target probability; likelihood-ratio reweighting is required.',fontsize=9,color='#475569')
    fig.subplots_adjust(left=.09,right=.98,top=.77,bottom=.18,wspace=.12)
    out=ROOT/'docs'/'figures'; out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'instanton_sampling.png',dpi=200,facecolor='white')
    fig.savefig(out/'instanton_sampling.svg',facecolor='white')
    plt.close(fig)
    print(f'tau={sol.tau}, action={sol.action:.6f}, exact drawdown={sol.dexact_tau:.6f}')
    for name,s in zip(('P','Q'),stats):
        print(f'{name}: hit rate={s.hit_rate_under_Q:.6f}, estimate={s.p_hat:.8f}, '
              f'SE={s.se:.8f}, ESS={s.ess:.2f}, mean weight={s.mean_weight:.6f}')
    print('Path-recorder checks passed for both measures.')

if __name__ == '__main__':
    main()
