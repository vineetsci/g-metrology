#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
from g_metrology.data import load_yaml

OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def fig_historical():
    d = load_yaml(ROOT / 'data' / 'codata_2018_g.yaml')
    y = np.asarray(d.y) / 1e-11
    u = np.asarray(d.u) / 1e-11
    lab = list(d.labels)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    idx = np.arange(len(y))

    hist = json.load(open(ROOT / 'results' / 'machine_readable' / 'historical_dispersion.json'))
    mu = hist['fits']['all']['beta_si'][0] / 1e-11
    mu_se_reported = hist['fits']['all']['beta_se_si'][0] / 1e-11
    lambda_mle = float(hist['fits']['all']['lambda_mle'])
    mu_se_inflated = lambda_mle * mu_se_reported

    # Both GLS SE bands sit behind everything else. The wider band is the
    # lambda-inflated SE in pale pink; the narrower band is the reported
    # SE in pale blue. The teal is lighter in appearance than the blue so
    # it reads as the outer envelope without competing with the GLS line.
    ax.axvspan(mu - mu_se_inflated, mu + mu_se_inflated,
               color="#FCCCC9", alpha=0.85, zorder=0)
    ax.axvspan(mu - mu_se_reported, mu + mu_se_reported,
               color="#BBD6EC", alpha=0.85, zorder=0.5)

    # Orange inflated uncertainty bars.
    ax.hlines(
        idx,
        y - lambda_mle * u,
        y + lambda_mle * u,
        linewidth=6,
        color='#F4CFA8',
        alpha=0.85,
        zorder=1,
    )
    reported = ax.errorbar(
        y,
        idx,
        xerr=u,
        fmt='o',
        ms=4,
        capsize=3,
        label=r'Reported $G\;\pm\;u$',
        zorder=2,
    )
    gls = ax.axvline(
        mu,
        linestyle='--',
        color="#1A1A1A",
        linewidth=0.5,
        label='GLS mean',
        zorder=3,
    )
    gls_band_reported = Patch(
        facecolor="#BBD6EC",
        edgecolor='none',
        alpha=0.85,
        label=r'GLS mean $\pm 1$ SE (reported)',
    )
    gls_band_inflated = Patch(
        facecolor="#FCCCC9",
        edgecolor='none',
        alpha=0.85,
        label=r'GLS mean $\pm 1$ SE ($\hat\lambda$-inflated)',
    )
    inflated = Patch(
        facecolor='#F4CFA8',
        edgecolor='none',
        label=fr'Inflated uncertainty ($\lambda_{{\mathrm{{MLE}}}}={lambda_mle:.3f}$)',
    )

    ax.set_yticks(idx)
    ax.set_yticklabels(lab)
    ax.invert_yaxis()
    ax.set_xlabel(r'$G\;(10^{-11}\,\mathrm{m^3\,kg^{-1}\,s^{-2}})$')
    ax.set_ylabel('Historical determination')
    ax.set_title('Historical determinations with quoted and inflated uncertainties')
    ax.grid(axis='both', linewidth=0.7, alpha=0.22)

    ax.legend(
        handles=[gls, gls_band_inflated, gls_band_reported, reported, inflated],
        loc='lower left',
        bbox_to_anchor=(0.0, 0.045),
        frameon=True,
        framealpha=0.92,
        borderpad=0.45,
        handlelength=2.4,
    )
    save(fig, 'fig1_historical_archive')

#NIST 2026 free-minus-servo torque differences
def fig_torque():
    d = json.load(open(ROOT / 'results' / 'machine_readable' / 'torque_offset.json'))
    labels = d['configuration_order']
    y = np.asarray(d['delta_torque_pNm'], float)
    u = np.asarray(d['u_typeA_pNm'], float)
    m = float(d['common_offset_pNm'])
    se = float(d['standard_error_pNm'])
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(x, y, yerr=u, fmt='o', capsize=4, label='Free minus servo')
    ax.axhline(m, linestyle='--', label=fr'Common offset = {m:.3f} pN m')
    ax.fill_between([-0.3, len(x)-0.7], m-se, m+se, alpha=0.15, label='±1 standard error')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r'$\Delta\tau$ (pN m)')
    ax.set_title('NIST/BIPM 2026 free-minus-servo torque differences')
    ax.legend(frameon=False)
    save(fig, 'fig3_torque_common_offset')

# NIST internal-dispersion figure with proportional lambda inflation.
def fig_nist_dispersion():
    from g_metrology.data import load_yaml
    d = load_yaml(ROOT / 'data' / 'nist2026_configurations.yaml')
    y = np.asarray(d.y) / 1e-11
    u = np.asarray(d.u) / 1e-11
    lab = [s.replace('_', ' ') for s in d.labels]
    idx = np.arange(len(y))

    nf = json.load(open(ROOT / 'results' / 'machine_readable' / 'nist_factorial_results.json'))
    mu = nf['gls_mean_si'] / 1e-11
    mu_se = nf['gls_mean_se_si'] / 1e-11
    q = nf['intercept_only_q']
    nu = len(y) - 1
    lambda_internal = float(np.sqrt(q / nu))

    lambda_show = lambda_internal

    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    ax.axvspan(mu - mu_se, mu + mu_se, color='#BBD6EC', alpha=0.55, zorder=0)
    ax.hlines(idx, y - lambda_show * u, y + lambda_show * u,
              linewidth=6, color='#F4CFA8', alpha=0.85, zorder=1)
    reported = ax.errorbar(y, idx, xerr=u, fmt='o', ms=4, capsize=3,
                           label=r'Reported $G\;\pm\;u$', zorder=2)
    gls = ax.axvline(mu, linestyle='--', color='#2C6CA8',
                     label='GLS mean (this analysis)', zorder=3)
    gls_band = Patch(facecolor='#BBD6EC', edgecolor='none', alpha=0.55,
                     label=r'GLS mean $\pm 1$ SE')
    inflated = Patch(facecolor='#F4CFA8', edgecolor='none',
                     label=fr'Inflated uncertainty ($\lambda={lambda_show:.3f}$)')

    ax.set_yticks(idx)
    ax.set_yticklabels(lab)
    ax.invert_yaxis()
    ax.set_xlabel(r'$G\;(10^{-11}\,\mathrm{m^3\,kg^{-1}\,s^{-2}})$')
    ax.set_ylabel('NIST 2026 configuration')
    ax.set_title('NIST 2026 four-configuration internal scatter')
    ax.grid(axis='both', linewidth=0.7, alpha=0.22)

    ax.text(0.985, 0.97,
            fr'$Q_{{\rm int}}={q:.2f}$ on $\nu={nu}$,' '\n'
            fr'$\sqrt{{Q/\nu}}={lambda_internal:.2f}$ (internal)',
            transform=ax.transAxes, ha='right', va='top', fontsize=8)

    ax.legend(handles=[gls, gls_band, reported, inflated], loc='lower right',
              bbox_to_anchor=(1.0, 0.065),
              frameon=True, framealpha=0.92, borderpad=0.45, handlelength=2.4, fontsize=9)
    save(fig, 'fig2_nist_internal_dispersion')

def main():
    fig_historical()
    fig_torque()
    fig_nist_dispersion()
    print(f'Generated figures in {OUT}')


if __name__ == '__main__':
    main()
