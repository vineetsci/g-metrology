#!/usr/bin/env python3
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict:
    with open(ROOT / "results" / "machine_readable" / name, encoding="utf-8") as f:
        return json.load(f)


def write_csv(name: str, header, rows) -> None:
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *rows])


def write_tex(name: str, caption: str, label: str, spec: str, header, rows) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\small",
        "\\begin{tabular}{" + spec + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    lines.extend([" & ".join(row) + " " + "\\\\" for row in rows])
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(OUT / name, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


h = load("historical_dispersion.json")
p = load("prospective_nist.json")
g = load("gaussian_predictive_band.json")
st = load("archived_student_t_predictive_band.json")
t = load("torque_offset.json")
m = load("precision_model_compare.json")

ha = h["fits"]["all"]
hp = h["fits"]["exclude_BIPM14_only"]

write_csv(
    "table1_historical_dispersion.csv",
    ["Analysis", "Q", "dof", "p", "lambda_MLE"],
    [
        ("All 16", f"{ha['q']:.12g}", ha["dof"], f"{ha['p']:.6g}", f"{ha['lambda_mle']:.12g}"),
        ("BIPM-14 excluded", f"{hp['q']:.12g}", hp["dof"], f"{hp['p']:.6g}", f"{hp['lambda_mle']:.12g}"),
    ],
)

write_tex(
    "table1_historical_dispersion.tex",
    r"Consistency of the CODATA-2018 archive under the proportional covariance-scale model. The final column gives $\mhat$ in units of $10^{-11}\ \mathrm{m^{3}\,kg^{-1}\,s^{-2}}$.",
    "tab:historical",
    "lrrrrrrr",
    [r"Configuration", r"$N$", r"$Q$", r"$\nu$", r"$p\,(\chi^{2}_{\nu})$", r"$\lhatMLE$", r"$R_{\mathrm{B}}$", r"$\mhat$"],
    [
        ["Complete archive", "16", f"{ha['q']:.3f}", "15", r"$1.60 \times 10^{-33}$", f"{ha['lambda_mle']:.3f}", f"{ha['birge_ratio']:.3f}", f"{ha['beta_si'][0]/1e-11:.6f}"],
        ["BIPM-14 excluded", "15", f"{hp['q']:.3f}", "14", r"$2.63 \times 10^{-21}$", f"{hp['lambda_mle']:.3f}", f"{hp['birge_ratio']:.3f}", f"{hp['beta_si'][0]/1e-11:.6f}"],
    ],
)

pc = p["configurations"]
primary = pc["exclude_BIPM14_only"]
allhist = pc["all_history"]
write_tex(
    "table4_prospective_nist.tex",
    r"Prospective predictive comparison for the lineage-related 2026 NIST configuration vector. Uniform covariance scaling is the primary convention; mean-only scaling is a sensitivity analysis. Finite-sample $F$ tails are primary; plug-in chi-square tails are diagnostics. The quantity $\mhat$ is reported in units of $10^{-11}\ \mathrm{m^{3}\,kg^{-1}\,s^{-2}}$.",
    "tab:prospective",
    "lrr",
    [r"Quantity", r"BIPM-14 excluded (primary)", r"Complete archive"],
    [
        ["Historical determinations", "15", "16"],
        [r"$\mhat$", "6.674227", "6.674300"],
        [r"$\lhatMLE$", f"{primary['lambda_mle']:.3f}", f"{allhist['lambda_mle']:.3f}"],
        [r"$\Qzero$ (reported covariance)", f"{primary['Q_without_inflation']:.3f}", f"{allhist['Q_without_inflation']:.3f}"],
        [r"$\Qlam$ (uniform scaling)", f"{primary['Q_with_uniform_inflation']:.3f}", f"{allhist['Q_with_uniform_inflation']:.3f}"],
        [r"Uniform tail, finite-sample $F$ (primary)", f"{primary['p_with_uniform_inflation_finite_sample']:.3f}", f"{allhist['p_with_uniform_inflation_finite_sample']:.3f}"],
        [r"Uniform tail, $\chi^{2}_{4}$ plug-in (diagnostic)", f"{primary['p_with_uniform_inflation_chi2']:.3f}", f"{allhist['p_with_uniform_inflation_chi2']:.3f}"],
        [r"$Q$ (mean-only scaling)", f"{primary['Q_with_mean_only_inflation']:.3f}", f"{allhist['Q_with_mean_only_inflation']:.3f}"],
        [r"Mean-only tail, $\chi^{2}_{4}$ plug-in", r"$2.4 \times 10^{-7}$", r"$1.3 \times 10^{-7}$"],
    ],
)

ab = t["absolute_offset_model"]
frac = t["fractional_scale_model"]
write_tex(
    "table6_torque_models.tex",
    r"Comparison of the absolute-offset and fractional-scale torque models.",
    "tab:torque-models",
    "lrrr",
    [r"Model", r"Parameter", r"$\chi^{2}/\nu$", r"$p$"],
    [
        [r"Common absolute offset", r"$\dN = \qty{1.918 \pm 0.215}{\pico\newton\metre}$", f"${ab['chi2']:.3f}/{ab['dof']}$", f"{ab['p_value_chi2_upper_tail']:.3f}"],
        [r"Fractional scale", r"$\hat f = 67.48 \pm 8.15\ \ppm$", f"${frac['chi2']:.3f}/{frac['dof']}$", f"{frac['p_value_chi2_upper_tail']:.4f}"],
        [r"Difference", r"$\Delta\chi^{2} = 11.041$", r"---", r"---"],
    ],
)

models_sorted = sorted(m["models"], key=lambda x: x["bic"])
model_names = {"prop": "Proportional scale", "power": "Power law", "both": r"Proportional $+$ floor", "floor": "Additive floor"}
write_tex(
    "table2_variance_models.tex",
    r"BIC comparison of covariance-inflation models for the complete archive. Relative weights are normalized over the four models listed and are not model probabilities.",
    "tab:bic",
    "lrrrr",
    [r"Model", r"Free parameters", r"BIC", r"$\Delta$BIC", r"Relative weight"],
    [[model_names[x["model"]], str(x["k"]), f"{x['bic']:.3f}", f"{m['bic_delta'][x['model']]:.3f}", f"{x['bic_weight']:.3f}"] for x in models_sorted],
)

# Auxiliary Paper-suite output; it is not a numbered manuscript table.
gb = g["configurations"]["exclude_BIPM14_only"]
st_intervals = {float(x["u_ppm"]): x for x in st["future_intervals"]}
write_tex(
    "gaussian_predictive_band.tex",
    r"Additional predictive-band output for one future independent scalar $G$ determination, conditional on its quoted $k=1$ standard uncertainty. The primary interval uses the Gaussian proportional common-scale model with BIPM-14 excluded and the exact finite-sample $F$ coverage factor. The 1.96 column is a Gaussian plug-in diagnostic. The archived Student-$t$ column uses the complete 16-point historical archive and is retained only as a secondary comparison.",
    "tab:gaussian_predictive_band",
    "lrrr",
    [r"Future $u$ (ppm)", r"Gaussian proportional common-scale (F)", r"Gaussian plug-in (1.96)", r"Archived Student-$t$ (all 16)"],
    [
        [
            f"{item['u_ppm']:.0f}",
            f"[{item['lower_ppm_relative_to_reference']:.1f}, {item['upper_ppm_relative_to_reference']:.1f}]",
            f"[{item['gaussian_diagnostic_lower_ppm_relative_to_reference']:.1f}, {item['gaussian_diagnostic_upper_ppm_relative_to_reference']:.1f}]",
            f"[{st_intervals[item['u_ppm']]['q025_ppm_relative_to_reference']:.1f}, {st_intervals[item['u_ppm']]['q975_ppm_relative_to_reference']:.1f}]",
        ]
        for item in gb["intervals"]
    ],
)

print(f"Generated manuscript-linked and auxiliary tables in {OUT}")
