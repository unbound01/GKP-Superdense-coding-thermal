"""
thermal_analysis.py -- Phase 2: thermal-loss generalisation (periodic model, both rates).
Needs sweep_periodic.py alongside.  python3 thermal_analysis.py
Writes ./figures/fig9..fig11 (.pdf/.png) and ./tables/table_thermal_*.csv/.tex and prints all numbers quoted in the paper.

Key facts
  sigma_eff^2 = kappa*Delta^2 + (1-eta)(n_th+1)                       (Sec. III D)
  threshold: s = -10 log10( 2 (sigma*^2 - f)/kappa ),  f = (1-eta)(n_th+1)
  => delta_s = s_sq - s_hex = 10 log10[ (sigma_hex*^2 - f) / (sigma_sq*^2 - f) ]   (exact; independent of kappa)
     which increases monotonically with the noise floor f and equals the intrinsic advantage at f = 0.
Benchmarks use the corrected N_S = (10^(s/10)-1)/2 (Zhuang thermal formulas).
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sweep_periodic as sp

os.makedirs("figures", exist_ok=True); os.makedirs("tables", exist_ok=True); os.makedirs("results_periodic", exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9, "figure.dpi": 150})
SQC, HXC = "#6a3d9a", "#e66101"
NTH = [0.0, 0.01, 0.05, 0.1]; NTH_COL = {0.0: "k", 0.01: "#1b9e77", 0.05: "#d95f02", 0.1: "#7570b3"}
C_SQ = {"soft": sp.C_square_soft, "hard": sp.C_square_hard}
C_HX = {"soft": sp.C_hex_soft, "hard": sp.C_hex_hard}
C_SQF = {"soft": sp.C_square_soft_fast, "hard": sp.C_square_hard}
C_HXF = {"soft": sp.C_hex_soft_fast, "hard": sp.C_hex_hard}

def save(fig, name):
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight"); fig.savefig(f"figures/{name}.png", bbox_inches="tight", dpi=300); plt.close(fig); print("  saved", name)

# ---- thermal benchmark capacities (Zhuang App. A), N_S corrected
def g_(x): x = np.clip(x, 1e-15, None); return (x + 1) * np.log2(x + 1) - x * np.log2(x)
def C_hol_th(NS, eta, nth): NB = (1 - eta) * nth; return g_(eta * NS + NB) - g_(NB)
def C_EA_th(NS, eta, nth):
    NB = (1 - eta) * nth; E = eta * NS + NB
    D = np.sqrt(max((NS + E + 1) ** 2 - 4 * eta * NS * (NS + 1), 0))
    return g_(NS) + g_(E) - g_(max((D - 1 + (E - NS)) / 2, 1e-15)) - g_(max((D - 1 - (E - NS)) / 2, 1e-15))

def thr(Cf, eta, T, nth): return sp.threshold(Cf, eta, nth, T)

# ---------------------------------------------------------------- Fig 9: capacity vs squeezing (soft), eta = 0.80, 0.90
def fig9():
    s = np.arange(3.0, 20.05, 0.05); fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, eta in zip(axs, (0.80, 0.90)):
        for nth in NTH:
            sg = [sp.sigma_eff(x, eta, nth) for x in s]
            ax.plot(s, [C_HXF["soft"](g) if g <= 1.5 else 0 for g in sg], color=NTH_COL[nth], lw=1.8)
            ax.plot(s, [C_SQF["soft"](g) if g <= 1.5 else 0 for g in sg], color=NTH_COL[nth], lw=1.3, ls="--")
        ax.axhline(1.8, color="gray", ls=":", lw=1); ax.set_title(fr"$\eta={eta}$"); ax.set_xlabel("$s$ (dB)"); ax.set_ylim(0, 2.05)
    axs[0].set_ylabel("Capacity (bits)")
    axs[1].legend(handles=[Line2D([0], [0], color=NTH_COL[n], lw=2, label=fr"$\bar n_{{\rm th}}={n}$") for n in NTH] +
                          [Line2D([0], [0], color="gray", lw=2, label="hexagonal"), Line2D([0], [0], color="gray", lw=1.3, ls="--", label="square")], frameon=False, loc="lower right")
    fig.tight_layout(); save(fig, "fig9_thermal_capacity")

# ---------------------------------------------------------------- Fig 10: threshold vs eta (1.8 bits, soft)
ETA_GRID = np.round(np.arange(0.80, 0.9951, 0.0025), 4)
def curves(T, rate):
    return {(lat, n): np.array([thr((C_SQF if lat == "sq" else C_HXF)[rate], e, T, n) for e in ETA_GRID]) for lat in ("sq", "hex") for n in NTH}
def fig10(cv):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for n in NTH:
        for lat, ls, lw in (("hex", "-", 2), ("sq", "--", 1.3)):
            y = cv[(lat, n)]; m = ~np.isnan(y); ax.plot(ETA_GRID[m], y[m], color=NTH_COL[n], ls=ls, lw=lw)
    ax.set_ylim(5, 30); ax.set_xlim(0.80, 0.995); ax.grid(alpha=0.25); ax.set_xlabel(r"Transmissivity $\eta$"); ax.set_ylabel("Threshold squeezing (dB)")
    ax.set_title(r"capacity $\geq1.8$ bits (soft decisions)")
    ax.legend(handles=[Line2D([0], [0], color=NTH_COL[n], lw=2, label=fr"$\bar n_{{\rm th}}={n}$") for n in NTH] +
                      [Line2D([0], [0], color="gray", lw=2, label="hexagonal"), Line2D([0], [0], color="gray", lw=1.3, ls="--", label="square")], frameon=False, loc="upper right")
    save(fig, "fig10_thermal_thresholds")

# ---------------------------------------------------------------- Fig 11: delta_s vs eta
def fig11(cv_soft, cv_hard):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for n in NTH:
        for cv, ls, lw in ((cv_soft, "-", 2), (cv_hard, "--", 1.2)):
            d = cv[("sq", n)] - cv[("hex", n)]; m = ~np.isnan(d) & (cv[("sq", n)] < 30); ax.plot(ETA_GRID[m], d[m], color=NTH_COL[n], ls=ls, lw=lw)
    ax.axhline(10 * np.log10(2 / np.sqrt(3)), color="gray", ls=":", lw=1.2); ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(-0.05, 2.5); ax.set_xlim(0.85, 0.995); ax.set_xlabel(r"Transmissivity $\eta$"); ax.set_ylabel(r"Squeezing saved by hex, $\delta s$ (dB)")
    ax.set_title(r"target $1.8$ bits")
    ax.legend(handles=[Line2D([0], [0], color=NTH_COL[n], lw=2, label=fr"$\bar n_{{\rm th}}={n}$") for n in NTH] +
                      [Line2D([0], [0], color="k", lw=2, label="soft"), Line2D([0], [0], color="k", lw=1.2, ls="--", label="hard")], frameon=False, loc="upper right")
    save(fig, "fig11_thermal_advantage")

def _f(v): return "N/A" if np.isnan(v) else f"{v:.2f}"
def thermal_table():
    rows = []
    for eta in (0.90, 0.95, 0.99):
        for nth in NTH:
            r = {"eta": eta, "n_th": nth}
            for rate in ("soft", "hard"):
                a, b = thr(C_SQ[rate], eta, 1.8, nth), thr(C_HX[rate], eta, 1.8, nth)
                r[f"sq_{rate}"], r[f"hex_{rate}"], r[f"ds_{rate}"] = a, b, a - b
            rows.append(r)
    with open("tables/table_thermal_thresholds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    L = [r"\begin{table*}", r"\caption{Threshold squeezing (dB) for a $1.8$-bit target under thermal loss (periodic model, $\kappa=2$). $\delta s=s^{\rm sq}-s^{\rm hex}$. $\bar n_{\rm th}=0$ rows reproduce Tables~\ref{tab:soft} and~\ref{tab:hard}.}",
         r"\label{tab:thermal}", r"\begin{ruledtabular}\begin{tabular}{cc ccc ccc}", r" & & \multicolumn{3}{c}{capacity (soft)} & \multicolumn{3}{c}{hard-decision rate}\\",
         r"$\eta$ & $\bar n_{\rm th}$ & $s^{\rm sq}$ & $s^{\rm hex}$ & $\delta s$ & $s^{\rm sq}$ & $s^{\rm hex}$ & $\delta s$\\ \hline"]
    for r in rows:
        L.append(f"{r['eta']:.2f} & {r['n_th']:.2f} & {_f(r['sq_soft'])} & {_f(r['hex_soft'])} & {_f(r['ds_soft'])} & {_f(r['sq_hard'])} & {_f(r['hex_hard'])} & {_f(r['ds_hard'])}" + r"\\")
    L += [r"\end{tabular}\end{ruledtabular}", r"\end{table*}"]
    open("tables/table_thermal_thresholds.tex", "w").write("\n".join(L))
    print("\nTHERMAL THRESHOLDS (C>=1.8): eta n_th | soft sq hex ds | hard sq hex ds")
    for r in rows: print(f"  {r['eta']:.2f} {r['n_th']:.2f} | {_f(r['sq_soft']):>6} {_f(r['hex_soft']):>6} {_f(r['ds_soft']):>5} | {_f(r['sq_hard']):>6} {_f(r['hex_hard']):>6} {_f(r['ds_hard']):>5}")
    # check the closed form delta_s = 10log10[(sh^2 - f)/(ss^2 - f)]
    err = 0
    for eta in (0.9, 0.95, 0.99):
        for nth in NTH:
            f_ = (1 - eta) * (nth + 1)
            for rate in ("soft", "hard"):
                ss, sh = sp.sigma_star(C_SQ[rate], 1.8), sp.sigma_star(C_HX[rate], 1.8)
                cf = 10 * np.log10((sh ** 2 - f_) / (ss ** 2 - f_)); dv = thr(C_SQ[rate], eta, 1.8, nth) - thr(C_HX[rate], eta, 1.8, nth)
                err = max(err, abs(cf - dv))
    print(f"closed-form delta_s vs direct: worst |diff| = {err:.1e} dB")
    for T in (1.5, 1.8):
        for rate in ("soft", "hard"):
            print(f"  eta_min(C>={T},{rate}, sq/hex) at n_th=0 / 0.1: " + " / ".join(f"{sp.eta_min(C_SQ[rate], T, n):.4f}-{sp.eta_min(C_HX[rate], T, n):.4f}" for n in (0.0, 0.1)))

def benchmarks():
    print("\nTHERMAL BENCHMARK SCAN (corrected N_S)")
    rows = []
    for nth in (0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0):
        worst = 0.0; viol_ea = 0; viol_order = 0; n_pts = 0
        for eta in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
            for s in np.arange(0.5, 25, 0.5):
                sg = sp.sigma_eff(s, eta, nth)
                if sg > 1.5: continue
                NS = sp.mean_photon_number(s); cg = max(sp.C_square_soft_fast(sg), sp.C_hex_soft_fast(sg))
                h, e = C_hol_th(NS, eta, nth), C_EA_th(NS, eta, nth); n_pts += 1
                worst = max(worst, cg / max(h, 1e-12)); viol_ea += cg > e + 1e-9; viol_order += e < h - 1e-9
        rows.append({"n_th": nth, "max_C_over_Chol": worst, "points": n_pts, "viol_EA": viol_ea, "viol_EA_ge_hol": viol_order})
        print(f"  n_th={nth:5.2f}: max C_gkp/C_hol = {worst:.3f} over {n_pts} points; C_gkp > C_EA at {viol_ea}; C_EA < C_hol at {viol_order}")
    with open("tables/table_thermal_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # pure-loss limit check of thermal formulas
    e0 = max(abs(C_hol_th(sp.mean_photon_number(s), 0.9, 0.0) - sp.mean_photon_number(s) * 0 - (g_(0.9 * sp.mean_photon_number(s)))) for s in (5, 10, 15))
    print(f"  n_th=0 reduces to g(eta N_S): worst diff {e0:.1e}")

if __name__ == "__main__":
    cv_soft = curves(1.8, "soft"); cv_hard = curves(1.8, "hard")
    fig9(); fig10(cv_soft); fig11(cv_soft, cv_hard); thermal_table(); benchmarks()
