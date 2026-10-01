"""
figures_periodic.py -- regenerate the pure-loss figures and tables under the PERIODIC model.
Needs sweep_periodic.py alongside.   python3 figures_periodic.py

Two quantities are shown for both lattices:
  SOFT (solid lines)  = capacity of the measured channel: mutual information between the sent class
                        and the raw folded outcome (q_-, p_+) mod 2L.
  HARD (dashed lines) = achievable rate with hard-decision nearest-lattice-point decoding, i.e. the
                        protocol exactly as described in Sec. II C.
Highlight target: 90% of the 2-bit ceiling (1.8 bits); 1.5 bits is shown for comparison.
Fig. 6 shows PER-SYMBOL error of the nearest-point decoder for both lattices.
Fig. 5 benchmarks use the corrected photon number N_S=(10^(s/10)-1)/2 (not Delta^-2).
Hardware reference lines in Fig. 8 are carried over from the earlier draft and are NOT verified.

Outputs (./figures_periodic/): fig3..fig8 (.pdf/.png), table1_capacity_soft, table2_rate_hard,
table3_advantage_vs_target (.tex/.csv)
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sweep_periodic as sp

OUT = "figures_periodic"; os.makedirs(OUT, exist_ok=True)
os.makedirs("results_periodic", exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9, "figure.dpi": 150})
SQC, HXC = "#6a3d9a", "#e66101"
HIGHLIGHT, COMPARE = 1.8, 1.5
GAIN = 10 * np.log10(2 / np.sqrt(3))
ETAS = [0.70, 0.80, 0.90, 0.95, 0.99]

C_SQ = {"soft": sp.C_square_soft_fast, "hard": sp.C_square_hard}
C_HX = {"soft": sp.C_hex_soft_fast, "hard": sp.C_hex_hard}
C_SQ_DIRECT = {"soft": sp.C_square_soft, "hard": sp.C_square_hard}
C_HX_DIRECT = {"soft": sp.C_hex_soft, "hard": sp.C_hex_hard}

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig); print("  saved", name)

def thr(Cfun, eta, T): return sp.threshold(Cfun, eta, 0.0, T)
def var_adv_dB(rate, T):
    """Hex advantage in tolerable noise variance (dB); positive = hex tolerates more noise."""
    return 10 * np.log10(sp.sigma_star(C_HX[rate], T) ** 2 / sp.sigma_star(C_SQ[rate], T) ** 2)

# ---- benchmarks: energy-matched at the CORRECTED mean photon number N_S = (10^(s/10)-1)/2 of the transmitted mode
#      (the earlier draft used N_S = Delta^-2, which is ~4-5x too large; see validation.py Section 5)
def g_(x): x = np.clip(x, 1e-15, None); return (x + 1) * np.log2(x + 1) - x * np.log2(x)
def C_EA(s, eta):
    NS = sp.mean_photon_number(s); D = np.sqrt(max((NS + eta * NS + 1) ** 2 - 4 * eta * NS * (NS + 1), 0))
    Ap = max((D - 1 + NS * (eta - 1)) / 2, 1e-15); Am = max((D - 1 - NS * (eta - 1)) / 2, 1e-15)
    return g_(NS) + g_(eta * NS) - g_(Ap) - g_(Am)
def C_hol(s, eta): return g_(eta * sp.mean_photon_number(s))

# ---------------------------------------------------------------- Fig 3
def fig3():
    s = np.arange(3.0, 20.05, 0.05)
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
    for ax, eta in zip(axs, ETAS):
        sg = [sp.sigma_eff(x, eta) for x in s]
        for col, C in ((SQC, C_SQ), (HXC, C_HX)):
            ax.plot(s, [C["soft"](g) for g in sg], color=col, lw=1.8)
            ax.plot(s, [C["hard"](g) for g in sg], color=col, lw=1.2, ls="--")
        ax.axhline(HIGHLIGHT, color="gray", ls=":", lw=1)
        ax.set_title(fr"$\eta={eta}$"); ax.set_xticks([5, 10, 15, 20]); ax.set_ylim(0, 2.05); ax.set_xlabel("$s$ (dB)")
    axs[0].set_ylabel("Rate (bits)")
    fig.legend(handles=[Line2D([0], [0], color=SQC, lw=2, label="square"), Line2D([0], [0], color=HXC, lw=2, label="hexagonal"),
                        Line2D([0], [0], color="k", lw=1.8, label="soft (capacity)"), Line2D([0], [0], color="k", lw=1.2, ls="--", label="hard-decision rate")],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(); save(fig, "fig3_capacity_vs_squeezing")

# ---------------------------------------------------------------- Fig 4
ETA_GRID = np.round(np.arange(0.75, 0.9951, 0.0025), 4)
def curves(T):
    out = {}
    for rate in ("soft", "hard"):
        out[("sq", rate)] = np.array([thr(C_SQ[rate], e, T) for e in ETA_GRID])
        out[("hex", rate)] = np.array([thr(C_HX[rate], e, T) for e in ETA_GRID])
    return out

def fig4(cache):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, T in zip(axs, (COMPARE, HIGHLIGHT)):
        for lat, col in (("sq", SQC), ("hex", HXC)):
            for rate, ls, lw in (("soft", "-", 2), ("hard", "--", 1.3)):
                y = cache[T][(lat, rate)]; m = ~np.isnan(y)
                ax.plot(ETA_GRID[m], y[m], color=col, ls=ls, lw=lw)
        ax.set_ylim(5, 30); ax.set_xlim(0.75, 0.995); ax.grid(alpha=0.25); ax.set_xlabel(r"Transmissivity $\eta$")
        ax.set_title(fr"target $C\geq{T}$ bits" + ("  (90% of ceiling)" if T == HIGHLIGHT else ""))
    axs[0].set_ylabel("Threshold squeezing (dB)")
    axs[1].legend(handles=[Line2D([0], [0], color=SQC, lw=2, label="square"), Line2D([0], [0], color=HXC, lw=2, label="hexagonal"),
                           Line2D([0], [0], color="k", lw=2, label="soft (capacity)"), Line2D([0], [0], color="k", lw=1.3, ls="--", label="hard-decision")],
                  frameon=False, loc="upper right")
    fig.tight_layout(); save(fig, "fig4_threshold_vs_eta")

# ---------------------------------------------------------------- Fig 5
def fig5():
    s = np.arange(3.0, 20.05, 0.05); eta = 0.90; sg = [sp.sigma_eff(x, eta) for x in s]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.semilogy(s, [C_EA(x, eta) for x in s], color="teal", lw=1.6, label=r"$C_{\rm EA}$")
    ax.semilogy(s, [C_hol(x, eta) for x in s], color="gray", lw=1.6, ls="--", label=r"$C_{\rm hol}$")
    ax.semilogy(s, [C_HX["soft"](g) for g in sg], color=HXC, lw=1.8, label=r"hex, soft (capacity)")
    ax.semilogy(s, [C_SQ["soft"](g) for g in sg], color=SQC, lw=1.8, label=r"square, soft (capacity)")
    ax.semilogy(s, [C_HX["hard"](g) for g in sg], color=HXC, lw=1.0, ls=":", label="hex, hard-decision")
    ax.semilogy(s, [C_SQ["hard"](g) for g in sg], color=SQC, lw=1.0, ls=":", label="square, hard-decision")
    ax.set_xlabel("Squeezing $s$ (dB)"); ax.set_ylabel("Capacity / rate (bits, log scale)"); ax.legend(frameon=False, fontsize=8); ax.set_xlim(3, 20)
    ax.set_title(r"$N_S=(10^{s/10}-1)/2$", fontsize=10)
    save(fig, "fig5_benchmark")

# ---------------------------------------------------------------- Fig 6 (per-symbol errors of the nearest-point decoder)
def fig6():
    s = np.arange(3.0, 20.05, 0.05); eta = 0.90
    psq = np.array([1 - (1 - sp.p_square_periodic(sp.sigma_eff(x, eta))) ** 2 for x in s])
    phx = np.array([sp.p_hex_periodic(sp.sigma_eff(x, eta)) for x in s])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogy(s, psq, color=SQC, lw=1.8, label="square (per symbol)"); ax.semilogy(s, phx, color=HXC, lw=1.8, label="hexagonal (per symbol)")
    ax.set_xlabel("Squeezing $s$ (dB)"); ax.set_ylabel("Symbol error probability (log scale)"); ax.set_xlim(3, 20)
    ax.legend(frameon=False); ax.grid(alpha=0.25, which="both"); save(fig, "fig6_error_probabilities")

# ---------------------------------------------------------------- Fig 7
def fig7(cache):
    Cs = np.round(np.arange(0.30, 1.981, 0.01), 3)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axs[0]
    ax.plot(Cs, [var_adv_dB("soft", T) for T in Cs], color="#1f4e79", lw=2, label="soft (capacity)")
    ax.plot(Cs, [var_adv_dB("hard", T) for T in Cs], color="#1f4e79", lw=1.3, ls="--", label="hard-decision")
    ax.axhline(0, color="k", lw=0.8); ax.axhline(GAIN, color="gray", ls=":", lw=1.2)
    ax.text(0.32, GAIN - 0.035, r"geometric gain $10\log_{10}(2/\sqrt{3})$", color="gray", fontsize=9)
    ax.axvline(HIGHLIGHT, color=HXC, ls="--", lw=1.0)
    ax.set_xlabel("Target rate (bits)"); ax.set_ylabel("Hex advantage in tolerable noise variance (dB)"); ax.legend(frameon=False, loc="lower right")
    ax.set_title(r"Intrinsic advantage (independent of $\eta$)")
    sec = ax.secondary_xaxis("top", functions=(lambda c: c / 2, lambda f: 2 * f)); sec.set_xlabel("fraction of 2-bit ceiling")
    ax = axs[1]
    for T, col in ((COMPARE, "#2b83ba"), (HIGHLIGHT, "#d7191c")):
        for rate, ls, lw in (("soft", "-", 2), ("hard", "--", 1.3)):
            d = cache[T][("sq", rate)] - cache[T][("hex", rate)]; m = ~np.isnan(d)
            ax.plot(ETA_GRID[m], d[m], color=col, ls=ls, lw=lw)
    ax.axhline(GAIN, color="gray", ls=":", lw=1.2); ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(-0.1, 3); ax.set_xlim(0.85, 0.995); ax.set_xlabel(r"Transmissivity $\eta$"); ax.set_ylabel(r"Squeezing saved by hex, $\delta s$ (dB)")
    ax.legend(handles=[Line2D([0], [0], color="#2b83ba", lw=2, label="1.5 bits"), Line2D([0], [0], color="#d7191c", lw=2, label="1.8 bits"),
                       Line2D([0], [0], color="k", lw=2, label="soft"), Line2D([0], [0], color="k", lw=1.3, ls="--", label="hard")], frameon=False)
    ax.set_title("Squeezing-threshold advantage")
    fig.tight_layout(); save(fig, "fig7_hex_advantage")

# ---------------------------------------------------------------- Fig 8
def fig8(cache):
    fig, ax = plt.subplots(figsize=(6.6, 5))
    for lat, col in (("sq", SQC), ("hex", HXC)):
        for rate, ls, lw in (("soft", "-", 2), ("hard", "--", 1.2)):
            y = cache[HIGHLIGHT][(lat, rate)]; m = ~np.isnan(y); ax.plot(ETA_GRID[m], y[m], color=col, ls=ls, lw=lw)
    ax.axhline(7.8, color="k", ls=":", lw=1); ax.text(0.755, 7.95, "7.8 dB (trapped ion) [22] -- unverified", fontsize=8)
    ax.axhspan(9.9, 14.8, color="gray", alpha=0.15); ax.text(0.755, 12.3, "9.9-14.8 dB band [19,20] -- unverified", fontsize=8, va="center")
    ax.set_ylim(5, 30); ax.set_xlim(0.75, 0.995); ax.set_xlabel(r"Transmissivity $\eta$"); ax.set_ylabel("Squeezing (dB)")
    ax.set_title(r"target $C\geq1.8$ bits (90% of ceiling)")
    ax.legend(handles=[Line2D([0], [0], color=SQC, lw=2, label="square"), Line2D([0], [0], color=HXC, lw=2, label="hexagonal"),
                       Line2D([0], [0], color="k", lw=2, label="soft"), Line2D([0], [0], color="k", lw=1.2, ls="--", label="hard")],
              frameon=False, loc="upper right")
    save(fig, "fig8_hardware_context")

# ---------------------------------------------------------------- Tables
def _fmt(v): return "N/A" if np.isnan(v) else f"{v:.2f}" + (r"$^\dagger$" if v > 20 else "")
def _fmtd(v): return "N/A" if np.isnan(v) else f"{v:.2f}"
def _plain(v): return "N/A" if np.isnan(v) else f"{v:.2f}" + ("*" if v > 20 else "")

def threshold_table(rate, name, caption, label):
    etas = [0.85, 0.88, 0.90, 0.95, 0.99]; targets = [1.5, 1.8, 1.9]; rows = []
    for eta in etas:
        r = {"eta": eta}
        for T in targets:
            a, b = thr(C_SQ_DIRECT[rate], eta, T), thr(C_HX_DIRECT[rate], eta, T)
            r[f"sq_{T}"], r[f"hex_{T}"], r[f"d_{T}"] = a, b, (a - b if not (np.isnan(a) or np.isnan(b)) else np.nan)
        rows.append(r)
    emin = {T: (sp.eta_min(C_SQ_DIRECT[rate], T), sp.eta_min(C_HX_DIRECT[rate], T)) for T in targets}
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    L = [r"\begin{table*}", rf"\caption{{{caption} $\delta s = s^{{\rm sq}}-s^{{\rm hex}}$. N/A: unreachable at any squeezing. $^\dagger$: exceeds 20~dB. Bottom row: minimum $\eta$ at infinite squeezing (square / hex).}}",
         rf"\label{{{label}}}", r"\begin{ruledtabular}\begin{tabular}{c ccc ccc ccc}",
         r" & \multicolumn{3}{c}{$\ge1.5$ bits} & \multicolumn{3}{c}{$\ge1.8$ bits (90\%)} & \multicolumn{3}{c}{$\ge1.9$ bits}\\",
         r"$\eta$ & $s^{\rm sq}$ & $s^{\rm hex}$ & $\delta s$ & $s^{\rm sq}$ & $s^{\rm hex}$ & $\delta s$ & $s^{\rm sq}$ & $s^{\rm hex}$ & $\delta s$\\ \hline"]
    for r in rows:
        L.append(f"{r['eta']:.2f} & " + " & ".join(f"{_fmt(r[f'sq_{T}'])} & {_fmt(r[f'hex_{T}'])} & {_fmtd(r[f'd_{T}'])}" for T in targets) + r"\\")
    L.append(r"\hline $\eta_{\min}$ & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{emin[T][0]:.3f} / {emin[T][1]:.3f}}}" for T in targets) + r"\\")
    L += [r"\end{tabular}\end{ruledtabular}", r"\end{table*}"]
    open(f"{OUT}/{name}.tex", "w").write("\n".join(L))
    print(f"\n{name.upper()} ({rate}); per target: sq / hex / delta_s   ('*' = exceeds 20 dB)")
    for r in rows:
        print(f"  eta={r['eta']:.2f} | " + " | ".join(f"{_plain(r[f'sq_{T}']):>7} {_plain(r[f'hex_{T}']):>7} {_fmtd(r[f'd_{T}']):>5}" for T in targets))
    for T in targets: print(f"  eta_min(>= {T}) sq / hex = {emin[T][0]:.4f} / {emin[T][1]:.4f}")

def advantage_table():
    fr = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]; rows = []
    for f_ in fr:
        T = 2 * f_; r = {"fraction": f_, "C": T}
        for rate in ("soft", "hard"):
            r[f"dvar_{rate}"] = var_adv_dB(rate, T)
            for eta in (0.90, 0.95, 0.99):
                a, b = thr(C_SQ[rate], eta, T), thr(C_HX[rate], eta, T)
                r[f"ds_{rate}_{eta}"] = a - b if not (np.isnan(a) or np.isnan(b)) else np.nan
        rows.append(r)
    with open(f"{OUT}/table3_advantage_vs_target.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    fd = lambda v: "N/A" if np.isnan(v) else f"{v:+.2f}"
    L = [r"\begin{table*}", r"\caption{Hexagonal advantage versus target rate. $\Delta\sigma^2$: advantage in tolerable noise variance (independent of $\eta$); $\delta s$: squeezing saved. Negative: square is better. N/A: unreachable.}",
         r"\label{tab:advantage}", r"\begin{ruledtabular}\begin{tabular}{cc cccc cccc}",
         r" & & \multicolumn{4}{c}{soft (capacity)} & \multicolumn{4}{c}{hard-decision rate}\\",
         r"fraction & $C$ & $\Delta\sigma^2$ & $\delta s_{0.90}$ & $\delta s_{0.95}$ & $\delta s_{0.99}$ & $\Delta\sigma^2$ & $\delta s_{0.90}$ & $\delta s_{0.95}$ & $\delta s_{0.99}$\\ \hline"]
    for r in rows:
        cells = [f"{r['dvar_soft']:+.3f}"] + [fd(r[f"ds_soft_{e}"]) for e in (0.90, 0.95, 0.99)] \
              + [f"{r['dvar_hard']:+.3f}"] + [fd(r[f"ds_hard_{e}"]) for e in (0.90, 0.95, 0.99)]
        L.append(f"{r['fraction']:.2f} & {r['C']:.2f} & " + " & ".join(cells) + r"\\")
    L += [r"\end{tabular}\end{ruledtabular}", r"\end{table*}"]
    open(f"{OUT}/table3_advantage_vs_target.tex", "w").write("\n".join(L))
    print("\nTABLE 3: dvar soft | ds soft (eta .90 .95 .99) || dvar hard | ds hard")
    for r in rows:
        print(f"  {r['fraction']:.2f} C={r['C']:.2f} | {r['dvar_soft']:+.3f} | " + " ".join(fd(r[f'ds_soft_{e}']) for e in (0.90, 0.95, 0.99))
              + f" || {r['dvar_hard']:+.3f} | " + " ".join(fd(r[f'ds_hard_{e}']) for e in (0.90, 0.95, 0.99)))

if __name__ == "__main__":
    print("threshold curves ..."); cache = {T: curves(T) for T in (COMPARE, HIGHLIGHT)}
    fig3(); fig4(cache); fig5(); fig6(); fig7(cache); fig8(cache)
    threshold_table("soft", "table1_capacity_soft", r"Threshold squeezing (dB) for the capacity (soft-decision) to reach a target, pure loss, periodic model.", "tab:soft")
    threshold_table("hard", "table2_rate_hard", r"Threshold squeezing (dB) for the hard-decision nearest-lattice-point rate to reach a target, pure loss, periodic model.", "tab:hard")
    advantage_table()
    print("\nall written to ./figures_periodic/")
