"""Fig. 1 (protocol schematic) and Fig. 2 (hexagonal Pauli lattice, classes mod 2L, Voronoi hexagon)."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
os.makedirs("figures", exist_ok=True)

def box(ax, x, y, w, h, text, fc="#f2f2f2", ec="k", ls="-", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, ls=ls, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
def arrow(ax, x0, y0, x1, y1, style="-|>", col="k", ls="-", lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, color=col, lw=lw, ls=ls))

def fig1():
    fig, ax = plt.subplots(figsize=(11, 3.7)); ax.set_xlim(0, 11.2); ax.set_ylim(0, 3.7); ax.axis("off")
    box(ax, 0.1, 1.05, 1.9, 1.6, "Bell-pair\nsource\n(GKP, finite\nsqueezing)", fc="#dbe9f6")
    ax.text(1.05, 0.85, r"Bell-observable noise $\kappa\Delta^2$", ha="center", fontsize=8, va="top")
    # mode A path (top), mode B path (bottom)
    ya, yb = 2.25, 1.45
    arrow(ax, 2.0, ya, 2.55, ya); ax.text(2.28, ya + 0.12, "A", fontsize=9)
    box(ax, 2.55, ya - 0.3, 1.05, 0.6, r"$D_k$", fc="#fff2cc"); ax.text(3.07, ya + 0.38, "Alice encodes 2 bits", ha="center", fontsize=7.5, va="bottom")
    arrow(ax, 3.6, ya, 4.15, ya)
    box(ax, 4.15, ya - 0.3, 1.25, 0.6, r"Amplifier" + "\n" + r"$\mathcal{A}[1/\eta]$", fc="#e2f0d9", fs=8); 
    arrow(ax, 5.4, ya, 5.95, ya)
    box(ax, 5.95, ya - 0.3, 1.45, 0.6, "Loss channel\n" + r"$\eta,\ \bar n_{\rm th}$", fc="#fce4d6", ec="#c00000", ls="--", fs=8)
    ax.text(6.7, ya + 0.45, r"adds $(1-\eta)(\bar n_{\rm th}+1)$", fontsize=7.5, ha="center")
    arrow(ax, 7.4, ya, 8.0, ya)
    arrow(ax, 2.0, yb, 8.0, yb); ax.text(5.0, yb - 0.12, "B (stays at Bob, no loss)", fontsize=8, va="top", ha="center")
    box(ax, 8.0, 1.1, 1.0, 1.55, "Bell\nmeas.\n50:50 BS", fc="#dbe9f6")
    arrow(ax, 9.0, 2.2, 9.55, 2.5); arrow(ax, 9.0, 1.55, 9.55, 1.3)
    box(ax, 9.55, 2.2, 1.6, 0.75, "homodyne\n" + r"$q_-=q_A-q_B$", fc="#e2f0d9", fs=7.5); box(ax, 9.55, 0.95, 1.6, 0.75, "homodyne\n" + r"$p_+=p_A+p_B$", fc="#e2f0d9", fs=7.5)
    box(ax, 9.3, 0.02, 1.9, 0.45, r"decoder $\to\hat k$", fc="#f2f2f2", fs=8.5)
    arrow(ax, 10.35, 0.95, 10.35, 0.47, col="#555555", lw=1.0)
    ax.text(0.1, 3.5, "Dashed: non-unitary loss channel, preceded by the quantum-limited pre-amplifier.", fontsize=7.5, color="#444444")
    fig.savefig("figures/fig1_protocol.pdf", bbox_inches="tight"); fig.savefig("figures/fig1_protocol.png", bbox_inches="tight", dpi=300); plt.close(fig)

def fig2():
    ell = np.sqrt(2 * np.pi / np.sqrt(3)); x = np.array([np.sqrt(3) / 2 * ell, ell / 2]); z = np.array([-np.sqrt(3) / 2 * ell, ell / 2]); y = x + z
    cols = {0: "#1f4e79", 2: "#c00000", 1: "#2e7d32", 3: "#e69f00"}; names = {0: "I", 2: "X", 1: "Z", 3: "Y"}   # class index = 2*(m%2)+(n%2); x=(1,0)->2, z=(0,1)->1
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    R = ell / np.sqrt(3); hexv = np.array([[R * np.cos(k * np.pi / 3), R * np.sin(k * np.pi / 3)] for k in range(6)])
    ax.add_patch(Polygon(hexv, closed=True, fc="#dbe9f6", ec="#1f4e79", lw=1.5, alpha=0.7, zorder=1))
    for m in range(-4, 5):
        for n in range(-4, 5):
            p = m * x + n * z
            if np.linalg.norm(p) > 3.9: continue
            c = (m % 2) * 2 + (n % 2); stab = (m % 2 == 0 and n % 2 == 0)
            ax.scatter(*p, s=120 if stab else 45, c=cols[c], edgecolors="k" if stab else "none", linewidths=1.2, zorder=3)
    for v, lab, col in ((x, r"$\mathbf{x}$", cols[2]), (z, r"$\mathbf{z}$", cols[1]), (y, r"$\mathbf{y}$", cols[3])):
        ax.add_patch(FancyArrowPatch((0, 0), tuple(v), arrowstyle="-|>", mutation_scale=12, color=col, lw=2, zorder=4))
        ax.text(v[0] * 1.12, v[1] * 1.12, lab, color=col, fontsize=13, ha="center", va="center", zorder=5)
    ax.text(0.0, -0.35, r"$I$", fontsize=12, ha="center", color=cols[0], zorder=5)
    ax.set_aspect("equal"); ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2); ax.axis("off")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=cols[k], label=f"class {names[k]}") for k in (0, 2, 3, 1)] +
                      [Line2D([0], [0], marker="o", ls="", color="w", markeredgecolor="k", markersize=9, label=r"stabilizer lattice $2L$")],
              loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.savefig("figures/fig2_hex_lattice.pdf", bbox_inches="tight"); fig.savefig("figures/fig2_hex_lattice.png", bbox_inches="tight", dpi=300); plt.close(fig)
fig1(); fig2(); print("ok")
