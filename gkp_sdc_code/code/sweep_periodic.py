"""
sweep_periodic.py -- GKP superdense-coding capacities, PERIODIC (physically
consistent) model for BOTH lattices.

Model
-----
Bob's outcome is (lattice point of the Pauli lattice L) + Gaussian noise
n ~ N(0, sigma_eff^2 I_2).  The four logical Paulis {I,X,Y,Z} are the four
classes of L / 2L (2L = stabilizer lattice).  Bob decodes the nearest point of L
and reports its class mod 2L.  The channel is group-covariant, so the uniform
prior is optimal and

        C = 2 - H(class of the decoded noise point)   [bits]

Square (spacing sqrt(pi)):  the two quadratures are independent BSCs with
        p = P( round(n/sqrt(pi)) is odd ),   C_sq = 2 (1 - h2(p)).
Hex (nearest-neighbour spacing ell, ell^2 = 2 pi / sqrt(3)):  the 60-degree
        rotation permutes the three non-identity classes, so each carries p/3:
        C_hex = 2 - h2(p) - p log2(3),   p = P(decoded class != 0).
        p is a 1-D integral (P(outside hexagon)) minus tiny wrap-around terms.

Two rates are computed for each lattice:
  * HARD  = achievable rate with hard-decision nearest-lattice-point decoding (only the decoded class is used);
            this is what the protocol as literally described (Sec. II C) achieves.
  * SOFT  = mutual information between the sent class and the raw folded outcome (q_-, p_+) mod 2L,
            i.e. the capacity of the measured channel (group-covariant, so uniform input is optimal).
            SOFT >= HARD by data processing.  Deterministic quadrature; no Monte Carlo.

n_th generalises sigma_eff^2 = 2 Delta^2 + (1-eta)(n_th+1)  (n_th = 0: pure loss).
Only sigma_eff enters, so Phase 2 (thermal) is just n_th > 0.

Usage
-----
    python3 sweep_periodic.py              # thresholds table + CSVs in results_periodic/
    python3 sweep_periodic.py --validate   # brute-force Monte Carlo cross-check
    python3 sweep_periodic.py --nth 0.1    # thermal (Phase 2)

Optional: if the old repo's sweep.py is importable, the table also shows the
old finite-4-point hex numbers for comparison.
"""
import os, sys, csv, argparse
import warnings
import numpy as np
from scipy.special import erfc
from scipy.integrate import quad
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
TARGET = 1.5
SQ = np.sqrt(np.pi)                       # square Pauli-lattice spacing
ELL = np.sqrt(2.0 * np.pi / np.sqrt(3.0)) # hex nearest-neighbour spacing
R_HEX = ELL / 2.0                         # hex Voronoi inradius
RC_HEX = ELL / np.sqrt(3.0)               # hex Voronoi circumradius

# --------------------------------------------------------------------------
# noise model
# --------------------------------------------------------------------------
KAPPA = 2.0   # Bell-observable noise = kappa*Delta^2 + (1-eta)(n_th+1).
              # 2: independent finite-squeezing displacement noise on each mode of an ideal Bell pair (the paper's model)
              # 1: Bell pair prepared by an ideal SUM gate from two finite-squeezed GKP states (noise inherited from one input)
              # (see validation.py, Section 4).  kappa only shifts ABSOLUTE squeezing thresholds by 10log10(kappa/2) dB;
              # it changes neither the hexagonal advantage delta_s nor eta_min.

def delta2(s_dB):
    return 0.5 * 10.0 ** (-s_dB / 10.0)

def sigma_eff(s_dB, eta, n_th=0.0, kappa=None):
    k = KAPPA if kappa is None else kappa
    return np.sqrt(k * delta2(s_dB) + (1.0 - eta) * (n_th + 1.0))

def mean_photon_number(s_dB):
    """Mean photon number of a finite-energy GKP state whose peaks have per-quadrature (density) variance
    Delta^2 = 0.5*10^(-s/10):  n ~ 1/(4 Delta^2) - 1/2 = (10^(s/10) - 1)/2  (asymptotic; checked numerically in
    validation.py, Section 5).  The earlier draft's N_S = Delta^-2 overstated this by about a factor 4."""
    return (10.0 ** (s_dB / 10.0) - 1.0) / 2.0

def h2(p):
    p = np.clip(p, 1e-300, 1.0 - 1e-16)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)

# --------------------------------------------------------------------------
# square lattice
# --------------------------------------------------------------------------
def p_square_periodic(sigma, kmax=25):
    """P(round(n/SQ) odd) for n ~ N(0, sigma^2); exact sum over odd bins."""
    a = SQ
    tot = 0.0
    for k in range(1, 2 * kmax, 2):          # k = 1,3,5,...  (double for -k)
        lo = (k - 0.5) * a / (np.sqrt(2.0) * sigma)
        hi = (k + 0.5) * a / (np.sqrt(2.0) * sigma)
        tot += 0.5 * (erfc(lo) - erfc(hi))
    return 2.0 * tot

def C_square_periodic(sigma):
    p = p_square_periodic(sigma)
    return 2.0 * (1.0 - h2(p))

def C_square_finite(sigma):
    """Finite 4-point square constellation: independent one-sided BSCs."""
    q = 0.5 * erfc(SQ / 2.0 / (np.sqrt(2.0) * sigma))
    return 2.0 * (1.0 - h2(q))

# --------------------------------------------------------------------------
# hexagonal lattice
# --------------------------------------------------------------------------
_X = np.array([np.sqrt(3.0) / 2.0 * ELL, ELL / 2.0])
_Z = np.array([-np.sqrt(3.0) / 2.0 * ELL, ELL / 2.0])

def _p_out_hexagon(sigma):
    """P(noise outside the Voronoi hexagon of L centred at 0)."""
    f = lambda th: np.exp(-R_HEX ** 2 / (2.0 * sigma ** 2 * np.cos(th) ** 2))
    val, _ = quad(f, 0.0, np.pi / 6.0, epsabs=0.0, epsrel=1e-13, limit=200)
    return (6.0 / np.pi) * val

_GL_N = 16
_gl_x, _gl_w = np.polynomial.legendre.leggauss(_GL_N)
_gl_x = 0.5 * (_gl_x + 1.0)
_gl_w = 0.5 * _gl_w
_HEX_VERTS = np.array([[RC_HEX * np.cos(k * np.pi / 3.0), RC_HEX * np.sin(k * np.pi / 3.0)]
                       for k in range(6)])

def _build_hex_quadrature():
    """Offsets/weights (relative to the cell centre) for a 6-triangle Gauss-Legendre rule."""
    offs, wts = [], []
    for k in range(6):
        B, C = _HEX_VERTS[k], _HEX_VERTS[(k + 1) % 6]
        det = abs(B[0] * C[1] - B[1] * C[0])
        for ui, uw in zip(_gl_x, _gl_w):
            for ti, tw in zip(_gl_x, _gl_w):
                w = ti * (1.0 - ui)
                offs.append(ui * B + w * C)
                wts.append(uw * tw * (1.0 - ui) * det)
    return np.array(offs), np.array(wts)

_QOFF, _QW = _build_hex_quadrature()

_STAB_SHELL = np.array([2.0 * (m * _X + n * _Z)
                        for m in range(-3, 4) for n in range(-3, 4)
                        if (m, n) != (0, 0) and np.linalg.norm(2.0 * (m * _X + n * _Z)) < 5.5 * ELL])

def _gauss_in_hexagons(centers, sigma):
    """Sum over the given hexagon centres of the integral of N(0, sigma^2 I) over the cell."""
    if len(centers) == 0:
        return 0.0
    pts = centers[:, None, :] + _QOFF[None, :, :]
    e = np.exp(-(pts ** 2).sum(-1) / (2.0 * sigma ** 2))
    return float((e * _QW[None, :]).sum() / (2.0 * np.pi * sigma ** 2))

def p_hex_periodic(sigma):
    """P(decoded class != I) for the periodic hexagonal code."""
    p_out = _p_out_hexagon(sigma)
    dist = np.linalg.norm(_STAB_SHELL, axis=1)
    near = _STAB_SHELL[dist - RC_HEX < 9.0 * sigma]
    wrap = _gauss_in_hexagons(near, sigma)
    return max(p_out - wrap, 0.0)

def C_hex_periodic(sigma):
    p = p_hex_periodic(sigma)
    return max(2.0 - h2(p) - p * np.log2(3.0), 0.0)

# --------------------------------------------------------------------------
# SOFT-decision capacity (mutual information with the folded continuous outcome)
# --------------------------------------------------------------------------
from scipy.special import logsumexp
from scipy.interpolate import CubicSpline

_KS = np.arange(-14, 15)
_EVEN = (_KS % 2 == 0)

def _I1_square(sigma):
    a = SQ
    def integrand(n):
        e = -(n - _KS * a) ** 2 / (2 * sigma ** 2)
        l0, l1 = logsumexp(e[_EVEN]), logsumexp(e[~_EVEN])
        lpost = l0 - np.logaddexp(l0, l1)
        return np.exp(-n * n / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma) * lpost / np.log(2)
    pts = [k * a / 2 for k in range(-6, 7) if abs(k * a / 2) < 9 * sigma]
    val, _ = quad(integrand, -9 * sigma, 9 * sigma, points=pts or None, epsabs=1e-13, epsrel=1e-12, limit=400)
    return 1.0 + val

def C_square_soft(sigma):
    return 2.0 * _I1_square(sigma)

_MS = np.array([(m, n) for m in range(-7, 8) for n in range(-7, 8)])
_P = _MS[:, 0:1] * _X[None, :] + _MS[:, 1:2] * _Z[None, :]
_CLS = (_MS[:, 0] % 2) * 2 + (_MS[:, 1] % 2)
_IDX = [np.where(_CLS == c)[0] for c in range(4)]

def C_hex_soft(sigma, n_rho=80, n_th=24, rho_max=9.0):
    """2 + E[log2 posterior of the true class]; the integrand has dihedral D6 symmetry (12 elements)."""
    xr, wr = np.polynomial.legendre.leggauss(n_rho)
    rho = 0.5 * rho_max * sigma * (xr + 1); wr = 0.5 * rho_max * sigma * wr
    xt, wt = np.polynomial.legendre.leggauss(n_th)
    th = 0.5 * (np.pi / 6) * (xt + 1); wt = 0.5 * (np.pi / 6) * wt
    R, T = np.meshgrid(rho, th, indexing="ij")
    pts = np.stack([R * np.cos(T), R * np.sin(T)], -1).reshape(-1, 2)
    W = (wr[:, None] * wt[None, :]).reshape(-1) * R.reshape(-1)
    e = -((pts[:, None, :] - _P[None, :, :]) ** 2).sum(-1) / (2 * sigma ** 2)
    lc = np.stack([logsumexp(e[:, ix], axis=1) for ix in _IDX], 1)
    lpost0 = lc[:, 0] - logsumexp(lc, axis=1)
    dens = np.exp(-(pts ** 2).sum(1) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return 2.0 + 12.0 * (W * dens * lpost0).sum() / np.log(2)

# fast lookup (cubic spline in log sigma) for curves; thresholds use direct evaluation
_SOFT_TABLE = {}
def _soft_table(kind):
    if kind in _SOFT_TABLE: return _SOFT_TABLE[kind]
    cache = os.path.join("results_periodic", "soft_table.npz")
    lg = np.linspace(np.log(0.03), np.log(1.5), 260)   # validity domain: sigma_eff <= 1.5 (lattice truncation beyond)
    if os.path.exists(cache):
        z = np.load(cache)
        if "lg" in z and np.allclose(z["lg"], lg):
            tab = {"sq": z["sq"], "hex": z["hex"]}
            _SOFT_TABLE.update({k: CubicSpline(lg, v) for k, v in tab.items()}); return _SOFT_TABLE[kind]
    os.makedirs("results_periodic", exist_ok=True)
    sq = np.array([C_square_soft(np.exp(x)) for x in lg]); hx = np.array([C_hex_soft(np.exp(x)) for x in lg])
    np.savez(cache, lg=lg, sq=sq, hex=hx)
    _SOFT_TABLE.update({"sq": CubicSpline(lg, sq), "hex": CubicSpline(lg, hx)}); return _SOFT_TABLE[kind]

def C_square_soft_fast(sigma): return float(_soft_table("sq")(np.log(sigma)))
def C_hex_soft_fast(sigma):    return float(_soft_table("hex")(np.log(sigma)))

# aliases with explicit names
C_square_hard = C_square_periodic
C_hex_hard = C_hex_periodic

# --------------------------------------------------------------------------
# thresholds (bisection to 1e-6 dB; Brent on a monotone function)
# --------------------------------------------------------------------------
_SS_CACHE = {}
def sigma_star(Cfun, T):
    """Noise level sigma_eff at which the rate equals T (monotone decreasing in sigma)."""
    key = (Cfun.__name__, round(T, 9))
    if key not in _SS_CACHE:
        _SS_CACHE[key] = brentq(lambda x: Cfun(x) - T, 0.03, 1.5, xtol=1e-13)
    return _SS_CACHE[key]

def threshold(Cfun, eta, n_th=0.0, T=None, kappa=None):
    """Threshold squeezing s (dB) for rate >= T.  Exact inversion of
    sigma_eff^2 = kappa Delta^2 + (1-eta)(n_th+1) = sigma*^2, with 2 Delta^2 = 10^{-s/10}."""
    T = TARGET if T is None else T
    k = KAPPA if kappa is None else kappa
    ss = sigma_star(Cfun, T)
    floor = (1.0 - eta) * (n_th + 1.0)
    if ss ** 2 <= floor:
        return float("nan")                    # unreachable at any squeezing
    return -10.0 * np.log10(2.0 * (ss ** 2 - floor) / k)

def eta_min(Cfun, T, n_th=0.0):
    """Smallest eta for which rate T is reachable at infinite squeezing."""
    return 1.0 - sigma_star(Cfun, T) ** 2 / (n_th + 1.0)

def _finite_hex_fn():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (here, os.path.join(here, "src"), "src", "."):
            if os.path.exists(os.path.join(cand, "sweep.py")):
                sys.path.insert(0, cand); break
        from sweep import C_hex as _old_hex
        import sweep as _sw
        # old code takes (s, eta); rebuild via sigma -> pick s that gives this sigma at eta=0.9 not needed:
        return _old_hex
    except Exception:
        return None

# --------------------------------------------------------------------------
# validation: brute-force Monte Carlo (no formulas shared with the analytics)
# --------------------------------------------------------------------------
def mc_check(sigma, N=2_000_000, seed=42):
    rng = np.random.default_rng(seed)
    # hex: nearest of many lattice points, class = (m mod 2, n mod 2)
    r = sigma * rng.standard_normal((N, 2))
    ms = [(m, n) for m in range(-5, 6) for n in range(-5, 6)]
    P = np.array([m * _X + n * _Z for m, n in ms])
    cls = np.array([(m % 2) * 2 + (n % 2) for m, n in ms])
    best = np.zeros(N, int); bd = np.full(N, np.inf)
    for i, p in enumerate(P):
        d = ((r - p) ** 2).sum(1); u = d < bd; bd[u] = d[u]; best[u] = i
    pc = np.bincount(cls[best], minlength=4) / N
    p_hex = 1.0 - pc[0]
    H = -(pc[pc > 0] * np.log2(pc[pc > 0])).sum()
    C_hex_mc = 2.0 - H
    # square: per-quadrature odd rounding
    odd = (np.round(r / SQ) % 2 != 0).mean(0)
    C_sq_mc = 2.0 - h2(odd[0]) - h2(odd[1])
    p_sq_mc = odd.mean()
    return p_hex, C_hex_mc, p_sq_mc, C_sq_mc, N

def run_validation(nth=0.0):
    pts = [(s, eta) for s in (5.0, 7.44, 10.5, 15.0) for eta in (0.70, 0.90, 0.99)]
    print(f"\nMonte Carlo validation (brute-force nearest lattice point, N=2e6, n_th={nth})")
    print(f"{'s':>6} {'eta':>5} | {'p_hex an/mc':>21} {'C_hex an/mc':>17} | {'p_sq an/mc':>21} {'C_sq an/mc':>17}")
    worst_C, worst_z = 0.0, 0.0
    for s, eta in pts:
        sg = sigma_eff(s, eta, nth)
        ph, Ch, ps, Cs, N = mc_check(sg)
        pha, Cha = p_hex_periodic(sg), C_hex_periodic(sg)
        psa, Csa = p_square_periodic(sg), C_square_periodic(sg)
        for pa, pm in ((pha, ph), (psa, ps)):
            se = np.sqrt(max(pa * (1 - pa), 1e-12) / N)
            worst_z = max(worst_z, abs(pa - pm) / se)
        worst_C = max(worst_C, abs(Cha - Ch), abs(Csa - Cs))
        print(f"{s:6.2f} {eta:5.2f} | {pha:10.6f}/{ph:10.6f} {Cha:8.5f}/{Ch:8.5f} | "
              f"{psa:10.6f}/{ps:10.6f} {Csa:8.5f}/{Cs:8.5f}")
    print(f"worst |dC| = {worst_C:.2e} bits ; worst probability deviation = {worst_z:.2f} sigma (MC)")
    print("PASS" if worst_z < 5 and worst_C < 3e-3 else "CHECK")

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    global TARGET
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--nth", type=float, default=0.0)
    ap.add_argument("--target", type=float, default=1.8, help="capacity target in bits (default: 90%% of the 2-bit ceiling)")
    ap.add_argument("--etas", type=float, nargs="+", default=[0.85, 0.88, 0.90, 0.95, 0.99])
    ap.add_argument("--kappa", type=float, default=2.0, help="Bell-observable noise = kappa*Delta^2+...: 2 (independent per-mode noise, default) or 1 (SUM-prepared pair)")
    args = ap.parse_args()
    global KAPPA
    KAPPA = args.kappa
    nth = args.nth; TARGET = args.target
    if args.validate:
        run_validation(nth); return
    os.makedirs("results_periodic", exist_ok=True)
    print(f"Threshold squeezing (dB) for rate >= {TARGET} bits, n_th = {nth}, kappa = {KAPPA}")
    print(f"{'eta':>5} | {'HARD sq':>8} {'hex':>8} {'adv':>6} | {'SOFT sq':>8} {'hex':>8} {'adv':>6}")
    rows = []
    fmt = lambda v: "   N/A " if np.isnan(v) else f"{v:8.3f}"
    for eta in args.etas:
        hs, hh = threshold(C_square_hard, eta, nth), threshold(C_hex_hard, eta, nth)
        ss, sh = threshold(C_square_soft, eta, nth), threshold(C_hex_soft, eta, nth)
        ah, as_ = hs - hh, ss - sh
        print(f"{eta:5.2f} | {fmt(hs)} {fmt(hh)} {ah:6.3f} | {fmt(ss)} {fmt(sh)} {as_:6.3f}")
        rows.append(dict(eta=eta, n_th=nth, target=TARGET, hard_sq=hs, hard_hex=hh, hard_adv=ah, soft_sq=ss, soft_hex=sh, soft_adv=as_))
    print(f"geometric coding gain 10log10(2/sqrt3) = {10*np.log10(2/np.sqrt(3)):.3f} dB")
    with open(f"results_periodic/thresholds_periodic_C{TARGET}_nth{nth}_kappa{KAPPA}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    s_vals = np.arange(3.0, 20.05, 0.05)
    with open(f"results_periodic/capacity_grid_periodic_nth{nth}_kappa{KAPPA}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["s_dB", "eta", "n_th", "sigma_eff", "p_sq_quadrature", "p_hex", "C_sq_hard", "C_hex_hard", "C_sq_soft", "C_hex_soft"])
        for eta in [0.70, 0.80, 0.90, 0.95, 0.99]:
            for s in s_vals:
                sg = sigma_eff(s, eta, nth)
                w.writerow([round(s, 2), eta, nth, sg, p_square_periodic(sg), p_hex_periodic(sg),
                            C_square_hard(sg), C_hex_hard(sg), C_square_soft_fast(sg), C_hex_soft_fast(sg)])
    print("wrote results_periodic/*.csv")

if __name__ == "__main__":
    main()
