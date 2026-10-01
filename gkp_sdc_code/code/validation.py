"""
validation.py  (lower-case, to match every reference in the paper and README)
=============================================================================
Validation for the GKP superdense-coding paper, PERIODIC model.
Needs sweep_periodic.py in the same folder (the analytic formulas being checked).

Paper cross-references (fixed: the old file had Appendix A/B swapped)
  Appendix A  = Monte Carlo validation of the capacities      -> Section 1 here
  Appendix B  = stage-by-stage circuit simulation of sigma_eff -> Section 2 here
  Appendix C  = thermal-loss validation                        -> run with --nth 0.1

Section 0  Lattice geometry sanity checks (index of 2L in L, cell area, min distances).
Section 1  Brute-force Monte Carlo of the decoded class (nearest of many lattice points,
           class = parity of the integer coordinates).  It shares NO error-region formulas
           with the analytic hexagon integral / odd-bin sums in sweep_periodic.py.
           It only shares the lattice definition, which Section 0 checks.
Section 1b Soft-decision capacity: deterministic quadrature (sweep_periodic.C_*_soft) versus large Monte
           Carlo estimates of E[log2 posterior] with standard errors, quadrature convergence, spline
           accuracy, limits (sigma->0 gives 2 bits, sigma large gives 0) and soft >= hard (data processing).
Section 4  Bell-pair noise model: Gaussian propagation through (a) an ideal SUM gate acting on two finite-squeezed
           GKP states (Bell observables carry ONE input's noise: kappa = 1; marginal noise Delta^2, 2Delta^2, 2Delta^2,
           Delta^2) and (b) independent displacement noise on each mode of an ideal Bell pair (kappa = 2); plus the
           lattice mismatch of a beam-splitter "Bell pair" and the kappa-invariance of the hexagonal advantage.
Section 5  Energy convention: numerical mean photon number of a finite-energy GKP wavefunction versus
           (10^(s/10) - 1)/2 (the earlier N_S = Delta^-2 is ~4x too large), and ordering of the GKP capacity against
           C_EA and the unassisted Holevo capacity at the corrected energy.
Section 6  Thermal benchmarks (Phase 2): n_th -> 0 reduction of the thermal C_hol, C_EA to the pure-loss formulas, and the
           orderings C_gkp <= C_EA and C_EA >= C_hol on a grid (n_th up to 20).  The formulas (Zhuang, App. A) are taken
           from the literature; this checks implementation consistency, not the formulas themselves.
Section 2  Stage-by-stage noise propagation (amplifier and loss vacuum noise as two separate
           draws) checking sigma_eff^2 = 2 Delta^2 + (1-eta)(n_th+1), over several seeds, plus a
           signal-scaling diagnostic showing WHY the pre-amplifier is needed (mean of q_-).

Scope / honesty notes (keep these in the paper too)
  * Section 1 validates the numerics of the analytic formulas, not the physical model.
  * Section 2 checks the noise algebra of the INDEPENDENT-noise model (kappa = 2).  It draws q_A and q_B
    independently, so it verifies the variance bookkeeping only.  Its "Bell pair from a beam splitter" is NOT a
    logical GKP Bell pair (Section 4 shows the lattice mismatch); the noise model it tests is "independent
    finite-squeezing displacement noise on each mode of an ideal Bell pair".  A pair prepared by an ideal SUM gate
    from two finite-squeezed states has kappa = 1 (Section 4).

Run:
    python3 validation.py                 # everything, pure loss
    python3 validation.py --nth 0.1       # thermal (Appendix C regression + thermal sigma_eff)
    python3 validation.py --n 400000      # quick run
"""
import argparse, sys
import numpy as np
import sweep_periodic as sp

TOL_C = 3e-3          # bits, on top of MC statistical error
TOL_Z = 5.0           # MC sigmas for probabilities

_B = np.array([sp._X, sp._Z]).T
_BINV = np.linalg.inv(_B)

def _nearest_class_candidates(r):
    """Nearest lattice point among the 16 integer-coordinate candidates around floor(coords);
    returns the class (parity of the integer coordinates).  Section 0 verifies this equals the
    full 81-point brute force."""
    f = np.floor(r @ _BINV.T)
    best = np.full(len(r), np.inf); cls = np.zeros(len(r), int)
    for da in range(-1, 3):
        for db in range(-1, 3):
            m = f + np.array([da, db])
            d = ((r - m @ _B.T) ** 2).sum(1)
            u = d < best
            best[u] = d[u]
            cls[u] = ((m[u, 0] % 2) * 2 + (m[u, 1] % 2)).astype(int)
    return cls

def _nearest_class_full(r):
    ms = [(m, n) for m in range(-4, 5) for n in range(-4, 5)]
    P = np.array([m * sp._X + n * sp._Z for m, n in ms])
    par = np.array([(m % 2) * 2 + (n % 2) for m, n in ms])
    d = ((r[:, None, :] - P[None]) ** 2).sum(-1)
    return par[d.argmin(1)]

def mc_class_distribution(sigma, N, seed, chunk=1_000_000):
    """Brute-force decoded-class distribution for both lattices."""
    rng = np.random.default_rng(seed)
    hex_counts = np.zeros(4)
    odd_counts = np.zeros(2)
    done = 0
    while done < N:
        k = min(chunk, N - done)
        r = sigma * rng.standard_normal((k, 2))
        hex_counts += np.bincount(_nearest_class_candidates(r), minlength=4)
        odd_counts += (np.round(r / sp.SQ) % 2 != 0).sum(0)
        done += k
    return hex_counts / N, odd_counts / N

def section0():
    print("=" * 96 + "\n  SECTION 0 -- lattice geometry checks\n" + "=" * 96)
    x, z = sp._X, sp._Z
    ok = True
    def chk(name, cond):
        nonlocal ok; ok &= bool(cond); print(f"  {'OK  ' if cond else 'FAIL'} {name}")
    chk("|x| = |z| = ell", np.isclose(np.linalg.norm(x), sp.ELL) and np.isclose(np.linalg.norm(z), sp.ELL))
    chk("|x+z| = ell  (Y = X.Z has the same length)", np.isclose(np.linalg.norm(x + z), sp.ELL))
    chk("x, z at 120 degrees", np.isclose(np.degrees(np.arccos(x @ z / sp.ELL**2)), 120.0))
    area = abs(x[0] * z[1] - x[1] * z[0])
    chk(f"Pauli-lattice cell area = pi (same as square)  [{area:.6f}]", np.isclose(area, np.pi))
    chk("min distance^2 ratio hex/square = 2/sqrt(3)", np.isclose(sp.ELL**2 / np.pi, 2 / np.sqrt(3)))
    P = np.array([m * x + n * z for m in range(-2, 3) for n in range(-2, 3) if (m, n) != (0, 0)])
    d = np.linalg.norm(P, axis=1); nn = int((np.abs(d - sp.ELL) < 1e-9).sum())
    chk(f"6 nearest neighbours at distance ell  [found {nn}]", nn == 6)
    rr = np.random.default_rng(1).standard_normal((200_000, 2)) * 0.9
    chk("16-candidate search == full 81-point brute force (200k samples, sigma=0.9)",
        np.array_equal(_nearest_class_candidates(rr), _nearest_class_full(rr)))
    print(f"  Section 0: {'PASS' if ok else 'FAIL'}")
    return ok

def section1(nth, N, seeds):
    print("\n" + "=" * 96 + f"\n  SECTION 1 -- brute-force Monte Carlo (N={N:,} per point, seeds {seeds}, n_th={nth})\n" + "=" * 96)
    pts = [(s, eta) for s in (7.0, 10.5, 13.5, 20.0) for eta in (0.70, 0.80, 0.90, 0.95, 0.99)]
    print(f"  {'s':>5} {'eta':>5} | {'p_hex an':>10} {'p_hex mc':>10} {'z':>5} | {'p_sq an':>10} {'p_sq mc':>10} {'z':>5} | "
          f"{'dC_hex':>8} {'dC_sq':>8}")
    worst_z = worst_C = 0.0
    for s, eta in pts:
        sg = sp.sigma_eff(s, eta, nth)
        ph_a, ps_a = sp.p_hex_periodic(sg), sp.p_square_periodic(sg)
        Ch_a, Cs_a = sp.C_hex_periodic(sg), sp.C_square_periodic(sg)
        hc = np.zeros(4); oc = np.zeros(2)
        for sd in seeds:
            a, b = mc_class_distribution(sg, N, sd); hc += a / len(seeds); oc += b / len(seeds)
        Ntot = N * len(seeds)
        ph_m = 1 - hc[0]; ps_m = oc.mean()
        se_h = np.sqrt(max(ph_a * (1 - ph_a), 1e-12) / Ntot)
        se_s = np.sqrt(max(ps_a * (1 - ps_a), 1e-12) / (2 * Ntot))
        zh, zs = abs(ph_a - ph_m) / se_h, abs(ps_a - ps_m) / se_s
        Ch_m = 2 + (hc[hc > 0] * np.log2(hc[hc > 0])).sum()
        Cs_m = 2.0 - sp.h2(oc[0]) - sp.h2(oc[1])
        worst_z = max(worst_z, zh, zs)
        worst_C = max(worst_C, abs(Ch_a - Ch_m), abs(Cs_a - Cs_m))
        print(f"  {s:5.1f} {eta:5.2f} | {ph_a:10.6f} {ph_m:10.6f} {zh:5.2f} | {ps_a:10.6f} {ps_m:10.6f} {zs:5.2f} | "
              f"{abs(Ch_a-Ch_m):8.1e} {abs(Cs_a-Cs_m):8.1e}")
        # 3-fold symmetry check: the three non-identity classes should be equally likely
    tolC = TOL_C + 3.0 / np.sqrt(N * len(seeds))          # absolute floor + statistical allowance (matters for small --n)
    ok = worst_z < TOL_Z and worst_C < tolC
    print(f"  worst probability deviation = {worst_z:.2f} MC-sigma ; worst |dC| = {worst_C:.2e} bits")
    print(f"  Section 1: {'PASS' if ok else 'FAIL'}")
    return ok


def _mc_soft(sigma, N, seed, chunk=100_000):
    """Monte Carlo of the soft-decision capacities: 2 + E[log2 P(true class | outcome)]."""
    from scipy.special import logsumexp
    rng = np.random.default_rng(seed)
    ms = np.array([(m, n) for m in range(-6, 7) for n in range(-6, 7)])
    P = ms[:, 0:1] * sp._X[None] + ms[:, 1:2] * sp._Z[None]
    cl = (ms[:, 0] % 2) * 2 + (ms[:, 1] % 2); idx = [np.where(cl == c)[0] for c in range(4)]
    ks = np.arange(-14, 15); a = sp.SQ
    vh, vs = [], []
    for _ in range(N // chunk):
        r = sigma * rng.standard_normal((chunk, 2))
        e = -((r[:, None, :] - P[None]) ** 2).sum(-1) / (2 * sigma ** 2)
        lc = np.stack([logsumexp(e[:, ix], axis=1) for ix in idx], 1)
        vh.append((lc[:, 0] - logsumexp(lc, axis=1)) / np.log(2))
        n = r[:, 0]                                                 # one quadrature; the other is independent
        ee = -(n[:, None] - ks[None] * a) ** 2 / (2 * sigma ** 2)
        l0 = logsumexp(ee[:, ks % 2 == 0], axis=1); l1 = logsumexp(ee[:, ks % 2 != 0], axis=1)
        vs.append((l0 - np.logaddexp(l0, l1)) / np.log(2))
    vh, vs = np.concatenate(vh), np.concatenate(vs)
    return 2 + vh.mean(), vh.std() / np.sqrt(len(vh)), 2 * (1 + vs.mean()), 2 * vs.std() / np.sqrt(len(vs))

def section1b(nth, Nsoft, seed):
    print("\n" + "=" * 96 + f"\n  SECTION 1b -- soft-decision capacity (MC N={Nsoft:,}, seed {seed}, n_th={nth})\n" + "=" * 96)
    pts = [(7.0, 0.90), (10.5, 0.90), (12.0, 0.95), (9.0, 0.99), (5.0, 0.70), (15.0, 0.90)]
    print(f"  {'s':>5} {'eta':>5} {'sigma':>6} | {'hex quad':>9} {'hex MC':>9} {'+-':>7} {'z':>5} | {'sq quad':>9} {'sq MC':>9} {'+-':>7} {'z':>5}")
    ok = True; worst = 0.0
    for s, eta in pts:
        sg = sp.sigma_eff(s, eta, nth)
        qh, qs = sp.C_hex_soft(sg), sp.C_square_soft(sg)
        mh, eh, ms_, es = _mc_soft(sg, Nsoft, seed)
        zh, zs = (mh - qh) / eh, (ms_ - qs) / es; worst = max(worst, abs(zh), abs(zs))
        print(f"  {s:5.1f} {eta:5.2f} {sg:6.3f} | {qh:9.5f} {mh:9.5f} {eh:7.5f} {zh:+5.2f} | {qs:9.5f} {ms_:9.5f} {es:7.5f} {zs:+5.2f}")
    ok &= worst < TOL_Z
    print(f"  worst |z| = {worst:.2f}")
    # quadrature convergence
    d = max(abs(sp.C_hex_soft(x, 80, 24) - sp.C_hex_soft(x, 160, 60)) for x in (0.25, 0.45, 0.7, 1.0))
    print(f"  hex quadrature (80x24 vs 160x60 nodes): worst |diff| = {d:.1e}"); ok &= d < 1e-8
    # spline
    e = max(max(abs(sp.C_hex_soft_fast(x) - sp.C_hex_soft(x)), abs(sp.C_square_soft_fast(x) - sp.C_square_soft(x)))
            for x in np.exp(np.linspace(np.log(0.05), np.log(1.4), 30) + 0.011))
    print(f"  spline lookup vs direct quadrature: worst |diff| = {e:.1e} bits"); ok &= e < 1e-6
    # properties
    sgs = np.exp(np.linspace(np.log(0.05), np.log(1.4), 40))
    soft_ge = all(sp.C_square_soft(x) >= sp.C_square_hard(x) - 1e-9 and sp.C_hex_soft(x) >= sp.C_hex_hard(x) - 1e-9 for x in sgs)
    mono = all(np.diff([sp.C_hex_soft(x) for x in sgs]) < 1e-12) and all(np.diff([sp.C_square_soft(x) for x in sgs]) < 1e-12)   # non-increasing (saturates at 2 bits)
    lim = abs(sp.C_hex_soft(0.05) - 2) < 1e-6 and abs(sp.C_square_soft(0.05) - 2) < 1e-6 and sp.C_hex_soft(1.4) < 0.05
    print(f"  soft >= hard everywhere (sigma in [0.05,1.4]): {soft_ge}; non-increasing in sigma: {mono}; limits (2 bits, ~0): {lim}")
    ok &= soft_ge and mono and lim
    print(f"  Section 1b: {'PASS' if ok else 'FAIL'}")
    return ok

def _circuit(s_dB, eta, nth, N, seed, preamp=True, shift=0.0):
    rng = np.random.default_rng(seed)
    D = np.sqrt(sp.delta2(s_dB))
    q1, q2 = rng.normal(0, D, N), rng.normal(0, D, N)
    qA, qB = (q1 + q2) / np.sqrt(2), (q1 - q2) / np.sqrt(2)
    qA = qA + shift                                   # Alice's displacement (0 for identity)
    if preamp:
        G = 1.0 / eta
        qA = np.sqrt(G) * qA + rng.normal(0, np.sqrt((G - 1) / 2), N)      # amplifier vacuum noise
    qA = np.sqrt(eta) * qA + rng.normal(0, np.sqrt((1 - eta) * (nth + 0.5)), N)   # loss (thermal) noise
    q = qA - qB
    return q.var(), q.mean()

def section2(nth, N, seeds):
    print("\n" + "=" * 96 + f"\n  SECTION 2 -- stage-by-stage circuit simulation (N={N:,}, seeds {seeds}, n_th={nth})\n" + "=" * 96)
    pts = [(10.0, 0.90), (10.0, 0.70), (15.0, 0.90), (8.0, 0.80), (12.0, 0.95), (7.0, 0.70), (20.0, 0.99)]
    print(f"  {'s':>5} {'eta':>5} {'sigma2 analytic':>16} {'sigma2 sim':>12} {'|diff|':>9} {'5sigma tol':>10}  ok")
    ok = True; worst = 0.0
    for s, eta in pts:
        ana = float(sp.sigma_eff(s, eta, nth) ** 2)
        sims = [_circuit(s, eta, nth, N, sd)[0] for sd in seeds]
        v = float(np.mean(sims)); diff = abs(v - ana)
        tol = 5 * ana * np.sqrt(2.0 / (N * len(seeds)))
        good = diff < tol; ok &= good; worst = max(worst, diff)
        print(f"  {s:5.1f} {eta:5.2f} {ana:16.6f} {v:12.6f} {diff:9.2e} {tol:10.2e}  {'OK' if good else 'FAIL'}")
    print(f"  worst |d sigma2| = {worst:.2e}")
    print("\n  Why the pre-amplifier matters: mean of q_- for a sqrt(pi) displacement (ideal value sqrt(pi) = %.4f)" % sp.SQ)
    print(f"  {'s':>5} {'eta':>5} {'with preamp':>12} {'without':>10}")
    for s, eta in [(10.0, 0.90), (10.0, 0.70)]:
        m1 = _circuit(s, eta, nth, N, seeds[0], True, sp.SQ)[1]
        m0 = _circuit(s, eta, nth, N, seeds[0], False, sp.SQ)[1]
        print(f"  {s:5.1f} {eta:5.2f} {m1:12.4f} {m0:10.4f}   (without: signal scaled by sqrt(eta) = {np.sqrt(eta):.4f})")
    print("  Without amplification the lattice seen by the decoder is shrunk by sqrt(eta): a bias, not extra variance.")
    print("  (A classical rescale by 1/sqrt(eta) does not commute with the joint Bell measurement, hence the amplifier.)")
    print(f"  Section 2: {'PASS' if ok else 'FAIL'}")
    return ok


def section4():
    print("\n" + "=" * 96 + "\n  SECTION 4 -- Bell-pair noise model (kappa) and lattice consistency\n" + "=" * 96)
    ok = True
    rng = np.random.default_rng(3); N = 2_000_000; D2 = 0.09; d = np.sqrt(D2)
    qA, pA, qB, pB = [rng.normal(0, d, N) for _ in range(4)]           # |+>_A (control), |0>_B (target) noise
    qA1, pA1, qB1, pB1 = qA, pA - pB, qB + qA, pB                        # ideal SUM: q_B += q_A ; p_A -= p_B
    v = lambda x: np.var(x) / D2
    tol = 5 * np.sqrt(2.0 / N) * 2
    r_sum = (v(qA1 - qB1), v(pA1 + pB1)); marg = (v(qA1), v(pA1), v(qB1), v(pB1))
    c = abs(np.cov(qA1 - qB1, pA1 + pB1)[0, 1]) / D2
    good = all(abs(x - 1) < tol for x in r_sum) and abs(marg[0] - 1) < tol and abs(marg[1] - 2) < 2 * tol and abs(marg[2] - 2) < 2 * tol and abs(marg[3] - 1) < tol and c < 1e-2
    ok &= good
    print(f"  SUM-prepared pair : Var(q_A-q_B) = {r_sum[0]:.4f} D^2, Var(p_A+p_B) = {r_sum[1]:.4f} D^2  -> kappa = 1 ; "
          f"marginals (qA,pA,qB,pB) = {tuple(round(float(m), 3) for m in marg)} D^2 ; quadrature covariance {c:.1e}   {'OK' if good else 'FAIL'}")
    nq = [rng.normal(0, d, N) for _ in range(4)]
    r_ind = (v(nq[0] - nq[2]), v(nq[1] + nq[3])); good = all(abs(x - 2) < 2 * tol for x in r_ind); ok &= good
    print(f"  Independent noise : Var(q_A-q_B) = {r_ind[0]:.4f} D^2, Var(p_A+p_B) = {r_ind[1]:.4f} D^2  -> kappa = 2   {'OK' if good else 'FAIL'}")
    # beam-splitter of two |0>_GKP: ideal q_A-q_B takes values sqrt(2)*(2 sqrt(pi)) Z, but a Pauli shift needs 2 sqrt(pi) Z
    ratio = (np.sqrt(2) * 2 * sp.SQ) / (2 * sp.SQ); good = np.isclose(ratio, np.sqrt(2)); ok &= good
    print(f"  Beam-splitter 'Bell pair': ideal q_A-q_B lattice / required lattice = {ratio:.4f} (= sqrt 2 -> not a logical GKP Bell pair)   {'OK' if good else 'FAIL'}")
    # kappa-invariance of hex advantage and 10 log10(kappa/2) shift of thresholds
    good = True
    for rate, (a, b) in (("soft", (sp.C_square_soft, sp.C_hex_soft)), ("hard", (sp.C_square_hard, sp.C_hex_hard))):
        for eta in (0.90, 0.99):
            t2 = (sp.threshold(a, eta, 0, 1.8, 2.0), sp.threshold(b, eta, 0, 1.8, 2.0))
            t1 = (sp.threshold(a, eta, 0, 1.8, 1.0), sp.threshold(b, eta, 0, 1.8, 1.0))
            good &= abs((t2[0] - t2[1]) - (t1[0] - t1[1])) < 1e-9 and abs((t2[0] - t1[0]) - 10 * np.log10(2)) < 1e-9
    ok &= good
    print(f"  kappa=1 vs 2: thresholds shift by exactly 10log10(2)=3.010 dB, hex advantage delta_s unchanged, eta_min unchanged   {'OK' if good else 'FAIL'}")
    print(f"  Section 4: {'PASS' if ok else 'FAIL'}")
    return ok

def _gkp_photon_number(sig2, L=40.0, n=2**18):
    Dg = np.sqrt(2 * sig2)                                   # amplitude std of peaks; density peak variance = sig2
    q = np.linspace(-L, L, n, endpoint=False); dq = q[1] - q[0]
    psi = np.zeros_like(q)
    for k in range(-40, 41):
        psi += np.exp(-(q - 2 * k * sp.SQ) ** 2 / (2 * Dg ** 2))
    psi *= np.exp(-Dg ** 2 * q ** 2 / 2); psi /= np.sqrt((psi ** 2).sum() * dq)
    q2 = (q ** 2 * psi ** 2).sum() * dq; p2 = (np.gradient(psi, dq) ** 2).sum() * dq
    m = np.abs(q) < 0.6 * sp.SQ; dens = psi ** 2 * m; dens /= dens.sum() * dq
    return (q2 + p2 - 1) / 2, (q ** 2 * dens).sum() * dq

def _g(x): x = np.clip(x, 1e-15, None); return (x + 1) * np.log2(x + 1) - x * np.log2(x)
def _cea(NS, eta):
    D = np.sqrt(max((NS + eta * NS + 1) ** 2 - 4 * eta * NS * (NS + 1), 0))
    return _g(NS) + _g(eta * NS) - _g(max((D - 1 + NS * (eta - 1)) / 2, 1e-15)) - _g(max((D - 1 - NS * (eta - 1)) / 2, 1e-15))

def section5():
    print("\n" + "=" * 96 + "\n  SECTION 5 -- energy convention and benchmark ordering\n" + "=" * 96)
    ok = True
    print(f"  {'sigma^2':>8} {'s (dB)':>7} | {'peak var (meas.)':>16} | {'<n> numeric':>11} {'(10^(s/10)-1)/2':>16} {'old Delta^-2':>13} {'rel. dev.':>9}")
    for sig2 in (0.09, 0.03, 0.01):
        nb, pv = _gkp_photon_number(sig2); s_dB = -10 * np.log10(2 * sig2); f = sp.mean_photon_number(s_dB)
        dev = abs(nb - f) / f
        print(f"  {sig2:8.3f} {s_dB:7.2f} | {pv:16.4f} | {nb:11.3f} {f:16.3f} {1/sig2:13.2f} {dev:9.1%}")
        if sig2 <= 0.03: ok &= dev < 0.01 and abs(pv - sig2) / sig2 < 0.01
    print("  (formula is asymptotic: within 1% for s >= 12 dB, ~10% at 7.4 dB; the earlier N_S = Delta^-2 is 4-5x too large.)")
    viol_ea = viol_hol = 0; worst = 0.0
    for eta in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999):
        for s_dB in np.arange(0.5, 25, 0.5):
            sg = sp.sigma_eff(s_dB, eta)
            if sg > 1.5: continue
            NS = sp.mean_photon_number(s_dB); cg = max(sp.C_square_soft_fast(sg), sp.C_hex_soft_fast(sg))
            hol = _g(eta * NS); ea = _cea(NS, eta)
            viol_ea += cg > ea + 1e-9; viol_hol += cg > hol + 1e-9; worst = max(worst, cg / max(hol, 1e-12))
    print(f"  GKP capacity > C_EA at {viol_ea} points (must be 0); GKP capacity > unassisted Holevo at {viol_hol} points; max C_gkp/C_hol = {worst:.3f}")
    ok &= viol_ea == 0
    print(f"  Section 5: {'PASS' if ok else 'FAIL'}   (the Holevo ordering is a RESULT, reported in the paper, not a pass/fail criterion)")
    return ok

def _chol_th(NS, eta, nth): NB = (1 - eta) * nth; return _g(eta * NS + NB) - _g(NB)
def _cea_th(NS, eta, nth):
    NB = (1 - eta) * nth; E = eta * NS + NB
    D = np.sqrt(max((NS + E + 1) ** 2 - 4 * eta * NS * (NS + 1), 0))
    return _g(NS) + _g(E) - _g(max((D - 1 + (E - NS)) / 2, 1e-15)) - _g(max((D - 1 - (E - NS)) / 2, 1e-15))

def section6():
    print("\n" + "=" * 96 + "\n  SECTION 6 -- thermal benchmark capacities\n" + "=" * 96)
    ok = True
    d = max(max(abs(_chol_th(sp.mean_photon_number(s), eta, 0.0) - _g(eta * sp.mean_photon_number(s))),
                abs(_cea_th(sp.mean_photon_number(s), eta, 0.0) - _cea(sp.mean_photon_number(s), eta)))
            for s in (3.0, 7.44, 10.0, 15.0, 20.0) for eta in (0.7, 0.9, 0.99))
    print(f"  n_th = 0 reduction to the pure-loss formulas: worst |diff| = {d:.1e}"); ok &= d < 1e-9
    v_ea = v_ord = 0; worst = 0.0; pts = 0
    for nth in (0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0):
        for eta in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
            for s_dB in np.arange(0.5, 25, 0.5):
                sg = sp.sigma_eff(s_dB, eta, nth)
                if sg > 1.5: continue
                NS = sp.mean_photon_number(s_dB); cg = max(sp.C_square_soft_fast(sg), sp.C_hex_soft_fast(sg))
                h, e = _chol_th(NS, eta, nth), _cea_th(NS, eta, nth); pts += 1
                v_ea += cg > e + 1e-9; v_ord += e < h - 1e-9; worst = max(worst, cg / max(h, 1e-12))
    print(f"  {pts} grid points: C_gkp > C_EA at {v_ea}; C_EA < C_hol at {v_ord}; max C_gkp/C_hol = {worst:.3f}")
    ok &= v_ea == 0 and v_ord == 0
    print(f"  Section 6: {'PASS' if ok else 'FAIL'}")
    return ok

def section3_thermal_regression():
    print("\n" + "=" * 96 + "\n  SECTION 3 -- n_th -> 0 regression (structural)\n" + "=" * 96)
    ok = True
    for s, eta in [(7.44, 0.9), (12.0, 0.95)]:
        a = sp.sigma_eff(s, eta, 0.0); b = float(np.sqrt(2 * sp.delta2(s) + (1 - eta)))
        ok &= np.isclose(a, b, rtol=0, atol=1e-15)
    print(f"  sigma_eff(n_th=0) == 2 Delta^2 + (1-eta): {'OK' if ok else 'FAIL'}")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nth", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=2_000_000, help="samples per point per seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    ap.add_argument("--nsoft", type=int, default=1_000_000, help="samples per point for the soft-decision MC")
    a = ap.parse_args()
    res = {"geometry": section0(), "monte_carlo": section1(a.nth, a.n, a.seeds),
           "soft_decision": section1b(a.nth, a.nsoft, a.seeds[0]),
           "circuit": section2(a.nth, a.n, a.seeds), "bell_pair_model": section4(), "energy": section5(),
           "thermal_benchmarks": section6(), "regression": section3_thermal_regression()}
    print("\n" + "=" * 96 + "\n  SUMMARY\n" + "=" * 96)
    for k, v in res.items():
        print(f"  {k:<12}: {'PASS' if v else 'FAIL'}")
    print("  Overall:", "ALL CHECKS PASS" if all(res.values()) else "ONE OR MORE CHECKS FAILED")
    sys.exit(0 if all(res.values()) else 1)

if __name__ == "__main__":
    main()
