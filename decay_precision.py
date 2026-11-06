#!/usr/bin/env python3
"""
Is the 1e-9 plateau in the atomic decay a numerical artifact?
=============================================================

Follow-up to section 6.2 of

    Eva Obeed, "Numerical Modelling of Some Quantum Systems",
    Master thesis, NTNU, Trondheim, May 2016.

Figure 6.3 of the thesis shows the survival probability p0(t) of an excited
two-level atom levelling off near 1e-9 before the revivals, and the text
says this "is probably an artifact of finite numerical accuracy". This
program tests that statement by computing p0(t) in four independent ways:

  1. double precision, full diagonalization (the thesis method),
  2. single precision, full diagonalization (to show what a real
     precision floor looks like),
  3. arbitrary precision (50 digits) from the secular equation of the
     arrowhead matrix,
  4. the continuum limit N -> infinity, by direct numerical integration
     of the spectral function.

Model (eq. 6.9): H is an (N+2) x (N+2) arrowhead matrix with diagonal
(Ea, E_0, ..., E_N), E_n evenly spaced on [0, Emax], and coupling g between
the atom and every photon mode, g = lamda * sqrt(Emax / N).

Secular equation. E is an eigenvalue of H exactly when

    f(E) = Ea - E - g^2 sum_n 1 / (E_n - E) = 0 ,

and the weight of the atomic state in that eigenvector is

    w = 1 / (1 + g^2 sum_n 1 / (E_n - E)^2) = -1 / f'(E) .

The survival amplitude is M0(t) = sum_k w_k exp(-i E_k t), p0 = |M0|^2.
"""

import os
import time

import numpy as np
from scipy.linalg import eigh
import mpmath as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def model(N=2000, lamda=0.1, Ea=1.0, Emax=8.0):
    """Return (Ea, Eph, g) for the thesis parameters."""
    Eph = np.linspace(0.0, Emax, N + 1)
    g = lamda * np.sqrt(Emax / N)
    return Ea, Eph, g


def arrowhead(Ea, Eph, g, dtype=np.float64):
    H = np.diag(np.concatenate(([Ea], Eph))).astype(dtype)
    H[0, 1:] = g
    H[1:, 0] = g
    return H


def golden_rule_rate(lamda=0.1):
    return 2 * np.pi * lamda ** 2


# ---------------------------------------------------------------------------
# Methods 1 and 2: full diagonalization in double / single precision
# ---------------------------------------------------------------------------

def survival_eigh(Ea, Eph, g, t, dtype=np.float64):
    """p0(t) by diagonalizing H in the given floating point type."""
    E, S = eigh(arrowhead(Ea, Eph, g, dtype))
    w = (S[0, :] ** 2).astype(dtype)
    Et = np.outer(t.astype(dtype), E)
    M = np.cos(Et) @ w - 1j * (np.sin(Et) @ w)
    return np.abs(M) ** 2, E.astype(float), w.astype(float)


# ---------------------------------------------------------------------------
# Method 3: arbitrary precision from the secular equation
# ---------------------------------------------------------------------------

def secular_spectrum_mp(Ea, Eph, g, E_guess, digits=50):
    """Eigenvalues and atomic weights to `digits` decimals.

    Each double precision eigenvalue is polished by Newton iteration on
    f(E), carried out with mpmath. Newton converges quadratically, so a few
    steps take 1e-15 accuracy to 1e-50.
    """
    mp.mp.dps = digits + 10
    Ea_mp = mp.mpf(Ea)
    g2 = mp.mpf(g) ** 2
    Eph_mp = [mp.mpf(float(e)) for e in Eph]
    # bracketing intervals: one root below the band, one between each pair
    # of photon energies, one above the band
    lo = [mp.mpf("-inf")] + Eph_mp
    hi = Eph_mp + [mp.mpf("inf")]
    tol = mp.mpf(10) ** (-digits)

    energies, weights = [], []
    for k, E0 in enumerate(sorted(E_guess)):
        E = mp.mpf(float(E0))
        for _ in range(60):
            s1 = mp.mpf(0)
            s2 = mp.mpf(0)
            for en in Eph_mp:
                r = 1 / (en - E)
                s1 += r
                s2 += r * r
            f = Ea_mp - E - g2 * s1
            df = -1 - g2 * s2
            step = f / df
            E_new = E - step
            if not (lo[k] < E_new < hi[k]):      # safeguard: stay in bracket
                raise RuntimeError("Newton left its bracket at root %d" % k)
            E = E_new
            if abs(step) < tol:
                break
        else:
            raise RuntimeError("Newton did not converge at root %d" % k)
        s2 = mp.fsum(1 / (en - E) ** 2 for en in Eph_mp)
        energies.append(E)
        weights.append(1 / (1 + g2 * s2))
    return energies, weights


def survival_mp(energies, weights, t):
    """p0(t) from high-precision eigenvalues and weights."""
    p0 = np.empty(len(t))
    for i, ti in enumerate(t):
        ti = mp.mpf(float(ti))
        M = mp.fsum(w * mp.expj(-E * ti) for E, w in zip(energies, weights))
        p0[i] = float(abs(M) ** 2)
    return p0


# ---------------------------------------------------------------------------
# Method 4: the continuum limit
# ---------------------------------------------------------------------------

def spectral_function(E, lamda=0.1, Ea=1.0, Emax=8.0):
    """Spectral function of the atomic state for N -> infinity.

    The atom acquires the self energy
        Sigma(E) = lamda^2 ln(E / (Emax - E)) - i pi lamda^2 ,  0 < E < Emax,
    so that A(E) = lamda^2 / [(E - Ea - Re Sigma)^2 + (pi lamda^2)^2].
    A is a Lorentzian of width Gamma0 = 2 pi lamda^2 near Ea, but it does
    not vanish towards the band edges E = 0 and E = Emax, where it is cut
    off sharply.
    """
    l2 = lamda ** 2
    with np.errstate(divide="ignore"):
        re_sigma = l2 * np.log(E / (Emax - E))
    return l2 / ((E - Ea - re_sigma) ** 2 + (np.pi * l2) ** 2)


def survival_continuum(t, lamda=0.1, Ea=1.0, Emax=8.0, npts=2 ** 21):
    """p0(t) for N -> infinity: M0(t) = int_0^Emax A(E) exp(-iEt) dE,
    by the midpoint rule on a very fine grid (far finer than 1/t)."""
    dE = Emax / npts
    E = (np.arange(npts) + 0.5) * dE
    A = spectral_function(E, lamda, Ea, Emax)
    norm = A.sum() * dE                       # ~1; rest sits in bound states
    p0 = np.empty(len(t))
    for i, ti in enumerate(t):
        p0[i] = np.abs(np.sum(A * np.exp(-1j * E * ti)) * dE) ** 2
    return p0, norm


def edge_estimate(t, lamda=0.1, Ea=1.0, Emax=8.0, N=None):
    """Contribution of the sharp band edges to M0(t), returned as |M0|^2.

    The spectral function A(E) does not vanish at the band edges; it stops
    abruptly, with value A_0 at E = 0. Its Fourier transform therefore has a
    slowly decaying tail.

    Continuum (N = None): integrating by parts, the edge at E = 0
    contributes A_0 / (i t), so p0 ~ 1 / t^2.

    Finite N: the modes near the edge have weights A_0 * dE and phases
    exp(-i n dE t), a geometric series,
        A_0 dE / (1 - exp(-i dE t)) = A_0 / (i t) + A_0 dE / 2 + ...
    The first term is the continuum tail, the second is a constant that
    comes from the discreteness of the spectrum, and the pole at
    dE t = 2 pi is the revival.

    A(E) varies only logarithmically near the edges, so it is evaluated at
    a distance 1/t from each edge. Both edges are added with their phases.
    """
    t = np.asarray(t, dtype=float)
    eps = 1.0 / t
    A_low = spectral_function(eps, lamda, Ea, Emax)
    A_high = spectral_function(Emax - eps, lamda, Ea, Emax)
    if N is None:
        low, high = 1 / (1j * t), -1 / (1j * t)
    else:
        dE = Emax / N
        low = dE / (1 - np.exp(-1j * dE * t))
        high = dE / (1 - np.exp(1j * dE * t))
    M = A_low * low + A_high * high * np.exp(-1j * Emax * t)
    return np.abs(M) ** 2


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run(N=2000, n_times=260, digits=50, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    lamda = 0.1
    Ea, Eph, g = model(N=N, lamda=lamda)
    G0 = golden_rule_rate(lamda)
    t = np.linspace(0.0, 90.0 / G0, n_times)          # up to Gamma0 t = 90

    print("N = %d photon modes, Gamma0 = %.6f" % (N, G0))
    t0 = time.time()
    p64, E64, w64 = survival_eigh(Ea, Eph, g, t, np.float64)
    print("double precision eigh     : %6.1f s" % (time.time() - t0))
    t0 = time.time()
    p32, _, _ = survival_eigh(Ea, Eph, g, t, np.float32)
    print("single precision eigh     : %6.1f s" % (time.time() - t0))
    t0 = time.time()
    E_mp, w_mp = secular_spectrum_mp(Ea, Eph, g, E64, digits)
    pmp = survival_mp(E_mp, w_mp, t)
    print("%d-digit secular equation : %6.1f s" % (digits, time.time() - t0))
    t0 = time.time()
    pc, norm = survival_continuum(t, lamda)
    print("continuum integral        : %6.1f s" % (time.time() - t0))

    # -- accuracy of the double precision ingredients -----------------------
    E_ref = np.array([float(e) for e in E_mp])
    w_ref = np.array([float(w) for w in w_mp])
    print("\nAccuracy of double precision eigh against %d digits:" % digits)
    print("  max |E - E_ref|            = %.2e" % np.abs(E64 - E_ref).max())
    print("  max |w - w_ref|            = %.2e" % np.abs(w64 - w_ref).max())
    print("  |1 - sum of weights| (reference) = %.1e"
          % abs(float(1 - mp.fsum(w_mp))))
    print("  weight of bound state below band = %.3e" % w_ref[0])
    print("  weight of bound state above band = %.3e" % w_ref[-1])

    # -- the plateau ------------------------------------------------------------
    x = G0 * t
    plateau = (x > 30) & (x < 80)
    err64 = np.abs(p64 - pmp)
    err32 = np.abs(p32 - pmp)
    print("\nPlateau region 30 < Gamma0 t < 80:")
    print("  p0, %d digits      : %.3e .. %.3e"
          % (digits, pmp[plateau].min(), pmp[plateau].max()))
    print("  p0, double         : %.3e .. %.3e"
          % (p64[plateau].min(), p64[plateau].max()))
    print("  p0, continuum      : %.3e .. %.3e"
          % (pc[plateau].min(), pc[plateau].max()))
    print("  max |double - ref| : %.2e   (relative %.1e)"
          % (err64[plateau].max(), (err64 / pmp)[plateau].max()))
    print("  max |single - ref| : %.2e" % err32[plateau].max())
    print("  max |continuum/ref - 1| : %.3f"
          % np.abs(pc / pmp - 1)[plateau].max())
    est = edge_estimate(t[1:], lamda, N=N)
    ratio = (est / pmp[1:])[plateau[1:]]
    print("  band edge formula / ref  : %.2f .. %.2f"
          % (ratio.min(), ratio.max()))
    print("  continuum spectral weight inside band = %.12f" % norm)

    # -- figure 1: the four calculations ------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.semilogy(x, p32, color="0.6", lw=1, label="single precision (7 digits)")
    ax.semilogy(x, p64, "b", lw=2.2, label="double precision (16 digits)")
    ax.semilogy(x, pmp, "y", lw=1, label="%d digits" % digits)
    ax.semilogy(x, pc, "r--", lw=1.2, label=r"continuum limit $N\to\infty$")
    ax.semilogy(x[1:], est, "k:", lw=1.2, label="band edge formula (finite N)")
    ax.semilogy(x, np.exp(-x), "g-.", lw=1, label=r"Golden rule $e^{-\Gamma_0 t}$")
    ax.set(xlim=(0, 90), ylim=(1e-12, 2), xlabel=r"$\Gamma_0\,t$",
           ylabel=r"$p_0(t)$",
           title="Survival probability, N = %d: the plateau is real" % N)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(os.path.join(outdir, "plateau_four_methods.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # -- figure 2: where the precision limits actually are ------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.semilogy(x, pmp, "k", lw=1.5, label=r"$p_0(t)$, %d digits" % digits)
    ax.semilogy(x, np.maximum(err32, 1e-40), color="0.6", lw=1,
                label="error of single precision")
    ax.semilogy(x, np.maximum(err64, 1e-40), "b", lw=1,
                label="error of double precision")
    ax.set(xlim=(0, 90), ylim=(1e-26, 2), xlabel=r"$\Gamma_0\,t$",
           ylabel="probability",
           title="Numerical error compared with the signal")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(os.path.join(outdir, "precision_errors.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    return dict(t=t, p64=p64, p32=p32, pmp=pmp, pc=pc)


def power_law_check(outdir="figures"):
    """In the continuum limit the tail should fall like 1/t^2 (up to
    logarithms): fit the slope of log p0 against log t."""
    lamda = 0.1
    G0 = golden_rule_rate(lamda)
    t = np.geomspace(25 / G0, 2000 / G0, 60)
    pc, _ = survival_continuum(t, lamda)
    est = edge_estimate(t, lamda)
    slope = np.polyfit(np.log(t), np.log(pc), 1)[0]
    print("\nContinuum tail for 25 < Gamma0 t < 2000:")
    print("  fitted power law p0 ~ t^(%.3f)   (band edge theory: -2 "
          "with logarithmic corrections)" % slope)
    print("  band edge estimate / continuum: %.2f .. %.2f"
          % ((est / pc).min(), (est / pc).max()))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.loglog(G0 * t, pc, "r", lw=1.5, label=r"continuum limit")
    ax.loglog(G0 * t, est, "k:", lw=1.5, label="band edge estimate")
    ax.loglog(G0 * t, pc[0] * (t / t[0]) ** -2, "b--", lw=1,
              label=r"$\propto t^{-2}$")
    ax.set(xlabel=r"$\Gamma_0\,t$", ylabel=r"$p_0(t)$",
           title="Long time tail without revivals: power law, not exponential")
    ax.legend()
    fig.savefig(os.path.join(outdir, "power_law_tail.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    return slope


def atomic_energy_dependence():
    """The tail should depend on how far the atomic level sits from the
    band edge at E = 0. Vary Ea and compare with the band edge estimate."""
    print("\nPlateau height at Gamma0 t = 40 for different atomic energies "
          "(continuum):")
    G0 = golden_rule_rate(0.1)
    t = np.array([40 / G0])
    for Ea in (0.5, 1.0, 2.0, 4.0):
        pc, _ = survival_continuum(t, 0.1, Ea=Ea)
        est = edge_estimate(t, 0.1, Ea=Ea)
        print("  Ea = %.1f: p0 = %.3e   band edge estimate %.3e"
              % (Ea, pc[0], est[0]))


if __name__ == "__main__":
    run()
    power_law_check()
    atomic_energy_dependence()
