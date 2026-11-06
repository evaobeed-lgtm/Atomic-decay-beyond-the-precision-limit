# Atomic decay beyond the precision limit

**Is the 10⁻⁹ plateau in the decay of a two-state atom a numerical artifact? No. It is real, and this project shows where it comes from.**

This is a follow-up to section 6.2 of

> Eva Obeed, *Numerical Modelling of Some Quantum Systems*, master thesis, NTNU, Trondheim, May 2016.

## The open question

The thesis models an excited two-level atom coupled to N + 1 equally spaced photon modes and computes the probability p₀(t) that the atom is still excited. The decay is exponential at first, as the Fermi Golden Rule predicts. At long times p₀(t) stops falling and levels off near 10⁻⁹ before the revivals set in (figure 6.3 of the thesis). The thesis comments:

> "The plateau of minimum p₀(t) ≈ 10⁻⁹ which seems to develop is probably an artifact of finite numerical accuracy."

and the summary names double precision as the limiting factor of the model. This project tests that statement.

## Method

p₀(t) is computed in four independent ways, for the thesis parameters (N = 2000, λ = 0.1, Eₐ = 1, E_max = 8):

| Method | Precision | Purpose |
|---|---|---|
| Full diagonalization, `float64` | 16 digits | the thesis calculation |
| Full diagonalization, `float32` | 7 digits | shows what a genuine precision floor looks like |
| Secular equation with `mpmath` | 50 digits | reference free of rounding error |
| Continuum integral, N → ∞ | 16 digits | removes the discreteness of the photon spectrum |

**Secular equation.** The Hamiltonian is an arrowhead matrix, so its eigenvalues are the roots of

```
f(E) = Ea - E - g² Σ_n 1/(E_n - E) = 0
```

and the weight of the atomic state in each eigenvector is w = −1/f′(E). Each double-precision eigenvalue is polished to 50 digits by Newton iteration, and the survival amplitude M₀(t) = Σ w_k exp(−iE_k t) is summed at 50 digits.

**Continuum limit.** For N → ∞ the atom has the spectral function

```
A(E) = λ² / [ (E - Ea - λ² ln(E/(Emax - E)))² + (πλ²)² ],   0 < E < Emax
```

and M₀(t) is its Fourier transform, evaluated on a grid of two million points.

## Results

![Four calculations of the survival probability](figures/plateau_four_methods.png)

**1. The plateau is not a rounding artifact.** In the plateau region (30 < Γ₀t < 80) the double-precision result and the 50-digit result agree to 1.0 × 10⁻¹⁷, eight orders of magnitude below the plateau itself (5.6 × 10⁻¹⁰ to 1.8 × 10⁻⁹). The eigenvalues from `eigh` are accurate to 4 × 10⁻¹⁵ and the weights to 3 × 10⁻¹⁴.

![Numerical error against signal](figures/precision_errors.png)

The single-precision run shows what a real precision floor looks like: noisy, and at the level of its own rounding error (about 10⁻⁸).

**2. The plateau is the edge of the photon band.** The photon energies stop abruptly at E = 0 and E = E_max, but the spectral function A(E) is not zero there. A sharp cut-off in energy gives a slowly decaying tail in time. For finite N the modes near the edge add up to a geometric series,

```
M₀(t) ≈ A₀ ΔE / (1 - exp(-i ΔE t))  =  A₀/(i t)  +  A₀ ΔE/2  + ...
```

with ΔE = E_max/N and A₀ the spectral function at the edge. This formula reproduces the 50-digit result to within 5% over the whole plateau, and its pole at ΔE·t = 2π is the revival. It has two parts:

- **A₀/(it)** is a power-law tail that survives for N → ∞. Exponential decay cannot continue forever when the energy spectrum is bounded.
- **A₀ΔE/2** is a constant proportional to 1/N, caused by the discreteness of the spectrum. For N = 2000 it is of similar size to the power-law term, which is why the curve looks flat.

**3. The continuum tail is a power law.** Without revivals (N → ∞) the survival probability falls as p₀ ∝ t^(−1.96) for 25 < Γ₀t < 2000, against the predicted t⁻² with logarithmic corrections.

![Power law tail](figures/power_law_tail.png)

**4. The tail depends on the distance to the band edge.** Moving the atomic level away from the lower edge lowers the tail, as the formula predicts:

| Eₐ | p₀ at Γ₀t = 40 (continuum) | band edge estimate |
|---|---|---|
| 0.5 | 8.80 × 10⁻⁹ | 8.43 × 10⁻⁹ |
| 1.0 | 3.72 × 10⁻¹⁰ | 3.66 × 10⁻¹⁰ |
| 2.0 | 2.23 × 10⁻¹¹ | 2.21 × 10⁻¹¹ |
| 4.0 | 4.02 × 10⁻¹² | 4.01 × 10⁻¹² |

## Conclusion

Double precision was not the limit of the thesis calculation: its results in figure 6.3 are correct to at least eight significant digits. The plateau is a property of the model, a mixture of the power-law decay that any system with a bounded spectrum must show at long times and a finite-N correction. The thesis observation that the plateau is roughly "independent of N" is explained by the N-independent power-law part.

## Usage

```bash
pip install -r requirements.txt
python decay_precision.py
```

The run takes about three minutes; most of it is the 50-digit calculation. Figures are written to `figures/`.

The functions can be used separately:

```python
import numpy as np
from decay_precision import (model, survival_eigh, secular_spectrum_mp,
                             survival_mp, survival_continuum, edge_estimate)

Ea, Eph, g = model(N=500)
t = np.linspace(0, 500, 100)
p_double, E, w = survival_eigh(Ea, Eph, g, t)
E_mp, w_mp = secular_spectrum_mp(Ea, Eph, g, E, digits=50)
p_exact = survival_mp(E_mp, w_mp, t)
```

## Files

| File | Content |
|---|---|
| `decay_precision.py` | the four methods, the band edge formula and the experiments |
| `figures/` | the three figures shown above |
| `requirements.txt` | NumPy, SciPy, Matplotlib, mpmath |

## Limitations

- The band edge formula evaluates A(E) at a distance 1/t from the edge, a simple treatment of its logarithmic variation. It is accurate to about 5% here (12% at the latest times in the continuum), not exact.
- A real atom couples to an unbounded photon spectrum with an energy-dependent coupling. The plateau height found here belongs to this model, with its flat coupling and sharp band edges.
