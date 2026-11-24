"""Closed forms and exact samplers for halfspace caps of the L2 ball (PREREG 3.1, 3.2).

A cap is {x in ball(R) : u.x > c R} for a unit vector u. With s = u.x / R:
  ball mass    p(c)   = 0.5 I_{1-c^2}((d+1)/2, 1/2)
  sphere mass  p_s(c) = 0.5 I_{1-c^2}((d-1)/2, 1/2)    (uniform on the boundary sphere)
for c >= 0. Under the uniform ball law, s has density proportional to (1 - s^2)^((d-1)/2) and,
given s, the component orthogonal to u is uniform in the (d-1)-ball of radius R sqrt(1 - s^2).
"""
import math

import numpy as np
from scipy import optimize, special


def ball_mass(c, d):
    c = np.asarray(c, dtype=float)
    if np.any(c < 0):
        raise ValueError("c must be >= 0")
    return 0.5 * special.betainc((d + 1) / 2, 0.5, 1 - c * c)


def sphere_mass(c, d):
    c = np.asarray(c, dtype=float)
    if np.any(c < 0):
        raise ValueError("c must be >= 0")
    return 0.5 * special.betainc((d - 1) / 2, 0.5, 1 - c * c)


def c_for_ball_mass(p, d):
    """Solve ball_mass(c) = p for c in [0, 1); p in (0, 0.5]."""
    if not 0 < p <= 0.5:
        raise ValueError("p must be in (0, 0.5]")
    if p == 0.5:
        return 0.0
    # Direct inverse: 1 - c^2 = I^{-1}(2p); refine with brentq on log mass for full relative accuracy.
    c0 = math.sqrt(max(0.0, 1 - special.betaincinv((d + 1) / 2, 0.5, 2 * p)))
    f = lambda c: math.log(ball_mass(c, d)) - math.log(p)
    lo, hi = max(0.0, c0 - 1e-3), min(1 - 1e-15, c0 + 1e-3)
    if f(lo) < 0 or f(hi) > 0:
        lo, hi = 0.0, 1 - 1e-15
    return optimize.brentq(f, lo, hi, xtol=1e-15, rtol=1e-15)


def median_s_in_cap(c, d):
    """Median of s = u.x / R for x uniform in the cap {s > c}."""
    m = ball_mass(c, d)
    return s_from_upper_mass(0.5 * m, d)


def s_from_upper_mass(q, d):
    """s such that ball_mass(s) = q (inverse upper tail)."""
    q = np.asarray(q, dtype=float)
    return np.sqrt(np.clip(1 - special.betaincinv((d + 1) / 2, 0.5, 2 * q), 0, 1))


# ---------------- samplers ----------------
def sample_ball(rng, n, d, R):
    g = rng.standard_normal((n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g * (R * rng.random(n) ** (1.0 / d))[:, None]


def sample_sphere(rng, n, d, R):
    g = rng.standard_normal((n, d))
    return R * g / np.linalg.norm(g, axis=1, keepdims=True)


def sample_cap(rng, n, d, R, u, c):
    """Exact sample of x uniform in {x in ball(R): u.x > c R} (inverse-CDF on s, uniform slice)."""
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    m = ball_mass(c, d)
    s = s_from_upper_mass(rng.random(n) * m, d)
    s = np.maximum(s, c)  # guard rounding at the boundary
    # Uniform in the orthogonal (d-1)-ball of radius R sqrt(1 - s^2).
    z = rng.standard_normal((n, d))
    z -= np.outer(z @ u, u)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    rho = R * np.sqrt(1 - s * s) * rng.random(n) ** (1.0 / (d - 1))
    return np.outer(R * s, u) + z * rho[:, None]


# ---------------- independent checks (do not use the incomplete-beta identity) ----------------
def _log_ball_volume(k, r):
    return (k / 2) * math.log(math.pi) + k * math.log(r) - special.gammaln(k / 2 + 1)


def _log_sphere_area(k, r):
    """Area of the k-dimensional sphere of radius r embedded in R^(k+1)."""
    return math.log(2) + ((k + 1) / 2) * math.log(math.pi) + k * math.log(r) - special.gammaln((k + 1) / 2)


def cap_mass_importance(rng, c, d, n=200_000, sphere=False):
    """Importance-sampling estimate of the cap mass using only ball/sphere volume formulas.

    Integrates the slice measure over t = u.x in [cR, R] with a truncated-exponential proposal
    fitted at the cap edge. Returns (estimate, relative standard error).
    """
    R = 1.0  # masses are scale-free
    tau = c * R
    rho2 = lambda t: np.maximum(R * R - t * t, 1e-300)
    k = d - 1 if not sphere else d - 2
    # Rate of the log slice measure at the edge, for the proposal.
    rate = max(k * tau / max(rho2(tau), 1e-300), 1.0 / (R - tau))
    a = R - tau
    uu = rng.random(n)
    # Truncated exponential on [0, a]: t = tau + x.
    x = -np.log1p(-uu * (1 - math.exp(-rate * a))) / rate
    t = tau + x
    log_q = math.log(rate) - rate * x - math.log1p(-math.exp(-rate * a))
    if not sphere:
        log_f = (k / 2) * np.log(rho2(t)) + _log_ball_volume(k, 1.0) - _log_ball_volume(d, R)
    else:
        # Surface element at height t: area of (d-2)-sphere of radius rho times R / rho.
        log_f = (k / 2) * np.log(rho2(t)) + _log_sphere_area(k, 1.0) + math.log(R) - 0.5 * np.log(rho2(t)) \
            - _log_sphere_area(d - 1, R)
    w = np.exp(log_f - log_q)
    est = w.mean()
    return est, w.std(ddof=1) / math.sqrt(n) / est
