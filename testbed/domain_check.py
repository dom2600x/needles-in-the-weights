"""domain_check.py: numerical check behind PREREG section 3.1 (box versus ball). Becomes a G1 unit test.
Needs numpy and scipy; runs in about 5 s: python3 domain_check.py

How hard is a rare halfspace 'needle' for a random-vertex sampler (box domain)
versus a boundary sampler (ball domain)?  d = 32.

Box: x ~ U[-1,1]^d, needle region {u.x > tau}.
  p   = P_unif(u.x > tau)   -> saddlepoint (Lugannani-Rice), checked by Monte Carlo at a moderate tau
  p_v = P_vertex(u.s > tau) -> exact, meet-in-the-middle over 2^16 x 2^16 sign patterns
Ball: x ~ U(ball of radius R), region {u.x/R > c}.
  p_ball   = 0.5 * I_{1-c^2}((d+1)/2, 1/2)
  p_sphere = 0.5 * I_{1-c^2}((d-1)/2, 1/2)   (sampler on the boundary sphere)
"""
import numpy as np
from scipy import special, stats, optimize

d = 32
rng = np.random.default_rng(0)


def box_tail_saddle(u, tau):
    """Lugannani-Rice approximation of P(sum u_i x_i > tau), x_i ~ U[-1,1]."""
    a = np.abs(u)

    def K(t):
        z = a * t
        return np.sum(np.where(z < 1e-8, z**2 / 6, np.log(np.sinh(z) / z)))

    def K1(t):
        z = a * t
        return np.sum(a * np.where(z < 1e-8, z / 3, 1 / np.tanh(z) - 1 / z))

    def K2(t):
        z = a * t
        return np.sum(a**2 * np.where(z < 1e-8, 1 / 3, 1 / z**2 - 1 / np.sinh(z) ** 2))

    t = optimize.brentq(lambda t: K1(t) - tau, 1e-9, 5000.0)
    w = np.sqrt(2 * (t * tau - K(t)))
    v = t * np.sqrt(K2(t))
    return stats.norm.sf(w) + stats.norm.pdf(w) * (1 / v - 1 / w)


def tau_for_p(u, p):
    f = lambda tau: np.log(box_tail_saddle(u, tau)) - np.log(p)
    return optimize.brentq(f, 0.3, 0.9 * np.abs(u).sum())


def vertex_mass(u, tau):
    """Exact P over uniform random sign vertices of u.s > tau (meet in the middle)."""
    h = d // 2
    def sums(v):
        s = np.zeros(1)
        for x in v:
            s = np.concatenate([s + x, s - x])
        return s
    A = sums(u[:h]); B = np.sort(sums(u[h:]))
    cnt = len(B) - np.searchsorted(B, tau - A, side="right")
    return cnt.sum() / (len(A) * len(B))


print("=== BOX domain, dense random unit u, d=32 ===")
# Monte-Carlo check of the saddlepoint at a moderate tail
u = rng.standard_normal(d); u /= np.linalg.norm(u)
tau_mc = 2.0
x = rng.uniform(-1, 1, size=(4_000_000, d))
print(f"MC check: tau={tau_mc}: MC p={np.mean(x @ u > tau_mc):.3e}  saddle p={box_tail_saddle(u, tau_mc):.3e}")
del x
print(f"{'p target':>9} {'tau':>6} {'||u||_1':>7} {'p_v (exact)':>12} {'p_v/p':>9} {'S1b hit @Q=1e3':>16} {'@Q=1e6':>8}")
for p in (1e-5, 1e-7, 1e-9):
    for k in range(3):
        u = rng.standard_normal(d); u /= np.linalg.norm(u)
        tau = tau_for_p(u, p)
        pv = vertex_mass(u, tau)
        print(f"{p:9.0e} {tau:6.3f} {np.abs(u).sum():7.3f} {pv:12.3e} {pv/p:9.1e} "
              f"{1-(1-pv)**1e3:16.3f} {1-(1-pv)**1e6:8.3f}")

print()
print("=== BALL domain (uniform in L2 ball), d=32 ===")
# Monte-Carlo check of the closed form at a moderate tail (unit ball, so c = tau / R is the threshold on u.x)
g = rng.standard_normal((4_000_000, d)); g /= np.linalg.norm(g, axis=1, keepdims=True)
r = rng.uniform(size=(4_000_000, 1)) ** (1 / d)
xb = g * r
c_mc = 0.45
print(f"MC check: c={c_mc}: MC ball p={np.mean(xb[:, 0] > c_mc):.3e}  closed form={0.5*special.betainc((d+1)/2, 0.5, 1-c_mc**2):.3e};"
      f"  MC sphere p={np.mean(g[:, 0] > c_mc):.3e}  closed form={0.5*special.betainc((d-1)/2, 0.5, 1-c_mc**2):.3e}")
del g, xb, r
print(f"{'p target':>9} {'c=tau/R':>8} {'p_sphere':>11} {'p_sphere/p':>11} {'sphere hit @Q=1e3':>18} {'@Q=1e6':>8}")
for p in (1e-5, 1e-7, 1e-9, 1e-10):
    c = optimize.brentq(lambda c: np.log(0.5*special.betainc((d+1)/2, 0.5, 1-c**2)) - np.log(p), 1e-3, 0.999)
    ps = 0.5*special.betainc((d-1)/2, 0.5, 1-c**2)
    print(f"{p:9.0e} {c:8.4f} {ps:11.3e} {ps/p:11.2f} {1-(1-ps)**1e3:18.2e} {1-(1-ps)**1e6:8.3f}")
