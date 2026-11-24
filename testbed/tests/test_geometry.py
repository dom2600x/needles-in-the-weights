import numpy as np
import pytest
from scipy import stats

from testbed import constants as K
from testbed import geometry as geo


@pytest.mark.parametrize("d", K.D_VALUES)
@pytest.mark.parametrize("p", K.P_GRID)
def test_solver_inverts_closed_form(d, p):
    c = geo.c_for_ball_mass(p, d)
    assert abs(float(geo.ball_mass(c, d)) / p - 1) < 1e-9


@pytest.mark.parametrize("d", K.D_VALUES)
@pytest.mark.parametrize("p", [1e-1, 1e-4, 1e-8, 1e-12])
def test_importance_sampling_matches_closed_form(d, p):
    rng = np.random.default_rng(0)
    c = geo.c_for_ball_mass(p, d)
    for sphere, closed in ((False, geo.ball_mass(c, d)), (True, geo.sphere_mass(c, d))):
        est, rse = geo.cap_mass_importance(rng, c, d, sphere=sphere)
        assert abs(est / float(closed) - 1) < max(0.01, 5 * rse)


def test_monte_carlo_matches_closed_form():
    rng = np.random.default_rng(1)
    d, c = 32, 0.3
    x = geo.sample_ball(rng, 2_000_000, d, 1.0)
    assert abs(np.mean(x[:, 0] > c) / float(geo.ball_mass(c, d)) - 1) < 0.03
    y = geo.sample_sphere(rng, 2_000_000, d, 1.0)
    assert abs(np.mean(y[:, 0] > c) / float(geo.sphere_mass(c, d)) - 1) < 0.03


def test_ball_sampler_variance_is_one_third():
    d = 32
    x = geo.sample_ball(np.random.default_rng(2), 400_000, d, K.radius(d))
    assert abs(x.var(axis=0).mean() - 1 / 3) < 0.005
    assert np.all(np.linalg.norm(x, axis=1) <= K.radius(d) * (1 + 1e-12))


@pytest.mark.parametrize("p", [1e-2, 1e-12])
def test_cap_sampler_is_exact(p):
    rng = np.random.default_rng(3)
    d, R = 32, K.radius(32)
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    c = geo.c_for_ball_mass(p, d)
    x = geo.sample_cap(rng, 100_000, d, R, u, c)
    assert np.all(x @ u >= c * R * (1 - 1e-12))
    assert np.all(np.linalg.norm(x, axis=1) <= R * (1 + 1e-12))
    s = x @ u / R
    cdf = lambda z: 1 - geo.ball_mass(np.clip(z, c, 1), d) / geo.ball_mass(c, d)
    assert stats.kstest(s, cdf).pvalue > 1e-3
    # Orthogonal part: its norm relative to the slice radius is distributed as U^(1/(d-1)).
    perp = x - np.outer(x @ u, u)
    r = np.linalg.norm(perp, axis=1) / (R * np.sqrt(1 - s ** 2))
    assert stats.kstest(r ** (d - 1), "uniform").pvalue > 1e-3
