import numpy as np
import pytest

from harness import score as SC
from solvers import fuzz as F
from solvers import structural as S
from testbed import constants as K
from testbed import generator as gen
from testbed import geometry as geo
from testbed import serialise as ser


class Counting:
    def __init__(self, f):
        self.f, self.rows = f, 0

    def __call__(self, x):
        self.rows += len(x)
        return self.f(x)


@pytest.fixture(scope="module")
def easy():
    s = gen.new_secret()
    o = gen.make_origin(s, p=1e-2)
    return s, o, gen.build_level(s, o, "O0")


@pytest.mark.parametrize("name", ["S0", "S1", "S1b"])
def test_fuzzer_hits_are_verified_and_counted(easy, name):
    s, o, L = easy
    for seed in range(5):
        idx, x = F.FUZZERS[name](L.net, o.teacher, o.d, o.R, np.random.default_rng(seed))
        assert idx is not None and SC.verify_counterexample(x, L.net, o.teacher, o.R)
    idx, x = F.FUZZERS[name](L.net, o.teacher, o.d, o.R, np.random.default_rng(0), qmax=1)
    assert (idx is None) == (x is None)


def test_s1_counts_only_queried_rows(easy):
    s, o, L = easy
    clean = gen.make_origin(gen.new_secret(), faulty=False)
    spec = Counting(clean.teacher)
    idx, _ = F.s1(clean.teacher, spec, clean.d, clean.R, np.random.default_rng(0), qmax=5000)
    assert idx is None and spec.rows == 5000


def test_s1_walk_is_uniform_on_the_ball():
    """With no hits the walk is a random walk with rejection outside the ball, whose stationary
    law is uniform: (||x|| / R)^d should be roughly Uniform(0, 1)."""
    o = gen.make_origin(gen.new_secret(), faulty=False)
    trace = []
    F.s1(o.teacher, o.teacher, o.d, o.R, np.random.default_rng(1), qmax=2_000_000, trace=trace)
    r = (np.linalg.norm(np.array(trace[200::10]), axis=1) / o.R) ** o.d
    assert np.all(r <= 1 + 1e-9)
    assert abs(r.mean() - 0.5) < 0.08


def test_s0_hit_rate_matches_p_v():
    s = gen.new_secret()
    o = gen.make_origin(s, p=1e-1)
    net = gen.build_level(s, o, "O0").net
    idx = [F.s0(net, o.teacher, o.d, o.R, np.random.default_rng(k), qmax=10**4)[0] for k in range(400)]
    # First-hit index is geometric with success probability p_V.
    assert abs(np.mean(idx) * o.masses["p_V"] - 1) < 0.2


@pytest.mark.parametrize("level,decoys", [("O0", "mixed"), ("O1", "mixed"), ("O2", "mixed"), ("O3", "mixed"),
                                          ("O2", "cancel_only"), ("O3", "cancel_only")])
def test_s5_finds_needle_and_s4_merges(level, decoys):
    for _ in range(10):
        s = gen.new_secret()
        o = gen.make_origin(s, p=1e-12)
        L = gen.build_level(s, o, level, decoys=decoys)
        sn = ser.serialised_net(L.net)
        spec = Counting(o.teacher)
        a = S.s5(sn, spec)
        r = SC.score(a, L.net, o.teacher, o.R, L.needle_units)
        assert a["faulty"] and r["success"] and r["localised"] and spec.rows <= S.S5_MAX_TESTS
        groups = S.merged_groups(sn)
        n_pairs = L.roles.count("benign_pair") // 2 + L.roles.count("mimic_pair") // 2
        n_needle_groups = 1
        assert len(groups) == K.H - 2 * n_pairs - len(L.needle_units) + n_needle_groups


@pytest.mark.parametrize("level", K.LEVELS)
def test_s5_never_alarms_on_clean(level):
    for _ in range(20):
        s = gen.new_secret()
        o = gen.make_origin(s, faulty=False)
        L = gen.build_level(s, o, level)
        a = S.s5(ser.serialised_net(L.net), o.teacher)
        assert a["faulty"] is False and a["counterexample"] is None


def test_scorer_rejects_bad_counterexamples(easy):
    s, o, L = easy
    x = geo.sample_cap(np.random.default_rng(0), 1, o.d, o.R, o.u, o.c_V + 0.5 * (1 - o.c_V))[0]
    assert SC.verify_counterexample(x, L.net, o.teacher, o.R)
    assert not SC.verify_counterexample(x * 1.01 * o.R / np.linalg.norm(x), L.net, o.teacher, o.R)  # outside
    assert not SC.verify_counterexample(np.full(o.d, np.nan), L.net, o.teacher, o.R)
    assert not SC.verify_counterexample(x[:-1], L.net, o.teacher, o.R)
    assert not SC.verify_counterexample(None, L.net, o.teacher, o.R)
    assert not SC.verify_counterexample(np.zeros(o.d), L.net, o.teacher, o.R)   # not in A
