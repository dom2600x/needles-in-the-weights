import numpy as np
import pytest

from testbed import constants as K
from testbed import generator as gen
from testbed import geometry as geo


@pytest.fixture(scope="module")
def secret():
    return gen.new_secret()


def _normalised(net, j):
    n = np.linalg.norm(net.W1[:, j])
    return np.concatenate([net.W1[:, j] / n, [net.b1[j] / n]]), net.v[j] * n


def test_secret_required():
    with pytest.raises(ValueError):
        gen.make_origin(b"short")
    with pytest.raises(ValueError):
        gen.make_origin(12345)
    assert gen.new_secret() != gen.new_secret()


def test_deterministic_given_secret(secret):
    a = gen.build_level(secret, gen.make_origin(secret), "O2")
    b = gen.build_level(secret, gen.make_origin(secret), "O2")
    assert np.array_equal(a.net.W1, b.net.W1) and a.roles == b.roles
    other = gen.make_origin(gen.new_secret())
    assert not np.allclose(gen.make_origin(secret).teacher.W1, other.teacher.W1)


@pytest.mark.parametrize("faulty", [True, False])
@pytest.mark.parametrize("decoys", ["mixed", "cancel_only"])
@pytest.mark.parametrize("level", K.LEVELS)
def test_composition_and_exactness(secret, level, decoys, faulty):
    o = gen.make_origin(secret, faulty=faulty, p=1e-5)
    L = gen.build_level(secret, o, level, decoys=decoys)
    comp = K.COMPOSITION if decoys == "mixed" else K.COMPOSITION_CANCEL_ONLY
    col = {"O0": 0, "O1": 0, "O2": 2, "O3": 4}[level] + (0 if faulty else 1)
    for role, counts in comp.items():
        assert L.roles.count(role) == counts[col], role
    x = geo.sample_ball(np.random.default_rng(0), 100_000, o.d, o.R)
    assert np.max(np.abs(L.net(x) - gen.reference(o)(x))) < 1e-9
    assert L.net.W1.shape == (o.d, K.H)


def test_teacher_preactivation_sd_about_one():
    sds = []
    for _ in range(40):
        o = gen.make_origin(gen.new_secret(), faulty=False)
        x = geo.sample_ball(np.random.default_rng(), 20_000, o.d, o.R)
        sds.append(np.std(o.teacher.preact(x)))
    assert 0.9 < np.mean(sds) < 1.1


@pytest.mark.parametrize("d", K.D_VALUES)
@pytest.mark.parametrize("p", [1e-1, 1e-5, 1e-12])
@pytest.mark.parametrize("g", K.G_VALUES)
def test_needle_parameters(d, p, g):
    s = gen.new_secret()
    o = gen.make_origin(s, d=d, p=p, g=g)
    assert abs(o.G / (g * o.median_teacher_norm) - 1) < 1e-12
    assert abs(o.masses["p"] / p - 1) < 1e-9
    # Median fault magnitude over A is FAULT_MEDIAN (by exact conditional sampling).
    x = geo.sample_cap(np.random.default_rng(0), 200_000, d, o.R, o.u, o.c_cap)
    L = gen.build_level(s, o, "O0")
    fault = np.abs(L.net(x) - o.teacher(x))
    assert abs(np.median(fault) / K.FAULT_MEDIAN - 1) < 0.01
    # The needle unit is active exactly on A (pre-activation sign).
    j = L.needle_units[0]
    assert np.all(L.net.preact(x)[:, j] > -1e-9)
    y = geo.sample_ball(np.random.default_rng(1), 200_000, d, o.R)
    outside = y @ o.u <= o.c_cap * o.R
    assert np.all(L.net.preact(y)[outside, j] <= 1e-9)


def test_o1_is_exact_transform_of_o0(secret):
    o = gen.make_origin(secret)
    a, b = gen.build_level(secret, o, "O0"), gen.build_level(secret, o, "O1")
    na = sorted(tuple(np.round(np.append(*_normalised(a.net, j)), 10)) for j in range(K.H))
    nb = sorted(tuple(np.round(np.append(*_normalised(b.net, j)), 10)) for j in range(K.H))
    assert na == nb
    assert np.any(b.scale_exp != 0) and np.all(np.isin(b.scale_exp, K.RESCALE_EXPONENTS))


def test_o3_split_and_o2_decoys_shared(secret):
    o = gen.make_origin(secret)
    L2, L3 = gen.build_level(secret, o, "O2"), gen.build_level(secret, o, "O3")
    i, j = L3.needle_units
    fi, vi = _normalised(L3.net, i)
    fj, vj = _normalised(L3.net, j)
    assert np.allclose(fi, fj, atol=1e-12)
    assert abs((vi + vj) / (o.v_n * o.G) - 1) < 1e-12
    # Decoys (mimics) identical between O2 and O3 as multisets of normalised forms.
    m2 = sorted(tuple(np.round(_normalised(L2.net, k)[0], 9)) for k, r in enumerate(L2.roles) if r.startswith("mimic"))
    m3 = sorted(tuple(np.round(_normalised(L3.net, k)[0], 9)) for k, r in enumerate(L3.roles) if r.startswith("mimic"))
    assert m2 == m3


@pytest.mark.parametrize("level", ["O0", "O2"])
def test_filler_properties(secret, level):
    o = gen.make_origin(secret)
    L = gen.build_level(secret, o, level)
    net, R = L.net, o.R
    reach = net.b1 + R * np.linalg.norm(net.W1, axis=0)   # max pre-activation over the ball
    for j, role in enumerate(L.roles):
        if role == "benign_dead":
            assert reach[j] <= -K.DEAD_MARGIN_BENIGN[0] * (1 - 1e-9) * np.ldexp(1.0, L.scale_exp[j])
        if role == "mimic_dead":
            assert reach[j] <= -K.DEAD_MARGIN_MIMIC[0] * (1 - 1e-9) * np.ldexp(1.0, L.scale_exp[j])
        if role in ("benign_pair", "mimic_pair"):
            k = L.pair_of[j]
            fj, vj = _normalised(net, j)
            fk, vk = _normalised(net, k)
            assert L.roles[k] == role and L.pair_of[k] == j
            assert np.allclose(fj, fk, atol=1e-12) and abs(vj + vk) < 1e-12 * max(1, abs(vj))
        if role == "mimic_pair":
            assert reach[j] > 0  # alive on a rare region
        if role == "needle":
            assert reach[j] > 0


def test_mimics_look_like_the_needle():
    """O2 decoys have needle-like incoming norms and path norms (PREREG 3.3)."""
    s = gen.new_secret()
    o = gen.make_origin(s)
    L = gen.build_level(s, o, "O2")
    sc = np.ldexp(1.0, -L.scale_exp)   # undo rescaling to compare natural norms
    norms = np.linalg.norm(L.net.W1, axis=0) * sc
    path = np.linalg.norm(L.net.W1, axis=0) * np.abs(L.net.v)
    j = L.needle_units[0]
    mim = [k for k, r in enumerate(L.roles) if r.startswith("mimic")]
    assert np.all((norms[mim] > 0.79 * norms[j]) & (norms[mim] < 1.26 * norms[j]))
    assert np.all((path[mim] > 0.39 * path[j]) & (path[mim] < 2.6 * path[j]))
