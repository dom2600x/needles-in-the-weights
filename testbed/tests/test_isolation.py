"""PREREG 3.4 check 5 (isolation) and the metered oracles (PREREG 7).

The proposer child must not be able to reach teacher (spec) parameters in any mode, nor the
network weights in mode B. Probed with the audit hook off (strongest attacker) and on.
"""
import hashlib, json, os, secrets

import numpy as np
import pytest

from harness.oracles import MeteredOracle, make_oracles
from host import REPO, SandboxKernel
from testbed import constants as K
from testbed import generator as gen
from testbed import geometry as geo
from testbed import serialise as ser

SCRATCH = os.path.join(REPO, "runs", "g1", "sandbox_scratch")

PROBE = r'''
import sys
_found, _seen = [], set()
def _visit(_o, _depth=0):
    if id(_o) in _seen or _depth > 4:
        return
    _seen.add(id(_o))
    if isinstance(_o, np.ndarray):
        if _o.dtype.kind == "f" and _o.size:
            _a = np.ascontiguousarray(_o, dtype=np.float64)
            _found.append(__import__("hashlib").sha256(_a.tobytes()).hexdigest())
            if _a.ndim == 2:
                for _j in range(_a.shape[1]):
                    _found.append(__import__("hashlib").sha256(np.ascontiguousarray(_a[:, _j]).tobytes()).hexdigest())
        return
    try:
        if isinstance(_o, dict):
            _items = list(_o.values())
        elif isinstance(_o, (list, tuple, set, frozenset)):
            _items = list(_o)
        else:
            _items = None
    except Exception:
        _items = None
    if _items is not None:
        for _v in _items:
            _visit(_v, _depth + 1)
    elif isinstance(_o, type(_visit)):  # plain functions: closures, defaults, attributes
        for _c in (_o.__closure__ or ()):
            try:
                _visit(_c.cell_contents, _depth + 1)
            except ValueError:
                pass
        _visit(getattr(_o, "__defaults__", None), _depth + 1)
        _visit(getattr(_o, "__dict__", None), _depth + 1)
# Everything reachable from module globals (the kernel's own module included) and this namespace.
for _m in list(sys.modules.values()):
    _visit(getattr(_m, "__dict__", None))
_visit(dict(globals()))
# All live objects, when gc is importable (the audit hook refuses it; the OS-only run allows it).
try:
    import gc as _gc
    for _o in _gc.get_objects():
        _visit(_o)
except ImportError:
    pass
print(len(_found))
'''


def _hashes(*arrays):
    out = set()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.float64)
        out.add(hashlib.sha256(a.tobytes()).hexdigest())
        if a.ndim == 2:
            for j in range(a.shape[1]):
                out.add(hashlib.sha256(np.ascontiguousarray(a[:, j]).tobytes()).hexdigest())
    return out


def _child_hashes(k):
    r = k.exec(PROBE)
    assert r["status"] == "ok", r
    n = int(r["stdout"].strip())
    got = set()
    for i in range(0, n, 50):  # 50 x 67 chars stays under the 4,000-char output cap
        rr = k.exec(f"print(__import__('json').dumps(_found[{i}:{i + 50}]))")
        got |= set(json.loads(rr["stdout"]))
    return got


@pytest.fixture
def task(tmp_path):
    s = gen.new_secret()
    o = gen.make_origin(s, p=1e-5)
    L = gen.build_level(s, o, "O2")
    return o, L, tmp_path


@pytest.mark.parametrize("hook", [False, True], ids=["os-only", "hook+os"])
@pytest.mark.parametrize("mode", ["W", "B"])
def test_child_cannot_reach_hidden_weights(task, mode, hook):
    o, L, tmp = task
    path = ser.write_blind(str(tmp), ser.new_task_id(), mode, net=L.net if mode == "W" else None, d=o.d)
    blind = ser.read_blind(path)
    oracles = make_oracles(o.teacher, L.net, o.d, o.R, mode)
    k = SandboxKernel(os.path.join(SCRATCH, secrets.token_hex(8)), oracles=oracles, audit_hook=hook)
    k.start()
    try:
        if mode == "W":
            for name in ("W1", "b1", "v"):
                k.set_array(name, blind["weights"][name])
            k.set_array("c", np.array(blind["weights"]["c"]))
        # Exercise the stubs so any cached state would be present.
        x = geo.sample_ball(np.random.default_rng(0), 50, o.d, o.R)
        k.set_array("_x", x)
        r = k.exec("y = spec(_x)\n" + ("z = net(_x[:10])\n" if mode == "B" else "") + "print(y.shape)")
        assert r["status"] == "ok" and r["stdout"] == "(50,)\n", r
        got = _child_hashes(k)
        teacher = _hashes(o.teacher.W1, o.teacher.b1, o.teacher.v)
        assert not (got & teacher), "teacher parameters reachable in the child"
        if mode == "W":  # positive control: the probe does see arrays that are in the child
            assert _hashes(blind["weights"]["W1"]) <= got
        if mode == "B":
            assert not (got & _hashes(L.net.W1, L.net.b1, L.net.v)), "network weights reachable in mode B"
            names = set(json.loads(k.exec("print(__import__('json').dumps(sorted(globals())))")["stdout"]))
            assert not names & {"W1", "b1", "v", "c"}, names
        # Scratch holds nothing but what the child wrote (and the generated profile).
        assert set(os.listdir(k.scratch)) <= {".profile.sb"}
        # Spec oracle has accounted for exactly the rows sent.
        assert oracles["spec"].used == 50 and (mode == "W" or oracles["net"].used == 10)
    finally:
        k.close()


def test_blind_b_package_has_no_weights(tmp_path):
    path = ser.write_blind(str(tmp_path), ser.new_task_id(), "B", d=32)
    obj = ser.read_blind(path)
    assert "weights" not in obj and obj["net_rows_cap"] == K.NET_ROWS_CAP


def test_metering_and_refund():
    d, R = 4, 1.0
    o = MeteredOracle(lambda x: x.sum(axis=1), d, R, cap=10, name="net")
    assert o(np.zeros((3, d))).shape == (3,) and o.used == 3
    with pytest.raises(ValueError):          # outside the ball: whole call rejected, nothing counted
        o(np.ones((2, d)))
    with pytest.raises(ValueError):          # over budget: rejected, nothing counted
        o(np.zeros((8, d)))
    with pytest.raises(ValueError):          # wrong shape
        o(np.zeros((2, d + 1)))
    with pytest.raises(ValueError):          # non-finite
        o(np.full((1, d), np.nan))
    assert o.used == 3 and o.rejected == 4
    x = np.zeros((1, d)); x[0, 0] = R * (1 + 0.5 * K.BALL_TOL)   # boundary tolerance
    o(x)
    assert o.used == 4
    def boom(x):
        raise RuntimeError("oracle crashed")
    b = MeteredOracle(boom, d, R, cap=10, name="net")
    with pytest.raises(RuntimeError):
        b(np.zeros((2, d)))
    assert b.used == 0                        # refunded on error


def test_spec_cap_in_sandbox():
    """spec over budget is rejected inside the child as a RuntimeError, not a crash."""
    o = gen.make_origin(gen.new_secret(), faulty=False)
    oracles = make_oracles(o.teacher, None, o.d, o.R, "W")
    oracles["spec"].cap = 100
    k = SandboxKernel(os.path.join(SCRATCH, secrets.token_hex(8)), oracles=oracles)
    k.start()
    try:
        r = k.exec("x = np.zeros((101, 32))\ntry:\n    spec(x)\nexcept RuntimeError as e:\n    print('rejected:', e)\n"
                   "print(spec(np.zeros((100, 32))).shape)")
        assert "rejected" in r["stdout"] and "remaining" in r["stdout"] and "(100,)" in r["stdout"], r
        assert oracles["spec"].used == 100
    finally:
        k.close()
