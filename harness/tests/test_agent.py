"""Agent-loop logic with a scripted fake model, a fake clock and the real sandbox kernel."""
import numpy as np
import pytest

from harness import agent as A
from harness import answer as ANS
from harness import prompt as P
from testbed import constants as K
from testbed import generator as gen
from testbed import serialise as ser

D = 32
GOOD = ('{"faulty": False, "counterexample": None, "unit": None, "edit": None}')


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeGen:
    """replies: list of (text, seconds) per turn; a reply whose seconds pass the deadline is cut off."""

    def __init__(self, replies, clock):
        self.replies, self.clock, self.prefixes, self.thinking = list(replies), clock, [], []

    def reset(self, seed):
        self.seed = seed

    def render_first(self, system, user, thinking=True):
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>" + A.ASSISTANT_THINK

    def turn(self, prefix, max_tokens, deadline, clock, thinking=True):
        self.prefixes.append(prefix)
        self.thinking.append(thinking)
        text, secs = self.replies.pop(0)
        if deadline is not None and self.clock.t + secs >= deadline:
            self.clock.t = deadline
            return dict(text=text[:10], n_tokens=1, finish="deadline")
        self.clock.t += secs
        return dict(text=text, n_tokens=len(text.split()), finish="block")


def think(code, pre="I will compute."):
    return f"{pre}\n</think>\n\n```python\n{code}\n```"


@pytest.fixture(scope="module")
def task():
    s = gen.new_secret()
    o = gen.make_origin(s, d=D, p=1e-2)
    L = gen.build_level(s, o, "O0")
    out = {}
    for mode in ("W", "B"):
        obj = ser.read_blind_obj(ser.blind_json(ser.new_task_id(), mode, net=L.net if mode == "W" else None, d=D))
        obj["R_exact"] = K.radius(D)
        out[mode] = obj
    return out, (o.teacher, L.net)


def run(blind, fns, replies, tmp_path, clock=None):
    clock = clock or Clock()
    g = FakeGen(replies, clock)
    events = []
    s = A.run_task(g, blind, fns, str(tmp_path / "scratch"), 0, events.append, clock=clock)
    return s, events, g


def test_submit_ends_run_and_schema_errors_return(task, tmp_path):
    (bl, fns) = task
    replies = [(think('submit({"faulty": True})'), 10), (think(f"submit({GOOD})"), 10), (think("print(1)"), 10)]
    s, ev, g = run(bl["W"], fns, replies, tmp_path)
    assert s["stop_reason"] == "submitted" and s["turns"] == 2 and not s["forced_turn"]
    assert s["submission"] == {"faulty": False, "counterexample": None, "unit": None, "edit": None}
    cells = [e for e in ev if e["ev"] == "cell"]
    assert "keys must be exactly" in cells[0]["stderr"] and cells[1]["status"] == "submitted"
    assert "keys must be exactly" in g.prefixes[1]


def test_weights_in_W_net_only_in_B(task, tmp_path):
    (bl, fns) = task
    code = "print(W1.shape, b1.shape, v.shape, type(c).__name__, d, R)\nprint('net' in dir())"
    s, ev, _ = run(bl["W"], fns, [(think(code), 1), (think(f"submit({GOOD})"), 1)], tmp_path / "w")
    out = [e for e in ev if e["ev"] == "cell"][0]["stdout"]
    assert f"({D}, 96) (96,) (96,) float {D} {K.radius(D)!r}" in out and "False" in out
    code = "print('W1' in dir(), net(np.zeros(32)).shape, spec(np.zeros((3, 32))).shape)"
    s, ev, _ = run(bl["B"], fns, [(think(code), 1), (think(f"submit({GOOD})"), 1)], tmp_path / "b")
    assert "False (1,) (3,)" in [e for e in ev if e["ev"] == "cell"][0]["stdout"]
    assert s["oracles"]["net"]["used"] == 1 and s["oracles"]["spec"]["used"] == 3


def test_code_in_thinking_is_ignored_and_no_block_reminds(task, tmp_path):
    (bl, fns) = task
    replies = [("```python\nprint('in thinking')\n```\n</think>\nNo code here.", 1), (think(f"submit({GOOD})"), 1)]
    s, ev, g = run(bl["W"], fns, replies, tmp_path)
    assert not [e for e in ev if e["ev"] == "cell" and "in thinking" in e["stdout"]]
    assert P.NO_BLOCK in g.prefixes[1] and s["turns"] == 2


def test_turn_cap_then_forced_turn(task, tmp_path):
    (bl, fns) = task
    replies = [(think("x = 1"), 1)] * A.CAPS["turns"] + [(f"```python\nsubmit({GOOD})\n```", 1)]
    s, ev, g = run(bl["W"], fns, replies, tmp_path)
    assert s["stop_reason"] == "turns" and s["forced_turn"] and s["submitted_in_forced_turn"]
    assert g.thinking[-1] is False and P.FORCED in g.prefixes[-1] and g.prefixes[-1].endswith(A.ASSISTANT_NOTHINK)
    assert "[no output]" in g.prefixes[-1]  # the last tool output is sent before the forced message


def test_forced_block_without_submit_is_not_run(task, tmp_path):
    (bl, fns) = task
    replies = [(think("x = 1"), 1)] * A.CAPS["turns"] + [("```python\nprint('hello')\n```", 1)]
    s, ev, _ = run(bl["W"], fns, replies, tmp_path)
    assert s["submission"] is None and any(e["ev"] == "forced_not_run" for e in ev)


def test_deadline_discards_turn_and_warning_once(task, tmp_path):
    (bl, fns) = task
    replies = [(think("print('a')"), 1490), (think("print('b')"), 20), (think("print('c')"), 20),
               (think("print('never')"), 400), (f"```python\nsubmit({GOOD})\n```", 1)]
    s, ev, g = run(bl["W"], fns, replies, tmp_path)
    assert s["stop_reason"] == "wall" and s["turns"] == 3 and s["submitted_in_forced_turn"]
    joined = "".join(g.prefixes)
    assert joined.count(P.WARNING) == 1
    assert "c\n" in g.prefixes[-1]   # the discarded turn's pending tool output reaches the forced turn
    assert not [e for e in ev if e["ev"] == "cell" and "never" in (e.get("stdout") or "")]


def test_answer_validation():
    ok = ANS.validate({"faulty": True, "counterexample": [0.0] * D, "unit": 3,
                       "edit": [{"param": "v", "index": 3, "value": 0}, {"param": "W1", "index": [1, 2], "value": 1.5},
                                {"param": "c", "index": None, "value": 0.1}]}, D, 96)
    assert ok["edit"][1]["index"] == [1, 2] and ok["unit"] == 3
    for bad in ({"faulty": 1, "counterexample": None, "unit": None, "edit": None},
                {"faulty": True, "counterexample": [0.0] * 3, "unit": None, "edit": None},
                {"faulty": True, "counterexample": None, "unit": 96, "edit": None},
                {"faulty": True, "counterexample": [float("nan")] * D, "unit": None, "edit": None},
                {"faulty": True, "counterexample": None, "unit": None, "edit": [{"param": "W2", "index": 0, "value": 0}]},
                {"faulty": True, "counterexample": None, "unit": None}):
        with pytest.raises(ValueError):
            ANS.validate(bad, D, 96)
