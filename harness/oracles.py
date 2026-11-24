"""Metered RPC oracles (PREREG 7). They run in the parent process; the sandboxed child sees stubs.

spec(x): the clean function, at most SPEC_ROWS_CAP rows per run.
net(x):  mode B only, the faulty-or-clean network (original float64), at most NET_ROWS_CAP rows.
A call is all-or-nothing: if any row is malformed or outside the ball, or the call would exceed
the remaining budget, it raises and nothing is counted (refund on error). Rows on the boundary
are accepted up to ||x|| <= R (1 + BALL_TOL).
"""
import numpy as np

from testbed import constants as K


class MeteredOracle:
    def __init__(self, fn, d, R, cap, name):
        self.fn, self.d, self.R, self.cap, self.name = fn, d, R, int(cap), name
        self.used, self.calls, self.rejected = 0, 0, 0

    @property
    def remaining(self):
        return self.cap - self.used

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        try:
            if x.ndim == 1:
                x = x[None, :]
            if x.ndim != 2 or x.shape[1] != self.d:
                raise ValueError(f"{self.name}: expected an array of shape (n, {self.d}), got {x.shape}")
            n = x.shape[0]
            if n == 0:
                raise ValueError(f"{self.name}: empty input")
            if not np.all(np.isfinite(x)):
                raise ValueError(f"{self.name}: non-finite values")
            out = np.linalg.norm(x, axis=1) > self.R * (1 + K.BALL_TOL)
            if out.any():
                raise ValueError(f"{self.name}: {int(out.sum())} of {n} rows are outside the ball of radius "
                                 f"{self.R:.6g}; call rejected, nothing counted")
            if n > self.remaining:
                raise ValueError(f"{self.name}: {n} rows requested, {self.remaining} remaining; call rejected")
            y = self.fn(x)
        except Exception:
            self.rejected += 1
            raise
        self.used += n
        self.calls += 1
        return y

    def log(self):
        return dict(name=self.name, used=self.used, cap=self.cap, calls=self.calls, rejected=self.rejected)


def make_oracles(teacher, net, d, R, mode):
    """teacher, net: callables (testbed.generator.Net). Mode W gets spec only; mode B also net."""
    oracles = {"spec": MeteredOracle(teacher, d, R, K.SPEC_ROWS_CAP, "spec")}
    if mode == "B":
        oracles["net"] = MeteredOracle(net, d, R, K.NET_ROWS_CAP, "net")
    elif mode != "W":
        raise ValueError(mode)
    return oracles
