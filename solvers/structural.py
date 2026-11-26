"""White-box structural auditors S2a, S2b, S3, S4, S5 (PREREG 5; G2 protocol).

Input: a Net as the proposer sees it (serialised weights) and, for S5, a spec callable.
Each returns {unit, score, faulty, counterexample, group}. No truth is visible here.
"""
import numpy as np

from testbed import constants as K

GROUP_TOL = 1e-6
ZERO_TOL = 1e-6
S5_MAX_TESTS = 5


def _top(scores):
    j = int(np.argmax(scores))
    return dict(unit=j, score=float(scores[j]), group=[j], counterexample=None, faulty=None)


def s2a(net, spec=None):
    return _top(np.linalg.norm(net.W1, axis=0))


def s2b(net, spec=None):
    return _top(np.abs(net.v))


def s3(net, spec=None):
    return _top(np.linalg.norm(net.W1, axis=0) * np.abs(net.v))


def merged_groups(net):
    """Group units with equal normalised forms; return live groups sorted by |summed outgoing|."""
    n = np.linalg.norm(net.W1, axis=0)
    live = np.flatnonzero(n > 0)
    F = np.vstack([net.W1[:, live] / n[live], (net.b1[live] / n[live])[None]]).T   # (m, d+1)
    parent = list(range(len(live)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    D = np.max(np.abs(F[:, None, :] - F[None, :, :]), axis=2)
    for i, j in zip(*np.nonzero(np.triu(D <= GROUP_TOL, 1))):
        parent[find(i)] = find(j)
    groups = {}
    for i in range(len(live)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        parts = net.v[live[members]] * n[live[members]]
        total = float(parts.sum())
        if abs(total) <= ZERO_TOL * float(np.max(np.abs(parts))):
            continue
        k = members[0]
        out.append(dict(units=[int(live[m]) for m in members], score=abs(total),
                        direction=F[k, :-1], bias_n=float(F[k, -1])))
    out.sort(key=lambda g: -g["score"])
    return out


def s4(net, spec=None):
    g = merged_groups(net)[0]
    return dict(unit=g["units"][0], score=g["score"], group=g["units"], counterexample=None, faulty=None)


def s5(net, spec):
    R = K.radius(net.d)
    groups = [g for g in merged_groups(net) if g["bias_n"] + R > 0]   # reachable on the ball
    for g in groups[:S5_MAX_TESTS]:
        x = R * g["direction"] / np.linalg.norm(g["direction"])
        if abs(float(net(x[None])[0]) - float(spec(x[None])[0])) > K.DELTA:
            return dict(unit=g["units"][0], score=g["score"], group=g["units"], counterexample=x, faulty=True)
    top = groups[0] if groups else dict(units=[0], score=0.0)
    return dict(unit=top["units"][0], score=top["score"], group=top["units"], counterexample=None, faulty=False)


STRUCTURAL = {"S2a": s2a, "S2b": s2b, "S3": s3, "S4": s4, "S5": s5}
