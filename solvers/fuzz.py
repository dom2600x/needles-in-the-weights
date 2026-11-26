"""Black-box fuzzers S0, S1, S1b (PREREG 5; G2 protocol). Each returns the 1-based index of the
first violating query (None if none within qmax) and that point. They see only callables."""
import numpy as np

from testbed import constants as K
from testbed import geometry as geo

CHUNK = 100_000


def _violates(net, spec, X):
    return np.abs(net(X) - spec(X)) > K.DELTA


def s0(net, spec, d, R, rng, qmax=10**6):
    done = 0
    while done < qmax:
        k = min(CHUNK, qmax - done)
        X = geo.sample_ball(rng, k, d, R)
        hit = np.flatnonzero(_violates(net, spec, X))
        if hit.size:
            return done + int(hit[0]) + 1, X[hit[0]]
        done += k
    return None, None


def s1b(net, spec, d, R, rng, qmax=10**6):
    done = 0
    while done < qmax:
        k = min(CHUNK, qmax - done)
        X = geo.sample_sphere(rng, k, d, R)
        hit = np.flatnonzero(_violates(net, spec, X))
        if hit.size:
            return done + int(hit[0]) + 1, X[hit[0]]
        done += k
    return None, None


def s1(net, spec, d, R, rng, qmax=10**6, batch=1000, step=0.25, max_empty_rounds=1000, trace=None):
    """Hill-climb fuzz: perturb the best-so-far point; out-of-ball proposals are not queried.
    Step x* + step * R * z / sqrt(d), z ~ N(0, I): step length about step * R."""
    x = geo.sample_ball(rng, 1, d, R)[0]
    h = float(np.abs(net(x[None]) - spec(x[None]))[0])
    count = 1
    if h > K.DELTA:
        return 1, x
    empty = 0
    while count < qmax:
        P_all = x + (step * R / np.sqrt(d)) * rng.standard_normal((batch, d))
        inside = np.linalg.norm(P_all, axis=1) <= R
        pick = int(rng.integers(batch))        # Metropolis candidate among all B proposals
        P = P_all[inside]
        if len(P) == 0:
            empty += 1
            if empty >= max_empty_rounds:
                raise RuntimeError("S1 walk stalled: no in-ball proposals")
            continue
        empty = 0
        P = P[: qmax - count]
        hv = np.abs(net(P) - spec(P))
        hit = np.flatnonzero(hv > K.DELTA)
        if hit.size:
            return count + int(hit[0]) + 1, P[hit[0]]
        j = int(np.argmax(hv))
        if hv[j] > h:
            x, h = P[j], float(hv[j])
        elif inside[pick] and int(np.count_nonzero(inside[:pick])) < len(P):
            k = int(np.count_nonzero(inside[:pick]))   # position of the candidate among queried rows
            x, h = P[k], float(hv[k])
        # else: candidate outside the ball (or beyond the query budget): stay
        count += len(P)
        if trace is not None:
            trace.append(x.copy())
    return None, None


FUZZERS = {"S0": s0, "S1": s1, "S1b": s1b}
