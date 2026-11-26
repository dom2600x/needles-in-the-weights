"""Network generator (PREREG 3). Toy one-hidden-layer ReLU networks only.

An *origin* fixes the clean function (teacher) and, if faulty, the needle. Each equivalence
level O0..O3 is a network computing the same function teacher + needle, built from that origin:

    f(x) = sum_j v_j relu(W1[:, j] . x + b1[j]) + c

All randomness comes from a 256-bit secret (secrets.token_bytes(32)) through numpy's SeedSequence,
so the generator seed cannot be searched (PREREG 3.4 leak test a). Unit order is randomly
permuted at every level, O0 included, so the needle's index carries no information (leak test 4).
"""
import math
import secrets
from dataclasses import dataclass, field

import numpy as np

from . import constants as K
from . import geometry as geo


@dataclass
class Net:
    W1: np.ndarray   # (d, H)
    b1: np.ndarray   # (H,)
    v: np.ndarray    # (H,)
    c: float

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        return np.maximum(x @ self.W1 + self.b1, 0.0) @ self.v + self.c

    def preact(self, x):
        return np.asarray(x, dtype=np.float64) @ self.W1 + self.b1

    def copy(self):
        return Net(self.W1.copy(), self.b1.copy(), self.v.copy(), float(self.c))

    @property
    def d(self):
        return self.W1.shape[0]

    @property
    def H(self):
        return self.W1.shape[1]


@dataclass
class Origin:
    """Clean function and (optional) needle. Everything here is truth."""
    d: int
    R: float
    teacher: Net                       # the spec: N_TEACHER units
    faulty: bool
    p: float | None = None             # needle region mass (uniform on ball)
    g: float | None = None
    u: np.ndarray | None = None        # needle direction
    c_cap: float | None = None         # tau / R
    G: float | None = None             # needle incoming norm
    v_n: float | None = None           # needle outgoing weight (signed)
    c_V: float | None = None           # violation cap threshold (s > c_V)
    masses: dict = field(default_factory=dict)  # p, p_s, p_V, p_Vs (closed form)
    median_teacher_norm: float = 0.0
    secret_fingerprint: str = ""       # sha256 prefix of the secret, for audit only


@dataclass
class Level:
    """One equivalence-level network built from an origin, plus unit roles (truth)."""
    level: str
    net: Net
    roles: list            # role of each hidden unit, e.g. "teacher", "needle", "benign_pair", ...
    pair_of: list          # partner index for pair units, else -1
    scale_exp: np.ndarray  # k_j with s_j = 2^k_j (zeros at O0)
    decoys: str            # "mixed" or "cancel_only"

    @property
    def needle_units(self):
        return [j for j, r in enumerate(self.roles) if r == "needle"]


def new_secret():
    return secrets.token_bytes(32)


def _rng(secret, *stream):
    if not isinstance(secret, (bytes, bytearray)) or len(secret) != 32:
        raise ValueError("secret must be 32 bytes from secrets.token_bytes(32)")
    words = [int.from_bytes(secret[i:i + 4], "little") for i in range(0, 32, 4)]
    return np.random.default_rng(np.random.SeedSequence(words + [hash_stream(s) for s in stream]))


def hash_stream(s):
    import hashlib
    return int.from_bytes(hashlib.sha256(str(s).encode()).digest()[:4], "little")


def _fingerprint(secret):
    import hashlib
    return hashlib.sha256(b"fingerprint:" + bytes(secret)).hexdigest()[:16]


def _loguniform(rng, lo, hi, size=None):
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size))


def make_origin(secret, d=32, faulty=True, p=1e-12, g=K.G_DEFAULT):
    """Build the clean teacher and, if faulty, the needle parameters."""
    if d not in K.D_VALUES:
        raise ValueError(f"d must be one of {K.D_VALUES}")
    rng = _rng(secret, "origin")
    R = K.radius(d)
    Wt = rng.normal(0.0, K.SIGMA / math.sqrt(d), size=(d, K.N_TEACHER))
    bt = rng.normal(0.0, K.TEACHER_BIAS_SD, size=K.N_TEACHER)
    vt = rng.normal(0.0, K.TEACHER_OUT_SD, size=K.N_TEACHER)
    ct = float(rng.normal(0.0, K.OUTPUT_BIAS_SD))
    teacher = Net(Wt, bt, vt, ct)
    med = float(np.median(np.linalg.norm(Wt, axis=0)))
    o = Origin(d=d, R=R, teacher=teacher, faulty=faulty, median_teacher_norm=med,
               secret_fingerprint=_fingerprint(secret))
    if faulty:
        if not 0 < p <= 0.5:
            raise ValueError("p must be in (0, 0.5]")
        u = rng.standard_normal(d); u /= np.linalg.norm(u)
        c_cap = geo.c_for_ball_mass(p, d)
        G = g * med
        s_med = float(geo.median_s_in_cap(c_cap, d))
        mag = K.FAULT_MEDIAN / (G * R * (s_med - c_cap))      # |v_n|: median fault in A = FAULT_MEDIAN
        v_n = float(mag * rng.choice([-1.0, 1.0]))
        c_V = c_cap + K.DELTA / (abs(v_n) * G * R)
        if c_V >= 1:
            raise ValueError("violation set is empty for these parameters")
        o.p, o.g, o.u, o.c_cap, o.G, o.v_n, o.c_V = p, g, u, c_cap, G, v_n, c_V
        o.masses = dict(p=float(geo.ball_mass(c_cap, d)), p_s=float(geo.sphere_mass(c_cap, d)),
                        p_V=float(geo.ball_mass(c_V, d)), p_Vs=float(geo.sphere_mass(c_V, d)))
    return o


def reference(origin):
    """The exact function teacher + needle with no fillers (25 units), for exactness checks."""
    t = origin.teacher
    if not origin.faulty:
        return t.copy()
    W = np.column_stack([t.W1, origin.G * origin.u])
    b = np.append(t.b1, -origin.G * origin.c_cap * origin.R)
    v = np.append(t.v, origin.v_n)
    return Net(W, b, v, t.c)


def _teacher_like_unit(rng, d):
    return rng.normal(0.0, K.SIGMA / math.sqrt(d), size=d), rng.normal(0.0, K.TEACHER_BIAS_SD)


def build_level(secret, origin, level, decoys="mixed"):
    """Build the network for one equivalence level. Deterministic given (secret, level, decoys)."""
    if level not in K.LEVELS:
        raise ValueError(level)
    if decoys not in ("mixed", "cancel_only"):
        raise ValueError(decoys)
    d, R = origin.d, origin.R
    comp = K.COMPOSITION if decoys == "mixed" else K.COMPOSITION_CANCEL_ONLY
    col = {"O0": 0, "O1": 0, "O2": 2, "O3": 4}[level] + (0 if origin.faulty else 1)
    counts = {role: v[col] for role, v in comp.items()}
    # O0 and O1 share fillers (O1 is an exact transform of O0); O2 and O3 share theirs.
    family = "O01" if level in ("O0", "O1") else "O23"
    rng = _rng(secret, "fillers", family, decoys)

    W, b, v, roles, pair_of = [], [], [], [], []

    def add(w, bias, out, role, partner=-1):
        W.append(np.asarray(w, dtype=float)); b.append(float(bias)); v.append(float(out))
        roles.append(role); pair_of.append(partner)

    t = origin.teacher
    for k in range(K.N_TEACHER):
        add(t.W1[:, k], t.b1[k], t.v[k], "teacher")

    # Reference outgoing scale for decoys: the needle's |v_n|, or (clean) what it would be at p = 1e-5.
    if origin.faulty:
        G_ref, v_ref = origin.G, abs(origin.v_n)
    else:
        G_ref = K.G_DEFAULT * origin.median_teacher_norm
        c_r = geo.c_for_ball_mass(1e-5, d)
        v_ref = K.FAULT_MEDIAN / (G_ref * R * (float(geo.median_s_in_cap(c_r, d)) - c_r))

    if counts["needle"]:
        wn, bn = origin.G * origin.u, -origin.G * origin.c_cap * R
        if counts["needle"] == 1:
            add(wn, bn, origin.v_n, "needle")
        else:  # O3 split: same incoming form, outgoing weights summing to v_n
            lam = _rng(secret, "split", decoys).uniform(*K.SPLIT_FRACTION)
            add(wn, bn, lam * origin.v_n, "needle")
            add(wn, bn, origin.v_n - lam * origin.v_n, "needle")

    for _ in range(counts["benign_pair"] // 2):
        w, bias = _teacher_like_unit(rng, d)
        a = abs(rng.normal(0.0, K.TEACHER_OUT_SD))
        i = len(W)
        add(w, bias, a, "benign_pair", i + 1)
        add(w, bias, -a, "benign_pair", i)

    for _ in range(counts["mimic_dead"]):
        dirn = rng.standard_normal(d); dirn /= np.linalg.norm(dirn)
        w = G_ref * _loguniform(rng, *K.MIMIC_NORM_FACTOR) * dirn
        margin = rng.uniform(*K.DEAD_MARGIN_MIMIC)
        out = v_ref * _loguniform(rng, *K.MIMIC_OUT_FACTOR) * rng.choice([-1.0, 1.0])
        add(w, -R * np.linalg.norm(w) - margin, out, "mimic_dead")

    for _ in range(counts["mimic_pair"] // 2):
        dirn = rng.standard_normal(d); dirn /= np.linalg.norm(dirn)
        Gm = G_ref * _loguniform(rng, *K.MIMIC_NORM_FACTOR)
        cm = geo.c_for_ball_mass(float(_loguniform(rng, *K.MIMIC_PAIR_MASS)), d)
        a = v_ref * _loguniform(rng, *K.MIMIC_OUT_FACTOR)
        i = len(W)
        add(Gm * dirn, -Gm * cm * R, a, "mimic_pair", i + 1)
        add(Gm * dirn, -Gm * cm * R, -a, "mimic_pair", i)

    # Drawn last, so O2 and O3 (14 vs 15 benign dead units) share every other filler.
    for _ in range(counts["benign_dead"]):
        w, _b = _teacher_like_unit(rng, d)
        margin = rng.uniform(*K.DEAD_MARGIN_BENIGN)
        add(w, -R * np.linalg.norm(w) - margin, rng.normal(0.0, K.TEACHER_OUT_SD), "benign_dead")

    assert len(W) == K.H, len(W)
    W1, b1, vv = np.column_stack(W), np.array(b), np.array(v)

    # Base permutation (all levels; makes the needle index uninformative).
    perm = rng.permutation(K.H)
    scale_exp = np.zeros(K.H, dtype=int)
    if level != "O0":
        # O1-O3: an extra permutation and exact power-of-two rescaling.
        lrng = _rng(secret, "transform", level, decoys)
        perm = perm[lrng.permutation(K.H)]
        scale_exp = lrng.choice(K.RESCALE_EXPONENTS, size=K.H)
    inv = np.empty(K.H, dtype=int); inv[perm] = np.arange(K.H)
    W1, b1, vv = W1[:, perm], b1[perm], vv[perm]
    roles = [roles[j] for j in perm]
    pair_of = [inv[pair_of[j]] if pair_of[j] >= 0 else -1 for j in perm]
    s = np.ldexp(1.0, scale_exp)   # exact powers of two
    W1, b1, vv = W1 * s, b1 * s, vv / s
    return Level(level=level, net=Net(W1, b1, vv, t.c), roles=roles, pair_of=pair_of,
                 scale_exp=scale_exp, decoys=decoys)
