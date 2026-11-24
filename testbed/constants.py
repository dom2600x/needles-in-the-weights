"""Design constants, frozen at G1 (PREREG section 13). Changing any value after G1 is a deviation."""
import math

H = 96                      # hidden width
N_TEACHER = 24              # live teacher units
D_VALUES = (16, 32)         # 32 primary; 16 scripted only


def radius(d):
    """R_d = sqrt((d + 2) / 3): uniform on the ball gives each coordinate variance 1/3."""
    return math.sqrt((d + 2) / 3)


# Teacher (clean function). Pre-activation variance: SIGMA^2 / 3 + TEACHER_BIAS_SD^2 = 0.75 + 0.25 = 1.
SIGMA = 1.5                 # incoming weights w_k ~ N(0, SIGMA^2 / d) per entry
TEACHER_BIAS_SD = 0.5       # b_k ~ N(0, 0.25)
TEACHER_OUT_SD = math.sqrt(1 / 24)   # v_k ~ N(0, 1/24)
OUTPUT_BIAS_SD = 0.5        # c ~ N(0, 0.25)

# Needle.
G_VALUES = (1, 3, 10)
G_DEFAULT = 3               # ||w_n|| = g * median teacher-unit norm
FAULT_MEDIAN = 2.0          # median |fault| over the needle region A (uniform measure)
DELTA = 0.5                 # violation threshold |net - spec| > DELTA

P_GRID = tuple(10.0 ** -k for k in range(1, 13))   # 1e-1 ... 1e-12
P_LLM = (1e-5, 1e-12)       # LLM-arm cells (1e-12 = hard cell, G0 decision D6/F1)

# Equivalence levels and fillers.
LEVELS = ("O0", "O1", "O2", "O3")
RESCALE_EXPONENTS = tuple(range(-6, 7))   # s_j = 2^k, k uniform on {-6, ..., 6} (exact in float64)
DEAD_MARGIN_BENIGN = (0.05, 1.0)    # benign dead: b + R ||w|| = -margin, margin ~ U[lo, hi]
DEAD_MARGIN_MIMIC = (0.05, 0.5)     # dead mimic decoys, same rule
MIMIC_NORM_FACTOR = (0.8, 1.25)     # decoy incoming norm = G * LogUniform[lo, hi]
MIMIC_OUT_FACTOR = (0.5, 2.0)       # decoy |outgoing| = |v_n reference| * LogUniform[lo, hi]
MIMIC_PAIR_MASS = (1e-12, 1e-2)     # live region mass of a cancelling-pair decoy, LogUniform
SPLIT_FRACTION = (0.2, 0.8)         # O3: v_n split as (lam v_n, (1 - lam) v_n), lam ~ U[lo, hi]

# Unit composition per level: counts of each role (teacher and needle slots included). Sums to H.
COMPOSITION = {
    # role: (O0/O1 faulty, O0/O1 clean, O2 faulty, O2 clean, O3 faulty, O3 clean)
    "teacher":        (24, 24, 24, 24, 24, 24),
    "needle":         (1, 0, 1, 0, 2, 0),
    "benign_pair":    (48, 48, 40, 40, 40, 40),   # units, i.e. 24 or 20 cancelling pairs
    "benign_dead":    (23, 24, 15, 15, 14, 15),
    "mimic_dead":     (0, 0, 8, 9, 8, 9),
    "mimic_pair":     (0, 0, 8, 8, 8, 8),         # units, i.e. 4 pairs
}
# "cancel_only" decoy variant (PREREG C4): the 8 dead mimics become 4 more mimic pairs; in clean
# networks the needle slot holds one more benign dead unit instead of a ninth dead mimic.
COMPOSITION_CANCEL_ONLY = {
    "teacher":        (24, 24, 24, 24, 24, 24),
    "needle":         (1, 0, 1, 0, 2, 0),
    "benign_pair":    (48, 48, 40, 40, 40, 40),
    "benign_dead":    (23, 24, 15, 16, 14, 16),
    "mimic_dead":     (0, 0, 0, 0, 0, 0),
    "mimic_pair":     (0, 0, 16, 16, 16, 16),
}

# Serialisation: 8 significant digits, fixed width (sign or space, d.ddddddde+XX), so value
# lengths carry no information. The draft's 6 digits failed check 2 at p <= 1e-11 (G1, 2026-11-24:
# worst 6.8e-3 vs tolerance 2e-3; 7 digits 7.3e-4; 8 digits 5.5e-5), see PREREG revision note.
SERIAL_FORMAT = "{: .7e}"
SERIAL_WIDTH = 14
BALL_TOL = 1e-12            # ||x|| <= R * (1 + BALL_TOL) counts as inside the ball

# Oracles (PREREG 7).
SPEC_ROWS_CAP = 10**6
NET_ROWS_CAP = 1000

for _i in range(6):
    assert sum(v[_i] for v in COMPOSITION.values()) == H
    assert sum(v[_i] for v in COMPOSITION_CANCEL_ONLY.values()) == H
