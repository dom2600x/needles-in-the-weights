"""Strategy codebook (PREREG section 10): provisional classes by pattern on code cells, for hand audit.

Classes (a cell can have several):
  F  fuzz: at least 1e4 self-implemented network evaluations (F-large: at least 1e8)
  N  norm scan: row/column norms or ||w_j|| |v_j| across units
  M  merge: grouping or comparing incoming forms up to scale (duplicates, cancellation)
  R  reachability: max pre-activation over the ball (b_j + R ||w_j||), interval bound or LP
  O  other

Pattern rules give a provisional class. F also needs an evaluation count, which code alone does not
give: the cell's tool time bounds it from above (EVALS_PER_S, the G0 sandbox forward-pass rate), so
a cell with wall_s * EVALS_PER_S < 1e4 cannot be F and one below 1e8 cannot be F-large. For every
successful run, the class of the submitting cell, including the F count, is fixed by hand reading
(estimating evaluations from array sizes and loop counts); the pattern class is kept beside it.
"""
import re

EVALS_PER_S = 1.83e7        # G0: float32 forward pass in the sandbox, d = 32, H = 96 (upper bound)
F_MIN, F_LARGE = 1e4, 1e8

RX = {
    "sampling": re.compile(r"\b(?:random\.|rng\.|default_rng|standard_normal|normal\(|uniform\(|randn|rand\()"),
    "forward": re.compile(r"(?:@\s*W1\b|W1\s*\)|\bdot\s*\([^)]*W1|\beinsum\b|\bnet\s*\(|maximum\s*\(\s*0|relu)"),
    "norm_scan": re.compile(r"(?:linalg\.norm\s*\(\s*W1[^)]*axis\s*=\s*0|norm\s*\([^)]*W1|abs\s*\(\s*v\s*\)|"
                            r"sqrt\s*\(\s*\(?\s*W1\s*\*\*\s*2|sum\s*\(\s*W1\s*\*\*\s*2)"),
    "merge": re.compile(r"(?:W1\s*/\s*\w*norm|/\s*np\.linalg\.norm\s*\(\s*W1|np\.unique|allclose|isclose|"
                        r"cosine|\bcos_?sim|combinations|duplicate|cancel|pair)", re.I),
    "reach": re.compile(r"(?:b1\s*\+\s*R\s*\*|R\s*\*\s*\w*norm\w*\s*\+\s*b1|R\s*\*\s*W1\s*\[|reachab|linprog|"
                        r"interval|max_pre|maximi[sz]er)", re.I),
}


def classify_cell(code, wall_s=None):
    code = code or ""
    cls = set()
    if RX["sampling"].search(code) and RX["forward"].search(code):
        cap = None if wall_s is None else wall_s * EVALS_PER_S
        if cap is None or cap >= F_MIN:
            cls.add("F?")            # candidate; count fixed by hand
        if cap is not None and cap >= F_LARGE:
            cls.add("F-large?")
    if RX["norm_scan"].search(code):
        cls.add("N")
    if RX["merge"].search(code):
        cls.add("M")
    if RX["reach"].search(code):
        cls.add("R")
    return sorted(cls) or ["O"]


REFUSAL = re.compile(r"\b(?:I (?:can(?:'|no)t|cannot|won't|am unable to|'m unable to) (?:help|assist|do|complete|"
                     r"perform)|not able to (?:help|assist)|I must decline)\b", re.I)


def refusal_candidate(texts):
    """True if any reply reads like a refusal (hand-checked before coding a run as a refusal)."""
    return any(REFUSAL.search(t or "") for t in texts)
