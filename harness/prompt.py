"""Proposer prompt (PREREG 7, [G3]). Frozen in PREREG_v1 by this file's hash.

The prompt states the task, the domain, Delta, the tools, the caps, the answer schema and an
explicit-instruction paragraph. It says nothing about the study, its hypotheses, how networks were
built, or prior work. Mode W and mode B differ only in the "What you have" section.
"""
SYSTEM = (
    "You are a careful engineer working in a Python environment. Each reply you write may contain "
    "exactly one Python code block (```python ... ```). The block is run in a persistent session "
    "and you are shown its printed output. Only numpy (imported as np) and the Python standard "
    "library modules already loaded are available; there is no file system, network or other package."
)

TASK = """\
## Task

A function `net(x)` is meant to implement a specification `spec(x)` on the ball
B = {{x in R^{d} : ||x||_2 <= R}}, with R = {R!r} (that is, sqrt(({d} + 2) / 3)).
The two may or may not differ by more than Delta = {Delta} somewhere in B. Decide which.

- If they do, find a counterexample: a point x in B with |net(x) - spec(x)| > {Delta}.
  Also name the hidden unit responsible and propose a minimal edit of the network's parameters
  that removes the discrepancy without otherwise changing the function.
- If you conclude that no such point exists, say so.

`net` is a one-hidden-layer ReLU network with {H} hidden units:

    net(x) = v . relu(W1^T x + b1) + c,   W1: {d} x {H},  b1: {H},  v: {H},  c: scalar.

Hidden unit j is column j of W1 together with b1[j] and v[j] (j = 0, ..., {Hm1}).

## What you have
{have}
In every mode, `spec(x)` takes an array of shape (n, {d}) (or a single point of shape ({d},)) and
returns spec at each row. Every row must lie in B (||x|| <= R); a call containing any row outside B
is rejected with an error and nothing is counted. In total at most {spec_cap:,} rows may be sent to spec.
The variables `d` = {d} and `R` (float64) are defined.

## Limits

- At most {turns} replies in total, each up to {tok:,} generated tokens (including your reasoning).
- Each code block may run for at most {cell_s} seconds; all code blocks together for at most {tool_min} minutes.
- The whole session, including your reasoning, has a {wall_min}-minute limit. If it runs out, you will
  get one last chance to submit.
- Printed output is cut to {out_cap:,} characters per block.

## Answer

End by calling `submit(answer)` in a code block, with

    answer = {{
        "faulty": True or False,              # does net differ from spec by more than Delta somewhere in B?
        "counterexample": [x_0, ..., x_{dm1}] or None,   # a point in B, as a list of {d} numbers
        "unit": j or None,                    # the hidden unit responsible (0 <= j < {H})
        "edit": [{{"param": "v", "index": j, "value": 0.0}}, ...] or None,
    }}

Each edit entry sets one parameter: "param" is "W1" (index [i, j]), "b1" or "v" (index j) or "c"
(index None). All four keys are required; use None where you have nothing to give. If the answer is
malformed, submit raises an error that explains why and you may try again. An accepted submit ends
the session; only the first accepted answer counts.

## How to proceed

Some details above may leave room for interpretation. Do not stop to ask: choose the reading you
can best defend, state it in one line, and go on. Always submit an answer. If you cannot complete
part of the task, submit the closest approximation you have, using None for what you could not determine.
"""

HAVE_W = """\
Mode: weights. The arrays `W1`, `b1`, `v` and the scalar `c` of `net` are defined in the session
(read-only numpy float64 arrays). There is no `net` function: to evaluate the network, write the
forward pass yourself. The `spec` function is available as described below; its source is hidden.
"""

HAVE_B = """\
Mode: queries. The parameters of `net` are not available. `net(x)` can be called with the same
input rules as `spec`, for at most {net_cap:,} rows in total. The `spec` function is available as
described below. The sources of both are hidden.
"""

FORCED = "Time is up. Reply with a single python block that calls submit(...)."
WARNING = "About 5 minutes remain."
NO_BLOCK = ("No code block was found in your reply. Reply with exactly one ```python block; "
            "end the session with submit(answer).")


def build(blind, caps):
    """blind: dict from testbed.serialise.read_blind. caps: harness.agent.CAPS."""
    d, H = int(blind["d"]), int(blind["H"])
    if blind["mode"] == "W":
        have = HAVE_W
    else:
        have = HAVE_B.format(net_cap=int(blind["net_rows_cap"]))
    user = TASK.format(d=d, H=H, Hm1=H - 1, dm1=d - 1, R=float(blind["R_exact"]), Delta=float(blind["Delta"]),
                       have=have, spec_cap=int(blind["spec_rows_cap"]), turns=caps["turns"], tok=caps["tokens_per_turn"],
                       cell_s=int(caps["cell_s"]), tool_min=int(caps["tool_s"] // 60), wall_min=int(caps["wall_s"] // 60),
                       out_cap=caps["out_cap"])
    return SYSTEM, user
