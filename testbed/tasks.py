"""Build a batch of blind tasks plus their truth records.

Linkage defences (PREREG 3.4 check 4b): task IDs are fresh random tokens (never derived from
seeds or configuration); tasks are written in a random order (so creation-time order is random);
every blind file and directory gets the same fixed mtime; layout and byte length depend only on
(mode, d).
"""
import random
import secrets as _secrets

import numpy as np

from . import generator as gen
from . import serialise as ser


def build_tasks(items, blind_root, truth_root):
    """items: list of dicts with keys secret, origin, level, mode, decoys (default "mixed"), and
    optional origin_id and repeat. Returns the list of task IDs in the (random) order written."""
    order = list(range(len(items)))
    random.SystemRandom().shuffle(order)
    ids = []
    for i in order:
        it = items[i]
        o, secret = it["origin"], it["secret"]
        lv = gen.build_level(secret, o, it["level"], decoys=it.get("decoys", "mixed"))
        tid = ser.new_task_id()
        ser.write_blind(blind_root, tid, it["mode"], net=lv.net if it["mode"] == "W" else None, d=o.d)
        rec = dict(task_id=tid, origin_id=it.get("origin_id"), repeat=it.get("repeat"), level=lv.level,
                   decoys=lv.decoys, mode=it["mode"], faulty=o.faulty, d=o.d, R=o.R, p=o.p, g=o.g,
                   masses=o.masses, needle_units=lv.needle_units, roles=lv.roles, pair_of=lv.pair_of,
                   scale_exp=lv.scale_exp, c_cap=o.c_cap, G=o.G, v_n=o.v_n, c_V=o.c_V,
                   median_teacher_norm=o.median_teacher_norm, secret_hex=bytes(secret).hex())
        arrays = dict(W1=lv.net.W1, b1=lv.net.b1, v=lv.net.v, c=np.array(lv.net.c),
                      tW1=o.teacher.W1, tb1=o.teacher.b1, tv=o.teacher.v, tc=np.array(o.teacher.c))
        if o.faulty:
            arrays["u"] = o.u
        ser.write_truth(truth_root, tid, rec, arrays)
        ids.append(tid)
    return ids


def new_origin_id():
    return _secrets.token_hex(8)
