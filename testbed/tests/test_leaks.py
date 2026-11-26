"""PREREG 3.4 check 4: no leakage through unit index, metadata, seeds, linkage or truth fields."""
import json, os, re

import numpy as np
import pytest

from testbed import constants as K
from testbed import generator as gen
from testbed import serialise as ser
from testbed import tasks

CHANCE_MARGIN = 0.05


@pytest.mark.parametrize("level", K.LEVELS)
def test_needle_index_not_predictable(level):
    """A classifier given only the unit index (fit on 100 networks, scored on 100 more) does no
    better than chance + 0.05. The classifier picks the most frequent needle index seen in training."""
    idx = []
    for _ in range(200):
        s = gen.new_secret()
        o = gen.make_origin(s)
        idx.append(gen.build_level(s, o, level).needle_units[0])
    train, test = np.array(idx[:100]), np.array(idx[100:])
    best = np.bincount(train, minlength=K.H).argmax()
    acc = np.mean(test == best)
    assert acc <= 1 / K.H + CHANCE_MARGIN, acc


def _skeleton(text):
    return re.sub(r"[0-9a-f]", "0", re.sub(r"[ -][0-9]\.[0-9]{7}e[+-][0-9]{2}", "#", text))


@pytest.mark.parametrize("mode", ["W", "B"])
def test_blind_metadata_is_constant(tmp_path, mode):
    """Every blind package of the same (mode, d) has the same file set, byte length and layout:
    once the digits are masked, the files are identical, so only values carry information."""
    sizes, skels = set(), set()
    for k in range(40):
        s = gen.new_secret()
        o = gen.make_origin(s, faulty=bool(k % 2), p=[1e-12, 1e-5, 1e-1][k % 3], g=K.G_VALUES[k % 3])
        L = gen.build_level(s, o, K.LEVELS[k % 4], decoys=["mixed", "cancel_only"][k % 2])
        path = ser.write_blind(str(tmp_path), ser.new_task_id(), mode, net=L.net if mode == "W" else None, d=32)
        assert set(os.listdir(path)) == ser.BLIND_FILES
        f = os.path.join(path, "task.json")
        sizes.add(os.path.getsize(f))
        skels.add(_skeleton(open(f).read()))
        assert int(os.stat(f).st_mtime) == ser.FIXED_MTIME and int(os.stat(path).st_mtime) == ser.FIXED_MTIME
    assert len(sizes) == 1 and len(skels) == 1


def test_seed_recovery_fails():
    """Attacker knows the generator and searches plausible seed spaces (small integers, recent
    UNIX times, all-zero/one byte patterns). With 256-bit secrets nothing matches."""
    target = gen.make_origin(gen.new_secret()).teacher.W1[:, 0]
    import time
    now = int(time.time())
    candidates = [i.to_bytes(32, "little") for i in range(20_000)]
    candidates += [t.to_bytes(32, "big") for t in range(now - 5_000, now + 1)]
    candidates += [bytes([b]) * 32 for b in range(256)]
    for c in candidates:
        rng = gen._rng(c, "origin")
        w0 = rng.normal(0.0, K.SIGMA / np.sqrt(32), size=(32, K.N_TEACHER))[:, 0]
        assert not np.allclose(w0, target)


def test_linkage_of_equivalent_copies(tmp_path):
    """Two copies of each of 60 origins (O0 W and O2 W, or W and B). An attacker pairs each task
    with its neighbour in listing, inode, ctime or mtime order, or with the most similar task ID.
    Accuracy must stay within chance + 0.05."""
    items, pairs = [], {}
    for k in range(60):
        s = gen.new_secret(); o = gen.make_origin(s); oid = tasks.new_origin_id()
        copies = [("O0", "W"), ("O2", "W")] if k % 2 else [("O1", "W"), ("O1", "B")]
        for lv, mode in copies:
            items.append(dict(secret=s, origin=o, level=lv, mode=mode, origin_id=oid))
    blind, truth = str(tmp_path / "blind"), str(tmp_path / "truth")
    ids = tasks.build_tasks(items, blind, truth)
    by_id = {}
    for t in ids:
        rec = json.load(open(os.path.join(truth, t, "truth.json")))
        by_id[t] = rec["origin_id"]
    partner = {t: [u for u in ids if u != t and by_id[u] == by_id[t]][0] for t in ids}
    n = len(ids)
    chance = 1 / (n - 1)

    def acc_of_order(order):
        """Attacker guesses one neighbour (the next item, or the previous one for the last)."""
        order = list(order)
        guess = {t: order[i + 1] if i + 1 < n else order[i - 1] for i, t in enumerate(order)}
        return np.mean([guess[t] == partner[t] for t in ids])

    stats = {}
    stats["listdir"] = acc_of_order(os.listdir(blind))
    stats["inode"] = acc_of_order(sorted(ids, key=lambda t: os.stat(os.path.join(blind, t, "task.json")).st_ino))
    stats["ctime"] = acc_of_order(sorted(ids, key=lambda t: os.stat(os.path.join(blind, t, "task.json")).st_ctime_ns))
    stats["mtime"] = acc_of_order(sorted(ids, key=lambda t: (os.stat(os.path.join(blind, t, "task.json")).st_mtime_ns, t)))
    stats["name"] = acc_of_order(sorted(ids))

    def sim(a, b):
        return sum(x == y for x, y in zip(a, b))
    stats["id_similarity"] = np.mean([partner[t] == max((u for u in ids if u != t), key=lambda u: sim(t, u)) for t in ids])
    for k, v in stats.items():
        assert v <= chance + CHANCE_MARGIN, (k, v, chance)


def test_reader_rejects_unknown_fields_and_files(tmp_path):
    s = gen.new_secret(); o = gen.make_origin(s); L = gen.build_level(s, o, "O0")
    path = ser.write_blind(str(tmp_path), ser.new_task_id(), "W", net=L.net)
    ser.read_blind(path)  # fine as written
    f = os.path.join(path, "task.json")
    obj = json.load(open(f))
    os.chmod(f, 0o644)
    for bad in ({**obj, "needle": 3}, {**obj, "weights": {**obj["weights"], "roles": []}},
                {k: v for k, v in obj.items() if k != "R"}):
        json.dump(bad, open(f, "w"))
        with pytest.raises(ValueError):
            ser.read_blind(path)
    json.dump(obj, open(f, "w"))
    ser.read_blind(path)
    open(os.path.join(path, "notes.txt"), "w").write("x")
    with pytest.raises(ValueError):
        ser.read_blind(path)


def test_blind_text_has_no_truth_values(tmp_path):
    items = []
    for lv in K.LEVELS:
        s = gen.new_secret(); o = gen.make_origin(s)
        items.append(dict(secret=s, origin=o, level=lv, mode="W", origin_id=tasks.new_origin_id()))
    blind, truth = str(tmp_path / "blind"), str(tmp_path / "truth")
    for t in tasks.build_tasks(items, blind, truth):
        text = open(os.path.join(blind, t, "task.json")).read()
        rec = json.load(open(os.path.join(truth, t, "truth.json")))
        for word in ("needle", "teacher", "mimic", "benign", "dead", "pair", "role", "origin", "secret", "scale", "level"):
            assert word not in text.lower(), word
        for val in (rec["secret_hex"], rec["origin_id"], f"{rec['p']:.6g}", f"{rec['G']: .7e}"):
            assert str(val) not in text
        assert ser.read_blind(os.path.join(blind, t))["task_id"] == t


def test_linkage_attack_has_power_without_shuffle(tmp_path, monkeypatch):
    """Negative control: if tasks were written in input order, the inode/ctime attack would pair
    copies far above chance. Shows the linkage test above can fail."""
    import random
    monkeypatch.setattr(random.SystemRandom, "shuffle", lambda self, x: None)
    items = []
    for _ in range(30):
        s = gen.new_secret(); o = gen.make_origin(s); oid = tasks.new_origin_id()
        items += [dict(secret=s, origin=o, level="O0", mode="W", origin_id=oid),
                  dict(secret=s, origin=o, level="O2", mode="W", origin_id=oid)]
    blind, truth = str(tmp_path / "blind"), str(tmp_path / "truth")
    ids = tasks.build_tasks(items, blind, truth)
    oid = {t: json.load(open(os.path.join(truth, t, "truth.json")))["origin_id"] for t in ids}
    order = sorted(ids, key=lambda t: os.stat(os.path.join(blind, t, "task.json")).st_ino)
    hits = np.mean([oid[order[i]] == oid[order[i + 1] if i + 1 < len(order) else order[i - 1]] for i in range(len(order))])
    assert hits > 0.4
