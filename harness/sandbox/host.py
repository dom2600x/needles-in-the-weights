"""Parent side of the proposer sandbox.

    k = SandboxKernel(scratch_dir, oracles={"spec": fn})   # fn: np.ndarray -> np.ndarray, runs HERE
    k.start(); k.set_array("W1", W1); r = k.exec("print(spec(np.zeros((1, 32))))"); k.close()

Layers: (1) macOS sandbox-exec profile (profile.sb): deny by default, no network, no fork,
reads limited to OS libraries + Python + venv + scratch, writes to scratch only;
(2) an audit hook in the child (kernel.py); (3) rlimits, a per-cell timeout (soft SIGALRM in the
child, hard kill here), a total tool-time budget, an RSS cap and capped output.
Oracles execute in this process; the child sees only stubs. Messages are JSON + raw float64 bytes.
"""
import json, os, resource, select, struct, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.realpath(os.path.join(HERE, "..", ".."))
VENV = os.path.join(REPO, ".venv")
KERNEL = os.path.join(HERE, "kernel.py")
PROFILE = os.path.join(HERE, "profile.sb")
MAX_HEADER = 1 << 20          # 1 MB JSON header cap from the child
MAX_PAYLOAD = 64 << 20        # 64 MB array payload cap from the child (8M float64)


def _ancestors(path):
    out, p = [], os.path.dirname(path)
    while True:
        out.append(p)
        if p == "/":
            return out
        p = os.path.dirname(p)


def build_profile(scratch):
    py_venv = os.path.join(VENV, "bin", "python")
    py_real = os.path.realpath(py_venv)
    # The Python framework root, e.g. /opt/homebrew/Cellar/python@3.13/3.13.18
    py_home = py_real.split("/Frameworks/")[0]
    # The framework build's bin stub re-execs this binary (posix_spawn with SETEXEC).
    py_app = os.path.join(os.path.dirname(os.path.dirname(py_real)), "Resources", "Python.app", "Contents", "MacOS", "Python")
    subs = {"@PY_VENV@": py_venv, "@PY_REAL@": py_real, "@PY_APP@": py_app, "@PY_HOME@": py_home, "@VENV@": VENV,
            "@KERNEL@": KERNEL, "@SCRATCH@": scratch}
    meta = set()
    for p in [py_venv, py_real, py_home, VENV, KERNEL, scratch, "/opt/homebrew/opt/python@3.13"]:
        meta.update(_ancestors(p))
    # Symlink hops Python resolves on the way to its binary and libraries.
    for p in ["/opt/homebrew/opt/python@3.13", "/opt/homebrew/opt/python@3.13/bin",
              "/opt/homebrew/opt/python@3.13/bin/python3.13", os.path.join(VENV, "bin", "python3"),
              "/tmp", "/var", "/etc", "/private/etc/localtime", "/etc/localtime"]:
        meta.add(p)
    meta_rules = "(allow file-read-metadata\n" + "\n".join(f'  (literal "{m}")' for m in sorted(meta)) + ")"
    # Symlinks on the exec path need read access to be followed.
    link_rules = ('(allow file-read* (literal "/opt/homebrew/opt/python@3.13") '
                  '(literal "/opt/homebrew/opt/python@3.13/bin/python3.13"))')
    txt = open(PROFILE).read()
    for k, v in subs.items():
        if '"' in v or "\\" in v:
            raise ValueError(f"unsafe path for SBPL: {v!r}")
        txt = txt.replace(k, v)
    return txt.replace("@METADATA_ALLOWS@", meta_rules + "\n" + link_rules), py_venv


class SandboxKernel:
    def __init__(self, scratch, oracles=None, audit_hook=True, cell_timeout=60.0, tool_budget=600.0,
                 rss_cap_gb=4.0, file_size_cap_mb=64, blas_threads=1):
        os.makedirs(scratch, exist_ok=True)
        self.scratch = os.path.realpath(scratch)
        self.oracles = dict(oracles or {})
        self.audit_hook, self.cell_timeout, self.tool_budget = audit_hook, cell_timeout, tool_budget
        self.rss_cap_kb, self.fsize = int(rss_cap_gb * 1024 * 1024), file_size_cap_mb << 20
        self.blas_threads = int(blas_threads)
        self.tool_used, self.proc, self.info, self._last_check = 0.0, None, None, 0.0

    # ---- framing ----
    def _read_exact(self, n, deadline):
        buf = b""
        while len(buf) < n:
            if time.monotonic() - self._last_check > 0.5:
                self._check(deadline)
            ready, _, _ = select.select([self.r], [], [], 0.5)
            if not ready:
                self._check(deadline)
                continue
            chunk = os.read(self.r, n - len(buf))
            if not chunk:
                raise EOFError("kernel closed its pipe")
            buf += chunk
        return buf

    def _recv(self, deadline):
        (hl,) = struct.unpack(">I", self._read_exact(4, deadline))
        if hl > MAX_HEADER:
            raise ValueError("kernel header too large")
        hdr = json.loads(self._read_exact(hl, deadline))
        nb = int(hdr.get("nbytes") or 0)
        if nb > MAX_PAYLOAD:
            raise ValueError("kernel payload too large")
        return hdr, (self._read_exact(nb, deadline) if nb else b"")

    def _send(self, hdr, data=b""):
        h = json.dumps(dict(hdr, nbytes=len(data))).encode()
        msg = struct.pack(">I", len(h)) + h + data
        view = memoryview(msg)
        while view:
            view = view[os.write(self.w, view):]

    def _check(self, deadline):
        self._last_check = time.monotonic()
        if self.proc.poll() is not None:
            raise EOFError(f"kernel exited with code {self.proc.returncode}")
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("hard timeout")
        try:
            rss = int(subprocess.run(["ps", "-o", "rss=", "-p", str(self.proc.pid)],
                                     capture_output=True, text=True, timeout=5).stdout.strip() or 0)
        except Exception:  # noqa: BLE001
            rss = 0
        if rss > self.rss_cap_kb:
            raise MemoryError(f"kernel RSS {rss // 1024} MB over cap")

    # ---- lifecycle ----
    def start(self):
        profile, py = build_profile(self.scratch)
        with open(os.path.join(self.scratch, ".profile.sb"), "w") as f:
            f.write(profile)
        c_r, p_w = os.pipe()   # parent -> child
        p_r, c_w = os.pipe()   # child -> parent
        fsize = self.fsize
        cpu = int(self.tool_budget + 60)

        def limits():
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))

        argv = ["/usr/bin/sandbox-exec", "-p", profile, py, "-I", "-B", KERNEL, str(c_r), str(c_w), self.scratch]
        if not self.audit_hook:
            argv.append("--no-audit-hook")
        env = {"HOME": self.scratch, "TMPDIR": self.scratch, "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
               "OPENBLAS_NUM_THREADS": str(self.blas_threads), "VECLIB_MAXIMUM_THREADS": str(self.blas_threads)}
        self.proc = subprocess.Popen(argv, cwd=self.scratch, env=env, pass_fds=(c_r, c_w),
                                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     preexec_fn=limits, close_fds=True)
        os.close(c_r); os.close(c_w)
        self.r, self.w = p_r, p_w
        try:
            hdr, _ = self._recv(time.monotonic() + 30)
        except Exception as e:  # noqa: BLE001
            err = self.proc.stderr.read().decode(errors="replace") if self.proc.poll() is not None else ""
            self.kill()
            raise RuntimeError(f"kernel failed to start: {e}\n{err[-2000:]}") from None
        assert hdr.get("op") == "ready", hdr
        for name in self.oracles:
            self._send({"op": "rpc_stub", "name": name}); self._recv(time.monotonic() + 10)
        self.info = hdr
        return hdr

    def set_array(self, name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        self._send({"op": "setvar", "name": name, "shape": list(a.shape)}, a.tobytes())
        self._recv(time.monotonic() + 10)

    def exec(self, code):
        remaining = self.tool_budget - self.tool_used
        if remaining <= 0:
            return {"status": "budget_exhausted", "stdout": "", "stderr": "tool-time budget exhausted", "wall_s": 0.0}
        soft = min(self.cell_timeout, remaining)
        t0 = time.monotonic()
        deadline = t0 + soft + 5.0
        self._send({"op": "exec", "code": code, "timeout": soft})
        try:
            while True:
                hdr, data = self._recv(deadline)
                if hdr.get("op") == "rpc":
                    self._serve_rpc(hdr, data)
                    continue
                if hdr.get("op") == "result":
                    break
                raise ValueError(f"unexpected message {hdr.get('op')!r}")
        except (TimeoutError, MemoryError, EOFError, ValueError) as e:
            self.kill()
            hdr = {"status": "killed", "stdout": "", "stderr": f"{type(e).__name__}: {e}", "wall_s": time.monotonic() - t0}
        self.tool_used += time.monotonic() - t0
        return {k: hdr.get(k) for k in ("status", "stdout", "stderr", "wall_s")}

    def _serve_rpc(self, hdr, data):
        fn = self.oracles.get(hdr.get("name"))
        try:
            if fn is None:
                raise KeyError(f"no oracle {hdr.get('name')!r}")
            shape = [int(s) for s in hdr["shape"]]
            if int(np.prod(shape)) * 8 != len(data):
                raise ValueError("payload size does not match shape")
            x = np.frombuffer(data, dtype=np.float64).reshape(shape)
            y = fn(x)
            if isinstance(y, (dict, list, str, int, float, bool)) or y is None:
                self._send({"ok": True, "json": y})
            else:
                y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
                self._send({"ok": True, "shape": list(y.shape)}, y.tobytes())
        except Exception as e:  # noqa: BLE001  (oracle errors go back to the proposer as text)
            self._send({"ok": False, "error": f"{type(e).__name__}: {e}"})

    def kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill(); self.proc.wait(5)

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self._send({"op": "shutdown"}); self.proc.wait(5)
            except Exception:  # noqa: BLE001
                self.kill()
        for fd in (getattr(self, "r", None), getattr(self, "w", None)):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *a):
        self.close()
