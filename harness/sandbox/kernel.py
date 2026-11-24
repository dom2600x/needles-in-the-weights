"""Persistent proposer kernel. Runs INSIDE the sandbox, started by host.py.

Protocol on inherited pipes (fd numbers in argv): each message is a 4-byte big-endian header
length, a JSON header, then `nbytes` raw bytes (float64, C order) if the header has nbytes > 0.
No pickle in either direction.

Defence in depth: after setup the kernel installs an audit hook (unless started with
--no-audit-hook, used only by the escape tests to check the OS layer on its own).
"""
import io, json, os, signal, struct, sys, time, traceback

import numpy as np  # noqa: F401  (pre-imported so user code can use it)
import numpy.random, numpy.linalg, numpy.fft, numpy.polynomial  # noqa: E401,F401  (they import stdlib modules the hook would refuse)
import math, random, itertools, functools, collections, heapq, bisect, statistics, re  # noqa: E401,F401
import fractions, decimal, string, operator, copy, dataclasses, typing  # noqa: E401,F401

FD_IN, FD_OUT, SCRATCH = int(sys.argv[1]), int(sys.argv[2]), os.path.realpath(sys.argv[3])
AUDIT = "--no-audit-hook" not in sys.argv
OUT_CAP = 4000
fin, fout = os.fdopen(FD_IN, "rb", buffering=0), os.fdopen(FD_OUT, "wb", buffering=0)


def _read_exact(n):
    buf = b""
    while len(buf) < n:
        chunk = fin.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def recv():
    (hl,) = struct.unpack(">I", _read_exact(4))
    hdr = json.loads(_read_exact(hl))
    data = _read_exact(hdr["nbytes"]) if hdr.get("nbytes") else b""
    return hdr, data


def send(hdr, data=b""):
    hdr = dict(hdr, nbytes=len(data))
    h = json.dumps(hdr).encode()
    fout.write(struct.pack(">I", len(h)) + h + data)


def _to_array(hdr, data):
    a = np.frombuffer(data, dtype=np.float64).reshape(hdr["shape"]).copy()
    a.setflags(write=False)
    return a


def _make_rpc(name):
    def rpc(x):
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
        send({"op": "rpc", "name": name, "shape": list(x.shape)}, x.tobytes())
        hdr, data = recv()
        if not hdr.get("ok"):
            raise RuntimeError(hdr.get("error", "oracle error"))
        if hdr.get("json") is not None:
            return hdr["json"]
        return _to_array(hdr, data)
    rpc.__name__ = name
    return rpc


class CappedWriter(io.TextIOBase):
    def __init__(self):
        self.parts, self.n, self.total = [], 0, 0
    def write(self, s):
        self.total += len(s)
        if self.n < OUT_CAP:
            take = s[: OUT_CAP - self.n]
            self.parts.append(take); self.n += len(take)
        return len(s)
    def value(self):
        v = "".join(self.parts)
        return v + (f"\n[output truncated: {self.total} chars, showing {OUT_CAP}]" if self.total > OUT_CAP else "")


class _Timeout(BaseException):  # BaseException so user `except Exception` cannot swallow it
    pass


def _alarm(signum, frame):
    raise _Timeout()


# ---------- audit hook (interpreter layer) ----------
ALLOWED_NEW_IMPORTS = {"numpy"}  # plus numpy.* submodules (lazy loads) and stdlib already pre-imported
_lib_roots = tuple(os.path.realpath(p) for p in {sys.prefix, sys.base_prefix, os.path.dirname(np.__file__)})
_BLOCK_PREFIXES = ("os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
                   "subprocess.", "socket.", "ctypes.", "_ctypes.", "os.kill", "os.killpg", "os.putenv",
                   "os.unsetenv", "os.chroot", "os.setuid", "webbrowser.", "urllib.", "http.", "ftplib.",
                   "smtplib.", "imaplib.", "poplib.", "telnetlib.", "nntplib.", "_posixsubprocess.",
                   "sqlite3.", "winreg.", "msvcrt.", "resource.setrlimit", "sys.setprofile",
                   "sys.settrace", "pty.", "fcntl.", "shutil.")
_PATH_EVENTS = {"os.listdir", "os.scandir", "os.chdir", "os.remove", "os.rename", "os.rmdir", "os.mkdir",
                "os.symlink", "os.link", "os.truncate", "os.chmod", "os.chown", "os.utime", "glob.glob"}


def _inside(path, roots):
    try:
        p = os.path.realpath(os.fsdecode(path)) if not isinstance(path, int) else None
    except Exception:  # noqa: BLE001
        return False
    return p is not None and any(p == r or p.startswith(r + os.sep) for r in roots)


def _hook(event, args):
    if event == "import":
        mod = args[0]
        if mod in ALLOWED_NEW_IMPORTS or mod.startswith("numpy.") or mod in sys.modules:
            return
        raise ImportError(f"sandbox: import of '{mod}' is not allowed (numpy only)")
    if event.startswith(_BLOCK_PREFIXES):
        raise PermissionError(f"sandbox: '{event}' is not allowed")
    if event == "open":
        path, mode = args[0], args[1] or "r"
        if isinstance(path, int):
            return
        writing = any(c in str(mode) for c in "wax+") if not isinstance(mode, int) else bool(mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if _inside(path, (SCRATCH,)):
            return
        if not writing and _inside(path, _lib_roots):
            return
        raise PermissionError(f"sandbox: open('{path}') outside scratch is not allowed")
    if event in _PATH_EVENTS:
        for a in args:
            if isinstance(a, (str, bytes, os.PathLike)) and not _inside(a, (SCRATCH,)):
                raise PermissionError(f"sandbox: '{event}' outside scratch is not allowed")


# ---------- main loop ----------
os.chdir(SCRATCH)
user_ns = {"__name__": "__main__", "np": np, "numpy": np}
signal.signal(signal.SIGALRM, _alarm)
if AUDIT:
    sys.addaudithook(_hook)
send({"op": "ready", "audit_hook": AUDIT, "numpy": np.__version__, "python": sys.version.split()[0]})

while True:
    try:
        hdr, data = recv()
    except EOFError:
        break
    op = hdr.get("op")
    if op == "setvar":
        user_ns[hdr["name"]] = _to_array(hdr, data) if hdr.get("nbytes") else hdr.get("value")
        send({"op": "ok"})
    elif op == "rpc_stub":
        user_ns[hdr["name"]] = _make_rpc(hdr["name"])
        send({"op": "ok"})
    elif op == "exec":
        out, err = CappedWriter(), CappedWriter()
        old = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        status, t0 = "ok", time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, float(hdr.get("timeout", 60)))
        try:
            exec(compile(hdr["code"], "<cell>", "exec"), user_ns)  # noqa: S102
        except _Timeout:
            status = "timeout"; err.write("TimeoutError: cell exceeded its time limit\n")
        except SystemExit:
            status = "error"; err.write("SystemExit is not allowed in a cell\n")
        except BaseException:  # noqa: BLE001
            status = "error"; err.write(traceback.format_exc(limit=5))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            sys.stdout, sys.stderr = old
        send({"op": "result", "status": status, "stdout": out.value(), "stderr": err.value(),
              "wall_s": round(time.perf_counter() - t0, 4)})
    elif op == "shutdown":
        break
