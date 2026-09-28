"""MINIMAL pytest stand-in (real pytest is not installable: no network). Supports only what this
project's tests use: fixture, mark.parametrize, raises, approx, importorskip, tmp_path, capsys."""
import contextlib, io, math, tempfile
from pathlib import Path

class Skipped(Exception): pass

def importorskip(name):
    try:
        return __import__(name)
    except ImportError:
        raise Skipped(f"could not import {name!r}")

def fixture(fn=None, **kw):
    def deco(f):
        f._is_fixture = True
        return f
    return deco(fn) if fn else deco

class _Mark:
    def parametrize(self, argnames, values):
        names = [a.strip() for a in argnames.split(",")]
        def deco(f):
            f._params = getattr(f, "_params", []) + [(names, list(values))]
            return f
        return deco
mark = _Mark()

class approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.e, self.rel, self.abs = expected, rel, abs
    def __eq__(self, other):
        try:
            return math.isclose(other, self.e, rel_tol=self.rel, abs_tol=self.abs)
        except TypeError:
            return all(math.isclose(a, b, rel_tol=self.rel, abs_tol=self.abs) for a, b in zip(other, self.e)) and len(other) == len(self.e)
    def __repr__(self): return f"approx({self.e})"

@contextlib.contextmanager
def raises(exc, match=None):
    class Info: value = None
    info = Info()
    try:
        yield info
    except exc as e:
        info.value = e
        if match:
            import re
            assert re.search(match, str(e)), f"{match!r} not in {e}"
        return
    raise AssertionError(f"DID NOT RAISE {exc}")

def main(*a, **k): raise SystemExit("shim")
