import sys, inspect, importlib, itertools, traceback, tempfile, contextlib, io, types
from pathlib import Path
sys.path.insert(0, "tools/pytest_standin"); sys.path.insert(0, ".")
import pytest as P
files = sorted(Path("tests").glob("test_*.py"))
passed = failed = skipped = 0; fails = []
for f in files:
    try:
        mod = importlib.import_module(f"tests.{f.stem}")
    except Exception as e:
        failed += 1; fails.append((f.name, "IMPORT", traceback.format_exc())); print("IMPORT FAIL", f.name); continue
    tests = []
    for name, obj in vars(mod).items():
        if name.startswith("test_") and callable(obj): tests.append((None, name, obj))
        elif name.startswith("Test") and inspect.isclass(obj):
            for n2, o2 in vars(obj).items():
                if n2.startswith("test_"): tests.append((obj, n2, o2))
    for cls, name, fn in tests:
        combos = [{}]
        for names, vals in getattr(fn, "_params", []):
            new = []
            for c in combos:
                for v in vals:
                    vv = v if (len(names) > 1) else (v,)
                    if len(names) > 1 and not isinstance(v, (tuple, list)): vv = (v,)
                    d = dict(c); d.update(dict(zip(names, vv))); new.append(d)
            combos = new
        for kw in combos:
            label = f"{f.stem}::{(cls.__name__+'::') if cls else ''}{name}{kw if kw else ''}"
            tmpdirs = []; cap = io.StringIO()
            try:
                sig = inspect.signature(fn); args = dict(kw)
                for p in sig.parameters:
                    if p in args or p == "self": continue
                    if p == "tmp_path":
                        td = tempfile.TemporaryDirectory(); tmpdirs.append(td); args[p] = Path(td.name)
                    elif p == "capsys":
                        class CS:
                            def readouterr(s):
                                class R: out = cap.getvalue(); err = ""
                                return R
                        args[p] = CS()
                    elif p == "monkeypatch":
                        args[p] = types.SimpleNamespace()
                    elif hasattr(mod, p) and getattr(getattr(mod, p), "_is_fixture", False):
                        args[p] = getattr(mod, p)()
                    else:
                        raise RuntimeError(f"unsupported fixture {p}")
                target = fn
                if cls: 
                    inst = cls(); target = lambda **a: fn(inst, **a)
                with contextlib.redirect_stdout(cap):
                    target(**args)
                passed += 1
            except P.Skipped as e:
                skipped += 1; print("SKIP", label, e)
            except Exception:
                failed += 1; fails.append((label, "FAIL", traceback.format_exc())); print("FAIL", label)
            finally:
                for td in tmpdirs: td.cleanup()
print(f"\nRESULT: passed={passed} failed={failed} skipped={skipped}")
for l, k, tb in fails: print("="*70, "\n", l, "\n", tb[-1500:])
