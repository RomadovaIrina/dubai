"""r15 T: ne_guard windowing test (fake LLM, no GPU): a short script = ONE call with the byte-identical prompt of the old code; a long one = windows
<= 3000 tokens, entities merged without duplicates; a failing window does not lose the others."""
import importlib.util, json, re, sys
# python scripts/r15/test_ne_windows.py [<old ne_guard.py>]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
new = load(str(ROOT / "scripts" / "pilot" / "ne_guard.py"), "ne_new")
old = load(sys.argv[1], "ne_old") if len(sys.argv) > 1 else None   # optional: the pre-r15 ne_guard.py (git show 439d793:scripts/pilot/ne_guard.py)
new.rare = (lambda w: True)   # no tokenizer file needed
if old: old.rare = new.rare


class FakeLLM:
    def __init__(self, fail_window=None): self.prompts = []; self.fail = fail_window
    def n_ctx(self): return 4096
    def tokenize(self, b, add_bos=False): return list(range(len(b) // 3))
    def create_chat_completion(self, msgs, max_tokens, temperature):
        p = msgs[1]["content"]; self.prompts.append(p)
        if self.fail is not None and len(self.prompts) - 1 == self.fail: raise ValueError("boom")
        if len(p.encode()) // 3 > 4096 - max_tokens: raise ValueError(f"Requested tokens ({len(p.encode()) // 3}) exceed context window of 4096")
        items = [{"unit": int(m.group(1)), "source": "Сочи", "english": "Sochi", "respell": "SOH-chee"} for m in re.finditer(r"^(\d+): RU:[^|]*Сочи", p, re.M)]
        return {"choices": [{"message": {"content": json.dumps(items)}}]}


def units(n):
    return [{"id": i, "text": f"Мы приехали в город Сочи номер {i}, здесь очень красиво и тепло" if i % 7 == 0 else f"Это обычное предложение номер {i} без имён",
             "translation": f"We came to the city of Sochi number {i}, it is very beautiful and warm here" if i % 7 == 0 else f"This is an ordinary sentence number {i} with no names"} for i in range(n)]


# 1) short: identical single call
a, b = FakeLLM(), FakeLLM(); ua, ub = units(20), units(20)
sb = new.extract(b, ub, log=lambda *x: None); assert len(b.prompts) == 1 and "windows" not in sb, "short script: not one call"
if old:
    sa = old.extract(a, ua, log=lambda *x: None)
    assert a.prompts == b.prompts, "short script: prompt changed"
    assert sa == sb and [u.get("ne") for u in ua] == [u.get("ne") for u in ub], "short script: result changed"
print("short: 1 call" + (", identical prompt + result to the old code" if old else "") + "; kept", sb["kept"])
# 2) long: the old code fails, the new one windows
a, b = FakeLLM(), FakeLLM(); ua, ub = units(400), units(400)
sb = new.extract(b, ub, log=lambda *x: None)
if old:
    sa = old.extract(a, ua, log=lambda *x: None); assert "error" in sa and "exceed context window" in sa["error"], sa
want = len([u for u in ub if u["id"] % 7 == 0])
assert "error" not in sb and sb["kept"] == want and sb["windows"] > 1, sb
assert all(len(p.encode()) // 3 <= new.WINDOW_TOKENS + 50 for p in b.prompts), [len(p.encode()) // 3 for p in b.prompts]
assert all(len(u.get("ne") or []) <= 1 for u in ub), "duplicate entity from the overlap"
print(f"long: new = {sb['windows']} windows, kept {sb['kept']} / {want}, max window {max(len(p.encode()) // 3 for p in b.prompts)} tokens")
# 3) one window fails: the others still count
c = FakeLLM(fail_window=1); uc = units(400); sc = new.extract(c, uc, log=lambda *x: None)
assert "error" not in sc and 0 < sc["kept"] < want and sc.get("window_errors"), sc
print(f"one failing window: kept {sc['kept']} / {want}, window_errors {len(sc['window_errors'])}")
print("NE_WINDOW_TESTS_OK")
