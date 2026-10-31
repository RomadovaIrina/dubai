"""r9: does the batched Qwen engine shift the JUDGE (score / natural) vs one-by-one calls? recorded calls, judge prompts only:
sequential replay x2 (its own spread) vs batched with flash attention on / off"""
import glob, json, os, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
R = [json.loads(l) for l in open(sys.argv[1])]
R = [r for r in R if "strict bilingual reviewer" in r["msgs"][0]["content"]]
from llama_cpp import Llama
m = pathlib.Path(__file__).resolve().parents[2] / "models" / "qwen2.5-14b-instruct-gguf"
llm = Llama(model_path=sorted(glob.glob(str(m / "*q5_k_m*.gguf")))[0], n_ctx=4096, n_gpu_layers=-1, verbose=False)
def sc(t):
    a = re.search(r'"score"\s*:\s*(\d)', t); b = re.search(r'"natural"\s*:\s*(\d)', t)
    return (int(a.group(1)) if a else None, int(b.group(1)) if b else None)
rec = [sc(r["answer"]) for r in R]
seq = [sc(llm.create_chat_completion(r["msgs"], max_tokens=r["max_tokens"], temperature=0.0)["choices"][0]["message"]["content"]) for r in R]
out = {}
for fl in ("1", "0"):
    os.environ["DUB_QWEN_FLASH"] = fl
    import importlib, qwen_batch; importlib.reload(qwen_batch)
    e = qwen_batch.Engine(llm); out[fl] = [sc(t) for t in e.chat([(r["msgs"], r["max_tokens"]) for r in R])]
    del e
def cmp(a, b, name):
    d = [(x, y) for x, y in zip(a, b) if x != y]
    ds = sum((y[0] or 0) - (x[0] or 0) for x, y in zip(a, b)); dn = sum((y[1] or 0) - (x[1] or 0) for x, y in zip(a, b))
    lo = sum(1 for x, y in zip(a, b) if (y[0] or 0) < (x[0] or 0) or (y[1] or 0) < (x[1] or 0)); hi = sum(1 for x, y in zip(a, b) if (y[0] or 0) > (x[0] or 0) or (y[1] or 0) > (x[1] or 0))
    print(f"{name}: {len(R) - len(d)}/{len(R)} same (score, natural); sum dscore {ds:+d} dnatural {dn:+d}; lower {lo} higher {hi}")
print(len(R), "judge calls")
cmp(rec, seq, "recorded -> sequential replay")
cmp(rec, out["1"], "recorded -> batched flash")
cmp(rec, out["0"], "recorded -> batched no flash")
cmp(seq, out["1"], "replay -> batched flash")
