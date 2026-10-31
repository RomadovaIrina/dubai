"""r9: does the batched Qwen engine write systematically SHORTER / longer candidate versions than one-by-one calls? (the stack's coverage misses
are 'en_short'). Recorded non-judge calls (budgeted versions, piece candidates): answer word counts, recorded vs a sequential replay vs batched
(flash on / off), and how often each lands within the asked budget (+-1)."""
import glob, json, os, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
R = [json.loads(l) for l in open(sys.argv[1])]
R = [r for r in R if "strict bilingual reviewer" not in r["msgs"][0]["content"] and r["temperature"] == 0.0]
from llama_cpp import Llama
m = pathlib.Path(__file__).resolve().parents[2] / "models" / "qwen2.5-14b-instruct-gguf"
llm = Llama(model_path=sorted(glob.glob(str(m / "*q5_k_m*.gguf")))[0], n_ctx=4096, n_gpu_layers=-1, verbose=False)
W = lambda t: len(re.findall(r"[\w']+", t))
def budget(r):
    b = re.search(r"with (\d+) words \(between", r["msgs"][-1]["content"])
    return int(b.group(1)) if b else None
rec = [W(r["answer"]) for r in R]
seq = [W(llm.create_chat_completion(r["msgs"], max_tokens=r["max_tokens"], temperature=0.0)["choices"][0]["message"]["content"]) for r in R]
out = {}
for fl in ("1", "0"):
    os.environ["DUB_QWEN_FLASH"] = fl
    import importlib, qwen_batch; importlib.reload(qwen_batch)
    e = qwen_batch.Engine(llm); out[fl] = [W(t) for t in e.chat([(r["msgs"], r["max_tokens"]) for r in R])]; del e
bs = [budget(r) for r in R]
def show(name, v):
    d = [x - y for x, y in zip(v, rec)]; ib = [abs(x - b) <= 1 for x, b in zip(v, bs) if b]
    print(f"{name}: words total {sum(v)} (recorded {sum(rec)}), mean diff vs recorded {sum(d) / len(d):+.2f}, shorter {sum(x < 0 for x in d)} longer {sum(x > 0 for x in d)}; in budget {sum(ib)}/{len(ib)}")
print(len(R), "non-judge calls")
show("recorded", rec); show("sequential replay", seq); show("batched flash", out["1"]); show("batched no flash", out["0"])
