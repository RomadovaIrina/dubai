"""r9 speed: batched Qwen engine (scripts/r9/qwen_batch.py) vs one-by-one create_chat_completion on calls recorded from a real pipeline run
(DUB_QWEN_RECORD). Prints the identity rate and the times.
    python scripts/r9/test_qwen_batch.py <rec.jsonl> [--n 0] [--wave 32]"""
import argparse, glob, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qwen_batch

ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("--n", type=int, default=0); ap.add_argument("--wave", type=int, default=0)
ap.add_argument("--seq", choices=["on", "off"], default="on")
a = ap.parse_args()
R = [json.loads(l) for l in open(a.rec)]
R = [r for r in R if r["temperature"] == 0.0][: a.n or None]
from llama_cpp import Llama
m = pathlib.Path(__file__).resolve().parents[2] / "models" / "qwen2.5-14b-instruct-gguf"
t = time.perf_counter(); llm = Llama(model_path=sorted(glob.glob(str(m / "*q5_k_m*.gguf")))[0], n_ctx=4096, n_gpu_layers=-1, verbose=False)
print(f"load {time.perf_counter() - t:.1f} s, {len(R)} calls", flush=True)
if a.seq == "on":
    t = time.perf_counter(); seq = [llm.create_chat_completion(r["msgs"], max_tokens=r["max_tokens"], temperature=0.0)["choices"][0]["message"]["content"] for r in R]
    ts = time.perf_counter() - t
    print(f"sequential {ts:.1f} s; = recorded {sum(s == r['answer'] for s, r in zip(seq, R))}/{len(R)}", flush=True)
else:
    seq = [r["answer"] for r in R]; ts = sum(r["s"] for r in R)
eng = qwen_batch.Engine(llm)
t = time.perf_counter(); eng.chat([(R[0]["msgs"], 4)]); print(f"engine warm {time.perf_counter() - t:.2f} s")
for w in ([a.wave] if a.wave else [1, 4, 16, 32]):
    eng.n_seq = w; t = time.perf_counter(); out = eng.chat([(r["msgs"], r["max_tokens"]) for r in R]); tb = time.perf_counter() - t
    same = sum(o == s for o, s in zip(out, seq))
    print(f"batched n_seq {w}: {tb:.1f} s (x{ts / tb:.1f}); identical to sequential {same}/{len(R)}", flush=True)
    for o, s, r in list((o, s, r) for o, s, r in zip(out, seq, R) if o != s)[:4]:
        print("  DIFF seq  :", s[:150].replace("\n", " ")); print("       batch:", o[:150].replace("\n", " "))
print(qwen_batch.STATS)
