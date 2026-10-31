"""r9: Qwen 14B Q5_K_M on the CPU (n_gpu_layers=0) for the recorded m01 calls: one-by-one on a subset, and the batched engine (qwen_batch) on all"""
import glob, json, os, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
R = [json.loads(l) for l in open(sys.argv[1])]; R = [r for r in R if r["temperature"] == 0.0]
th = int(sys.argv[2]) if len(sys.argv) > 2 else 32; nsub = int(sys.argv[3]) if len(sys.argv) > 3 else 6
from llama_cpp import Llama
m = pathlib.Path(__file__).resolve().parents[2] / "models" / "qwen2.5-14b-instruct-gguf"
t = time.perf_counter(); llm = Llama(model_path=sorted(glob.glob(str(m / "*q5_k_m*.gguf")))[0], n_ctx=4096, n_gpu_layers=0, n_threads=th, n_threads_batch=th, verbose=False)
print(f"CPU load {time.perf_counter() - t:.1f} s, threads {th}, {len(R)} calls", flush=True)
t = time.perf_counter(); [llm.create_chat_completion(r["msgs"], max_tokens=r["max_tokens"], temperature=0.0) for r in R[:nsub]]; ts = time.perf_counter() - t
rec_s = sum(r["s"] for r in R[:nsub])
print(f"one by one, {nsub} calls: {ts:.1f} s ({ts / nsub:.2f} s/call; the same calls on the GPU one by one: {rec_s:.1f} s)", flush=True)
os.environ["DUB_QWEN_FLASH"] = "1"
import qwen_batch
e = qwen_batch.Engine(llm, n_seq=32, n_ctx=20480)
t = time.perf_counter(); e.chat([(r["msgs"], r["max_tokens"]) for r in R]); tb = time.perf_counter() - t
print(f"batched engine on the CPU, all {len(R)} calls: {tb:.1f} s (GPU batched: 11.2 s; GPU one by one: 65.6 s); {qwen_batch.STATS}", flush=True)
