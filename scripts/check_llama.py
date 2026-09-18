"""Qwen2.5-7B-Instruct GGUF Q8_0 (production quant, pilot 0.9; q4_k_m fallback) via llama.cpp, CPU only, explicit thread count."""
from smoke_common import *
import glob
from llama_cpp import Llama
files = sorted(glob.glob(str(MODELS / "qwen2.5-7b-instruct-gguf" / "*q8_0*.gguf")))   # Q8_0 = production quant (pilot 0.9)
files = files or sorted(glob.glob(str(MODELS / "qwen2.5-7b-instruct-gguf" / "*q4_k_m*.gguf")))
if not files: fail("no q8_0 / q4_k_m gguf found")
print("shards:", [os.path.basename(f) for f in files])
with Timer() as tl:
    llm = Llama(model_path=files[0], n_ctx=4096, n_threads=THREADS, n_threads_batch=THREADS, n_gpu_layers=0, verbose=False)
print(f"loaded in {tl.s:.1f}s, n_threads={THREADS}")
msgs = [{"role": "system", "content": "You are a professional translator. Reply with the translation only."},
        {"role": "user", "content": "Translate to Russian: The quick brown fox jumps over the lazy dog."}]
with Timer() as t:
    r = llm.create_chat_completion(msgs, max_tokens=64, temperature=0.0)
txt = r["choices"][0]["message"]["content"].strip(); u = r["usage"]
print("output:", txt)
print(f"prompt {u['prompt_tokens']} tok, completion {u['completion_tokens']} tok in {t.s:.1f}s -> {u['completion_tokens']/t.s:.1f} tok/s (gen incl. prompt eval)")
if not txt: fail("empty completion")
print("PASS llama.cpp CPU")
