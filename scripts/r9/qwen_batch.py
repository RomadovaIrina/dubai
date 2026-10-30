"""r9 speed (audio lever Q): Qwen 14B chat calls BATCHED on the resident llama.cpp model (greedy = temperature 0, as every dur_translate call).

A Qwen call in the pipeline is decode-bound (r9 m01: 146 calls, ~1000 prompt tokens + ~40 answer tokens each, 64 s): one sequence per decode
step leaves the GPU idle. Here N chat requests are decoded together in ONE llama.cpp context with N sequences on the SAME weights (a second
context on llm._model: only the KV cache is new, no second model load):
  - the prompt tokens are exactly those create_chat_completion would send (the chat handler is run and intercepted at Llama.generate);
  - prompt prefixes are evaluated once: the calibration shots of every prompt kind (judge, candidates, pieces) stay RESIDENT in cache sequences
    of the unified KV and are shared into each request (llama_memory_seq_cp shares the cells, no copy); requests of one wave share their common
    prefix the same way;
  - every step decodes the last token of every live sequence in one batch; argmax (= the greedy sampler at temperature 0), stop on an
    end-of-generation token, a stop string or max_tokens, exactly as create_completion.
Differences to one-by-one calls are last-bit rounding of the batched matmuls (another greedy path where two logits nearly tie). One-by-one calls
are not reproducible either (llama-cpp-python reuses the previous call's KV prefix): a replay of 150 recorded m01 calls gave 111 / 150 identical.
    eng = Engine(llm); texts = eng.chat([(msgs, max_tokens), ...])
"""
from __future__ import annotations

import ctypes
import os
import time

import numpy as np

STATS = {"calls": 0, "waves": 0, "steps": 0, "prompt_tok": 0, "eval_tok": 0, "cached_tok": 0, "out_tok": 0, "seconds": 0.0, "prompt_s": 0.0, "decode_s": 0.0}
N_SEQ = int(os.environ.get("DUB_QWEN_NSEQ", "32"))
N_CTX = int(os.environ.get("DUB_QWEN_BCTX", "20480"))
N_CACHE = int(os.environ.get("DUB_QWEN_NCACHE", "6"))      # resident prefixes (cache sequences)
MIN_CACHE = 160                                           # tokens: a prefix shorter than this is not worth a cache slot
FLASH = os.environ.get("DUB_QWEN_FLASH", "1") == "1"
N_BATCH = 2048


class _Got(Exception):
    pass


def prompt_tokens(llm, msgs, max_tokens: int):
    """(tokens, stop strings) of llm.create_chat_completion(msgs, max_tokens, temperature=0): the handler formats + tokenizes, Llama._create_completion
    adds its own prompt handling; we stop it at self.generate (instance attributes shadow the bound methods for this call only)."""
    got = {}
    cls_cc = type(llm).create_completion

    def cc(*a, **kw):
        got["stop"] = kw.get("stop")
        return cls_cc(llm, *a, **kw)

    def gen(tokens, *a, **kw):
        got["tokens"] = list(tokens)
        raise _Got()

    llm.create_completion = cc; llm.generate = gen
    try:
        llm.create_chat_completion(msgs, max_tokens=max_tokens, temperature=0.0)
    except _Got:
        pass
    finally:
        del llm.create_completion; del llm.generate
    stop = got.get("stop") or []
    return got["tokens"], [stop] if isinstance(stop, str) else list(stop)


def _lcp(a, b) -> int:
    k, m_ = 0, min(len(a), len(b))
    while k < m_ and a[k] == b[k]: k += 1
    return k


class Engine:
    def __init__(self, llm, n_seq: int = N_SEQ, n_ctx: int = N_CTX, n_cache: int = N_CACHE):
        import llama_cpp
        from llama_cpp import _internals
        self.L, self.llm, self.n_seq, self.n_ctx, self.n_cache = llama_cpp, llm, n_seq, n_ctx, n_cache; self.seq_base = n_seq   # n_seq (wave size) may be lowered later, the cache sequences stay above seq_base
        p = llama_cpp.llama_context_params(); ctypes.memmove(ctypes.byref(p), ctypes.byref(llm.context_params), ctypes.sizeof(p))
        p.n_ctx = n_ctx; p.n_batch = N_BATCH; p.n_ubatch = min(512, N_BATCH); p.n_seq_max = n_seq + n_cache; p.kv_unified = True
        if FLASH: p.flash_attn_type = llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
        self.ctx = _internals.LlamaContext(model=llm._model, params=p, verbose=False)
        self.c = self.ctx.ctx; self.mem = llama_cpp.llama_get_memory(self.c)
        self.vocab = llama_cpp.llama_model_get_vocab(llm._model.model); self.nv = llm.n_vocab()
        self.b = llama_cpp.llama_batch_init(N_BATCH, 0, n_seq + n_cache)
        self.cache = []     # [{"seq": id, "tok": prefix tokens, "used": tick}]
        self.recent = []    # recent prompts (a prefix shared with one of them becomes a cache entry)
        self.tick = 0
        llama_cpp.llama_memory_clear(self.mem, True)

    def cached_cells(self) -> int:
        return sum(len(c["tok"]) for c in self.cache)

    def _flush(self, rows):
        """rows = [(tok, pos, seq, want)] decoded in n_batch chunks; returns {row index: argmax} for the wanted rows"""
        L, b, out = self.L, self.b, {}
        for s in range(0, len(rows), N_BATCH):
            ch = rows[s:s + N_BATCH]; b.n_tokens = len(ch); want = []
            for i, (tok, pos, seq, w) in enumerate(ch):
                b.token[i] = tok; b.pos[i] = pos; b.n_seq_id[i] = 1; b.seq_id[i][0] = seq; b.logits[i] = bool(w)
                if w: want.append(i)
            rc = L.llama_decode(self.c, b)
            if rc != 0: raise RuntimeError(f"llama_decode {rc} (n_tokens {len(ch)}, KV {self.n_ctx}, cached {self.cached_cells()})")
            for i in want:
                lg = np.ctypeslib.as_array(L.llama_get_logits_ith(self.c, i), shape=(self.nv,))
                out[s + i] = int(lg.argmax())
        return out

    def _wave(self, reqs):
        """reqs = [(tokens, stop, max_tokens)] (<= n_seq, KV-fitting): resident / shared prefixes once, then greedy decode of all together"""
        L = self.L; n = len(reqs); t_p = time.perf_counter(); self.tick += 1
        for j in range(self.seq_base): L.llama_memory_seq_rm(self.mem, j, -1, -1)
        # 1. a resident prefix per request (the longest cache entry that is a proper prefix of its prompt)
        base, src = [0] * n, [None] * n
        for j, (t, _, _) in enumerate(reqs):
            for c in self.cache:
                k = len(c["tok"])
                if base[j] < k < len(t) and t[:k] == c["tok"]: base[j], src[j] = k, c
        # 2. requests with the same base whose next tokens agree share their longer common prefix (evaluated once on the first member)
        groups = {}
        for j, (t, _, _) in enumerate(reqs): groups.setdefault((base[j], tuple(t[base[j]:base[j] + 16])), []).append(j)
        own = [0] * n; rows = []
        for (b0, _), js in groups.items():
            t0 = reqs[js[0]][0]; lcp = min(len(reqs[j][0]) for j in js) - 1
            for j in js[1:]: lcp = min(lcp, _lcp(t0, reqs[j][0]))
            if len(js) == 1: lcp = b0
            lcp = max(lcp, b0)
            if src[js[0]] is not None: L.llama_memory_seq_cp(self.mem, src[js[0]]["seq"], js[0], 0, b0); src[js[0]]["used"] = self.tick
            rows += [(t0[pos], pos, js[0], False) for pos in range(b0, lcp)]
            for j in js: own[j] = lcp
        if rows: self._flush(rows)
        STATS["eval_tok"] += len(rows)
        for (b0, _), js in groups.items():
            for j in js[1:]: L.llama_memory_seq_cp(self.mem, js[0], j, 0, own[j])
        rows, last_row = [], {}
        for j, (t, _, _) in enumerate(reqs):
            for pos in range(own[j], len(t)):
                rows.append((t[pos], pos, j, pos == len(t) - 1))
                if pos == len(t) - 1: last_row[len(rows) - 1] = j
            STATS["prompt_tok"] += len(t); STATS["cached_tok"] += base[j]
        STATS["eval_tok"] += len(rows)
        nxt = {last_row[r]: tok for r, tok in self._flush(rows).items()}
        # 3. new resident prefixes: what a prompt shares with a recent one (>= MIN_CACHE tokens, not covered yet), copied from its evaluated sequence
        for j, (t, _, _) in enumerate(reqs):
            k = max([_lcp(t, r) for r in self.recent] + [0])
            k = min(k, len(t) - 1)
            if k >= MIN_CACHE and not any(len(c["tok"]) >= 0.9 * k and t[:len(c["tok"])] == c["tok"] for c in self.cache):
                self._add_cache(j, t[:k])
            self.recent = (self.recent + [t])[-24:]
        STATS["prompt_s"] += time.perf_counter() - t_p; t_d = time.perf_counter()
        outs = [[] for _ in range(n)]; texts = [""] * n; live = set(range(n)); pos = [len(t) for t, _, _ in reqs]
        while live:
            rows, order = [], []
            for j in sorted(live):
                tok = nxt[j]
                if L.llama_vocab_is_eog(self.vocab, tok): live.discard(j); continue
                outs[j].append(tok); texts[j] = self.llm.detokenize(outs[j]).decode("utf-8", errors="ignore")
                stop = reqs[j][1]; hit = [texts[j].index(s_) for s_ in stop if s_ and s_ in texts[j]]
                if hit: texts[j] = texts[j][:min(hit)]; live.discard(j); continue
                if len(outs[j]) >= reqs[j][2]: live.discard(j); continue
                rows.append((tok, pos[j], j, True)); order.append(j); pos[j] += 1
            if not rows: break
            got = self._flush(rows); STATS["steps"] += 1
            for r, j in enumerate(order): nxt[j] = got[r]
        STATS["out_tok"] += sum(len(o) for o in outs); STATS["decode_s"] += time.perf_counter() - t_d
        return texts

    def _add_cache(self, j, tok):
        L = self.L
        if len(self.cache) >= self.n_cache:   # evict the least recently used
            old = min(self.cache, key=lambda c: c["used"]); L.llama_memory_seq_rm(self.mem, old["seq"], -1, -1); self.cache.remove(old); seq = old["seq"]
        else:
            seq = self.seq_base + len(self.cache)
        L.llama_memory_seq_cp(self.mem, j, seq, 0, len(tok)); self.cache.append({"seq": seq, "tok": list(tok), "used": self.tick})

    def chat(self, reqs):
        """reqs = [(msgs, max_tokens)] -> answer texts, in order (waves of <= n_seq requests that fit the KV cache)"""
        t0 = time.perf_counter(); P = [(*prompt_tokens(self.llm, m, mt), mt) for m, mt in reqs]; res = [None] * len(P)
        idx = sorted(range(len(P)), key=lambda j: P[j][0][:64])   # similar prompts together (prefix sharing)
        room = self.n_ctx - self.cached_cells() - 2 * max([len(P[j][0]) for j in idx] + [0]) - 64   # the new cache entries of a wave need cells too
        wave, need = [], 0
        def run(w):
            for j, t in zip(w, self._wave([P[j] for j in w])): res[j] = t
            STATS["waves"] += 1
        for j in idx:   # KV cells: a request's own tokens beyond what it shares with the previous one (sorted, so shared prefixes are adjacent) + its answer
            t = P[j][0]; k = _lcp(t, P[wave[-1]][0]) if wave else 0
            k = max(k, max([len(c["tok"]) for c in self.cache if len(c["tok"]) < len(t) and t[:len(c["tok"])] == c["tok"]] + [0]))
            c = len(t) - k + P[j][2] + 8
            if wave and (len(wave) >= self.n_seq or need + c > room):
                run(wave); wave, need = [], 0
                room = self.n_ctx - self.cached_cells() - 2 * len(t) - 64
            wave.append(j); need += c
        if wave: run(wave)
        STATS["calls"] += len(P); STATS["seconds"] += time.perf_counter() - t0
        return res


_ENGINES = {}


def engine(llm):
    if id(llm) not in _ENGINES: _ENGINES[id(llm)] = Engine(llm)
    return _ENGINES[id(llm)]
