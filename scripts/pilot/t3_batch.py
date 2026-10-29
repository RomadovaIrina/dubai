"""Chatterbox batched generation (r8 speed, DUB_TTS_BATCH=1): N takes decoded TOGETHER, each exactly as `torch.manual_seed(seed); m.generate(...)`
would make it, except for last-bit GEMM rounding of the shared decode step (measured: 4 of 6 takes bit-identical to upstream generate, the
others diverge from one token on, the same rate as t3_fast vs upstream).

  generate_batch(m, [dict(text=..., seed=..., temperature=0.8, cfg_weight=0.5, language_id="en"), ...]) -> [wav tensor (1, n) | None, ...]
  ladder_batch(m, jobs, assess)  the QA retry ladders of many takes, round by round, with the sequential loop's selection

Per take, unchanged from mtl_tts.generate / t3_fast: punc_norm + tokenisation, the CFG pair (cond + uncond rows), the T3 prefill through the
upstream patched HF model (identical numerics), the sampling chain op for op (CFG combine, repetition penalty, temperature, min_p, top_p,
softmax, multinomial, EOS), drop_invalid_tokens, S3Gen (with the speech-token guard), the trailing-token trim and the watermark.
Batched: only the per-token transformer step, for all 2N rows at once over a static KV cache (per-row position, RoPE and mask), captured as
CUDA graphs per 128-slot length bucket (as t3_fast). The decode step is launch-bound at batch 2 (~8 ms / token on a 5090), so 2N rows cost
about the same as 2.
RNG: each take samples with its OWN CUDA generator seeded with its seed (= the default generator state right after torch.manual_seed(seed)),
and S3Gen then runs with the default generator set to that take state, so every take consumes the same random numbers as when made alone.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

BUCKET = 128
CHUNK = 16      # takes per decode batch (2 x CHUNK rows); the decoder and its graphs are reused across chunks
MAXLEN = 1536   # conditioning (~150-400 tokens) + up to 1000 speech tokens + headroom


class BatchDecoder:
    def __init__(self, t3, rows: int, maxlen: int, cdt):
        self.t3 = t3; self.tf = t3.tfmr; cfg = self.tf.config; p = next(self.tf.parameters()); self.dev, self.wdt = p.device, p.dtype
        self.dt = cdt; self.amp = self.dt != self.wdt; self.R = rows; self.M = maxlen
        self.L, self.H, self.D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        self.KV = getattr(cfg, "num_key_value_heads", self.H)
        self.k = torch.zeros(self.L, rows, self.KV, maxlen, self.D, device=self.dev, dtype=self.dt); self.v = torch.zeros_like(self.k)
        self.x = torch.zeros(rows, 1, cfg.hidden_size, device=self.dev, dtype=self.dt)
        self.pos = torch.zeros(rows, dtype=torch.long, device=self.dev); self.ridx = torch.arange(rows, device=self.dev)
        self.ar = torch.arange(maxlen, device=self.dev); self.graphs = {}; self.outs = {}; self.pool = None; self.scale = self.D ** -0.5

    def _step(self, n: int):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        x = self.x; R = self.R
        cos, sin = self.tf.rotary_emb(x, self.pos.view(R, 1))
        mask = torch.where(self.ar[:n].view(1, n) <= self.pos.view(R, 1), 0.0, float("-inf")).view(R, 1, 1, n)
        for i, layer in enumerate(self.tf.layers):
            a = layer.self_attn; h = layer.input_layernorm(x)
            q = a.q_proj(h).view(R, 1, self.H, self.D).transpose(1, 2)
            k = a.k_proj(h).view(R, 1, self.KV, self.D).transpose(1, 2)
            v = a.v_proj(h).view(R, 1, self.KV, self.D).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            self.k[i][self.ridx, :, self.pos] = k[:, :, 0]; self.v[i][self.ridx, :, self.pos] = v[:, :, 0]
            kk, vv = self.k[i, :, :, :n], self.v[i, :, :, :n]
            if self.KV != self.H: kk, vv = repeat_kv(kk, self.H // self.KV), repeat_kv(vv, self.H // self.KV)
            s = torch.matmul(q, kk.transpose(-1, -2)).float() * self.scale + mask
            o = torch.matmul(torch.softmax(s, dim=-1).to(self.dt), vv)
            x = x + a.o_proj(o.transpose(1, 2).reshape(R, 1, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.t3.speech_head(self.tf.norm(x))[:, -1, :]

    def replay(self, maxpos: int):
        n = min(self.M, (maxpos // BUCKET + 1) * BUCKET)
        if n not in self.graphs:
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            amp = lambda: torch.autocast("cuda", dtype=self.dt, enabled=self.amp, cache_enabled=False)
            with torch.cuda.stream(st), torch.inference_mode(), amp():
                for _ in range(2): self._step(n)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g, pool=self.pool), amp():
                self.outs[n] = self._step(n)
            self.pool = g.pool(); self.graphs[n] = g
        self.graphs[n].replay(); return self.outs[n]

    def load_prefix(self, row: int, past, n: int):   # the take rows (cond, uncond) = rows row, row + 1
        for i in range(self.L):
            kk, vv = (past.key_cache[i], past.value_cache[i]) if hasattr(past, "key_cache") else past[i]
            self.k[i, row:row + 2, :, :n].copy_(kk); self.v[i, row:row + 2, :, :n].copy_(vv)
            self.k[i, row:row + 2, :, n:].zero_(); self.v[i, row:row + 2, :, n:].zero_()


_DECODERS = {}
LAST_TOKENS: list = []   # the speech tokens of the last decode_batch (tests)
STATS = {"takes": 0, "chunks": 0}


@torch.inference_mode()
def _decode_chunk(m, items, max_new_tokens, repetition_penalty, min_p, top_p):
    from chatterbox.mtl_tts import punc_norm
    from chatterbox.models.t3.t3 import _ensure_BOT_EOT
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    from transformers.generation.logits_process import MinPLogitsWarper, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper
    t3 = m.t3; dev = t3.device; N = len(items); C = max(CHUNK, N)
    patched = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb, speech_head=t3.speech_head)
    bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=dev)
    pre = []
    for it in items:   # per take: tokens + embeds + the upstream prefill (B = 2: cond / uncond), exactly as generate / t3_fast
        tt = m.tokenizer.text_to_tokens(punc_norm(it["text"]), language_id=it.get("language_id", "en")).to(dev)
        tt = torch.cat([tt, tt], dim=0); tt = F.pad(tt, (1, 0), value=t3.hp.start_text_token); tt = F.pad(tt, (0, 1), value=t3.hp.stop_text_token)
        _ensure_BOT_EOT(tt, t3.hp)
        embeds, _ = t3.prepare_input_embeds(t3_cond=m.conds.t3, text_tokens=tt, speech_tokens=t3.hp.start_speech_token * torch.ones_like(tt[:, :1]), cfg_weight=it.get("cfg_weight", 0.5))
        be = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0); inp = torch.cat([embeds, torch.cat([be, be])], dim=1)
        out = patched(inputs_embeds=inp, past_key_values=None, use_cache=True, output_attentions=False, output_hidden_states=True, return_dict=True)
        pre.append((inp.size(1), out))
    cdt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else next(t3.tfmr.parameters()).dtype
    mx = max(MAXLEN, ((max(n0 for n0, _ in pre) + max_new_tokens + 1) // BUCKET + 1) * BUCKET); key = (2 * C, mx, cdt)
    if key not in _DECODERS:
        _DECODERS.clear(); torch.cuda.empty_cache(); _DECODERS[key] = BatchDecoder(t3, 2 * C, mx, cdt)
    dec = _DECODERS[key]
    for j, (n0, out) in enumerate(pre): dec.load_prefix(2 * j, out.past_key_values, n0)
    dec.k[:, 2 * N:].zero_(); dec.v[:, 2 * N:].zero_()   # unused rows of a short chunk: an empty prefix, ignored
    gens = [torch.Generator(device=dev).manual_seed(int(it["seed"])) for it in items]
    temp = torch.tensor([float(it.get("temperature", 0.8)) for it in items], device=dev).view(N, 1)
    cfgw = torch.tensor([float(it.get("cfg_weight", 0.5)) for it in items], device=dev).view(N, 1)
    rep = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty)); mp = MinPLogitsWarper(min_p=min_p); tp = TopPLogitsWarper(top_p=top_p)
    logits_step = torch.cat([out.logits[:, -1, :] for _, out in pre], dim=0)   # (2N, V): rows 2j = cond, 2j + 1 = uncond
    ldt = logits_step.dtype
    generated = bos.repeat(N, 1); toks = [[] for _ in range(N)]; done = [False] * N
    n0s = torch.tensor([n0 for n0, _ in pre] + [1] * (C - N), device=dev)
    for i in range(max_new_tokens):
        cond, uncond = logits_step[0:2 * N:2], logits_step[1:2 * N:2]
        logits = cond + cfgw.to(cond.dtype) * (cond - uncond)
        logits = rep(generated, logits)
        logits = logits / temp.to(logits.dtype)
        logits = tp(generated, mp(generated, logits))
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.cat([torch.multinomial(probs[j:j + 1], num_samples=1, generator=gens[j]) if not done[j] else
                         torch.full((1, 1), t3.hp.stop_speech_token, dtype=torch.long, device=dev) for j in range(N)], dim=0)
        generated = torch.cat([generated, nxt], dim=1)
        nl = nxt.view(-1).tolist()
        for j in range(N):
            if done[j]: continue
            toks[j].append(nl[j])
            if nl[j] == t3.hp.stop_speech_token: done[j] = True
        if all(done): break
        e = t3.speech_emb(nxt) + t3.speech_pos_emb.get_fixed_embedding(i + 1)   # (N, 1, d)
        x = e.repeat_interleave(2, dim=0).to(dec.dt)
        if C > N: x = torch.cat([x, torch.zeros(2 * (C - N), 1, x.size(-1), device=dev, dtype=dec.dt)])
        dec.x.copy_(x); dec.pos.copy_((n0s + i).repeat_interleave(2))
        logits_step = dec.replay(int(n0s[:N].max().item()) + i)[:2 * N].to(ldt)
    STATS["chunks"] += 1; STATS["takes"] += N
    return [(t, g) for t, g in zip(toks, gens)]


def decode_batch(m, items: list[dict], max_new_tokens: int = 1000, repetition_penalty: float = 1.2, min_p: float = 0.05, top_p: float = 1.0,
                 exaggeration: float = 0.5) -> list:
    """the speech tokens (+ the take generator, for vocode) of every item, decoded CHUNK at a time"""
    from chatterbox.mtl_tts import T3Cond
    assert m.conds is not None, "prepare_conditionals first"
    if float(exaggeration) != float(m.conds.t3.emotion_adv[0, 0, 0].item()):
        c = m.conds.t3; m.conds.t3 = T3Cond(speaker_emb=c.speaker_emb, cond_prompt_speech_tokens=c.cond_prompt_speech_tokens, emotion_adv=exaggeration * torch.ones(1, 1, 1)).to(device=m.device)
    out = []
    for s in range(0, len(items), CHUNK): out += _decode_chunk(m, items[s:s + CHUNK], max_new_tokens, repetition_penalty, min_p, top_p)
    LAST_TOKENS[:] = [list(t) for t, _ in out]
    return out


@torch.inference_mode()
def vocode(m, item: dict, dec):
    """S3Gen of one decoded take with the default generator at this take state (the same draws as when made alone) -> wav (1, n) | None"""
    from chatterbox.mtl_tts import drop_invalid_tokens, S3GEN_SR, S3_TOKEN_RATE
    toks, gen = dec; dev = m.t3.device
    st = drop_invalid_tokens(torch.tensor(toks, dtype=torch.long, device=dev)).to(dev)
    if st.numel() == 0: return None
    torch.cuda.set_rng_state(gen.get_state(), device=dev); torch.default_generator.manual_seed(int(item["seed"]))   # CPU draws too, as after manual_seed
    wav, _ = m.s3gen.inference(speech_tokens=st, ref_dict=m.conds.gen)
    wav = wav.squeeze(0).detach().cpu().numpy(); wav = wav[: max(1, int(st.shape[-1]) - 1) * (S3GEN_SR // S3_TOKEN_RATE)]
    return torch.from_numpy(m.watermarker.apply_watermark(wav, sample_rate=m.sr)).unsqueeze(0)


def generate_batch(m, items: list[dict], **kw) -> list:
    return [vocode(m, it, d) for it, d in zip(items, decode_batch(m, items, **kw))]


def ladder_batch(m, jobs: list[list[dict]], assess):
    """the QA retry ladders of many takes at once, with the SAME selection as the sequential loop
        take = None
        for entry in ladder: wav = generate(entry); key = assess(...); (take = this if take is None or key < take key); if not key[0]: break
    round r decodes entry r of every job still searching (one batch), then vocodes + assesses them in job order.
    assess(job, r, item, wav) -> (key, payload); returns per job (wav, payload, key) of the chosen take, or None for an empty ladder"""
    best = [None] * len(jobs); active = [j for j, lad in enumerate(jobs) if lad]; r = 0
    while active:
        items = [jobs[j][r] for j in active]; decs = decode_batch(m, items); nxt = []
        for j, it, d in zip(active, items, decs):
            wav = vocode(m, it, d); key, pay = assess(j, r, it, wav)
            if best[j] is None or key < best[j][2]: best[j] = (wav, pay, key)
            if key[0] and r + 1 < len(jobs[j]): nxt.append(j)
        active = nxt; r += 1
    return best
