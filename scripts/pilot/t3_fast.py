"""Chatterbox T3 fast decode (speed track, --tts-fast-t3 on): the upstream T3.inference loop runs one eager HF Llama forward per speech
token (DynamicCache, ~8 ms/token on a 5090: Python / launch bound, 30 layers x ~10 kernels). This wrapper keeps everything that decides the
take and replaces only the per-token transformer call:

  - prefill: the upstream patched HF model on the full conditioning (identical numerics), its DynamicCache copied into a static KV cache;
  - decode step: the model's OWN modules (input/post layernorms, q/k/v/o projections, MLP, final norm, speech_head), llama3 RoPE at the
    absolute position (HF: position_ids = cache_position), SDPA over the static cache with the not-yet-written slots masked;
    attention as two small GEMMs over the filled cache only; captured as CUDA graphs per 128-slot length bucket (batch 2 = CFG, lazily,
    one memory pool) and replayed per token. Measured: torch.compile made the step slower (24 vs 7.6 ms: the cache writes), SDPA over the whole
    static cache was 57 % of the step;
  - sampling: unchanged, op for op (CFG combine, repetition penalty, temperature, min_p, top_p, softmax, multinomial, EOS check every step).

Takes are statistically the same model, not bit-identical (masked SDPA over the padded cache rounds differently in the last bits, and one
flipped sample changes the rest of the take): the speed track's quality gate decides. No upstream file is edited.
    from t3_fast import install_fast_t3; install_fast_t3(m)      # after ChatterboxMultilingualTTS.from_local (+ any dtype cast)"""
from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

MAXLEN = 2048   # conditioning (~150-350 tokens) + up to 1000 speech tokens + headroom

log = logging.getLogger("t3_fast")


class FastDecoder:
    BUCKET = 128   # one graph per filled-length bucket: attention over the filled cache only (over all MAXLEN slots it was 57 % of the step)

    def __init__(self, t3, cdt=None):
        self.t3 = t3; self.tf = t3.tfmr; cfg = self.tf.config
        p = next(self.tf.parameters()); self.dev, self.wdt = p.device, p.dtype
        self.dt = cdt or p.dtype   # compute dtype: the caller's autocast dtype (retake: fp32 weights under fp16 autocast) or the weights'
        self.amp = self.dt != self.wdt
        self.L, self.H, self.D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        self.k = torch.zeros(self.L, 2, self.H, MAXLEN, self.D, device=self.dev, dtype=self.dt); self.v = torch.zeros_like(self.k)
        self.x = torch.zeros(2, 1, cfg.hidden_size, device=self.dev, dtype=self.dt)
        self.pos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.ar = torch.arange(MAXLEN, device=self.dev)
        self.graphs = {}; self.outs = {}; self.pool = None; self.scale = self.D ** -0.5

    def _step(self, n: int):
        """One decode step attending to cache slots [0, n) (n = the bucket end >= pos + 1; slots > pos are masked)."""
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        x = self.x; pid = self.pos.view(1, 1)
        cos, sin = self.tf.rotary_emb(x, pid)
        mask = torch.where(self.ar[:n] <= self.pos, 0.0, float("-inf")).view(1, 1, 1, n)   # fp32 (scores are upcast)
        for i, layer in enumerate(self.tf.layers):
            a = layer.self_attn; h = layer.input_layernorm(x)
            q = a.q_proj(h).view(2, 1, self.H, self.D).transpose(1, 2)
            k = a.k_proj(h).view(2, 1, self.H, self.D).transpose(1, 2)
            v = a.v_proj(h).view(2, 1, self.H, self.D).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            self.k[i].index_copy_(2, self.pos, k); self.v[i].index_copy_(2, self.pos, v)
            s = torch.matmul(q, self.k[i, :, :, :n].transpose(-1, -2)).float() * self.scale + mask   # q_len 1: two small GEMMs beat the fused kernels
            o = torch.matmul(torch.softmax(s, dim=-1).to(self.dt), self.v[i, :, :, :n])
            x = x + a.o_proj(o.transpose(1, 2).reshape(2, 1, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.t3.speech_head(self.tf.norm(x))[:, -1, :]

    def replay(self, pos: int):
        n = min(MAXLEN, (pos // self.BUCKET + 1) * self.BUCKET)
        if n not in self.graphs:   # lazily, one per bucket, all in one memory pool
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            amp = lambda: torch.autocast("cuda", dtype=self.dt, enabled=self.amp, cache_enabled=False)   # no autocast weight-cast cache inside a graph
            with torch.cuda.stream(st), torch.inference_mode(), amp():
                for _ in range(2): self._step(n)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g, pool=self.pool), amp():
                self.outs[n] = self._step(n)
            self.pool = g.pool(); self.graphs[n] = g
        self.graphs[n].replay(); return self.outs[n]

    def load_prefix(self, past, n: int):
        for i in range(self.L):
            kk, vv = (past.key_cache[i], past.value_cache[i]) if hasattr(past, "key_cache") else past[i]
            self.k[i, :, :, :n].copy_(kk); self.v[i, :, :, :n].copy_(vv)
            self.k[i, :, :, n:].zero_(); self.v[i, :, :, n:].zero_()


def install_fast_t3(m) -> None:
    t3 = m.t3; orig = t3.inference
    from transformers.generation.logits_process import MinPLogitsWarper, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    state = {"dec": None, "dtype": None, "fallbacks": 0, "calls": 0}

    @torch.inference_mode()
    def inference(*, t3_cond, text_tokens, initial_speech_tokens=None, prepend_prompt_speech_tokens=None, num_return_sequences=1, max_new_tokens=None,
                  stop_on_eos=True, do_sample=True, temperature=0.8, top_p=0.95, min_p=0.05, length_penalty=1.0, repetition_penalty=1.2, cfg_weight=0.5):
        from chatterbox.models.t3.t3 import _ensure_BOT_EOT
        _ensure_BOT_EOT(text_tokens, t3.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=t3.device)
        if text_tokens.size(0) != 2 or not t3.device.type == "cuda":
            state["fallbacks"] += 1; return orig(t3_cond=t3_cond, text_tokens=text_tokens, initial_speech_tokens=initial_speech_tokens, max_new_tokens=max_new_tokens,
                                                  temperature=temperature, top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty, cfg_weight=cfg_weight)
        state["calls"] += 1
        if initial_speech_tokens is None:
            initial_speech_tokens = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=initial_speech_tokens, cfg_weight=cfg_weight)
        patched = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb, speech_head=t3.speech_head)
        dev = embeds.device
        bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=dev)
        bos_e = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0); bos_e = torch.cat([bos_e, bos_e])
        inputs = torch.cat([embeds, bos_e], dim=1); n0 = inputs.size(1); max_new = max_new_tokens or t3.hp.max_speech_tokens
        if n0 + max_new + 1 > MAXLEN:
            state["fallbacks"] += 1; return orig(t3_cond=t3_cond, text_tokens=text_tokens, initial_speech_tokens=initial_speech_tokens, max_new_tokens=max_new_tokens,
                                                  temperature=temperature, top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty, cfg_weight=cfg_weight)
        out = patched(inputs_embeds=inputs, past_key_values=None, use_cache=True, output_attentions=False, output_hidden_states=True, return_dict=True)
        cdt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else next(t3.tfmr.parameters()).dtype
        if state["dec"] is None or state["dtype"] != cdt:
            state["dec"] = FastDecoder(t3, cdt); state["dtype"] = cdt
        dec = state["dec"]; dec.load_prefix(out.past_key_values, n0)
        generated = bos.clone(); predicted = []
        top_p_w = TopPLogitsWarper(top_p=top_p); min_p_w = MinPLogitsWarper(min_p=min_p); rep = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
        logits_step = out.logits[:, -1, :]
        for i in range(max_new):
            cond, uncond = logits_step[0:1, :], logits_step[1:2, :]
            logits = cond + torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype) * (cond - uncond)
            ids = generated[:1, ...]; logits = rep(ids, logits)
            if temperature != 1.0: logits = logits / temperature
            logits = top_p_w(ids, min_p_w(ids, logits))
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            predicted.append(nxt); generated = torch.cat([generated, nxt], dim=1)
            if nxt.view(-1) == t3.hp.stop_speech_token: break
            e = t3.speech_emb(nxt) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            dec.x.copy_(torch.cat([e, e]).to(dec.dt)); dec.pos.fill_(n0 + i)
            logits_step = dec.replay(n0 + i).to(out.logits.dtype)
        return torch.cat(predicted, dim=1)

    t3.inference = inference; t3._fast_state = state
