"""r9 speed (audio lever S, DUB_S3_BATCH=1): S3Gen of many takes at once, each with the same random draws as when vocoded alone (t3_batch.vocode).

S3Gen per take = the flow (conformer encoder + 10 CFM Euler steps of the causal U-Net estimator, CFG = 2 rows) + HiFiGAN + watermark:
~0.26 s at batch 1, 0.21 s of it the flow, i.e. ~21 ms per Euler step on a few hundred mel frames (launch-bound). Here:
  - per take, exactly the single-take path up to the CFM: the flow's own inference() runs and is stopped at its decoder call (the encoder output
    mu, mask, speaker embedding, prompt condition), and the CFM's initial noise z = randn_like(mu) is drawn there with the take's RNG state (the
    first random draw of S3Gen, as in t3_batch.vocode);
  - the Euler solve of all takes together (padded at the END to the longest: the estimator is causal (left-padded convs, LayerNorm per frame)
    and its attention is masked, so the valid frames see no padding);
  - per take again: its mel frames, HiFiGAN with the RNG state right after its z draw, the trim fade, the watermark.
Differences to one-by-one vocoding are last-bit rounding of the batched estimator matmuls."""
from __future__ import annotations
import torch

CHUNK = 16
STATS = {"takes": 0, "chunks": 0}


class _Cap(Exception):
    pass


@torch.inference_mode()
def _prep(m, item, dec):
    """the CFM inputs of one take + its RNG states, by running the single-take path and stopping it at the CFM"""
    from chatterbox.mtl_tts import drop_invalid_tokens
    toks, gen = dec; dev = m.t3.device
    st = drop_invalid_tokens(torch.tensor(toks, dtype=torch.long, device=dev)).to(dev)
    if st.numel() == 0: return None
    torch.cuda.set_rng_state(gen.get_state(), device=dev); torch.default_generator.manual_seed(int(item["seed"]))
    flow = m.s3gen.flow; dec_ = flow.decoder; got = {}
    def cap(mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, noised_mels=None, meanflow=False):
        assert noised_mels is None and not meanflow and temperature == 1.0
        got.update(mu=mu, mask=mask, spks=spks, cond=cond, n=n_timesteps, z=torch.randn_like(mu))   # = CausalConditionalCFM.forward's first line
        raise _Cap()
    dec_.forward = cap   # the CFM module itself stays; its forward is shadowed for this call
    try:
        m.s3gen.flow_inference(st, ref_dict=m.conds.gen, finalize=True)
    except _Cap:
        pass
    finally:
        del dec_.forward
    got["mel1"] = m.conds.gen["prompt_feat"].shape[1]; got["n_tok"] = int(st.shape[-1])
    got["rng"] = (torch.cuda.get_rng_state(device=dev), torch.default_generator.get_state())
    return got


@torch.inference_mode()
def _solve(m, ps):
    """the CFM Euler solve of several takes in one batch -> per take its mel (1, 80, T_i)"""
    cfm = m.s3gen.flow.decoder; T = max(p["mu"].shape[2] for p in ps); B = len(ps)
    pad = lambda x: torch.nn.functional.pad(x, (0, T - x.shape[2]))
    mu = torch.cat([pad(p["mu"]) for p in ps]); z = torch.cat([pad(p["z"]) for p in ps]); mask = torch.cat([pad(p["mask"]) for p in ps])
    cond = torch.cat([pad(p["cond"]) for p in ps]); spks = torch.cat([p["spks"] for p in ps])
    n = ps[0]["n"]; assert all(p["n"] == n for p in ps)
    t_span = torch.linspace(0, 1, n + 1, device=mu.device, dtype=mu.dtype)
    if cfm.t_scheduler == "cosine": t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    feat = cfm.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, meanflow=False)
    return [feat[b:b + 1, :, p["mel1"]:p["mu"].shape[2]] for b, p in enumerate(ps)]


@torch.inference_mode()
def vocode_batch(m, items, decs):
    """= [t3_batch.vocode(m, it, d) for it, d in zip(items, decs)] (wav (1, n) CPU tensor | None per take)"""
    from chatterbox.mtl_tts import S3GEN_SR, S3_TOKEN_RATE
    dev = m.t3.device; ps = [_prep(m, it, d) for it, d in zip(items, decs)]; mels = [None] * len(ps)
    live = [k for k, p in enumerate(ps) if p is not None]
    for s in range(0, len(live), CHUNK):
        ks = live[s:s + CHUNK]
        for k, mel in zip(ks, _solve(m, [ps[k] for k in ks])): mels[k] = mel
        STATS["chunks"] += 1
    out = []
    s3 = m.s3gen
    for k, (it, p) in enumerate(zip(items, ps)):
        if p is None: out.append(None); continue
        torch.cuda.set_rng_state(p["rng"][0], device=dev); torch.default_generator.set_state(p["rng"][1])
        wav, _ = s3.hift_inference(mels[k].to(dtype=s3.dtype), None)
        wav[:, :len(s3.trim_fade)] *= s3.trim_fade
        wav = wav.squeeze(0).detach().cpu().numpy(); wav = wav[: max(1, p["n_tok"] - 1) * (S3GEN_SR // S3_TOKEN_RATE)]
        out.append(torch.from_numpy(m.watermarker.apply_watermark(wav, sample_rate=m.sr)).unsqueeze(0)); STATS["takes"] += 1
    return out
