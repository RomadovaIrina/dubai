# LatentSync 1.6 execution analysis (before any change)

Source: `third_party/latentsync` @ a229c3948406, env `/venv/dabai` (torch 2.7.1+cu128, RTX 5090 sm_120).

## A. Does LatentSync use `torch.nn.functional.scaled_dot_product_attention`?

**Yes.** `latentsync/models/attention.py::Attention.forward` and `motion_module.py::VersatileAttention.forward` both call
`F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)` (comment in source: "Use PyTorch native
implementation of FlashAttention-2"). There is no xformers path and no custom softmax attention. The VAE (diffusers
AutoencoderKL 0.32.2) uses diffusers' default AttnProcessor2_0, i.e. SDPA as well.

Attention sites per UNet call (stage2_512, CFG batch 2 x 16 frames = 32 rows):

| site | q shape (rows, heads, seq, head_dim) | kv seq |
|---|---|---|
| spatial self-attn, 64x64 latent | (32, 8, 4096, 40) | 4096 |
| spatial self-attn, 32x32 | (32, 8, 1024, 80) | 1024 |
| spatial self-attn, 16x16 / mid | (32, 8, 256, 160) | 256 |
| audio cross-attn (whisper 50 tokens x 384) | (32, 8, N, hd) | 50 |
| temporal self-attn (motion module, `(b s) f c`) | (2*h*w, 8, 16, hd) | 16 |

## B. Does PyTorch on RTX 5090 dispatch the fused FLASH_ATTENTION backend for these tensors?

**Yes, by default, with no fallback.** Probe (`torch.profiler`, fp16, default dispatch, no `sdpa_kernel` context):
the CUDA kernel recorded for every representative shape is `pytorch_flash::flash_fwd_kernel<...cutlass::half_t...>`.
Forcing backends with `torch.nn.attention.sdpa_kernel`:

| case | default kernel | FLASH_ATTENTION | EFFICIENT_ATTENTION | MATH | flash vs math max abs diff |
|---|---|---:|---:|---:|---:|
| spatial_self_r1 (4096) | flash_fwd_kernel | 4.692 ms | 9.403 ms | OOM (16 GiB scores) | - |
| spatial_self_r2 (1024) | flash_fwd_kernel | 0.479 ms | 1.341 ms | 8.853 ms | 2.4e-4 |
| spatial_self_r3 (256) | flash_fwd_kernel | 0.079 ms | 0.200 ms | 0.879 ms | 4.9e-4 |
| audio_cross_r1 (4096 x 50) | flash_fwd_kernel | 0.193 ms | 0.165 ms | 2.133 ms | 9.8e-4 |
| temporal_self_r1 (16) | flash_fwd_kernel | 1.336 ms | 0.464 ms | 2.509 ms | 2.0e-3 |

`FLASH_SDPA = AVAILABLE` (PyTorch native fused FlashAttention-2 kernel, `torch.backends.cuda.flash_sdp_enabled() == True`).
It is already the active backend for the dominant spatial attention, so there is no attention-backend speedup left to
collect; the third-party `flash-attn` pip package is not installed and not needed (pilot 0.3 only asked whether it builds).
Note: for the tiny temporal windows (seq 16) EFFICIENT_ATTENTION is faster than flash, but that site is a small share of
the UNet; the sweep records `--sdpa-backend` results rather than guessing.

## C. Are temporal windows processed sequentially with batch dimension = 1?

**Yes.** `LipsyncPipeline.__call__`: `num_inferences = ceil(len(whisper_chunks) / num_frames)`; `for i in range(num_inferences)`
takes `latents = all_latents[:, :, i*16:(i+1)*16]` (shape (1, 4, 16, 64, 64)), builds mask / masked-image / reference
latents for that window, doubles everything for classifier-free guidance (`torch.cat([x]*2)` -> batch 2:
[unconditional, conditional]) and runs the full 20-step DDIM loop before moving to the next window. Windows are
independent (no cross-window state; the initial noise is one latent repeated over all frames), so they can be stacked
on the batch dimension without changing the math.

## D. What `num_frames` means

`num_frames = 16` (configs/unet/stage2_512.yaml) is the TEMPORAL length of one inference window: how many consecutive
video frames the 3D UNet sees at once (`(b, c, f=16, h, w)`, temporal attention across those 16). It is NOT a batch
size. The batch dimension of every UNet call in the upstream loop is 1 (2 with CFG). The last window is shorter when
`len(chunks) % 16 != 0` (e.g. 04.mp4: 1349 frames -> 84 full windows + 1 window of 5 frames = 85 UNet loops of 20 steps).

## Other execution facts relevant to the optimization

- DeepCache (`DeepCacheSDHelper`, cache_interval 3, branch 0) wraps `unet.forward` and block forwards; it keys the
  cache on the DDIM timestep index (`list(scheduler.timesteps).index(t.item())`), refreshed at every step with index % 3 == 0.
  Each window/batch restarts at step 0 (a full pass), so cached tensors never leak between batches of different size.
- `ImageProcessor` (insightface detector + aligner) is re-created on every `__call__`; the accelerated wrapper creates it once.
- `read_video` re-encodes the input to 25 fps CFR (`ffmpeg -r 25 -crf 18`) into `./temp`; the output is written with
  imageio (crf 13) then muxed with `-c:v libx264 -crf 18 -c:a aac`.
- Baseline measurement on 04.mp4 (reports/pilot/optim/latentsync_baseline.json): model load 6.5 s; inference cold
  348.0 s / warm 348.5 s = 388 s per source-video-minute; nvidia-smi process peak 20.9 GB; torch peak allocated 15.1 GB,
  reserved 19.7 GB. Run-to-run determinism: see baseline_determinism_r1_vs_r2.json.
