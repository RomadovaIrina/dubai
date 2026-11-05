### r6_s4: val per checkpoint (EMA; ckpt_val.py, 24 held-out windows)

| ckpt | latent MSE | mouth pix L1 | jitter | stable-SyncNet sim | opening / teacher |
|---|---|---|---|---|---|
| step_00250_ema.pt **pick** | 0.0047 | 0.0234 | 0.0318 | 0.944 | 1.056 |
| step_00500_ema.pt | 0.0042 | 0.0235 | 0.0321 | 0.941 | 1.041 |
| step_00750_ema.pt | 0.0044 | 0.0238 | 0.0316 | 0.945 | 0.972 |
| step_01000_ema.pt | 0.0046 | 0.0236 | 0.0318 | 0.944 | 1.023 |
| step_01250_ema.pt | 0.0048 | 0.0241 | 0.0319 | 0.943 | 1.022 |

pick: step_00250_ema.pt (opening ratio 1.056 >= 1.03; of the 2 within 20% of the lowest MSE (0.0042, step_00500_ema.pt) the highest val sync (0.944, MSE 0.0047))

### End-to-end (dabai_job, q2 + F3 composite, LS_FACE_UNITS=1; base = q2u)

| video | arm | sync_speech | flicker mean | flicker pops | opening vs q2u (mean / speed) | near-closed voiced | GPU s / video-min |
|---|---|---|---|---|---|---|---|
| old03 | q2u | 7.805 | 0.0060 | 15 | 1 | 0.14 | 305 |
| old03 | s4p1500c8u | 7.099 | 0.0057 | 1 | 0.913 / 0.983 | 0.20 | 257 |
| old03 | r5fc8u | 7.056 | 0.0058 | 0 | 0.915 / 0.899 | 0.17 | 260 |
| old03 | r6c8u | 7.929 | 0.0059 | 8 | 1.009 / 0.981 | 0.14 | 262 |
| old03 | r6s500c8u | 8.075 | 0.0061 | 11 | 0.988 / 1.011 | 0.16 | 263 |
| old03 | r6s1250c8u | 8.175 | 0.0061 | 16 | 1.007 / 0.995 | 0.14 | 263 |
| m03 | q2u | 8.779 | 0.0098 | 11 | 1 | 0.16 | 286 |
| m03 | s4p1500c8u | 7.586 | 0.0095 | 12 | 0.981 / 0.965 | 0.16 | 262 |
| m03 | r5fc8u | 7.322 | 0.0095 | 11 | 0.967 / 0.883 | 0.14 | 274 |
| m03 | r6c8u | 8.467 | 0.0097 | 12 | 1.033 / 0.990 | 0.13 | 264 |
| m03 | r6s500c8u | 8.712 | 0.0099 | 12 | 1.029 / 0.949 | 0.15 | 264 |
| m03 | r6s1250c8u | 8.784 | 0.0099 | 12 | 1.046 / 0.907 | 0.12 | 262 |
| m18 | q2u | 8.331 | 0.0152 | 8 | 1 | 0.57 | 326 |
| m18 | r6c8u | 8.202 | 0.0154 | 8 | 1.071 / 0.928 | 0.53 | 314 |
