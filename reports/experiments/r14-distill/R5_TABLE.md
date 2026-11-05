### r5_s4: val per checkpoint (EMA; ckpt_val.py, 24 held-out windows)

| ckpt | latent MSE | mouth pix L1 | jitter | stable-SyncNet sim | opening / teacher |
|---|---|---|---|---|---|
| step_00250_ema.pt | 0.0054 | 0.0229 | 0.0302 | 0.944 | 1.184 |
| step_00500_ema.pt | 0.0055 | 0.0300 | 0.0305 | 0.946 | 1.274 |
| step_00750_ema.pt | 0.0075 | 0.0336 | 0.0295 | 0.947 | 1.221 |
| step_01000_ema.pt | 0.0076 | 0.0313 | 0.0295 | 0.949 | 1.217 |
| step_01250_ema.pt **pick** | 0.0065 | 0.0250 | 0.0298 | 0.949 | 1.141 |
| step_01500_ema.pt | 0.0062 | 0.0243 | 0.0299 | 0.948 | 1.176 |

pick: step_01250_ema.pt (opening ratio 1.141 >= 1.03; of the 4 within 20% of the lowest MSE (0.0054, step_00250_ema.pt) the highest val sync (0.949, MSE 0.0065))

### End-to-end (dabai_job, q2 + F3 composite, LS_FACE_UNITS=1; base = q2u)

| video | arm | sync_speech | flicker mean | flicker pops | opening vs q2u (mean / speed) | near-closed voiced | GPU s / video-min |
|---|---|---|---|---|---|---|---|
| old03 | q2u | 7.805 | 0.0060 | 15 | 1 | 0.14 | 305 |
| old03 | s4p1500c8u | 7.099 | 0.0057 | 1 | 0.913 / 0.983 | 0.20 | 257 |
| old03 | r5c8u | 7.000 | 0.0058 | 0 | 0.898 / 0.900 | 0.18 | 268 |
| old03 | r5fc8u | 7.056 | 0.0058 | 0 | 0.915 / 0.899 | 0.17 | 260 |
| m03 | q2u | 8.779 | 0.0098 | 11 | 1 | 0.16 | 286 |
| m03 | s4p1500c8u | 7.586 | 0.0095 | 12 | 0.981 / 0.965 | 0.16 | 262 |
| m03 | r5fc8u | 7.322 | 0.0095 | 11 | 0.967 / 0.883 | 0.14 | 274 |
