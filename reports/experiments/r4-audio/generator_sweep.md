# r4 LatentSync generator sweep on the worst lip-sync windows (audio v1)

Windows (7, 8 s each, worst per-second dub-only SyncNet conf_at0 of a full v4-generator run on audio v1): 03 20-28 s, 03 40-48 s, 03 67-75 s, 04 10-18 s, 04 32-40 s, 04 42-50 s, 05 31-39 s

Generator code = v4 (904c5d2: A R3 generator, fp32 AlignRestore, flow:4 anchors, head mask) via `ls_window_ab.py`; one setting changed per arm.
Metrics: conf_at0 = SyncNet confidence at offset 0 per second over seconds where the dub speaks (mean / worst second); sync_conf = window SyncNet;
lip_r = inner-lip gap vs driving audio log-RMS (Pearson); flash = single-frame lip spikes vs the master (r3_flash); jaw = jaw-contour jitter / master
(1.0 = as steady as the source); s/frame = LatentSync seconds per generated frame (GPU shared with other jobs part of the time: indicative).

| arm | setting | conf_at0_speech_mean | conf_at0_worst_s | sync_conf | lip_r | flash | jaw_ratio | s_per_ls_frame | wins vs base (conf_at0) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_base | v4 / R3: guidance 2.0, 20 steps, bridge 8x4, seed 1247 | 5.495 | 3.705 | 4.609 | 0.571 | 9.857 | 0.942 | 0.376 |  |
| A5_seed7 | seed 7 | 5.257 | 3.533 | 4.444 | 0.589 | 8.857 | 0.952 | 0.327 | 0/7 |
| A6_seed99 | seed 99 | 5.266 | 3.531 | 4.436 | 0.549 | 8.429 | 0.949 | 0.336 | 0/7 |
| A1_g15 | guidance 1.5 | 5.125 | 3.614 | 4.325 | 0.492 | 3.714 | 0.936 | 0.260 | 0/7 |
| A2_g25 | guidance 2.5 | 5.913 | 4.066 | 4.955 | 0.634 | 17.429 | 0.975 | 0.292 | 7/7 |
| A3_s30 | 30 steps | 5.494 | 3.684 | 4.589 | 0.562 | 10.429 | 0.955 | 0.401 | 4/7 |
| A4_nobr | bridge off | 5.490 | 3.771 | 4.591 | 0.569 | 10.286 | 0.952 | 0.261 | 4/7 |

Seed spread (max - min over the three seeds, same settings) = the noise floor: conf_at0_speech_mean 0.238, conf_at0_worst_s 0.174, sync_conf 0.173, lip_r 0.040, flash 1.429, jaw_ratio 0.011, s_per_ls_frame 0.049

## Per window: conf_at0 (speech mean) per arm

| window | A0_base | A5_seed7 | A6_seed99 | A1_g15 | A2_g25 | A3_s30 | A4_nobr |
|---|---:|---:|---:|---:|---:|---:|---:|
| 03 20-28 s | 5.74 | 5.57 | 5.59 | 4.96 | 6.23 | 5.72 | 5.66 |
| 03 40-48 s | 7.58 | 7.35 | 7.33 | 6.87 | 7.80 | 7.59 | 7.60 |
| 03 67-75 s | 6.59 | 6.43 | 6.27 | 5.77 | 7.08 | 6.68 | 6.67 |
| 04 10-18 s | 2.68 | 2.51 | 2.37 | nan | 3.19 | 2.69 | 2.70 |
| 04 32-40 s | 4.15 | 4.05 | 4.03 | 3.58 | 4.34 | 4.10 | 4.08 |
| 04 42-50 s | 6.63 | 6.16 | 6.36 | 5.24 | 7.28 | 6.67 | 6.48 |
| 05 31-39 s | 5.09 | 4.73 | 4.91 | 4.34 | 5.48 | 5.01 | 5.25 |

## Verdict (conf_at0 speech mean vs base, noise floor = seed spread)

- A2_g25 (guidance 2.5): +0.418 vs base, wins 7/7 -> BEATS seed noise
- A3_s30 (30 steps): -0.001 vs base, wins 4/7 -> within seed noise
- A4_nobr (bridge off): -0.005 vs base, wins 4/7 -> within seed noise
- A1_g15 (guidance 1.5): -0.371 vs base, wins 0/7 -> within seed noise

Notes: A1 (guidance 1.5) on 04 10-18 s has no speech-second SyncNet value (face lost by SyncNet's tracker in that clip), its mean is over 6 windows.
Seed 1247 (production) scored above seeds 7 / 99 on 7/7 windows: the base sits at the top of its own seed distribution.

## Recommendation
- **guidance 2.5** is the only arm beyond seed noise on lip-sync: conf_at0 +0.42 (seed spread 0.24), worst second +0.36, lip-energy r 0.571 -> 0.634,
  better on 7/7 windows (04 10-18 s 2.68 -> 3.19, 04 42-50 s 6.63 -> 7.28). Cost: single-frame lip flash candidates 9.9 -> 17.4 per window (seed spread
  1.4) and jaw jitter 0.94 -> 0.98 of the source. -> Worth a FULL-video A/B on the video side WITH the r3fix sanitizer (which removes generator outlier
  frames) and the video gate's flash review / jaw metric; do not adopt on SyncNet alone.
- 30 steps, bridge off: within seed noise on every metric. Bridge off is ~30 % less LatentSync time per frame (0.26 vs 0.38 s, shared GPU: indicative)
  for no measurable quality change -> a free speed option.
- guidance 1.5: worse sync (-0.37, 0/7) but far fewer flash candidates (3.7) -> not recommended.
- Clips: `outbox/sweep_04_10s_ARMS[_half].mp4`, `outbox/sweep_04_42s_ARMS[_half].mp4` (3x2: base, seed 7, g1.5 / g2.5, 30 steps, bridge off).
