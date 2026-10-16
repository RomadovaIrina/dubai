# Track B review clips (Server B)

| file | what | timestamp | why picked | metric values (mouth_jitter_hr, vs source) |
|---|---|---|---|---|
| `p2-twitch-03-15s_orig_vs_cand5_speed1-1.mp4` / `_speed1-2.mp4` | ORIGINAL \| cand5 baseline, face crop, real speed and half speed, no audio | 03 15.0-17.4 s (frames 375-435) | visible twitch: frames 393 -> 394 -> 395 the LatentSync mouth snaps from open to pressed shut and the lower lip skews for one frame; window boundary at 396 | 03 whole video: LS window-boundary mouth jump 1.37x the inner frames (source 0.97), LS-core mouth texture residual 1.32 |
| `p2-decomp-03-42s-60s_stages_4panel.mp4` | ORIGINAL \| LatentSync raw \| LS + CodeFormer \| cand5 final, same frames, with the run's audio | 03 42.0-60.2 s (frames 1050-1506) | decomposition window: LS->orig switch 44.56 s, orig->LS 46.04 s, continuous speech + the largest head turn of 03 (46-50 s), LS->orig 56.24 s, orig->LS 58.96 s | LS-core mouth: texture residual LS raw 1.05 / +CodeFormer 1.43 / final 1.43; sharpness 0.61 / 0.99 / 0.99; window-boundary jump (d1) 1.29 / 1.21 / 1.21 (source 0.99) |
| `p2-sbs-03-42s-60s_orig_cand5_C2_C1C2gfpgan.mp4` | ORIGINAL \| BASELINE cand5 \| C2 (window-boundary interpolation, no restorer) \| C1+C2 with GFPGAN v1.4 detail (alpha 0.5, gain 1.3), with the run's audio | 03 42.0-60.2 s | Phase 2 candidates on the decomposition window | LS-core mouth texture resid / sharpness / boundary jump / SyncNet dub-only: cand5 1.43 / 0.99 / 1.21 / 6.12; C2 1.01 / 0.60 / 0.59 / 6.31; C1+C2 GFPGAN 1.01 / 0.71 / 0.59 / 6.00 |
| `p2-sbs-03-46s-52s_mouthcrop_halfspeed.mp4` | same four, face crop, half speed, no audio | 03 46.0-52.0 s | head turn + continuous speech; look at lips / teeth shimmer (cand5) vs softer but steady lower face (C2, C1+C2) | as above |

What to look for: (1) in "LatentSync raw" the mouth is soft but its texture is steady; small jumps every ~0.64 s (window boundaries);
(2) in "LS + CodeFormer" / cand5 the lips and teeth are sharp but their texture shimmers from frame to frame.
(3) C2 / C1+C2: no jump every 0.64 s and no shimmer, but the lower face (chin stubble) inside the paste ellipse is softer than the source.
