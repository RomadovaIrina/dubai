# Track B review clips (Server B)

| file | what | timestamp | why picked | metric values (mouth_jitter_hr, vs source) |
|---|---|---|---|---|
| `p2-twitch-03-15s_orig_vs_cand5_speed1-1.mp4` / `_speed1-2.mp4` | ORIGINAL \| cand5 baseline, face crop, real speed and half speed, no audio | 03 15.0-17.4 s (frames 375-435) | visible twitch: frames 393 -> 394 -> 395 the LatentSync mouth snaps from open to pressed shut and the lower lip skews for one frame; window boundary at 396 | 03 whole video: LS window-boundary mouth jump 1.37x the inner frames (source 0.97), LS-core mouth texture residual 1.32 |
| `p2-decomp-03-42s-60s_stages_4panel.mp4` | ORIGINAL \| LatentSync raw \| LS + CodeFormer \| cand5 final, same frames, with the run's audio | 03 42.0-60.2 s (frames 1050-1506) | decomposition window: LS->orig switch 44.56 s, orig->LS 46.04 s, continuous speech + the largest head turn of 03 (46-50 s), LS->orig 56.24 s, orig->LS 58.96 s | LS-core mouth: texture residual LS raw 1.05 / +CodeFormer 1.43 / final 1.43; sharpness 0.61 / 0.99 / 0.99; window-boundary jump (d1) 1.29 / 1.21 / 1.21 (source 0.99) |
| `p2-sbs-03-42s-60s_orig_cand5_C2_C1C2gfpgan.mp4` | ORIGINAL \| BASELINE cand5 \| C2 (window-boundary interpolation, no restorer) \| C1+C2 with GFPGAN v1.4 detail (alpha 0.5, gain 1.3), with the run's audio | 03 42.0-60.2 s | Phase 2 candidates on the decomposition window | LS-core mouth texture resid / sharpness / boundary jump / SyncNet dub-only: cand5 1.43 / 0.99 / 1.21 / 6.12; C2 1.01 / 0.60 / 0.59 / 6.31; C1+C2 GFPGAN 1.01 / 0.71 / 0.59 / 6.00 |
| `p2-sbs-03-46s-52s_mouthcrop_halfspeed.mp4` | same four, face crop, half speed, no audio | 03 46.0-52.0 s | head turn + continuous speech; look at lips / teeth shimmer (cand5) vs softer but steady lower face (C2, C1+C2) | as above |
| `p3-03-h24-25s-frozenmouth_orig_cand5_clean0_opt1_opt2_{full,zoom,zoom_halfspeed}.mp4` | ORIGINAL \| CAND5 \| CLEAN0 \| OPT1 still lower face \| OPT2 original pass-through (both on the F2 source-fps base: C2 + skin texture, flow retiming); all panels retimed to 30 fps by time | 03 22.5-27.5 s | human mark 24-25 s: dub "the start" fills 0.5 s of a 1.1 s source burst, then dub silent while the source talks | see PHASE3.md table |
| `p3-03-h35-36s-frozenmouth_orig_cand5_clean0_opt1_opt2_{full,zoom,zoom_halfspeed}.mp4` | same panels | 03 33.5-38.5 s | human mark 35-36 s: dub silent 34.22-35.90 s while the source talks | OPT2 frozen-mouth index at 35 s 0.99 (rank 74 / 90) |
| `p4-03-h24-25s-orig_cand5_clean0_MAIN_opt1_opt2_{full,zoom,zoom_halfspeed}.mp4` | ORIGINAL \| CAND5 \| CLEAN0 \| **MAIN** (source fps, nearest + rigid retiming, C2, source skin + lip texture, sharpen 1.5 + lips 1.5, 12 px edge) \| MAIN + OPT1 still \| MAIN + OPT2 orig | 03 22.5-28.5 s | human mark 24-25 s | PHASE4.md |
| `p4-03-h35-36s-orig_cand5_clean0_MAIN_opt1_opt2_{full,zoom,zoom_halfspeed}.mp4` | same six panels | 03 33.5-39.5 s | human mark 35-36 s | PHASE4.md |
| `p4-03-w42-48s-orig_cand5_clean0_MAIN_opt1_opt2_{full,zoom,zoom_halfspeed}.mp4` | same six panels | 03 42.0-48.0 s | LS -> original 44.56 s, original -> LS 46.04 s, head turn 46-48 s | PHASE4.md |

What to look for: (1) in "LatentSync raw" the mouth is soft but its texture is steady; small jumps every ~0.64 s (window boundaries);
(2) in "LS + CodeFormer" / cand5 the lips and teeth are sharp but their texture shimmers from frame to frame.
(3) C2 / C1+C2: no jump every 0.64 s and no shimmer, but the lower face (chin stubble) inside the paste ellipse is softer than the source.
(4) p4 clips: MAIN vs CLEAN0 -> sharper lips / teeth / stubble, no 0.64 s seam jump, source frame rate (no 1-in-6 judder); OPT1 / OPT2 differ only in dub-silent stretches.

## Full-length copies (no new renders; copies of the evaluated outputs)
| folder | content |
|---|---|
| `fps_variants_03/` | 03 source-fps variants: F0 (clean0 composite at source fps, 12 px edge), F1 (+ C2), F2 (+ skin texture), F3 (+ lip texture), F0n / F0r (nearest / nearest + rigid retiming) — PHASE3.md table |
| `sharpening_03/` | 03 sharpening finalists: L4 (global unsharp 1.5), Q1 (1.5 + 1.5 on the lip contour) — flow retiming |
| `frozen_mouth_03/` | 03 OPT1 still lower face / OPT2 original pass-through on the F2 base |
| `phase4/` | MAIN on 01 / 02 / 03 / 04 / 05 + 03 MAIN+OPT1 / MAIN+OPT2 — PHASE4.md |
