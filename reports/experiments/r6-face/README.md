# r6 F: face twitch (blocker #0)

## Reference scores (r5_twitch.py, coordinator's uploads, 27.10)
See `twitch_refs.md` (twitch_sum / n_events / top20_mean / p99 per video; v4, r4_video_v1, m3).
- twitch_sum: v4 01 8.9 / 02 0 / 03 52.8 / 04 47.3 / 05 33.0; r4_video_v1 11.5 / 0 / 49.6 / 56.6 / 23.1; m3 (ref) - / - / 45.0 / 45.5 / 36.4;
  m3 local re-render 01 11.5 / 02 0.
- 02 has no face (room walkthrough), 01 a face in 260 of 2542 frames: twitch is decided on 03 / 04 / 05.
- m3 fails "<= v4" on 05 (36.4 vs 33.0) only.

## What the top events are (contact sheets + 9-frame grids src / v4 / r4_video_v1 / m3)
- **05 23.67-23.80 s (m3 8.2, r4_video_v1 10.6, v4 clean):** the whole pasted lower face is a blurred, mis-coloured square patch with no
  mouth for 3-4 frames (looks like a corrupted / mis-registered generator frame, or the sanitize layer on one), then snaps back. THE
  visible twitch; most likely the user's "23-24 s". It is in the r4-era renders (r4_video_v1 and m3), not in v4 -> the r4 video stack.
- **05 34.7 s (m3 11.0):** the known raw-generator lip flash (33.7-36 s), item 2 of the brief.
- **04 27.8 / 18.3 s and the other 04 "lips" events (head speed >= 2 iod/s):** mouth opening / closing that the source also does; not a
  visible jump at the 9-frame grid. The lips channel over-fires on articulation during fast head motion (same in v4).
- 03: all "lips" <= 2.2 at low head speed; nothing stands out on the sheet.
Conclusion: the twitches that matter are single / few-frame LAYER failures (a corrupt paste, a flash), not the LS|LS joints or the
LS <-> source crossfades the r5 flags address. Next: find the failing frames in m3's raw LatentSync layer (local 05 render) and add a
generic per-frame layer sanity pass in fps_composite (blur / colour / mouth-less patch vs the neighbours -> flow interpolation from the
nearest good frames, source if the run is long).

## Root cause (local m3 re-render of 05, raw layer + composite log, 27.10 10:30)
- The raw LatentSync layer is clean at 23.7 s (master 588-598) and at 7.8 s (master 190-200).
- `fps_composite --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8` flagged 137 and REPLACED 131 of 1323 layer frames on 05
  (03: 277 / 2496, 04: 110 / 1359): the flag (lip-box Laplacian or contrast < 0.83 x the neighbours' median) also trips on articulation.
- The replaced frames at 23.7 s are master 592-593 (+ 587-588), at 7.8 s master 193-195: EXACTLY the smeared patches. Texture mode keeps the
  frame's own low frequencies and takes the high frequencies from a flow interpolation of the neighbours; on a small (iod 45) moving face
  that interpolation is misaligned -> a mouth-less smear for 2-4 frames = the face twitch the user sees.
- Fix (generic, default on): `--sanitize-guard on`: a flagged frame is replaced only if the lip-box high frequencies go UP (a real flash has
  no lip line, the fix restores it; a misaligned fix lowers them). Arms on m3's raw layers: sg (guard) vs s0 (sanitizer off) vs m3.

## Arms on m3's raw layers (composite only), twitch_sum (lower = better); v4 reference 03 52.8 / 04 47.3 / 05 33.0
| arm | 03 | 04 | 05 | note |
|---|---|---|---|---|
| m3 (local) | 43.7 | 44.4 | 36.1 | sanitizer replaces 277 / 110 / 131 frames |
| s0 = sanitizer off | 64.4 | 40.3 | 15.1 | 03's real flashes come back (6 events >= 2) |
| sg = sharpness guard (hf up) | - | 40.9 | 27.9 | 23.8 s smear still through: a misaligned warp can add edge energy |
| sa = skin-ring alignment NCC >= 0.8 | 43.7 | 46.7 | **9.1** | guard kept 0 / 9 / 9 of 319 / 115 / 137 flagged; both 05 smears gone (grid checked) |
| m4 = sa + --mouth-region 0.38 | pending | 46.1 | 8.2 | the m4 candidate; 04's top events = abrupt generator lip-shape changes (22.4 s window joint, 6.9 s), not paste jumps |
Probe (`/workspace/dub/artifacts/r6f/san_probe.py`): on 05 every flagged frame has interp confidence 0 (small face, flow unreliable); the
smear frames have ring NCC -0.13..0.14, 03's real flashes 0.95-0.98, 05 p10 0.83.

## m4 per video (twitch_sum; v4 in brackets): 03 42.5 (52.8), 04 46.1 (47.3), 05 8.2 (33.0); 01 / 02 compositing at 12:55
## Why the two new defaults should hold on unseen videos (user rule: the E2E works on ANY video)
- `--sanitize-guard on` / `--sanitize-align 0.8`: a normalised grey cross-correlation on the skin ring around the lip box (ring sized by the
  mouth width): scale-, resolution- and brightness-free. It tests the physics of a flash (lips change, the skin ring does not) against the
  failure (a misaligned warp moves the skin). 0.8 is not fitted: real flashes 0.95-0.98 (03, 277 frames), smears -0.13..0.14 (05), the 10th
  percentile of all 05 fixes 0.83. An undefined NCC (textureless ring) = not aligned = the original frame stays (safe side). Depends only on
  the generator landmarks the sanitizer already needs.
- `--mouth-region 0.38`: iod units (scale-free), the hull of the source AND layer lips (an opening jaw widens it); depends on MediaPipe
  landmarks like the whole stack. Risk: a chin moving > 0.38 iod below the lips hull; the r4 sweep kept the jaw drop (SyncNet within +-0.2).
- Both are composite-side: no generator / model change (G: keep guidance 2.0, bridge on).

## m4 WATCH (27.10 13:35): gate vs m3 PASS on all 5 (`gate_m4_vs_m3.md`, `r6_m4_WATCH.md`); outbox: <id>_r6_m4_WATCH.mp4 + _sbs (M3 | M4) + r6_m4_twitch/

## m5 = m4 face stack + audio r6_audio_v1 (one command r6_e2e.sh), 27.10 14:20
twitch_sum m5 / m4 / v4: 01 12.1 / 10.2 / 8.9; 02 0 / 0 / 0; 03 47.7 / 42.5 / 52.8; 04 50.5 / 46.1 / 47.3; 05 19.0 / 8.2 / 33.0.
The audio of record moves the LS segments and the generator's lip motion; the extra events are mouth-shape changes at the new speech
boundaries (05 0:26.9 / 0:20.2 grid-checked: articulation, no paste smear). Gate vs m4 + WATCH: `r6_m5_WATCH.md`, `gate_m5_vs_m4.md` (03 / 05 / 04
first; 01 / 02 re-emitted if the server lived long enough). Speed: 01 191 s/video-min, 03 748, 05 ~700 (over the 300 s norm, as m3).
