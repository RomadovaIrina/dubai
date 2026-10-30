# r9fx: full set, m18, twitch, flashes (30.10.2026, 18:10 UTC)

My messages to the coordinator session are paused by the app (too many session-to-session messages), so the numbers are here.

## Delivered
- **r9fx set** (shipped generator + fix = layer repair v2 + inner-mouth CodeFormer detail 0.4, composite only), both mixes, `<id>_DabAI_r9fx_{demucs,spleeter}.mp4`:
  - r9-a `/workspace/dub/outbox/r9fx/`: m03 m15 m17 m18 m19 m20 m21 (no face, = r9) + m6185 (IMG_6185);
  - r9-f `/workspace/dub/outbox/r9fx/`: m01 m04.
- **IMG_6185 ref fixed (lead 0.5) + fix**, on the same dub: r9-a `/workspace/dub/outbox/r9_e2e/m6185_DabAI_r9fixedfx_{demucs,spleeter}.mp4`.
  - 2-panel WATCH (r9 + fix | fixed + fix): `/workspace/dub/outbox/r9_e2e/watch_mouth/m6185_WATCH_r9fix_vs_fixedfix.mp4`.
- **WATCH r9 | r9fx** (demucs mix), r9-a `/workspace/dub/outbox/r9fx/watch/`:
  - m18;
  - m03;
  - m15 (the worst twitch video).

## Twitch (round-6 face-twitch detector vs the raw source; it had never run on r9: missing source landmarks)
| video | r9 twitch sum (events, strong ≥ 2) | r9fx | worst r9 moments (s) |
|---|---|---|---|
| **m15** | **126.8 (57, 22)** | **43.1 (24, 2)** | 8.0, 12.7 (score 15), 15.4, 20.4, 23.3, 23.8, 24.2, 28.3 |
| m17 | 45.7 (27, 8) | 46.6 (29, 5) | 4.4, 4.7, 12.2, 16.1, 20.4, 22.5, 25.0, 37.9 |
| m03 | 35.7 (24, 3) | 23.7 (18, 1) | |
| m04 | 29.8 (20, 1) | 28.5 (20, 1) | |
| m20 | 27.1 (20, 1) | 29.3 (23, 0) | |
| m18 | 11.5 (9, 0) | 9.2 (8, 0) | |
| m19 | 10.8 (9, 0) | 11.4 (9, 0) | |
| m01 | 6.1 (5, 0) | (pending) | |

- Almost all twitch events are on the lips channel, not the skin.
- m15 is most likely the "she twitches in various places" video: the fix cuts its strong events 22 → 2.
- m17's twitches (phone partly over the face) are untouched by the fix.
- Sheets: `outbox/r9fx/twitch_sheets/`.

## Mouth flashes (the gate detector r3_flash: 1-frame spikes of texture / brightness / colour in the lip box)
| video | r9 flash frames (strong) | r9fx | main kind |
|---|---|---|---|
| m18 | 56 (7) | 63 (8) | **texture** (42 / 55) |
| m03 | 85 (26) | 77 (19) | brightness (58 / 49) |
| m20 | 68 (17) | 63 (14) | brightness |
| m17 | 59 (6) | 49 (7) | brightness |
| m04 | 58 (26) | 58 (28) | brightness |
| m15 | 52 (10) | 45 (12) | brightness / colour |
| m19 | 49 (4) | 41 (3) | texture / brightness |

## m18 ("a total mess, explosions in the mouth")
- **Shape artefacts are few.** r9 → r9fx: stick / unstick 3 → 2, amplitude 1 → 0, slide 2 → 1.
- **The "explosions" are TEXTURE flicker of the mouth detail**, which the fix does not remove. At 30 fps, where the layers line up frame for frame with the source, the texture flash frames are:

  | layer | texture flash frames |
  |---|---|
  | raw LatentSync | 10 |
  | + CodeFormer | 25 |
  | + detail transfer + layer SR | 120 |
  | final (the composite blends the source back) | 45 |

- **Cause:** CodeFormer's detail changes from frame to frame, and the layer SR amplifies it. On a dim, low-texture face with glasses, that detail is most of the visible texture.
- **Not the cause:** the 25 → 30 fps retime. With LatentSync at 30 fps (no retime) m18 still has 45.
- **Test:** detail at half strength + ±2-frame smoothing gives texture 55 → 36, but strong flashes 8 → 12. Partial only.
- **Next:** a temporal stabiliser for the detail / SR layer (flow-aligned median over ±2–3 frames), or no layer SR on low-texture faces; verify on m18 + m03.

## "The mouth slides off" in the shift16 / fixed panels
- **Measure:** the gate's chin-wave row (low-frequency jaw / chin motion that is not in her real head motion) is 35–75 % higher with the non-current references:

  | video | r9 + fix | shift16 + fix | fixed + fix |
  |---|---|---|---|
  | m03 | 3.9 | 5.3 | 5.4 |
  | m19 | 3.3 | 4.6 | 5.5 |
  | m15 | 4.1 | 5.1 | 5.5 |

- **Flat:** lateral lip offset and shear.
- **Reading:** the generated lower face half-follows the reference clip's pose, which round 3 described. The lip-colour slide detector only sees sideways lip jumps.
