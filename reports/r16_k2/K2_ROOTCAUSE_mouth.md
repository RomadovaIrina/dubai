# K2 root-cause notes: black hole / soft mouth / lip shape (r16 day 3 analysis, 09.11)

**Data:** the delivered 08.11 e2e files (what Max saw), each against its own source on the same 25 fps grid.
- Videos: m03, m18, old03, q01, q02, q03, q04, q05, and long5's dark-haired shot (story 3 = long5 107.8–141.4 s).
- Skipped: q06 (no KeySync) and q01_subs (= q01).
- **Tool:** scripts/keysync/r16/k2_mouth_atlas.py (MediaPipe on ours + source, per frame) on r16-k2.
- **Outputs:** /workspace/dub/outbox/r16_k2/analysis/:
  - K2_MOUTH_ATLAS_ROWS.md: 94 rows in the taxonomy format, for Q's atlas;
  - out_<video>.json: per-frame series + events;
  - old03_arms_sharpness.txt.
- **Generator inputs recorded per event:**
  - the identity frame: e2e = ref 0 = the first frame of each face segment; its opening, lip thickness and teeth share;
  - the dub at that moment: RMS dB + the share of 300–1000 Hz (F1 band) energy, an "open vowel" proxy;
  - ours vs source opening / width.
- **Gate state per segment is NOT in the delivered logs.** Where it matters I argue from controls (golden = no gate; the gate-off arms of 09.11).

Definitions:
- **black** = open > 0.15 and the inner mouth ≥ 40 % near-black (L* < 15) or mean L* < 15, with < 5 % teeth pixels;
- **soft** = (lower lip + chin Laplacian / nose band Laplacian) ours ÷ source < 0.5 over ≥ 10 frames;
- **lip shape** = lip thickness / IOD vs the speaker's median (eaten < 0.75, swollen > 1.35).

## 1. Black hole: 77 events

| video | black share of open frames, ours vs source | events | reading |
|---|---|---|---|
| m03 | 27 % vs 7 % | 12 | generator |
| m18 | 50 % vs 38 % | 11 | generator, plus a dark source |
| q01 | 17 % vs 48 % | 8 | mostly fine, the source is darker |
| old03 (e2e) | 2.2 % vs 0.4 % | 2 | rare. The 66.2 s void Irina saw is in TODAY's chain: frames 1655–1657, 50–61 % near-black; the delivered e2e has it at 65.88–65.92 s |
| q04 | 100 % vs 100 % | 13 | source: a beard + a dark mouth; ours just opens wider |
| q02 | 47 % vs 96 % | 5 | source dark |
| long5 shot | 66 % vs 83 % | 25 | source dark (dim warm light); plus the paste colour lift → navy (see 4) |
| q03 | 0.5 % | 0 | none |
| q05 | 4.9 % | 1 | none |

**Trigger:** on m03 / m18 / q01 / old03 the void appears when all three hold:
1. The generator opens the mouth WIDE: ours 0.17–0.34 vs the source 0.02–0.15 at the same frame.
2. On a loud vowel: dub 64–78 dB, often the onset straight after a short dip (old03 66.20 s: 47–53 dB → 75–78 dB).
3. The identity frame has NO visible teeth: teeth share 0.00 in every e2e ref 0 (m03, m18, old03, q03, q04); 0.02 on q01. KeySync has no teeth exemplar, so a wide opening is filled with a dark cavity.

**What it is NOT:**
- **Not the gate:**
  - golden m18 (K1 clean5, no gate) has the same voids at 4 / 8 / 12 s (K1: black 0.39 / 0.56 / 0.58 vs e2e 0.36 / 0.42 / 0.54);
  - old03 66.2 s is void in both today's renders, which differ only in the reference (A ref 0, B refpick #1518 with teeth).
- **Not the reference alone:**
  - B (teeth ref) still voids at old03 66.2 s;
  - on m18 the teeth ref made the black share WORSE on 09.11 (24 → 30 %).
  - Day 1's crop metric (m03 16 → 5 %) does not hold e2e on every video.
- **Not the paste:** raw KeySync is already near-black (long5 shot, 8.2 s: raw L 2.8, the guard L 1.2). The paste only lifts it to a navy grey (L 7.6, b* ≈ 0) through the colour-match offset (K1 has --cm-shadow).

**Second trigger: an open mouth on SILENCE.**
- m18 0.16–0.76 s (dub 23–24 dB), m18 16.16–16.40 s (38 dB), q02 13.44–14.08 s (31 dB), q02 7.8 s (17 dB, thin lips).
- The generator holds or opens the mouth with no speech. This is the "lips parted in pauses" effect that pause arm b (k2_pause warp) closes; Irina said "гуд, делаем", but it is default OFF in e2e.

**Severity 2 occurrences:**

| video | time |
|---|---|
| m03 | 25.28–25.80 s |
| m18 | 10.00–10.48 s |
| q02 | 13.44–14.08 s (silence) |
| long5 shot | 1.84–2.48, 12.80–13.36, 18.12–18.60 s |
| q04 | 10.56–11.12, 18.84–19.32, 21.60–22.52, 23.56–24.24, 40.08–40.52 s (source-dark) |
| old03, today's chain | 66.20–66.28 s (A and B) |

## 2. Soft mouth: 4 events, all old03

- **old03 whole video:** lower / upper sharpness ours 0.44 vs source 0.65 (0.67 of the source).
- **Events:** 56.04–56.76, 72.96–73.88, 87.16–87.84, 89.32–89.84 s, ratio 0.46–0.49. All in pauses (dub 51 dB, F1 band 0.08–0.12), i.e. where the gate keeps the 2-step student.
- **Cause on old03 = the GENERATOR (verified in crop space):**
  - old03's box is 464 px, so the 512 KeySync render is scaled DOWN 0.91× into the frame, not up;
  - the RAW KeySync crop is already soft: lower / upper sharpness 0.37 vs 0.67 for the source crop at the same 512 scale, i.e. 0.56 of the source before any paste;
  - so it's the 2-step LCM student on a low-res source;
  - the reference does not matter: B 0.66, the same as A 0.72.
- **Cause on m18 = the PASTE (K1, measured on raw → post → final):**
  - raw KeySync ≈ the source (0.87–0.93 at 9–10 s);
  - the final is 0.68–0.78 of raw in every second;
  - m18's box is 720 px, so the 512 render is UPSCALED 1.4× and warped. That is K1's file.
- **Rule:** box > 512 → soft from the paste upscale; box ≤ 512 → soft from the generator.
- **Other videos are not soft:**

| video | sharpness vs source |
|---|---|
| m03 | 1.27 |
| q03 | 1.25 |
| q02 | 1.20 |
| m18 | 0.99 |
| q04 | 0.98 |
| long5 shot | 0.94 |
| q05 | 0.93 |
| q01 | 0.84 |

  K1 finds m18 10–11 s locally soft (12.9 vs 19.8).
- **Measured remedy:**
  - paste-side `--sharp auto` (unsharp 0.60, auto) takes old03 to 0.80 of the source (63–67 s: 0.90);
  - **More steps do NOT sharpen it:** the 8-step-everywhere arm (gate on every segment, 250 GPU-s/vmin) has raw lower / upper sharpness 0.373, identical to 2 steps (0.374), both 0.56 of the source crop. The softness is the model's output detail on a low-res input, not the step count, so neither the gate nor the teacher steps can fix it; only a paste-side sharpen (or a better generator) can.

## 3. Lip shape: 13 events

- **Swollen / puckered (11):**
  - m18 6.16–6.36, 7.00–7.52, 10.88–11.20 s;
  - q01 1.08–1.44, 21.08–21.56, 28.56–28.84, 43.92–44.16 s;
  - long5 shot 20.72–21.52 s;
  - old03 86.6 s (via its black row).
  - Lips 1.35–1.46× the speaker's median thickness (by IOD) while the mouth is NARROWER: width / IOD 0.49–0.54 vs the source 0.56–0.62.
  - Always on strong open / rounded vowels (F1 band 0.71–0.85).
  - **Cause:** generator overshoot on rounded vowels (KeySync's audio → mouth mapping). Not the reference: ref 0 lips are 0.82–1.08× the median.
  - q05 12.36–13.16 s is the cup (landmarks on the cup): occlusion class, not lip shape.
- **Eaten (2):**
  - q02 7.80–8.08 s: thin lips on silence (17 dB);
  - m18 9.5–10.5 s: Irina's "губы съелись". The 08.11 file has a NEW TTS dub (uncorrelated with the accepted one, r = −0.003); its spread vowel at 0:10 + ref 0 with thin lips (upper 0.141 / lower 0.172 vs median 0.151 / 0.185) give thin, rolled-in lips. With the accepted dub, every arm (gate on / off / hi3) matches golden (0.166 / 0.232 vs 0.169 / 0.236). A neutral-lips ref (frame 81) restores most of it (lower 0.190 → 0.205, opening 0.116 → 0.157) at −0.25 speech@0. This is pending Irina.
- **Note:** an earlier width-normalised lip measure flagged m03 / q03 "eaten". Those were artefacts of wide spread vowels, removed by the IOD normalisation.

## 4. m18: the mouth problems Max saw (with K1)

Candidates (Max gave no timestamps), ranked:

| t (s) | class | numbers | cause |
|---|---|---|---|
| 10.0–10.5 | black hole + lip shape | ours open 0.20 vs src 0.03, void; lips thin 9.5–10.5 | generator: the new TTS dub's vowel + a teeth-free / thin-lipped ref 0 |
| 6.2–7.5 | lip shape (swollen / pucker) | lips 1.36–1.46×, width 0.49 vs 0.60; K1: at 7 s the mouth stays shut (gap 0.033 vs golden 0.108) | generator rounded-vowel overshoot |
| 10.9–11.2 | swollen + black + soft | lips 1.41×; K1 sharp 12.9 vs 19.8 | generator |
| 16.2–16.4 | black hole on near-silence | open 0.17 at 38 dB | generator: mouth open in a pause (pause arm b not on) |
| 0.2–0.8 | black hole on silence | 23–24 dB | same |
| 3.2, 8.3, 13.5–14.1 | black hole | open 0.19–0.24 vs 0.03–0.13 | generator, a teeth-free ref; partly shared with golden (K1: 4 / 8 / 12 s) |
| 18.3–20.0 | tail | | T: the zero-slot chunk + the paste writing N−1 frames |

Paste-side (K1, measured raw → post → final on today's default render, r16-k2:/workspace/dub/k1/ana/m18/m18_raw_post_final.json):
- **Lip colour is NOT the paste.** Lip-ring a*: raw 21–24 → post 20–23 → final 18.5–20.6 vs the source 16.6–19.0. The colour match pulls the red DOWN, so the "painted / swollen" look is the generator.
- **Black share** (raw → post → final): 10 s 0.50 → 0.64 → 0.30; 16 s 0.44 → 0.56 → 0.30; 8 s 0.69 → 0.77 → 0.48. The generator makes it, the post chain (tsmooth / snap / guard) deepens it by 0.08–0.14, the paste's colour lift reduces it.
- **The post chain also darkens the lips** by ~2.2 L* (K1 to check in ks_snap / guard).
- **The m18 softness at 10–11 s is the paste upscale,** not the generator (see 2).
- **Seam wobble is worst exactly at the swollen-lip seconds:** 6 s 3.92, 7 s 4.17 (median 2.18). The generated chin moves a lot there (dchin 0.025).

## 5. What the analysis says about fixes (not done; for the coordinator / Irina to pick)
1. **Black hole:** the cause is a teeth-free identity frame + wide openings. Options, by evidence:
   - pause-closure (arm b) for the silence voids;
   - a per-video "ref with teeth only when ref 0 has none AND the source shows teeth" rule (q04 / q02 / long5 have dark sources: leave them alone);
   - K1's --cm-shadow for the navy tint;
   - the loud-onset voids (old03 66.2 s) survive the teeth ref: generator-limited.
2. **Soft mouth (old03):** --sharp auto (0.67 → 0.80 of the source) is cheap; the 8-step answer is pending.
3. **Lip shape:** the m18 0:10 neutral-ref option (−0.25 speech@0) is pending Irina. The puckers are a generator / audio-mapping property: no cheap lever measured.
