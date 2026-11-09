# K2 day 3 (09.11): m18 0:10 lips, long5 blue mouth

Box r16-k2. KeySync installed from the internet, TRT rebuilt (kf 58.1 dB, interp 58.4 dB), LoRA uploaded from the laptop's C:\repos\chkpts (md5 verified). All renders: T's int head 2d53197, ks_face defaults (probe delay; it chose +40 ms everywhere) unless stated.

## 1. m18 0:10, "the lips got eaten": cause = the NEW dub, not the gate, not ref 0
- **Irina's file vs golden:**
  - Irina's e2e (full_m18, pipeline 286b0ec) is a full raw-source job, so it has a freshly generated TTS dub. Golden (K1 clean5) used the accepted dub.
  - The two audio tracks are uncorrelated (r = −0.003), so at 0:10 they say different words.
- **Lip thickness / mouth width on open frames at 9.5–10.5 s** (upper lip 0–13, lower lip 14–17, both relative to mouth width):

| render | audio | upper | lower | inner opening | speech@0 |
|---|---|---|---|---|---|
| golden (clean5, no gate, ref 0) | accepted | 0.169 | 0.236 | 0.103 | 7.96 |
| A: int default (ref 0 + gate hi8) | accepted | 0.166 | 0.232 | 0.094 | 7.86 |
| B: gate off | accepted | 0.171 | 0.241 | 0.098 | 8.13 |
| C: gate hi3 | accepted | 0.162 | 0.229 | 0.101 | 7.89 |
| Irina's e2e (delivered) | new dub | 0.139 | 0.187 | 0.124 | |
| D: ref 0 + gate | new dub (e2e mix) | 0.148 | 0.190 | 0.116 | 7.92 |
| E: ref 0, gate off | new dub | 0.141 | 0.179 | 0.130 | |
| F: neutral ref 81 + gate | new dub | 0.152 | 0.205 | 0.157 | 7.67 |
| G: neutral ref 81 + gate | accepted | 0.182 | 0.250 | 0.073 | 7.61 |

- **With the accepted dub,** every arm has golden-like lips at 0:10, gate or not. With the new dub, every ref-0 arm has the thin lips, gate or not.
- **The new dub at 0:10 is a spread vowel.** The mouth is wider (width/IOD 0.55 vs 0.51) and KeySync stretches the lips thin.
- **Ref 0 (m18 frame 0) has thinner lips than her median** (upper 0.141 / lower 0.172 vs 0.151 / 0.185).
- **A neutral reference** (frame 81: median mouth width, slightly parted, fuller lips):
  - on the new dub: opening 0.116 → 0.157, outer lip height 0.478 → 0.531, lower lip 0.190 → 0.205;
  - on the accepted dub it keeps the golden look;
  - but speech@0 drops by ~0.25 on both audios (the same kind of trade-off that took refpick off).
- **Irina's eye clip:** outbox/r16_k2/day3/m18_lips_e2edub_delivered_ref0_ref81_source_8-12s_half_speed.mp4 (delivered e2e | ref 0 now | ref 81 neutral | source).
- **Options:**
  - (a) accept: it's this dub's vowel, and the accepted look is intact;
  - (b) a neutral-lips reference rule (cheap: pick by lip thickness + median width, no teeth / smile terms) at about −0.25 speech@0;
  - (c) a lip-thickness floor in the paste (K1).

## 2. long5 1:56 blue mouth: cause = the paste colour match (sent to K1)
- **Re-rendered the dark-haired shot** (story-1774250573471-666527943.MP4 = long5 107.8–141.4 s). It reproduces Irina's frame.
- **Darkest 25 % of mouth pixels at 8.2 s (= 1:56):**

| stage | L | b* | BGR |
|---|---|---|---|
| raw KeySync | 2.8 | 4.4 | (0, 2, 28) |
| after the post chain (guard) | 1.2 | 2.0 | (0, 0, 19) |
| after ks_plain | 7.6 | 0.6 | (23, 20, 32) |
| Irina's e2e | 8.4 | 1.5 | (22, 21, 34) |

- **Mechanism:** this shot's global colour match is Lab gain [0.964, 0.929, 0.900], offset [+7.53, +0.16, −2.07]. It lifts the black floor into a neutral grey, and next to the warm skin in a yellow-lit scene that grey reads as navy.
- K1 has added `--cm-shadow L0:L1` (fade the correction out by raw L) and `--cm-lonly`, and is testing 10:30 on this shot.

## 3. The tail (m18 0:19)
- In the last 1.2 s, sampled every 5th frame, neither the delivered e2e nor my int re-run shows the broken face Irina saw, and the gate on / off arms look the same.
- Nothing points at the gate or window code yet. If T finds the cause is in the keyframe scheduler, I'll look.

## Files (r16-k2:/workspace/dub/k2/)
- m18/: arms A–G (*.mp4, *.npz), lips.py, open.py, refs.jpg, strip10.jpg;
- s3/: raw / guard / final / e2e cut, mouthlab.py, cmp_8.2.jpg, cmp_18.2.jpg.
