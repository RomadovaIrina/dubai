# HANDOFF K2, 08.11 (r16 day 2)

- **Branch:** experiment/r16-mouth on r16-k2:/workspace/dub/k2wt, on top of int 66d6041.
- **Bundle:** outbox/r16_k2/k2-r16-mouth.bundle (66d6041..HEAD).
- **NOT pushed:** GitHub rejects this box's deploy key (`Permission denied (publickey)`). Irina has to add key B.

Commits today:

| commit | what |
|---|---|
| 3d66d9e | alpha-paste (ks_dissolve.py); ks_shots keeps KS_REF=0 |
| 9187ef1 | `ks_dissolve check` (linear-crossfade test + flag); Irina's acceptance quote |
| 688e28a | ks_shots: split a faceless gap only if the picture changes (q06: 18 pieces → 3) |
| 4c9874b | ks_dissolve unmix streams (Q audit P3, RAM) |

## 1. q03 alpha-paste: ACCEPTED by Irina
- Quote: "альфа пейст не плавает не дергается" (it doesn't float, doesn't jerk).
- **Final:** outbox/r16_k2/q03/q03_multishot_alphapaste.mp4. Shot B is pasted with shift +1 on its own run. speech@0 5.87 vs the hard switch 6.15; the difference is inside shot B's 6 s window, within noise.
- **Auto-on rule (`check`):** a linear crossfade is one with worst residual < 0.10 × mean|B−T|, alpha monotone, span ≥ 0.8. Otherwise the hard switch stays and a `KS WARN` flag is printed.
  - q03 22.76 s: 0.029, ok;
  - q03 30.4 s: 0.031, ok;
  - q03 37.72 s (fresco pan): 0.285, flagged.
- Wiring for T is in the docstring. T has the bundle.
- **P3 RAM: fixed.** unmix now holds only the T / B frames and the dissolve window, and streams the rest (q03 shot B peak 408 MB; output md5 identical). assemble already streamed.

## 2. Black hole (e2e ks_face, delay 0, shift 0; table + method: outbox/r16_k2/bh/RESULTS.md)

| video | speech@0 plain / r0g / refpick | black share src / plain / r0g / refpick | open frames plain / r0g / refpick |
|---|---|---|---|
| q01 | 6.85 / 6.91 / 7.31 | 2.2 / 2.0 / 2.5 / 2.2 % | 661 / 653 / 312 |
| q02 | (unscorable) | 82 / 52 / 36 / 0 % | 321 / 475 / 98 |
| q03 | 5.49 / 5.43 / 6.16 | 0 / 0.7 / 0.7 / 0.4 % | 138 / 140 / 252 |
| q04 | 5.19 / 5.23 / 5.47 | 77 / 85 / 82 / 97 % | 75 / 76 / 68 |
| m18 | 8.15 / 8.09 / 7.20 | 6.7 / 24 / 21 / 30 % | 125 / 164 / 103 |

Arms: plain = ref 0, no gate; r0g = ref 0 + gate (the default); refpick = auto reference + gate.

- **The gate (r0g) is slightly less black than plain and opens the mouth as much or more.** It agrees with Irina's choice.
- **Refpick causes "the mouth opens less"** (q01 / q02) and is worse on m18 / q04. Keep it OFF, as already decided.
- **The real black hole is m18:** 21–24 % vs the source's 7 %. q02 and q04 have dark mouths in the source too.
- **Failures:**
  - q05 crashes in ks_tsmooth on NaN landmarks (audit 10; T has it);
  - q06 is correctly kept as source (no-face rule).

## 3. A cheaper gate (m03 + m18, ref 0, delay 0, shift 0)
- GPU cost proxy = UNet chunk calls. Wall seconds are noisy because of contention.
- Eye clips: outbox/r16_k2/gate/{m18_..._2-6s, m03_..._23-27s}_half_speed.mp4, showing gate off | hi8 (current) | hi3 | source.

| arm | m18 speech@0 | m03 speech@0 | UNet calls m18 / m03 |
|---|---|---|---|
| off | 8.15 | 8.48 | n/a |
| hi8 (current) | 8.14 | 8.38 | 104 / 192 |
| hi4 | 7.99 | 8.37 | 68 / 110 |
| **hi3** | **8.13** | **8.38** | **59 / 111** |
| thr .18 cap .15 | 7.98 | 8.47 | 74 / 150 |

- **Recommendation: hi=3.** Same speech@0 as hi8 at ~45 % fewer UNet calls, well inside the +20 s per video-minute budget. Needs Irina's eye on the clips that it keeps the open-mouth gain.
- **In this chain the gate costs only −0.01 (m18) and −0.10 (m03), not −0.37 / −0.28.**

## 4. Delay: T's +40 ms vs my 0 ms
- T's head (b372dae) ks_face.sh, run directly on m18 with the raw story video + paste_inputs audio at delay 0 / gate off, gives speech@0 8.13 at offset 0. My chain gives 8.15 at offset 0. T's e2e at delay 0 gives 5.43 at offset −1.
- **So the extra frame is UPSTREAM of ks_face, in what e2e hands it (the master or the audio dir), or in the metric (eval_final vs sync_perframe).** It is not in ks_face.
- **Also:** the probe's raw-crop SyncNet says −1 while the pasted full frame scores 0. That is why the probe asks for +40.
- **T should check:** do the e2e job's master_25fps / dubbed_16k start on the same frame / sample as the raw story / paste_inputs?
- **Side finding:** T-head paste wrote 497 frames for a 501-frame master. That is the `-shortest` truncation (audit item 7).

## Needed from Irina
1. **Gate clips:** is hi3 as good as hi8 on the open mouth? If yes, make it the default (KS_GATE=thr=0.12,cap=0.25,hi=3,lo=2).
2. **Black hole on m18** (21 % vs source 7 %): refpick is off, and on day 1 "(a) bank" (mouth-interior fill) was rejected as a teeth grille. Options: try an upper-teeth-only interior mock (stills first), wait for L's B2p, or accept.
3. **Delay rule** (Q's proposal "0 unless conf − conf_at0 > 2"): this is her call. See point 4 first: the 1-frame difference may be a pipeline offset, not the content.

## Next steps
- Wire the alpha-paste into e2e (T).
- hi3 as the default if Irina agrees.
- Find the e2e 1-frame offset (T + K2).
- m18 black hole after Irina picks an approach.
- q05 tsmooth fix (T).
