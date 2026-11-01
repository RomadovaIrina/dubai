# r11 SYNC: composite-stage SyncNet fixes on the LatentSync r9 + fix default (01.11.2026)

- **Branch:** `r11-composite-sync`, worktree `/workspace/dub/r11_sync` on r11-q (local commits only; the push has no deploy key yet).
- **Scope:** every lever is a new flag, default off. Production output is unchanged.
- **Inputs:** the r9 + fix layers / caches of m18, m03, old03, old05.
  - old03 / old05 had no r9 + fix baseline on r11-q. I generated them with `scripts/r11s/base_old.sh` (E2E_FX=1, as `scripts/r10/base.sh` without the stab arm), plus their gate bases `r10/gate/old0{3,5}base`.
- **Gate:** the r9-speed-face rule, `scripts/r10/gate.sh` + `gate_cmp.py` vs `<id>base`.
- **Repro:** `scripts/r11s/arm.sh` reproduces the gate SyncNet. m18 7.757 vs 7.743, m03 7.655 vs 7.651, old03 7.188 vs 7.198. Noise is about ±0.02.

## User verdict (01.11, m03 + m18 clips: r9+fix vs k5)
- Quote: "да, лучше, но всё ещё не идеально. но реально лучше. артефактов как будто тоже поменьше".
- **k5 is ACCEPTED as the direction**, including the m03 closures that the flash counter flags (the eye reads them as fine).
- The eye overrides the m03 flash count; the old03 / old05 gate rows below are still open.

## Where the ~1.1 goes
| m18 / m03 | SyncNet |
|---|---|
| r9 + fix final | 7.74 / 7.65 |
| unrepaired layer, no fallback (same composite) | 8.22 / 8.39 |
| repaired layer (rep2), no fallback | 7.91 / 8.01 |
| colour match off | 8.10 / 7.70 |
| retime nearest | 7.94 / 7.74 |

**The fallback + layer-repair chain is the biggest lever.**
- Contact sheets (`/workspace/dub/r11s/fb_m18.jpg`, `fb_m03.jpg`) show nearly all fb7 "wash" flags fire on a CLOSED generated mouth over an OPEN source mouth: the source cavity / teeth make the source box darker.
- The "dark" flags fire on legitimately open generated mouths.
- The repair's opening-spike class removes real 1–2-frame closures.

## Per-fix table
SyncNet / strong flashes / stick / slide. **Bold** = full gate; the other flash counts come from the gate's own `r3_flash` (`scripts/r11s/flash.sh`).

Base (r9 + fix): m18 **7.743 / 8 / 2 / 1**; m03 **7.651 / 13 / 1 / 1**; old03 **7.198 / 36 / 0 / 0**; old05 **3.882 / 3 / 0 / 4**.

| fix | flags | m18 | m03 | old03 | old05 | verdict |
|---|---|---|---|---|---|---|
| 1 colour match, mouth excluded | `--cm-mouth-excl 0.5` (mean) | 7.896 / 9 | 7.731 / 13 | 7.255 / **44** | 4.214 / 6 | flashes on old03 / old05: not gate-safe |
| 1 | `--cm-mouth-excl 0.5 --cm-mouth-mode zero` | 7.974 / 12 | 7.687 | | | +4 flashes on m18 |
| 1 | `--cm-mouth-excl 0.5 --cm-mouth-mode field` | 7.922 / 10 | 7.715 / 15 | 7.270 / 41 | 4.037 / 6 | flashes |
| 2 retime mouth guard | `--retime-mouth nearest` | 7.929 / 21 | 7.737 / 19 | | | REJECTED: 1-in-6 frame step = flashes |
| 2 | `--retime-mouth warp` | 7.881 / 18 | 7.733 / 18 | | | REJECTED |
| 2 | `--retime-mouth hf` | 7.754 | 7.671 | | | no gain |
| 3 fallback fix (s3w) | r8 `--open-gate 0.04 --dark-transient on --thr-wash-lips 0.2` + repair `--spikes off` | **8.011 / 7 / 1 / 1 PASS** | **8.071 / 19 / 1 / 1 FAIL flashes** | 7.270 / 34 | **3.985 / 5 / 0 / 9 FAIL slides** | best gate-safe lever (see m03 / old05 notes) |
| 3 (s3ogns, no lip-ring test) | r8 `--open-gate 0.04 --dark-transient on` + `--spikes off` | **8.174 / 10 / 1 / 1 PASS** | **8.255 / 24 / 3 / 1 FAIL** | | | misses the m18 410–412 pale pop |
| 3 stricter repair | + `--slide 0.09 --red-off 0.09` | 8.113 / 11 | 8.262 / 25 | | 4.144 / 5 | no gain |
| 4 mouth region | `--mouth-region 0.55` | **7.838 / 7 / 2 / 1, occluder ring 0→4 FAIL** | 7.696 / 12 | 7.371 / 32 | **3.968 / 4 / 0 / 7, occl 1 FAIL** | wider paste = occluder ring |
| 4 | `--mouth-region 0.45` | 7.821 | 7.668 | | | small |
| 4 registration | `--register sim` / `rigid` | 7.745 / 7.790 | 7.628 / 7.629 | | | no gain |

Fallback % of LatentSync frames (r8 score > 0.5; frames left on the source mouth after the repair):

| | m18 | m03 | old03 | old05 |
|---|---|---|---|---|
| fb7 (now) | 4.6 % / 2.1 % | 5.2 % / 5.0 % | 2.9 % / 0 % | 0.5 % / 6.0 % (repair: 53 lip_colour + 22 slide) |
| fix 3 (s3w) | 3.0 % / 0 % | 2.7 % / 2.5 % | 2.2 % / 0 % | 1.0 % / 5.1 % |

## Stacks (full gate)
| stack | m18 | m03 | old03 | old05 |
|---|---|---|---|---|
| **s3w** = fix 3 alone | 8.011 (+0.27) PASS | 8.071 (+0.42), flashes 13→19 | 7.270 (+0.07), flashes 34 (fast) | 3.985 (+0.10), slides 4→9 |
| k7 = s3w + `--mouth-region 0.55` | 8.121 (+0.38), occl 0→1 | 8.084 (+0.43), flashes 18 | **7.478 (+0.28) PASS** | 4.160 (+0.28), slides 7, occl 2 |
| k5 = k7 + `--cm-mouth-excl 0.5` | **8.334 (+0.59) PASS** | 8.207 (+0.56), flashes 17 | 7.574 (+0.38), occl 0→1 | 4.228 (+0.35), flashes 3→11 |

**Recommended flag set (fix 3 only, gate-safe on m18; behind flags):**
```
FB_ARGS="$(cat reports/experiments/r8-face/fb7.args) --open-gate 0.04 --dark-transient on --thr-wash-lips 0.2"   # r8_fallback.py
REP_ARGS="--spikes off"                                                                                         # r9_layer_repair.py
```
**Higher-sync option, pending the eye check:** add `EXTRA="--tex-tau 1000 --tex-sigma 2.5 --cm-mouth-excl 0.5 --mouth-region 0.55"` (k5).

## Open issues
- **m03 flashes:** every fallback variant that gains SyncNet on m03 raises strong flashes 13 → 17–24.
  - The new flashes sit on 2–3-frame closures of the generated mouth while the source keeps talking with an open mouth (master 85–87, 92–94, 361–362, 516–518, 528–531; strip `/workspace/dub/r11s/strip_m03.jpg`).
  - They look like natural lip closures, and their speed matches the person's own (layer |Δopen| p99 0.083 vs source 0.076).
  - The flash detector counts them because the source doesn't move. fb7 hid them via repair / the source fallback, which is where the SyncNet went.
  - Re-flagging short closures (`--og-short 3`) does not remove them (19–20).
  - **Needs an eye check + a decision on the flash rule.** Clips are in outbox.
- **old05 slides 4 → 7–9 for any arm:** consistent with the face agent's note that old05's landmarks are unstable (every arm fails there). SyncNet on old05 covers only 10 s.
- **Colour match:** its loss is real on m18 (+0.34 when off), but every exclusion mode adds flashes on old03 / old05. The per-frame correction stabilises the mouth tone there. A temporally smoothed colour match (two-pass, like `--reg-smooth`) is the untested next step.
- **Retime:** the flow blend's SyncNet cost (+0.17 on m18 for nearest) cannot be recovered without the flash detector firing at the interpolated frames.

## Clips
`/workspace/dub/outbox/r11_sync/` on r11-q. Full length, LEFT r9 + fix (now) | RIGHT candidate, demucs mix, H.264, about 23 MB each:
- `m03_r9fix_vs_sync_s3w.mp4`, `m18_r9fix_vs_sync_s3w.mp4`: the recommended set (fix 3).
- `*_k5.mp4`: the higher-sync stack.
- `*_k7.mp4`: fix 3 + mouth region 0.55.

k0 (fb7 unchanged + `--cm-mouth-excl 0.5 --mouth-region 0.55`), SyncNet / fast strong flashes:
- m18 8.078 (+0.32)
- m03 7.719 (+0.07)
- old03 7.451 (+0.26) / 41 (base 35)
- old05 4.069 (+0.19) / 6 (base 3)

## Left mid-test / next (not done)
1. **E2E wiring:** wire k5 into `scripts/r9/e2e.sh` as `E2E_SYNC=1` (default off):
   - `FB_ARGS` += `--open-gate 0.04 --dark-transient on --thr-wash-lips 0.2`;
   - `REP_ARGS="--spikes off"` (r7_m7_comp.sh passes `REP_ARGS`);
   - `EXTRA` += `--cm-mouth-excl 0.5 --mouth-region 0.55`;
   - `REP_LABEL` ≠ rep2 so the caches don't collide.
   - Then show E2E_FX=1 E2E_SYNC=1 on m18 = 8.33.
2. **Remaining old03 / old05 failures with k5:**
   - old03: occluder ring 0→1 (from the mouth region 0.55).
   - old05: flashes 3→11 (from the colour exclusion); slides fail on every arm.
   - Candidates: k5 with `--mouth-region 0.45`, or with a temporally smoothed colour match.
3. **More gain:**
   - the remaining source fallback: old05 5 % from the repair's lip_colour / slide classes; m03 2.5 % from long runs;
   - the CodeFormer detail inner alpha (0.4) vs sync;
   - sanitize-layer (replaces about 6 % of layer frames on m18; off = +0.04);
   - paste guard / face clip (KeySync saw gains from turning them off).
4. **Use the existing baselines:** old03 / old05 also exist on r11-s as qo3 / qo5 (face agent). My old03 / old05 here are fresh E2E_FX=1 runs on r11-q (`r9g_old03` etc.), which is fine for relative numbers.
5. **Face agent's `--colour-mouth-flat`:** committed as 204a1aa in /workspace/dub/dubai, equivalent to `--cm-mouth-excl D --cm-mouth-mode mean`. Not cherry-picked, to avoid a duplicate flag.

## Resume commands (r11-q)
```
cd /workspace/dub/r11_sync     # branch experiment/r11-composite-sync
FA="--open-gate 0.04 --dark-transient on --thr-wash-lips 0.2"
bash scripts/r11s/fb_arm.sh <id> k5 "$FA" "--spikes off" "--cm-mouth-excl 0.5 --mouth-region 0.55"   # composite + SyncNet
bash scripts/r11s/flash.sh <id> s0base k5                                                         # fast strong flashes
bash scripts/r11s/gate_arm.sh <id> k5                                                             # full gate vs r10/gate/<id>base
bash scripts/r11s/clip2.sh <id> k5 "r9+fix + sync fixes"                                          # 2-panel user clip -> outbox/r11_sync
```

## Tools (`scripts/r11s/`)
- `arm.sh <id> <tag> "<flags>"`: composite + SyncNet.
- `fb_arm.sh <id> <tag> "<r8 flags>" "<repair flags>" "<composite flags>"`.
- `sweep.sh`.
- `flash.sh <id> <tags>`: strong flashes, fast.
- `gate_arm.sh <id> <tag>`: full gate.
- `fb_sheet.py`: contact sheet of the flagged frames.
- `clip2.sh <id> <tag> "<label>"`.
- `base_old.sh`.
