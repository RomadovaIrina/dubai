# r8-sep: Demucs instead of Spleeter for the bed (29.10.2026, branch `experiment/r8-sep`, worktree `/workspace/dub/sep`)

## LICENCE: deliberate NC deviation
`htdemucs` / `htdemucs_ft` (Demucs v4, MIT code) and torchaudio `HDEMUCS_HIGH_MUSDB_PLUS` have weights trained on MUSDB18-HQ (non-commercial data).
The user accepted that risk on 29.10 for this experiment. It is a **deliberate deviation** from the round-8 rule "commercially clean models only (no HDEMUCS)".
- Every NC path is refused unless `DUB_ALLOW_NC_BED=1`. The default stays `--sep spleeter` (bit-exact r8v4) until the user approves.
- The weights live in `/workspace/dub/quarantine_nc/torch/hub` (721 MB), OUTSIDE the roots that `clean_face_stack.banned_files()` scans (repo `models/`, `third_party/`, `~/.cache/torch/hub`). The licence guard of every other run on this server stays green.
- demucs 4.0.1 is pip-installed with `--target /workspace/dub/sep_pylib` (+ dora-search, julius, einops, openunmix, lameenc, treetable, retrying, submitit; 3.8 MB). The shared `/venv/dabai` is untouched.

## Flags (all off / spleeter by default)
| where | flag | what |
|---|---|---|
| `scripts/quality/r3_remix.py` | `--sep spleeter\|hdemucs\|htdemucs\|htdemucs_ft` | the bed's separator. Non-spleeter stems come from `/workspace/dub/r8a/stems_<sep>[_res]/<id>/` (separated on the fly when missing) |
| | `--sep-bg stems\|residual` | the bed = sum of the non-vocal stems \| source − vocals |
| | `--bed-pauses sep\|source` | in the source pauses (speech_intervals ±0.3 s off) the bed IS the source (bed + vocals stems); 150-ms linear joins inside the pause |
| | `--bed-noise-back off\|on` | the vocals stem's stationary noise (noisereduce, profile = that stem in the pauses), clamped frame-wise to its pause level +3 dB, goes back to the bed (htdemucs files 02's room noise under "vocals") |
| | `--room-fill on\|off` | mix_background's room-tone top-up (was hard-wired on) |
| `scripts/pilot/background_bed.py` | `separate_background(..., separator=htdemucs\|htdemucs_ft\|hdemucs)`, env `DUB_SEP_BG`, `DUB_DEMUCS_SHIFTS` (1) | |
| `scripts/r8/sep_stems.py` | one separator over N videos, model loaded once, per-video timing | |
| `scripts/r8/sep_mix.sh <tag> <sep> <stems\|residual> "<extra>"` | the r8v4c mix with ONLY the bed changed (same `mix_src` = dub, placement, hush, soften; same remix flags), + mixdrop, clicks, gap_metrics vs r8v4c | |
| `scripts/r8/bed_metrics.py`, `mix_nat.py`, `sep_table.py`, `listen_sep.py`, `sep_watch.sh`, `derive_res.py` | the measurements / deliverables below | |

Proof that only the bed changes: `sep_mix.sh sepS0 spleeter stems ""` rebuilds r8v4c's final mix **bit-exact** (max diff 0.0). The dub files of the candidate have samples identical to the audio-r8-v4 handoff (max diff 0.0; the md5 differs only in the WAV PEAK-chunk header).

## Step 2: the raw beds (before gap_clean / cap / fill), against r8v4's dub timing (`/workspace/dub/r8a/sep_bed_metrics.json`)
S = source speech, P = source pauses, GS = r8v4 dub gaps where the source speaks (the bed is heard alone where the voice was removed).

| per video 01 / 02 / 03 / 04 / 05 | Spleeter (r8v4) | htdemucs | htdemucs_ft | HDEMUCS (torchaudio) |
|---|---|---|---|---|
| Silero voice in the bed, S (s) | 0 / **1.06** / 0.36 / 0 / **1.62** | **0 / 0 / 0 / 0 / 0** | 0 / 0.04 / 0 / 0 / 0 | 0 / 0.26 / **1.0** / 0 / 0.24 |
| Silero voice in GS (s) | 0 / **0.48** / 0 / 0 / 0.04 | 0 everywhere | 0 everywhere | 0 / 0 / 0.62 / 0 / 0.02 |
| Whisper words in GS | 0 | 0 | 0 | 03: **1** |
| voice-band excess in S (dB) | 1.05 / 1.37 / 1.16 / 0.68 / 0.78 | 0.74 / 1.26 / 0.44 / 0.33 / 0.45 | 0.85 / 1.53 / 0.51 / 0.29 / 0.55 | 0.6 / **4.67** / 0.34 / 0.29 / 0.48 |
| clicks in the bed (01 = music) | 50 / 2 / 4 / 0 / 3 | 30 / 0 / 1 / 0 / 1 | 20 / 0 / 0 / 0 / 0 | 1 / 2 / 4 / 0 / 0 |
| bed in P vs source in P: LSD (dB) | 7.0 / 2.0 / 3.7 / 8.8 / 1.6 | 0.5 / **10.5** / 3.3 / 2.3 / 7.0 | 1.5 / 12.3 / 4.5 / 5.5 / 7.6 | 0.3 / 14.4 / 2.3 / 1.5 / 5.4 |
| bed in S vs the source room: LSD (dB) | 16.7 / 20.7 / 14.2 / 15.2 / 9.9 | 11.0 / 16.0 / 10.4 / 8.1 / 12.1 | 12.0 / 16.4 / 11.4 / 10.7 / 11.6 | 12.2 / 16.0 / 14.0 / 11.3 / 13.3 |
| energy > 11.5 kHz in P (dB rel. total; source) | −139 / −95 / −71 / −81 / −84 (src −47 / −55 / −39 / −46 / −70) | −42 / −37 / −44 / −42 / −50 | −33 / −35 / −36 / −39 / −47 | |
| separate s per video-min (RTX 5090) | 0.04 (torch port) | 0.54 | 2.1 | 0.29 |

- **Spleeter 2stems processes only up to 11 kHz**: the r8v4 bed has no top end at all, under the voice AND in the pauses (the "dead / dull" bed). Demucs keeps the full band.
- **htdemucs** = the only candidate with zero detected voice in the bed on all five. HDEMUCS leaks (03: a Whisper word in a gap; 02 voice band 3.4× Spleeter's): OUT. htdemucs_ft is 4× slower and leaks slightly on 02: no gain.
- **htdemucs's one weakness:** it files part of 02's steady room noise under "vocals" (the pause bed is −7 dB vs the source, with no speech in it), and under the voice it takes the room noise out (02 bed −65 dB vs −52 dB ambience). Hence `--bed-pauses source` and `--bed-noise-back on`.
- `source − vocals` (residual) beds: no leak change, a little closer to the room, but more source transients (01). Not needed with `--bed-pauses source`.

## Step 3: the r8v4 mix on the new bed (same dub / placement / tags), workarounds relaxed
Gate per video vs r8v4c: `/workspace/dub/r8a/sep_table.md` (all variants). Variants (the r8v4 remix flags + ...):
- **H0** htdemucs.
- **H0p** + pauses = source.
- **H1** src-duck 0.
- **H2p** + no gap_clean denoise.
- **H3p** + room fill off.
- **H4p** all workarounds off.
- **S1** Spleeter with src-duck 0 + no denoise (the control).
- **H6p** pauses = source + noise-back (unclamped) + src-duck −3.
- **H8p** = H6p with the clamped noise-back = **the candidate**.

What relaxing taught:
- **gap_clean denoise and room fill must stay.** Room fill off = 11–12 dead-air events on 02, because htdemucs takes 02's room out with the voice.
- Without denoise the bed under the removed voice shows more voice-band warble, on Spleeter much more (S1 1.0–1.5 dB vs 0.3–1.45). So these were not only Spleeter workarounds.
- **The source duck (−6 dB) can be halved.** −3 keeps every leak number at or under r8v4. At 0, 01 picks up Silero hits on the music bed (0.34 s vs 0.14), although the raw bed itself has no voice.
- gap_clean's floor 3 → 2: no help (02 abrupt stays, 01 voice up).

### The candidate sepH8p vs r8v4c (final mix, dub gaps)
| | 01 | 02 | 03 | 04 | 05 |
|---|---|---|---|---|---|
| gap words (Whisper on the mix gaps) | 0 | 0 | 0 | 0 | 0 |
| leaked voice, Silero in gaps (s) r8v4 → sep | 0.14 → 0.14 | 0.02 → 0.00 | 0.02 → 0.02 | 0 → 0 | 0.32 → 0.24 |
| voice-band excess where the source speaks (dB) | 0.61 → 0.60 | 0.89 → **1.23** | 0.37 → 0.22 | 0.09 → 0.06 | 0.23 → 0.29 |
| dead / abrupt after a dub word (mixdrop) | 1/1 → 1/1 | 0/0 → 0/**1** | 1/0 → 1/0 | 1/0 → **0/0** | 2/0 → 2/0 |
| new bed clicks in gaps (not in the source) | 0 | **1** (75.14 s, score 18.8 vs thr 18) | 0 | 0 | 0 |
| level where the voice was removed vs the pauses (dB) | −3.0 → −1.7 | −4.9 → −2.7 | −6.8 → −5.2 | −6.9 → −6.5 | −8.2 → −8.4 |
| spectral shape there vs the pauses (dB, lower = the same room) | 4.9 → **2.3** | 1.8 → 2.0 | 5.2 → **2.2** | 6.2 → **2.2** | 4.1 → **1.9** |
| air > 11.5 kHz in gaps (source in its pauses) | −83 → −47 (−47) | −59 → −40 (−55) | −66 → −48 (−39) | −69 → −46 (−46) | −80 → −59 (−70) |
| bed gain / LUFS | same ±0.3 dB / −16.0 | | | | |

**Verdict:** a clear win on 01 / 03 / 04 / 05: the same or less leak, no new dead air or clicks, the bed under the removed voice sounds like the room (shape 2× closer), and the top end is back. **On 02 the strict gate is NOT fully met:**
- 1 borderline "abrupt" at 50.8 s (post −56.4 vs a limit of −55.9 dB; no dead air);
- 1 marginal click at 75.14 s;
- a higher quiet voice-band excess (1.23 vs 0.89 dB; Silero 0 s, Whisper 0 words).

So the handoff is staged as a CANDIDATE, without READY, until the user listens (and approves the licence deviation).

## Deliverables
- `/workspace/dub/outbox/sep/{02,04,05}_audio_sep_WATCH_AB_r8v4-then-sep.mp4`: A = r8v4 (Spleeter bed), a 1-s black "B" card, B = sep (htdemucs bed).
- `/workspace/dub/outbox/sep/{02,04,05}_audio_sep_over-original.mp4`.
- `/workspace/dub/outbox/sep/{02,04,05}_audio_sep_listen.md`: the gaps where A and B differ most, with A and B timecodes.
- `/workspace/dub/handoff/out/F/audio-r8-v5-candidate/` (all five, same layout as audio-r8-v4; NO READY).

## Reproduce
```
cd /workspace/dub/sep; export DUB_ALLOW_NC_BED=1 PYTHONPATH=/workspace/dub/sep_pylib
flock /workspace/dub/gpu.lock python scripts/r8/sep_stems.py --sep htdemucs --out /workspace/dub/r8a/stems_htdemucs 01 02 03 04 05
bash scripts/r8/sep_mix.sh sepH8p htdemucs stems "--bed-pauses source --bed-noise-back on --src-duck-db -3" 01 02 03 04 05
bash scripts/r8/sep_watch.sh sepH8p 02 04 05
```
In the one-command chain: `REMIX_ARGS="<r8v4 flags> --sep htdemucs --bed-pauses source --bed-noise-back on --src-duck-db -3"` with `DUB_ALLOW_NC_BED=1` and the PYTHONPATH above.

## Cost
- Disk: 60 → about 67 GB of 150 GB (stems ~1.5 GB, NC weights 0.7 GB, mixes ~3 GB).
- Download: about 0.8 GB (weights + pip).
