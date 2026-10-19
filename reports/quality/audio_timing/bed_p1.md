# Agent C: clean background bed, first A/B on 03 (CPU, 2026-10-16)

Task (coordinator): replace the cand5 background bed (torchaudio HDEMUCS_HIGH_MUSDB_PLUS, weights without a licence → NO) with a separator
whose code AND weights are MIT / Apache-2.0, run it on the CPU, rebuild the bed with the existing `background_bed.mix_background`, and compare
with the cand5 mix. Licences: `LICENSES.md` (this folder). Script: `scripts/quality/bed_ab.py` (separator → bg / vocals stems → the
UNCHANGED `mix_background` with the same cand5 dub track, so only the bed differs → metrics on the bed as mixed).

## Candidates

| bed | licence verdict | how | CPU s (90 s video, 2 threads) |
|---|---|---|---:|
| HDEMUCS (cand5) | NO (reference only) | the reference run's own stems | – (GPU) |
| **Spleeter 2stems** | YES, note: weights = release asset of the MIT repo, no separate model licence file | own venv (`/workspace/dub/venv-spleeter`, TF 2.9.3, numpy < 2), bg = accompaniment | 10.6 |
| Open-Unmix UMX-HQ | YES (Zenodo: MIT) | `openunmix` 1.3.0 (no-deps target dir), bg = drums + bass + other | 14.3 |
| Open-Unmix UMX-SE | YES (Zenodo: MIT) | speech enhancement at 16 kHz, bg = noise stem (upsampled) | 1.3 |
| room tone (no model) | YES | the source itself outside source speech (± 0.15 s, 30 ms crossfades) + the video's own steady non-speech frames, shuffled, under speech | 1.4 |

## Results on 03 (bed as mixed; risk = dub pauses while the source speaks, 11.1 s; ambience = pauses where nobody speaks)

| bed | voice residue: Silero speech prob in risk (s > 0.5) | prob in ambience | bed dB risk / ambience | continuity (risk − ambience) | 100 ms level jumps p95 in dub pauses | mix dB in dub pauses (dead air = no bed) |
|---|---|---:|---|---:|---:|---:|
| HDEMUCS (NC, ref) | 0.048 (0.22 s) | 0.012 | −39.1 / −36.1 | −3.0 dB | 7.4 dB | −37.0 |
| **Spleeter** | **0.003 (0.0 s)** | 0.003 | −35.4 / −35.3 | **−0.1 dB** | 4.5 dB | −35.7 |
| UMX-SE | 0.021 (0.0 s) | 0.004 | −38.7 / −35.4 | −3.3 dB | 4.1 dB | −36.5 |
| room tone | 0.003 (0.0 s) | 0.004 | −38.0 / −35.6 | −2.4 dB | 3.7 dB | −36.4 |
| UMX-HQ | **0.305 (3.2 s)** | 0.066 | −37.1 / −35.2 | −1.9 dB | 4.0 dB | −35.9 |

Reading: Spleeter 2stems is the best candidate on 03: no detectable voice in the bed where the dub is silent and the speaker talks (HDEMUCS:
0.22 s, incl. the known 88.6 s residue), and the ambience level does not change when the speaker starts talking (−0.1 dB vs −3.0 dB: HDEMUCS
removes part of the ambience with the voice). UMX-HQ is out: it leaves the Russian voice in its "other" stem (3.2 s detected). UMX-SE
and the model-free room tone are clean but dip by 2.4–3.3 dB under the source speech (UMX-SE also has 16 kHz bandwidth; room tone has no
music under speech). Caveats: one video, outdoor ambience without music; Silero speech probability is a proxy for audible residue.

Listening clips: `/workspace/dub/outbox/audio-bed-p1/` (README inside): original frames, the same cand5 dub, 7 beds incl. no-bed and the NC
reference, windows 43–48 s, 64.5–69.5 s, 85.5–90.3 s.

Next: confirm on 01 (music?), 02 (no face; long pauses), 04, 05 (outdoor noise) and wire `--bed-separator spleeter|roomtone` into
`background_bed.separate_background` (subprocess into the Spleeter venv; the HDEMUCS loader stays only behind the NC guard) after the
listening verdict.
