# r7 audio v3 variants on the face (F): v2 | v3a | v3b | v3c on 05 and 04

Files: `05_r7_audio_v3_variants_WATCH.mp4`, `04_r7_audio_v3_variants_WATCH.mp4` — one row of four (both videos are portrait), labelled left to right: **v2 (m8b)**,
**v3a (m8v3a)**, **v3b (m8v3b)**, **v3c (m8v3c)**. Four audio tracks, one per variant (titled v2 / v3a / v3b / v3c;
the first = v2): switch the track to hear the variant you watch. Every variant is the same m8 face stack (the default, unchanged since m8),
LatentSync re-run on that variant's dub; the single-variant files are in `artifacts/r4m/m8v3{a,b,c}/<id>_m8v3?.mp4`.

v3a = re-translated to the source length (duration-aware Qwen + Chatterbox re-take, tempo 1.0); v3b = the v2 take cut between words and
tempo-fitted (0.92-1.08) onto the speaker's own bursts; v3c = v3a's re-takes + v3b's aligner (audio agent's README_v3_variants.md).

## Closed mouth while the source talks (scripts/r7/r7_closed_mouth.py: output landmarks, inner opening < 0.02 IOD while the source's > 0.05 IOD; seconds)
| video | variant | closed mouth while the source talks (s) | ... of it dub-silent (s) = the pressed-lips case | open mouth while the dub is silent (s) |
|---|---|---:|---:|---:|
| 05 | v2 (m8b) | 9.56 | 8.69 | 5.01 |
| 05 | v3a | 8.22 | 6.85 | 7.49 |
| 05 | v3b | 8.52 | 7.69 | 6.48 |
| 05 | **v3c** | **7.08** | **5.88** | 7.99 |
| 04 | v2 (m8b) | 7.23 | 4.67 | 1.90 |
| 04 | v3a | 5.33 | 3.13 | 4.07 |
| 04 | **v3b** | 5.27 | **2.20** | 4.67 |
| 04 | v3c | **5.23** | 2.57 | 5.07 |

The dub-silent subset uses an energy VAD on the variant's dubbed_16k.wav (< -45 dB, >= 0.3 s), so it also counts the pauses inside a take;
the audio agent's "speaker on screen talks, dub silent" (05: 4.3 -> 1.5 s on v3a; 04: 5.25 -> 2.35 s) is the source-speech-based number.
"Open mouth while the dub is silent" rises on every v3 variant: the longer takes carry more short internal pauses in which the generator
keeps the mouth moving (the other direction of the user's rule; 0.3-0.5 s pieces, visible as a mouth that keeps going between words).

## SyncNet / flash / twitch (m8 stack; lower twitch and flash = better, higher SyncNet = better)
| video | variant | SyncNet dub-speech | lip flash strong | twitch_sum | mouth/dub mismatch s (gate) |
|---|---|---:|---:|---:|---:|
| 05 | v2 (m8b) | 3.91 | 5 | 17.83 | 2.17 |
| 05 | v3a | 3.58 | 6 | 15.59 | 1.84 |
| 05 | v3b | **4.33** | 6 | **9.77** | 1.54 |
| 05 | v3c | 3.72 | 8 | 12.50 | **1.44** |
| 04 | v2 (m8b) | 6.18 | **9** | **36.01** | **0.73** |
| 04 | v3a | **6.42** | 16 | 38.83 | **0.73** |
| 04 | v3b | 6.11 | 11 | 37.92 | 0.97 |
| 04 | v3c | 6.28 | 12 | 41.17 | 1.03 |

Reading (F): on the pressed-lips number all three v3 variants beat v2 on both videos; v3c is best on 05 (5.9 s, from 8.7) and v3b / v3c on
04 (2.2 / 2.6 s, from 4.7). Lip-sync and twitch do not separate them beyond noise except 05 v3b (SyncNet 4.33, twitch 9.8: the best 05 yet)
and 04 v3a's flash count (16, from 9: the re-takes' consonant closures). Whole-frame gate PASS on all (face stack unchanged). The user judges
by eye: watch 05 24-25.5 s and 04 1.5-5.5 / 28-29 / 51-52 s in all four quadrants.

## Twitch table (r6_twitch_table.py; columns v2 (m8b) | v3a | v3b | v3c)
| metric | video | v2 (m8b) | v3a | v3b | v3c |
|---|---|---:|---:|---:|---:|
| twitch_sum | 04 | 36.01 | 38.83 | 37.92 | 41.17 |
| twitch_sum | 05 | 17.83 | 15.59 | 9.77 | 12.5 |
| skin_sum | 04 | 1.36 | 5.74 | 0 | 1.48 |
| skin_sum | 05 | 4.34 | 2.45 | 3.26 | 2.25 |
