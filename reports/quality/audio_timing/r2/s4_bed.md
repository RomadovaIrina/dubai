# r2 step 4: clean background bed (Spleeter 2stems) on 01–05

Default since r2: `--background on --bed-separator spleeter` (MIT code; weights = release asset of the MIT repo, sha256 = the release
checksum.json, `scripts/setup_spleeter.sh`; HDEMUCS refused unless `DUB_ALLOW_NC_BED=1`). Metrics (`bed_ab.py`, bed as mixed = mix / peak − dub):
**residue** = Silero speech in the bed where the source speaks and the dub is silent (risk zone), **continuity** = bed level in the risk zone −
bed level in pure ambience (0 = the ambience does not change when the speaker starts talking), **jump p95** = 100 ms bed level jumps in dub pauses.

| video | HDEMUCS (NC reference, old duck) residue s / continuity dB / jump p95 | Spleeter, old duck, no residue duck | **delivery** (Spleeter + residue duck + duck −4 dB / 250 ms) | bed character |
|---|---|---|---|---|
| 01 | 0.0 / −1.4 / 16.5 | 0.0 / −1.5 / 18.6 | **0.0 / −1.2 / 16.3** | music (the jumps are the music's own hits, also in the stem) |
| 02 | 0.0 / **−8.9** / 6.9 | **0.45** / +0.4 / 6.9 | **0.0 / −3.4 / 7.2** | quiet room (−47 dB), room-tone fill on |
| 03 | 0.16 / −4.1 / 6.8 | 0.13 / +1.2 / 4.5 | **0.13 / +0.2 / 4.5** | outdoor ambience |
| 04 | 0.0 / −4.4 / 3.4 | 0.0 / −2.7 / 3.1 | **0.0 / −3.2 / 2.8** | outdoor |
| 05 | 0.0 / −0.5 / 4.1 | 0.0 / −0.2 / 4.2 | **0.0 / −0.6 / 3.9** | loud outdoor noise (−21 dB) |

Yesterday (03, cand5 dub track): Spleeter residue 0.0 s, continuity −0.1 dB, jump p95 4.5 dB; HDEMUCS 0.22 s / −3.0 dB / 7.4 dB. Today's 03
delivery: 0.13 s / +0.2 dB / 4.5 dB (the 0.13 s is the 59.3 s consonant onset, 75 % of it under the start of the next dub phrase).

**Residue duck** (new, `--bg-residue-duck-db -15`): Spleeter put a vocal sound of the speaker into the bed on 02 at 39.0–39.4 s (Silero p 0.99,
−32 dB, the dub is silent there: an audible Russian syllable between English phrases); HDEMUCS had filed it under vocals. Silero runs on the
bed stem; only where it detects speech while the source speaks and the dub is silent is the bed lowered by 15 dB (40 ms / 120 ms ramps):
02 0.6 s, 03 0.15 s, 05 0.36 s, 01 / 04 0 s. 02 residue 0.45 → 0 s.

Spleeter vs HDEMUCS in one line: HDEMUCS removes part of the ambience together with the voice (continuity down to −8.9 dB on 02: the room
"goes quiet" whenever the speaker talks under a dub pause); Spleeter keeps it (−3.4…+0.2 dB) and, with the residue duck, leaves no voice.
Separation time: 2–7 s per video on the CPU (in the runner: `background_sep` 5.1–7.3 s incl. TF load).
Listening: `sbs/02_037.50-041.50s_{H,I}_*` (residue duck), every `B_old` vs `C_new` (bed + timing).
