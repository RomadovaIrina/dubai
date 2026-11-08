# r16 Q: the "spring" metric (beards bounce on KeySync), 08.11

Irina 08.11: KeySync beats LatentSync on the beard panels, but on beards the face "springs".

The measurement is K1's (spring3.py on r16-k1b, definition kept):
- per frame pair, DIS flow in each part box;
- a RANSAC similarity fit (tx, ty, scale);
- extra = the final's motion − the source's motion for the same box.

Q's wrapper: `scripts/quality/r16q_spring.py --final F --master SRC_SAME_GRID --out S.json [--region x0,y0,x1,y1]`. It adds:
- a bitrate-matched floor: the master re-encoded at the final's own video bitrate and measured the same way;
- spring = cheek |scale| extra; ratio = spring / floor;
- a 95 % block-bootstrap CI over 1 s blocks;
- the worst seconds;
- an "upper" control: the unpasted band, which must stay ≈ the floor.

## Validation: cheek |scale| extra, % per frame [95 % CI]
| clip | KeySync r16 (K2 day 1) | LatentSync clean | floor, crf 30 | floor, bitrate-matched |
|---|---|---|---|---|
| q01 | 0.280 [0.257, 0.305] | 0.181 [0.164, 0.203] | 0.215 | 0.156 [0.142, 0.171] |
| q03 | 0.306 [0.244, 0.366] | 0.147 [0.117, 0.179] | 0.226 | 0.156 [0.124, 0.191] |
| q04 | 0.299 [0.270, 0.330] | 0.160 [0.141, 0.178] | 0.219 | 0.171 [0.150, 0.194] |

Wrapper runs:
- KeySync q01 ratio 1.79 (repeat 1.87: x264 non-determinism in the floor gives ±0.01 %);
- KeySync q04 1.79, with its region found automatically;
- LatentSync q01 1.37, against its own (higher-bitrate) floor.
The upper control ratio is 1.05–1.22 everywhere.

## What it can and cannot say
- **KeySync vs LatentSync: yes.**
  - On all 3 beard clips KeySync is ~1.8× its floor and its CI is clear of LatentSync's. This matches Irina's "KeySync springs".
  - The excess is in the pasted cheeks / nose, not in the unpasted band above (upper ≈ floor). So it is the paste / generator, not the camera.
  - LatentSync's paste mask leaves the cheeks mostly source pixels, so a low value there is partly "nothing generated", not "stable generated".
- **Ranking q01 / q03 / q04 against each other: no.** KeySync 0.28 / 0.31 / 0.30 % overlap within their CIs. Irina's per-clip order is not known to Q yet; when it is, check against it.
- **The floor must match the compression.** A crf 30 floor (0.22 %) is almost as high as KeySync, while the matched floor is 0.16 %. Never compare against a fixed number.
  - Beard finals flicker ~2× old03 (K1: 0.30 vs 0.17 %).
  - Use ratio = spring / matched floor, or compare with the golden final of the same clip.
- old03 e2e = golden on this metric (K1: cheek 0.16 / 0.09 vs 0.17 / 0.10).
- The content leaks in: LatentSync q01 and KeySync q01 share the worst second (15 s, a fast head move), so per-second peaks partly follow the source motion. Use the whole-clip ratio for decisions and the worst seconds only for where to look.

Suggested gate (not yet eye-validated): spring_ratio ≤ golden ratio + 0.2, or ≤ 1.3 where there is no golden. KeySync beards (1.8) fail it.
