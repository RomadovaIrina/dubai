# r16 Q: visual defect detectors (`scripts/quality/r16q_detect.py`)

Usage:

```
/venv/mediapipe/bin/python scripts/quality/r16q_detect.py --src SRC --ours OURS --out det.json
python3 scripts/quality/r16q_clips.py ...
```

The second command makes 0.5x source | ours face-crop clips of the flagged ranges.

How a run works:
- Decode both videos at 25 fps.
- MediaPipe finds the faces in the source: the largest face is the target, the rest are the "others".
- Take a 512 crop around the target and compute the per-frame signals.
- Flag a range when a signal is over its threshold for at least the minimum number of frames.

| detector | what it measures | threshold / min frames |
|---|---|---|
| black | inner mouth L < 15 in an open mouth, where the source is not black | 2 % / 3 |
| slide | Farneback flow source→ours, per region (upper / cheeks / chin, mouth excluded), % IOD | 4.0 / 2 |
| swap | mouth \|ours − src\| < 0.4 × median: the source face shown (old ks_fix guard); its edges are the "floats" | 3 |
| overlap | changed pixels inside the hulls of the other faces | 3 % / 3 |
| texture | jaw HF energy ours / src, 13-frame mean | < 0.6 / 8 |
| explode | 2nd difference of mouth gap / width or open ratio, or a pixel spike | 0.10 / 0.15 / z 3.5 |
| edge | p90 gradient of blur(ours − src) in a band at the cheek sides of the oval | 3.0 / 5 |
| flicker | lip-ring Lab 2nd difference, 13-frame mean, > 2× the source's and > 0.9 | 6 |

## Calibration on known defects: all HIT

| known defect | video | detector | flagged |
|---|---|---|---|
| m03 0:25 black hole | m03 plain / cfhf / q2u | black | 25.32–25.64 |
| m18 3.7 s slide (jelly) | m18 plain | slide | 3.64–3.72 |
| m03 floats 4.3 / 7.9 / 11.7 s (old cfhf guard) | m03 cfhf | swap | 4.12–4.56, 7.72–8.16, 11.52–12.0 |
| m18 float | m18 cfhf | swap | 3.52–3.92 |
| m03 8.64–8.71 LS tear | q2u m03 | explode (open ratio) | 8.68 (flicker 8.56–9.2) |
| cheek into the child | m03 cfhf vs plain | overlap | 70 frames vs 0 |
| old03 waxy jaw | old03 plain | texture | median 0.646 |
| LS vs KeySync lip flicker | q2u vs plain | flicker | 125 vs 33 frames |

## Known false positives

- explode on a cutaway with no face (q06 40.6 s): gate it by face presence.
- black on a naturally dark / toothless mouth (q02): compare it with the source's black share per range.
- swap is meaningless on `*_src` proxies, where the audio is the same, and in pauses.
