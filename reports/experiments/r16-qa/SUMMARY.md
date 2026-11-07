# r16 Q: KeySync fast2 + plain paste on new open-licence clips. Interim summary (07.11, 16:30)

## What these runs are

- Generator input: the clip's own Russian audio (`*_src` proxy). This isolates the visual defects; the visual problems will be the same with the dub.
- Generator: fast2 (LCM 2-step + TRT).
- Paste: plain, `--shift 0`.
- The dub was delayed by the measured raw offset before generating.
- Real-dub e2e audio is done for q01–q08 (PARTIAL_SUCCESS = Spleeter bed, no htdemucs on this server); the KeySync runs on the dub are waiting for the GPU.
- The GPU was yielded to T at 16:20.
- Clips + licences: `reports/experiments/r16-qa/CLIPS.md`.

## Per-clip AV offset (raw KeySync output vs. its input audio, SyncNet)

| clip | raw offset | conf | action | final offset after the redub | final conf |
|---|---|---|---|---|---|
| q01_amir | −1 | 3.9 | dub +40 ms | **0** | 3.9 |
| q02_vera | −1 | 5.1 | dub +40 ms | **0** | 4.7 |
| q03_priest | −1 | 3.4–5.4 | dub +40 ms, generated | pending (GPU) | |
| q04_farikh | −1 | | dub +40 ms, generated | pending | |
| q05_rozhkov | −3 | **1.5** (unreliable) | none | −6 | 1.6 (no sync: the face is broken) |
| q06_suvorov | 0 | 0.9 | none | 0 | 1.9 |
| q08_albats | −1 | | dub +40 ms, generated | pending | |
| q09_vertical | −1 | | dub +40 ms, generated | pending | |
| q07_altapress | n/a | | MediaPipe finds no face (small face in a crowd) → no KeySync crop | | |

Every clip with a reliable measurement (conf > 3; 6 of 6) shows a 1-frame (40 ms) lag. Delaying the dub by 40 ms brings the final to 0. So the lag is systematic, not per-clip.

## Flags (detector frames as % of 1125 frames; ★ = checked by eye)

| clip | black | slide | swap | overlap | texture | explode | edge | flicker | verdict |
|---|---|---|---|---|---|---|---|---|---|
| q01_amir (beard, low light) | 0.7 | 0 | 0.7 | 0 | 6.9 (median 0.84) | 0.4 | 2.6 | 6.2 | **OK.** Beard slightly softer ★; mild flicker |
| q02_vera (older, glasses, hands) | 10.6 | 7.2 | 1.0 | 0 | 12.8 | 1.4 | 0 | 12.5 | **Hand over the face → a black rectangle + white patch on the cheek ★** (34.8–38.6 s); toothless mouth darker / bigger (12.9 s, borderline) |
| q05_rozhkov (head turns, cup, stubble) | 0.2 | **37.8** | 6.8 | 0 | **88** (median 0.32) | 0.7 | 1.2 | 6.6 | **BROKEN for the whole video ★**: teal / white blob instead of the jaw, even with no cup in frame |
| q06_suvorov (low light, profile) | 0 | 0.7 | 8.4 | 0 | 0 | 5.5 | 0.6 | 6.8 | **Yellow glowing blob over the mouth + a dark jaw ★** (27.6–30.6 s); the face is found in only 487 / 1125 frames |

## Root causes (hypotheses)

1. **q05 / q06: the static crop is too wide for a moving head.**
   - ks_prep builds ONE box for the whole clip. The head moves 419 px (q05) and 395 px (q06), so the box side is 1066 / 808 px for a face 380 / 243 px tall.
   - The face then fills only ~0.30–0.36 of the 512 KeySync input (q01: 0.39 OK; q02: 0.54 OK).
   - The generator's raw output in that case (checked at 5 / 27 / 40 s, q05) is a teal-and-white smear: the failure happens in KeySync itself, not in the paste.
   - Fix direction: a tracking or segment-wise crop (a new box per shot / large move), or refuse when face_h / side < ~0.4.
2. **Occlusion (q02 hand, q05 cup): KeySync paints through the hand.**
   - The `fix_occlusion` flag in ks_fast is False.
   - Fix direction: an occlusion mask (hand / object segmentation) → keep the source pixels there.
3. **q07: no face found** in crowd shots → the pipeline should take the largest face at a higher detection resolution, or skip the clip with a warning.

## Worst spot clips (0.5x, face crop, source | ours, timecode burnt in)

- `q05_rozhkov_src/q05_rozhkov_src_flag07_23.28-29.28_*.mp4`: jaw replaced by a teal blob, no occlusion
- `q05_rozhkov_src/q05_rozhkov_src_flag05_11.32-17.32_slide+texture.mp4`: drinking from a cup → white smear
- `q02_vera_src/q02_vera_src_flag07_31.04-36.20_explode+flicker+slide.mp4`: a hand in front → black rectangle
- `q06_suvorov_src/q06_suvorov_src_flag06_27.00-32.80_explode+flicker+slide.mp4`: yellow blob over the mouth

Server: `/workspace/dub/outbox/r16_q/<clip>/` (also `*_ALL_FLAGS_05x.mp4`, `*_flags.tsv`, the full `*_keysync_fast2plain.mp4`).

## Detector notes from this batch

- q06 at 40.6 s: explode fires on a cutaway with no face. False positive; it should be gated by face presence.
- q02 black: Vera's toothless mouth is naturally dark (the source mouth is dark 72 % of the time vs ours 29 %), so many of the black flags are borderline.
- q05 texture (88 %) and slide (38 %) correctly catch the whole-video failure.
- The calibration on m03 / m18 / old03 was all HIT: `reports/experiments/r16-qa/DETECTORS.md` (to be written).
