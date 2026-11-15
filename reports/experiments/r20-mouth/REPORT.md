# r20-mouth (15.11.2026): the black mouth, 04's band above the lip, the small-face smear of 05

Branch `experiment/r20-mouth` (off r18-quality e26a3f7, merged with r18-quality ffbbb61). Everything here is OPT-IN: the defaults are bit-identical until Irina picks a variant in her blind A/B (`Videos\DabAI_AB\r20_mouth\`).

## 1. The black mouth is keyframe guidance over-shoot

Method: a fixed noise seed (`KS_SEED`) and keyframe-only runs (`KS_KF_ONLY=1`, `KS_KF_DUMP=<dir>`), so two arms differ only by the knob. Black = share of L* < 6 pixels in the mouth box over all keyframes.

| keyframe guidance (audio / ref) | 04 | m03 | 01 |
|---|---|---|---|
| 5 / 2 (the authors' config, our default) | 0.042 | 0.028 | 0.056 |
| 3 / 2 | 0.027 | 0.018 | 0.044 |
| 2 / 2 | 0.019 | 0.013 | 0.036 |
| 1 / 1 (no guidance) | 0.001 | 0.000 | 0.024 (mouth soft, barely opens) |
| 2.5 / 1 | 0.006 | 0.010 | 0.041 |
| steps 4-8 without guidance (`sched:8:1:1`) | 0.008 | 0.010 | 0.028 (soft lips) |
| steps 1-2 without guidance (`hi:30:1:1`) | 0.035 | 0.026 | – |
| CFG rescale 1.0 | 0.020 | 0.024 | 0.058 |
| luma floor (`floor:1`) | – | 0.008, but a NAVY cavity | – |
| **`floorv:1`** | **0.003** | **0.003** | **0.018** |
| reference 431 / 84, local closed / teeth | 0.052–0.070 | 0.028–0.031 | – |

- The black is already in the decoded keyframes. Only the keyframe model sees the identity frame; the interpolation student is conditioned on the two keyframes.
- It forms in sampling steps 3-4 of 8 (sigma 11.3 and 5.0 of 700, 40, 11.3, 4.97, 1.79, 0.79, 0.26, 0.05).
- The reference frame does not reduce it; it decides what is drawn beside it (teeth on m03).
- The post chain darkens the raw render stage by stage (m03 black share raw 0.128 → seam 0.184) but the paste's colour match lifts it back (final 0.061): not the cause.

**`KS_KF_MODE=floorv:1`** (`scripts/keysync/r16/ks_lref.py`, `guide_hook`): per latent pixel, where the conditional prediction c (audio + reference, no extrapolation) is dark (latent luma < -0.3, soft over 0.3) and the guided prediction e is darker than c, e is pulled back to c (all four channels; luma alone leaves the chroma over-shoot = a navy cavity). Elsewhere the guidance is untouched, so the mouth shape, the lips and the lit areas keep the full guidance.

Full e2e jobs, the same saved dub and seed (n = as now, F = floorv):

| video | SyncNet conf n / F | black share in the open mouth, final n / F (source) | guard-replaced frames n / F |
|---|---|---|---|
| m03 | 7.39 / 7.40 | 0.061 / 0.043 (0.039) | 1 / 0 |
| 04 | 3.60 / 3.56 | median interior L* 19.4 / 26.6 | – |
| old03 | 5.54 / 5.65 | 0.000 / 0.000 | 48 / – |
| 01 | – | – | 0 / 0 |

Irina, 15.11, m03 pairs: the rule wins, "ещё кажись починил дёрганья щеки".

Side effect, measured: the K2 gate (`KS_GATE`, more interpolation steps on wide-open segments) keys on dark pixels (luma < 0.18), so with floorv it flags fewer segments (04 15 % → 6 %, m03 19 % → 10 %, old03 5 % → 0). `KS_GATE=...,luma=0.25` (new key in `ks_gate.py`, default 0.18) restores the share on 04 / old03 / 01 and over-flags on m03 (32 %). Mouth-opening fidelity vs "now" is the same either way (correlation 0.80–0.95 without, 0.83–0.97 with), so the candidate is floorv alone; the luma key is there if the gate share ever matters.

The guider hook composes with r18-quality's `ks_quiet.py`: both wrap `AudioRefMultiCondGuider.__call__` and call the previous one; per-step ratios travel through `self.audio_ratio` / `self.ref_ratio`.

## 2. A time-local reference: `KS_REF=local`

The keyframe model was trained with the identity frame taken right next to the generated clip (`sgm/data/video_dataset_latent.py`: one extra frame before the clip). At inference one frame serves the whole run.

- `ks_lrefpick.py` picks one frame per 169-frame keyframe batch, near that batch (± 3 s first): lower-face tone like the batch, frontal, sharp, eyes open, the face where the batch has it, the speaker's typical lip redness and thickness. `--mouth auto`: a clean-teeth frame only if the speaker shows clean upper teeth in ≥ 2 % of the speech frames (m03 0.11: teeth; old03 0.001, 04 0, 05 0.004: closed mouth). Long videos: one plan per `ks_chunk` chunk.
- `ks_lref.py` feeds the plan to the keyframe model (a json path as the 4th job column of `ks_fast.py`).
- 04 (selfie walk, frame 0 dark and green under trees): the greenish band above the lip is the tone of frame 0, doubled by the ref guidance. With the local plan [97, 225, 396, 396, 793, 936, 1058, 1058] the band is gone on the pasted frames and the lips stay close to the source (the r19 `KS_REF=skin` pick 667 gave fuller, pinker lips). Reducing the low frequencies of the ref term (`reflp`) does not remove the band.
- m03 with floorv: teeth rows instead of the hole (final black share 0.027, below the source's 0.039).

## 3. The small-face smear of 05 ("сыр", Irina 15.11 on 05_tiles at 0:37)

Already in the raw generator output; mouth detail vs the source (Laplacian std in the mouth box) 0.62 over the whole video (m03 0.87, 04 0.75). Three causes:
1. tiles mode (`KS_MP_TILES=1 DUB_ROUTE=off`) runs the video as one piece, so the reference is frame 0, where he wears SUNGLASSES: dark blobs under the eyes and almost no mouth after he takes them off → `KS_REF=local`;
2. the face is 43 % of the generator input (m03 / 04: ~71 %) → `KS_TRACK_K=1.25 KS_TRACK_SIGMA=3` (65 %). The tight crop alone is worse (it exposes cause 1);
3. at 0:34–0:38 the dub is silent while the source speaks; the silent mouth was a pressed flat blob → r18-quality's `KS_QUIET=ar=1.0,gap=0.3,ramp=0.4`.

With all three: mouth detail 0.73, smear frames 12 % → 4 % of the drawn frames, none left in 0:30–0:40. Fallback if Irina prefers the source: keep the source below ~70 px face width.

## Test switches (never set in production)
`KS_SEED`, `KS_KF_ONLY`, `KS_KF_DUMP`, `KS_KF_TAG`, `KS_KF_LIVE` + `KS_KF_AR` / `KS_KF_RR` / `KS_KF_NORM`, other `KS_KF_MODE` values (`sched`, `hi`, `rescale`, `floor`, `reflp`, `audlp`, `mix`), per-job `K=V` columns after the reference column of a `ks_fast.py` job line.

## Open
- Irina's picks on 04 (three-way), 01, old03, m03 teeth, 05.
- floorv on a bearded speaker (q01 / q04) not checked.
- The boss files in `r19_начальство` use fresh dubs: never pair them with renders from `dubs_1411.tar`.

## Defaults unchanged (check, 15.11)
Seeded 12.8 s render of 01 with the new modules in the kit vs without them (ks_lref / ks_quiet / ks_kfout removed): PSNR 48.5 dB; two identical runs: 47.8 dB (TensorRT / compiled-VAE run-to-run floor). Without a plan or KS_KF_MODE the wrappers call the original functions with the original arguments.
