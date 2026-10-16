# You are AGENT B: face restoration / temporal stabilization track (Server B)

Branch: `experiment/restoration-temporal` (create it from the verified `tts-intelligibility-fix` SHA). Track id for paths:
`restoration-temporal`. Server B: vast.ai, 1× RTX 5090 32 GB, AMD Ryzen 9 9950X3D (16 vCPU share), ~46 GB RAM allocation, 120 GB disk, France.

## Your research question

**Do not change the lip-sync generator in the first phase.** Take the output of the current LatentSync and find out how to remove:

* face flicker;
* mouth texture instability;
* expression jumps;
* pasteback discontinuities.

## First, a decomposition (before any new model)

Answer with numbers on the same frames: how much of the jitter is created by

1. LatentSync itself (raw generator output);
2. CodeFormer (current settings: optimized, mouth scope);
3. pasteback (mouth-only composite, mask, landmark smoothing);
4. transitions (LS↔original switches, gate merge).

Minimum set, all on the same frames: **LatentSync raw**, **LatentSync + current CodeFormer**, and later
**LatentSync + candidate restoration**. Save the intermediate frames (raw LS, after CodeFormer, after composite) for one 03 window
so that each stage is measurable on its own.

If it turns out that CodeFormer is not a meaningful source of the problem, **do not run an endless restoration sweep**. Move on to
temporal pasteback / stabilization.

## Your search space

* Replacements or additions for CodeFormer: at least look at **VQFR** and **RestoreFormer / RestoreFormer++**, plus other face
  restoration models with permissive licenses. Search on your own.
* **Video / temporal** restoration and stabilization methods, not only frame-by-frame enhancers. For example, temporally
  consistent face restoration, optical flow (RAFT-family) guided propagation, landmark warping, stable-mouth-texture
  propagation, temporal smoothing limited to the mouth ROI, and smarter blending at LS↔original transitions.
* Out of scope: replacing the lip-sync generator (no MuseTalk / VideoReTalking / Wav2Lip-class models) and the TTS / translation /
  audio pipeline.
* Do not use CoTracker as a production dependency if the license of that specific version forbids commercial use.

## License notes for your track

Check every candidate's code license and weights license, and the license of any face detector, parser or landmark model it
pulls in. Also audit the restoration dependency the baseline already uses: CodeFormer and its weights, plus facexlib / basicsr
assets. Your experiment directly evaluates it, so record its real license status, with links, in `LICENSES.md`.

## A/B protocol

Same input video, same LatentSync raw output, same dub audio, same resolution / FPS / frame count. Start with a short 03 window that
contains continuous speech, a head turn, a closed mouth and a speech↔silence switch. Only then run whole videos. Report the
decomposition first, then candidates, with the full common metric set and side-by-side clips.

Then follow the common rules below.
