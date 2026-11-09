#!/usr/bin/env python3
"""r16c T: fix of the KeySync keyframe scheduler (the broken last 0.3-1.2 s of m03 / m18 / q01, Irina 09.11). With add_zero_flag every chunk
of 14 keyframe slots = the reference "zero" slot + 13 keyframes. The authors count the zero slots as len(range(0, K + (K + 1) % 14, 14)), which
is short of ceil(K / 13) for most keyframe counts K (617 of K = 2..799; m03: K 83 -> 6 inserts, 7 needed). Then the LAST chunk(s) run without
the reference slot (m03: 5 keyframes + 9 duplicates) and those windows break (blotches, smeared nose, misplaced mouth), on the real frames
whenever that chunk reaches back before the pad. Fix: exactly ceil(K / 13) zero slots, same positions 0, 14, 28, ...
KS_ZEROFIX=0 at run time = the authors' count (A/B). Idempotent; run from the keysync code dir: python3 ks_zeroflag.py"""
import pathlib
p = pathlib.Path("scripts/sampling/dubbing_pipeline_raw.py"); s = p.read_text()
old = """        len_audio_image_preds = (
            len(audio_image_preds) + (len(audio_image_preds) + 1) % num_frames
        )
        for i in range(0, len_audio_image_preds, num_frames):"""
new = """        len_audio_image_preds = (
            len(audio_image_preds) + (len(audio_image_preds) + 1) % num_frames
        )
        if __import__("os").environ.get("KS_ZEROFIX", "1") != "0":   # r16c T (ks_zeroflag.py): one zero slot per 13 keyframes, the last chunk included
            len_audio_image_preds = -(-len(audio_image_preds) // (num_frames - 1)) * num_frames
        for i in range(0, len_audio_image_preds, num_frames):"""
if "KS_ZEROFIX" not in s:
    assert s.count(old) == 1, "unexpected dubbing_pipeline_raw.py (add_zero_flag block)"
    p.write_text(s.replace(old, new))
print("ZEROFIX_OK")
