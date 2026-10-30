#!/usr/bin/env python3
"""r8-sep: residual bed of a stem cache: bg = source - vocals stem (keeps what the model filed under no stem, e.g. the room noise under the voice).
    python scripts/r8/derive_res.py /workspace/dub/r8a/stems_htdemucs 01 02 ...   -> /workspace/dub/r8a/stems_htdemucs_res/<id>/"""
import os
import json, pathlib, subprocess, sys
import numpy as np, soundfile as sf
RAW = pathlib.Path(os.environ.get("DUB_INPUTS", "/workspace/dub/inputs/test_videos_raw"))
src_dir = pathlib.Path(sys.argv[1]); out = src_dir.with_name(src_dir.name + "_res")
for id_ in sys.argv[2:]:
    d = out / id_; d.mkdir(parents=True, exist_ok=True); w = d / "_src.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(next(RAW.glob(f"{id_}.*"))), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", str(w)], check=True)
    y = sf.read(str(w), dtype="float32")[0]; w.unlink(); voc = sf.read(str(src_dir / id_ / "vocals_44k.wav"), dtype="float32")[0]
    m = min(len(y), len(voc)); sf.write(str(d / "bg_44k.wav"), (y[:m] - voc[:m]).astype(np.float32), 44100, subtype="FLOAT")
    (d / "vocals_44k.wav").unlink(missing_ok=True); (d / "vocals_44k.wav").symlink_to(src_dir / id_ / "vocals_44k.wav")
    j = json.loads((src_dir / id_ / "sep.json").read_text()); j.update(bg_mode="residual", bg=str(d / "bg_44k.wav")); (d / "sep.json").write_text(json.dumps(j, indent=1))
    print(id_, "residual", d)
