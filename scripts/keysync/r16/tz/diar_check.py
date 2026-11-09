#!/usr/bin/env python3
"""r16 TZ 6.3: how many voices does the DUBBED track have? pyannote 3.1 (the pipeline's own diarizer) on the final audio, CPU.
    diar_check.py final.mp4 [--device cpu]"""
import collections, json, os, subprocess, sys, tempfile
from pyannote.audio import Pipeline
import torch
src = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
wav = tempfile.mktemp(suffix=".wav"); subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", wav], check=True)
tok = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
p = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=tok); p.to(torch.device(dev))
d = p(wav); s = collections.Counter()
for seg, _, spk in d.itertracks(yield_label=True): s[spk] += seg.end - seg.start
tot = sum(s.values())
print("DIAR", json.dumps({"input": src, "speakers": len(s), "speech_s": {k: round(v, 1) for k, v in s.most_common()}, "top_share": round(s.most_common(1)[0][1] / tot, 3) if tot else None}))
os.remove(wav)
