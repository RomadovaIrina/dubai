#!/bin/bash
# Round 3 D: r3cand finals with C's audio-r3b (same speech as audio-r3, loudness-normalised -16 LUFS / <= -1.2 dBTP) muxed in.
#   bash scripts/quality/r3_remux_r3b.sh [ids]
# Checks that the dub voice is the one LatentSync was driven with (audio-r3 vs audio-r3b dubbed_16k: correlation of the envelopes, gain-free),
# copies the video stream of <id>_r3cand_fix30.mp4 and the r3b final mix -> <id>_r3candb_final.mp4 (+ manifest for the scorecard),
# publishes outbox/<id>_r3cand_audio-r3_final.mp4 and outbox/<id>_sbs_CLEAN0-R3F-R3CAND.mp4 (+ _half).
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
V=/workspace/dub/artifacts/r3d/video; CI=/workspace/dub/handoff/in/C; OB=/workspace/dub/outbox; ids=${*:-"01 02 03 04 05"}
for id in $ids; do
  a3=$CI/audio-r3/$id; b3=$CI/audio-r3b/$id; f=$V/${id}_r3cand_fix30.mp4
  [ -f $f ] && [ -f $b3/final_mix_aac.m4a ] || { echo "WARN $id: missing $f or $b3/final_mix_aac.m4a"; continue; }
  python - $a3/dubbed_16k.wav $b3/dubbed_16k.wav <<'PY' || { echo "WARN $id: audio-r3b dub voice differs from audio-r3 (lips would not match)"; continue; }
import sys, numpy as np, soundfile as sf
a, _ = sf.read(sys.argv[1]); b, _ = sf.read(sys.argv[2]); n = min(len(a), len(b))
def env(x): x = np.abs(x[:n]); k = 160; return x[:len(x) // k * k].reshape(-1, k).mean(1)
r = float(np.corrcoef(env(a), env(b))[0, 1]); print(f"  dub voice r3 vs r3b envelope r = {r:.4f}, length {len(a)} / {len(b)}")
sys.exit(0 if r > 0.99 and abs(len(a) - len(b)) < 1600 else 1)
PY
  out=$V/${id}_r3candb_final.mp4
  ffmpeg -loglevel error -y -i $f -i $b3/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c copy -shortest $out
  python - $V/${id}_r3cand_fix30.manifest.json $out $b3 <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text()); out, b3 = sys.argv[2], sys.argv[3]
m["output_pre_remux"] = m["output"]; m["output"] = out; m["audio_dir"] = b3
m["candidate"] = {**(m.get("candidate") or {}), "dubbed_dir": b3, "muxed_audio": "C audio-r3b (loudness-normalised); lips rendered on audio-r3 (same speech)"}
m["dubbed_track"] = {**m["dubbed_track"], "aac": str(pathlib.Path(b3) / "final_mix_aac.m4a")}
pathlib.Path(out).with_suffix(".manifest.json").write_text(json.dumps(m, indent=1))
PY
  cp $out $OB/${id}_r3cand_audio-r3_final.mp4
  SRC=$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/$id.* | head -1)
  C0=$V/${id}_clean0a3.mp4; [ -f $C0 ] || C0=$V/${id}_clean0.mp4
  HALF=1 bash scripts/quality/r3d_sbs.sh $OB/${id}_sbs_CLEAN0-R3F-R3CAND.mp4 ORIGINAL=$SRC "CLEAN0 25fps"=$C0 "R3F (audio-r2)"=$V/${id}_r3F_fix30.mp4 "R3CAND (audio-r3b)"=$out > /dev/null 2>&1
  echo "== $id r3cand final -> $OB/${id}_r3cand_audio-r3_final.mp4"
done
