#!/bin/bash
# Round 3 D: publish r3final v3 (A R3 generator + B fps master; composite: flow_safe retime, sanitize-layer 0.83, lip sharpen cap 0.8,
# face clip; audio-r4b = fixed phrase cuts + remixed bed + static loudness) for <ids>, with a scorable manifest, + before / after clips.
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
V=/workspace/dub/artifacts/r3d/video; OB=/workspace/dub/outbox; A4B=/workspace/dub/artifacts/r3d/audio-r4b; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
for id in "$@"; do
  VER=${VER:-v3}; f=$V/${id}_r3final_$VER.mp4; [ -f $f ] || { echo "skip $id"; continue; }
  ffmpeg -loglevel error -y -i $f -i $A4B/$id/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c copy $V/${id}_r3final${VER}b.mp4
  python - $V/${id}_r3final_$VER.manifest.json $V/${id}_r3final${VER}b.mp4 $A4B/$id <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text()); out, d = sys.argv[2], sys.argv[3]
m["output_pre_remux"] = m["output"]; m["output"] = out; m["audio_dir"] = d
m["candidate"] = {**(m.get("candidate") or {}), "dubbed_dir": d, "muxed_audio": "audio-r4b"}
m["dubbed_track"] = {**m["dubbed_track"], "aac": str(pathlib.Path(d) / "final_mix_aac.m4a")}
pathlib.Path(out).with_suffix(".manifest.json").write_text(json.dumps(m, indent=1))
PY
  cp $V/${id}_r3final${VER}b.mp4 $OB/${id}_r3final_fixed.mp4; echo "== $id -> $OB/${id}_r3final_fixed.mp4"
done
lab() { echo "drawtext=fontfile=$F:text='$1':x=10:y=10:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.6"; }
clip() {   # id t0 t1 [crop: full | jaw]
  id=$1; t0=$2; t1=$3; d=$(python -c "print(round($t1-$t0,3))"); B=$OB/${id}_r3cand_audio-r3_final.mp4; A=$OB/${id}_r3final_fixed.mp4; n=$OB/${id}_${t0}-${t1}s
  [ -f $B ] && [ -f $A ] || return
  ffmpeg -loglevel error -y -ss $t0 -t $d -i $B -ss $t0 -t $d -i $A -filter_complex "[0:v]$(lab 'BEFORE (r3cand)')[a];[1:v]$(lab 'AFTER (fixed)')[b];[a][b]hstack=inputs=2[v]" \
    -map "[v]" -map 1:a:0 -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac -b:a 192k ${n}_BEFORE-AFTER.mp4
  ffmpeg -loglevel error -y -i ${n}_BEFORE-AFTER.mp4 -filter_complex "[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 16 -pix_fmt yuv420p ${n}_BEFORE-AFTER_half.mp4
  ffmpeg -loglevel error -y -ss $t0 -t $d -i $B -ss $t0 -t $d -i $A -filter_complex "[0:v]$(lab BEFORE)[a];[1:v]$(lab AFTER)[b];[a][0:a][b][1:a]concat=n=2:v=1:a=1[v][au]" \
    -map "[v]" -map "[au]" -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac -b:a 192k ${n}_before_then_after.mp4
  echo "== clips $id $t0-$t1"
}
bgclip() {   # 04 17-22 s: full frame + 2x crop of the lower face / jaw side, half speed, BEFORE | AFTER
  id=$1; t0=$2; t1=$3; d=$(python -c "print(round($t1-$t0,3))"); B=$OB/${id}_r3cand_audio-r3_final.mp4; A=$OB/${id}_r3final_fixed.mp4
  CR=$(python - $id $t0 <<'PY'
import sys, numpy as np, av
id_, t0 = sys.argv[1], float(sys.argv[2]); lm = np.load(f"/workspace/dub/artifacts/r3d/timing/{id_}_source_lm_F.npz")["lm"]
c = av.open([p for p in __import__("glob").glob(f"/workspace/dub/inputs/test_videos_raw/{id_}.*")][0]); fps = float(c.streams.video[0].average_rate); c.close()
k = int(t0 * fps); seg = lm[k:k + int(5 * fps)]; ov = seg[:, [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454], :2].reshape(-1, 2)
ov = ov[np.isfinite(ov).all(1)]; x0, y0 = ov.min(0) - 30; x1, y1 = ov.max(0) + 30; y0 = max(y0, ov[:, 1].mean() - 40)
w, h = int(x1 - x0) // 2 * 2, int(y1 - y0) // 2 * 2; print(f"{w}:{h}:{max(0, int(x0))}:{max(0, int(y0))}")
PY
)
  ffmpeg -loglevel error -y -ss $t0 -t $d -i $B -ss $t0 -t $d -i $A -filter_complex \
    "[0:v]split[a0][a1];[1:v]split[b0][b1];[a1]crop=$CR,scale=iw*2:ih*2,$(lab 'BEFORE 2x')[ac];[b1]crop=$CR,scale=iw*2:ih*2,$(lab 'AFTER 2x')[bc];[ac][bc]hstack=inputs=2,scale=-2:848,setsar=1[cc];[a0]$(lab BEFORE)[af];[b0]$(lab AFTER)[bf];[af][bf]hstack=inputs=2,setsar=1[ff];[ff][cc]hstack=inputs=2,setpts=2*PTS[v]" \
    -map "[v]" -an -c:v libx264 -crf 16 -pix_fmt yuv420p $OB/${id}_bg_BEFORE-AFTER_half.mp4 && echo "== bg clip $id ($CR)"
}
clip 04 20 25; clip 03 35 40; bgclip 04 17 22
