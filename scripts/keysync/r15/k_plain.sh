#!/bin/bash
# r15 K: the plain-paste full video of one KeySync render: bash k_plain.sh <q18|m03|qo3> <render> <name> [ks_plain args]
id=$1; R=$2; NAME=$3; shift 3; K=/workspace/dub/keysync; cd $K
case $id in q18) v=m18;; m03) v=m03;; qo3) v=old03;; esac
G=/workspace/dub/artifacts/r4m/r9g_$id; MST=$G/work_${id}_r9g_$id/video/master_25fps.mp4; O=$K/plain/$NAME; mkdir -p $O /workspace/dub/outbox/r15_k/full
SH=$(python3 -c "import json; print(-int(json.load(open(\"$K/fast/fast2c/eval_$id/raw_sync.json\"))[\"global\"][\"av_offset\"]))")
OTH=""; [ $id = m03 ] && OTH="--others $K/lm/m03_faces2.npz"
KS_CMS=1 /venv/dabai/bin/python ks_plain.py --ks $R --master $MST --box data/box_$id.json --lm $G/timing/${id}_source_lm_F.npz $OTH --shift $SH --audio /workspace/dub/dubaudio/$v/final_mix_aac.m4a --out $O/${v}.mp4 "$@" > $O/${v}.log 2>&1 || { echo "FAIL $v $NAME"; tail -3 $O/${v}.log; exit 1; }
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $O/${v}.mp4); BR=$(python3 -c "print(int(min(8000, 25e6 * 8 / $D * 0.92 / 1000 - 160)))")
ffmpeg -v error -y -i $O/${v}.mp4 -c:v libx264 -preset slow -crf 18 -maxrate ${BR}k -bufsize $((2 * BR))k -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv -movflags +faststart -c:a copy /workspace/dub/outbox/r15_k/full/${v}_keysync_${NAME}.mp4
grep -E "colour|plain paste:" $O/${v}.log; ls -la /workspace/dub/outbox/r15_k/full/${v}_keysync_${NAME}.mp4; echo "PLAIN_DONE $v $NAME"
