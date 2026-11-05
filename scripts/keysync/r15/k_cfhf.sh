#!/bin/bash
# r15 K: full-frame, full-length KeySync finals through the PRODUCTION r13 cfhf chain (scripts/keysync/ks_final.sh, the r13_s8final.sh env):
# KS_CMS=1, ks_layer --end-fade 12, ks_fix --guard 3 --chin tex --skin-cm 12 --cm-robust 1 --guard-fade 10, alpha 0.8 tsmooth, KS_HF 3 / KS_SHARP 0.3,
# the q2 dub mix, <= 25 MB. r15-k has no E2E artefacts, so the layout ks_final.sh reads is rebuilt here from the source + the q2 dub:
#   artifacts/r4a/r9_<id>_htd/<id>/{dubbed_16k.wav, final_mix_aac.m4a}; artifacts/r4m/r9g_<id>/work_<id>_r9g_<id>/video/master_25fps.mp4;
#   artifacts/r4m/r9g_<id>/timing/<id>_source_lm_F.npz (the source mediapipe landmarks + j = the source frame of each 25 fps master frame).
# usage: bash k_cfhf.sh <id: q18|m03|qo3> <arm name> <KeySync render 512^2 25 fps>
id=$1; NAME=$2; R=$3; K=/workspace/dub/keysync; cd $K
case $id in q18) SRC=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4; v=m18;; qo3) SRC=/workspace/dub/inputs/test_videos_raw/03.mp4; v=old03;; m03) SRC=/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV; v=m03;; esac
A=/workspace/dub/artifacts/r4a/r9_${id}_htd/$id; G=/workspace/dub/artifacts/r4m/r9g_$id; W=$G/work_${id}_r9g_$id/video
mkdir -p $A $W $G/timing
ln -sfn /workspace/dub/dubaudio/$v/dubbed_16k.wav $A/dubbed_16k.wav; ln -sfn /workspace/dub/dubaudio/$v/final_mix_aac.m4a $A/final_mix_aac.m4a
( flock 9
  [ -s $W/master_25fps.mp4 ] || ffmpeg -v error -y -i $SRC -an -vf fps=25 -c:v libx264 -crf 12 -pix_fmt yuv420p $W/master_25fps.mp4
  [ -s $G/timing/${id}_source_lm_F.npz ] || /venv/dabai/bin/python - $K/final/$id/src.mediapipe.npz $SRC $W/master_25fps.mp4 $G/timing/${id}_source_lm_F.npz <<'PY'
import sys, subprocess, numpy as np
lmf, src, mst, out = sys.argv[1:5]
fps = lambda v: (lambda n, d: float(n) / float(d))(*subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", v], capture_output=True, text=True).stdout.strip().split("/"))
n = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", mst], capture_output=True, text=True).stdout.strip())
d = np.load(lmf); lm = d["lm"]; fs = fps(src)
j = np.clip(np.round(np.arange(n) * fs / 25.0).astype(int), 0, len(lm) - 1)
np.savez(out, lm=lm[..., :2], j=j); print("source_lm_F", lm.shape, "master frames", n, "src fps", fs)
PY
) 9> /tmp/k_cfhf_$id.lock
O=$K/final/${id}_cfhf_$NAME; mkdir -p $O/ks/$id; ln -sfn $R $O/ks/$id/${id}_$id.mp4; cp $K/final/$id/src.mediapipe.npz $O/
OUT=/workspace/dub/outbox/r15_k/full; mkdir -p $OUT
KS_CMS=1 KS_HF=3 KS_SHARP=0.3 KS_REDO=1 KS_TSMOOTH=0.8 KS_FINALDIR=$O KS_OUTDIR=$OUT KS_LAYERARGS="--end-fade 12" KS_FIXARGS="--skin-cm 12 --cm-robust 1 --guard-fade 10" \
  bash ks_final.sh $id $SRC ${v}_$NAME > $O/final.log 2>&1 || { echo "FAIL $id $NAME"; tail -5 $O/final.log; exit 1; }
mv $OUT/${v}_${NAME}_KeySync.mp4 $OUT/${v}_keysync_$NAME.mp4 && ls -la $OUT/${v}_keysync_$NAME.mp4 && echo "CFHF_DONE $v $NAME"
