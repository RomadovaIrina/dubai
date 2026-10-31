#!/bin/bash
# r10: face_only.sh's gate rows for a post-processed final (artifacts/r4m/r9f_<id><v>) whose generator run is r9g_<id><gv> -> /workspace/dub/r9/face/<id><v>/
#   bash gate_post.sh <id> <v> <gv>
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=on
id=$1; V=$2; GV=$3; A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; AT=r9_$id; GT=r9g_$id$GV; FT=r9f_$id$V
OUT=/workspace/dub/r9/face/$id$V; L=$OUT/logs; mkdir -p $L; export DUB_INPUTS=/workspace/dub/r9/runs/$id/work/in; SRC=$(ls $DUB_INPUTS/$id.* | head -1); VID=$B/$FT/${id}_$FT.mp4
{ python scripts/r4/r4_gate.py score --label $FT --dir $B/$FT --ids $id --force > $L/face_score.log 2>&1 || echo "WARN face score"
  MAN=$B/$FT/%s_$FT.manifest.json bash scripts/r6/r6_twitch.sh $FT $B/$FT _$FT $id | grep "=="
  python scripts/r9/occl_scan.py --src $SRC --out $VID --lm $B/$GT/timing/${id}_source_lm_F.npz --sheet $OUT/occlusion_scan.jpg --json $OUT/occlusion.json | tail -1
  mkdir -p /workspace/dub/r9/diag $B/r9g_$id/work_${id}_r9g_$id; [ -e $B/r9g_$id/work_${id}_r9g_$id/video ] || ln -sfn $B/$GT/work_${id}_$GT/video $B/r9g_$id/work_${id}_r9g_$id/video
  L3=$(ls $B/$GT/work_${id}_$GT/${id}_${GT}_25fps_ls_cfd_ia04_srt1_rep*.mp4 2>/dev/null | grep -v "\.mediapipe" | tail -1)
  CUDA_VISIBLE_DEVICES= python scripts/r9/artefacts.py $id --tag-suffix "$V" --gen $GT ${L3:+--l3 $L3} --out $OUT/art.json | tail -1
  bash scripts/r7/r7_frame_gate.sh $B/$FT:$FT $id | grep "^=="
  python scripts/r7/r7_baseline.py freeze --tag $FT --audio $A/$AT --ids $id --out $OUT/face_gate.json | tail -2; } > $L/metrics.log 2>&1
echo "GATE_POST_DONE $id$V"
