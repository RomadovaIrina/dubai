#!/bin/bash
# r15 Z step 1, ALL stages incl. the crop space, on the q2d arm (production q2 + LS_DUMP_DIR): a source | a25 master | b input crops (+ the
# affine placement) | c generated crops | d raw layer pasted | e1 CodeFormer | e2 CF-detail | e3 SR | f repair | g final.
# Two passes: VIDEO-mode landmarks (colour + LF slide; m03 only this: two faces confuse per-frame detection) and IMAGE-mode (HF geometry).
#   bash scripts/r15/zstage.sh <old03|m03|m18>
set -u
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export OMP_NUM_THREADS=8
id=$1; X=/workspace/dub/r15w/eval; Z=/workspace/dub/r15w/zstage; mkdir -p $Z
G=r9g_q2_${id}_q2d; W=/workspace/dub/artifacts/r4m/$G/work_q2_${id}_$G; P=$W/q2_${id}_${G}_25fps; SRC=$(ls $X/q2_$id/e2e/work/in/q2_$id.* | head -1)
ST="a25_master=$W/video/master_25fps.mp4 d_raw_paste=${P}_ls_pre.mp4 e1_codeformer=${P}_ls_raw.mp4 e2_cf_detail=${P}_ls_cfd_ia04.mp4 e3_sr=${P}_ls_cfd_ia04_srt1.mp4 f_repair=${P}_ls_cfd_ia04_srt1_rep2s.mp4 g_final=$X/q2d_$id/q2d_$id.mp4"
python scripts/r15/stage_jitter.py --lm video --src $SRC --stage $ST --crops /workspace/dub/r15w/zdump/$id --master-lm $W/video/master_mediapipe.npz \
  --json $Z/zstage_video_$id.json 2>&1 | grep -vE "^W0000|^I0000|^INFO|^WARNING" | tee $Z/zstage_video_$id.txt
[ $id = m03 ] || python scripts/r15/stage_jitter.py --lm img --no-colour --src $SRC --stage $ST --crops /workspace/dub/r15w/zdump/$id \
  --json $Z/zstage_img_$id.json 2>&1 | grep -vE "^W0000|^I0000|^INFO|^WARNING" | tee $Z/zstage_img_$id.txt
echo "ZSTAGE_DONE $id"
