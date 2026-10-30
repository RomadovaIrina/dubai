#!/bin/bash
# IMG_6185 with LatentSync ref fixed (lead 0.5) + fx, on the SAME dub as the r9fx run (only the face differs), then both mixes + a 2-panel WATCH
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=off DUB_INPUTS=/workspace/dub/r9/runs/m6185/work/in
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; D=/workspace/dub/r9/diag; O=/workspace/dub/outbox/r9_e2e; id=m6185; G=r9g_m6185fixedL05; F=r9f_m6185fixedL05fx
t0=$(date +%s)
AUDIO=$A/r9_$id TAG=$G SKIP_SCORE=1 VIDEO_EXTRA="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16 --ref-mode fixed --audio-lead 0.5" bash scripts/r7/r7_e2e.sh $id > $D/m6185f.gen.log 2>&1
t1=$(date +%s)
SKIP_SCORE=1 TAG=$F SRC_TAG=$G CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$(cat reports/experiments/r8-face/fb7.args)" REP=1 bash scripts/r7/r7_m7_comp.sh $id > $D/m6185f.comp.log 2>&1
t2=$(date +%s); echo "== gen $((t1 - t0)) s comp $((t2 - t1)) s $(grep -h layer_repair $D/m6185f.comp.log | cut -c1-200)"
V=$B/$F/${id}_$F.mp4; [ -f $V ] || { echo FAIL; exit 1; }
for v in demucs:$A/r9_${id}_htd/$id/final_mix_aac.m4a spleeter:$A/r9_$id/$id/final_mix_aac.m4a; do ffmpeg -loglevel error -y -i $V -i ${v#*:} -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart $O/${id}_DabAI_r9fixedfx_${v%%:*}.mp4; done
ffmpeg -loglevel error -y -i /workspace/dub/r9/runs/$id/${id}_DabAI_r9_demucs.mp4 -i $O/${id}_DabAI_r9fixedfx_demucs.mp4 -filter_complex "[0:v]scale=-2:960,drawtext=text='r9 + fix':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6[a];[1:v]scale=-2:960,drawtext=text='ref fixed + fix':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6[b];[a][b]hstack" -map 0:a:0 -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a copy $O/watch_mouth/${id}_WATCH_r9fix_vs_fixedfix.mp4
ls -la $O/${id}_DabAI_r9fixedfx_*.mp4 $O/watch_mouth/${id}_WATCH_r9fix_vs_fixedfix.mp4; echo JOB6185F_DONE
