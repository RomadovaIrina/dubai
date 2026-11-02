#!/bin/bash
# r11 SPEED (REPORT r10-speed §8): dynamic UNet compile warmed on DIFFERENT videos -> the UNSEEN 4-video queue -> the no-compile reference
# -> the m03 face gate (_ucd vs _b). One GPU job at a time. Markers: R11_<STEP>_DONE in this log.
set -u
until grep -q R9_MODELS_DONE /workspace/dub/r9_models.log; do sleep 30; done
echo "R11_MODELS_DONE $(date +%T)"
cd /workspace/dub/dubai; R=/workspace/dub/r10; mkdir -p $R
M=/workspace/dub/inputs/mamakris; O=/workspace/dub/inputs/test_videos_raw
printf "01 $O/01.MP4\n02 $O/02.mp4\nm19 $M/story-1774250832175-746471562.mp4\n" > $R/q3warm.txt
printf "m03 $M/story-1774249919482-367803614.MOV\n03 $O/03.mp4\nm15 $M/story-1774250573471-666527943.MP4\nm18 $M/story-1774250817106-616905094.mp4\n" > $R/q4u.txt
DYN="LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1"
W() { /venv/dabai/bin/python scripts/r10/worker.py --queue $R/$1 --out $R/$2 --inflight 2 --gen-worker --env E2E_METRICS=off ${3:-} 2>&1 | grep -E "QUEUE_DONE|done "; grep "gen_worker\] job" $R/$2/gen_spool/worker.log; }
W q3warm.txt q3warm_dyn "$DYN"; echo "R11_WARM_DONE $(date +%T)"
W q4u.txt q4u_dyn "$DYN"; echo "R11_Q4U_DONE $(date +%T)"
W q4u.txt q4u_ref; echo "R11_REF_DONE $(date +%T)"
mkdir -p /workspace/dub/r9/runs/m03/work; ln -sfn $R/q4u_ref/m03/work/in /workspace/dub/r9/runs/m03/work/in
export GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12 LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
V=_b bash scripts/r9/face_only.sh m03 2>&1 | grep "== timing\|metrics done"
LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1 V=_ucd bash scripts/r9/face_only.sh m03 2>&1 | grep "== timing\|metrics done"
for v in m03_b m03_ucd; do until grep -q FRAME_GATE_DONE /workspace/dub/r9/face/$v/logs/frame_gate.log 2>/dev/null; do sleep 20; done; done
python3 scripts/r9/gate_cmp.py --base /workspace/dub/r9/face/m03_b --cand /workspace/dub/r9/face/m03_ucd 2>&1 | tail -3
echo "R11_GATE_DONE $(date +%T)"
