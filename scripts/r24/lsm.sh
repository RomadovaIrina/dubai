#!/bin/bash
# (r24 measuring tools: lsm.sh runs inside the image on the VM layout /r18; see the r24 handoff)
# lsm.sh <name> <src in /r18/in> <final mp4 path under /r18>   (inside the image): upright source, landmarks of both (cached), lsmove.py
set -u; cd /workspace/dub/dubai; O=/r18/out/mv; mkdir -p $O; nm=$1; src=$2; fin=$3; b=${src%.mp4}
[ -s $O/${b}_up.mp4 ] || ffmpeg -v error -y -i /r18/in/$src -an -c:v libx264 -crf 10 -pix_fmt yuv420p $O/${b}_up.mp4
[ -s $O/${b}_up.npz ] || /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $O/${b}_up.mp4 --out $O/${b}_up.npz
ffmpeg -v error -y -i $fin -an -c:v copy $O/${nm}_v.mp4
[ -s $O/${nm}.npz ] || /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $O/${nm}_v.mp4 --out $O/${nm}.npz
/venv/dabai/bin/python /r18/lsmove.py --src $O/${b}_up.mp4 --fin $O/${nm}_v.mp4 --lm-src $O/${b}_up.npz --lm-fin $O/${nm}.npz --out $O/${nm}.json > /dev/null; python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(\"LSMOVE\", {k: r[k] for k in (\"jitter_in_patch\",\"jitter_outside(noise_floor)\",\"slide_in_patch\",\"flicker\",\"lip_twitch_final\",\"lip_twitch_source\",\"opening_final_p50_p90\",\"opening_source_p50_p90\")}, {k: (v[\"max_over_median\"], v[\"argmax\"]) for k, v in r[\"seams\"].items()}, r[\"by_quartile(low->high)\"], \"WJ\", r[\"worst_jitter_1s\"], \"WF\", r[\"worst_flicker_1s\"], \"WT\", r[\"worst_lip_twitch_1s\"])" $O/${nm}.json
