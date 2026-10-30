#!/bin/bash
# Round 6 F (blocker #0): score a VERSION (one mp4 per video) with the whole-video face twitch detector against the raw sources.
#   bash scripts/r6/r6_twitch.sh <label> <dir> <suffix> [ids]
#     <dir>/<id><suffix>.mp4 (e.g. /workspace/dub/reference/v4 _v4) -> /workspace/dub/artifacts/r6f/twitch/<label>/<id>.twitch.json
#   env: MAN=<printf pattern with %s = id> manifest path (annotates events with LS joints; references have none);
#        LM_DIR (source landmarks from r5_srclm.py, default /workspace/dub/artifacts/r6f/lm); THR (default 1.0); SHEET=1 -> top-20 sheet + half-speed clips
# CPU only (DIS flow): OMP capped at 8, videos sequential (vast CPU quota). ~16 s per video-minute... per video.
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
LABEL=$1; DIR=$2; SUF=$3; shift 3; IDS=${*:-01 02 03 04 05}
LM_DIR=${LM_DIR:-/workspace/dub/artifacts/r6f/lm}; O=/workspace/dub/artifacts/r6f/twitch/$LABEL; mkdir -p $O
for id in $IDS; do
  v=$DIR/${id}${SUF}.mp4; src=$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/$id.* | head -1); lm=$LM_DIR/${id}_src_lm.npz
  [ -f "$v" ] || { echo "no video $v"; continue; }; [ -f "$lm" ] || { echo "no landmarks $lm"; continue; }
  m=(); [ -n "${MAN:-}" ] && { mf=$(printf "$MAN" $id); [ -f "$mf" ] && m=(--manifest "$mf"); }
  s=$(date +%s)
  python scripts/r5/r5_twitch.py score --video $v --src $src --lm $lm "${m[@]}" --thr ${THR:-1.0} --out $O/${id}.twitch.json > $O/${id}.log 2>&1 \
    || { echo "FAIL $LABEL $id"; tail -3 $O/${id}.log; continue; }
  echo "== $LABEL $id $(( $(date +%s) - s )) s $(tail -1 $O/${id}.log | python -c 'import json,sys; d=json.load(sys.stdin); print({k:d[k] for k in ("twitch_sum","n_events","top20_mean","p99","n_events_ge2")})')"
  [ -n "${SHEET:-}" ] && python scripts/r5/r5_twitch.py sheet --twitch $O/${id}.twitch.json --lm $lm --top 20 --outdir $O/sheets --stem ${id}_$LABEL --clips > /dev/null 2>&1
done
echo R6_TWITCH_DONE $LABEL
