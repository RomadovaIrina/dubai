#!/bin/bash
# r17: the proof KS_PROBE_BY_CHECK needs before it may be a default (coordinator 12.11): on the clips whose r16 probe did NOT give 40 ms (old03: 0,
# q02: low conf -> 0, q03: 40 but 1 frame late) and the 40 ms ones (m03, m18), the probe path vs probe-by-check, each a fresh job (the dub is
# deterministic). Same = the same final delay per face AND the same off-screen faces; the frames within the run-to-run floor.
#   bash infra/docker/probe_proof.sh <image> <clips dir: m03.mp4 m18.mp4 old03.mp4 q02.mp4 q03.mp4> <out dir> [code snapshot dir]
set -uo pipefail
IMG=${1:?image}; CL=$(readlink -f "${2:?clips}"); O=$(mkdir -p "${3:?out}" && readlink -f "$3"); C=${4:-}; VOL=${VOL:-dabai-models}
log(){ echo "[$(date -u +%T)] $*" | tee -a $O/progress.log; }
MNT=(); [ -n "$C" ] && MNT=(-v $C/scripts:/workspace/dub/dubai/scripts:ro -v $C/infra:/workspace/dub/dubai/infra:ro)
run(){   # run <name> <clip> <pbc 0/1>
  local N=$1; mkdir -p $O/$N
  printf '{"job_id": "%s", "input": "/in.mp4", "out_dir": "/data/jobs/%s", "target_lang": "en", "output": {"keep_intermediate": true}, "debug_env": {"KS_PROBE_BY_CHECK": "%s"}}\n' $N $N $3 > $O/$N/job.json
  local t0=$(date +%s)
  docker run --rm --gpus all --shm-size 16g "${MNT[@]}" -v $VOL:/workspace/dub/dubai/models -v $O/$N:/data -v $O/$N/job.json:/job.json:ro -v $2:/in.mp4:ro $IMG job /job.json > $O/$N/run.log 2>&1
  echo $(( $(date +%s) - t0 )) > $O/$N/wall.txt
}
delays(){   # delays <name>: "f0=40 f1=0 | off: f2," = the LAST delay each non-off-screen face was rendered with (a re-render logs twice) + the off-screen faces
  local F=$(find $O/$1 -name face_keysync.log | head -1)
  local OFF=$(grep -oE 'KS_FLAG ks_offscreen_voice f[0-9]+' $F | awk '{print $3}' | sort -u | tr '
' ',')
  echo "$(grep -oE 'KS_CHECK f[0-9]+ delay [-0-9]+ ms' $F | awk -v off=",$OFF" '{ if (index(off, "," $2 ",") == 0) d[$2] = $4 } END { for (f in d) print f "=" d[f] }' | sort | tr '
' ' ')| off: $OFF"
}
cp /root/dubai/scripts/keysync/r16/tz/out_compare.py $O/out_compare.py 2>/dev/null || cp "$(dirname "$0")/../../scripts/keysync/r16/tz/out_compare.py" $O/out_compare.py
ALL=1
for c in m03 m18 old03 q02 q03; do
  [ -s $CL/$c.mp4 ] || { log "$c: no clip, skipped"; continue; }
  run ${c}_probe $CL/$c.mp4 0; run ${c}_pbc $CL/$c.mp4 1
  A=$(delays ${c}_probe); B=$(delays ${c}_pbc)
  RR=$(grep -c "(re-rendered)" $(find $O/${c}_pbc -name face_keysync.log | head -1) 2>/dev/null)
  V=$(docker run --rm -v $O:$O --entrypoint /venv/dabai/bin/python $IMG $O/out_compare.py --a $O/${c}_probe/jobs/${c}_probe/${c}_probe.mp4 --b $O/${c}_pbc/jobs/${c}_pbc/${c}_pbc.mp4 --src $CL/$c.mp4 2>/dev/null | grep OUT_COMPARE | cut -c13-)
  SAME=$([ "$(echo $A | tr ' ' '\n' | sort | tr '\n' ' ')" = "$(echo $B | tr ' ' '\n' | sort | tr '\n' ' ')" ] && echo yes || echo NO); [ $SAME = yes ] || ALL=0
  log "$c: probe [$A] $(cat $O/${c}_probe/wall.txt) s | by-check [$B] $(cat $O/${c}_pbc/wall.txt) s, re-rendered $RR | same delays: $SAME | frames $V"
done
log "PROBE_PROOF $([ $ALL = 1 ] && echo 'PASS: every clip ends with the probe delay -> KS_PROBE_BY_CHECK may be a default' || echo 'FAIL: keep the probe')"
