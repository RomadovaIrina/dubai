#!/bin/bash
# r18 quality (Irina 13.11: restore the 08.11 look + ONLY the chin / neck seam fix). Runs on a Docker GPU VM with the published image; this
# checkout's scripts are mounted over the image's. One KeySync layer per video (GPU), every candidate = a CPU re-paste of that layer, so the
# candidates differ only by the paste (KeySync is unseeded: two renders of the same input differ, PSNR floor ~44 dB).
#   r18.sh dub <id> <08.11 mp4 in ref/> [dir with the accepted dubbed_16k.wav]   the accepted mix, stream-copied out of the 08.11 file (never
#                                                  re-encoded) + the voice KeySync hears: the given wav, else Spleeter vocals of that mix
#   r18.sh job <id> <input in in/> <route: off | reports/.../x.json | /r18/...json> [K=V ...]   the e2e job with DUB_AUDIO_FROM + the 08.11 seam rule
#   r18.sh repaste <id> <variant> [K=V ...]        CPU re-paste (KS_POST_REUSE=1: the finished post layer) -> out/<id>/<variant>.mp4
#   r18.sh sh <command ...>                        a shell command inside the image (CPU; /r18 = the r18 root, /data = the jobs)
# Layout under R18_ROOT (/root/r18): code/ (this checkout) in/ ref/ dubs/<id>/ data/jobs/<id>/ out/<id>/ log/
set -uo pipefail
B=${R18_ROOT:-/root/r18}; CODE=${R18_CODE:-$B/code}; IMG=${R18_IMG:-ghcr.io/romadovairina/dubai:latest}; VOL=${R18_VOL:-dabai-models}
mkdir -p $B/in $B/ref $B/dubs $B/data $B/out $B/log
DR=(docker run --rm --shm-size 16g -v $VOL:/workspace/dub/dubai/models -v $CODE/scripts:/workspace/dub/dubai/scripts -v $CODE/reports:/workspace/dub/dubai/reports:ro -v $B:/r18 -v $B/data:/data)
SEAM0811="--ratio-full 2.0"   # ks_seam: the 08.11 step rule (ratio > 2 -> full smoothing of every 13-frame boundary)
log(){ echo "[$(date -u +%T)] $*" | tee -a $B/log/r18.log; }
cmd=${1:?dub | job | repaste | sh}; shift
case $cmd in
dub) id=$1; ref=$2; vd=${3:-}; D=$B/dubs/$id; mkdir -p $D
  "${DR[@]}" --entrypoint ffmpeg $IMG -v error -y -i /r18/ref/$ref -vn -c:a copy /r18/dubs/$id/final_mix_aac.m4a || { log "DUB $id: mix extract failed"; exit 1; }
  if [ -n "$vd" ]; then cp $vd/dubbed_16k.wav $D/dubbed_16k.wav; for m in $vd/*.manifest.json; do [ -f "$m" ] && cp "$m" $D/; done; src="given $vd"
  else   # no voice stem of the 08.11 take survives: Spleeter 2stems (MIT, the pipeline's own separator) vocals of the accepted mix, 16 kHz mono
    "${DR[@]}" --gpus all --entrypoint bash $IMG -c "set -e; cd /workspace/dub/dubai; D=/r18/dubs/$id
      ffmpeg -v error -y -i \$D/final_mix_aac.m4a -ar 44100 -ac 2 -c:a pcm_s16le \$D/mix44k.wav
      /venv/dabai/bin/python scripts/pilot/spleeter_torch.py separate --weights models/spleeter/2stems_tf_vars.npz --wav \$D/mix44k.wav --bg \$D/bg44k.wav --vocals \$D/voc44k.wav
      ffmpeg -v error -y -i \$D/voc44k.wav -ar 16000 -ac 1 -c:a pcm_s16le \$D/dubbed_16k.wav; rm -f \$D/mix44k.wav \$D/bg44k.wav" > $B/log/dub_$id.log 2>&1 || { log "DUB $id: spleeter failed (log/dub_$id.log)"; exit 1; }
    src="spleeter vocals of the 08.11 mix"; fi
  log "DUB $id: mix from $ref ($(stat -c %s $D/final_mix_aac.m4a) B), voice $src ($(stat -c %s $D/dubbed_16k.wav) B)";;
job) id=$1; inp=$2; route=$3; shift 3
  [ -s $B/in/$inp ] && [ -s $B/dubs/$id/dubbed_16k.wav ] && [ -s $B/dubs/$id/final_mix_aac.m4a ] || { log "JOB $id: input or dub missing"; exit 1; }
  E="\"DUB_AUDIO_FROM\": \"/r18/dubs/$id\", \"KS_SEAM_ARGS\": \"${KS_SEAM_ARGS-$SEAM0811}\""
  case $route in off) E="$E, \"DUB_ROUTE\": \"off\"";; auto) ;; /*) E="$E, \"DUB_ROUTE_JSON\": \"$route\"";; *) E="$E, \"DUB_ROUTE_JSON\": \"/workspace/dub/dubai/$route\"";; esac
  for kv in "$@"; do E="$E, \"${kv%%=*}\": \"${kv#*=}\""; done
  printf '{"job_id": "%s", "input": "/r18/in/%s", "out_dir": "/data/jobs/%s", "target_lang": "en", "timeout_s": 21600, "output": {"keep_intermediate": true}, "debug_env": {%s}}\n' $id "$inp" $id "$E" > $B/log/$id.job.json
  rm -rf $B/data/jobs/$id; t0=$(date +%s); log "JOB $id start: $(cat $B/log/$id.job.json | cut -c1-400)"
  "${DR[@]}" --gpus all $IMG job /r18/log/$id.job.json > $B/log/$id.run.log 2>&1; rc=$?
  st=$(grep -o '"status": "[A-Z_]*"' $B/data/jobs/$id/status.json 2>/dev/null | head -1)
  log "JOB $id rc=$rc $(( $(date +%s) - t0 )) s $st $(grep -E "^== $id (SUCCESS|PARTIAL_SUCCESS|ERROR)" $B/data/jobs/$id/job.log 2>/dev/null | cut -c1-300)";;
repaste) id=$1; var=$2; shift 2; EV=(); for kv in "$@"; do EV+=(-e "$kv"); done; t0=$(date +%s)
  "${DR[@]}" -e KS_POST_REUSE=${KS_POST_REUSE:-1} -e "KS_SEAM_ARGS=${KS_SEAM_ARGS-$SEAM0811}" "${EV[@]}" $IMG bash /workspace/dub/dubai/scripts/r18/repaste_in.sh $id $var > $B/log/${id}_$var.repaste.log 2>&1; rc=$?
  log "REPASTE $id $var [$*] rc=$rc $(( $(date +%s) - t0 )) s $(grep -E "^REPASTE_DONE|^REPASTE .*failed" $B/log/${id}_$var.repaste.log | tail -1 | cut -c1-200)";;
sh) "${DR[@]}" ${R18_GPU:+--gpus all} --entrypoint bash $IMG -c "cd /workspace/dub/dubai && $*";;
*) echo "unknown command $cmd"; exit 2;;
esac
