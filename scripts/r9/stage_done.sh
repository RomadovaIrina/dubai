#!/bin/bash
# r19 speed (long videos, output-identical): what can start the moment the audio stage's own-process TTS stage has exited. From then on the audio
# stage is its CPU chain (placement, remix: ~1 min on a 5-min video) and the GPU is idle and empty. Started by scripts/r8/r8_audio.sh (R9_ON_STAGE_DONE,
# set by e2e.sh). Both parts are optional accelerators: e2e.sh uses their results only when they are there and valid, else it works as before.
#   stage_done.sh <route_pre dir> <pipeline manifest (r4a)> <log dir> <device> [resident dir to resume]
#   1. the paused resident worker (dabai_job.py stopped it for this long video) comes back NOW instead of after the job: the KeySync stage is then
#      served by the warm daemon (no cold model load / compile warm-up inside the job);
#   2. the route's active-speaker pass (S3FD tracks + SyncNet on the SOURCE): it needs the master, the detector prep (both made during the audio
#      stage) and the diarization turns, which the pipeline manifest already has (e2e.sh checks them against the final manifest).
R=$(cd "$(dirname "$0")/../.." && pwd); cd $R; RPRE=$1; MAN=$2; L=$3; DEV=${4:-cuda}; RES=${5:-}
if [ -n "$RES" ]; then setsid bash scripts/job/resident.sh resume $RES > $L/resident_resume.log 2>&1 < /dev/null & fi
[ -n "$RPRE" ] && [ -d "$RPRE" ] && [ -s "$MAN" ] || exit 0
touch $RPRE/as_early.started
until [ -f $RPRE/finished ]; do sleep 0.5; done
. scripts/keysync/r16/gpulock.sh
if [ -f $RPRE/master.done ] && gpu_run python scripts/quality/r16q_active_speaker.py --video $RPRE/master_25fps.mp4 --manifest $MAN --out $RPRE/as_early.json --device $DEV \
     $([ -f $RPRE/as_prep/done ] && echo "--prep $RPRE/as_prep") > $L/as_early.log 2>&1 && [ -s $RPRE/as_early.json ]; then
  cp $MAN $RPRE/as_early.manifest.json; touch $RPRE/as_early.done
else touch $RPRE/as_early.failed; fi
