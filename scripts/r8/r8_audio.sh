#!/bin/bash
# Round 8 AUDIO (audio-r8-v1): the dub speaks whenever the source speaks, on and off screen (management on final_r7: 02 / 05 "keeps
# disappearing", the silent "about to vomit" mouth), and sentence starts follow the speaker (user on 04: "strange starts and pauses").
# On a base of r6/r7 takes (REUSE tag with mix_src/<id> = work dirs on the pre-placement timeline; v3a1 rebuilt by scripts/r8/rebase.py):
#   1. retake.py --cover-all on       units whose source speech outlasts the take (on OR off screen) re-translated longer (Qwen 14B duration-aware,
#                                     same judge) and re-taken (normal / deliberate pace); every other unit bit-identical
#   2. burst_align.py --cover-all on  the take aligned to the source bursts, tempo R8_TMIN..R8_TMAX (0.9-1.1), where it beats the placement
#   3. the r7 chain with lip_place --pace on --cover-all on (onset anchor, sentence gap floor, gate-aware blocks, dictation splits), hush, remix
#   then scripts/r8/coverage.py + pacing.py rows per video (<tag>/cov/).
#     bash scripts/r8/r8_audio.sh <tag> [ids]                         one command from the raw input (defaults = audio-r8-v4)
#     REUSE=v3a1r R7_PREV=r8repro2 bash scripts/r8/r8_audio.sh <tag>  on the rebuilt v3a1 takes (how audio-r8-v4 was built); R8_ALIGN / R8_CLAUSE / ... = off to skip a step
set -u
DUB_ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd $DUB_ROOT; R=$DUB_ROOT; . scripts/env.sh >/dev/null; DUB_ROOT=$R; export DUB_ROOT
export DUB_TTS_BATCH=${DUB_TTS_BATCH:-1}   # batched Chatterbox (scripts/pilot/t3_batch.py): the default since the user accepted the r8b5 AB (DUB_TTS_BATCH=0 = one by one)
TAG=$1; shift; IDS=${*:-01 02 03 04 05}; A=/workspace/dub/artifacts/r4a; S=$A/${TAG}src; mkdir -p $S/mix_src $S/src16
s=$(date +%s)
RT_ARGS="${R8_RETAKE_ARGS---short-s 0.3 --budget-scales 1.15,1.3,1.5,1.75 --cap-scale 1.2 --min-piece 0.2 --max-tts 4 --natural-cuts on --fillers on}"
BA_ARGS="${R8_ALIGN_ARGS---min-piece 0.2 --sil-ratio 1.5 --natural-cuts on}"
LIPA="--max-shift 2.0 --grow-unit 1.5 --grow 0.5 --pace on --cover-all on ${R8_LIP_ARGS---dsplit off}"   # the r7 chain's LIP_ARGS / REMIX_ARGS (below; also the in-process chain job)
REMA="--src-duck-db -6 --unit-level off --gap-bed-cap-db 10 --gap-bed-cap-scope edges --gap-clean on --gap-clean-nr-ns 0.7 --dub-gate on --gap-clean-floor ${R8_FLOOR:-3}"
[ "${R9_CHAIN_INPROC:-off}" = on ] && [ "${R9_ONEPROC:-off}" != on ] && { echo "R9_CHAIN_INPROC needs R9_ONEPROC=on"; exit 1; }
[ "${R9_ONEPROC:-off}" = on ] && { [ -z "${REUSE:-}" ] && [ "${R8_ONEPROC:-on}" = on ] || { echo "R9_ONEPROC needs the raw-input path and R8_ONEPROC=on"; exit 1; }; R9_MP_BG=on; }
mkjobs() {   # the retake -> clause takes -> burst_align jobs (stdin: "id|work dir|faith arg"); $1 = one json for all ids, or a dir: one oneproc_jobs_<id>.json each (R9_ONEPROC)
  S=$S A=$A RT="$RT_ARGS" BA="$BA_ARGS" CA="${R8_CLAUSE_ARGS:-}" CL="${R8_CLAUSE:-on}" AL="${R8_ALIGN:-on}" TMIN=${R8_TMIN:-0.9} TMAX=${R8_TMAX:-1.1} OUT=$1 python3 -c "
import json, os, shlex, sys
S, A = os.environ['S'], os.environ['A']; jobs = []
for ln in sys.stdin:
    id_, w, f = ln.rstrip('\n').split('|'); fc = f'{A}/sep/{id_}/source_mediapipe.npz'; s16 = f'{S}/src16/{id_}_16k.wav'
    st = [['retake', ['--work', w, '--out', f'{S}/rt/{id_}', '--face', fc, *shlex.split(f), '--src16', s16, '--cover-all', 'on', *shlex.split(os.environ['RT'])]]]
    wk = f'{S}/rt/{id_}'
    if os.environ['CL'] == 'on': st.append(['clause', ['--work', wk, '--out', f'{S}/ct/{id_}', '--src16', s16, '--face', fc, *shlex.split(os.environ['CA'])]]); wk = f'{S}/ct/{id_}'
    if os.environ['AL'] == 'on': st.append(['burst_align', ['--work', wk, '--out', f'{S}/ba/{id_}', '--face', fc, '--cover-all', 'on', '--tmin', os.environ['TMIN'], '--tmax', os.environ['TMAX'], *shlex.split(os.environ['BA'])]])
    jobs.append({'id': id_, 'steps': st, 'logs': {'retake': f'{S}/rt_{id_}.log', 'clause': f'{S}/ct_{id_}.log', 'burst_align': f'{S}/ba_{id_}.log'}})
o = os.environ['OUT']
if os.path.isdir(o):
    for j in jobs: json.dump({'wait': [f'{A}/sep/{j[\"id\"]}/source_mediapipe.npz'], 'jobs': [j]}, open(f'{o}/oneproc_jobs_{j[\"id\"]}.json', 'w'), indent=1)
else: json.dump(jobs, open(o, 'w'), indent=1)"
}
src16() { ffmpeg -loglevel error -y -i "$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/$1.* | head -1)" -vn -ac 1 -ar 16000 -c:a pcm_s16le $S/src16/${1}_16k.wav; }
if [ -z "${REUSE:-}" ]; then   # from the raw input (one-command E2E): the r6 pipeline's takes only (ASR, diarization, Qwen 14B, Chatterbox), no mix / metrics / judge
  BASE=${TAG}pre; MPP=()
  if [ "${R9_MP_BG:-off}" = on ]; then   # r9 speed: the source face track (MediaPipe, CPU only, ~1 s per video-second) runs while the GPU pipeline runs (same file; <npz>.done when written)
    for id in $IDS; do mkdir -p $A/sep/$id; npz=$A/sep/$id/source_mediapipe.npz; [ -f $npz ] && { touch $npz.done; continue; }
      ( python -c "import sys; sys.path.insert(0,'scripts/pilot'); import clean_face_stack as c, pathlib; c.run_mediapipe(pathlib.Path('$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/$id.* | head -1)'), pathlib.Path('$npz'))"; touch $npz.done ) > /dev/null 2>&1 & MPP+=($!); done
  fi
  if [ "${R9_ONEPROC:-off}" = on ]; then   # r9 speed L: the stage runs inside the pipeline process (scripts/r9/audio_proc.py), its jobs written now (the work dir is known)
    for id in $IDS; do src16 $id; echo "$id|$A/$BASE/runs/$id/work_$id|--faith judge"; done | mkjobs $S
    export R9_PIPE_PY=scripts/r9/audio_proc.py R9_AFTER_JOBS=$S
    if [ "${R9_CHAIN_INPROC:-off}" = on ]; then   # r9 speed C: the chain job (the same argv / env the bash chain gets: r7 -> r6 -> r5 -> r4_audio.sh, PAR=1)
      for id in $IDS; do rm -f $S/chain_done_$id
        ID_=$id S_=$S A_=$A TAG_=$TAG LIPA="$LIPA" REMA="$REMA" SRC_="$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/$id.* | head -1)" python3 -c "
import json, os
e = os.environ; th = str(max(2, 24 // int(e.get('PAR', '1') or 1)))
json.dump({'S': e['S_'], 'A': e['A_'], 'id': e['ID_'], 'tag': e['TAG_'], 'close': e.get('R8_CLOSE', 'on') == 'on', 'soften': e.get('R8_SOFTEN', 'on') == 'on',
           'tmin': e.get('R8_TMIN', '0.9'), 'lip_args': e['LIPA'], 'remix_args': e['REMA'], 'src': e['SRC_'],
           'env': {'DUB_CUT_SNAP': '1', 'DUB_EQPOWER_FADE': '1', 'DUB_CLOSE_INSIDE_DB': e.get('DUB_CLOSE_INSIDE_DB', '30'), 'DUB_THREADS': th, 'OMP_NUM_THREADS': th,
                   'MKL_NUM_THREADS': th, 'REUSE': e['TAG_'] + 'src', 'RETAKE': 'off', 'FINAL_SOFTEN': e.get('FINAL_SOFTEN', 'on'), 'HUSH_PLACED': 'on', 'LIP_PLACE': 'on'}},
          open(e['S_'] + '/chain_' + e['ID_'] + '.json', 'w'), indent=1)"; done
    fi
    python3 -c "import json, os; json.dump(dict(os.environ), open('$S/stage_env.json', 'w'))"; export R9_STAGE_ENV=$S/stage_env.json   # the stage runs with THIS env (as when r8_tts_stage.py is started from here), not the one r4_audio.sh gives the pipeline (DUB_CUT_SNAP ...)
  fi
  t0p=$(date +%s.%N)
  R4_PIPELINE_ONLY=1 R5_FAITH=off PAR=${PAR:-1} R6_EXTRA="${R8_PIPE_EXTRA:-}" bash scripts/r6/r6_audio.sh $BASE $IDS || exit 1
  unset R9_PIPE_PY R9_AFTER_JOBS
  for p in "${MPP[@]}"; do wait $p; done
  for id in $IDS; do mkdir -p $A/sep/$id; [ -f $A/sep/$id/bg_44k.wav ] || cp $(readlink -f $A/$BASE/mix_src/$id)/bg_44k.wav $(readlink -f $A/$BASE/mix_src/$id)/vocals_44k.wav $A/sep/$id/
    [ -f $A/sep/$id/source_mediapipe.npz ] || python -c "import sys; sys.path.insert(0,'scripts/pilot'); import clean_face_stack as c, pathlib; c.run_mediapipe(pathlib.Path('$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/$id.* | head -1)'), pathlib.Path('$A/sep/$id/source_mediapipe.npz'))" > /dev/null 2>&1; done
  FAITH_ARG="--faith judge"   # retake judges the pipeline's own text (same prompt / 14B), no separate faith pass
else BASE=$REUSE; FAITH_ARG=""; fi
echo "BASE_DONE $(( $(date +%s) - s )) s" >> $S/times.txt
if [ "${R9_ONEPROC:-off}" = on ]; then   # the stage already ran inside the pipeline process: its time is in BASE_DONE
  echo "ONEPROC_DONE in-process (BASE incl. the stage, $(python3 -c "import time;print(round(time.time()-$t0p,1))") s pipeline + stage)" >> $S/times.txt
elif [ "${R8_ONEPROC:-on}" = on ]; then   # speed L5: retake -> clause takes -> burst_align of ALL videos in one process, models loaded once (scripts/r8/r8_tts_stage.py)
  J=$S/oneproc_jobs.json; t0p=$(date +%s.%N)
  for id in $IDS; do
    src16 $id
    f="$FAITH_ARG"; [ -z "$f" ] && for t in ${R7_PREV:-} $BASE; do [ -f $A/$t/$id.metrics.json ] && { f="--faith $A/$t/$id.metrics.json"; break; }; done
    [ -z "$f" ] && f="--faith judge"
    echo "$id|$(readlink -f $A/$BASE/mix_src/$id)|$f"
  done | mkjobs $J
  PYTHONPATH=$DUB_ROOT/third_party/llama_cuda:${PYTHONPATH:-} python scripts/r8/r8_tts_stage.py --jobs $J > $S/oneproc.log 2>&1 || echo "WARN oneproc (see $S/oneproc.log)"
  echo "ONEPROC_DONE $(python3 -c "import time;print(round(time.time()-$t0p,1))") s" >> $S/times.txt
fi
IDS_CHAIN=""   # the ids whose chain still has to run here (R9_CHAIN_INPROC: done inside the audio process -> skipped)
for id in $IDS; do [ -f $S/chain_done_$id ] || IDS_CHAIN="$IDS_CHAIN $id"; done
for id in $IDS_CHAIN; do
  if [ "${R8_ONEPROC:-on}" = on ]; then   # the one-process stage already ran: take its furthest output, then close_inner / soften as below
    t0=$(date +%s.%N); [ -f $S/rt/$id/dubbed_24k.wav ] || { echo "FAIL retake $id (see $S/rt_$id.log)"; continue; }
    W=$S/rt/$id; [ -f $S/ct/$id/dubbed_24k.wav ] && W=$S/ct/$id; [ -f $S/ba/$id/dubbed_24k.wav ] && W=$S/ba/$id; t1=$t0; t2=$t0
  else
  w=$(readlink -f $A/$BASE/mix_src/$id); s16=$S/src16/${id}_16k.wav; fc=$A/sep/$id/source_mediapipe.npz
  ffmpeg -loglevel error -y -i "$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/$id.* | head -1)" -vn -ac 1 -ar 16000 -c:a pcm_s16le $s16
  f="$FAITH_ARG"; [ -z "$f" ] && for t in ${R7_PREV:-} $BASE; do [ -f $A/$t/$id.metrics.json ] && { f="--faith $A/$t/$id.metrics.json"; break; }; done
  [ -z "$f" ] && f="--faith judge"   # no faith rows anywhere (a fresh base): retake judges the text itself
  t0=$(date +%s.%N)
  PYTHONPATH=$DUB_ROOT/third_party/llama_cuda:${PYTHONPATH:-} python scripts/r7/retake.py --work $w --out $S/rt/$id --face $fc $f --src16 $s16 --cover-all on $RT_ARGS > $S/rt_$id.log 2>&1 \
    || { echo "FAIL retake $id (see $S/rt_$id.log)"; continue; }
  t1=$(date +%s.%N); W=$S/rt/$id
  if [ "${R8_CLAUSE:-on}" = on ]; then   # v4: one take per clause for sentences over source pauses (scripts/r8/clause_takes.py)
    python scripts/r8/clause_takes.py --work $W --out $S/ct/$id --src16 $s16 --face $fc ${R8_CLAUSE_ARGS:-} > $S/ct_$id.log 2>&1 && W=$S/ct/$id || echo "WARN clause_takes $id (see $S/ct_$id.log)"
  fi
  if [ "${R8_ALIGN:-on}" = on ]; then
    python scripts/r7/burst_align.py --work $W --out $S/ba/$id --face $fc --cover-all on --tmin ${R8_TMIN:-0.9} --tmax ${R8_TMAX:-1.1} $BA_ARGS > $S/ba_$id.log 2>&1 && W=$S/ba/$id || echo "WARN burst_align $id (see $S/ba_$id.log)"
  fi
  t2=$(date +%s.%N)
  fi
  if [ "${R8_CLOSE:-on}" = on ]; then   # v2: the take's inner pauses where the source talks closed, the piece slowed to keep its span (scripts/r8/close_inner.py)
    python scripts/r8/close_inner.py --work $W --out $S/ci/$id --tmin ${R8_TMIN:-0.9} > $S/ci_$id.log 2>&1 && W=$S/ci/$id || echo "WARN close_inner $id (see $S/ci_$id.log)"
  fi
  if [ "${R8_SOFTEN:-on}" = on ]; then   # v3: abrupt word ends / starts in the take softened (scripts/r8/soften_edges.py)
    python scripts/r8/soften_edges.py --work $W --out $S/se/$id > $S/se_$id.log 2>&1 && W=$S/se/$id || echo "WARN soften $id (see $S/se_$id.log)"
  fi
  t3=$(date +%s.%N); ln -sfn $W $S/mix_src/$id
  python3 -c "import json,sys; t=[float(x) for x in sys.argv[1:]]; json.dump({'retake':t[1]-t[0],'burst_align':t[2]-t[1],'close_inner':t[3]-t[2]},open('$S/$id.steps.json','w'))" $t0 $t1 $t2 $t3
  echo "== $id src $(( $(date +%s) - s )) s"
done
[ -n "${IDS_CHAIN// /}" ] && FINAL_SOFTEN=${FINAL_SOFTEN:-on} RETAKE=off REUSE=${TAG}src LIP_ARGS="$LIPA" REMIX_ARGS="$REMA" \
  R7_PREV=${R7_PREV:-} bash scripts/r7/r7_audio.sh $TAG $IDS_CHAIN
mkdir -p $A/$TAG/cov
[ "${METRICS:-on}" = on ] && for id in $IDS; do
  python scripts/r8/coverage.py --run $A/$TAG/$id --id $id --label $TAG --face $A/sep/$id/source_mediapipe.npz --out $A/$TAG/cov/cov_$id 2>/dev/null | tail -1
  python scripts/r8/pacing.py --run $A/$TAG/$id --id $id --label $TAG --face $A/sep/$id/source_mediapipe.npz --out $A/$TAG/cov/pace_$id 2>/dev/null | tail -1
  python scripts/r8/cuts.py --run $A/$TAG/$id --id $id --label $TAG --out $A/$TAG/cov/cuts_$id 2>/dev/null | tail -1
  python scripts/r8/mixdrop.py --run $A/$TAG/$id --id $id --label $TAG --out $A/$TAG/cov/mix_$id 2>/dev/null | tail -1
done
echo "R8_AUDIO_DONE $TAG $(( $(date +%s) - s )) s"
