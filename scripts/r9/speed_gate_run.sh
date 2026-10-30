#!/bin/bash
# r9 audio speed gate run: gate_run.sh <tag> <id> [seed offset 0|1|2]   (artifacts under the tag <tag>_<id>)
#   the e2e audio stage alone (timed, lever flags from the caller env), the htdemucs mix of the same dub, then the e2e metrics block on BOTH mixes
#   (r4 metrics + WER, faith judge, coverage / pacing / cuts / mixdrop, gap words; the htdemucs mix: mixdrop, clicks, gap words)
set -u
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1; export DUB_ROOT=/workspace/dub/dubai   # env.sh points DUB_ROOT at /workspace/dub (models not found)
B=${1:?tag}; id=${2:?id}; SO=${3:-0}; TAG=${B}_$id;   # one tag per video (a run clears its own tag dirs)
 A=/workspace/dub/artifacts/r4a; HT=${TAG}_htd; L=/workspace/dub/r9s/logs
d=/workspace/dub/r9/runs/$id/work/in; [ -d $d ] || d=/workspace/dub/inputs/test_videos_raw
export METRICS=off DUB_ALLOW_NC=1 DUB_ALLOW_NC_BED=1 DUB_INPUTS=$d
D="--short-s 0.3 --budget-scales 1.15,1.3,1.5,1.75 --cap-scale 1.2 --min-piece 0.2 --max-tts 4 --natural-cuts on --fillers on"
if [ "$SO" != 0 ]; then export R8_PIPE_EXTRA="--seed $((1247 + SO))" R8_RETAKE_ARGS="$D --seed $((7 + SO))" R8_CLAUSE_ARGS="--seed $((11 + SO))"; fi
if [ "${METRICS_ONLY:-0}" != 1 ]; then   # METRICS_ONLY=1: the metrics block again on an existing run
rm -rf "/workspace/dub/artifacts/r4a/sep/${id:?}" "/workspace/dub/artifacts/r4a/${TAG:?}" "/workspace/dub/artifacts/r4a/${TAG:?}src" "/workspace/dub/artifacts/r4a/${TAG:?}pre" "/workspace/dub/artifacts/r4a/${TAG:?}_htd"
t0=$(date +%s.%N); bash scripts/r8/r8_audio.sh $TAG $id > $L/$TAG.$id.audio.log 2>&1; t1=$(date +%s.%N)
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $(ls $d/$id.* | head -1))
python3 -c "import sys, json; t=float(sys.argv[1]) - float(sys.argv[2]); d=float(sys.argv[3]); print(json.dumps({'tag': sys.argv[4], 'id': sys.argv[5], 'audio_s': round(t, 1), 'dur_s': d, 'audio_s_per_vmin': round(t / d * 60, 1)}))" $t1 $t0 $dur $TAG $id | tee -a /workspace/dub/r9s/timing.jsonl
O=$A/$TAG; [ -f $O/$id/final_mix_aac.m4a ] || { echo "GATE_FAIL $TAG $id: no audio"; exit 1; }
# the htdemucs mix (stems cached per id: they depend on the source only)
export DUB_SEP_STEMS=/workspace/dub/r8a
[ -f $DUB_SEP_STEMS/stems_htdemucs/$id/bg.wav ] || [ -n "$(ls $DUB_SEP_STEMS/stems_htdemucs/$id 2>/dev/null)" ] || \
  ( export PYTHONPATH=/workspace/dub/sep_pylib:${PYTHONPATH:-}; flock /workspace/dub/gpu.lock python scripts/r8/sep_stems.py --sep htdemucs --out $DUB_SEP_STEMS/stems_htdemucs $id ) > $L/$TAG.$id.stems.log 2>&1
SEP_SRC=$TAG SEP_PREV=$TAG bash scripts/r8/sep_mix.sh $HT htdemucs stems "--bed-pauses source --bed-noise-back on --src-duck-db -3" $id > $L/$TAG.$id.sepmix.log 2>&1
fi
export METRICS=on
{ src=$(ls $d/$id.* | head -1)
  python scripts/quality/r4_audio_metrics.py --mix $O/${id}_over-original.mp4 --src $src --sep $A/sep/$id --run $O/$id --manifest $O/$id/${id}_r4b.manifest.json --wer \
    --series $O/${id}.series.npz --json $O/${id}.metrics.json || echo "WARN audio metrics"
  flock /workspace/dub/gpu.lock env DUB_QWEN_SIZE=14b PYTHONPATH=third_party/llama_cuda:${PYTHONPATH:-} python scripts/r5/r5_faith.py $TAG $id > $O/runs/faith.log 2>&1 || echo "WARN faith"
  mkdir -p $O/cov
  python scripts/r8/coverage.py --run $O/$id --id $id --label $TAG --face $A/sep/$id/source_mediapipe.npz --out $O/cov/cov_$id | tail -1
  python scripts/r8/pacing.py --run $O/$id --id $id --label $TAG --face $A/sep/$id/source_mediapipe.npz --out $O/cov/pace_$id | tail -1
  python scripts/r8/cuts.py --run $O/$id --id $id --label $TAG --out $O/cov/cuts_$id | tail -1
  python scripts/r8/mixdrop.py --run $O/$id --id $id --label $TAG --out $O/cov/mix_$id | tail -1
  flock /workspace/dub/gpu.lock python scripts/r7/gap_metrics.py --tag $TAG --clips $O/gaps $id || echo "WARN gaps spleeter"
  if [ -f $A/$HT/$id/final_mix_aac.m4a ]; then mkdir -p $A/$HT/cov
    python scripts/r8/mixdrop.py --run $A/$HT/$id --id $id --label $HT --out $A/$HT/cov/mix_$id | tail -1
    python scripts/r7/clicks.py --tag $HT --prev $TAG --json $A/$HT/cov/clicks.json $id
    flock /workspace/dub/gpu.lock python scripts/r7/gap_metrics.py --tag $HT --prev $TAG --clips $A/$HT/gaps $id || echo "WARN gaps demucs"
  else echo "WARN no htdemucs mix"; fi
} > $L/$TAG.$id.metrics.log 2>&1
echo "GATE_RUN_DONE $TAG $id"
