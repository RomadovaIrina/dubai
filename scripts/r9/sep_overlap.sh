#!/bin/bash
# r9 speed lever O (R9_SEP_OVERLAP=on): the htdemucs (demucs) mix of a video made WHILE its audio stage runs, with the same commands and inputs
# as the e2e sep stage (identical files):
#   start  sep_overlap.sh start <id> <stems dir> <marker dir> <log dir>      the stems (they depend on the source only) in the background
#   mix    sep_overlap.sh mix <id> <audio tag> <htd tag> <marker dir> <log dir>   called by r4_audio.sh (R9_ON_MIXSRC) the moment the final dub
#          is linked (mix_src): waits for the stems, then the sepH8p remix of that dub, in parallel with the Spleeter remix
#   wait   sep_overlap.sh wait <marker dir>                                     until the demucs mix is done (exit 1 when the stems failed)
set -u
cd "$(dirname "$0")/../.."
case "$1" in
  start)
    id=$2; ST=$3; M=$4; L=$5; rm -f $M/stems.ok $M/stems.done $M/sepmix.done
    ( export PYTHONPATH=/workspace/dub/sep_pylib:${PYTHONPATH:-}; python scripts/r8/sep_stems.py --sep htdemucs --out $ST $id > $L/sep_stems.log 2>&1 && touch $M/stems.ok; touch $M/stems.done ) &
    ;;
  mix)
    id=$2; AT=$3; HT=$4; M=$5; L=$6
    while [ ! -f $M/stems.done ]; do sleep 0.5; done
    [ -f $M/stems.ok ] && SEP_SRC=$AT SEP_PREV=$AT bash scripts/r8/sep_mix.sh $HT htdemucs stems "--bed-pauses source --bed-noise-back on --src-duck-db -3" $id > $L/sep_mix.log 2>&1
    touch $M/sepmix.done
    ;;
  wait)
    M=$2
    while [ ! -f $M/sepmix.done ]; do sleep 0.5; done
    [ -f $M/stems.ok ]
    ;;
esac
