#!/bin/bash
# r15 K: KeySync inputs. Held-out (qo3 / q18 / m03, m03 = the speaker-only crop) get the q2 DUB voice; the training clips (tr_*) keep their own audio.
# usage: bash k_inputs.sh test|mamakris
K=/workspace/dub/keysync; T=/workspace/dub/inputs/test_videos_raw; M=/workspace/dub/inputs/mamakris; cd $K; mkdir -p lm data/videos data/audios final
prep() { id=$1; v=$2; [ -s data/videos/$id.mp4 ] && return; bash ks_prep_src.sh $id $v 2>&1 | tail -1; mkdir -p final/$id; cp lm/${id}_src_mp.npz final/$id/src.mediapipe.npz; }
dub() { ffmpeg -v error -y -i /workspace/dub/dubaudio/$2/dubbed_16k.wav -ac 1 -ar 16000 data/audios/$1.wav && echo "DUBAUDIO $1"; }
if [ "$1" = test ]; then
  unset KS_BOX_K KS_BOX_M; prep qo3 $T/03.mp4; dub qo3 old03
  for n in 01 02 04 05; do f=$(ls $T/$n.* | head -1); prep tr_t$n $f; done
else
  unset KS_BOX_K KS_BOX_M; prep q18 $M/story-1774250817106-616905094.mp4; dub q18 m18
  KS_BOX_K=1.25 KS_BOX_M=0.4 prep m03 $M/story-1774249919482-367803614.MOV; dub m03 m03
  i=0; for f in $M/story-*; do case $f in *1774250817106*|*1774249919482*) continue;; esac; i=$((i + 1)); prep tr_mk$i $f; done
fi
echo K_INPUTS_DONE $1
