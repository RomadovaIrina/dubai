#!/bin/bash
# r16c K2 day 3: engines, then m18 gate arms through T's int ks_face (lips at 0:10) and the long5 dark-haired shot (story 3) for the blue mouth
K=/workspace/dub/k2; L=/workspace/dub/k2/lora; F=/workspace/dub/intwt/scripts/keysync/r16/ks_face.sh; P=/workspace/dub/r16in/HANDOFF_K/paste_inputs
until [ "$(stat -c %s $L/lora_pick.pt 2>/dev/null)" = 461504071 ] && [ -s $L/lora_meta.json ]; do sleep 30; done; echo "LORA_OK $(date)"
export KS_LORA=$L/lora_pick.pt
cd /workspace/dub/intwt && bash scripts/keysync/r16/ks_engines.sh > $K/engines.log 2>&1; tail -1 $K/engines.log; grep TRT_VERIFY $K/engines.log | cut -c1-120
# m18 arms (int defaults = ref 0 + probe delay + gate hi8)
mkdir -p $K/m18/ad; cp $P/m18/dubbed_16k.wav $P/m18/final_mix_aac.m4a $K/m18/ad/
declare -A ARM=([A_e2e]="thr=0.12,cap=0.25,hi=8,lo=2" [B_off]="" [C_hi3]="thr=0.12,cap=0.25,hi=3,lo=2")
for arm in A_e2e B_off C_hi3; do
  env KS_GATE="${ARM[$arm]}" bash $F m18_$arm /workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4 $K/m18/ad $K/m18/w_$arm $K/m18/$arm.mp4 > $K/m18/$arm.log 2>&1
  echo "m18 $arm rc $? $(grep -hE "^KS_PROBE|^KS_CROP" $K/m18/$arm.log | cut -c1-100 | tr "\n" " ")"
done
# long5's dark-haired shot = story 3 (own audio drives the mouth; the question is colour)
S3=/workspace/dub/inputs/mamakris/story-1774250573471-666527943.MP4; mkdir -p $K/s3/ad
ffmpeg -nostdin -v error -y -i $S3 -vn -ac 1 -ar 16000 $K/s3/ad/dubbed_16k.wav; ffmpeg -nostdin -v error -y -i $S3 -vn -c:a aac -b:a 192k $K/s3/ad/final_mix_aac.m4a
bash $F s3 $S3 $K/s3/ad $K/s3/w $K/s3/s3.mp4 > $K/s3/s3.log 2>&1; echo "s3 rc $?"
echo DAY3_GPU_DONE
