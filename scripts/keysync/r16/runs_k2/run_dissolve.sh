#!/bin/bash
# r16b K2: q03 alpha-paste through the two cross-dissolves (A end 569->589 into the title, B start 736->761 out of it)
export KS_LORA=/workspace/dub/r16in/lora/lora_pick.pt KS_DELAY=0 KS_PASTE_SHIFT=0 KS_IN_SHOT=1
H=/workspace/dub/r16in/day1/HANDOFF_Q; M=/workspace/dub/r16k2/ms; D=$M/dis; K=/workspace/dub/k2wt/scripts/keysync/r16; PY=/venv/dabai/bin/python
while pgrep -f "[r]un_q03.sh" > /dev/null; do sleep 15; done
mkdir -p $D; MS=$M/w_q03/ks/master_25fps.mp4
for p in "A 0 587 592 566 567 587" "B 737 943 736 762 737 762"; do set -- $p; S=$D/$1; mkdir -p $S/audio
  $PY $K/ks_dissolve.py unmix --master $MS --k0 $2 --k1 $3 --d0 $6 --d1 $7 --t-frame $4 --b-frame $5 --out $S/unmixed.mp4
  t0=$(python3 -c "print($2/25)"); t1=$(python3 -c "print($3/25)")
  ffmpeg -nostdin -v error -y -i $H/mixes/q03_priest/dubbed_16k.wav -af "atrim=start_sample=$(($2*640)):end_sample=$(($3*640)),asetpts=PTS-STARTPTS,apad=whole_len=$((($3-$2)*640))" -ar 16000 -ac 1 $S/audio/dubbed_16k.wav
  ffmpeg -nostdin -v error -y -i $H/mixes/q03_priest/final_mix_aac.m4a -af "atrim=start=$t0:end=$t1,asetpts=PTS-STARTPTS,apad=whole_dur=$(python3 -c "print($t1-$t0+0.2)")" -c:a aac -b:a 192k $S/audio/final_mix_aac.m4a
  $PY $K/ks_shots.py route --route $H/routes/repo_routing/q03_priest.json --t0 $t0 --t1 $t1 --out $S/route.json
  KS_MASTER_IN=$S/unmixed.mp4 bash /workspace/dub/k2wt/scripts/keysync/r16/ks_face.sh q03_$1 $S/unmixed.mp4 $S/audio $S $S/out.mp4 $S/route.json > $S/face.log 2>&1; echo "part $1 rc $? $(grep -E "^KS_PASTE" $S/face.log | cut -c1-120)"
done
( cd $K && KS_CMS=1 $PY $K/ks_dissolve.py assemble --master $MS --audio $H/mixes/q03_priest/final_mix_aac.m4a --part $D/A/out.mp4:0:$D/A/unmixed.mp4 $D/B/out.mp4:737:$D/B/unmixed.mp4 --out $M/q03_dissolve.mp4 )
echo DISSOLVE_DONE
