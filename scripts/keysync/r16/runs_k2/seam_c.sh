#!/bin/bash
# r16c K2 seam variant (c): tight mask WITHOUT the generated chin (--mask tight, no --mask-chin), re-pasted on K1 "now" KeySync layers
set -o pipefail; R=/workspace/dub/k2/seamc; S=/workspace/dub/k1/seam; FK=/workspace/dub/k1/from_k1; Q=/workspace/dub/r16in/day1/HANDOFF_Q
WT=/workspace/dub/k1/wt/scripts/keysync; F=$WT/r16/ks_face.sh; EV=/workspace/dub/k1/qa_wt/scripts/r16q/eval_final.sh; PY=/venv/dabai/bin/python3
export KS_LORA=/workspace/dub/r16in/lora/lora_pick.pt KS_REF=0 KS_PASTE_SHIFT=0 KS_PASTE_JOBS=1 KS_ROOT=/workspace/dub/keysync
declare -A AD=( [m03]=/workspace/dub/r16in/HANDOFF_K/paste_inputs/m03 [q03a]=$FK/k1c/in/q03a_aud [old03]=$FK/r16in/HANDOFF_K/paste_inputs/old03 [q04]=$Q/mixes/q04_farikh [q01]=$Q/mixes/q01_amir )
declare -A W=( [m03]=$S/w_m03 [q03a]=$S/w_q03a [old03]=$S/w_old03 [q04]=$S/w_q04 [q01]=$S/w_q01 )
declare -A NOW=( [m03]=$S/m03_now.mp4 [q03a]=$S/q03a_now.mp4 [old03]=$S/old03_now.mp4 [q04]=$S/q04_now.mp4 [q01]=$S/q01_now.mp4 )
# q01 "now" (no K1 render on this box): same settings as K1 seam_test, probe delay
: # q01: K1 render pending
KT=$R/kit; mkdir -p $KT; cp $WT/r15/ks_plain.py $WT/r16/ks_plain_par.py $WT/ks_cms.py $WT/r16/face_parse.py $KT/
for id in q03a old03 m03 q04 q01; do KW=${W[$id]}/ks; P=$KW/post_f0
  [ -s $P/guard.mp4 ] || { echo "SKIP $id: no layers ($P)"; continue; }
  OTH=""; [ -s $KW/others_f0.npz ] && OTH="--others $KW/others_f0.npz --others-grow 1.3"
  B="--ks $P/guard.mp4 --ks-lm $P/raw_lm.npz --master $KW/master_25fps.mp4 --box $KW/data/box_f0.json --lm $KW/lm_f0.npz --audio ${AD[$id]}/final_mix_aac.m4a --shift 0 --absent skip $OTH --lfm 6 --upper-src 0.12:0.32 --keep-nose 0.04 --align 2 --align-ecc $P/ecc.npz --align-scale 0 --tex 2"
  [ -s $R/${id}_c.mp4 ] || ( cd $KT && KS_CMS=1 /venv/dabai/bin/python ks_plain_par.py --jobs 4 --out $R/${id}_c.mp4 -- $B --mask tight ) > $R/${id}_c.log 2>&1 || echo "FAIL paste $id"
  for t in now c; do f=$R/${id}_c.mp4; [ $t = now ] && f=${NOW[$id]}
    j=$($PY $WT/r16/seam_jitter.py --final $f --master $KW/master_25fps.mp4 --lm $KW/lm_f0.npz --out $R/${id}_$t.seam.json | sed "s/SEAM_JITTER //")
    bash $EV $f ${AD[$id]}/dubbed_16k.wav $R/ev/${id}_$t --noise 0 --label ${id}_$t > $R/ev_${id}_$t.log 2>&1
    echo "$id $t seam $(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get(\"excess_median\", d.get(\"ratio\")), d.get(\"excess_p95\", d.get(\"extra_mean\")), d[\"worst_seconds\"][:3])" "$j" 2>/dev/null) speech@0 $(grep -o "\"speech_at0\": [0-9.-]*" $R/ev/${id}_$t/eval.json 2>/dev/null | cut -d" " -f2)"
  done
done
echo SEAMC_DONE
