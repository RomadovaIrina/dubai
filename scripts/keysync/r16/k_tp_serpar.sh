# r16b K1 STEP 0 test: serial ks_plain vs ks_plain_par (4 chunks) on the raw KeySync fast2 layers (r16in/HANDOFF_K/paste_inputs), + m03 --sharp auto --cavity 0.5.
# Run with KS_PLAIN_CRF=0 (lossless): at crf 16 two independent x264 encodes alone differ by 42-44 dB. Masters: $T/ms/{m03,m18,old03}.mp4 = ffmpeg -i SRC -an -vf fps=25.
set -u
W=/workspace/dub/k1wt/scripts/keysync; T=/workspace/dub/k1test; mkdir -p $T/tools $T/out $T/in
cp $W/r15/ks_plain.py $W/r16/ks_plain_par.py $W/r16/ks_seam.py $W/r16/k_ecc.py $W/ks_cms.py $T/tools/
P=/workspace/dub/r16in/HANDOFF_K/paste_inputs; MS=$T/ms; PY=/workspace/dub/k1venv/bin/python; TAG=${TAG:-}
cd $T/tools
for v in m03 m18 old03; do [ -s $T/in/$v.ecc.npz ] && continue
  b=$(cat $P/$v/box.json); x=$(echo $b | $PY -c "import json,sys;d=json.load(sys.stdin);print(d['x'],d['y'],d['side'])")
  set -- $x; ffmpeg -v error -y -i $MS/$v.mp4 -vf "crop=$3:$3:$1:$2,scale=512:512:flags=lanczos" -c:v libx264 -crf 10 $T/in/${v}_src512.mp4
  ($PY k_ecc.py --ks $P/$v/keysync_fast2_raw512.mp4 --src $T/in/${v}_src512.mp4 --lm $P/$v/source_crop512.mediapipe.npz --out $T/in/$v.ecc.npz > $T/in/$v.ecc.log 2>&1) &
done; wait
args() { v=$1; SH=0; [ $v = old03 ] && SH=-1; OTH=""; [ $v = m03 ] && OTH="--others $P/m03/faces2.npz --others-grow 1.3"
  echo "--ks $P/$v/keysync_fast2_raw512.mp4 --ks-lm $P/$v/keysync_raw512.mediapipe.npz --master $MS/$v.mp4 --box $P/$v/box.json --lm $P/$v/source_lm_F.npz --audio $P/$v/final_mix_aac.m4a --shift $SH --lfm 6 --upper-src 0.12:0.32 --keep-nose 0.04 --align 2 --align-ecc $T/in/$v.ecc.npz --align-scale 0 --tex 2 $OTH"; }
for v in m18 m03 old03; do (KS_CMS=1 $PY ks_plain.py $(args $v) --out $T/out/${v}_ser$TAG.mp4 > $T/out/${v}_ser$TAG.log 2>&1) & done
(KS_CMS=1 $PY ks_plain.py $(args m03) --sharp auto --cavity 0.5 --out $T/out/m03x_ser$TAG.mp4 > $T/out/m03x_ser$TAG.log 2>&1) &
wait
for v in m18 m03 old03; do (KS_CMS=1 $PY ks_plain_par.py --jobs 4 --out $T/out/${v}_par$TAG.mp4 -- $(args $v) > $T/out/${v}_par$TAG.log 2>&1) & done
(KS_CMS=1 $PY ks_plain_par.py --jobs 4 --out $T/out/m03x_par$TAG.mp4 -- $(args m03) --sharp auto --cavity 0.5 > $T/out/m03x_par$TAG.log 2>&1) &
wait
for v in m18 m03 old03 m03x; do b=${v%x}; $PY $W/r16/k_tp_cmp.py $T/out/${v}_ser$TAG.mp4 $T/out/${v}_par$TAG.mp4 $P/$b/box.json $MS/$b.mp4; done
grep -h -E "WARN|colour match|sharp|cavity|Trace|Error" $T/out/*$TAG.log | sort | uniq -c | cut -c1-220
echo TP_DONE
