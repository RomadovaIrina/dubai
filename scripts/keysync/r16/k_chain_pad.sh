cd /workspace/dub/keysync; P=/workspace/dub/r16in/HANDOFF_K/paste_inputs; O=/workspace/dub/outbox/r16_k1; mkdir -p r16/pad
enc() { D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $1); BR=$(python3 -c "print(int(min(8000, 25e6 * 8 / $D * 0.92 / 1000 - 160)))"); ffmpeg -nostdin -v error -y -i $1 -c:v libx264 -profile:v high -level 4.1 -preset slow -crf 18 -maxrate ${BR}k -bufsize $((2 * BR))k -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv -movflags +faststart -c:a copy $2; }
for v in m18 m03 old03; do
  R=r16/out_pad/${v}p/${v}_pad_${v}p.mp4; W=r16/pad/$v
  (cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video /workspace/dub/keysync/$R --out /workspace/dub/keysync/$W.raw.mp.npz >/dev/null 2>&1 </dev/null)
  ./k_gate.sh $R $P/$v/dubbed_16k.wav $W.rawsync.json </dev/null | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('$v raw vs orig dub', d['global'])"
  # source crop landmarks for the padded length: the source crop lm (first N frames are what matters)
  /venv/dabai/bin/python k_ecc.py --ks $R --src data/videos/${v}_pad.mp4 --lm $P/$v/source_crop512.mediapipe.npz --out $W.ecc.npz </dev/null
  /venv/dabai/bin/python ks_tsmooth.py --layer $R --lm $W.raw.mp.npz --src-lm $P/$v/source_crop512.mediapipe.npz --out $W.ts.mp4 --alpha gate:0.5 </dev/null 2>&1 | grep gate:
  /venv/dabai/bin/python ks_snap.py --layer $W.ts.mp4 --lm $W.raw.mp.npz --out $W.snap.mp4 </dev/null | grep ks_snap
  OTH=""; [ $v = m03 ] && OTH="--others $P/m03/faces2.npz --others-grow 1.3"
  KS_CMS=1 /venv/dabai/bin/python ks_plain.py --ks $W.snap.mp4 --ks-lm $W.raw.mp.npz --master r16/master/$v.mp4 --box data/box_$v.json --lm $P/$v/source_lm_F.npz --audio $P/$v/final_mix_aac.m4a --shift 0 \
    --lfm 6 --upper-src 0.12:0.32 --keep-nose 0.04 --align 2 --align-ecc $W.ecc.npz --align-scale 0 $OTH --out r16/plain/${v}_clean4.mp4 </dev/null 2>&1 | grep -E "WARN|Trace|Error|plain paste:"
  ./k_gate.sh r16/plain/${v}_clean4.mp4 $P/$v/dubbed_16k.wav r16/plain/${v}_clean4.sync.json </dev/null | grep -o "speech_at0.*" | sed "s/^/$v clean4 /"
  /venv/dabai/bin/python /tmp/jump.py $v r16/plain/${v}_clean4.mp4 | grep "last 30"
  enc r16/plain/${v}_clean4.mp4 $O/full/${v}_keysync_clean4.mp4
done
prev() { case $1 in m18) echo r16/plain/m18_clean3.mp4;; m03) echo r16/plain/m03_clean3.mp4;; old03) echo r16/plain/old03_c_upnose.mp4;; esac; }
./k_clip3.sh $O/clips/m18_end_clean3_fixed_src_17-20.1s_half_speed.mp4 17 20.1 data/box_m18.json $P/m18/final_mix_aac.m4a clean3=r16/plain/m18_clean3.mp4 fixed_pad=r16/plain/m18_clean4.mp4 source=r16/master/m18.mp4 </dev/null
for v in m03 old03; do L=$(ffprobe -v error -show_entries format=duration -of csv=p=0 r16/master/$v.mp4); E=$(python3 -c "print(round($L-2.0,2))"); E1=$(python3 -c "print(round($L+0.05,2))")
  ./k_clip3.sh $O/clips/${v}_start_prev_fixed_src_0-2s_half_speed.mp4 0 2 data/box_$v.json $P/$v/final_mix_aac.m4a previous=$(prev $v) fixed_pad=r16/plain/${v}_clean4.mp4 source=r16/master/$v.mp4 </dev/null
  ./k_clip3.sh $O/clips/${v}_end_prev_fixed_src_last2s_half_speed.mp4 $E $E1 data/box_$v.json $P/$v/final_mix_aac.m4a previous=$(prev $v) fixed_pad=r16/plain/${v}_clean4.mp4 source=r16/master/$v.mp4 </dev/null; done
cd /workspace/dub/dubai; K=/workspace/dub/keysync; cp $K/ks_plain.py scripts/keysync/r15/; cp /tmp/tk.sh scripts/keysync/r16/k_chain_pad.sh; git add scripts/keysync; git -c user.name="DabAI agent K1" -c user.email="dabai-agent-K1@users.noreply.github.com" commit -q -m "r16 K1: no KeySync -> source switch at the end: pad the crop (mirrored 30 frames) + dub (silence) so KeySync covers every frame; ks_plain warns + 1 s fade only as a last resort; k_chain_pad.sh = the full r16 chain

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; git log --oneline -1
