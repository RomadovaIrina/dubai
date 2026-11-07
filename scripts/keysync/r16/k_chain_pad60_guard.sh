cd /workspace/dub/keysync; P=/workspace/dub/r16in/HANDOFF_K/paste_inputs; O=/workspace/dub/outbox/r16_k1
enc() { D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $1); BR=$(python3 -c "print(int(min(8000, 25e6 * 8 / $D * 0.92 / 1000 - 160)))"); ffmpeg -nostdin -v error -y -i $1 -c:v libx264 -profile:v high -level 4.1 -preset slow -crf 18 -maxrate ${BR}k -bufsize $((2 * BR))k -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv -movflags +faststart -c:a copy $2; }
# m18 from the 60-frame pad render (no duplicated seam frame)
R=r16/out_pad60/m18q/m18_pad60_m18q.mp4; W=r16/pad/m18q
(cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video /workspace/dub/keysync/$R --out /workspace/dub/keysync/$W.raw.mp.npz >/dev/null 2>&1 </dev/null)
./k_gate.sh $R $P/m18/dubbed_16k.wav $W.rawsync.json </dev/null | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('m18 pad60 raw vs orig dub', d['global'])"
/venv/dabai/bin/python k_ecc.py --ks $R --src data/videos/m18_pad60.mp4 --lm $P/m18/source_crop512.mediapipe.npz --out $W.ecc.npz </dev/null
/venv/dabai/bin/python ks_tsmooth.py --layer $R --lm $W.raw.mp.npz --src-lm $P/m18/source_crop512.mediapipe.npz --out $W.ts.mp4 --alpha gate:0.5 </dev/null 2>&1 | grep gate:
/venv/dabai/bin/python ks_snap.py --layer $W.ts.mp4 --lm $W.raw.mp.npz --out $W.snap.mp4 </dev/null | grep ks_snap
/venv/dabai/bin/python ks_guard.py --layer $W.snap.mp4 --src data/videos/m18_pad60.mp4 --n 501 --out $W.guard.mp4 </dev/null
KS_CMS=1 /venv/dabai/bin/python ks_plain.py --ks $W.guard.mp4 --ks-lm $W.raw.mp.npz --master r16/master/m18.mp4 --box data/box_m18.json --lm $P/m18/source_lm_F.npz --audio $P/m18/final_mix_aac.m4a --shift 0 \
  --lfm 6 --upper-src 0.12:0.32 --keep-nose 0.04 --align 2 --align-ecc $W.ecc.npz --align-scale 0 --out r16/plain/m18_clean5.mp4 </dev/null 2>&1 | grep -E "WARN|Trace|Error"
# m03 / old03: the guard on the existing padded layers
for v in m03 old03; do W=r16/pad/$v; OTH=""; [ $v = m03 ] && OTH="--others $P/m03/faces2.npz --others-grow 1.3"
  KS_CMS=1 /venv/dabai/bin/python ks_plain.py --ks $W.guard.mp4 --ks-lm $W.raw.mp.npz --master r16/master/$v.mp4 --box data/box_$v.json --lm $P/$v/source_lm_F.npz --audio $P/$v/final_mix_aac.m4a --shift 0 \
    --lfm 6 --upper-src 0.12:0.32 --keep-nose 0.04 --align 2 --align-ecc $W.ecc.npz --align-scale 0 $OTH --out r16/plain/${v}_clean5.mp4 </dev/null 2>&1 | grep -E "WARN|Trace|Error"; done
for v in m18 m03 old03; do ./k_gate.sh r16/plain/${v}_clean5.mp4 $P/$v/dubbed_16k.wav r16/plain/${v}_clean5.sync.json </dev/null | grep -o "speech_at0[^,]*" | sed "s/^/$v clean5 /"; enc r16/plain/${v}_clean5.mp4 $O/full/${v}_keysync_clean5.mp4; done
./k_clip3.sh $O/clips/m18_end_clean4_clean5_src_17-20.04s_half_speed.mp4 17 20.04 data/box_m18.json $P/m18/final_mix_aac.m4a clean4_broken_end=r16/plain/m18_clean4.mp4 clean5=r16/plain/m18_clean5.mp4 source=r16/master/m18.mp4 </dev/null
for v in m03 old03; do L=$(ffprobe -v error -show_entries format=duration -of csv=p=0 r16/master/$v.mp4); E=$(python3 -c "print(round($L-2.0,2))")
  ./k_clip3.sh $O/clips/${v}_end_clean4_clean5_src_last2s_half_speed.mp4 $E $L data/box_$v.json $P/$v/final_mix_aac.m4a clean4=r16/plain/${v}_clean4.mp4 clean5=r16/plain/${v}_clean5.mp4 source=r16/master/$v.mp4 </dev/null; done
./k_clip3.sh $O/clips/old03_seams_clean4_clean5_src_41-46s_half_speed.mp4 41 46 data/box_old03.json $P/old03/final_mix_aac.m4a clean4=r16/plain/old03_clean4.mp4 clean5_guard=r16/plain/old03_clean5.mp4 source=r16/master/old03.mp4 </dev/null
# last-frame check of every final
for v in m18 m03 old03; do /venv/dabai/bin/python /tmp/jump.py $v r16/plain/${v}_clean5.mp4 | grep "last 30" | sed "s/^/$v /"; done
cd /workspace/dub/dubai; K=/workspace/dub/keysync; cp $K/ks_guard.py $K/ks_seam.py $K/ks_cheeksmooth.py $K/k_cheekvel.py scripts/keysync/r16/; cp /tmp/tm.sh scripts/keysync/r16/k_chain_pad60_guard.sh; git add scripts/keysync; git -c user.name="DabAI agent K1" -c user.email="dabai-agent-K1@users.noreply.github.com" commit -q -m "r16 K1: the m18 broken last frame = KeySync degenerating at the duplicated mirror-pad seam frame; pad 60 frames WITHOUT the duplicate + ks_guard.py (per-frame lower-face sanity, in-between / hold); also ks_seam, ks_cheeksmooth, k_cheekvel

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; git log --oneline -1
