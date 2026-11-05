#!/bin/bash
# r15 K: k_eval.sh <arm> <id> -- the SAME post for every arm (= the r12/r13 ks_eval path + the r13 detail transfer):
#   SyncNet AV offset of the raw KeySync output -> alpha 0.8 lag smoothing (ks_tsmooth) -> raw paste at the source fps (KS_HF 3 / KS_SHARP 0.3, the q2 mix)
#   -> SyncNet mean conf at offset 0 over the dub-speech seconds (25 fps rendition; = scripts/r11/sync_speech.py) + lower-face tone flicker vs the source.
# Face GPU-s/vmin comes from ks_fast's KS_JOB line (KeySync stage, landmarks off the critical path as in prod). -> fast/<arm>/eval_<id>/eval.json
A=$1; id=$2; K=/workspace/dub/keysync; O=$K/fast/$A; KS=$O/out/$id/${id}_$id.mp4; E=$O/eval_$id; mkdir -p $E
case $id in q18) SRC=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4; v=m18;; qo3) SRC=/workspace/dub/inputs/test_videos_raw/03.mp4; v=old03;; m03) SRC=/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV; v=m03;; esac
DUB=/workspace/dub/dubaudio/$v/dubbed_16k.wav; MIX=/workspace/dub/dubaudio/$v/final_mix_aac.m4a; LM=$K/lm/${id}_lm.npz
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
[ -s $E/raw_sync.json ] || flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $KS --audio $DUB --out $E/raw_sync.json > $E/raw_sync.log 2>&1
SH=$(python3 -c "import json; print(-int(json.load(open('$E/raw_sync.json'))['global']['av_offset']))")
[ -s $E/ks.mediapipe.npz ] || /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $KS --out $E/ks.mediapipe.npz > /dev/null 2>&1
cd $K
python ks_tsmooth.py --layer $KS --lm $E/ks.mediapipe.npz --out $E/ks_ts.mp4 --alpha ${KS_TSMOOTH:-0.8} > /dev/null 2>&1
KS_SHIFT=$SH KS_HF=3 KS_SHARP=0.3 KS_HF_LM=$LM python ks_paste.py $E/ks_ts.mp4 $SRC data/box_$id.json $MIX $E/paste.mp4 > /dev/null 2>&1
ffmpeg -v error -y -i $E/paste.mp4 -vf fps=25 -c:v libx264 -crf 12 -an $E/p25.mp4
cd /workspace/dub/dubai
flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $E/p25.mp4 --audio $DUB --out $E/sync.json > $E/sync.log 2>&1
python $K/tone_flicker.py --video $E/paste.mp4 --src $SRC --src-lm $LM --label ${A}_$id 2>/dev/null | grep "== tone_flicker" | sed 's/== tone_flicker //' > $E/flicker.json
python3 - $E $SH $A $id $O/run.log <<'PY'
import json, sys, statistics as st
E, sh, A, id_, log = sys.argv[1:6]
r = json.load(open(f"{E}/raw_sync.json")); s = json.load(open(f"{E}/sync.json"))
try: f = json.load(open(f"{E}/flicker.json"))
except Exception: f = {}
sp = [p["conf_at0"] for p in s["per_second"] if p.get("conf_at0") is not None and (p.get("dub_act") or 0) >= 0.5]
jobs = [json.loads(l.split("KS_JOB ", 1)[1]) for l in open(log, errors="ignore") if "KS_JOB " in l]
j = [x for x in jobs if x["tag"] == id_][-1] if any(x["tag"] == id_ for x in jobs) else {}
st_ = j.get("stages_s", {}); lmk = st_.get("extract_video_landmarks", 0.0)
gs = round((j["wall_s"] - lmk) / j["video_s"] * 60, 1) if j else None
out = {"arm": A, "id": id_, "sync_speech_at0": round(st.mean(sp), 3), "raw_av_offset": r["global"]["av_offset"], "shift": int(sh),
       "face_gpu_s_per_vmin": gs, "interp_s": round(st_.get("sample_interpolation", 0), 1), "kf_s": round(st_.get("sample_keyframes", 0), 1),
       "vae_dec_interp_s": round(st_.get("vae_decode_interp", 0), 1), "video_s": j.get("video_s"),
       "flicker_low_mean": f.get("low", {}).get("mean"), "flicker_lips_mean": f.get("lips", {}).get("mean"), "flicker_ring_mean": f.get("ring", {}).get("mean"),
       "flicker_lips_p95": f.get("lips", {}).get("p95")}
json.dump(out, open(f"{E}/eval.json", "w")); print("K_EVAL " + json.dumps(out))
PY
