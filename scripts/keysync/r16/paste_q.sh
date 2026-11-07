# paste_q.sh <id> <tag> <ks.mp4> <box.json>
id=$1; t=$2; ks=$3; box=$4; cd /workspace/dub/keysync; MP=/venv/mediapipe/bin/python; Q=/workspace/dub/r16k2/q
[ -s $ks.mp.npz ] || $MP /workspace/dub/dubai/scripts/quality/mp_landmarks.py --video $ks --out $ks.mp.npz > /dev/null 2>&1
KS_CMS=1 $MP ks_plain.py --ks $ks --master $Q/${id}_master.mp4 --box $box --lm $Q/${id}_lmF.npz --shift 1 --audio $Q/${id}_mix.m4a --out $Q/${t}_final.mp4 --ks-lm $ks.mp.npz > $Q/${t}_paste.log 2>&1; tail -1 $Q/${t}_paste.log
