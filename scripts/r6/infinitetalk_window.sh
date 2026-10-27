#!/bin/bash
# r6 G: InfiniteTalk (MeiGen-AI, weights apache-2.0; Wan2.1-I2V-14B-480P apache-2.0; chinese-wav2vec2-base mit) as an alternative backend for
# ls_window_ab.py --precomputed. Video-to-video ("dubbing") mode, streaming, 480p bucket, fp8 quant, 40 steps (STEPS env), single speaker.
# The output is resized back to the input W x H and padded / cut to the input frame count (Wan re-renders the whole frame; ls_window_ab.py's
# mouth-only mask then keeps only the mouth region, like every other arm).
#   bash scripts/r6/infinitetalk_window.sh <window_25fps.mp4> <drive_16k.wav> <out.mp4>
set -euo pipefail
IN=$(readlink -f $1); AUD=$(readlink -f $2); OUT=$(readlink -f $3); IT=/workspace/dub/alt/infinitetalk; R=$IT/repo; W=$IT/weights
T=$(mktemp -d /workspace/dub/alt/infinitetalk/tmp.XXXX); mkdir -p $(dirname $OUT)
read Wd Hd N <<< $(ffprobe -v error -count_frames -select_streams v -show_entries stream=width,height,nb_read_frames -of csv=p=0 $IN | tr , ' ')
cat > $T/in.json <<EOF
{"prompt": "A man is talking to the camera", "cond_video": "$IN", "cond_audio": {"person1": "$AUD"}}
EOF
cd $R; t0=$(date +%s.%N)
$IT/venv/bin/python generate_infinitetalk.py --ckpt_dir $W/Wan2.1-I2V-14B-480P --wav2vec_dir $W/chinese-wav2vec2-base \
  --infinitetalk_dir $W/InfiniteTalk/single/infinitetalk.safetensors --input_json $T/in.json --size infinitetalk-480 --sample_steps ${STEPS:-40} \
  --mode streaming --motion_frame 9 --quant fp8 --quant_dir $W/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors \
  --num_persistent_param_in_dit ${PERSIST:-0} --audio_save_dir $T/audio --save_file $T/gen > $T/log.txt 2>&1 || { tail -30 $T/log.txt; exit 3; }
t1=$(date +%s.%N)
ffmpeg -loglevel error -y -i $T/gen.mp4 -vf "fps=25,scale=${Wd}:${Hd}:flags=lanczos,tpad=stop=$N:stop_mode=clone" -frames:v $N -an -c:v libx264 -qp 0 $OUT
python3 -c "import json;print(json.dumps({'backend':'infinitetalk_fp8','steps':${STEPS:-40},'gen_s':round($t1-$t0,1),'frames':$N,'gen_frames':int('$(ffprobe -v error -count_frames -select_streams v -show_entries stream=nb_read_frames -of csv=p=0 $T/gen.mp4)'),'gen_size':'$(ffprobe -v error -select_streams v -show_entries stream=width,height -of csv=p=0:s=x $T/gen.mp4)'}))" | tee ${OUT%.mp4}.json
cp $T/log.txt ${OUT%.mp4}.log; rm -rf $T
