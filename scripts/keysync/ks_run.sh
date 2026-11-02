#!/bin/bash
# KeySync bake-off: one video under gpu.lock, NVML-sampled. ks_run.sh <tag> [extra fire args]
set -u
T=$1; shift; K=/workspace/dub/keysync; O=$K/runs/$T${KS_CFG:-}; mkdir -p $O
cd $K/code; . $K/venv/bin/activate
/venv/main/bin/python /workspace/dub/dubai/scripts/r10/gpu_sampler.py --out $O/nvml.jsonl 2>$O/sampler.log & SP=$!
t0=$(date +%s.%N)
flock /workspace/dub/gpu.lock python scripts/sampling/dubbing_pipeline_raw.py \
  --filelist=$K/data/videos/$T.mp4 --filelist_audio=$K/data/audios/$T.wav \
  --decoding_t 1 --cond_aug 0. --resize_size=512 --force_uc_zero_embeddings='[cond_frames, audio_emb]' \
  --latent_folder=videos --video_folder=videos --model_config=scripts/sampling/configs/interpolation${KS_CFG:-}.yaml \
  --model_keyframes_config=scripts/sampling/configs/keyframe${KS_CFG:-}.yaml --chunk_size=2 --landmark_folder=landmarks \
  --audio_folder=audios --audio_emb_folder=audios --output_folder=$O/out \
  --keyframes_ckpt=pretrained_models/checkpoints/keyframe_dub.pt --interpolation_ckpt=pretrained_models/checkpoints/interpolation_dub.pt \
  --add_zero_flag=True --extra_audio=None --compute_until=end --audio_emb_type=hubert --recompute=True \
  --fix_occlusion=False --position=None --start_frame=0 "$@" > $O/run.log 2>&1
rc=$?; t1=$(date +%s.%N); sleep 1; kill $SP; wait $SP 2>/dev/null
cd /workspace/dub/dubai/scripts/r10 && /venv/main/bin/python -c "
import sys,json,gpu_sampler as G
s=G.load('$O/nvml.jsonl'); b,w=G.busy(s,$t0,$t1)
d=float(sys.argv[1]); print(json.dumps({'tag':'$T','rc':$rc,'wall_s':round($t1-$t0,1),'gpu_busy_s':round(b,1),'gpu_weighted_s':round(w,1),'video_s':d,'gpu_busy_s_per_vmin':round(b/d*60,1),'wall_s_per_vmin':round(($t1-$t0)/d*60,1)}))
" $(ffprobe -v error -show_entries format=duration -of csv=p=0 $K/data/videos/$T.mp4) | tee $O/gpu.json
echo KS_DONE $T rc $rc
