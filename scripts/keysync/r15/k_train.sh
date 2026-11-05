#!/bin/bash
# r15 K: dump the mamakris training clips, then the LCM-LoRA distillation of the interpolation UNet (ks_lcm.py) under gpu.lock.
# usage: bash k_train.sh <run name> [ks_lcm args]
N=$1; shift; K=/workspace/dub/keysync; cd $K
ls data/videos/tr_mk*.mp4 | while read f; do t=$(basename $f .mp4); [ -s dump/$t.pt ] || echo "$t $K/data/videos/$t.mp4 $K/data/audios/$t.wav"; done > dump/jobs_mk.txt
[ -s dump/jobs_mk.txt ] && bash k_dump.sh $K/dump/jobs_mk.txt
mkdir -p lcm/$N; cd code; . $K/venv/bin/activate
flock /workspace/dub/gpu.lock bash -c "/venv/main/bin/python /workspace/dub/dubai/scripts/r10/gpu_sampler.py --out $K/lcm/$N/nvml.jsonl 2>/dev/null & SP=\$!; python $K/ks_lcm.py --dump $K/dump --out $K/lcm/$N $* > $K/lcm/$N/train.log 2>&1; kill \$SP"
grep -E "VAL|TRAIN_DONE|Error" $K/lcm/$N/train.log | tail -20
