#!/bin/bash
# r15 K: TensorRT FP16 engines (full + shallow, DeepCache 2:0) for the keyframe (batch 42) and interpolation (batch 28 = the production chunk) UNets
K=/workspace/dub/keysync; cd $K/code; . $K/venv/bin/activate; mkdir -p $K/trt/eng
for w in kf interp; do
  [ -s $K/trt/eng/$w/shallow.plan ] || flock /workspace/dub/gpu.lock python $K/ks_trt.py --cap $K/trt/cap --which $w --out $K/trt/eng > $K/trt/make_$w.log 2>&1
  grep -a -E "TRT_VERIFY|BUILT|TORCH|Traceback|Error" $K/trt/make_$w.log | tail -6
done
