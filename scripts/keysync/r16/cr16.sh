#!/bin/bash
# r16 T: the image recipe WITHOUT docker (vast = unprivileged container, no Docker-in-Docker): a clean room that mirrors infra/docker exactly.
#   prep:  bash cr16.sh prep <git rev>     code = git archive <rev> (= COPY .), KeySync = ks_install.sh KS_NO_WEIGHTS=1 into $C/opt/keysync (= the builder
#                                         stage), a models "volume" ($C/models: the research weights linked read-only + keysync/ filled like fetch-models 8/8,
#                                         EMPTY trt_cache / inductor = a true cold start)
#   job:   bash cr16.sh job <job.json>     the entrypoint's env (Dockerfile ENV + DUB_PROFILE=clean) + link_runtime + check.sh, then the job
set -e
C=${CR16:-/workspace/dub/r16t/clean}; RM=/workspace/dub/dubai/models; KSRC=/workspace/dub/keysync
case $1 in
prep)
  rm -rf $C/root; mkdir -p $C/root $C/models/keysync/{checkpoints,student,trt_cap,trt_cache,inductor,torch} $C/data
  git -C /workspace/dub/int archive ${2:-HEAD} | tar -x -C $C/root
  mkdir -p $C/root/third_party; for d in llama_cuda latentsync; do [ -e $C/root/third_party/$d ] || ln -s /workspace/dub/dubai/third_party/$d $C/root/third_party/$d; done   # (= the builder's third_party; NO CodeFormer)
  for d in $(ls $RM); do [ -e $C/models/$d ] || ln -s $RM/$d $C/models/$d; done
  for f in WavLM-Base+.pt keyframe_dub.pt interpolation_dub.pt; do [ -e $C/models/keysync/checkpoints/$f ] || ln $KSRC/code/pretrained_models/checkpoints/$f $C/models/keysync/checkpoints/$f; done
  cp -L $KSRC/student/lora_pick.pt $KSRC/student/lora_meta.json $C/models/keysync/student/; cp $KSRC/trt/cap/kf.pt $KSRC/trt/cap/interp.pt $C/models/keysync/trt_cap/
  t0=$(date +%s); KS_ROOT=$C/opt/keysync KS_NO_WEIGHTS=1 bash $C/root/scripts/keysync/r16/ks_install.sh > $C/ks_install.log 2>&1; echo "ks_install rc=$? $(( $(date +%s) - t0 )) s"
  rm -rf $C/opt/keysync/code/.git; du -sh $C/opt/keysync $C/opt/keysync/venv; echo CR16_PREP_DONE;;
job)
  export DUB_ROOT=$C/root DUB_ENV=/venv/dabai VIRTUAL_ENV=/venv/dabai DUB_MEDIAPIPE_PY=/venv/mediapipe/bin/python VENV_ROOT=/venv \
    HF_HOME=$C/models/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false DUB_THREADS=16 \
    DUB_PROFILE=clean DUB_ALLOW_NC=0 DUB_ALLOW_NC_BED=0 DABAI_IMAGE=cleanroom-r16 DABAI_FACE_DEFAULT=keysync \
    KS_ROOT=$C/opt/keysync KS_LORA=$C/models/keysync/student/lora_pick.pt KS_TRT_CAP=$C/models/keysync/trt_cap KS_TRT_CACHE=$C/models/keysync/trt_cache \
    TORCHINDUCTOR_CACHE_DIR=$C/models/keysync/inductor KS_LMK_CACHE=$C/data/ks_lmk KS_TORCH_HOME=$C/models/keysync/torch \
    R10_HEAVY_LOCK=/workspace/dub/gpu.lock
  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
  [ -L $C/root/models ] || { rm -rf $C/root/models; ln -s $C/models $C/root/models; }
  mkdir -p $KS_ROOT/code/pretrained_models; ln -sfn $C/models/keysync/checkpoints $KS_ROOT/code/pretrained_models/checkpoints   # = entrypoint link_runtime
  [ -s $C/check.log ] || { bash $C/root/infra/docker/check.sh > $C/check.log 2>&1; echo "check rc=$? $(tail -1 $C/check.log)"; }
  exec /venv/dabai/bin/python $C/root/scripts/job/dabai_job.py $2;;
esac
