cd /workspace/dub/dubai-d; export DUB_ROOT=$PWD; . scripts/env.sh >/dev/null; export EVAL_DIR=/workspace/dub/r15x/eval
Q2=q2u Q2ENV="\"LS_FACE_UNITS\": \"1\"" OB=/workspace/dub/outbox/r15x LORA=/workspace/dub/r15x/ckpt/r6_s4/step_01250_ema.pt STEPS=4 TAG=r6s1250k XENV="\"LS_STRIDE\": \"8\", \"LS_OVERLAP\": \"continue\", \"LS_FACE_UNITS\": \"1\"" bash scripts/r14/eval_r14.sh m03
echo REGEN_DONE
