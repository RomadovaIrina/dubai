# r15 F GPU queue (waits for gpu.lock = D's training end): probe (original | validated rigid track) -> full m03 regeneration with
# LS_RIGID_VALIDATE on D's own master / landmarks / rigid track (prep cache) -> composite (q2 + F3, production repair) -> GPU_DONE -> gate + clips
cd /workspace/dub/dubai-f; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai-f DUB_ALLOW_NC=1
run=q2_m03; sfx=rv; B=/workspace/dub/artifacts/r4m; A=/workspace/dub/artifacts/r4a; O=/workspace/dub/outbox/r15_f; L=/workspace/dub/r15f/logs
W0=$B/r9g_$run/work_${run}_r9g_$run/video
# 1. probe 16-22 s (2 x ~150 frames)
echo "== probe waiting for gpu.lock $(date +%T)"
flock /workspace/dub/gpu.lock python scripts/r15/probe_anchor.py $run 16 22 $O/m03_16-22s_originalH_vs_validatedH_05x.mp4 --thr 0.02 > $L/probe_anchor.log 2>&1; grep "==\|rigid" $L/probe_anchor.log
cp $O/m03_16-22s_originalH_vs_validatedH_05x.f476.jpg /workspace/dub/r15f/ 2>/dev/null
# 2. full regeneration on D's prep (master, its wav, MediaPipe passes, rigid track): the validation is the only difference
PC=/workspace/dub/artifacts/r3d/prep/$run; mkdir -p $PC; cp -n $W0/master_25fps.mp4 $W0/*.wav $W0/master_mediapipe*.npz $W0/master_rigid.npz $PC/
export P2_PREP_CACHE=1 DUB_INPUTS=/workspace/dub/r14d/eval/$run/e2e/work/in LS_CMS_READ=1 FC_CMS=1 LS_RIGID_VALIDATE=0.02 \
       GEN_COMP=0 FAST_SRCPREP=0 FAST_FB_BG=1 FAST_ENC=ultrafast LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
P="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16"
echo "== regen start $(date +%T)"
AUDIO=$A/r9_$run TAG=r9g_$run$sfx SKIP_SCORE=1 VIDEO_EXTRA="$P" bash scripts/r7/r7_e2e.sh $run > $L/gen_$run$sfx.log 2>&1
grep -h "\[rigid\]" $B/r9g_$run$sfx/work_${run}_r9g_$run$sfx/*.log $B/r9g_$run$sfx/*.log 2>/dev/null | head -2; tail -n 2 $L/gen_$run$sfx.log
# 3. composite (layer SR on the GPU -> under the lock), production repair (rep2s, no guard) to isolate the fix
echo "== comp start $(date +%T)"
flock /workspace/dub/gpu.lock env GT=r9g_$run$sfx bash scripts/r15/comp_f.sh $run $sfx rep2s ""
touch /workspace/dub/r15f/GPU_DONE; echo "== GPU_DONE $(date +%T)"
# 4. CPU / SyncNet (no lock, VRAM check): gate vs the control c0, flicker, clips
until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 4096 ]; do sleep 30; done
bash scripts/r15/gate_f.sh $run $sfx > $L/gate_$run$sfx.log 2>&1; tail -n 2 $L/gate_$run$sfx.log | cut -c1-300
python scripts/r10/gate_cmp.py --base /workspace/dub/r15f/gate/${run}c0 --cand /workspace/dub/r15f/gate/$run$sfx --md /workspace/dub/r15f/gate.md | head -n 3
SRC=$(ls /workspace/dub/r14d/eval/$run/e2e/work/in/$run.* | head -1); SL=$B/r9g_$run/timing/${run}_source_lm_F.npz; V=$B/r9f_$run$sfx/${run}_r9f_$run$sfx.mp4
python scripts/r12/tone_flicker.py --video $V --src $SRC --src-lm $SL --label "$run $sfx" | tee -a /workspace/dub/r15f/flicker.txt
python scripts/r12/mouth_pop.py --video $V --src $SRC --src-lm $SL --label "$run $sfx" | tee -a /workspace/dub/r15f/flicker.txt
bash scripts/r15/clip_f.sh $run $B/r9f_${run}c0/${run}_r9f_${run}c0.mp4 "before (production)" $V "rigid validated" "17-21" $O/m03_17-21s_production_vs_rigidvalidated_05x.mp4
echo GEN_Q_DONE
