cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
# r15-D E2E retry: the 09:29 s6c8 runs failed in the audio stage (VRAM peak 32 GB while another agent ran a 13 GB q2_m03 face stage).
# dabai_job cannot hold gpu.lock for the whole run (e2e.sh takes it inside for sep / gaps), so each video waits for an idle GPU first,
# and /workspace/dub/r15d/E2E_RUNNING marks the run for the other agents
X=/workspace/dub/r14d; N=$X/R15_NOTES.txt; C=$X/ckpt/r4_s6; M=/workspace/dub/r15d; mkdir -p $M
note(){ echo "$(date +%T) $*" | tee -a $N; }
idle(){ local ok=0 u; for i in $(seq 1 240); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$u" -lt 2000 ] && [ ! -f /workspace/dub/r15t/E2E_RUNNING ] && flock -w 5 /workspace/dub/gpu.lock true; then ok=$((ok + 1)); [ $ok -ge 2 ] && return 0; else ok=0; fi; sleep 15; done
  note "GPU not idle after 60 min, going anyway"; }
C8='"LS_STRIDE": "8", "LS_OVERLAP": "continue"'
note "E2E retry (09:29 runs: audio_failed under VRAM contention; moved to $X/failed_0933)"
for spec in s6c8:step_00500_ema.pt:old03 s6c8:step_00500_ema.pt:m03 s6c8:step_00500_ema.pt:m18 s6fc8:step_02000_ema.pt:old03; do
  IFS=: read tag ck id <<< "$spec"; idle; echo "$tag $id $(date +%T)" > $M/E2E_RUNNING
  OB=/workspace/dub/outbox/r15_d LORA=$C/$ck STEPS=6 TAG=$tag XENV="$C8" bash scripts/r14/eval_r14.sh $id
  [ -s $X/eval/${tag}_$id/${tag}_$id.mp4 ] || note "$tag $id: no output video ($(tail -1 $X/eval/${tag}_$id.log 2>/dev/null | cut -c1-200))"
done
rm -f $M/E2E_RUNNING
bash scripts/r14/r15_clips.sh s6c8 > $X/r15_clips.log 2>&1 || note "r15 clips failed: $(tail -1 $X/r15_clips.log)"
echo "R15D_DONE $(date +%T)"
