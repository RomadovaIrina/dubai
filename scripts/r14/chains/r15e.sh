cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
# r15-D m03 with agent F's fix (d494c67, cherry-picked as 90fd069): LS_FACE_UNITS=1, the speaker -> face timeline follows the dubbed units
# (the old m03 runs lip-synced the child's face at 19.15-20.21 s, a 1 s SPEAKER_01 diarization turn). All three m03 arms rerun with it:
# q2u (q2 + F3 baseline), s4p1500c8u (r14 student), s6c8u (r15 student pick), then the clips s6c8u vs q2u / s4p1500c8u
X=/workspace/dub/r14d; N=$X/R15_NOTES.txt; M=/workspace/dub/r15d; mkdir -p $M
note(){ echo "$(date +%T) $*" | tee -a $N; }
idle(){ local ok=0 u; for i in $(seq 1 240); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$u" -lt 2000 ] && [ ! -f /workspace/dub/r15t/E2E_RUNNING ] && flock -w 5 /workspace/dub/gpu.lock true; then ok=$((ok + 1)); [ $ok -ge 2 ] && return 0; else ok=0; fi; sleep 15; done
  note "GPU not idle after 60 min, going anyway"; }
until grep -q R15D_DONE $X/r15d.log || ! pgrep -f "^bash r15d.sh" >/dev/null; do sleep 30; done
FU='"LS_FACE_UNITS": "1"'; C8='"LS_STRIDE": "8", "LS_OVERLAP": "continue"'
note "m03 rerun with LS_FACE_UNITS=1 (F's fix 90fd069): q2u, s4p1500c8u, s6c8u"
for spec in s4p1500c8u:$X/ckpt/r2_s4/step_01500_ema.pt:4 s6c8u:$X/ckpt/r4_s6/step_00500_ema.pt:6; do
  IFS=: read tag ck st <<< "$spec"
  idle; echo "$tag m03 $(date +%T)" > $M/E2E_RUNNING
  Q2=q2u Q2ENV="$FU" OB=/workspace/dub/outbox/r15_d LORA=$ck STEPS=$st TAG=$tag XENV="$C8, $FU" bash scripts/r14/eval_r14.sh m03
  rm -f $M/E2E_RUNNING
  [ -s $X/eval/${tag}_m03/${tag}_m03.mp4 ] || note "$tag m03: no output video ($(tail -1 $X/eval/${tag}_m03.log 2>/dev/null | cut -c1-200))"
done
[ -s $X/eval/q2u_m03/q2u_m03.mp4 ] || note "q2u m03: no output video ($(tail -1 $X/eval/q2u_m03.log 2>/dev/null | cut -c1-200))"
Q2=q2u bash scripts/r14/r15_clips.sh s6c8u s4p1500c8u > $X/r15e_clips.log 2>&1 || note "r15e clips failed: $(tail -1 $X/r15e_clips.log)"
echo "R15E_DONE $(date +%T)"
