cd /workspace/dub/dubai-f; B=/workspace/dub/artifacts/r4m
until grep -q NOSR_DONE /workspace/dub/r15f3/run_nosr.out; do sleep 30; done
for x in q2_m03:r9g_q2_m03fu:fu q2_m18:r9g_q2_m18: q2_old03:r9g_q2_old03:; do IFS=: read id ST A <<< "$x"
  ( FCX="--retime nearest" GT=$ST bash scripts/r15/comp_f4.sh $id ${A}rtn ia04 ) & ( FCX="--retime flow" GT=$ST bash scripts/r15/comp_f4.sh $id ${A}rtf ia04 ) & wait; done
. scripts/env.sh >/dev/null; J=/workspace/dub/r15f3/jitter.jsonl; : > $J; D=/workspace/dub/r15f3/jit
for x in q2_m03:fu q2_m18:c0 q2_old03:c0; do IFS=: read id P <<< "$x"; A=${P/c0/}; SRC=$(ls /workspace/dub/r14d/eval/$id/e2e/work/in/$id.* | head -1)
  [ -s $D/${id}_src.npz ] || /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $SRC --out $D/${id}_src.npz > /dev/null 2>&1
  for t in $P ${A}rtn ${A}rtf; do V=$B/r9f_$id$t/${id}_r9f_$id$t.mp4; [ -s $D/$id$t.npz ] || /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $V --out $D/$id$t.npz > /dev/null 2>&1 &
  done; wait
  for t in $P ${A}rtn ${A}rtf; do python scripts/r15/lip_jitter.py --lm $D/$id$t.npz --src-lm $D/${id}_src.npz --label $id$t --json $J; done; done
echo JIT_DONE $(date +%T)
