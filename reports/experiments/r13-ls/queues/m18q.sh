cd /workspace/dub/dubai; export FC_CMS=1 LS_CMS_READ=1 QUICK=1 FC_EXTRA_ADD="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.5"
B=/workspace/dub/artifacts/r4m; id=m18; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); SL=$B/r9g_$id/timing/${id}_source_lm_F.npz
for v in "cmsr" "e34q LS_RATE=3/4 LS_RATE_EVEN=1"; do set -- $v; t=$1; shift
  bash scripts/r12/ls_arm.sh $id $t "" "$@"
  ( . scripts/env.sh >/dev/null; V=$B/r9f_$id$t/${id}_r9f_$id$t.mp4
    python scripts/r12/lab_drift.py --video $V --src $SRC --src-lm $SL --label "$id $t"
    python scripts/r12/mouth_pop.py --video $V --src $SRC --src-lm $SL --label "$id $t @6.25" --cad-hz 6.25
    bash scripts/r10/gate.sh $id Y$t $V > /workspace/dub/r12/logs/gate_$id$t.log 2>&1
    python scripts/r10/gate_cmp.py --base /workspace/dub/r10/gate/${id}base --cand /workspace/dub/r10/gate/${id}Y$t --md /workspace/dub/r12/gate.md | head -n 3
    bash scripts/r12/clip_ls.sh $id $t "k5 + $t" | tail -2 ) &
done; wait; echo M18Q_DONE
