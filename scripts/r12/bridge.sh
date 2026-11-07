cd /workspace/dub/dubai; export FC_CMS=1 LS_CMS_READ=1 QUICK=1 FC_EXTRA_ADD="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"
B=/workspace/dub/artifacts/r4m; id=old03; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1)
P="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16"
for v in "br12|--bridge-len 12 --bridge-steps 6" "br16|--bridge-len 16 --bridge-steps 6"; do t=${v%%|*}; x=${v#*|}
  bash scripts/r12/ls_arm.sh $id $t "$P $x"
  ( . scripts/env.sh >/dev/null; python scripts/r12/tone_flicker.py --video $B/r9f_$id$t/${id}_r9f_$id$t.mp4 --src $SRC --src-lm $B/r9g_$id/timing/${id}_source_lm_F.npz --label "$id $t" | tee -a /workspace/dub/r14/abl/$id.txt ) &
done; wait; echo BRIDGE_DONE
