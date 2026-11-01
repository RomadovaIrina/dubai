id=$1; cd /workspace/dub/r10; OS="--region paste --shade on --hf off --radius-s 0.1 --shade-sigma 0.07 --keep-cavity on"
OSTAB_IN=/workspace/dub/r10/abl/$id/${id}_reg_smooth=4.mp4 bash ostab_try.sh $id f1l $OS --chin-reg on --chin-reg-vert 0
cd /workspace/dub/dubai; bash scripts/r10/gate.sh $id F1L /workspace/dub/r10/abl/$id/${id}_f1l.mp4; python scripts/r10/gate_cmp.py --base /workspace/dub/r10/gate/${id}base --cand /workspace/dub/r10/gate/${id}F1L --md /workspace/dub/r10/gate/gate5.md | head -n 3; echo G5_DONE $id
