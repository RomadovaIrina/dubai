cd /workspace/dub/dubai; . scripts/env.sh >/dev/null
for id in m03 old03; do (
  W=/workspace/dub/artifacts/r4m/r9g_$id/work_${id}_r9g_$id; P=$W/${id}_r9g_${id}_25fps; L=${P}_ls_cfd_ia04_srt1_rep2s.mp4; O=${P}_ls_cfd_ia04_srt1_rep2s_tsm.mp4
  [ -s $O ] || python scripts/r12/ks_tsmooth.py --layer $L --lm ${L%.mp4}.mediapipe.npz --out $O --alpha auto --src-lm $W/video/master_mediapipe.npz | tail -1
  cp ${L%.mp4}.mediapipe.npz ${O%.mp4}.mediapipe.npz; cat $O.alpha.json; echo
  FB=$(ls ${P}_fb_fb7s_r9f_${id}_rep2s.npz) LS=$O THR=16 bash scripts/r11s/arm.sh $id ktsm "--mouth-region 0.55 --cm-mouth-excl 0.5"
  python scripts/r12/mouth_pop.py --video ${P}_ktsm.mp4 --src $(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1) --src-lm /workspace/dub/artifacts/r4m/r9g_$id/timing/${id}_source_lm_F.npz --label "$id k5+tsmooth"
  python scripts/r12/mouth_pop.py --video /workspace/dub/artifacts/r4m/r9f_$id/${id}_r9f_$id.mp4 --src $(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1) --src-lm /workspace/dub/artifacts/r4m/r9g_$id/timing/${id}_source_lm_F.npz --label "$id k5" ) &
done; wait; echo TSM_DONE
