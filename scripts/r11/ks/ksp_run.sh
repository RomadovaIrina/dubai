bash ksp.sh m18 ksg_m18 box_q18.json ls_srt1 sp_plain &
bash ksp.sh m18 ksg_m18 box_q18.json ls_kfglt sp_fix &
bash ksp.sh qo3 ksga_qo3 box_qo3.json ls_srt1 sp_plain &
bash ksp.sh qo3 ksga_qo3 box_qo3.json ls_kfglt sp_fix &
wait; echo KSP_DONE
