bash ksp.sh m18 ksg_m18 box_q18.json ls_kfglt sp_fixh --interp hybrid &
bash ksp.sh m18 ksg_m18 box_q18.json ls_kfglt sp_fixh5 --interp hybrid --lip-dil 0.5 &
bash ksp.sh qo3 ksga_qo3 box_qo3.json ls_kfglt sp_fixh --interp hybrid &
bash ksp.sh qo3 ksga_qo3 box_qo3.json ls_kfglt sp_fixh5 --interp hybrid --lip-dil 0.5 &
wait; echo KSP3_DONE
