bash ksp.sh m18 ksg_m18 box_q18.json ls_kfglt sp_fixn --interp nearest &
bash ksp.sh m18 ksg_m18 box_q18.json ls_kfglt sp_fixw --dil 0.06 --feather 0.08 &
bash ksp.sh m18 ksg_m18 box_q18.json ls_raw sp_rawn --interp nearest &
bash ksp.sh qo3 ksga_qo3 box_qo3.json ls_kfglt sp_fixn --interp nearest &
wait; echo KSP2_DONE
