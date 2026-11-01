bash ksv.sh m18 ksg_m18 ls_kfglt kl "ln|--colour-mouth-flat 0.6 --register off --face-clip off --paste-guard 0 --retime nearest" &
bash ksv.sh qo3 ksga_qo3 ls_kfglt kol "ln|--colour-mouth-flat 0.6 --register off --face-clip off --paste-guard 0 --retime nearest" &
KSV_ROOT=/workspace/dub/r11_sync bash ksv.sh m18 ksg_m18 ls_kfglt kr "lrm|--colour-mouth-flat 0.6 --register off --face-clip off --paste-guard 0 --retime-mouth nearest" "lrmx|--colour-mouth-flat 0.6 --register off --face-clip off --paste-guard 0 --retime-mouth nearest --cm-mouth-excl 0.5 --cm-mouth-mode zero --colour-mouth-flat 0" &
KSV_ROOT=/workspace/dub/r11_sync bash ksv.sh qo3 ksga_qo3 ls_kfglt kor "lrm|--colour-mouth-flat 0.6 --register off --face-clip off --paste-guard 0 --retime-mouth nearest" "lrmx|--colour-mouth-flat 0.6 --register off --face-clip off --paste-guard 0 --retime-mouth nearest --cm-mouth-excl 0.5 --cm-mouth-mode zero --colour-mouth-flat 0" &
KSV_ROOT=/workspace/dub/r11_sync bash ksv.sh qo3 ksga_qo3 ls_kfglt kox "nfx|--retime-mouth nearest --cm-mouth-excl 0.5 --cm-mouth-mode zero" &
wait; echo KSFIX4_DONE
