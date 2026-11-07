cd /workspace/dub/dubai
export QUICK=1
bash scripts/r12/ls_arm.sh m18 t0c ""
bash scripts/r12/ls_arm.sh m18 ucd "" LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1
bash scripts/r12/ls_arm.sh m03 t0c ""
bash scripts/r12/ls_arm.sh m03 ucd "" LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1
bash scripts/r12/ls_arm.sh old03 t0c ""
bash scripts/r12/ls_arm.sh old03 ucd "" LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1
echo COMPILE_DONE
