cd /workspace/dub/dubai
until [ -s /workspace/dub/r10/gate/m18base/face_gate.json ]; do sleep 10; done
bash scripts/r12/ls_arm.sh m18 t0 ""
bash scripts/r12/ls_arm.sh m18 t3 "" LS_CFG_TRUNC=3
bash scripts/r12/ls_arm.sh m18 t2 "" LS_CFG_TRUNC=2
bash scripts/r12/ls_arm.sh m18 t5 "" LS_CFG_TRUNC=5
echo TRUNC_M18_DONE
