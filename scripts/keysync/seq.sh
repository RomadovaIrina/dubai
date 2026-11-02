#!/bin/bash
# r12 KS: let the old03 baseline arm finish, stop the rest of r12_arms1 (KeySync holds ~30 GB: the E2E audio's llama context failed next to it), run the 3-video E2E, then resume the arms
cd /workspace/dub/keysync
until grep -a -q "KSF_DONE r12_base" r12_arms1.log; do sleep 15; done
P=$(cat r12_arms1.pid 2>/dev/null); for p in $(pgrep -f "bash r12_arms1.sh"); do kill $p; done; sleep 2
for p in $(pgrep -f "ks_fast.py --jobs /workspace/dub/keysync/fast/r12_stack"); do kill $p; done
cd /workspace/dub/r12k && rm -rf runs/* && bash e2e3.sh > e2e3.log 2>&1
cd /workspace/dub/keysync && cat > r12_arms1b.sh <<'EOS'
#!/bin/bash
cd /workspace/dub/keysync; export KS_DUB=$PWD/lm/qo3_src16k.wav KS_MIX=$PWD/lm/qo3_src.m4a KS_LM=$PWD/lm/qo3_lm.npz IDS=qo3
KS_DC=2:0 bash ks_fast.sh r12_stack_warm --cfg _s8 --vae-half --compile unet
KS_DC=2:0 bash ks_fast.sh r12_stack --cfg _s8 --vae-half --compile unet
KS_DC=2:0 KS_IGUIDE=trunc:3 bash ks_fast.sh r12_t3 --cfg _s8 --vae-half --compile unet
echo R12_ARMS1_DONE
EOS
bash r12_arms1b.sh >> r12_arms1.log 2>&1
