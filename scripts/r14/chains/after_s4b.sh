cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
# r14 chain, reordered (coordinator): the s4 arm's eval (pid $1) finishes m03 / m18, then step 1500 (best-by-pix) on old03 EARLY; if it syncs
# better than the final EMA it runs on m03 / m18 next, before the overlap arms s4f8 / s4c8; otherwise it goes last.
X=/workspace/dub/r14d; C=$X/ckpt/r2_s4; L=$C/step_03000_ema.pt; P=$C/step_01500_ema.pt; E=$X/eval
while kill -0 $1 2>/dev/null; do sleep 30; done
echo "s4 post m03 m18 $(date +%T)"; LORA=$L STEPS=4 TAG=s4 bash scripts/r14/eval_r14.sh m03 m18
echo "eval s4p1500 old03 $(date +%T)"; LORA=$P STEPS=4 TAG=s4p1500 bash scripts/r14/eval_r14.sh old03
sy(){ grep -o "[0-9.]*" <<< "$(sed 's/.*: //' $E/$1.sync.txt)" | head -1; }
better=$(python3 -c "print(1 if float('$(sy s4p1500_old03)' or 0) > float('$(sy s4_old03)' or 0) else 0)")
echo "old03 sync s4 $(sy s4_old03) s4p1500 $(sy s4p1500_old03) -> 1500 first: $better"
[ "$better" = 1 ] && { echo "eval s4p1500 m03 m18 $(date +%T)"; LORA=$P STEPS=4 TAG=s4p1500 bash scripts/r14/eval_r14.sh m03 m18; }
echo "eval s4f8 $(date +%T)"; LORA=$L STEPS=4 TAG=s4f8 XENV='"LS_STRIDE": "8", "LS_OVERLAP": "fuse"' bash scripts/r14/eval_r14.sh old03 m03 m18
echo "eval s4c8 $(date +%T)"; LORA=$L STEPS=4 TAG=s4c8 XENV='"LS_STRIDE": "8", "LS_OVERLAP": "continue"' bash scripts/r14/eval_r14.sh old03 m03 m18
[ "$better" = 1 ] || { echo "eval s4p1500 m03 m18 $(date +%T)"; LORA=$P STEPS=4 TAG=s4p1500 bash scripts/r14/eval_r14.sh m03 m18; }
echo "AFTER_S4_DONE $(date +%T)"
