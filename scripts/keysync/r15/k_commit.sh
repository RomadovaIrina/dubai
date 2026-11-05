#!/bin/bash
# r15 K: copy the scripts, the KeySync patch and the report material into the repo, commit on experiment/r15-keysync,
# write the incremental bundle (^origin/experiment/r13-keysync) and verify it.
set -e; K=/workspace/dub/keysync; R=/workspace/dub/dubai; E=$R/reports/experiments/r15-keysync; O=/workspace/dub/r15k; mkdir -p $E/logs $O $R/scripts/keysync/r15
cp $K/ks_fast.py $R/scripts/keysync/ks_fast.py
for f in ks_dump.py ks_lcm.py ks_student.py ks_trt.py ks_trt_vae.py k_setup.sh k_inputs.sh k_base.sh k_eval.sh k_clip.sh k_clips.sh k_arm.sh k_arm2.sh k_trt.sh k_dump.sh k_train.sh \
         k_queue.sh k_queue_b.sh k_queue2.sh k_after_train.sh k_after_train2.sh k_cut_vb.sh k_table.py k_commit.sh tone_flicker.py mouth_open.py; do [ -f $K/$f ] && cp $K/$f $R/scripts/keysync/r15/; done
(cd $K/code && git diff 5e2503b) > $R/scripts/keysync/keysync_5e2503b_sm120_r15.patch
for a in $(ls $K/fast); do [ -f $K/fast/$a/run.log ] || continue; mkdir -p $E/logs/$a; grep -a -E "^\{|KS_JOB|TRT |LORA|LCM sigmas|TRT VAE" $K/fast/$a/run.log > $E/logs/$a/ks_jobs.txt || true
  cp $K/fast/$a/eval_*/eval.json $E/logs/$a/ 2>/dev/null || true; for i in q18 m03 qo3; do [ -f $K/fast/$a/eval_$i/eval.json ] && cp $K/fast/$a/eval_$i/eval.json $E/logs/$a/eval_$i.json; [ -f $K/fast/$a/eval_$i/flicker.json ] && cp $K/fast/$a/eval_$i/flicker.json $E/logs/$a/flicker_$i.json; done; rm -f $E/logs/$a/eval.json; done
for w in kf interp; do [ -f $K/trt/eng/$w/verify.json ] && cp $K/trt/eng/$w/verify.json $E/logs/trt_verify_$w.json; done
[ -f $K/trt/eng_s/interp/verify.json ] && cp $K/trt/eng_s/interp/verify.json $E/logs/trt_verify_student.json; [ -f $K/trt/vae/verify.json ] && cp $K/trt/vae/verify.json $E/logs/trt_verify_vae.json
for r in r0_bad_scaling r1; do [ -f $K/lcm/$r/train.log ] && grep -a -E "GRID|LORA|SEGMENTS|^VAL|STEP|TRAIN_DONE" $K/lcm/$r/train.log > $E/logs/train_$r.txt; [ -f $K/lcm/$r/lora_meta.json ] && cp $K/lcm/$r/lora_meta.json $E/logs/lora_meta_$r.json; done
cp $K/fast/table.json $E/table.json 2>/dev/null || true; ls /workspace/dub/outbox/r15_k > $E/outbox_files.txt
cd $R; git add scripts/keysync/ks_fast.py scripts/keysync/keysync_5e2503b_sm120_r15.patch scripts/keysync/r15 reports/experiments/r15-keysync
git commit -q -m "r15 K: KeySync speed pilot: TensorRT FP16 engines (UNets + VAE decoder) + LCM-LoRA guided distillation of the interpolation UNet

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git log --oneline -1
git bundle create $O/r15-keysync.bundle experiment/r15-keysync ^origin/experiment/r13-keysync && git bundle verify $O/r15-keysync.bundle && ls -la $O/r15-keysync.bundle
cp $E/REPORT.md $O/REPORT.md; date > $O/DONE; echo K_COMMIT_DONE
