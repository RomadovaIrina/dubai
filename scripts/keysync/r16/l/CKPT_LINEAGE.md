# r16 L: checkpoint lineage (global steps)

Global step = the r16 continuation steps since `lora_pick.pt` (r15 lcm_r1, 2984 steps), so g0 = lora_pick. The old names used per-run local steps (audit #6). From be047fa+ on, `ks_lcm_jit.py` names checkpoints by the global step and writes a `lora_step_N.json` sidecar with every hyperparameter.

All A runs: lr 5e-5, swap 0.5, τ 0, LoRA r64 / α64, 12-min chunks, fast2 student.

**LEAKAGE:** the training dump of every run below = q01–q09 (the QA test set). The init lora_pick was trained on other mamakris stories (the same speaker and room as m03 / m18). Their metrics on q-clips are invalid; on m03 / m18 they are optimistic.

| old name | global | init (global) | λ | loss | band | spike-top | pair | server |
|---|---|---|---|---|---|---|---|---|
| jit1_s387 | g387 | lora_pick (g0) | 1 | mean hinge | cheek + jaw | – | – | r16-qt |
| jit1_s778 | g778 | lora_pick (g0) | 1 | mean hinge | cheek + jaw | – | – | r16-qt |
| jit10_s388 | g1166 | jit1_s778 (g778) | 10 | mean hinge | cheek + jaw | – | – | r16-qt |
| jit10_s776 | g1554 | jit1_s778 (g778) | 10 | mean hinge | cheek + jaw | – | – | r16-qt |
| spk1_s284 | g1838 | jit10_s776 (g1554) | 10 | CVaR | cheek + jaw | 0.05 | 0.5 | r16-k1 |
| spk3_s284 | g1838 | jit10_s776 (g1554) | 3 | CVaR | cheek + jaw | 0.05 | 0.5 | r16-k1 |
| spk3c_s284 | g1838 | jit10_s776 (g1554) | 3 | CVaR | cheeks only | 0.05 | 0.5 | r16-k1 |
| spk3c_s563 | g2117 | jit10_s776 (g1554) | 3 | CVaR | cheeks only | 0.05 | 0.5 | r16-k1 |

λ is not comparable between the mean and CVaR runs: the spike-top 0.05 term is up to ~20× the mean.

B (keyframe LoRA r16, per-person, self-reconstruction on the m03 source):
- adapt_m03_90: 90 s, lr 2e-4, 95 steps.
- adapt_m03s: 240 s, lr 1e-3, 260 steps; 120 s snapshot.
