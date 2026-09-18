# Pilot status

**10/11 formally closed**

| ID | state | task | verdict/evidence |
|---|---|---|---|
| 0.1 | CLOSED | Blackwell image | PASS |
| 0.2 | CLOSED | LatentSync compatibility | PASS |
| 0.3 | CLOSED | FlashAttention sm_120 | CLOSED — flash-attn 2.8.3.post1 builds and runs on sm_120 (build PASS, runtime PASS); not installed into /venv/dabai (native SDPA flash kernel already used) |
| 0.4 | CLOSED | VRAM residency | RESIDENT_ALL |
| 0.5 | CLOSED | GPU-min/video-min | CLOSED — PASS_TARGET; median 4.6365 GPU-min/video-min vs target 5.0 |
| 0.6 | CLOSED | CodeFormer timing | CLOSED |
| 0.7 | CLOSED | CodeFormer vs SyncNet | CLOSED — NO_STRICT_SAFE_WEIGHT |
| 0.8 | CLOSED | SyncNet real content | CLOSED — 5/5 final E2E outputs valid; SyncNet measured on 4/5, 1 N/A (no face); contract threshold still UNDECIDED |
| 0.9 | CLOSED | Qwen Q8 vs Q5 | CLOSED — Q8_0 selected (speed winner Q5_K_M 1.48x; quality decision after manual review) |
| 0.10 | MISSING | Cold start | - |
| 0.11 | CLOSED | Sequential S | CLOSED — 5/5 TIMELINE-PRESERVING S MEASURED |
