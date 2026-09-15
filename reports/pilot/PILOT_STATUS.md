# Pilot status

**6/11 formally closed**

| ID | state | task | verdict/evidence |
|---|---|---|---|
| 0.1 | CLOSED | Blackwell image | PASS |
| 0.2 | CLOSED | LatentSync compatibility | PASS |
| 0.3 | MISSING | FlashAttention sm_120 | - |
| 0.4 | CLOSED | VRAM residency | RESIDENT_ALL |
| 0.5 | MISSING | GPU-min/video-min | - |
| 0.6 | CLOSED | CodeFormer timing | CLOSED |
| 0.7 | CLOSED | CodeFormer vs SyncNet | CLOSED — NO_STRICT_SAFE_WEIGHT |
| 0.8 | OPEN | SyncNet real content | 0.8 = BLOCKED — 2/5 real-content videos produced a LatentSync output; 01, 02, 05 fail upstream LatentSync with 'Face not detected' (content has no face / face below 80 px); SyncNet on original 02 also fails (no face) |
| 0.9 | OPEN | Qwen Q8 vs Q5 | INCOMPLETE |
| 0.10 | MISSING | Cold start | - |
| 0.11 | CLOSED | Sequential S | CLOSED — 5/5 TIMELINE-PRESERVING S MEASURED |
