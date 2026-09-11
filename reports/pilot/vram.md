# VRAM measurements

| label | ok | time | peak proc | peak device (Δ) | torch alloc/reserved | mode | env |
|---|---|---|---|---|---|---|---|
| self-test/alloc-2GiB | PASS | 1.50s | 2434 MiB | 2458 MiB (Δ2048) | 2048/2048 MiB | in-process | `20260911_182453-8125bfd3` |
| retinaface-smoke | PASS | 5.16s | 798 MiB | 822 MiB (Δ806) | - | subprocess | `20260911_182453-8125bfd3` |
| LatentSync | PASS | 131.51s | 20358 MiB | 20382 MiB (Δ20366) | - | subprocess | `20260911_182453-8125bfd3` |
| stage/vad_split | PASS | 7.18s | 386 MiB | 410 MiB (Δ0) | 0/0 MiB | in-process | `20260911_182453-8125bfd3` |
| stage/asr | PASS | 1.75s | 2612 MiB | 2636 MiB (Δ402) | 0/0 MiB | in-process | `20260911_182453-8125bfd3` |
| stage/concat | PASS | 0.23s | 394 MiB | 418 MiB (Δ0) | 0/0 MiB | in-process | `20260911_182453-8125bfd3` |
| stage/subtitle_burn | PASS | 2.38s | 394 MiB | 418 MiB (Δ0) | 0/0 MiB | in-process | `20260911_182453-8125bfd3` |
