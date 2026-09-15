# 0.11 — LEGACY_INVALID_TIMELINE

The table below was produced by the old measure_stages.py that concatenated **speech segments only** (53.9 s source -> 29.5 s output, timeline compressed, subtitle cues shifted). It is kept for provenance only and is **not** 0.11 evidence. The valid, timeline-preserving results are in `reports/pilot/0.11_s_dataset.md` / `.json` (per-video `0.11_stages_XX.json`).


| input | source | dur | split | segs | T_vad_split | T_concat | T_subtitle_burn | **S** | s/min | env |
|---|---|---|---|---|---|---|---|---|---|---|
| 04.mp4 | 464x848@30.0 h264 | 53.9s | reencode | 15 | 7.18s | 0.23s | 2.38s | **9.79s** | 10.9 | `20260911_182453-8125bfd3` |
