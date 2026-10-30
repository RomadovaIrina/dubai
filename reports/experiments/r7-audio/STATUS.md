# r7-audio status (audio agent A, 2026-10-28)

**Published: r7_audio_v1 WATCH** (tag r7v1b) = the published r6_audio_v1 takes (handoff/in/audio-r6-v1) + lip placement + gap clean.
outbox/<id>_r7_audio_v1_WATCH_*, handoff/out/F/audio-r7-v1 (42 MB, READY at the coordinator's request). Details, numbers and the 4 classified
gate rows: README_v1.md. 7B vs 14B: qwen_7b_vs_14b.md (recommendation: 14B Q5_K_M with --dur-rounds 1, same speed as 7B Q8_0, best faithfulness).

Reproduce: `REUSE=r6v1src R7_PREV=r6v1 REMIX_ARGS="--src-duck-db -6 --unit-level off --gap-bed-cap-db 10 --gap-bed-cap-scope edges --gap-clean on --gap-clean-nr-ns 0.7" bash scripts/r7/r7_audio.sh r7v1b`
(r6v1src = the uploaded r6v1 dub + manifest with this server's Spleeter stems), then `FORCE_WATCH=1 bash scripts/r7/r7_publish.sh v1 r7v1b r6v1`.
From raw input: `bash scripts/r7/r7_audio.sh <tag>` (r6 pipeline + both r7 changes).

Findings:
- r6_audio_v1 does NOT reproduce on another server (14B: 3-20 units per video change text; llama.cpp / Chatterbox are not deterministic across builds and GPUs),
  so m6 audio is built on the published takes.
- The "leaked words" in gaps were Whisper hallucinations on joined gaps; per gap with no_speech_prob <= 0.6: 0 words in r6v1 and r7.
- Speed: r7 adds ~10 s per video (placement 5-7 s incl. Whisper words, gap clean 2-5 s); audio stage 14B 110.7 s/video-min (+~8 with r7), 7B 99.0.
- Disk 50/150 GB; bandwidth: 14B GGUF 10.5 GB (approved once), spleeter 2stems ~80 MB, pip ~600 MB.
- git push: the deploy key fails ("check access rights"); all commits are local on experiment/r7-audio.
