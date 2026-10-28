# r7 subs: spec block 9 (subtitles + karaoke + watermark), CPU only

Code: `scripts/r7/r7_subs.py` (ffmpeg + libass only, 4 threads). Wired into `scripts/r6/r6_e2e.sh` as a post-step after the final mux.

## Timing (9.1 / 9.2)
- **dub:** each English phrase's real placed interval comes from `units[].alignment.placements[].placed`, which is the time after the fit and atempo. Word times come from the TTS CTC forced-align (`units[].tts.word_spans`), mapped linearly from the phrase `clip_span` onto `placed`, so atempo is accounted for. If the spans don't match the phrase's words, the phrase is split proportionally by characters instead.
- **src:** whisper word timestamps (`asr.segments[].words`).
- Cues break on a pause > 0.7 s, a sentence end, a length > 6 s, or > 2 lines. Cues never overlap.

## Styles (9.3)
- Presets: `white`, `yellow`, `trendy` (big caps, green highlight), `outline` (boxed).
- Karaoke uses `\kf` per word, with `\k` for gaps.
- Lines: at most 2, balanced, preferring a break at punctuation.
- Chars per line = min(42, what fits the frame width). Portrait 464 px gets about 20–30.

Previews at t=6.5 s on 03, with a watermark: `preview_{white,yellow,trendy,outline}.png`, plus `preview_grid.png`.

## Burn (9.4)
- Encode: libx264 crf 18 at the source resolution and fps, yuv420p, aspect kept, `-c:a copy`.
- Watermark: PNG scaled to 18 % of the short side, corner `tl|tr|bl|br`, opacity via the alpha multiplier.

## E2E flags (all default off)
`--subs on|off --subs-lang src|dub --subs-style <preset> --watermark <png> --wm-corner br --wm-opacity 0.6`
With these flags the E2E writes an extra `<id>_<TAG>_final.mp4`. The plain `<id>_<TAG>.mp4` is unchanged.

## Demo
`/workspace/dub/outbox/S__03_subs_demo.mp4`: yellow karaoke + watermark, br corner, 0.6 opacity.

**Caveat:** no m5 lip-synced 03 mp4 exists on disk (`artifacts/r4m` is absent). The demo therefore burns onto the *source* 03 video muxed with the r6 dub audio (`final_mix_aac.m4a`). The subtitle timing is the real dub timing either way.

## Not done
- The flags were not run through a full GPU E2E, by design: this agent is CPU-only. `bash -n` passes.
- Only the 03 demo was rendered.
