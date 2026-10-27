# r6 F: face twitch (blocker #0)

## Reference scores (r5_twitch.py, coordinator's uploads, 27.10)
See `twitch_refs.md` (twitch_sum / n_events / top20_mean / p99 per video; v4, r4_video_v1, m3).
- twitch_sum: v4 01 8.9 / 02 0 / 03 52.8 / 04 47.3 / 05 33.0; r4_video_v1 11.5 / 0 / 49.6 / 56.6 / 23.1; m3 (ref) - / - / 45.0 / 45.5 / 36.4;
  m3 local re-render 01 11.5 / 02 0.
- 02 has no face (room walkthrough), 01 a face in 260 of 2542 frames: twitch is decided on 03 / 04 / 05.
- m3 fails "<= v4" on 05 (36.4 vs 33.0) only.

## What the top events are (contact sheets + 9-frame grids src / v4 / r4_video_v1 / m3)
- **05 23.67-23.80 s (m3 8.2, r4_video_v1 10.6, v4 clean):** the whole pasted lower face is a blurred, mis-coloured square patch with no
  mouth for 3-4 frames (looks like a corrupted / mis-registered generator frame, or the sanitize layer on one), then snaps back. THE
  visible twitch; most likely the user's "23-24 s". It is in the r4-era renders (r4_video_v1 and m3), not in v4 -> the r4 video stack.
- **05 34.7 s (m3 11.0):** the known raw-generator lip flash (33.7-36 s), item 2 of the brief.
- **04 27.8 / 18.3 s and the other 04 "lips" events (head speed >= 2 iod/s):** mouth opening / closing that the source also does; not a
  visible jump at the 9-frame grid. The lips channel over-fires on articulation during fast head motion (same in v4).
- 03: all "lips" <= 2.2 at low head speed; nothing stands out on the sheet.
Conclusion: the twitches that matter are single / few-frame LAYER failures (a corrupt paste, a flash), not the LS|LS joints or the
LS <-> source crossfades the r5 flags address. Next: find the failing frames in m3's raw LatentSync layer (local 05 render) and add a
generic per-frame layer sanity pass in fps_composite (blur / colour / mouth-less patch vs the neighbours -> flow interpolation from the
nearest good frames, source if the run is long).
