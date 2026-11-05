# r15 F: LS_FACE_UNITS (default ON since r15, =0 off): the two-face speaker -> face timeline follows the DUBBED units, not raw diarization turns.
# m03: pyannote adds a SPEAKER_01 turn 19.15-20.21 s (the child, no dubbed line) inside SPEAKER_00 18.1-22.36 s -> frames 478-505 were
# lip-synced on the child's face; flow:4 spread the jump to 475-477 = the second mouth on the left. Run from the worktree root.
import pathlib
p = pathlib.Path("scripts/pilot/clean_face_stack.py"); s = p.read_text()
assert "import os" in s
old = "def assign_speakers(tracks: np.ndarray, turns: list,"
new = '''def face_turns(man: dict):
    """Turns for the speaker -> face selection. LS_FACE_UNITS=1 (r15 F): the dubbed units (speaker, start, end) replace the raw
    diarization turns, so a short overlap turn of a speaker with no dubbed line (m03 19 s: the child) never moves LatentSync's crop."""
    turns = (man.get("diarization") or {}).get("turns")
    if os.environ.get("LS_FACE_UNITS", "1") != "1": return turns
    units = [u for u in (man.get("units") or []) if u.get("speaker") is not None and u.get("end", 0) > u.get("start", 0)]
    if not units: return turns
    print(f"  [faces] LS_FACE_UNITS: {len(units)} dubbed units replace {len(turns or [])} diarization turns", flush=True)
    return [{"speaker": u["speaker"], "start": float(u["start"]), "end": float(u["end"])} for u in units]


def assign_speakers(tracks: np.ndarray, turns: list,'''
assert s.count(old) == 1; s = s.replace(old, new, 1); p.write_text(s)
for f in ("scripts/pilot/run_clean_pipeline_05.py", "scripts/quality/candidate_video_stage.py"):
    p = pathlib.Path(f); s = p.read_text(); old = 'turns=(man.get("diarization") or {}).get("turns")'
    assert s.count(old) == 1, f; p.write_text(s.replace(old, "turns=cfs.face_turns(man)"))
print("patched face_turns")
