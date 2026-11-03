#!/bin/bash
cd /workspace/dub/dubai && bash scripts/setup_mediapipe.sh 2>&1 | tail -2
cd /workspace/dub/keysync && bash r13_prep.sh > r13_prep.log 2>&1
