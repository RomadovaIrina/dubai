#!/bin/bash
# old03 cfhf + the management default alpha 0.8 smoothing (the m18 / m03 cfhf finals use it; the viewer clip was v1 = no smoothing)
cd /workspace/dub/keysync; export KS_HF=3 KS_SHARP=0.3; CF_TS=0.8 bash r13_colorfix.sh cfts; echo CFTS_DONE
