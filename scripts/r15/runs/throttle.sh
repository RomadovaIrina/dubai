#!/bin/bash
# r15 F3: keep at most MAXRUN of my composite trees running (the oldest), the rest SIGSTOPped; everything of mine niced to 19
MAXRUN=${MAXRUN:-2}
kids(){ for c in $(ps -o pid= --ppid $1); do kids $c; echo $c; done; }
while [ ! -e /workspace/dub/r15f3/THROTTLE_OFF ]; do
  for d in $(ps -eo pid,args | grep "bash /workspace/dub/r15f3/run_" | grep -v grep | awk "{print \$1}"); do renice -n 19 -p $d $(kids $d) > /dev/null 2>&1; done
  i=0; for p in $(ps -eo pid,etimes,args --sort=-etimes | grep "bash scripts/r15/comp_f[234].sh" | grep -v grep | awk "{print \$1}"); do
    T="$p $(kids $p)"; if [ $i -lt $MAXRUN ]; then kill -CONT $T 2>/dev/null; else kill -STOP $T 2>/dev/null; fi; i=$((i+1)); done
  sleep 15; done
for p in $(ps -eo pid,args | grep "bash scripts/r15/comp_f[234].sh" | grep -v grep | awk "{print \$1}"); do kill -CONT $p $(kids $p) 2>/dev/null; done
