#!/bin/bash
cd /Users/alexchuang/Documents/flashkv0516/turbo-fieldfare-github-official
BIN=.build/release/TurboFieldfareCLI
M=/Users/alexchuang/Documents/flashkv0516/models/gemma4.gturbo
P="Write a short technical explanation of how a B-tree index speeds up database lookups."
env TURBO_FIELDFARE_EXPERT_SLOTS=16 TURBO_FIELDFARE_EXPERT_STATS=1 \
  $BIN --model "$M" --prompt "$P" --max-new 160 >/dev/null 2>/tmp/prof_run.log &
PID=$!
sleep 4
sample $PID 14 1 -f /tmp/sample.txt >/dev/null 2>&1
wait $PID
echo PROF-DONE
