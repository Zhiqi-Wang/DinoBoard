#!/bin/bash
cd ~/projects/DinoBoard
LOG="$(ls -dt games/splendor/train/runs/nopeek_* 2>/dev/null | head -n1)/train.log"
grep "eval benchmark" "$LOG"
