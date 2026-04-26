#!/bin/bash
cd ~/projects/DinoBoard
tail -f "$(ls -dt games/splendor/train/runs/nopeek_* | head -n1)/train.log"
