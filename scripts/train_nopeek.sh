#!/bin/bash
cd ~/projects/DinoBoard
ORT_ROOT=/root/projects/DinoBoard/third_party/onnxruntime-linux-x64-1.17.3
RUN_DIR="games/splendor/train/runs/nopeek_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
nohup env PYTHONPATH=. LD_LIBRARY_PATH="$ORT_ROOT/lib:$LD_LIBRARY_PATH" \
  ./.venv_debug_shared/bin/python -u -m games.splendor.train.plugin \
  --config games/splendor/train/train_config.server_nopeek.json \
  --output "$RUN_DIR" \
  --steps 20000 \
  --workers 56 \
  --worker-pool process \
  --warm-start-episodes 10000 \
  --warm-start-engine heuristic \
  --warm-start-simulations 1 \
  --warm-start-train-passes 2 \
  --episodes 100 \
  --updates-per-step 6 \
  --batch-size 2048 \
  --buffer-size 300000 \
  --schedule-start-simulations 1000 \
  --schedule-end-simulations 3000 \
  --selfplay-temp-initial 0.5 \
  --selfplay-temp-final 0.1 \
  --selfplay-temp-decay-plies 50 \
  --eval-every 100 \
  --eval-workers 56 \
  --eval-games 100 \
  --history-best-games 100 \
  --eval-simulations 1000 \
  --save-every 100 \
  > "$RUN_DIR/train.log" 2>&1 &
echo "PID=$!"
echo "LOG=$RUN_DIR/train.log"
