#!/bin/bash
cd ~/projects/DinoBoard
export BOARD_AI_WITH_ONNX=1
export BOARD_AI_ONNXRUNTIME_ROOT=/root/projects/DinoBoard/third_party/onnxruntime-linux-x64-1.17.3
export LD_LIBRARY_PATH="$BOARD_AI_ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH"
.venv_debug_shared/bin/pip install -e games/splendor/debug_service/
